"""Turning contours into a point stream a galvanometer can actually draw.

This is the part that separates projecting on a laser from drawing on a screen.
A screen renders every pixel at once; a laser has one beam that must physically
travel the whole picture, 25 or more times a second, with a mirror that has mass
and cannot turn a corner instantly. Four consequences drive everything here:

1. **Travel between shapes is wasted time.** Paths are ordered to minimise the
   distance the blanked beam covers getting from one to the next.
2. **Brightness is dwell time.** The beam deposits light in proportion to how
   long it lingers, so points must be spaced evenly along the path or the line
   is bright where samples bunch up and dim where they spread out.
3. **Corners need warning.** The galvos overshoot a sharp turn taken at speed;
   repeating the corner point buys them time to arrive.
4. **The frame budget is hard.** 4096 points per frame, and the scan rate
   divided by the target frame rate is usually a much tighter limit than that.
   Something has to be dropped when the scene is busy, and it should be the
   least interesting thing rather than whatever happened to come last.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .dac.base import MAX_POINTS
from .frame import COORD_CENTER, LaserFrame, LaserPoint


@dataclass(slots=True)
class LaserPath:
    """One contour, already in DAC coordinates."""

    points: np.ndarray                       #: ``(N, 2)`` int array
    closed: bool = False
    color: tuple[int, int, int] = (0, 255, 0)

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float64).reshape(-1, 2)

    @property
    def length(self) -> float:
        if len(self.points) < 2:
            return 0.0
        pts = self.points
        if self.closed:
            pts = np.vstack([pts, pts[:1]])
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


@dataclass(slots=True)
class OptimiserConfig:
    """Every knob that shapes the point stream.

    The defaults are tuned for a 30 kpps scan rate on typical 30 kpps-rated
    galvos; slower scanners want a larger ``step_draw`` and a lower pps.
    """

    #: Spacing between consecutive lit samples, in DAC units. Smaller is
    #: smoother and brighter but costs points. 60 units is ~1.5% of full scale.
    step_draw: float = 60.0
    #: Spacing while blanked. The beam is off but the mirrors still travel, and
    #: an un-interpolated jump rings into the start of the next shape.
    step_blank: float = 180.0
    #: Turn angle, in degrees, above which a vertex gets dwell points.
    corner_threshold_deg: float = 25.0
    #: Degrees of turn per extra dwell point.
    corner_degrees_per_point: float = 30.0
    corner_max_points: int = 8
    #: Blanked points parked at a path's first vertex while the galvos settle.
    dwell_blank_start: int = 8
    #: Lit points at the start of a path, so its first millimetre is not faint.
    dwell_lit_start: int = 2
    dwell_lit_end: int = 2
    #: Blanked points at the last vertex before travelling away.
    dwell_blank_end: int = 6

    @property
    def path_overhead(self) -> int:
        """Fixed point cost of drawing any path at all, however short."""
        return (
            self.dwell_blank_start
            + self.dwell_lit_start
            + self.dwell_lit_end
            + self.dwell_blank_end
        )


def point_budget(pps: int, fps: int, max_points: int = MAX_POINTS) -> int:
    """How many points fit in one frame at this scan rate and frame rate.

    The frame's duration is ``n / pps``. To refresh ``fps`` times a second the
    frame can hold at most ``pps / fps`` points -- almost always a tighter bound
    than the firmware's 4096-point buffer.
    """
    if pps <= 0 or fps <= 0:
        raise ValueError("pps and fps must be positive")
    return max(1, min(int(max_points), MAX_POINTS, int(pps // fps)))


# --- ordering ----------------------------------------------------------------


def _endpoints(path: LaserPath) -> tuple[np.ndarray, np.ndarray]:
    return path.points[0], path.points[-1]


def order_paths(
    paths: list[LaserPath],
    start: tuple[float, float] = (COORD_CENTER, COORD_CENTER),
) -> list[LaserPath]:
    """Greedy nearest-neighbour ordering that minimises blanked travel.

    Open paths may be traversed in either direction; closed paths are rotated so
    that drawing begins at whichever vertex is nearest the beam's current
    position. With at most a few dozen contours the O(n^2) search is free, and
    it beats the arbitrary order ``findContours`` returns by a wide margin.

    ``start`` is the previous frame's finishing position, so consecutive frames
    do not fling the beam back to the centre between refreshes.
    """
    remaining = list(paths)
    pen = np.asarray(start, dtype=np.float64)
    ordered: list[LaserPath] = []

    while remaining:
        best_idx = 0
        best_cost = math.inf
        best_path: LaserPath | None = None

        for i, path in enumerate(remaining):
            if len(path.points) == 0:
                continue
            if path.closed:
                # Rotate the vertex ring so it starts at the nearest vertex.
                d = np.linalg.norm(path.points - pen, axis=1)
                j = int(np.argmin(d))
                cost = float(d[j])
                candidate = LaserPath(np.roll(path.points, -j, axis=0), True, path.color)
            else:
                head, tail = _endpoints(path)
                d_head = float(np.linalg.norm(head - pen))
                d_tail = float(np.linalg.norm(tail - pen))
                if d_tail < d_head:
                    cost = d_tail
                    candidate = LaserPath(path.points[::-1].copy(), False, path.color)
                else:
                    cost = d_head
                    candidate = path
            if cost < best_cost:
                best_cost, best_idx, best_path = cost, i, candidate

        if best_path is None:
            break
        ordered.append(best_path)
        pen = best_path.points[-1] if not best_path.closed else best_path.points[0]
        remaining.pop(best_idx)

    return ordered


def travel_distance(paths: list[LaserPath], start: tuple[float, float] = (COORD_CENTER, COORD_CENTER)) -> float:
    """Total blanked travel for a given path order -- the thing ordering minimises."""
    pen = np.asarray(start, dtype=np.float64)
    total = 0.0
    for p in paths:
        if len(p.points) == 0:
            continue
        total += float(np.linalg.norm(p.points[0] - pen))
        pen = p.points[0] if p.closed else p.points[-1]
    return total


# --- emission ----------------------------------------------------------------


def _turn_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle in degrees by which the direction changes at ``b``."""
    v1 = b - a
    v2 = c - b
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos = float(np.dot(v1, v2) / (n1 * n2))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _interpolate(a: np.ndarray, b: np.ndarray, step: float) -> list[np.ndarray]:
    """Points from just after ``a`` up to and including ``b``, spaced ~``step``."""
    dist = float(np.linalg.norm(b - a))
    n = max(1, int(math.ceil(dist / max(step, 1e-6))))
    return [a + (b - a) * (k / n) for k in range(1, n + 1)]


def _emit_path(path: LaserPath, cfg: OptimiserConfig, step_draw: float) -> list[LaserPoint]:
    """Render one path to lit points, with corner and end dwells."""
    verts = path.points
    if len(verts) == 0:
        return []
    if len(verts) == 1:
        # A degenerate contour would park the beam; the safety layer would
        # reject the frame, so refuse to emit it here instead.
        return []

    ring = np.vstack([verts, verts[:1]]) if path.closed else verts
    r, g, b = path.color
    out: list[LaserPoint] = []

    def lit(p: np.ndarray, dwell: bool = False) -> LaserPoint:
        return LaserPoint(int(round(p[0])), int(round(p[1])), r, g, b, blank=False, dwell=dwell)

    # Lit dwell at the entry vertex: the galvos have settled, now let the beam
    # register before it starts moving.
    out.extend(lit(ring[0], dwell=True) for _ in range(cfg.dwell_lit_start))

    for i in range(len(ring) - 1):
        out.extend(lit(p) for p in _interpolate(ring[i], ring[i + 1], step_draw))

        # Corner dwell at the vertex we just arrived at, if the path turns there.
        is_interior = i + 2 < len(ring)
        if is_interior or path.closed:
            nxt = ring[i + 2] if is_interior else ring[1]
            angle = _turn_angle(ring[i], ring[i + 1], nxt)
            if angle >= cfg.corner_threshold_deg:
                reps = max(
                    1,
                    min(
                        cfg.corner_max_points,
                        int(round(angle / cfg.corner_degrees_per_point)),
                    ),
                )
                out.extend(lit(ring[i + 1], dwell=True) for _ in range(reps))

    out.extend(lit(ring[-1], dwell=True) for _ in range(cfg.dwell_lit_end))
    return out


def _estimate_cost(path: LaserPath, cfg: OptimiserConfig, step_draw: float) -> int:
    """Approximate point cost of a path, used for budgeting before emission."""
    n_draw = int(math.ceil(path.length / max(step_draw, 1e-6)))
    # Assume roughly one corner dwell per simplified vertex that turns; two
    # points each is a reasonable average for camera contours.
    n_corner = 2 * max(0, len(path.points) - 2)
    return cfg.path_overhead + n_draw + n_corner


def build_frame(
    paths: list[LaserPath],
    budget: int,
    cfg: OptimiserConfig | None = None,
    start: tuple[float, float] = (COORD_CENTER, COORD_CENTER),
) -> tuple[LaserFrame, dict]:
    """Assemble an ordered, dwelled, budget-respecting frame.

    Returns the frame and a stats dict (``kept``, ``dropped``, ``step``,
    ``travel``) that the preview HUD displays -- when the optimiser starts
    dropping contours, the operator should be able to see that happening.
    """
    cfg = cfg or OptimiserConfig()
    budget = max(1, min(int(budget), MAX_POINTS))

    live = [p for p in paths if len(p.points) >= 2]
    if not live:
        return LaserFrame(), {"kept": 0, "dropped": len(paths), "step": cfg.step_draw, "travel": 0.0}

    # Selection: keep the visually dominant contours first, so a busy scene
    # degrades by losing detail rather than losing whichever shape sorted last.
    by_importance = sorted(live, key=lambda p: p.length, reverse=True)
    step = cfg.step_draw
    kept: list[LaserPath] = []
    spent = 0
    for path in by_importance:
        cost = _estimate_cost(path, cfg, step)
        if spent + cost > budget and kept:
            continue
        kept.append(path)
        spent += cost

    dropped = len(live) - len(kept)
    ordered = order_paths(kept, start=start)

    # Emit, then correct. The cost model is an estimate; this loop makes the
    # budget a guarantee. Coarsening the step trades smoothness for fit, which
    # is the right trade when the alternative is dropping the whole shape.
    frame = LaserFrame()
    for _ in range(4):
        frame = _assemble(ordered, cfg, step, start)
        if len(frame) <= budget:
            break
        step *= len(frame) / budget * 1.05

    while len(frame) > budget and len(ordered) > 1:
        # Last resort: the scene genuinely does not fit. Shed the smallest path.
        ordered.pop()
        dropped += 1
        frame = _assemble(ordered, cfg, step, start)

    if len(frame) > budget:
        frame = LaserFrame(frame.points[:budget])

    return frame, {
        "kept": len(ordered),
        "dropped": dropped,
        "step": step,
        "travel": travel_distance(ordered, start),
    }


def _assemble(
    ordered: list[LaserPath],
    cfg: OptimiserConfig,
    step_draw: float,
    start: tuple[float, float],
) -> LaserFrame:
    """Stitch emitted paths together with blanked travel and anchor dwells."""
    frame = LaserFrame()
    pen = np.asarray(start, dtype=np.float64)
    step_blank = max(cfg.step_blank, step_draw)

    for path in ordered:
        body = _emit_path(path, cfg, step_draw)
        if not body:
            continue
        entry = path.points[0]

        # Travel there with the beam off, interpolated so the mirrors are not
        # asked for an instantaneous jump.
        for p in _interpolate(pen, entry, step_blank):
            frame.append(LaserPoint(int(round(p[0])), int(round(p[1])), 0, 0, 0, blank=True))

        # Sit blanked on the entry vertex while the galvos settle. This is the
        # single most important detail for a clean-looking start to a line.
        for _ in range(cfg.dwell_blank_start):
            frame.append(
                LaserPoint(int(round(entry[0])), int(round(entry[1])), 0, 0, 0, blank=True, dwell=True)
            )

        frame.extend(body)

        exit_pt = body[-1]
        for _ in range(cfg.dwell_blank_end):
            frame.append(LaserPoint(exit_pt.x, exit_pt.y, 0, 0, 0, blank=True, dwell=True))
        pen = np.array([exit_pt.x, exit_pt.y], dtype=np.float64)

    # A frame that begins lit would flash on the travel move from wherever the
    # previous frame ended. Guarantee the first point is dark.
    if frame.points and frame.points[0].lit():
        first = frame.points[0]
        frame.points.insert(0, LaserPoint(first.x, first.y, 0, 0, 0, blank=True, dwell=True))

    return frame


__all__ = [
    "LaserPath",
    "OptimiserConfig",
    "build_frame",
    "order_paths",
    "point_budget",
    "travel_distance",
]
