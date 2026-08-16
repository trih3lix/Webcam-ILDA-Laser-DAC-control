"""Beam-safety controls that live in code, not only in the README.

The honest framing first, because it matters more than any of the code below:

**This is not a scan-fail safeguard.** A software watchdog cannot detect a
seized galvanometer. If a mirror stops moving, the DAC still reports ready and
still accepts frames, and the beam sits in one place at full power. Only
hardware -- the projector's own scan-fail circuit and the ILDA connector's
interlock loop -- can catch that. What follows is a set of *contributing*
controls that make the software side of the system fail dark.

What it does do:

* refuses to emit a frame that would park the beam;
* refuses to emit a frame that has collapsed to a point;
* caps optical power at the point of no return, so nothing upstream can bypass it;
* blanks output when the pipeline stalls;
* blanks on every exit path there is.

The last one carries more weight than it looks. The Helios loops the last frame
it was given until it is replaced, so a host crash leaves the projector running
that frame forever. The validators therefore guarantee that *any* frame which
could end up looping is a moving, power-capped pattern.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .frame import COORD_MAX, COORD_MIN, LaserFrame, blank_frame

log = logging.getLogger(__name__)

SAFETY_BANNER = """
================================ LASER SAFETY =================================
 This software drives a real laser projector. Before arming:

  * Never look into the aperture, and never place your eye in the beam path --
    including reflections off glass, polished metal, or a phone screen.
  * Do not scan an audience. Keep beams above head height, or terminate them on
    a surface. Audience scanning is a regulated activity requiring a variance.
  * Check the whole beam path, including where it lands beyond your target.
  * Have a physical means to kill output that does not depend on this software:
    an e-stop, an interlock, or a hand on the power switch.
  * Confirm the projector's own scan-fail safeguard and interlock loop work.
    Software cannot detect a stalled scanner. This program is not a substitute.

 Emergency stop: press ESC in the preview window, or Ctrl+C in the terminal.
===============================================================================
""".strip()


@dataclass(slots=True)
class SafetyConfig:
    """Limits enforced on every frame, immediately before it is written."""

    #: Optical power ceiling as a fraction of full scale. Applied last, in the
    #: pump, so no upstream stage can raise it.
    max_brightness: float = 0.5
    #: Longest run of identical consecutive lit points tolerated. At 30 kpps,
    #: 12 points is 0.4 ms -- long enough for legitimate corner dwells, far too
    #: short to be a parked beam.
    max_static_run: int = 12
    #: Minimum bounding-box diagonal of the lit content, as a fraction of full
    #: scale. A picture that has degenerated to a dot is the hazard case.
    min_extent_frac: float = 0.02
    #: Seconds without a fresh frame before the pump blanks output.
    watchdog_s: float = 2.0

    def __post_init__(self) -> None:
        self.max_brightness = max(0.0, min(1.0, float(self.max_brightness)))


class FrameRejected(Exception):
    """Internal signal that a frame failed validation. Carries the reason."""


def check_bounds(frame: LaserFrame) -> None:
    for i, p in enumerate(frame.points):
        if not (COORD_MIN <= p.x <= COORD_MAX and COORD_MIN <= p.y <= COORD_MAX):
            raise FrameRejected(f"point {i} at ({p.x}, {p.y}) is outside the projector field")


def check_static_beam(frame: LaserFrame, cfg: SafetyConfig) -> None:
    """Reject a frame in which the lit beam stops moving for too long."""
    run = 0
    prev = None
    for i, p in enumerate(frame.points):
        if not p.lit():
            run = 0
            prev = None
            continue
        if prev is not None and p.same_position(prev):
            run += 1
            if run >= cfg.max_static_run:
                raise FrameRejected(
                    f"lit beam stationary at ({p.x}, {p.y}) for {run + 1} consecutive "
                    f"points (limit {cfg.max_static_run})"
                )
        else:
            run = 0
        prev = p


def check_extent(frame: LaserFrame, cfg: SafetyConfig) -> None:
    """Reject a frame whose lit content has collapsed towards a single point."""
    lit = frame.lit_points
    if not lit:
        return  # A fully blanked frame is safe by definition.
    min_x, min_y, max_x, max_y = frame.bounds(lit_only=True)
    diag = ((max_x - min_x) ** 2 + (max_y - min_y) ** 2) ** 0.5
    limit = cfg.min_extent_frac * (COORD_MAX - COORD_MIN)
    if diag < limit:
        raise FrameRejected(
            f"lit content spans only {diag:.0f} DAC units (minimum {limit:.0f}); "
            "the beam would be effectively stationary"
        )


def check_point_count(frame: LaserFrame, max_points: int) -> None:
    if len(frame) == 0:
        raise FrameRejected("frame is empty")
    if len(frame) > max_points:
        raise FrameRejected(f"frame has {len(frame)} points, limit is {max_points}")


def validate(frame: LaserFrame, cfg: SafetyConfig, max_points: int) -> LaserFrame:
    """Return a frame that is safe to write, or a blank one.

    Never raises: the pump's job is to keep the scanners fed, and the correct
    response to a bad frame is darkness, not a stalled output thread. The reason
    is logged so a persistent rejection is visible rather than silent.

    The brightness cap is applied *after* validation and to a copy, so the cap
    cannot be defeated by any upstream stage and the caller's frame is untouched.
    """
    try:
        check_point_count(frame, max_points)
        check_bounds(frame)
        check_static_beam(frame, cfg)
        check_extent(frame, cfg)
    except FrameRejected as exc:
        log.warning("frame rejected, blanking output: %s", exc)
        return blank_frame()

    safe = frame.copy()
    if cfg.max_brightness < 1.0:
        safe.scale_brightness(cfg.max_brightness)
    return safe


def arm(interactive: bool = True, acknowledged: bool = False) -> bool:
    """Show the safety banner and require an explicit arming step.

    Returns True if output may be enabled. ``--ack-safety`` sets
    ``acknowledged`` and skips the prompt -- the banner still prints, because
    somebody else may be reading the terminal.
    """
    print(SAFETY_BANNER)
    if acknowledged:
        print("\nSafety acknowledged via --ack-safety. Arming laser output.\n")
        return True
    if not interactive:
        print("\nRefusing to arm without --ack-safety in a non-interactive session.\n")
        return False
    try:
        response = input("\nType 'arm' and press Enter to enable laser output: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    if response != "arm":
        print("Not armed. Exiting.\n")
        return False
    print("Armed.\n")
    return True


__all__ = [
    "SAFETY_BANNER",
    "FrameRejected",
    "SafetyConfig",
    "arm",
    "check_bounds",
    "check_extent",
    "check_point_count",
    "check_static_beam",
    "validate",
]
