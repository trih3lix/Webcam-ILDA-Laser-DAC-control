"""The lingua franca between pipeline stages: points and frames in DAC space.

Every stage after :mod:`webcam_ilda.geometry` speaks in these types. Coordinates
are always DAC units -- 12-bit, ``0..4095``, origin bottom-left, y up. Colours are
8-bit per channel. A blanked point still has a position (the galvos travel to it)
but emits no light.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Iterator, Sequence

# 12-bit DAC coordinate space.
COORD_MIN = 0
COORD_MAX = 4095
COORD_CENTER = 2048
COORD_SPAN = COORD_MAX - COORD_MIN


@dataclass(slots=True)
class LaserPoint:
    """A single sample handed to the galvos.

    ``blank`` is authoritative: a blanked point is emitted with all colour
    channels forced to zero at encode time, whatever ``r``/``g``/``b`` hold.
    Keeping the colour around makes it possible to un-blank a travel move for
    debugging without losing the path's colour.
    """

    x: int
    y: int
    r: int = 0
    g: int = 255
    b: int = 0
    blank: bool = False
    #: Dwell points are exempt from resampling -- they exist precisely because
    #: the beam should sit still there for a moment while the galvos catch up.
    dwell: bool = False

    @property
    def intensity(self) -> int:
        """The Helios ``i`` channel. Blanked points are dark by construction."""
        if self.blank:
            return 0
        return max(self.r, self.g, self.b)

    def lit(self) -> bool:
        return not self.blank

    def with_color(self, r: int, g: int, b: int) -> "LaserPoint":
        return replace(self, r=r, g=g, b=b)

    def same_position(self, other: "LaserPoint") -> bool:
        return self.x == other.x and self.y == other.y


@dataclass(slots=True)
class LaserFrame:
    """An ordered point stream, ready for a DAC backend.

    A frame is what the Helios loops on the galvos until it is replaced, so a
    frame is a complete closed thought: it starts blanked, draws, and leaves the
    beam somewhere predictable.
    """

    points: list[LaserPoint] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self) -> Iterator[LaserPoint]:
        return iter(self.points)

    def __getitem__(self, idx: int) -> LaserPoint:
        return self.points[idx]

    def append(self, point: LaserPoint) -> None:
        self.points.append(point)

    def extend(self, points: Iterable[LaserPoint]) -> None:
        self.points.extend(points)

    @property
    def lit_points(self) -> list[LaserPoint]:
        return [p for p in self.points if p.lit()]

    def bounds(self, lit_only: bool = True) -> tuple[int, int, int, int]:
        """``(min_x, min_y, max_x, max_y)`` of the frame; zeros if empty."""
        pts: Sequence[LaserPoint] = self.lit_points if lit_only else self.points
        if not pts:
            return (0, 0, 0, 0)
        xs = [p.x for p in pts]
        ys = [p.y for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))

    def scale_brightness(self, factor: float) -> None:
        """Clamp every colour channel in place. Used by the safety validator."""
        factor = max(0.0, min(1.0, factor))
        for p in self.points:
            p.r = int(p.r * factor)
            p.g = int(p.g * factor)
            p.b = int(p.b * factor)

    def copy(self) -> "LaserFrame":
        return LaserFrame([replace(p) for p in self.points])


def blank_frame(n: int = 16, x: int = COORD_CENTER, y: int = COORD_CENTER) -> LaserFrame:
    """A frame that emits no light.

    This is the value written on shutdown, on watchdog expiry, and in place of
    any frame the safety validator rejects. It has real points rather than being
    empty because the Helios needs a frame to loop; an empty write is an error.
    """
    return LaserFrame([LaserPoint(x, y, 0, 0, 0, blank=True) for _ in range(n)])
