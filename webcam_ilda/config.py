"""Runtime configuration: one dataclass, loadable from and savable to YAML.

Calibration is the reason this exists. Working out the flip, rotation and scale
that make the projection land correctly is a fiddly job done once, standing at
the projector, and it should not have to be redone at every launch. Pressing
``s`` in the preview writes the current tuning back to ``webcam_ilda.yaml``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .color import parse_color
from .geometry import Calibration
from .pathopt import OptimiserConfig
from .safety import SafetyConfig
from .vision import VisionConfig

DEFAULT_CONFIG_NAME = "webcam_ilda.yaml"


class ConfigError(ValueError):
    """Raised with a message naming the offending key."""


@dataclass
class AppConfig:
    """Everything that can be set from the CLI or the config file."""

    source: str = "0"
    source_kind: str = "auto"
    dac: str = "auto"
    device: int = 0
    dll_path: str | None = None

    pps: int = 30000
    fps: int = 25
    max_points: int = 4096

    color: tuple[int, int, int] = (0, 255, 0)
    color_mode: str = "fixed"

    headless: bool = False
    preview_size: int = 720
    show_travel: bool = True
    show_camera: bool = True

    vision: VisionConfig = field(default_factory=VisionConfig)
    calibration: Calibration = field(default_factory=Calibration)
    optimiser: OptimiserConfig = field(default_factory=OptimiserConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    # -- serialisation --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["color"] = list(self.color)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        if not isinstance(data, dict):
            raise ConfigError("config file must contain a mapping at the top level")

        nested = {
            "vision": VisionConfig,
            "calibration": Calibration,
            "optimiser": OptimiserConfig,
            "safety": SafetyConfig,
        }
        kwargs: dict[str, Any] = {}
        known = {f.name for f in fields(cls)}

        for key, value in data.items():
            if key not in known:
                raise ConfigError(f"unknown key {key!r}")
            if key in nested:
                if not isinstance(value, dict):
                    raise ConfigError(f"{key}: expected a mapping")
                sub_cls = nested[key]
                sub_known = {f.name for f in fields(sub_cls)}
                bad = set(value) - sub_known
                if bad:
                    raise ConfigError(f"{key}: unknown key(s) {', '.join(sorted(bad))}")
                try:
                    kwargs[key] = sub_cls(**value)
                except (TypeError, ValueError) as exc:
                    raise ConfigError(f"{key}: {exc}") from exc
            elif key == "color":
                kwargs[key] = _coerce_color(value)
            else:
                kwargs[key] = value

        try:
            return cls(**kwargs)
        except (TypeError, ValueError) as exc:
            raise ConfigError(str(exc)) from exc

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AppConfig":
        """Load from ``path``, or return defaults when the file is absent."""
        p = Path(path) if path else Path(DEFAULT_CONFIG_NAME)
        if not p.exists():
            return cls()
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"could not parse {p}: {exc}") from exc
        try:
            return cls.from_dict(raw)
        except ConfigError as exc:
            raise ConfigError(f"config error in {p}: {exc}") from exc

    def save(self, path: str | Path | None = None) -> Path:
        p = Path(path) if path else Path(DEFAULT_CONFIG_NAME)
        p.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        return p


def _coerce_color(value: Any) -> tuple[int, int, int]:
    if isinstance(value, str):
        return parse_color(value)
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(int(v) for v in value)  # type: ignore[return-value]
    raise ConfigError("color: expected 'R,G,B' or a three-element list")


__all__ = ["DEFAULT_CONFIG_NAME", "AppConfig", "ConfigError"]
