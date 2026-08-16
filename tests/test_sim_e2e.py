"""End-to-end runs against the simulator: no camera, no DAC, no laser."""

from __future__ import annotations

import numpy as np
import pytest

from webcam_ilda.cli import contours_to_paths, main
from webcam_ilda.config import AppConfig
from webcam_ilda.dac.base import MAX_POINTS
from webcam_ilda.dac.simulator import SimulatorDac
from webcam_ilda.frame import LaserFrame, LaserPoint
from webcam_ilda.pathopt import build_frame, point_budget
# Imported under an alias: pytest would otherwise collect `test_pattern` as a test.
from webcam_ilda.patterns import PATTERNS, SyntheticSource
from webcam_ilda.patterns import test_pattern as build_test_pattern
from webcam_ilda.pump import DacPump
from webcam_ilda.safety import SafetyConfig, validate
from webcam_ilda.vision import VisionPipeline


def test_synthetic_source_through_the_whole_pipeline():
    cfg = AppConfig()
    source = SyntheticSource()
    vision = VisionPipeline(cfg.vision)
    budget = point_budget(cfg.pps, cfg.fps, cfg.max_points)

    for _ in range(5):
        contours, working = vision.extract(source.read())
        assert contours, "synthetic scene should always yield contours"
        paths = contours_to_paths(contours, working, cfg)
        frame, _stats = build_frame(paths, budget, cfg.optimiser)

        assert frame.lit_points
        assert len(frame) <= budget
        assert all(0 <= p.x <= 4095 and 0 <= p.y <= 4095 for p in frame)
        assert frame.points[0].blank


@pytest.mark.parametrize("mode", ["canny", "threshold", "motion", "color"])
def test_every_extraction_mode_runs(mode):
    cfg = AppConfig()
    cfg.vision.mode = mode
    vision = VisionPipeline(cfg.vision)
    source = SyntheticSource()
    for _ in range(3):
        contours, working = vision.extract(source.read())
        paths = contours_to_paths(contours, working, cfg)
        frame, _ = build_frame(paths, 1200, cfg.optimiser)
        assert len(frame) <= 1200


@pytest.mark.parametrize("name", PATTERNS)
def test_test_patterns_build_valid_frames(name):
    frame, _ = build_frame(build_test_pattern(name), 2000)
    assert frame.lit_points
    assert all(0 <= p.x <= 4095 and 0 <= p.y <= 4095 for p in frame)


def test_simulator_enforces_hardware_limits():
    dac = SimulatorDac()
    dac.open()
    with pytest.raises(Exception):
        dac.write_frame(LaserFrame([LaserPoint(5000, 0)]), 30000)
    with pytest.raises(Exception):
        dac.write_frame(LaserFrame([LaserPoint(0, 0) for _ in range(MAX_POINTS + 1)]), 30000)


def test_simulator_models_frame_duration():
    """Readiness should track n/pps, so the pump paces the same as on hardware."""
    now = [0.0]
    dac = SimulatorDac(clock=lambda: now[0])
    dac.open()
    assert dac.ready()
    dac.write_frame(LaserFrame([LaserPoint(i, i, 0, 255, 0) for i in range(1000)]), 1000)
    assert not dac.ready()          # a 1000-point frame at 1000 pps lasts 1 s
    now[0] = 1.01
    assert dac.ready()


def test_pump_validates_and_blanks_a_hazardous_frame():
    dac = SimulatorDac()
    dac.open()
    pump = DacPump(dac, pps=30000, safety=SafetyConfig(max_brightness=1.0))
    parked = LaserFrame([LaserPoint(2048, 2048, 0, 255, 0) for _ in range(200)])
    pump._write(validate(parked, pump.safety, pump.max_points), 0)
    assert dac.last_frame is not None
    assert not dac.last_frame.lit_points


def test_pump_blank_and_stop_leaves_output_dark():
    dac = SimulatorDac()
    dac.open()
    dac.set_shutter(True)
    pump = DacPump(dac, pps=30000)
    pump.blank_and_stop()
    assert dac.stopped
    assert dac.shutter_open is False
    assert not dac.last_frame.lit_points


def test_cli_dry_run_headless_returns_zero():
    rc = main(
        [
            "--dry-run", "--headless", "--ack-safety",
            "--source", "synthetic", "--frames", "3",
            "--log-level", "WARNING",
        ]
    )
    assert rc == 0


def test_cli_test_pattern_dry_run_returns_zero():
    rc = main(
        [
            "--dry-run", "--headless", "--ack-safety",
            "--test-pattern", "square", "--frames", "2",
            "--log-level", "WARNING",
        ]
    )
    assert rc == 0


def test_cli_projects_the_ilda_fixture():
    from pathlib import Path

    fixture = Path(__file__).parent / "data" / "ILDATEST.ILD"
    rc = main(
        [
            "--dry-run", "--headless", "--ack-safety",
            "--ild", str(fixture), "--frames", "2",
            "--log-level", "WARNING",
        ]
    )
    assert rc == 0


def test_cli_rejects_a_bad_colour():
    rc = main(["--dry-run", "--headless", "--ack-safety", "--color", "nope", "--frames", "1"])
    assert rc != 0
