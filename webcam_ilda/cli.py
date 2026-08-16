"""Command-line entry point: argument parsing, wiring, and the main loop.

Thread layout, and why:

* **capture** reads the source as fast as it will go and keeps only the newest
  frame, so network stream latency cannot accumulate;
* **main** does vision, geometry, path optimisation and preview, then hands the
  finished frame over;
* **pump** owns the DAC exclusively, validates every frame, and blanks on stall.

The main loop is deliberately allowed to be slow. The Helios loops its current
frame, so a 12 fps vision pipeline still produces a rock-steady 25 fps
projection -- the two rates are decoupled by design.
"""

from __future__ import annotations

import argparse
import atexit
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import __version__, ilda
from .capture import CaptureError, CaptureThread, FrameSource
from .color import MODES as COLOR_MODES, assign_colors, parse_color
from .config import AppConfig, ConfigError
from .dac import BACKENDS, DacError, open_dac
from .frame import LaserFrame
from .geometry import image_to_dac
from .pathopt import LaserPath, build_frame, point_budget
from .patterns import PATTERNS, test_pattern
from .preview import HOTKEY_HELP, Preview
from .pump import DacPump
from .safety import arm
from .vision import MODES as VISION_MODES, VisionPipeline, contour_is_closed

log = logging.getLogger("webcam_ilda")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="webcam-ilda",
        description="Draw what a camera sees on an ILDA laser projector, via a Helios DAC.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=HOTKEY_HELP,
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    src = p.add_argument_group("source")
    src.add_argument(
        "--source", default=None,
        help="webcam index (0), rtsp:// or http:// URL, video/image file, or 'synthetic'",
    )
    src.add_argument(
        "--source-kind", default=None, choices=["auto", "snapshot"],
        help="force an http:// URL to be treated as a still-image snapshot endpoint",
    )
    src.add_argument("--ild", default=None, help="project an ILDA file instead of a camera")
    src.add_argument("--test-pattern", default=None, choices=list(PATTERNS),
                     help="project an alignment pattern instead of a camera")

    out = p.add_argument_group("output")
    out.add_argument("--dac", default=None, choices=list(BACKENDS), help="DAC backend (default: auto)")
    out.add_argument("--device", type=int, default=None, help="DAC device index (default: 0)")
    out.add_argument("--dll-path", default=None, help="path to HeliosLaserDAC.dll")
    out.add_argument("--dry-run", action="store_true",
                     help="use the simulator: no hardware needed, everything else identical")
    out.add_argument("--pps", type=int, default=None, help="scan rate in points per second (default: 30000)")
    out.add_argument("--fps", type=int, default=None, help="target projected frame rate (default: 25)")
    out.add_argument("--max-points", type=int, default=None, help="hard point cap per frame (<=4096)")
    out.add_argument("--record", default=None, metavar="FILE.ild", help="also write frames to an ILDA file")

    look = p.add_argument_group("appearance")
    look.add_argument("--mode", default=None, choices=list(VISION_MODES), help="extraction mode")
    look.add_argument("--color", default=None, metavar="R,G,B", help="fixed draw colour (default: 0,255,0)")
    look.add_argument("--color-mode", default=None, choices=list(COLOR_MODES))
    look.add_argument("--scale", type=float, default=None, metavar="PCT", help="projected size, %% of full scale")
    look.add_argument("--flip-x", action="store_true")
    look.add_argument("--flip-y", action="store_true")
    look.add_argument("--rotate", type=int, default=None, choices=[0, 90, 180, 270])
    look.add_argument("--simplify", type=float, default=None, metavar="PX",
                      help="Douglas-Peucker tolerance in pixels (default: 2.0)")

    safe = p.add_argument_group("safety")
    safe.add_argument("--max-brightness", type=float, default=None, metavar="PCT",
                      help="optical power ceiling, %% of full scale (default: 50)")
    safe.add_argument("--ack-safety", action="store_true",
                      help="acknowledge the safety briefing without the interactive prompt")

    run = p.add_argument_group("runtime")
    run.add_argument("--headless", action="store_true", help="no preview window")
    run.add_argument("--frames", type=int, default=None, help="process N frames then exit")
    run.add_argument("--config", default=None, help=f"config file (default: webcam_ilda.yaml)")
    run.add_argument("--save-config", action="store_true", help="write the resolved settings and exit")
    run.add_argument("--log-level", default="INFO",
                     choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def apply_overrides(cfg: AppConfig, args: argparse.Namespace) -> AppConfig:
    """CLI arguments win over the config file; unset arguments change nothing."""
    if args.source is not None:
        cfg.source = args.source
    if args.source_kind is not None:
        cfg.source_kind = args.source_kind
    if args.dac is not None:
        cfg.dac = args.dac
    if args.dry_run:
        cfg.dac = "sim"
    if args.device is not None:
        cfg.device = args.device
    if args.dll_path is not None:
        cfg.dll_path = args.dll_path
    if args.pps is not None:
        cfg.pps = args.pps
    if args.fps is not None:
        cfg.fps = args.fps
    if args.max_points is not None:
        cfg.max_points = min(4096, max(1, args.max_points))
    if args.mode is not None:
        cfg.vision.mode = args.mode
    if args.color is not None:
        cfg.color = parse_color(args.color)
    if args.color_mode is not None:
        cfg.color_mode = args.color_mode
    if args.scale is not None:
        cfg.calibration.scale_pct = args.scale
    if args.flip_x:
        cfg.calibration.flip_x = not cfg.calibration.flip_x
    if args.flip_y:
        cfg.calibration.flip_y = not cfg.calibration.flip_y
    if args.rotate is not None:
        cfg.calibration.rotate = args.rotate
    if args.simplify is not None:
        cfg.vision.simplify_px = args.simplify
    if args.max_brightness is not None:
        cfg.safety.max_brightness = args.max_brightness / 100.0
    if args.headless:
        cfg.headless = True
    return cfg


def contours_to_paths(
    contours: list[np.ndarray],
    image: np.ndarray,
    cfg: AppConfig,
) -> list[LaserPath]:
    """Map extracted contours into DAC space and colour them."""
    h, w = image.shape[:2]
    colors = assign_colors(contours, cfg.color_mode, cfg.color, image)
    paths: list[LaserPath] = []
    for contour, color in zip(contours, colors):
        pts = image_to_dac(contour, w, h, cfg.calibration)
        if len(pts) >= 2:
            paths.append(LaserPath(pts, closed=contour_is_closed(contour), color=color))
    return paths


def _hud(cfg: AppConfig, pump: DacPump, stats: dict, n_points: int, cv_fps: float) -> str:
    state = "MUTED" if pump.muted else "ARMED"
    return (
        f"{state} | {cfg.pps} pps | {n_points} pts | "
        f"scan {pump.scan_fps:5.1f} fps | cv {cv_fps:4.1f} fps | "
        f"{cfg.vision.mode} | paths {stats.get('kept', 0)}"
        f"(-{stats.get('dropped', 0)}) | bright {int(cfg.safety.max_brightness * 100)}%"
    )


def _handle_key(key: int, cfg: AppConfig, vision: VisionPipeline, preview: Preview, pump: DacPump) -> str:
    """Apply a hotkey. Returns 'quit', 'freeze', or ''."""
    if key in (27,):  # ESC
        return "quit"
    ch = chr(key) if 32 <= key < 127 else ""

    if ch == " ":
        pump.set_muted(not pump.muted)
    elif ch == "m":
        idx = VISION_MODES.index(cfg.vision.mode)
        cfg.vision.mode = VISION_MODES[(idx + 1) % len(VISION_MODES)]
        log.info("mode: %s", cfg.vision.mode)
    elif ch == "[":
        cfg.vision.canny_lo = max(0, cfg.vision.canny_lo - 10)
    elif ch == "]":
        cfg.vision.canny_lo = min(255, cfg.vision.canny_lo + 10)
    elif ch == "{":
        cfg.vision.canny_hi = max(0, cfg.vision.canny_hi - 10)
    elif ch == "}":
        cfg.vision.canny_hi = min(255, cfg.vision.canny_hi + 10)
    elif ch == "a":
        cfg.vision.auto_threshold = not cfg.vision.auto_threshold
    elif ch == "f":
        cfg.calibration.flip_x = not cfg.calibration.flip_x
    elif ch == "g":
        cfg.calibration.flip_y = not cfg.calibration.flip_y
    elif ch == "r":
        cfg.calibration.rotate = (cfg.calibration.rotate + 90) % 360
    elif ch == "-":
        cfg.calibration.scale_pct = max(5.0, cfg.calibration.scale_pct - 5.0)
    elif ch == "=":
        cfg.calibration.scale_pct = min(100.0, cfg.calibration.scale_pct + 5.0)
    elif ch == ",":
        cfg.safety.max_brightness = max(0.0, cfg.safety.max_brightness - 0.05)
    elif ch == ".":
        cfg.safety.max_brightness = min(1.0, cfg.safety.max_brightness + 0.05)
    elif ch == "t":
        preview.show_travel = not preview.show_travel
    elif ch == "c":
        preview.show_camera = not preview.show_camera
    elif ch == "b":
        vision.reset_background()
    elif ch == "p":
        return "freeze"
    elif ch == "s":
        path = cfg.save()
        log.info("saved settings to %s", path)
    return ""


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        cfg = apply_overrides(AppConfig.load(args.config), args)
    except (ConfigError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.save_config:
        print(f"wrote {cfg.save(args.config)}")
        return 0

    simulated = cfg.dac == "sim"
    if not simulated:
        interactive = sys.stdin is not None and sys.stdin.isatty()
        if not arm(interactive=interactive, acknowledged=args.ack_safety):
            return 1
    elif not args.ack_safety:
        log.info("simulator backend: no laser output, safety prompt skipped")

    try:
        dac = open_dac(cfg.dac, device=cfg.device, dll_path=cfg.dll_path)
    except DacError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    log.info("DAC: %s", dac.name)

    pump = DacPump(dac, pps=cfg.pps, safety=cfg.safety, max_points=cfg.max_points)
    shutdown_done = {"done": False}

    def shutdown() -> None:
        if shutdown_done["done"]:
            return
        shutdown_done["done"] = True
        log.info("blanking output and closing DAC")
        try:
            pump.stop()
        finally:
            dac.close()

    # Every exit path converges here: normal return, signal, unhandled exception,
    # interpreter teardown. A laser that stays lit because of an unhandled
    # KeyError is not an acceptable failure mode.
    atexit.register(shutdown)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        except (ValueError, OSError):
            pass

    try:
        dac.set_shutter(True)
        pump.start()
        if args.ild:
            return _run_ilda(args, cfg, pump)
        if args.test_pattern:
            return _run_pattern(args, cfg, pump)
        return _run_camera(args, cfg, pump)
    except KeyboardInterrupt:
        log.info("interrupted")
        return 0
    except CaptureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4
    finally:
        shutdown()


def _finish_recording(frames: list[LaserFrame], path: Optional[str]) -> None:
    if path and frames:
        ilda.write(path, frames)
        log.info("wrote %d frames to %s", len(frames), path)


def _run_pattern(args: argparse.Namespace, cfg: AppConfig, pump: DacPump) -> int:
    """Project a static alignment pattern. The safest thing to switch on with."""
    paths = test_pattern(args.test_pattern, cfg.calibration.scale_pct, cfg.color)
    budget = point_budget(cfg.pps, cfg.fps, cfg.max_points)
    frame, stats = build_frame(paths, budget, cfg.optimiser)
    log.info("test pattern %r: %d points (budget %d)", args.test_pattern, len(frame), budget)

    preview = None if cfg.headless else Preview(cfg.preview_size, cfg.show_travel, cfg.show_camera)
    deadline = args.frames if args.frames else None
    count = 0
    try:
        while True:
            pump.submit(frame)
            count += 1
            if deadline and count >= deadline:
                return 0
            if preview is not None:
                key = preview.show(preview.render(frame, _hud(cfg, pump, stats, len(frame), 0.0)))
                if key != 255 and _handle_key(key, cfg, VisionPipeline(cfg.vision), preview, pump) == "quit":
                    return 0
            else:
                time.sleep(0.05)
    finally:
        if preview is not None:
            preview.close()


def _run_ilda(args: argparse.Namespace, cfg: AppConfig, pump: DacPump) -> int:
    """Play an ILDA file. Useful for first light and for checking calibration."""
    frames = ilda.load_laser_frames(args.ild)
    if not frames:
        print(f"error: no frames in {args.ild}", file=sys.stderr)
        return 4
    log.info("loaded %d frame(s) from %s", len(frames), args.ild)

    preview = None if cfg.headless else Preview(cfg.preview_size, cfg.show_travel, cfg.show_camera)
    interval = 1.0 / max(1, cfg.fps)
    count = 0
    try:
        while True:
            for frame in frames:
                pump.submit(frame)
                count += 1
                if args.frames and count >= args.frames:
                    return 0
                if preview is not None:
                    hud = f"ILDA {Path(args.ild).name} | {len(frame)} pts | {cfg.pps} pps"
                    key = preview.show(preview.render(frame, hud))
                    if key != 255 and _handle_key(key, cfg, VisionPipeline(cfg.vision), preview, pump) == "quit":
                        return 0
                else:
                    time.sleep(interval)
    finally:
        if preview is not None:
            preview.close()


def _run_camera(args: argparse.Namespace, cfg: AppConfig, pump: DacPump) -> int:
    """The main event: camera in, laser out."""
    source = FrameSource.create(cfg.source, cfg.source_kind)
    log.info("source: %s", source.description)
    capture = CaptureThread(source).start()

    vision = VisionPipeline(cfg.vision)
    preview = None if cfg.headless else Preview(cfg.preview_size, cfg.show_travel, cfg.show_camera)
    record_path: Optional[str] = args.record
    recorded: list[LaserFrame] = []

    budget = point_budget(cfg.pps, cfg.fps, cfg.max_points)
    pen = (2048.0, 2048.0)
    frozen: Optional[LaserFrame] = None
    processed = 0
    cv_fps = 0.0
    t_last = time.monotonic()

    # Wait briefly for the first frame rather than spinning on None.
    deadline = time.monotonic() + 10.0
    while capture.latest() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if capture.latest() is None:
        capture.stop()
        raise CaptureError(f"no frames from {source.description} after 10s")

    try:
        while True:
            image = capture.latest()
            if image is None:
                time.sleep(0.01)
                continue

            if frozen is not None:
                frame, stats = frozen, {"kept": 0, "dropped": 0}
            else:
                vision.config = cfg.vision
                contours, working = vision.extract(image)
                paths = contours_to_paths(contours, working, cfg)
                frame, stats = build_frame(paths, budget, cfg.optimiser, start=pen)
                if frame.points:
                    last = frame.points[-1]
                    pen = (float(last.x), float(last.y))

            pump.submit(frame)
            if record_path:
                recorded.append(frame.copy())

            processed += 1
            now = time.monotonic()
            dt = now - t_last
            if dt > 0:
                cv_fps = 0.85 * cv_fps + 0.15 * (1.0 / dt) if cv_fps else 1.0 / dt
            t_last = now

            if args.frames and processed >= args.frames:
                return 0

            if preview is not None:
                warning = pump.last_error or ""
                canvas = preview.render(
                    frame, _hud(cfg, pump, stats, len(frame), cv_fps),
                    camera=image,
                    warning=warning,
                )
                key = preview.show(canvas)
                if key != 255:
                    action = _handle_key(key, cfg, vision, preview, pump)
                    if action == "quit":
                        return 0
                    if action == "freeze":
                        frozen = None if frozen is not None else frame
                        log.info("frame %s", "released" if frozen is None else "frozen")
            else:
                time.sleep(0.005)
    finally:
        capture.stop()
        if preview is not None:
            preview.close()
        _finish_recording(recorded, record_path)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
