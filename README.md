# Webcam → ILDA Laser

Point a camera at the world and a laser draws what it sees.

Live frames — a webcam, an RTSP camera, a video file, or a still — go through an
OpenCV pipeline that finds the lines worth drawing, and those lines become a
real laser point stream on an ILDA projector driven by a
[Helios DAC](https://bitlasers.com/helios-laser-dac/).

> ## ⚠ This drives a real laser
>
> A laser projector can cause permanent eye injury faster than you can blink,
> and a Class 4 projector can do it from a reflection off a window. Before you
> connect anything:
>
> - **Never look into the aperture.** Assume every reflective surface in the
>   room — glass, polished metal, a phone screen — is part of the beam path.
> - **Do not scan an audience.** Keep beams above head height or terminate them
>   on a surface. Audience scanning is a regulated activity that requires a
>   variance in most jurisdictions.
> - **Have a hardware kill.** An e-stop, an interlock, or a hand on the power
>   switch — something that does not depend on this software running correctly.
> - **This software is not a scan-fail safeguard.** It cannot detect a stalled
>   galvanometer. Read [Safety](#safety) before you switch anything on.
>
> Start every session with `--test-pattern square --max-brightness 20`.

## Features

- **Any camera, or none.** Webcam index, `rtsp://` stream, HTTP snapshot
  endpoint, video file, still image, or a built-in synthetic scene.
- **Four extraction modes** — `canny` edges, adaptive `threshold` silhouettes,
  `motion` (background subtraction, so the laser outlines only what moves), and
  an HSV `color` key. Switch between them live with `m`.
- **Real laser path optimisation**, not just contour plotting: nearest-neighbour
  path ordering to minimise blanked travel, constant-arc-length resampling for
  even brightness, angle-proportional corner dwell, blanked anchor dwell at path
  entry and exit, and a hard point budget with graceful degradation.
- **Safety enforced in code** — an arming step, a stall watchdog, a brightness
  ceiling, and a validator that refuses any frame which would park the beam.
- **Runs with no hardware.** `--dry-run` swaps in a simulator that enforces the
  same limits and models frame timing, so the whole pipeline — and the entire
  test suite — works on a laptop with nothing plugged in.
- **ILDA in and out.** Read `.ild` files to project them; `--record` writes a
  session back out as ILDA.
- **Live tuning.** Thresholds, flips, rotation, scale and brightness are all
  hotkeys, and `s` saves them to `webcam_ilda.yaml`.

## Hardware

| Part | Notes |
| --- | --- |
| Helios Laser DAC | USB `1209:E500`. The Windows SDK DLL ships vendored in this repo (x64, MIT — see [THIRD_PARTY.md](THIRD_PARTY.md)). |
| ILDA projector | Any projector with a standard DB25 ILDA input. Match `--pps` to your scanners' rating. |
| Camera | Optional. Anything OpenCV opens, or an RTSP/HTTP camera on the network. |

## Requirements

- Python 3.10+
- Windows, Linux or macOS. The vendored DLL is Windows/x64; other platforms use
  the `[usb]` extra.
- `numpy`, `opencv-python`, `PyYAML` (installed automatically).

## Install

```bash
pip install .
```

```bash
pip install ".[usb]"     # adds the pyusb backend (needed on Linux/macOS)
```

```bash
pip install ".[test]"    # adds pytest
```

## Quickstart

Nothing plugged in — check the pipeline works:

```bash
webcam-ilda --dry-run --source synthetic
```

First light, at low power, with known geometry:

```bash
webcam-ilda --test-pattern square --max-brightness 20 --pps 30000
```

Project a known-good ILDA file to check calibration:

```bash
webcam-ilda --ild tests/data/ILDATEST.ILD --max-brightness 20
```

A camera:

```bash
webcam-ilda --source 0 --mode motion --max-brightness 30
```

A network camera:

```bash
webcam-ilda --source rtsp://user:pass@192.168.0.142:7447/stream --mode canny
```

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--source SPEC` | `0` | Webcam index, `rtsp://`/`http://` URL, file path, or `synthetic` |
| `--source-kind` | `auto` | Force an `http://` URL to be polled as a still-image snapshot |
| `--ild FILE` | — | Project an ILDA file instead of a camera |
| `--test-pattern` | — | `square`, `circle`, `grid` or `cross` |
| `--dac` | `auto` | `auto`, `dll`, `pyusb` or `sim` |
| `--dry-run` | off | Use the simulator — no hardware needed |
| `--pps N` | `30000` | Scan rate. Match your scanners; too high rounds off corners |
| `--fps N` | `25` | Target projected frame rate; sets the point budget |
| `--max-points N` | `4096` | Hard per-frame cap (firmware maximum is 4096) |
| `--mode` | `canny` | `canny`, `threshold`, `motion`, `color` |
| `--color R,G,B` | `0,255,0` | Fixed draw colour |
| `--color-mode` | `fixed` | `fixed`, `sample` (read from the image), `rainbow` |
| `--scale PCT` | `70` | Projected size as a percentage of full scale |
| `--flip-x` / `--flip-y` / `--rotate` | — | Calibration transform |
| `--simplify PX` | `2.0` | Douglas–Peucker tolerance — the biggest lever on point cost |
| `--max-brightness PCT` | `50` | Optical power ceiling |
| `--ack-safety` | off | Skip the interactive arming prompt (required for `--headless`) |
| `--headless` | off | No preview window |
| `--frames N` | — | Process N frames then exit |
| `--record FILE.ild` | — | Write the session out as ILDA |
| `--config PATH` | `webcam_ilda.yaml` | Config file |
| `--save-config` | — | Write resolved settings and exit |

## Hotkeys

| Key | Action | Key | Action |
| --- | --- | --- | --- |
| `ESC` | **Emergency stop** | `f` / `g` | Flip X / Y |
| `SPACE` | Mute output (soft blank) | `r` | Rotate 90° |
| `m` | Cycle extraction mode | `-` / `=` | Scale down / up |
| `[` `]` | Canny low threshold ∓10 | `,` / `.` | Brightness cap down / up |
| `{` `}` | Canny high threshold ∓10 | `t` | Show/hide blanked travel |
| `a` | Toggle automatic thresholds | `c` | Show/hide camera inset |
| `p` | Freeze the current frame | `b` | Relearn background (motion mode) |
| `s` | Save tuning to `webcam_ilda.yaml` | | |

## Calibration

Project `--test-pattern square` at low power and adjust until it lands where you
want it:

1. **Orientation** — `f` and `g` flip, `r` rotates. A projector mounted upside
   down behind a mirror needs both.
2. **Size** — `-` and `=` change `scale_pct`. Keep some margin; a pattern that
   reaches the edge of the scan field is being drawn with the galvos at their
   limits, where they are least accurate.
3. **Scan rate** — if the corners of the square look rounded or the shape
   wobbles, `--pps` is above what your scanners can follow. Drop it.
4. Press `s`. The settings land in `webcam_ilda.yaml` and load automatically.

## Safety

The controls below are implemented in [`safety.py`](webcam_ilda/safety.py) and
[`pump.py`](webcam_ilda/pump.py), and every frame passes through them regardless
of which part of the program produced it.

**What the software does**

- **Arming.** With a hardware backend, the program prints a safety briefing and
  waits for you to type `arm`. `--ack-safety` skips the prompt for automated
  runs; it does not skip the briefing.
- **No parked beams.** A lit point repeated more than `max_static_run` times
  (12 by default — 0.4 ms at 30 kpps) is rejected and the frame is blanked. A
  stationary beam is the difference between a light show and a cutting tool.
- **No collapsed frames.** If the lit content shrinks below 2% of the scan
  field, the frame is blanked — a scene that has degenerated to a dot delivers
  its entire power into one spot.
- **Brightness ceiling.** `--max-brightness` is applied in the pump, after
  validation, so no upstream stage can raise it.
- **Stall watchdog.** If the vision pipeline stops delivering frames for two
  seconds — a dropped RTSP stream, an exception, a wedged camera — output is
  blanked.
- **Blanking on every exit.** Normal return, `Ctrl+C`, `SIGTERM`, an unhandled
  exception, and interpreter teardown all converge on the same shutdown that
  writes a blank frame, stops output and closes the shutter.
- **Intrinsically safe frames.** The Helios *loops the last frame it was given*
  until something replaces it, so a host crash leaves the projector running that
  frame indefinitely. The validators exist so that any frame which could end up
  looping is a moving, power-capped pattern.

**What the software cannot do**

It is not a scan-fail safeguard, and nothing in this repository should be
treated as one. If a galvanometer seizes, the DAC carries on reporting ready and
accepting frames while the beam sits still at full power. Detecting that
requires the projector's own scan-fail circuit and the ILDA connector's
interlock loop. This program is a *contributing* control on the software side of
the system — it makes the software fail dark. The safety system is hardware.

## Configuration

`webcam_ilda.yaml` in the working directory, if present, sets the defaults.
Command-line arguments override it. Write a starting point with
`webcam-ilda --save-config`:

```yaml
source: "0"
dac: auto
pps: 30000
fps: 25
color: [0, 255, 0]
color_mode: fixed
vision:
  mode: canny
  canny_lo: 80
  canny_hi: 160
  simplify_px: 2.0      # Douglas-Peucker tolerance, pixels
  min_arc_px: 30.0      # contours shorter than this are noise
  max_paths: 48
calibration:
  rotate: 0
  flip_x: false
  flip_y: false
  scale_pct: 70.0
optimiser:
  step_draw: 60.0       # spacing between lit samples, DAC units
  step_blank: 180.0     # spacing while travelling blanked
  corner_threshold_deg: 25.0
  dwell_blank_start: 8  # settling points before the beam comes on
  dwell_blank_end: 6
safety:
  max_brightness: 0.5
  max_static_run: 12
  watchdog_s: 2.0
```

## How it works

```
camera ──▶ capture thread ──▶ vision ──▶ geometry ──▶ path optimiser ──▶ pump thread ──▶ DAC
          (keeps newest       (contours)  (DAC coords)  (ordering, dwell,   (validate,
           frame only)                                   budget)            blank on stall)
```

**Capture** runs on its own thread and keeps only the most recent frame. This is
not an optimisation — OpenCV buffers decoded frames, so an RTSP consumer that is
slower than the stream falls progressively further behind until it is projecting
the past.

**Vision** downscales to 640 px wide, extracts contours according to the current
mode, and simplifies them with Douglas–Peucker. A raw contour has one point per
boundary pixel; the laser has a budget of about a thousand points for the whole
scene.

**Geometry** maps image pixels into the DAC's 12-bit square field, preserving
aspect ratio and inverting y, then applies the calibration transform.

**The path optimiser** is where a laser stops behaving like a screen. One beam
has to physically travel the entire picture 25 times a second, through mirrors
that have mass:

- *Ordering* — greedy nearest-neighbour over contour endpoints, with open paths
  reversible and closed paths rotated to start at their nearest vertex. Frames
  begin where the previous frame ended, so the beam is not flung back to the
  centre between refreshes.
- *Resampling* — points are emitted every `step_draw` DAC units of arc length.
  Brightness is dwell time, so unevenly spaced samples produce a line that is
  bright where they bunch and dim where they spread.
- *Corner dwell* — a vertex turning through θ° is repeated `round(θ/30)` times,
  capped at 8. Without it the galvos overshoot every corner.
- *Anchor dwell* — 8 blanked points sit on a path's first vertex while the
  mirrors settle, then the beam comes on. This single detail is the difference
  between a clean line start and a smear.
- *Budget* — `min(4096, pps / fps)`. At the defaults that is 1200 points for the
  entire scene. Contours are kept longest-first, so a busy frame loses fine
  detail rather than losing whichever shape happened to sort last; if a single
  contour still will not fit, the sample step is coarsened instead.

**The pump** owns the DAC exclusively, validates every frame, and paces itself
against `GetStatus`. Because the DAC loops its current frame, the vision loop is
free to run slower than the projection: a 12 fps pipeline still yields a steady
25 fps image.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `could not load HeliosLaserDAC.dll` | 32-bit Python against the x64 DLL. Use a 64-bit interpreter, or pass `--dll-path`. |
| `no Helios DAC found` | Check the USB cable, and that the device enumerates (`1209:E500`). `--dry-run` confirms the rest of the pipeline is fine. |
| Image is mirrored or upside down | `f`, `g`, `r` — then `s` to save. |
| Lines dim or dashed | `step_draw` too large, or `--pps` too low for the point count. |
| Visible flicker | Frame rate below ~20 fps. Raise `--pps`, raise `--simplify`, or lower `--fps` to shrink the budget. |
| Corners rounded, shapes wobble | `--pps` exceeds what the scanners can follow. Lower it. |
| RTSP feed lags seconds behind | Should not happen — the capture thread drops stale frames. If it does, the stream is delivering B-frames faster than they decode; lower the camera's resolution. |
| Contours flicker in and out | Canny thresholds sitting at the edge of the scene's contrast. Try `a` for automatic thresholds, or `threshold` mode. |
| `frame rejected, blanking output` in the log | Working as designed — something produced a frame that would have parked the beam. The message says which check failed. |

## Development

```bash
pip install ".[test]"
pytest
```

The suite runs entirely against the simulator — no DAC, no laser, no camera —
and covers the ILDA parser (against a real `.ild` file), the geometry
transforms, the path-optimiser invariants, point-budget enforcement, the safety
validators, and the USB wire encoding including the firmware's 64-byte packet
workaround.

CI runs on Ubuntu and Windows against Python 3.10 and 3.13.

## License

MIT — see [LICENSE](LICENSE).

The vendored `HeliosLaserDAC.dll` is from the MIT-licensed
[Helios DAC SDK](https://github.com/Grix/helios_dac) by Gitle Mikkelsen. See
[THIRD_PARTY.md](THIRD_PARTY.md).
