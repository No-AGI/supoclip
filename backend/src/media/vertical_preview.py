"""Preview a gameplay crop for a vertical export — measurement only.

SupoClip's vertical renderer crops to a tracked face and falls back to a
blurred-background full-frame fit when it cannot find a usable subject. On
gameplay footage there is usually no face and often no text to protect, so the
fallback wastes most of a 1080x1920 frame on blur. This tool measures where the
movement is inside a window and renders a **preview** of a 9:16 crop so the
choice can be judged on pictures.

It is deliberately not part of the renderer: nothing here is called by the
pipeline, it writes a local preview file, and it needs no paid service.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import visual_events as ve

PREVIEW_FPS = 4.0
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920


def measure_crop(path: Path, start: float, end: float) -> dict:
    """Where the movement sits inside one window, and how steady it is."""
    duration = max(0.1, end - start)
    frames = ve.thumbnails(path, fps=PREVIEW_FPS, start=start, duration=duration)
    if len(frames) < 2:
        return {"frames": int(len(frames)), "steady": False}
    centres = ve.smoothed_centroid(ve.motion_centroid(frames))
    live = centres[centres != 0.5]
    reference = live if len(live) else centres
    return {
        "frames": int(len(frames)),
        "centre_mean": round(float(reference.mean()), 3),
        "centre_median": round(float(np.median(reference)), 3),
        "centre_std": round(float(reference.std()), 3),
        "centre_min": round(float(reference.min()), 3),
        "centre_max": round(float(reference.max()), 3),
        "moved_samples": int(len(live)),
        "steady": bool(float(reference.std()) <= 0.12),
    }


def render_preview(
    path: Path,
    start: float,
    end: float,
    out_path: Path,
    *,
    centre: Optional[float] = None,
) -> dict:
    """Render the window twice: current blur-fit look and the measured crop."""
    probe = ve._run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path),
        ]
    ).decode().strip()
    width, height = (int(part) for part in probe.split(",")[:2])
    measurement = measure_crop(path, start, end)
    if centre is None:
        centre = measurement.get("centre_median", 0.5)
    box = ve.crop_window_for_centroid(float(centre), width, height)
    duration = max(0.1, end - start)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop = (
        f"crop={box['width']}:{box['height']}:{box['x']}:{box['y']},"
        f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:flags=lanczos,setsar=1"
    )
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
            "-i", str(path), "-vf", crop, "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-y", str(out_path),
        ],
        check=True,
    )
    return {
        "source": str(path),
        "start_s": round(start, 3),
        "end_s": round(end, 3),
        "source_size": [width, height],
        "crop_box": box,
        "crop_share_of_frame": round(
            (box["width"] * box["height"]) / float(max(1, width * height)), 3
        ),
        "measured": measurement,
        "preview": str(out_path),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Preview a measured 9:16 gameplay crop")
    parser.add_argument("source", type=Path)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--centre", type=float, default=None, help="Override 0..1")
    parser.add_argument("--report", type=Path, help="Write the measurement JSON here")
    args = parser.parse_args(argv)

    report = render_preview(
        args.source, args.start, args.end, args.out, centre=args.centre
    )
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
