"""Build a 1080x1350 carousel from frames, one caption per slide.

Instagram's 4:5 feed slot is the brand's carousel format. Each slide keeps the
same frame pipeline as the covers (crop, sharpen, shade) but shows one short
line instead of a two-line hook, so a story reads left to right. The last slide
carries the call to action.

Composition only: a human still chooses frames and words. Pillow is the only
dependency.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFilter

from .cover_render import BRAND, PINK, WHITE, YELLOW, _font, _fit_text, DEFAULT_FONT, DEFAULT_WEIGHT

SLIDE_SIZE = (1080, 1350)
DEFAULT_CROP_OFFSET = 0.62


def _prepare(frame_path: Path, crop_offset: float, dim: float) -> Image.Image:
    source = Image.open(frame_path).convert("RGB")
    scale = SLIDE_SIZE[1] / source.height
    scaled = source.resize(
        (max(SLIDE_SIZE[0], int(source.width * scale)), SLIDE_SIZE[1]), Image.LANCZOS
    )
    max_left = scaled.width - SLIDE_SIZE[0]
    left = int(max_left * max(0.0, min(1.0, crop_offset)))
    slide = scaled.crop((left, 0, left + SLIDE_SIZE[0], SLIDE_SIZE[1]))
    slide = slide.filter(ImageFilter.UnsharpMask(radius=2, percent=145, threshold=2))

    shade = Image.new("L", SLIDE_SIZE, 0)
    draw = ImageDraw.Draw(shade)
    for y in range(0, SLIDE_SIZE[1]):
        # Heavier at the bottom: the caption sits there and the game HUD lives
        # in the upper corners.
        value = int(dim * 255 * (y / SLIDE_SIZE[1]) ** 1.3)
        draw.line([(0, y), (SLIDE_SIZE[0], y)], fill=value)
    return Image.composite(Image.new("RGB", SLIDE_SIZE, (0, 0, 0)), slide, shade)


def _caption(slide: Image.Image, text: str, index: int, total: int, *, last: bool) -> None:
    draw = ImageDraw.Draw(slide)
    limit = int(SLIDE_SIZE[0] * 0.86)
    font, width, _ = _fit_text(draw, text.upper(), DEFAULT_FONT, limit, 86, weight=DEFAULT_WEIGHT)

    baseline = int(SLIDE_SIZE[1] * 0.72)
    box = draw.textbbox((0, 0), text.upper(), font=font, stroke_width=6)
    x = (SLIDE_SIZE[0] - (box[2] - box[0])) / 2 - box[0]
    draw.text(
        (x, baseline),
        text.upper(),
        font=font,
        fill=WHITE,
        stroke_width=6,
        stroke_fill=(0, 0, 0, 220),
    )

    accent = PINK if last else YELLOW
    bar_width = min(int(SLIDE_SIZE[0] * 0.34), int(width * 0.9))
    bar_x = (SLIDE_SIZE[0] - bar_width) // 2
    bar_y = baseline + (box[3] - box[1]) + 54
    draw.rounded_rectangle(
        (bar_x, bar_y, bar_x + bar_width, bar_y + 20), radius=10, fill=accent
    )

    if last:
        cta = _font(DEFAULT_FONT, 44, DEFAULT_WEIGHT)
        cta_box = draw.textbbox((0, 0), BRAND, font=cta)
        draw.text(
            ((SLIDE_SIZE[0] - (cta_box[2] - cta_box[0])) / 2 - cta_box[0], bar_y + 74),
            BRAND,
            font=cta,
            fill=YELLOW,
            stroke_width=4,
            stroke_fill=(0, 0, 0, 200),
        )

    counter = _font(DEFAULT_FONT, 34, DEFAULT_WEIGHT)
    label = f"{index}/{total}"
    counter_box = draw.textbbox((0, 0), label, font=counter)
    draw.text(
        (SLIDE_SIZE[0] - (counter_box[2] - counter_box[0]) - 48, 44),
        label,
        font=counter,
        fill=(255, 255, 255, 200),
        stroke_width=3,
        stroke_fill=(0, 0, 0, 180),
    )


def render_carousel(
    slides: Sequence[tuple[Path, str]],
    out_dir: Path,
    *,
    crop_offset: float = DEFAULT_CROP_OFFSET,
    dim: float = 0.55,
) -> dict[str, Any]:
    if not slides:
        raise ValueError("a carousel needs at least one slide")
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    total = len(slides)
    for index, (frame_path, caption) in enumerate(slides, start=1):
        slide = _prepare(frame_path, crop_offset, dim)
        _caption(slide, caption, index, total, last=index == total)
        path = out_dir / f"slide_{index:02d}.png"
        slide.save(path, "PNG", optimize=True)
        written.append(str(path))
    return {
        "out_dir": str(out_dir),
        "size": list(SLIDE_SIZE),
        "ratio": round(SLIDE_SIZE[0] / SLIDE_SIZE[1], 4),
        "caption_count": total,
        "files": written,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="JSON list of {frame, caption}")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--crop-offset", type=float, default=DEFAULT_CROP_OFFSET)
    parser.add_argument("--dim", type=float, default=0.55)
    args = parser.parse_args(argv)

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    slides = [(Path(item["frame"]), item["caption"]) for item in spec["slides"]]
    report = render_carousel(
        slides, args.out_dir, crop_offset=args.crop_offset, dim=args.dim
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
