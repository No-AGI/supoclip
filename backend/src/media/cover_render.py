"""Render a 1080x1920 cover from a gameplay frame plus brand typography.

The brand contract lives in ``docs/04_COVER_CONTRACT.md`` of the handoff:
one short title (two lines: white then pink), a pink separator, the yellow
nickname and a pink underline, no neon capsules and no Instagram icons. This
module only composes; a human still chooses the frame and the words.

Everything is drawn with Pillow so no paid vision or design service is needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont

COVER_SIZE = (1080, 1920)
BRAND = "@GTAVI.FRAME"
WHITE = (255, 255, 255)
PINK = (255, 45, 120)
YELLOW = (255, 216, 74)
SHADOW = (0, 0, 0)

# The streamer's webcam sits in the bottom-left of the source frame, so the
# vertical crop is normally pushed right of it.
DEFAULT_CROP_OFFSET = 0.62
# Archivo Black, Bebas and Poppins ship no Cyrillic, so the brand's default title
# face is Montserrat at weight 900: a poster weight that covers Russian.
DEFAULT_FONT = "Montserrat-Variable-wght.ttf"
DEFAULT_WEIGHT = 900


def _font(name: str, size: int, weight: int | None = None) -> ImageFont.FreeTypeFont:
    path = Path(__file__).resolve().parents[2] / "fonts" / name
    if not path.is_file():
        raise FileNotFoundError(f"font not found: {path}")
    font = ImageFont.truetype(str(path), size)
    if weight:
        try:
            font.set_variation_by_axes([weight])
        except Exception:
            # A static face keeps its own weight; nothing to select.
            pass
    return font


def _shear(layer: Image.Image, amount: float) -> Image.Image:
    """Slant a transparent layer to the right, keeping it inside the canvas."""
    width, height = layer.size
    # A positive x-offset that grows with y leans the text forward.
    data = (1, -amount, amount * height, 0, 1, 0)
    return layer.transform(layer.size, Image.AFFINE, data, resample=Image.BICUBIC)


def _fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_name: str,
    max_width: int,
    start_size: int,
    stroke: int = 0,
    weight: int | None = None,
) -> tuple[ImageFont.FreeTypeFont, int, int]:
    size = start_size
    while size > 24:
        font = _font(font_name, size, weight)
        box = draw.textbbox((0, 0), text, font=font, stroke_width=stroke)
        if box[2] - box[0] <= max_width:
            return font, box[2] - box[0], box[3] - box[1]
        size -= 4
    font = _font(font_name, 24, weight)
    box = draw.textbbox((0, 0), text, font=font, stroke_width=stroke)
    return font, box[2] - box[0], box[3] - box[1]


def _title_layer(
    lines: list[tuple[str, tuple[int, int, int]]],
    font_name: str,
    max_width: int,
    shear: float,
    weight: int | None = None,
) -> tuple[Image.Image, int]:
    """Draw the stacked title on its own layer so it can be slanted at once."""
    probe = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    pad = 120
    rendered: list[tuple[ImageFont.FreeTypeFont, str, tuple[int, int, int], int]] = []
    for text, colour in lines:
        font, width, _ = _fit_text(probe, text, font_name, max_width, 190, stroke=0, weight=weight)
        box = probe.textbbox((0, 0), text, font=font, stroke_width=10)
        rendered.append((font, text, colour, box[3] - box[1]))

    total_height = sum(item[3] for item in rendered) + 26 * (len(rendered) - 1) + pad * 2
    layer = Image.new("RGBA", (COVER_SIZE[0], total_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    y = pad
    for font, text, colour, height in rendered:
        box = draw.textbbox((0, 0), text, font=font, stroke_width=10)
        x = (COVER_SIZE[0] - (box[2] - box[0])) / 2 - box[0]
        draw.text(
            (x, y - box[1]),
            text,
            font=font,
            fill=colour,
            stroke_width=10,
            stroke_fill=(0, 0, 0, 235),
        )
        y += height + 26
    return _shear(layer, shear), total_height


def _brush(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], colour: tuple[int, int, int, int]) -> None:
    """A paint stroke: several offset bars so the ends look torn, not machined."""
    left, top, right, bottom = box
    height = bottom - top
    draw.rounded_rectangle(box, radius=height // 2, fill=colour)
    draw.rounded_rectangle(
        (left + 18, top - 5, right - 34, top + height // 2),
        radius=height // 3,
        fill=colour,
    )
    draw.rounded_rectangle(
        (left + 4, top + height // 2, right - 6, bottom + 4),
        radius=height // 3,
        fill=colour,
    )


def render_cover(
    frame_path: Path,
    out_path: Path,
    *,
    white_line: str,
    pink_line: str,
    cta: str = "ПОДПИШИСЬ",
    crop_offset: float = DEFAULT_CROP_OFFSET,
    font_name: str = DEFAULT_FONT,
    font_weight: int | None = DEFAULT_WEIGHT,
    shear: float = 0.20,
    dim: float = 0.35,
) -> dict[str, Any]:
    """Compose a cover and return measurable facts about the result."""
    source = Image.open(frame_path).convert("RGB")
    target_ratio = COVER_SIZE[0] / COVER_SIZE[1]
    # Scale by height first, then take the widest crop that still fills 9:16.
    scale = COVER_SIZE[1] / source.height
    scaled = source.resize(
        (max(COVER_SIZE[0], int(source.width * scale)), COVER_SIZE[1]),
        Image.LANCZOS,
    )
    if scaled.width < COVER_SIZE[0]:
        scaled = source.resize(
            (COVER_SIZE[0], int(source.height * COVER_SIZE[0] / source.width)),
            Image.LANCZOS,
        )
        top = max(0, (scaled.height - COVER_SIZE[1]) // 2)
        scaled = scaled.crop((0, top, COVER_SIZE[0], top + COVER_SIZE[1]))
    max_left = scaled.width - COVER_SIZE[0]
    left = int(max_left * max(0.0, min(1.0, crop_offset)))
    cover = scaled.crop((left, 0, left + COVER_SIZE[0], COVER_SIZE[1]))
    cover = cover.filter(ImageFilter.UnsharpMask(radius=2, percent=145, threshold=2))

    # A top-down shade keeps white type readable on bright sky, and a bottom
    # shade separates the call to action from the HUD.
    shade = Image.new("L", COVER_SIZE, 0)
    shade_draw = ImageDraw.Draw(shade)
    for y in range(0, 900):
        value = int(dim * 255 * (1 - y / 900) ** 1.4)
        shade_draw.line([(0, y), (COVER_SIZE[0], y)], fill=value)
    for y in range(1500, COVER_SIZE[1]):
        value = int(dim * 255 * ((y - 1500) / 420) ** 1.2)
        shade_draw.line([(0, y), (COVER_SIZE[0], y)], fill=value)
    cover = Image.composite(Image.new("RGB", COVER_SIZE, SHADOW), cover, shade.point(lambda v: int(v * 0.9)))

    # Title block.
    layer, layer_height = _title_layer(
        [(white_line.upper(), WHITE), (pink_line.upper(), PINK)],
        font_name,
        int(COVER_SIZE[0] * 0.80),
        shear,
        weight=font_weight,
    )
    title_top = int(COVER_SIZE[1] * 0.085)
    # A slanted layer moves sideways, so anchor it by its real ink bounds.
    ink = layer.getbbox()
    paste_x = 0 if ink is None else -ink[0] + (COVER_SIZE[0] - (ink[2] - ink[0])) // 2
    cover.paste(layer, (paste_x, title_top), layer)

    # Pink brush stroke under the two title lines.
    stroke_layer = Image.new("RGBA", COVER_SIZE, (0, 0, 0, 0))
    stroke_draw = ImageDraw.Draw(stroke_layer)
    stroke_width = int(COVER_SIZE[0] * 0.36)
    stroke_height = 34
    sx = COVER_SIZE[0] // 2 - stroke_width // 2
    sy = title_top + layer_height - 40
    _brush(
        stroke_draw,
        (sx, sy, sx + stroke_width, sy + stroke_height),
        (*PINK, 215),
    )
    # Two thinner passes break the silhouette so it reads as paint, not a pill.
    stroke_draw.rounded_rectangle(
        (sx + 40, sy - 7, sx + stroke_width - 70, sy + stroke_height // 2),
        radius=9,
        fill=(*PINK, 120),
    )
    stroke_draw.rounded_rectangle(
        (sx + 12, sy + stroke_height // 3, sx + stroke_width - 18, sy + stroke_height + 6),
        radius=8,
        fill=(*PINK, 140),
    )
    stroke_layer = stroke_layer.filter(ImageFilter.GaussianBlur(1.1))
    stroke_layer = _shear(stroke_layer, shear * 0.35)
    cover.paste(stroke_layer, (0, 0), stroke_layer)

    # Call to action.
    cta_layer = Image.new("RGBA", COVER_SIZE, (0, 0, 0, 0))
    cta_draw = ImageDraw.Draw(cta_layer)
    gap = 26
    cta_size = 62
    limit = int(COVER_SIZE[0] * 0.90)
    while cta_size > 28:
        cta_font = _font(font_name, cta_size, font_weight)
        handle_font = _font(font_name, cta_size, font_weight)
        cta_box = cta_draw.textbbox((0, 0), cta, font=cta_font)
        handle_box = cta_draw.textbbox((0, 0), BRAND, font=handle_font)
        total = (cta_box[2] - cta_box[0]) + gap * 2 + (handle_box[2] - handle_box[0])
        if total <= limit:
            break
        gap = max(14, gap - 2)
        cta_size -= 3
    x = (COVER_SIZE[0] - total) / 2
    baseline = int(COVER_SIZE[1] * 0.855)
    cta_draw.text((x, baseline), cta, font=cta_font, fill=WHITE + (255,))
    x += (cta_box[2] - cta_box[0]) + gap
    dot_r = 9
    dot_y = baseline + int(cta_size * 0.45)
    cta_draw.ellipse((x, dot_y, x + dot_r * 2, dot_y + dot_r * 2), fill=PINK + (255,))
    x += dot_r * 2 + gap
    cta_draw.text((x, baseline), BRAND, font=handle_font, fill=YELLOW + (255,))
    underline_y = baseline + int(cta_size * 1.42)
    cta_draw.rounded_rectangle(
        (x - 6, underline_y, x + (handle_box[2] - handle_box[0]) + 6, underline_y + 7),
        radius=4,
        fill=PINK + (230,),
    )
    cover.paste(cta_layer, (0, 0), cta_layer)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cover.save(out_path, "PNG", optimize=True)

    return {
        "out": str(out_path),
        "size": list(cover.size),
        "source_size": list(source.size),
        "crop_left": left,
        "crop_offset": crop_offset,
        "target_ratio": round(target_ratio, 4),
        "actual_ratio": round(cover.width / cover.height, 4),
        "title_height": layer_height,
        "title_top": title_top,
        "lines": [white_line, pink_line],
    }


def crop_checks(path: Path) -> dict[str, Any]:
    """Verify the 3:4 and 4:5 crops the brand contract prescribes."""
    image = Image.open(path)
    results: dict[str, Any] = {"path": str(path), "size": list(image.size)}
    for name, box in (("3:4", (0, 240, 1080, 1680)), ("4:5", (0, 285, 1080, 1635))):
        width = box[2] - box[0]
        height = box[3] - box[1]
        results[name] = {
            "box": list(box),
            "ratio": round(width / height, 4),
            "ok": abs(width / height - {"3:4": 3 / 4, "4:5": 4 / 5}[name]) < 0.01,
        }
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--white", required=True, help="first title line")
    parser.add_argument("--pink", required=True, help="accent line")
    parser.add_argument("--cta", default="ПОДПИШИСЬ")
    parser.add_argument("--crop-offset", type=float, default=DEFAULT_CROP_OFFSET)
    parser.add_argument("--font", default=DEFAULT_FONT)
    parser.add_argument("--weight", type=int, default=DEFAULT_WEIGHT)
    parser.add_argument("--shear", type=float, default=0.20)
    parser.add_argument("--json", type=Path, help="write the report here")
    args = parser.parse_args(argv)

    report = render_cover(
        args.frame,
        args.out,
        white_line=args.white,
        pink_line=args.pink,
        cta=args.cta,
        crop_offset=args.crop_offset,
        font_name=args.font,
        font_weight=args.weight,
        shear=args.shear,
    )
    report["crops"] = crop_checks(args.out)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json:
        args.json.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
