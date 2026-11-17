"""Cover composition: size, crops, Cyrillic coverage and safe margins."""
import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image, ImageDraw  # noqa: E402

from src.media import cover_render as cr  # noqa: E402


@pytest.fixture()
def frame(tmp_path):
    """A frame with the streamer-style webcam block in the bottom-left corner."""
    image = Image.new("RGB", (1280, 720), (40, 60, 90))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 430, 300, 720), fill=(200, 40, 40))
    draw.rectangle((400, 100, 900, 500), fill=(120, 160, 200))
    path = tmp_path / "frame.png"
    image.save(path)
    return path


def test_cover_is_exactly_the_instagram_story_size(frame, tmp_path):
    report = cr.render_cover(frame, tmp_path / "c.png", white_line="НОЧНАЯ", pink_line="ПОГОНЯ")
    assert report["size"] == [1080, 1920]
    assert report["actual_ratio"] == pytest.approx(1080 / 1920, abs=0.001)


def test_the_prescribed_crops_hold(frame, tmp_path):
    out = tmp_path / "c.png"
    cr.render_cover(frame, out, white_line="ЗА", pink_line="ЗАБОРОМ")
    checks = cr.crop_checks(out)
    assert checks["size"] == [1080, 1920]
    assert checks["3:4"]["ok"] and checks["4:5"]["ok"]


def test_the_default_font_actually_has_cyrillic():
    """Archivo Black and Bebas render every Russian letter as a tofu box."""
    font = cr._font(cr.DEFAULT_FONT, 60, cr.DEFAULT_WEIGHT)
    widths = {letter: round(font.getlength(letter), 2) for letter in "НОЧЕЙПБ"}
    assert len(set(widths.values())) > 3, widths
    tofu = round(font.getlength("\uFFFF"), 2)
    assert widths["Н"] != tofu


def test_the_webcam_corner_is_cropped_away(frame, tmp_path):
    """The red block is only in the left of the source, so it must not survive."""
    out = tmp_path / "c.png"
    report = cr.render_cover(frame, out, white_line="ТЕСТ", pink_line="КРОП")
    assert report["crop_left"] > 0
    cover = Image.open(out).convert("RGB")
    bottom_left = cover.crop((0, 1500, 120, 1900))
    red_pixels = sum(
        1 for r, g, b in bottom_left.get_flattened_data() if r > 150 and g < 90 and b < 90
    )
    assert red_pixels < 400


def test_a_different_crop_offset_moves_the_picture(frame, tmp_path):
    left = cr.render_cover(frame, tmp_path / "a.png", white_line="А", pink_line="Б", crop_offset=0.0)
    right = cr.render_cover(frame, tmp_path / "b.png", white_line="А", pink_line="Б", crop_offset=1.0)
    assert left["crop_left"] < right["crop_left"]


def test_a_long_title_stays_inside_the_canvas(frame, tmp_path):
    out = tmp_path / "long.png"
    cr.render_cover(
        frame,
        out,
        white_line="ОЧЕНЬ ДЛИННЫЙ ЗАГОЛОВОК",
        pink_line="ВТОРАЯ СТРОКА ТОЖЕ ДЛИННАЯ",
    )
    cover = Image.open(out).convert("RGB")
    # Nothing may be painted in the outermost columns of the title band.
    band = cover.crop((0, 150, 1080, 700))
    pixels = band.load()
    for x in list(range(0, 6)) + list(range(1074, 1080)):
        for y in range(0, band.height, 7):
            r, g, b = pixels[x, y]
            assert not (r > 200 and g > 200 and b > 200), f"white ink at x={x}"
