"""Carousel slides: the 4:5 size, the counter, and the closing call to action."""
import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image, ImageDraw  # noqa: E402

from src.media import carousel_render as car  # noqa: E402


@pytest.fixture()
def frames(tmp_path):
    paths = []
    for index, colour in enumerate([(30, 50, 70), (90, 40, 40), (20, 70, 60)], start=1):
        image = Image.new("RGB", (1280, 720), colour)
        draw = ImageDraw.Draw(image)
        # The streamer's webcam corner, which must not reach the slide.
        draw.rectangle((0, 430, 300, 720), fill=(220, 30, 30))
        path = tmp_path / f"f{index}.png"
        image.save(path)
        paths.append(path)
    return paths


def test_slides_are_the_feed_size(frames, tmp_path):
    report = car.render_carousel(
        [(frames[0], "ПЕРВЫЙ"), (frames[1], "ВТОРОЙ")], tmp_path / "out"
    )
    assert report["size"] == [1080, 1350]
    assert report["ratio"] == pytest.approx(0.8)
    assert len(report["files"]) == 2
    for path in report["files"]:
        assert Image.open(path).size == (1080, 1350)


def test_a_carousel_needs_at_least_one_slide(tmp_path):
    with pytest.raises(ValueError):
        car.render_carousel([], tmp_path / "out")


def test_the_last_slide_is_the_only_one_with_the_brand(frames, tmp_path):
    report = car.render_carousel(
        [(frames[0], "РАЗ"), (frames[1], "ДВА"), (frames[2], "ТРИ")], tmp_path / "out"
    )
    # Pixels, not text: the nickname is yellow only where the brand is printed.
    def yellow_pixels(path):
        slide = Image.open(path).convert("RGB").crop((0, 1100, 1080, 1350))
        return sum(
            1
            for r, g, b in slide.get_flattened_data()
            if r > 220 and g > 180 and b < 120
        )

    counts = [yellow_pixels(path) for path in report["files"]]
    assert counts[-1] > counts[0]
    assert car.BRAND == "@GTAVI.FRAME"


def test_every_slide_keeps_the_webcam_corner_out(frames, tmp_path):
    report = car.render_carousel([(frames[0], "ТЕСТ")], tmp_path / "out")
    slide = Image.open(report["files"][0]).convert("RGB")
    corner = slide.crop((0, 1100, 140, 1350))
    red = sum(
        1 for r, g, b in corner.get_flattened_data() if r > 150 and g < 80 and b < 80
    )
    assert red < 500, red


def test_a_long_caption_is_shrunk_to_fit(frames, tmp_path):
    long_text = "ЭТО ОЧЕНЬ ДЛИННАЯ ПОДПИСЬ КОТОРАЯ НЕ ДОЛЖНА ВЫЙТИ ЗА КРАЙ"
    report = car.render_carousel([(frames[0], long_text)], tmp_path / "out")
    slide = Image.open(report["files"][0]).convert("RGB")
    band = slide.crop((0, 900, 1080, 1200))
    for x in list(range(0, 5)) + list(range(1075, 1080)):
        for y in range(0, band.height, 5):
            r, g, b = band.load()[x, y]
            assert not (r > 235 and g > 235 and b > 235), f"white ink at x={x}"
