import pytest

from src.media.srt import CaptionGroup, build_srt


def test_builds_numbered_subrip_cues_and_preserves_gaps_and_text():
    result = build_srt(
        [
            CaptionGroup("Я", 0.1, 0.188),
            {"text": "Попал в беду", "start": 2.993, "end": 3.715},
        ]
    )
    assert result == (
        "1\n00:00:00,100 --> 00:00:00,188\nЯ\n\n"
        "2\n00:00:02,993 --> 00:00:03,715\nПопал в беду\n"
    )


def test_keeps_explicit_multiline_text_and_handles_long_hours():
    result = build_srt([{"text": "First line\nSecond line", "start": 360000, "end": 360001.25}])
    assert result == "1\n100:00:00,000 --> 100:00:01,250\nFirst line\nSecond line\n"


def test_empty_groups_produce_an_empty_file():
    assert build_srt([]) == ""


@pytest.mark.parametrize(
    "groups",
    [
        [{"text": "", "start": 0, "end": 1}],
        [{"text": "x", "start": -0.1, "end": 1}],
        [{"text": "x", "start": 0, "end": float("nan")}],
        [{"text": "x", "start": 1, "end": 1}],
        [{"text": "x", "start": 0, "end": 0.0004}],
        [
            {"text": "first", "start": 0, "end": 2},
            {"text": "second", "start": 1, "end": 3},
        ],
    ],
)
def test_rejects_invalid_or_overlapping_cues(groups):
    with pytest.raises(ValueError):
        build_srt(groups)


def test_requires_exact_text_and_timing_fields():
    with pytest.raises(ValueError, match="missing 'text'"):
        build_srt([{"start": 0, "end": 1}])
