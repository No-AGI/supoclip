"""Visual candidates are a reserve: they must never displace spoken segments."""
from src.services.video_service import VideoService


SPOKEN = [
    {
        "start_time": "00:07:00",
        "end_time": "00:07:30",
        "text": "говорит стример",
        "relevance_score": 0.9,
        "reasoning": "dialogue",
    }
]


def test_a_visual_window_overlapping_a_spoken_one_is_dropped():
    candidates = [{"start_s": 425, "end_s": 445, "score": 80, "event_type": "motion_burst"}]
    merged = VideoService.merge_visual_candidates(SPOKEN, candidates)
    assert merged == SPOKEN


def test_a_visual_window_between_clips_is_added_as_a_reserve():
    candidates = [{"start_s": 800, "end_s": 820, "score": 60, "event_type": "montage_cut"}]
    merged = VideoService.merge_visual_candidates(SPOKEN, candidates)
    assert len(merged) == 2
    added = merged[1]
    assert added["start_time"] in ("13:20", "00:13:20")
    assert added["end_time"] in ("13:40", "00:13:40")
    assert added["hook_type"] == "visual_signal"
    assert added["text"] == ""
    # Ranked below the spoken segment so it cannot take its slot.
    assert added["relevance_score"] < SPOKEN[0]["relevance_score"]


def test_no_candidates_returns_the_segments_untouched():
    assert VideoService.merge_visual_candidates(SPOKEN, []) == SPOKEN


def test_a_reversed_candidate_is_ignored():
    merged = VideoService.merge_visual_candidates(
        SPOKEN, [{"start_s": 900, "end_s": 880, "score": 50}]
    )
    assert merged == SPOKEN


def test_overlapping_candidates_are_added_once():
    candidates = [
        {"start_s": 800, "end_s": 820, "score": 60},
        {"start_s": 805, "end_s": 825, "score": 55},
    ]
    merged = VideoService.merge_visual_candidates(SPOKEN, candidates)
    assert len(merged) == 2
