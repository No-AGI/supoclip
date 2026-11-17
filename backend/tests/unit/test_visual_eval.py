"""Scoring the visual scan against reviewed intervals.

The label files here are tiny fixtures: they exist to prove the comparison maths
and the false-positive accounting, not to measure accuracy.
"""
import json

import pytest

from src.media import visual_eval as ev


def report(source, candidates):
    return {
        "source_id": source,
        "candidates": [
            {"start_s": start, "end_s": end, "score": score, "event_type": kind}
            for start, end, score, kind in candidates
        ],
    }


def labels(entries):
    return [
        {"source": source, "start_s": start, "end_s": end, "label": kind, "note": ""}
        for source, start, end, kind in entries
    ]


def test_a_candidate_that_covers_the_label_is_a_hit():
    result = ev.evaluate(
        report("src", [(110.0, 130.0, 90.0, "motion_burst")]),
        labels([("src", 112.0, 132.0, "event")]),
    )
    assert result["matched_positive"] == 1
    assert result["recall"] == 1.0
    assert result["missed_positive"] == 0


def test_a_candidate_that_only_clips_the_edge_does_not_count():
    result = ev.evaluate(
        report("src", [(0.0, 20.0, 80.0, "motion_burst")]),
        labels([("src", 18.0, 38.0, "event")]),
    )
    assert result["matched_positive"] == 0
    assert result["missed_positive"] == 1
    # The candidate covers 10% of the label, so it belongs to no labelled span
    # at this threshold: unlabelled, not a proven false positive.
    assert len(result["unlabelled"]) == 1


def test_a_slate_candidate_is_a_false_positive():
    result = ev.evaluate(
        report("src", [(500.0, 520.0, 22.0, "static_card"), (100.0, 120.0, 90.0, "motion_burst")]),
        labels([("src", 505.0, 528.0, "not_event"), ("src", 100.0, 120.0, "event")]),
    )
    assert result["matched_negative"] == 1
    assert result["matched_positive"] == 1
    # A slate offered as a candidate where a reviewer said "not a moment" is a
    # genuine false positive, not merely unlabelled.
    assert len(result["false_positives"]) == 1
    assert result["false_positives"][0]["event_type"] == "static_card"
    # Precision inside labelled spans: one hit and one slate = 1/2.
    assert result["precision_within_labelled_spans"] == 0.5


def test_candidates_outside_the_labels_are_unverified_not_negative():
    result = ev.evaluate(
        report("src", [(100.0, 120.0, 90.0, "motion_burst"), (300.0, 320.0, 70.0, "motion_burst")]),
        labels([("src", 100.0, 120.0, "event")]),
    )
    assert len(result["false_positives"]) == 0
    assert len(result["unlabelled"]) == 1
    assert result["unlabelled"][0]["start_s"] == 300.0
    # Precision counts only what the labels actually cover.
    assert result["precision_within_labelled_spans"] == 1.0


def test_labels_for_another_source_are_ignored():
    result = ev.evaluate(
        report("src-a", [(100.0, 120.0, 90.0, "motion_burst")]),
        labels([("src-b", 100.0, 120.0, "event")]),
    )
    assert result["labelled_intervals"] == 0
    assert len(result["unlabelled"]) == 1


def test_no_candidates_means_zero_recall_not_an_error():
    result = ev.evaluate(report("src", []), labels([("src", 10.0, 30.0, "event")]))
    assert result["recall"] == 0.0
    assert result["precision_within_labelled_spans"] is None


def test_unknown_label_is_rejected(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps([{"source": "s", "start_s": 0, "end_s": 5, "label": "maybe"}]))
    with pytest.raises(ValueError, match="unknown label"):
        ev.load_labels(path)


def test_reversed_interval_is_rejected(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps([{"source": "s", "start_s": 10, "end_s": 5}]))
    with pytest.raises(ValueError, match="after start"):
        ev.load_labels(path)


def test_missing_report_is_listed_not_crashed(tmp_path):
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps([{"source": "a", "start_s": 0, "end_s": 5}, {"source": "b", "start_s": 0, "end_s": 5}]))
    report_path = tmp_path / "a.json"
    report_path.write_text(json.dumps(report("a", [(0.0, 5.0, 50.0, "motion_burst")])))
    result = ev.evaluate_files([report_path], labels_path)
    assert result["sources_without_report"] == ["b"]
