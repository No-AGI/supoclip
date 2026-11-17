"""Score the visual candidate scan against manually reviewed intervals.

The scan is a heuristic. Its usefulness is a measurement question, not an
opinion, so this module compares a scan report with a small JSON label file
written after looking at frames:

    [{"source": "HQA7fxZ-_r0", "start_s": 112.3, "end_s": 132.3,
      "label": "event", "note": "shootout, verified by eye"}]

``label`` is ``event`` (worth a Reel) or ``not_event`` (slate, black card, dead
air). A candidate counts as a hit when it covers enough of a labelled interval,
so a candidate that only clips the edge does not inflate the score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

DEFAULT_OVERLAP = 0.5
KIND_EVENT = "event"
KIND_NOT_EVENT = "not_event"


def load_labels(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("labels file must contain a list")
    labels = []
    for entry in data:
        start = float(entry["start_s"])
        end = float(entry["end_s"])
        if end <= start:
            raise ValueError(f"interval end must be after start: {entry}")
        kind = entry.get("label", KIND_EVENT)
        if kind not in (KIND_EVENT, KIND_NOT_EVENT):
            raise ValueError(f"unknown label {kind!r}; use {KIND_EVENT}/{KIND_NOT_EVENT}")
        labels.append(
            {
                "source": str(entry["source"]),
                "start_s": start,
                "end_s": end,
                "label": kind,
                "note": entry.get("note", ""),
            }
        )
    return labels


def overlap(first: tuple[float, float], second: tuple[float, float]) -> float:
    """Share of the shorter interval that the two intervals share."""
    start = max(first[0], second[0])
    end = min(first[1], second[1])
    shared = max(0.0, end - start)
    shortest = min(first[1] - first[0], second[1] - second[0])
    if shortest <= 0:
        return 0.0
    return shared / shortest


def evaluate(
    report: dict,
    labels: Sequence[dict],
    *,
    min_overlap: float = DEFAULT_OVERLAP,
) -> dict:
    """Compare one scan report with the labels of the same source."""
    source = report.get("source_id") or ""
    source_labels = [item for item in labels if item["source"] == source]
    candidates = report.get("candidates") or []
    candidate_spans = [(item["start_s"], item["end_s"], item) for item in candidates]

    hits = []
    misses = []
    for label in source_labels:
        span = (label["start_s"], label["end_s"])
        matched = [
            (item, overlap(span, (start, end)))
            for start, end, item in candidate_spans
            if overlap(span, (start, end)) >= min_overlap
        ]
        if matched:
            best, score = max(matched, key=lambda pair: pair[1])
            hits.append(
                {
                    "label": label,
                    "candidate": {"start_s": best["start_s"], "end_s": best["end_s"], "score": best.get("score")},
                    "overlap": round(score, 3),
                }
            )
        else:
            misses.append(label)

    # Candidates are split three ways so a half-labelled source cannot be
    # reported as low precision: a candidate on a labelled ``not_event`` is a
    # real false positive, one on a labelled ``event`` is a hit, and one outside
    # every labelled span is simply unlabelled — nobody said it was wrong.
    false_positives = []
    unlabelled = []
    overlapping = []
    not_event_spans = [(lb["start_s"], lb["end_s"]) for lb in source_labels if lb["label"] == KIND_NOT_EVENT]
    all_spans = [(lb["start_s"], lb["end_s"]) for lb in source_labels]
    for start, end, item in candidate_spans:
        span = (start, end)
        if any(overlap(span, other) >= min_overlap for other in not_event_spans):
            false_positives.append(
                {"start_s": start, "end_s": end, "score": item.get("score"), "event_type": item.get("event_type")}
            )
        elif any(overlap(span, other) >= min_overlap for other in all_spans):
            overlapping.append(item)
        else:
            unlabelled.append(
                {"start_s": start, "end_s": end, "score": item.get("score"), "event_type": item.get("event_type")}
            )

    true_positive_labels = [hit for hit in hits if hit["label"]["label"] == KIND_EVENT]
    negative_hits = [hit for hit in hits if hit["label"]["label"] == KIND_NOT_EVENT]
    positive_labels = [lb for lb in source_labels if lb["label"] == KIND_EVENT]

    recall = len(true_positive_labels) / len(positive_labels) if positive_labels else None
    # Precision is only meaningful inside the labelled spans; the rest of the
    # timeline is unverified, not negative.
    labelled_candidates = len(overlapping) + len(false_positives)
    precision = len(true_positive_labels) / labelled_candidates if labelled_candidates else None
    return {
        "source_id": source,
        "labelled_intervals": len(source_labels),
        "positive_intervals": len(positive_labels),
        "candidates": len(candidates),
        "candidates_in_labelled_spans": len(overlapping) + len(false_positives),
        "unlabelled_candidates": len(unlabelled),
        "matched_positive": len(true_positive_labels),
        "matched_negative": len(negative_hits),
        "missed_positive": len([lb for lb in misses if lb["label"] == KIND_EVENT]),
        "recall": None if recall is None else round(recall, 3),
        "precision_within_labelled_spans": None if precision is None else round(precision, 3),
        "label_coverage": round(
            len(overlapping) + len(false_positives), max(1, len(candidates))
        ) if candidates else None,
        "false_positives": false_positives,
        "unlabelled": unlabelled,
        "misses": misses,
        "min_overlap": min_overlap,
        "note": (
            "Recall and precision are only as good as the label file. Precision "
            "is computed inside labelled spans; candidates outside them are "
            "unverified, not negative. A handful of intervals is a smoke check, "
            "not a benchmark."
        ),
    }


def evaluate_files(
    report_paths: Iterable[Path], labels_path: Path, *, min_overlap: float = DEFAULT_OVERLAP
) -> dict:
    labels = load_labels(labels_path)
    per_source = []
    for path in report_paths:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        per_source.append(evaluate(report, labels, min_overlap=min_overlap))
    labelled_sources = {item["source"] for item in labels}
    seen_sources = {item["source_id"] for item in per_source}
    return {
        "labels_file": str(labels_path),
        "sources_in_labels": sorted(labelled_sources),
        "sources_evaluated": sorted(seen_sources),
        "sources_without_report": sorted(labelled_sources - seen_sources),
        "per_source": per_source,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Score visual candidates against reviewed intervals")
    parser.add_argument("labels", type=Path)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--min-overlap", type=float, default=DEFAULT_OVERLAP)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    result = evaluate_files(args.reports, args.labels, min_overlap=args.min_overlap)
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
