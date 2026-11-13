# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Recall audit contracts for competition, low scores, and background images."""

from scripts.diagnose_tomato_recall import at_threshold, diagnose_misses, recall_curve


def test_curve_counts_duplicates_and_empty_images() -> None:
    """Duplicate predictions and predictions on negative images must reduce precision."""
    records = [
        dict(
            target_indices=[0],
            mask_iou=[[0.8], [0.7]],
            native_stems=[dict(query=0, score=0.8), dict(query=1, score=0.8)],
        ),
        dict(target_indices=[], mask_iou=[[]], native_stems=[dict(query=0, score=0.9)]),
    ]
    curve = recall_curve(records)
    assert len(curve) == 2
    assert at_threshold(curve, 0.8, 1) == dict(threshold=0.8, tp=1, fp=2, misses=0, precision=1 / 3, recall=1.0)
    assert at_threshold(curve, 1.0, 1)["tp"] == 0


def test_low_score_is_distinct_from_mask_failure() -> None:
    """All-query geometry separates a good low-score mask from a poorly segmented located box."""
    row = dict(
        image_id=1,
        target_indices=[4, 7],
        query_scores=[[0.8, 0.1, 0.0], [0.0, 0.8, 0.0]],
        mask_iou=[[0.9, 0.0], [0.0, 0.3]],
        box_iou=[[0.9, 0.0], [0.0, 0.9]],
        native_stems=[dict(query=1, score=0.8)],
    )
    misses = diagnose_misses(row)
    assert [m["reason"] for m in misses] == ["low_stem_score", "box_found_mask_below_iou50"]
    assert misses[0]["other_class_competition"]
    assert not misses[1]["other_class_competition"]


def test_one_prediction_cannot_recover_two_stems() -> None:
    """A duplicate GT match is an assignment conflict, not a recoverable low score."""
    row = dict(
        image_id=1,
        target_indices=[4, 7],
        query_scores=[[0.0, 0.8, 0.0]],
        mask_iou=[[0.9, 0.7]],
        box_iou=[[0.9, 0.7]],
        native_stems=[dict(query=0, score=0.8)],
    )
    assert recall_curve([row])[-1]["recall"] == 0.5
    assert diagnose_misses(row)[0]["reason"] == "assignment_conflict"


def test_topk_filter_is_not_a_score_threshold() -> None:
    """A good high-scoring query omitted by native top-k has a separate diagnostic label."""
    row = dict(
        image_id=1,
        target_indices=[0],
        query_scores=[[0.0, 0.4, 0.0]],
        mask_iou=[[0.9]],
        box_iou=[[0.9]],
        native_stems=[],
    )
    assert diagnose_misses(row)[0]["reason"] == "native_topk_filter"
