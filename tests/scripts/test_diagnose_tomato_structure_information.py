# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Non-tautological geometry and score-independent conditional-pair checks."""

import numpy as np
import pytest

from scripts.diagnose_tomato_structure_information import closest_pair, geometry


def test_equal_iou_distinguishes_center_break_from_end_trimming() -> None:
    """Equal overlap need not imply equal internal connectivity."""
    gt = np.zeros((11, 23), dtype=bool)
    gt[4:7, 2:21] = True
    cut = gt.copy()
    cut[:, 11] = False
    end = gt.copy()
    end[:, 20] = False
    broken = geometry(cut, gt)
    trimmed = geometry(end, gt)
    assert broken["mask_iou"] == pytest.approx(trimmed["mask_iou"])
    assert broken["split_components"] == 1
    assert trimmed["split_components"] == 0
    assert broken["connected_center_fraction"] < trimmed["connected_center_fraction"]


def test_center_coverage_does_not_protect_against_overgrowth() -> None:
    """A broad mask can cover the center while adding many false pixels."""
    gt = np.zeros((21, 31), dtype=bool)
    gt[9:12, 4:27] = True
    pred = np.zeros_like(gt)
    pred[6:15, 4:27] = True
    row = geometry(pred, gt)
    assert row["center_recall"] == 1
    assert row["fp_over_gt"] == pytest.approx(2)
    assert row["mask_iou"] == pytest.approx(1 / 3)


def test_empty_prediction_is_zero_quality_and_empty_gt_is_rejected() -> None:
    """Do not reward empty masks or fabricate skeleton quality for absent GT."""
    gt = np.zeros((9, 9), dtype=bool)
    gt[2:7, 4] = True
    row = geometry(np.zeros_like(gt), gt)
    assert row["cldice"] == row["center_recall"] == row["connected_center_fraction"] == 0
    with pytest.raises(ValueError, match="nonempty"):
        geometry(gt, np.zeros_like(gt))


def test_conditional_pair_ignores_scores_and_respects_caliper() -> None:
    """Match by IoU closeness; do not choose a pair because scores support a hypothesis."""
    rows = [
        dict(query=0, mask_iou=0.6, cldice=0.7, score=0.9),
        dict(query=1, mask_iou=0.601, cldice=0.9, score=0.1),
        dict(query=2, mask_iou=0.75, cldice=0.95, score=0.99),
    ]
    pair = closest_pair(rows, 0.02)
    assert pair == (0, 1)
    for row in rows:
        row["score"] = 1 - row["score"]
    assert closest_pair(rows, 0.02) == pair
    assert closest_pair(rows, 0.0001) is None
