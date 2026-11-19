# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Regression tests for visible-point denominators, duplicate predictions and geometric references."""

import numpy as np
import pytest
from pycocotools import mask as mask_utils

from scripts.evaluate_tomato_points import evaluate, geometry


def test_matching_denominators_and_native_point() -> None:
    """Keep missed stems in the denominator and evaluate the point of the matched prediction."""
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[2, 1:6] = 1
    rle = mask_utils.encode(np.asfortranarray(mask))
    images = [{"id": n, "height": 10, "width": 12} for n in (1, 2, 3)]
    annotations = [
        {"id": n, "image_id": n, "category_id": 1, "bbox": [1, 2, 5, 1], "segmentation": rle, "keypoints": [3, 2, 2]}
        for n in (1, 2)
    ]
    rows = [
        {"image_id": 1, "scores": [0.31, 0.9], "pred_masks": [rle, rle], "pick_points": [[3, 2], None]},
        {"image_id": 2, "scores": [], "pred_masks": [], "pick_points": []},
        {"image_id": 3, "scores": [0.5], "pred_masks": [rle], "pick_points": [[3, 2]]},
    ]
    dataset = {"images": images, "annotations": annotations}
    summary, _ = evaluate(dataset, rows)
    assert (summary["tp"], summary["fp"], summary["fn"]) == (1, 2, 1)
    for method in ("centroid", "skeleton"):
        assert summary["methods"][method]["pck_0.10"] == {"hits": 1, "conditional": 1, "end_to_end": 0.5}
    assert summary["methods"]["point_head"]["pck_0.10"]["hits"] == 0
    rows[0]["pick_points"][1] = [3, 2]
    assert evaluate(dataset, rows)[0]["methods"]["point_head"]["pck_0.10"]["hits"] == 1
    with pytest.raises(ValueError, match="coverage"):
        evaluate(dataset, rows[:2])


def test_skeleton_single_pixel_and_disconnected_mask() -> None:
    """Use the largest component for the skeleton and the complete mask for the centroid."""
    mask = np.zeros((9, 15), dtype=bool)
    mask[5, 10] = True
    assert geometry(mask)["skeleton"] == (10, 5)
    mask[1, 1:6] = True
    assert geometry(mask)["skeleton"] == (3, 1)
    assert geometry(mask)["centroid"] != (3, 1)
