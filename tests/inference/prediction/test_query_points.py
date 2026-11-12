# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Public segmentation detections preserve point association after class thresholds."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from pytest_mock import MockerFixture

from rfdetr._prediction import core
from tests.inference.helpers import _DummyRFDETR


class TestQueryPointPrediction:
    """Point rows follow every public-prediction filter, including empty results."""

    @pytest.mark.parametrize("threshold", [0.2, 0.8, 1.0])
    def test_optional_data_stays_with_masks(self, mocker: MockerFixture, threshold: float) -> None:
        rows = {
            "scores": torch.tensor([0.5, 0.9]),
            "labels": torch.tensor([0, 1]),
            "boxes": torch.zeros(2, 4),
            "masks": torch.ones(2, 1, 10, 20, dtype=torch.bool),
            "pick_points": torch.tensor([[4.0, 5.0], [10.0, 3.0]]),
            "pick_point_valid": torch.tensor([False, True]),
        }
        context = replace(_DummyRFDETR()._prediction_context(), postprocess=mocker.Mock(return_value=[rows]))
        prepared = core._PreparedBatch(
            context=context,
            batch_tensor=torch.zeros(1, 3, 10, 20),
            target_sizes=torch.tensor([[10, 20]]),
            orig_sizes=[(10, 20)],
            source_images=[],
            include_source_image=False,
            single_input=True,
            score_floor=0.1,
            threshold_by_class_id=torch.tensor([1.0, threshold, float("inf")]),
        )
        result = core._decode_batch(prepared, {})
        indices = [1] if threshold < 0.9 else []
        np.testing.assert_array_equal(result.data["pick_points"], rows["pick_points"][indices].numpy())
        np.testing.assert_array_equal(result.data["pick_point_valid"], rows["pick_point_valid"][indices].numpy())
        assert len(result.mask) == len(result.data["pick_points"])
