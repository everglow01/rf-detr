# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Topology metadata is opt-in and never changes predictions or model parameters."""

import pytest
import torch
from pytest_mock import MockerFixture

from rfdetr.config import RFDETRSegNanoConfig, SegmentationTrainConfig
from rfdetr.models.lwdetr import build_model_from_config
from rfdetr.utilities.tensors import NestedTensor
from tests.models.test_lwdetr_segmentation_eval import _build_seg_lwdetr, _images
from tests.models.test_soft_cldice_integration import _criterion


class TestClDiceModel:
    """Exercise the real mask head and model forward with fixed backbone features."""

    @pytest.mark.parametrize("enabled", [False, True])
    def test_builder_forwards_opt_in(self, mocker: MockerFixture, enabled: bool) -> None:
        """The public config builder reaches the model's metadata switch."""
        stub = _build_seg_lwdetr()
        mocker.patch("rfdetr.models.lwdetr.build_backbone", return_value=stub.backbone)
        mocker.patch("rfdetr.models.lwdetr.build_transformer", return_value=stub.transformer)
        mc = RFDETRSegNanoConfig(num_classes=3, hidden_dim=8, num_queries=3, group_detr=1, device="cpu")
        tc = SegmentationTrainConfig(
            dataset_dir="/tmp", mask_cldice_loss_coef=0.1 if enabled else 0.0, mask_cldice_class_ids=[1]
        )
        assert build_model_from_config(mc, tc).mask_cldice_enabled is enabled

    @pytest.mark.parametrize("training", [False, True])
    def test_metadata_does_not_change_predictions(self, training: bool) -> None:
        """Existing model tensors and state_dict keys remain bitwise identical."""
        model = _build_seg_lwdetr().train(training)
        keys = set(model.state_dict())
        expected = model(_images(), all_layer_masks=True)
        assert "mask_padding" not in expected
        model.mask_cldice_enabled = True
        actual = model(_images(), all_layer_masks=True)
        assert set(actual) - set(expected) == {"mask_padding"}
        assert set(model.state_dict()) == keys
        for left, right in zip(
            [expected, *expected["aux_outputs"], expected["enc_outputs"]],
            [actual, *actual["aux_outputs"], actual["enc_outputs"]],
        ):
            for key in ("pred_logits", "pred_boxes"):
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
            if isinstance(left["pred_masks"], dict):
                for key in left["pred_masks"]:
                    torch.testing.assert_close(left["pred_masks"][key], right["pred_masks"][key], rtol=0, atol=0)
            else:
                torch.testing.assert_close(left["pred_masks"], right["pred_masks"], rtol=0, atol=0)

    def test_inference_outputs_unchanged(self) -> None:
        """Normal prediction never emits topology metadata, even for an E1-trained model."""
        model = _build_seg_lwdetr().eval()
        expected = model(_images())
        model.mask_cldice_enabled = True
        actual = model(_images())
        assert actual.keys() == expected.keys()
        torch.testing.assert_close(actual["pred_masks"], expected["pred_masks"], rtol=0, atol=0)

    def test_padding_reference_and_mask_head_backward(self) -> None:
        """Model-to-criterion training wiring works with real sparse mask-head outputs."""
        model = _build_seg_lwdetr().train()
        model.mask_cldice_enabled = True
        padding = torch.zeros(2, 8, 8, dtype=torch.bool)
        padding[0, :, 7:] = True
        outputs = model(NestedTensor(_images(), padding))
        assert outputs["mask_padding"] is padding
        masks = torch.zeros(3, 8, 8, dtype=torch.bool)
        masks[:, 1:7, 4] = True
        targets = [{"labels": torch.tensor([0, 1, 2]), "masks": masks} for _ in range(2)]
        loss = _criterion()(outputs, targets)["loss_mask_cldice"]
        loss.backward()
        gradients = [parameter.grad for parameter in model.segmentation_head.parameters()]
        assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(gradient.abs().sum() for gradient in gradients) > 0
