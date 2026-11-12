# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Keep topology supervision on the GT grid during per-batch input resizing."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from pytorch_lightning import LightningModule
from torch import nn

from rfdetr.config import SegmentationTrainConfig
from rfdetr.models.criterion import SetCriterion
from rfdetr.models.soft_cldice import soft_cldice_loss
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.utilities.tensors import NestedTensor


@pytest.mark.parametrize("scale", [6, 18])
@pytest.mark.parametrize("enabled", [False, True])
def test_cldice_preserves_gt_canvas_during_multiscale(
    monkeypatch: pytest.MonkeyPatch, scale: int, enabled: bool
) -> None:
    """Down/upscaled inputs retain GT-grid loss/gradients; disabling topology keeps the original output padding."""
    module = RFDETRModelModule.__new__(RFDETRModelModule)
    LightningModule.__init__(module)
    module.train_config = SegmentationTrainConfig(
        dataset_dir="/tmp",
        multi_scale="per-batch",
        mask_cldice_loss_coef=0.1 if enabled else 0,
        mask_cldice_class_ids=[1],
        compact_train_metrics=False,
        compute_train_metrics=False,
    )
    module.model_config = SimpleNamespace(resolution=12, patch_size=1, num_windows=1)
    module._trainer = SimpleNamespace(global_step=0)
    module._inductor_cudagraphs = False
    module._cuda_graph_runner = None
    module._use_manual_optimization = False
    module.log = Mock()
    module.log_dict = Mock()
    module._log_train_progress_metrics = Mock()
    module.model = nn.Module()
    module.model.register_parameter("logits", nn.Parameter(torch.linspace(-2, 2, 12).reshape(1, 1, 3, 4)))
    monkeypatch.setattr(
        module.model,
        "forward",
        lambda samples, targets: {"pred_masks": module.model.logits, "mask_padding": samples.mask},
    )
    criterion = SetCriterion(
        3, None, {"loss_mask_cldice": 1}, 0.25, [], mask_cldice_class_ids=[1], mask_cldice_iterations=2
    )
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    seen = []
    monkeypatch.setattr(
        criterion,
        "forward",
        lambda outputs, targets: (
            seen.append(outputs["mask_padding"]),
            criterion.loss_mask_cldice(outputs, targets, indices)
            if enabled
            else {"loss_mask_cldice": module.model.logits.sum() * 0},
        )[1],
    )
    module.criterion = criterion
    monkeypatch.setattr("rfdetr.training.module_model.compute_multi_scale_scales", lambda *args: [scale])
    padding = torch.ones(1, 12, 16, dtype=torch.bool)
    padding[:, :9, :13] = False
    samples = NestedTensor(torch.zeros(1, 3, 12, 16), padding)
    masks = torch.zeros(1, 10, 14, dtype=torch.bool)
    masks[:, 2:9, 6:8] = True
    targets = [{"labels": torch.tensor([1]), "masks": masks}]
    module.on_train_batch_start((samples, targets), 0)
    assert samples.tensors.shape[-2:] == (scale, scale)
    assert targets[0]["masks"] is masks
    actual = module.training_step((samples, targets), 0)
    if enabled:
        assert seen[0] is padding
        aligned = F.interpolate(module.model.logits, (12, 16), mode="bilinear", align_corners=False)[0, :, :10, :14]
        valid = ~padding[0, :10, :14]
        expected = soft_cldice_loss(aligned.masked_fill(~valid, -torch.inf), masks & valid, iterations=2).mean()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, module.model.logits, retain_graph=True)[0],
            torch.autograd.grad(expected, module.model.logits)[0],
        )
    else:
        assert seen[0] is samples.mask
    module.train_config.multi_scale = "off"
    module.on_train_batch_start((samples, targets), 1)
    assert getattr(module, "_cldice_target_padding", None) is None
