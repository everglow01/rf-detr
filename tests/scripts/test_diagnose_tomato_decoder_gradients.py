# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Verify the shared autograd boundary and the meaning of gradient-conflict measurements."""

import torch

from scripts.diagnose_tomato_decoder_gradients import cut_decoder_output, measures, relation


def test_shared_boundary_preserves_both_head_gradients_without_upstream_gradients() -> None:
    """Both independent heads must use the same unchanged feature values after cutting the upstream graph."""
    original = torch.arange(12.0).reshape(2, 1, 2, 3).requires_grad_()
    other = torch.tensor([7.0])
    observed = cut_decoder_output((original, other, None))
    features = observed[0]
    assert features.is_leaf and features.requires_grad
    assert torch.equal(features, original) and observed[1] is other and observed[2] is None
    cls_weights = torch.tensor([1.0, 2.0, 3.0])
    mask_weights = torch.tensor([-2.0, 1.0, -3.0])
    native_cls = (original[-1] * cls_weights).square().sum()
    native_mask = (original[-1] * mask_weights).sum()
    observed_cls = (features[-1] * cls_weights).square().sum()
    observed_mask = (features[-1] * mask_weights).sum()
    for native, cut in ((native_cls, observed_cls), (native_mask, observed_mask)):
        expected = torch.autograd.grad(native, original, retain_graph=True)[0]
        actual = torch.autograd.grad(cut, features, retain_graph=True)[0]
        torch.testing.assert_close(native, cut, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert original.grad is None and features.grad is None
    assert torch.autograd.grad(observed_cls, original, allow_unused=True)[0] is None


def test_projection_detects_overwhelming_conflict_and_does_not_hide_zero_gradients() -> None:
    """A larger opposing term can reverse descent; zero gradients have undefined cosine."""
    cls = torch.tensor([1.0, 0.0])
    mask = torch.tensor([0.0, 1.0])
    topology = torch.tensor([-2.0, 0.0])
    result = measures(cls, mask, topology, torch.tensor([-1.0, 0.0]))
    assert result["cls_mask"]["cosine"] == 0
    assert result["cls_cldice"]["cosine"] == -1
    assert result["cls_cldice"]["right_to_left_norm"] == 2
    assert result["cls_native_sum"]["right_projection_over_left"] == 1
    assert result["cls_sum_with_cldice"]["right_projection_over_left"] == -1
    assert result["stem_logit_descent"] == {"cls": 1, "mask": 0, "cldice": -2}
    assert relation(cls, torch.zeros(2))["cosine"] is None
    assert relation(torch.zeros(2), cls)["right_to_left_norm"] is None
