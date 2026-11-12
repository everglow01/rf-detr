# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Small mathematical checks for the read-only E1 diagnostic."""

import importlib.util
from pathlib import Path

import pytest
import torch

from rfdetr.models.soft_cldice import _soft_skeleton

SPEC = importlib.util.spec_from_file_location(
    "diagnose_tomato_e1", Path(__file__).resolve().parents[2] / "scripts/diagnose_tomato_e1.py"
)
diagnose = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnose)


def test_skeleton_reference_exposes_truncated_thick_target():
    """A thick diamond has no early skeleton but a nonempty converged center."""
    y, x = torch.meshgrid(torch.arange(61), torch.arange(61), indexing="ij")
    mask = ((x - 30).abs() + (y - 30).abs() <= 20).unsqueeze(0)
    short, full, last, extinct = diagnose.skeleton_trace(mask)
    assert extinct.all()
    assert last.item() > 10
    assert full.sum() > short.sum()
    torch.testing.assert_close(short, _soft_skeleton(mask[:, None].float(), 10)[:, 0])
    torch.testing.assert_close(full, _soft_skeleton(mask[:, None].float(), 60)[:, 0])


def test_skeleton_reference_keeps_thin_target_complete():
    """A one-pixel line is covered at K=10."""
    mask = torch.zeros(1, 25, 25, dtype=torch.bool)
    mask[:, 3:22, 12] = True
    short, full, _, extinct = diagnose.skeleton_trace(mask)
    assert extinct.all()
    torch.testing.assert_close(short, full)
    assert full.sum() == 19


def test_full_foreground_is_reported_unresolved():
    """Pooling ignores outside pixels: all foreground is a non-extinguishing fixed point."""
    _, _, _, extinct = diagnose.skeleton_trace(torch.ones(1, 25, 25))
    assert not extinct.item()


def test_gradient_relation_counts_unused_parameter_as_zero():
    """An unused branch still contributes to the norm of the other branch."""
    result = diagnose.gradient_relation(
        [torch.tensor([3.0, 0.0]), torch.tensor([4.0])], [torch.tensor([-1.0, 0.0]), None]
    )
    assert result["first_norm"] == 5
    assert result["ratio"] == 0.2
    assert result["cosine"] == pytest.approx(-0.6)
    assert result["sum_norm"] == pytest.approx(20**0.5)


def test_misses_distinguish_topk_from_score_filter():
    """A high-IoU excluded query must not count as a selected detection."""
    result = diagnose.greedy_matches(
        torch.tensor([[0.9], [0.6]]), torch.tensor([0.9, 0.2]), torch.tensor([False, True]), 0.3
    )
    assert result == {}


def test_greedy_matching_does_not_reuse_target():
    """Duplicate predictions cannot inflate recall."""
    result = diagnose.greedy_matches(
        torch.tensor([[0.9, 0.1], [0.8, 0.7]]), torch.tensor([0.9, 0.8]), torch.tensor([True, True]), 0.3
    )
    assert result == {0: 0, 1: 1}
