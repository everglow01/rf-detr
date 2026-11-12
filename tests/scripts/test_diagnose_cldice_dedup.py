# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""The BF16 parity gate must not excuse a real change as numerical noise."""

import torch

from scripts.diagnose_cldice_dedup import verify_gradient_parity


def test_parity_accepts_small_error_with_measured_control_noise() -> None:
    """A tiny cross-path difference within repeated-backward noise is acceptable."""
    result = verify_gradient_parity(
        [torch.tensor([1.0, 2.0])],
        [torch.tensor([1.0 + 1e-6, 2.0])],
        [torch.tensor([1.0 + 2e-6, 2.0])],
        [torch.tensor([1.0 + 2e-6, 2.0 + 1e-6])],
    )
    assert result["passed"]


def test_parity_rejects_deterministic_gradient_change() -> None:
    """Different exact gradients cannot pass through the self-repeat allowance."""
    result = verify_gradient_parity(
        [torch.tensor([1.0])], [torch.tensor([1.0])], [torch.tensor([1.001])], [torch.tensor([1.001])]
    )
    assert not result["passed"]


def test_parity_rejects_large_error_even_with_noisy_controls() -> None:
    """The global L2 bound remains mandatory when repeated calls are noisy."""
    result = verify_gradient_parity(
        [torch.tensor([1.0])], [torch.tensor([1.01])], [torch.tensor([1.03])], [torch.tensor([1.04])]
    )
    assert not result["passed"]
