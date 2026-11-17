# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Calibration and paired-configuration invariants for the authorized CE experiment."""

from pathlib import Path

import pytest
import yaml

from scripts.diagnose_tomato_centerline_ce import choose_coefficient
from scripts.tomato_centerline_ce_short import ARMS, REPO, ce_config


@pytest.mark.parametrize("extreme, expected", [(0.2, 0.05), (10.0, 0.01)])
def test_coefficient_targets_random_baseline_and_caps_extremes(extreme: float, expected: float) -> None:
    """Known ratios give the declared scale; one hard outlier reduces both arms' common coefficient."""
    rows = [
        dict(arm="baseline", role="random", mode="centerline_local_ce", norm_ratio=0.2, max_seed_norm_ratio=0.2)
        for _ in range(16)
    ]
    rows.append(dict(arm="beta2", role="hard", mode="centerline_ce", norm_ratio=extreme, max_seed_norm_ratio=extreme))
    selected = choose_coefficient(rows)
    assert selected["coefficient"] == pytest.approx(expected)
    assert selected["observed_maximum_ratio"] <= 1


def test_three_arms_change_only_the_declared_loss(tmp_path: Path) -> None:
    """Keep initialization, native recipe, scheduler and callbacks equal; only mode/outer coefficient differ."""
    base = yaml.safe_load((REPO / "configs/tomato/seg_small_baseline.yaml").read_text())
    configs = [ce_config(base, tmp_path, arm, 0.023) for arm in ARMS]
    for config, arm in zip(configs, ARMS, strict=True):
        train = config["model"]["train_config"]["init_args"]
        assert train.pop("mask_cldice_loss_coef") == (0 if arm == "baseline" else 0.023)
        assert train.pop("mask_cldice_mode") == ("cldice" if arm == "baseline" else arm)
        assert train["epochs"] == 100 and train["mask_centerline_ce_radius"] == 3
    assert configs[0] == configs[1] == configs[2]
