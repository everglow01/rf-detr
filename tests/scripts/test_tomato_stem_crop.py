# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Verify the stem guard changes only severely truncated training crops."""

from types import SimpleNamespace

import pytest
import torch

from rfdetr.datasets import _torchvision as tv
from rfdetr.datasets.coco import _build_train_resize_transforms
from scripts.tomato_stem_crop import StemCropGuard, StemGuardedCrop


@pytest.mark.parametrize(
    ("label", "left", "stop", "fallback"),
    [(1, 4, 10, True), (1, 3, 8, False), (1, 6, 10, False), (0, 4, 10, False), (1, 0, 4, False)],
)
def test_partial_stem_guard(label, left, stop, fallback, monkeypatch):
    """At a fixed crop, handle a severe fragment, exact boundary, absent stem and other class."""
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor(0))
    image = torch.arange(100).reshape(1, 10, 10).to(torch.uint8)
    mask = torch.zeros(1, 10, 10, dtype=torch.bool)
    mask[0, 2, left:stop] = True
    target = {"labels": torch.tensor([label]), "masks": mask, "boxes": torch.tensor([[left, 2, stop, 3.0]])}
    native = tv.RandomSizedCrop((6, 6), (8, 8))
    expected = tv.Resize((8, 8)) if fallback else native
    actual_image, actual = StemGuardedCrop(native)(image, target)
    expected_image, expected_target = expected(image, target)
    assert torch.equal(actual_image, expected_image)
    for key in ("masks", "boxes", "labels"):
        assert torch.equal(actual[key], expected_target[key])


def test_guard_consumes_same_rng():
    """A background example stays bitwise equal, including the RNG for later transforms."""
    sample = torch.arange(144).reshape(1, 12, 12).to(torch.uint8)
    mask = torch.zeros(0, 12, 12, dtype=torch.bool)
    target = {"labels": torch.empty(0, dtype=torch.long), "masks": mask}
    native = tv.RandomSizedCrop((4, 11), (8, 8))
    for seed in range(12):
        torch.manual_seed(seed)
        expected, _ = native(sample, target)
        state = torch.get_rng_state()
        torch.manual_seed(seed)
        actual, _ = StemGuardedCrop(native)(sample, target)
        assert torch.equal(expected, actual)
        assert torch.equal(state, torch.get_rng_state())


def test_install_only_training_pipeline():
    """The optional fit callback leaves validation and the direct resize branch identical."""

    def build():
        return tv.Compose([_build_train_resize_transforms([504], square=True)])

    train, val = build(), build()
    direct = train.transforms[0].transform1
    trainer = SimpleNamespace(
        datamodule=SimpleNamespace(
            _dataset_train=SimpleNamespace(_transforms=train), _dataset_val=SimpleNamespace(_transforms=val)
        )
    )
    StemCropGuard().on_fit_start(trainer, None)
    assert train.transforms[0].transform1 is direct
    assert isinstance(train.transforms[0].transform2.transforms[1].transforms[0], StemGuardedCrop)
    assert type(val.transforms[0].transform2.transforms[1].transforms[0]) is tv.RandomSizedCrop
