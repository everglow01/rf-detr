# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Optional training-only crop guard for the tomato segmentation ablation."""

from typing import Any

import torch
from pytorch_lightning import Callback, LightningModule

from rfdetr.datasets import _torchvision as tv


class StemGuardedCrop:
    """Keep the full view when a native crop leaves under three fifths of a visible stem."""

    def __init__(self, native: tv.RandomSizedCrop) -> None:
        self.native = native

    def __call__(self, image: tv._ImageInput, target: tv._Target) -> tv._TransformResult:
        """Draw the native square window once and use the native spatial transforms for all labels."""
        if target is None:
            return self.native(image, target)
        stem = target["labels"] == 1
        if not stem.any():
            return self.native(image, target)
        height, width = tv._image_size(image)
        low, high = self.native.min_max_height
        high = min(high, height, width)
        low = min(low, high)
        # Match RandomSizedCrop's three conditional integer draws, including single-choice cases.
        side = int(torch.randint(low, high + 1, ()).item()) if high > low else high
        top = int(torch.randint(0, height - side + 1, ()).item()) if height > side else 0
        left = int(torch.randint(0, width - side + 1, ()).item()) if width > side else 0
        masks = target["masks"][stem]
        remaining = masks[:, top : top + side, left : left + side].sum((1, 2))
        original = masks.sum((1, 2))
        if not ((remaining > 0) & (remaining * 5 < original * 3)).any():
            image, target = tv.crop(image, target, top, left, side, side)
        return tv.Resize(self.native.size)(image, target)


class StemCropGuard(Callback):
    """Install the guard only when explicitly listed in the experiment's Lightning callbacks."""

    def on_fit_start(self, trainer: Any, pl_module: LightningModule) -> None:
        """Replace just the training crop leaves before the fit dataloader starts its workers."""
        train = trainer.datamodule._dataset_train
        selector = train._transforms.transforms[0]
        if not isinstance(selector, tv.RandomSelect) or not isinstance(selector.transform2, tv.Compose):
            raise ValueError("The stem crop guard requires RF-DETR's native random resize/crop training pipeline")
        choice = selector.transform2.transforms[-1]
        if not isinstance(choice, tv.RandomChoice) or not all(
            isinstance(t, tv.RandomSizedCrop) for t in choice.transforms
        ):
            raise ValueError("Expected native RandomSizedCrop leaves in the training pipeline")
        choice.transforms = [StemGuardedCrop(t) for t in choice.transforms]
        print("Enabled tomato stem crop guard: retain >=60% of any partial stem (class 1)", flush=True)
