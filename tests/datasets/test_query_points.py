# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Single-point annotation routing preserves the segmentation augmentation recipe."""

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import RFDETRSegNanoConfig, SegmentationTrainConfig
from rfdetr.datasets import build_dataset
from rfdetr.datasets._torchvision import RandomHorizontalFlip, Resize, crop
from rfdetr.datasets.transforms import Normalize
from rfdetr.utilities.tensors import nested_tensor_from_tensor_list


class TestQueryPointData:
    """Load masks and points together, retain class ids, and transform coordinates."""

    def test_coco_pipeline_preserves_baseline_images_masks_and_rng(self, tmp_path: Path) -> None:
        categories = [
            {"id": i, "name": name, "keypoints": ["pick"] if i == 1 else []}
            for i, name in enumerate(["ripe", "stem", "unripe"])
        ]
        annotations = [
            {
                "id": i,
                "image_id": 1,
                "category_id": i,
                "bbox": [4 + i * 20, 5, 12, 50],
                "area": 600,
                "iscrowd": 0,
                "segmentation": [[4 + i * 20, 5, 16 + i * 20, 5, 16 + i * 20, 55, 4 + i * 20, 55]],
                "keypoints": [30, 25, 2] if i == 1 else [],
                "num_keypoints": int(i == 1),
            }
            for i in range(3)
        ]
        for split in ("train", "valid"):
            folder = tmp_path / split
            folder.mkdir()
            Image.new("RGB", (100, 60), (20, 40, 80)).save(folder / "sample.png")
            (folder / "_annotations.coco.json").write_text(
                json.dumps(
                    {
                        "categories": categories,
                        "images": [{"id": 1, "file_name": "sample.png", "width": 100, "height": 60}],
                        "annotations": annotations,
                    }
                ),
                encoding="utf-8",
            )
        tc = SegmentationTrainConfig(
            dataset_dir=str(tmp_path), dataset_file="roboflow", augmentation_backend="torchvision"
        )
        datasets = []
        for enabled in (False, True):
            mc = RFDETRSegNanoConfig(num_classes=3, device="cpu", query_point_head=enabled, query_point_class_ids=[1])
            datasets.append(build_dataset("train", _namespace_from_configs(mc, tc), 384))
        assert datasets[0].cat2label == datasets[1].cat2label
        for seed in range(12):
            torch.manual_seed(seed)
            image0, target0 = datasets[0][0]
            rng0 = torch.get_rng_state()
            torch.manual_seed(seed)
            image1, target1 = datasets[1][0]
            assert torch.equal(rng0, torch.get_rng_state())
            torch.testing.assert_close(image0, image1, rtol=0, atol=0)
            for key in ("boxes", "labels", "masks"):
                torch.testing.assert_close(target0[key], target1[key], rtol=0, atol=0)
            assert target1["keypoints"].shape == (len(target1["labels"]), 1, 3)
            assert (target1["keypoints"][target1["labels"] != 1, :, 2] == 0).all()

    @pytest.mark.parametrize("x", [0.0, 10.0, 20.0])
    def test_non_square_flip_resize_normalize_and_padding(self, x: float) -> None:
        image = torch.zeros(3, 10, 20)
        target = {"keypoints": torch.tensor([[[x, 3.0, 2.0]]])}
        image, target = RandomHorizontalFlip(p=1)(image, target)
        image, target = Resize((20, 60))(image, target)
        image, target = Normalize()(image, target)
        torch.testing.assert_close(target["keypoints"][0, 0], torch.tensor([1 - x / 20, 0.3, 2.0]))
        before = target["keypoints"].clone()
        nested = nested_tensor_from_tensor_list([image, torch.zeros(3, 25, 70)])
        assert nested.mask[0, 20:, :].all() and nested.mask[0, :, 60:].all()
        # Coordinates follow the same per-image convention as boxes, not the padded canvas.
        assert torch.equal(before, target["keypoints"])

    def test_crop_retains_stem_but_invalidates_removed_point(self) -> None:
        target = {
            "boxes": torch.tensor([[0.0, 0.0, 20.0, 10.0]]),
            "labels": torch.tensor([1]),
            "masks": torch.ones(1, 10, 20, dtype=torch.bool),
            "keypoints": torch.tensor([[[18.0, 5.0, 2.0]]]),
        }
        _, target = crop(torch.zeros(3, 10, 20), target, 0, 0, 10, 10)
        assert target["labels"].tolist() == [1]
        assert target["masks"].any() and target["keypoints"][0, 0, 2] == 0
