# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""CPU contracts for the accepted tomato segmentation/keypoint dataset release."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from pycocotools import mask as mask_utils

from rfdetr.datasets._torchvision import RandomHorizontalFlip, Resize, crop
from rfdetr.datasets.coco import CocoDetection
from rfdetr.datasets.yolo import _parse_yolo_label_line, _polygon_to_mask
from tests._markers import _requires_symlink_privilege

#: Load the standalone preparation tool without turning scripts/ into a package.
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "prepare_tomato_keypoints.py"
#: Module spec for the exact tool under test.
SPEC = importlib.util.spec_from_file_location("prepare_tomato_keypoints", SCRIPT)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Create a tiny two-split source with one approved stem per image.

    Args:
        tmp_path: Pytest's isolated directory.

    Returns:
        Source, draft annotations, approval record, and new destination paths.

    Examples:
        >>> inputs(Path("unused"))  # doctest: +SKIP
        # A pytest fixture cannot be called directly.
    """
    source = tmp_path / "yolo"
    points = []
    for split in ("train", "valid"):
        images, labels = source / split / "images", source / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir()
        image, label = images / "sample.png", labels / "sample.txt"
        Image.new("RGB", (32, 32)).save(image)
        lines = [f"{cid} 0.125 0.125 0.75 0.125 0.75 0.75 0.125 0.75" for cid in (1, 0, 2)]
        label.write_text("\n".join(lines) + "\n", encoding="utf-8")
        Image.new("RGB", (32, 32)).save(images / "z_fruit_only.png")
        (labels / "z_fruit_only.txt").write_text(lines[1] + "\n", encoding="utf-8")
        points.append(
            {
                "id": f"{split}-1",
                "split": split,
                "image": f"{split}/images/sample.png",
                "label": f"{split}/labels/sample.txt",
                "line_number": 1,
                "label_line_sha256": hashlib.sha256(lines[0].encode()).hexdigest(),
                "label_file_sha256": hashlib.sha256(label.read_bytes()).hexdigest(),
                "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "width": 32,
                "height": 32,
                "keypoint_xy": [12.0, 16.0],
                "keypoint_normalized_xy": [12 / 32, 16 / 32],
                "status": "provisional_endpoint_uncertain",
                "endpoint_uncertain": True,
                "full_batch_human_reviewed": False,
                "strict_evaluation_review_needed": True,
            }
        )
    draft = tmp_path / "annotations.json"
    draft.write_text(json.dumps({"stage": "draft", "annotations": points}), encoding="utf-8")
    approval = tmp_path / "approval.json"
    approval.write_text(
        json.dumps(
            {
                "status": "human_approved",
                "scope": "all_annotations",
                "annotations_sha256": hashlib.sha256(draft.read_bytes()).hexdigest(),
                "annotation_count": 2,
                "user_statement": "All reviewed and accepted.",
            }
        ),
        encoding="utf-8",
    )
    return source, draft, approval, tmp_path / "release"


class TestRelease:
    """Verify provenance, complete coverage, and preservation of original masks."""

    @_requires_symlink_privilege
    def test_complete_release_and_native_targets(self, inputs: tuple[Path, Path, Path, Path]) -> None:
        source, draft, approval, destination = inputs
        report = prepare.build_dataset(source, draft, approval, destination)
        assert report["splits"]["train"] == {"images": 2, "instances": 4, "keypoints": 1}
        assert not (destination / "test").exists()
        accepted = json.loads((destination / "keypoints.approved.json").read_text())
        assert all(a["full_batch_human_reviewed"] for a in accepted["annotations"])
        assert all(a["endpoint_uncertain"] for a in accepted["annotations"])
        for split in ("train", "valid"):
            ann_path = destination / split / "_annotations.coco.json"
            coco = json.loads(ann_path.read_text())
            assert [a["num_keypoints"] for a in coco["annotations"]] == [1, 0, 0, 0]
            assert coco["annotations"][0]["keypoints"] == [12, 16, 2]
            label = source / split / "labels/sample.txt"
            for line_number, (line, ann) in enumerate(zip(label.read_text().splitlines(), coco["annotations"]), 1):
                _, _, polygon = _parse_yolo_label_line(line.split(), line_number, label, 3, 32, 32)
                np.testing.assert_array_equal(
                    mask_utils.decode(ann["segmentation"]),
                    _polygon_to_mask(polygon, (32, 32)),
                )
            baseline = CocoDetection(destination / split, ann_path, None, include_masks=True, remap_category_ids=True)
            joint = CocoDetection(
                destination / split,
                ann_path,
                None,
                include_masks=True,
                include_keypoints=True,
                num_keypoints_per_class=[0, 1, 0],
                remap_category_ids=True,
            )
            image, target = joint[0]
            _, fruit_target = joint[1]
            assert fruit_target["labels"].tolist() == [0]
            assert fruit_target["keypoints"].shape == (1, 1, 3)
            assert not fruit_target["keypoints"].any()
            _, base_target = baseline[0]
            assert "keypoints" not in base_target
            assert baseline.cat2label == joint.cat2label == {0: 0, 1: 1, 2: 2}
            for field in ("boxes", "labels", "masks", "area"):
                assert torch.equal(target[field], base_target[field])
            _, flipped = RandomHorizontalFlip(p=1)(image, target)
            assert flipped["keypoints"][0].tolist() == [[20, 16, 2]]
            _, resized = Resize((64, 64))(image, target)
            assert resized["keypoints"][0].tolist() == [[24, 32, 2]]
            _, cropped = crop(image, target, top=0, left=16, height=32, width=16)
            assert len(cropped["boxes"]) == 3
            assert cropped["keypoints"][0, 0, 2] == 0

    @pytest.mark.parametrize("field", ["image_sha256", "label_file_sha256", "label_line_sha256"])
    def test_rejects_changed_sources(self, inputs: tuple[Path, Path, Path, Path], field: str) -> None:
        source, draft, approval, destination = inputs
        data = json.loads(draft.read_text())
        data["annotations"][0][field] = "0" * 64
        draft.write_text(json.dumps(data))
        record = json.loads(approval.read_text())
        record["annotations_sha256"] = hashlib.sha256(draft.read_bytes()).hexdigest()
        approval.write_text(json.dumps(record))
        with pytest.raises(ValueError, match=field):
            prepare.build_dataset(source, draft, approval, destination)
        assert not destination.exists()

    @pytest.mark.parametrize(
        "change",
        [pytest.param("missing", marks=_requires_symlink_privilege), "duplicate", "out_of_bounds", "outside_mask"],
    )
    def test_rejects_invalid_points(self, inputs: tuple[Path, Path, Path, Path], change: str) -> None:
        source, draft, approval, destination = inputs
        data = json.loads(draft.read_text())
        if change == "missing":
            data["annotations"].pop()
        elif change == "duplicate":
            data["annotations"].append(data["annotations"][0])
        else:
            xy = [40, 16] if change == "out_of_bounds" else [2, 2]
            data["annotations"][0]["keypoint_xy"] = xy
            data["annotations"][0]["keypoint_normalized_xy"] = [v / 32 for v in xy]
        draft.write_text(json.dumps(data))
        record = json.loads(approval.read_text())
        record.update(
            annotations_sha256=hashlib.sha256(draft.read_bytes()).hexdigest(), annotation_count=len(data["annotations"])
        )
        approval.write_text(json.dumps(record))
        with pytest.raises(ValueError):
            prepare.build_dataset(source, draft, approval, destination)
        assert not destination.exists()

    def test_rejects_unapproved_artifact(self, inputs: tuple[Path, Path, Path, Path]) -> None:
        source, draft, approval, destination = inputs
        draft.write_text(draft.read_text() + "\n")
        with pytest.raises(ValueError, match="approval"):
            prepare.build_dataset(source, draft, approval, destination)
        assert not destination.exists()

    def test_refuses_overwrite(self, inputs: tuple[Path, Path, Path, Path]) -> None:
        source, draft, approval, destination = inputs
        destination.mkdir()
        marker = destination / "keep.txt"
        marker.write_text("keep")
        with pytest.raises(FileExistsError):
            prepare.build_dataset(source, draft, approval, destination)
        assert marker.read_text() == "keep"
