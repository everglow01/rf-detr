# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Release reviewed tomato points as COCO masks + keypoints, without changing model code."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from rfdetr.datasets.yolo import _parse_yolo_label_line, _polygon_to_mask

#: Keep the original YOLO class order; only the whole-truss stem has a keypoint.
CATEGORIES = [
    {"id": 0, "name": "ripe", "supercategory": "tomato", "keypoints": [], "skeleton": []},
    {"id": 1, "name": "stem", "supercategory": "tomato", "keypoints": ["whole_truss_candidate_cut"], "skeleton": []},
    {"id": 2, "name": "unripe", "supercategory": "tomato", "keypoints": [], "skeleton": []},
]


def sha256(path: Path) -> str:
    """Hash the exact bytes of a source or release artifact.

    Args:
        path: File to hash.

    Returns:
        SHA-256 hexadecimal digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: Any) -> None:
    """Write a deterministic UTF-8 JSON artifact.

    Args:
        path: Destination file.
        data: Serializable content.
    """
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def convert_split(
    source: Path,
    split: str,
    points: dict[tuple[str, int], dict[str, Any]],
) -> tuple[dict[str, Any], set[tuple[str, int]]]:
    """Convert every original instance and verify exact reviewed-point provenance.

    Args:
        source: Original YOLO segmentation dataset root.
        split: Original train or valid split.
        points: Reviewed points indexed by original label path and line number.

    Returns:
        COCO content and the reviewed-point keys consumed by this split.

    Raises:
        ValueError: Original hashes, point coverage, or point geometry do not agree.
    """
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    used: set[tuple[str, int]] = set()
    image_paths = sorted(
        p for p in (source / split / "images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise ValueError(f"Empty image split: {split}")
    label_paths = {source / split / "labels" / f"{p.stem}.txt" for p in image_paths}
    if len(label_paths) != len(image_paths) or label_paths != set((source / split / "labels").glob("*.txt")):
        raise ValueError(f"Image/label pairing mismatch: {split}")
    for image_id, image_path in enumerate(image_paths, 1):
        label_path = source / split / "labels" / f"{image_path.stem}.txt"
        label_rel, image_rel = label_path.relative_to(source).as_posix(), image_path.relative_to(source).as_posix()
        with Image.open(image_path) as image:
            width, height = image.size
        image_hash, label_hash = sha256(image_path), sha256(label_path)
        images.append(
            {
                "id": image_id,
                "file_name": f"images/{image_path.name}",
                "width": width,
                "height": height,
                "source_image": image_rel,
                "image_sha256": image_hash,
                "source_label": label_rel,
                "label_file_sha256": label_hash,
            }
        )
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            # These pinned upstream helpers preserve the exact YOLO float32 clipping and PIL rasterization.
            if len(line.split()) < 7:
                raise ValueError(f"Expected an instance polygon: {label_rel}:{line_number}")
            category, xyxy, polygon = _parse_yolo_label_line(
                line.split(),
                line_number,
                label_path,
                num_classes=3,
                width=width,
                height=height,
            )
            if category not in (0, 1, 2) or xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
                raise ValueError(f"Invalid category or box: {label_rel}:{line_number}")
            assert polygon is not None  # The polygon-only token-count guard above excludes box labels.
            mask = _polygon_to_mask(polygon, (width, height))
            rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("ascii")
            line_hash = hashlib.sha256(line.encode("utf-8")).hexdigest()
            annotation = {
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": category,
                "bbox": [float(xyxy[0]), float(xyxy[1]), float(xyxy[2] - xyxy[0]), float(xyxy[3] - xyxy[1])],
                "area": int(mask.sum()),
                "iscrowd": 0,
                "segmentation": rle,
                "keypoints": [],
                "num_keypoints": 0,
                "source": {"label": label_rel, "line_number": line_number, "label_line_sha256": line_hash},
            }
            key = (label_rel, line_number)
            if category == 1:
                if key not in points:
                    raise ValueError(f"Missing approved stem keypoint: {key}")
                point = points[key]
                for field, expected in {
                    "split": split,
                    "image": image_rel,
                    "width": width,
                    "height": height,
                    "image_sha256": image_hash,
                    "label_file_sha256": label_hash,
                    "label_line_sha256": line_hash,
                }.items():
                    if point[field] != expected:
                        raise ValueError(f"{field} mismatch for {key}")
                x, y = point["keypoint_xy"]
                if not (np.isfinite([x, y]).all() and 0 <= x < width and 0 <= y < height):
                    raise ValueError(f"Keypoint out of bounds: {key}")
                if not np.allclose(point["keypoint_normalized_xy"], [x / width, y / height], atol=1e-7, rtol=0):
                    raise ValueError(f"Keypoint normalization mismatch: {key}")
                if not mask[min(round(y), height - 1), min(round(x), width - 1)]:
                    raise ValueError(f"Keypoint outside original mask: {key}")
                annotation.update(
                    keypoints=[x, y, 2],
                    num_keypoints=1,
                    tomato_keypoint={
                        "annotation_id": point["id"],
                        "status": "human_approved",
                        "endpoint_uncertain": point.get("endpoint_uncertain", False),
                        "draft_status": point["status"],
                        "group_id": point.get("group_id"),
                    },
                )
                used.add(key)
            elif key in points:
                raise ValueError(f"Keypoint belongs to a non-stem instance: {key}")
            annotations.append(annotation)
    return {"images": images, "annotations": annotations, "categories": CATEGORIES}, used


def build_dataset(source: Path, draft_path: Path, approval_path: Path, destination: Path) -> dict[str, Any]:
    """Build a new, versioned release after whole-batch human acceptance.

    Original images and YOLO labels are never edited. Relative directory symlinks reuse the source images.
    A temporary sibling directory is published only when every point and source hash passes validation.

    Args:
        source: Original YOLO segmentation dataset root.
        draft_path: Exact reviewed annotation JSON.
        approval_path: User acceptance record bound to the reviewed JSON hash.
        destination: New COCO dataset directory; must not exist.

    Returns:
        Release counts, source hashes, and artifact checksums.

    Raises:
        FileExistsError: Destination already exists.
        ValueError: Approval, source provenance, or coverage is invalid.
    """
    source, destination = source.resolve(), destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    records = draft["annotations"]
    if (
        approval.get("status") != "human_approved"
        or approval.get("scope") != "all_annotations"
        or approval.get("annotations_sha256") != sha256(draft_path)
        or approval.get("annotation_count") != len(records)
    ):
        raise ValueError("Invalid approval or reviewed artifact checksum")
    points = {(point["label"], point["line_number"]): point for point in records}
    if len(points) != len(records) or len({p["id"] for p in records}) != len(records):
        raise ValueError("Duplicate keypoint instance or annotation id")
    used: set[tuple[str, int]] = set()
    report: dict[str, Any] = {"schema_version": "tomato-coco-keypoints-v1", "source_dataset": str(source), "splits": {}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=destination.name + ".building-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        for split in ("train", "valid"):
            coco, consumed = convert_split(source, split, points)
            used.update(consumed)
            split_dir = staging / split
            split_dir.mkdir()
            (split_dir / "images").symlink_to(
                os.path.relpath(source / split / "images", split_dir),
                target_is_directory=True,
            )
            coco["info"] = {"description": "User-approved whole-truss cut keypoints with original YOLO instance masks"}
            write_json(split_dir / "_annotations.coco.json", coco)
            report["splits"][split] = {
                "images": len(coco["images"]),
                "instances": len(coco["annotations"]),
                "keypoints": len(consumed),
            }
        if used != points.keys():
            raise ValueError(f"Unmatched reviewed keypoints: {len(points.keys() - used)}")
        accepted = copy.deepcopy(draft)
        accepted.update(stage="human_approved", approval=approval)
        for point in accepted["annotations"]:
            point["draft_status"] = point["status"]
            point.update(status="human_approved", full_batch_human_reviewed=True, strict_evaluation_review_needed=False)
        # Old summary/user_feedback describe draft review; preserve them under an explicitly historical key.
        accepted["draft_review_metadata"] = {
            key: accepted.pop(key) for key in ("summary", "user_feedback") if key in accepted
        }
        accepted["summary"] = {
            "total": len(records),
            "human_approved": len(records),
            "by_split": dict(Counter(p["split"] for p in records)),
        }
        write_json(staging / "keypoints.approved.json", accepted)
        write_json(staging / "approval.json", approval)
        report.update(
            reviewed_annotations_sha256=sha256(draft_path),
            approval_sha256=sha256(approval_path),
            preparation_script_sha256=sha256(Path(__file__)),
            files={p.relative_to(staging).as_posix(): sha256(p) for p in sorted(staging.rglob("*.json"))},
            image_storage="relative symlinks; keep original source dataset available",
            test_split=False,
            mask_encoding="COCO compressed RLE from pinned native YOLO rasterizer",
        )
        write_json(staging / "release.json", report)
        staging.rename(destination)
    return report


def main() -> None:
    """Parse explicit input paths and publish the accepted dataset on CPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.source, args.annotations, args.approval, args.destination), indent=2))


if __name__ == "__main__":
    main()
