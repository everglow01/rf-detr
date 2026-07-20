#!/usr/bin/env python3
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Prepare TOMATO COCO data and train LingBot RF-DETR instance segmentation."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

import torch

from rfdetr import RFDETRLingBotSmallSeg

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SOURCE_DATASET = _REPOSITORY_ROOT / "data" / "TOMATO.v7i.coco-segmentation"
_DEFAULT_PREPARED_DATASET = _DEFAULT_SOURCE_DATASET / "rfdetr_dataset"
_DEFAULT_OUTPUT_DIR = _REPOSITORY_ROOT / "output" / "tomato-lingbot-small-seg"
_EXPECTED_MISSING_TRAIN_IMAGES = {
    "IMG20250620110805_slice_1_png.rf.5174007bd540d2a9fc84822a05e89d50.jpg",
    "tomato_20220726052_slice_1_png.rf.c86501cbe4e3adf7b8d1b0910f191a17.jpg",
}
_EXPECTED_CATEGORIES = {1: "ripe", 2: "unripe", 3: "stem"}


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(file_descriptor, "w") as file:
            json.dump(data, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _ensure_image_link(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"Existing symlink points to the wrong image: {destination}")
        return
    if destination.exists():
        raise FileExistsError(f"Refusing to replace existing path: {destination}")
    destination.symlink_to(os.path.relpath(source, destination.parent))


def _prepare_split(source_root: Path, prepared_root: Path, split: str) -> tuple[int, int, int]:
    annotation_path = source_root / "annotations" / f"_annotations_{split}_fixed.json"
    image_root = source_root / "images" / split
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing annotation file: {annotation_path}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_root}")

    data = json.loads(annotation_path.read_text())
    categories = {category["id"]: category["name"] for category in data["categories"]}
    if categories != _EXPECTED_CATEGORIES:
        raise ValueError(f"Unexpected categories in {annotation_path}: {categories}")

    image_ids = [image["id"] for image in data["images"]]
    file_names = [image["file_name"] for image in data["images"]]
    if len(image_ids) != len(set(image_ids)) or len(file_names) != len(set(file_names)):
        raise ValueError(f"Duplicate image IDs or file names in {annotation_path}")

    missing_images = [image for image in data["images"] if not (image_root / image["file_name"]).is_file()]
    missing_names = {image["file_name"] for image in missing_images}
    allowed_missing = _EXPECTED_MISSING_TRAIN_IMAGES if split == "train" else set()
    unexpected_missing = missing_names - allowed_missing
    if unexpected_missing:
        raise FileNotFoundError(f"Unexpected missing {split} images: {sorted(unexpected_missing)}")

    removed_image_ids = {image["id"] for image in missing_images}
    cleaned_images = [image for image in data["images"] if image["id"] not in removed_image_ids]
    cleaned_annotations = [
        annotation for annotation in data["annotations"] if annotation["image_id"] not in removed_image_ids
    ]
    cleaned_image_ids = {image["id"] for image in cleaned_images}

    orphan_annotation_ids = [
        annotation["id"] for annotation in cleaned_annotations if annotation["image_id"] not in cleaned_image_ids
    ]
    if orphan_annotation_ids:
        raise ValueError(f"Annotations reference missing image IDs: {orphan_annotation_ids[:10]}")
    annotations_without_segmentation = [
        annotation["id"] for annotation in cleaned_annotations if not annotation.get("segmentation")
    ]
    if annotations_without_segmentation:
        raise ValueError(f"Annotations lack segmentation data: {annotations_without_segmentation[:10]}")

    destination_root = prepared_root / split
    destination_root.mkdir(parents=True, exist_ok=True)
    for image in cleaned_images:
        source_image = image_root / image["file_name"]
        _ensure_image_link(source_image, destination_root / image["file_name"])

    cleaned_data = {**data, "images": cleaned_images, "annotations": cleaned_annotations}
    _atomic_write_json(destination_root / "_annotations.coco.json", cleaned_data)
    return len(cleaned_images), len(cleaned_annotations), len(data["annotations"]) - len(cleaned_annotations)


def prepare_dataset(source_root: Path, prepared_root: Path) -> None:
    """Create RF-DETR's train/valid COCO layout without copying source images."""
    for split in ("train", "valid"):
        images, annotations, removed_annotations = _prepare_split(source_root, prepared_root, split)
        print(
            f"Prepared {split}: {images} images, {annotations} annotations, "
            f"removed {removed_annotations} annotations for missing images"
        )
    print(f"Prepared dataset: {prepared_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, default=_DEFAULT_SOURCE_DATASET)
    parser.add_argument("--prepared-dataset", type=Path, default=_DEFAULT_PREPARED_DATASET)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare and validate data, then exit without training",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--strategy", choices=("auto", "ddp"), default="ddp")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-encoder", type=float, default=1.5e-5)
    parser.add_argument("--lr-scheduler", choices=("step", "cosine"), default="cosine")
    parser.add_argument("--warmup-epochs", type=float, default=2.0)
    parser.add_argument("--lr-drop", type=int, default=40)
    parser.add_argument("--lr-min-factor", type=float, default=0.01)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--unfreeze-encoder", action="store_true")
    parser.add_argument(
        "--use-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use exponential moving average weights (disable with --no-use-ema)",
    )
    parser.add_argument("--multi-scale", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dataset = args.source_dataset.expanduser().resolve()
    prepared_dataset = args.prepared_dataset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    prepare_dataset(source_dataset, prepared_dataset)
    if args.prepare_only:
        return

    if args.resolution <= 0 or args.resolution % 16:
        raise ValueError("--resolution must be a positive multiple of LingBot's patch size (16)")
    if args.devices < 1:
        raise ValueError("--devices must be at least 1")
    if args.device == "cpu" and args.devices > 1:
        raise ValueError("Multi-device training requires a CUDA device")
    if args.warmup_epochs < 0:
        raise ValueError("--warmup-epochs must be non-negative")
    if args.lr_drop < 0:
        raise ValueError("--lr-drop must be non-negative")
    if not 0 <= args.lr_min_factor <= 1:
        raise ValueError("--lr-min-factor must be between 0 and 1")

    model = RFDETRLingBotSmallSeg(  # type: ignore[no-untyped-call]
        amp=not args.no_amp,
        freeze_encoder=not args.unfreeze_encoder,
        num_classes=len(_EXPECTED_CATEGORIES),
        resolution=args.resolution,
    )
    model.train(  # type: ignore[no-untyped-call]
        dataset_dir=prepared_dataset,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        num_workers=args.num_workers,
        device=args.device,
        devices=args.devices,
        strategy=args.strategy,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        lr_scheduler=args.lr_scheduler,
        warmup_epochs=args.warmup_epochs,
        lr_drop=args.lr_drop,
        lr_min_factor=args.lr_min_factor,
        checkpoint_interval=args.checkpoint_interval,
        resume=os.fspath(args.resume) if args.resume else None,
        use_ema=args.use_ema,
        multi_scale=args.multi_scale,
        expanded_scales=False,
        do_random_resize_via_padding=False,
        tensorboard=args.tensorboard,
        run_test=False,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
