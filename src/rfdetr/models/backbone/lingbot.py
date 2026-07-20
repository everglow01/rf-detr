# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""LingBot-Vision backbone adapter for RF-DETR."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from rfdetr.assets.model_weights import (
    download_pretrain_weights,
    get_model_cache_dir,
    validate_pretrain_weights,
)
from rfdetr.models.backbone._lingbot_vision.vit import LingBotVisionTransformer, vit_small
from rfdetr.models.backbone.base import BackboneBase
from rfdetr.models.backbone.projector import MultiScaleProjector
from rfdetr.utilities.logger import get_logger
from rfdetr.utilities.tensors import NestedTensor

logger = get_logger()

_LINGBOT_SOURCE_REVISION = "151e46321bae4399f8568829f190c7bdec216b49"
_LINGBOT_WEIGHT_REVISION = "127cbcec380de0bcd55bdc1b1fad3819850a6514"
_LEVEL_TO_SCALE = {"P3": 2.0, "P4": 1.0, "P5": 0.5}
_STATE_WRAPPER_KEYS = ("teacher", "model_state", "state_dict", "model", "backbone")


def build_lingbot_vision_small(img_size: int = 512) -> LingBotVisionTransformer:
    """Build the released LingBot-Vision Small architecture with trainable parameters.

    Args:
        img_size: Nominal image size stored by the patch embedding module.

    Returns:
        Freshly initialized LingBot-Vision Small backbone.
    """
    model = vit_small(
        img_size=img_size,
        patch_size=16,
        layerscale_init=1e-5,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        n_storage_tokens=4,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        mask_k_bias=True,
        untie_cls_and_patch_norms=False,
        untie_global_and_local_cls_norm=False,
        pos_embed_rope_base=100.0,
        pos_embed_rope_min_period=None,
        pos_embed_rope_max_period=None,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_shift_coords=None,
        pos_embed_rope_jitter_coords=None,
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        drop_path_rate=0.0,
    )
    model.init_weights()
    return model


def _unwrap_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    """Extract a tensor state dict from the wrapper formats released by LingBot.

    Args:
        checkpoint: Object loaded from a checkpoint file.

    Returns:
        Mapping from parameter names to tensors.

    Raises:
        ValueError: If no valid tensor state dict can be found.
    """
    state = checkpoint
    if isinstance(state, Mapping):
        for key in _STATE_WRAPPER_KEYS:
            value = state.get(key)
            if isinstance(value, Mapping):
                state = value
                break
    if not isinstance(state, Mapping) or not state:
        raise ValueError("could not locate LingBot backbone weights in checkpoint")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
        raise ValueError("LingBot checkpoint state dict must contain only string keys and tensors")
    return state


def _normalize_state_key(key: str) -> str:
    """Remove only the compiled-model and backbone prefixes supported upstream.

    Args:
        key: Raw checkpoint key.

    Returns:
        Key matching the vendored LingBot module.
    """
    normalized = key.replace("_orig_mod.", "")
    if normalized.startswith("backbone."):
        normalized = normalized[len("backbone.") :]
    return normalized


def load_lingbot_backbone_state(encoder: nn.Module, checkpoint_path: str | os.PathLike[str]) -> int:
    """Strictly load a LingBot backbone checkpoint without freezing the encoder.

    Args:
        encoder: LingBot encoder receiving the checkpoint.
        checkpoint_path: Local checkpoint path.

    Returns:
        Number of tensors loaded.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        RuntimeError: If keys do not exactly match the encoder.
        ValueError: If the checkpoint is not a supported tensor state dict.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    raw_state = _unwrap_state_dict(checkpoint)
    state = {_normalize_state_key(key): value for key, value in raw_state.items()}
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"LingBot backbone checkpoint mismatch: missing={list(missing)} unexpected={list(unexpected)}"
        )
    logger.info(
        "Loaded %d LingBot backbone tensors (source=%s, weights=%s)",
        len(state),
        _LINGBOT_SOURCE_REVISION,
        _LINGBOT_WEIGHT_REVISION,
    )
    return len(state)


def _vit_layer_id(name: str, num_layers: int) -> int:
    """Return the layer-wise decay index for one LingBot encoder parameter.

    Args:
        name: Parameter name relative to ``LingBotBackbone``.
        num_layers: Number of transformer blocks.

    Returns:
        Layer index in ``[0, num_layers + 1]``.
    """
    if name.startswith(("encoder.patch_embed.", "encoder.cls_token", "encoder.storage_tokens", "encoder.mask_token")):
        return 0
    if name.startswith("encoder.blocks."):
        block = int(name.split(".")[2])
        return block + 1
    return num_layers + 1


def _weight_decay(name: str, default: float) -> float:
    """Apply zero weight decay to affine and normalization-style parameters.

    Args:
        name: Parameter name.
        default: Default weight decay.

    Returns:
        Per-parameter weight decay.
    """
    if any(token in name for token in ("bias", "norm", ".gamma", "cls_token", "storage_tokens", "mask_token")):
        return 0.0
    return default


class LingBotBackbone(BackboneBase):
    """Adapt LingBot patch tokens to RF-DETR's spatial backbone contract."""

    def __init__(
        self,
        *,
        encoder: nn.Module | None = None,
        backbone_weights: str | os.PathLike[str] | None = None,
        out_channels: int = 256,
        out_feature_indexes: list[int] | None = None,
        projector_scale: list[str] | None = None,
        freeze_encoder: bool = False,
        layer_norm: bool = True,
        rms_norm: bool = False,
        target_shape: tuple[int, int] = (512, 512),
        gradient_checkpointing: bool = False,
    ) -> None:
        """Initialize the LingBot adapter.

        Args:
            encoder: Optional encoder override used by focused tests.
            backbone_weights: Raw LingBot backbone checkpoint.
            out_channels: Projected RF-DETR feature channels.
            out_feature_indexes: LingBot block indexes selected by the baseline.
            projector_scale: Requested feature levels. The migration baseline supports P4 only.
            freeze_encoder: Freeze only LingBot encoder parameters.
            layer_norm: Use channel-wise LayerNorm in the projector.
            rms_norm: Forwarded to the existing projector.
            target_shape: Nominal input shape used when constructing the encoder.
            gradient_checkpointing: Unsupported in the first migration milestone.

        Raises:
            NotImplementedError: If gradient checkpointing is requested.
            ValueError: If a non-P4 projector or incompatible encoder is requested.
        """
        super().__init__()
        if gradient_checkpointing:
            raise NotImplementedError("LingBot gradient checkpointing is not supported in the migration baseline")
        projector_scale = projector_scale or ["P4"]
        if projector_scale != ["P4"]:
            raise ValueError(f"LingBot migration baseline requires projector_scale=['P4'], got {projector_scale!r}")
        out_feature_indexes = out_feature_indexes or [11]
        if out_feature_indexes != [11]:
            raise ValueError(
                f"LingBot migration baseline consumes only final block index 11, got {out_feature_indexes!r}"
            )
        self.encoder = encoder or build_lingbot_vision_small(img_size=max(target_shape))
        raw_patch_size = self.encoder.patch_size
        if isinstance(raw_patch_size, tuple):
            if len(raw_patch_size) != 2 or raw_patch_size[0] != raw_patch_size[1]:
                raise ValueError(f"LingBot requires a square patch size, got {raw_patch_size!r}")
            raw_patch_size = raw_patch_size[0]
        self.patch_size = int(raw_patch_size)
        embed_dim = int(self.encoder.embed_dim)
        if backbone_weights is not None:
            weights_path = os.fspath(backbone_weights)
            if not os.path.dirname(weights_path):
                os.makedirs(get_model_cache_dir(), exist_ok=True)
                weights_path = os.path.join(get_model_cache_dir(), weights_path)
            else:
                os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            model_name = os.path.basename(weights_path)
            is_registered_asset = model_name == "lingbot-vision-vit-small-127cbcec.pt"
            if os.path.isfile(weights_path):
                if is_registered_asset:
                    download_pretrain_weights(weights_path)
                    validate_pretrain_weights(weights_path, strict=True)
            else:
                download_pretrain_weights(weights_path)
                if not os.path.isfile(weights_path):
                    raise FileNotFoundError(weights_path)
                if is_registered_asset:
                    validate_pretrain_weights(weights_path, strict=True)
            load_lingbot_backbone_state(self.encoder, weights_path)
        if freeze_encoder:
            self.encoder.requires_grad_(False)
        self.projector_scale = projector_scale
        self.projector = MultiScaleProjector(
            in_channels=[embed_dim],
            out_channels=out_channels,
            scale_factors=[_LEVEL_TO_SCALE[level] for level in projector_scale],
            layer_norm=layer_norm,
            rms_norm=rms_norm,
        )
        self.cross_attn_projector = None
        self._export = False

    def _extract_patch_map(self, images: torch.Tensor) -> torch.Tensor:
        """Extract and spatially restore normalized LingBot patch tokens.

        Args:
            images: Batched normalized images.

        Returns:
            Patch feature map in NCHW format.

        Raises:
            ValueError: If image dimensions are not divisible by patch size or token count is inconsistent.
        """
        height, width = images.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"LingBot input height and width must be divisible by patch_size={self.patch_size}, "
                f"got {(height, width)}"
            )
        output = self.encoder.forward_features(images)
        tokens = output["x_norm_patchtokens"]
        grid_height = height // self.patch_size
        grid_width = width // self.patch_size
        expected_tokens = grid_height * grid_width
        if tokens.shape[1] != expected_tokens:
            raise ValueError(f"LingBot returned {tokens.shape[1]} patch tokens, expected {expected_tokens}")
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], grid_height, grid_width)

    def _project(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Project one LingBot patch map to RF-DETR feature levels.

        Args:
            images: Batched normalized images.

        Returns:
            Projected spatial feature list.
        """
        return self.projector([self._extract_patch_map(images)])

    def forward(self, tensor_list: NestedTensor) -> tuple[list[NestedTensor], None]:
        """Return projected features and resized padding masks.

        Args:
            tensor_list: Batched images plus a padding mask.

        Returns:
            RF-DETR feature stream and no separate cross-attention stream.
        """
        features = self._project(tensor_list.tensors)
        if tensor_list.mask is None:
            raise ValueError("LingBotBackbone requires a padding mask")
        outputs = []
        for feature in features:
            mask = F.interpolate(tensor_list.mask[None].float(), size=feature.shape[-2:]).to(torch.bool)[0]
            outputs.append(NestedTensor(feature, mask))
        return outputs, None

    def forward_export(self, tensors: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor], None]:
        """Return tensor-only features and all-valid masks for export paths.

        Args:
            tensors: Batched normalized images.

        Returns:
            Features, boolean masks, and no separate cross-attention stream.
        """
        features = self._project(tensors)
        masks = [
            torch.zeros((feature.shape[0], *feature.shape[-2:]), dtype=torch.bool, device=feature.device)
            for feature in features
        ]
        return features, masks, None

    def export(self) -> None:
        """Switch the adapter to the tensor-only export contract."""
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export  # type: ignore[method-assign]

    def get_named_param_lr_pairs(self, args: Any, prefix: str = "backbone.0") -> dict[str, dict[str, Any]]:
        """Build layer-wise learning-rate groups for trainable LingBot encoder parameters.

        Args:
            args: Training namespace containing encoder LR and decay values.
            prefix: Full model prefix used by RF-DETR parameter grouping.

        Returns:
            Mapping from full parameter name to optimizer group settings.
        """
        num_layers = int(getattr(self.encoder, "n_blocks", len(getattr(self.encoder, "blocks", []))))
        groups: dict[str, dict[str, Any]] = {}
        for name, parameter in self.named_parameters():
            if not name.startswith("encoder.") or not parameter.requires_grad:
                continue
            layer_id = _vit_layer_id(name, num_layers)
            lr_scale = args.lr_vit_layer_decay ** (num_layers + 1 - layer_id)
            full_name = f"{prefix}.{name}"
            groups[full_name] = {
                "params": parameter,
                "lr": args.lr_encoder * lr_scale * args.lr_component_decay**2,
                "weight_decay": _weight_decay(name, args.weight_decay),
            }
        return groups
