# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the LingBot-Vision backbone adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rfdetr.models.backbone.lingbot import (
    LingBotBackbone,
    build_lingbot_vision_small,
    load_lingbot_backbone_state,
)
from rfdetr.utilities.tensors import NestedTensor


class _TinyEncoder(nn.Module):
    """Small patch encoder matching the LingBot structured-output contract."""

    patch_size = 16
    embed_dim = 24
    n_blocks = 2

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, self.embed_dim, kernel_size=16, stride=16)
        self.blocks = nn.ModuleList([nn.Linear(self.embed_dim, self.embed_dim) for _ in range(self.n_blocks)])
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        feature = self.patch_embed(images)
        tokens = feature.flatten(2).transpose(1, 2)
        for block in self.blocks:
            tokens = tokens + block(tokens)
        return {"x_norm_patchtokens": self.norm(tokens)}


def _nested_input(height: int, width: int, batch_size: int = 2) -> NestedTensor:
    images = torch.randn(batch_size, 3, height, width)
    masks = torch.zeros(batch_size, height, width, dtype=torch.bool)
    masks[0, height // 2 :, width // 2 :] = True
    return NestedTensor(images, masks)


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        lr_encoder=2e-5,
        lr_vit_layer_decay=0.8,
        lr_component_decay=0.7,
        weight_decay=1e-4,
    )


def test_build_lingbot_small_matches_released_architecture() -> None:
    """The local builder must reproduce the released Small backbone shape and configuration."""
    model = build_lingbot_vision_small(img_size=512)

    assert model.patch_size == 16
    assert model.embed_dim == 384
    assert model.n_blocks == 12
    assert model.num_heads == 6
    assert model.n_storage_tokens == 4
    assert model.rope_embed.dtype == torch.float32
    assert model.rope_embed.normalize_coords == "separate"
    assert model.rope_embed.rescale_coords == 2


def test_lingbot_backbone_outputs_p4_and_resized_mask() -> None:
    """A stride-16 token map should become one P4 NestedTensor with an aligned padding mask."""
    backbone = LingBotBackbone(
        encoder=_TinyEncoder(),
        out_feature_indexes=[11],
        out_channels=32,
        projector_scale=["P4"],
        layer_norm=True,
    )

    features, cross_attention = backbone(_nested_input(64, 96))

    assert cross_attention is None
    assert len(features) == 1
    assert features[0].tensors.shape == (2, 32, 4, 6)
    assert features[0].mask is not None
    assert features[0].mask.shape == (2, 4, 6)
    assert features[0].mask[0, -1, -1]


def test_lingbot_backbone_supports_non_square_inputs() -> None:
    """Spatial recovery must derive height and width separately rather than from sqrt(token_count)."""
    backbone = LingBotBackbone(
        encoder=_TinyEncoder(),
        out_channels=16,
        projector_scale=["P4"],
        layer_norm=True,
    )

    features, _ = backbone(_nested_input(32, 80, batch_size=1))

    assert features[0].tensors.shape[-2:] == (2, 5)


@pytest.mark.parametrize(
    ("height", "width"),
    [
        pytest.param(63, 64, id="height"),
        pytest.param(64, 79, id="width"),
    ],
)
def test_lingbot_backbone_rejects_non_divisible_inputs(height: int, width: int) -> None:
    """The adapter must reject dimensions that official PatchEmbed would silently truncate."""
    backbone = LingBotBackbone(
        encoder=_TinyEncoder(),
        out_channels=16,
        projector_scale=["P4"],
        layer_norm=True,
    )

    with pytest.raises(ValueError, match="divisible by patch_size=16"):
        backbone(_nested_input(height, width, batch_size=1))


def test_lingbot_backbone_export_contract() -> None:
    """Export forward should mirror the existing backbone's feature/mask/cross-attention tuple."""
    backbone = LingBotBackbone(
        encoder=_TinyEncoder(),
        out_channels=16,
        projector_scale=["P4"],
        layer_norm=True,
    )

    features, masks, cross_attention = backbone.forward_export(torch.randn(1, 3, 32, 48))

    assert len(features) == len(masks) == 1
    assert features[0].shape == (1, 16, 2, 3)
    assert masks[0].shape == (1, 2, 3)
    assert masks[0].dtype == torch.bool
    assert cross_attention is None


def test_freeze_encoder_keeps_projector_trainable() -> None:
    """Freezing LingBot must not freeze the randomly initialized detector projector."""
    backbone = LingBotBackbone(
        encoder=_TinyEncoder(),
        out_channels=16,
        projector_scale=["P4"],
        layer_norm=True,
        freeze_encoder=True,
    )

    assert all(not parameter.requires_grad for parameter in backbone.encoder.parameters())
    assert all(parameter.requires_grad for parameter in backbone.projector.parameters())


def test_trainable_encoder_receives_gradients() -> None:
    """The training adapter must not inherit the official inference loader's no-grad behavior."""
    backbone = LingBotBackbone(
        encoder=_TinyEncoder(),
        out_channels=16,
        projector_scale=["P4"],
        layer_norm=True,
    )

    features, _ = backbone(_nested_input(32, 32, batch_size=1))
    features[0].tensors.square().mean().backward()

    assert backbone.encoder.patch_embed.weight.grad is not None
    assert any(parameter.grad is not None for parameter in backbone.projector.parameters())


def test_lingbot_parameter_groups_cover_encoder_once_with_layer_decay() -> None:
    """Encoder parameters should be unique and deeper blocks should use larger learning rates."""
    backbone = LingBotBackbone(
        encoder=_TinyEncoder(),
        out_channels=16,
        projector_scale=["P4"],
        layer_norm=True,
    )

    groups = backbone.get_named_param_lr_pairs(_args(), prefix="backbone.0")
    encoder_parameters = {
        f"backbone.0.{name}": parameter
        for name, parameter in backbone.named_parameters()
        if name.startswith("encoder.")
    }

    assert set(groups) == set(encoder_parameters)
    assert len({id(group["params"]) for group in groups.values()}) == len(groups)
    assert groups["backbone.0.encoder.blocks.1.weight"]["lr"] > groups["backbone.0.encoder.blocks.0.weight"]["lr"]
    assert groups["backbone.0.encoder.norm.bias"]["weight_decay"] == 0.0


def test_load_lingbot_state_accepts_wrapped_prefixed_state(tmp_path) -> None:
    """The strict loader should accept official wrappers plus compiled/backbone prefixes."""
    source = _TinyEncoder()
    state = {f"_orig_mod.backbone.{key}": value.clone() for key, value in source.state_dict().items()}
    checkpoint = tmp_path / "wrapped.pt"
    torch.save({"teacher": state}, checkpoint)
    target = _TinyEncoder()

    loaded_keys = load_lingbot_backbone_state(target, checkpoint)

    assert loaded_keys == len(source.state_dict())
    for source_value, target_value in zip(source.state_dict().values(), target.state_dict().values()):
        assert torch.equal(source_value, target_value)


def test_load_lingbot_state_rejects_missing_keys(tmp_path) -> None:
    """A partially loaded backbone must fail rather than silently random-initialize layers."""
    source = _TinyEncoder()
    state = dict(source.state_dict())
    state.pop(next(iter(state)))
    checkpoint = tmp_path / "missing.pt"
    torch.save(state, checkpoint)

    with pytest.raises(RuntimeError, match="missing"):
        load_lingbot_backbone_state(_TinyEncoder(), checkpoint)


def test_load_lingbot_state_rejects_unexpected_keys(tmp_path) -> None:
    """Unexpected backbone tensors should fail strict provenance validation."""
    source = _TinyEncoder()
    state = dict(source.state_dict())
    state["not_a_real_parameter"] = torch.zeros(1)
    checkpoint = tmp_path / "unexpected.pt"
    torch.save(state, checkpoint)

    with pytest.raises(RuntimeError, match="unexpected"):
        load_lingbot_backbone_state(_TinyEncoder(), checkpoint)


def test_registered_lingbot_asset_hash_failure_is_fatal(tmp_path, monkeypatch) -> None:
    """The official pinned filename must never load after a failed SHA-256 validation."""
    checkpoint = tmp_path / "lingbot-vision-vit-small-127cbcec.pt"
    torch.save(_TinyEncoder().state_dict(), checkpoint)
    monkeypatch.setattr("rfdetr.models.backbone.lingbot.download_pretrain_weights", lambda *_args, **_kwargs: None)

    def reject_hash(*_args, **_kwargs):
        raise ValueError("SHA-256 hash validation failed")

    monkeypatch.setattr("rfdetr.models.backbone.lingbot.validate_pretrain_weights", reject_hash)

    with pytest.raises(ValueError, match="SHA-256 hash validation failed"):
        LingBotBackbone(
            encoder=_TinyEncoder(),
            backbone_weights=checkpoint,
            out_channels=16,
            projector_scale=["P4"],
            layer_norm=True,
        )


def test_gradient_checkpointing_is_rejected_explicitly() -> None:
    """Unsupported checkpointing must not be accepted and ignored."""
    with pytest.raises(NotImplementedError, match="gradient checkpointing"):
        LingBotBackbone(
            encoder=_TinyEncoder(),
            out_channels=16,
            projector_scale=["P4"],
            layer_norm=True,
            gradient_checkpointing=True,
        )
