from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn


def get_base_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    if hasattr(model, "module"):
        model = model.module
    return model


def get_state_dict_for_saving(model: nn.Module) -> dict[str, Tensor]:
    return get_base_model(model).state_dict()


def normalized_state_dict(state: dict[str, Tensor]) -> dict[str, Tensor]:
    cleaned = {}
    for key, value in state.items():
        for prefix in ("module.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    return cleaned


def load_pretrained(
    model: nn.Module, weight_path: str | Path, device: torch.device, strict: bool
) -> None:
    state = torch.load(weight_path, map_location=device, weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"{weight_path} did not contain a state_dict.")
    model.load_state_dict(normalized_state_dict(state), strict=strict)


def configure_trainable_parameters(model: nn.Module, mode: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    if mode == "all":
        modules: list[nn.Module] = [model]
    elif mode == "head":
        modules = [model.hypo_mlp_heads]
        if getattr(model, "arrival_time", False):
            modules.append(model.arrival_mlp_heads)
    elif mode == "head-and-tokens":
        modules = [model.hypo_mlp_heads]
        if getattr(model, "arrival_time", False):
            modules.append(model.arrival_mlp_heads)
        for name in ("hypo_token", "arrival_token"):
            token = getattr(model, name, None)
            if token is not None:
                token.requires_grad = True
        return
    elif mode == "no-patch-embedding":
        modules = [model.transformer, model.positional_encoding, model.hypo_mlp_heads]
        if getattr(model, "arrival_time", False):
            modules.append(model.arrival_mlp_heads)
    else:
        raise ValueError(f"Unknown freeze mode: {mode}")

    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
