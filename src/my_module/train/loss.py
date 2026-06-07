from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def gaussian_nll_from_cholesky(pred: Tensor, target: Tensor, eps: float = 1e-4) -> Tensor:
    if pred.shape[-1] != 9:
        raise ValueError(f"expected pred.shape[-1] == 9, got {pred.shape[-1]}")

    mean = pred[:, 0:3]
    l_00, l_11, l_22 = pred[:, 3:4], pred[:, 4:5], pred[:, 5:6]
    l_10, l_20, l_21 = pred[:, 6:7], pred[:, 7:8], pred[:, 8:9]

    batch = target.shape[0]
    chol = torch.zeros((batch, 3, 3), device=pred.device, dtype=pred.dtype)
    chol[:, 0, 0] = F.softplus(l_00.squeeze(-1)) + eps
    chol[:, 1, 0] = l_10.squeeze(-1)
    chol[:, 1, 1] = F.softplus(l_11.squeeze(-1)) + eps
    chol[:, 2, 0] = l_20.squeeze(-1)
    chol[:, 2, 1] = l_21.squeeze(-1)
    chol[:, 2, 2] = F.softplus(l_22.squeeze(-1)) + eps

    residual = target - mean
    y = torch.linalg.solve_triangular(
        chol, residual.unsqueeze(-1), upper=False
    ).squeeze(-1)

    quad = (y * y).sum(dim=1)
    logdet = torch.log(torch.diagonal(chol, dim1=1, dim2=2)).sum(dim=1)
    log_prob = -0.5 * (3 * math.log(2.0 * math.pi) + 2.0 * logdet + quad)
    return -log_prob.mean()


def gaussian_nll_1d(pred: Tensor, target: Tensor, eps: float = 1e-4) -> Tensor:
    if pred.shape[-1] != 2:
        raise ValueError(f"expected pred.shape[-1] == 2, got {pred.shape[-1]}")

    mean = torch.tanh(pred[:, 0])
    std = F.softplus(pred[:, 1]) + eps
    target = target.reshape(-1)
    return (
        0.5
        * (math.log(2.0 * math.pi) + 2.0 * torch.log(std) + ((target - mean) / std) ** 2)
    ).mean()


def compute_loss(
    pred: Tensor | tuple[Tensor, Tensor],
    target: Tensor,
    loss_function: str,
    arrival_target: Tensor | None,
    lambda_arrival: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    if isinstance(pred, tuple):
        hypo_pred, arrival_pred = pred
    else:
        hypo_pred, arrival_pred = pred, None

    if loss_function == "gaussian":
        source_loss = gaussian_nll_from_cholesky(hypo_pred, target)
    elif loss_function == "mse":
        if arrival_target is not None:
            raise ValueError("arrival_time training requires loss_function='gaussian'.")
        source_loss = F.mse_loss(hypo_pred, target)
    else:
        raise ValueError(f"Unknown loss_function: {loss_function}")

    losses = {"source": source_loss}
    total_loss = source_loss
    if arrival_target is not None:
        if arrival_pred is None:
            raise ValueError("Model did not return arrival-time prediction.")
        arrival_loss = gaussian_nll_1d(arrival_pred, arrival_target)
        total_loss = total_loss + lambda_arrival * arrival_loss
        losses["arrival"] = arrival_loss

    losses["total"] = total_loss
    return total_loss, losses
