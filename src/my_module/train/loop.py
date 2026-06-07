from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import ExperimentConfig
from .loss import compute_loss
from .state import get_state_dict_for_saving


def unpack_batch(
    batch: tuple[Tensor, ...], device: torch.device
) -> tuple[Tensor, Tensor, Tensor | None]:
    if len(batch) == 3:
        waveform, target, arrival = batch
        arrival = arrival.to(device, non_blocking=True)
    else:
        waveform, target = batch
        arrival = None

    return (
        waveform.to(device, non_blocking=True),
        target.to(device, non_blocking=True),
        arrival,
    )


def mean_losses(batch_losses: dict[str, list[float]]) -> dict[str, float]:
    return {key: float(np.mean(values)) for key, values in batch_losses.items()}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    cfg: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    batch_losses: dict[str, list[float]] = {}

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in tqdm(loader, total=len(loader), leave=False):
            waveform, target, arrival = unpack_batch(batch, device)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            pred = model(waveform)
            loss, losses = compute_loss(
                pred,
                target,
                loss_function=cfg.train.loss_function,
                arrival_target=arrival,
                lambda_arrival=cfg.train.lambda_arrival,
            )

            if optimizer is not None:
                loss.backward()
                optimizer.step()

            for key, value in losses.items():
                batch_losses.setdefault(key, []).append(float(value.detach().cpu()))

    return mean_losses(batch_losses)


def save_history(history: list[dict[str, float]], save_dir: str | Path) -> None:
    if not history:
        return

    save_dir = Path(save_dir)
    keys = list(history[0].keys())
    with (save_dir / "finetune_history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def finetune_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: ExperimentConfig,
    device: torch.device,
    logger: logging.Logger,
) -> None:
    save_dir = Path(cfg.output.save_dir)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.train.learning_rate,
        weight_decay=1e-4,
    )

    best_val = float("inf")
    patience_counter = 0
    history: list[dict[str, float]] = []

    for epoch in range(cfg.train.num_epochs):
        train_loss = run_epoch(model, train_loader, optimizer, cfg, device)
        val_loss = run_epoch(model, val_loader, None, cfg, device)

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_loss.items()},
            **{f"val_{k}": v for k, v in val_loss.items()},
        }
        history.append(row)

        logger.info(
            f"Epoch {epoch}: train_total={train_loss['total']:.4f}, "
            f"val_total={val_loss['total']:.4f}, best_val={best_val:.4f}"
        )

        if val_loss["total"] < best_val:
            best_val = val_loss["total"]
            patience_counter = 0
            torch.save(get_state_dict_for_saving(model), save_dir / "best_weight.pth")
            logger.info(f"Saved new best model to {save_dir / 'best_weight.pth'}")
        else:
            patience_counter += 1
            logger.info(
                f"No improvement. Patience {patience_counter}/"
                f"{cfg.train.early_stopping_patience}"
            )
            if patience_counter >= cfg.train.early_stopping_patience:
                logger.info("Early stopping triggered.")
                break

        if not cfg.output.save_best_only:
            torch.save(get_state_dict_for_saving(model), save_dir / f"weight_{epoch}.pth")

    save_history(history, save_dir)
    logger.info(f"Fine-tuning finished. Best validation loss: {best_val:.4f}")
