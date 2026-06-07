from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from my_module import create_logger
from my_module.config import ExperimentConfig
from my_module.models import build_vit_locator
from my_module.train import (
    build_dataset,
    build_loader,
    configure_trainable_parameters,
    finetune_loop,
    load_pretrained,
    set_random_seed,
)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the TremorViT single-station locator."
    )
    parser.add_argument(
        "--exp", default="vit_locator_v11", help="config/experiments/{exp}.json"
    )
    parser.add_argument(
        "--pretrained_weight",
        default="models/version1.0/vit_locator_v11/best_weight.pth",
        help="Path to the source locator weight.",
    )
    parser.add_argument("--dataset_dir", default=None, help="Override cfg.data.dataset_dir.")
    parser.add_argument("--train_name", default=None, help="Override training split name.")
    parser.add_argument("--val_name", default=None, help="Override validation split name.")
    parser.add_argument("--save_dir", default=None, help="Override cfg.output.save_dir.")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device. auto uses CUDA when available.",
    )
    parser.add_argument(
        "--freeze",
        default="all",
        choices=["all", "head", "head-and-tokens", "no-patch-embedding"],
        help="Which parameters to fine-tune.",
    )
    parser.add_argument("--strict", action="store_true", help="Use strict weight loading.")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile after loading.")
    return parser.parse_args()


def choose_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def apply_overrides(
    cfg: ExperimentConfig, args: argparse.Namespace, device: torch.device
) -> None:
    if args.dataset_dir is not None:
        cfg.data.dataset_dir = str(resolve_path(args.dataset_dir))
    else:
        cfg.data.dataset_dir = str(resolve_path(cfg.data.dataset_dir))

    if args.train_name is not None:
        cfg.data.train_name = args.train_name
    if args.val_name is not None:
        cfg.data.val_name = args.val_name

    if args.save_dir is not None:
        cfg.output.save_dir = str(resolve_path(args.save_dir))
    else:
        cfg.output.save_dir = str(resolve_path(cfg.output.save_dir))

    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.train.num_epochs = args.epochs
    if args.learning_rate is not None:
        cfg.train.learning_rate = args.learning_rate
    if args.patience is not None:
        cfg.train.early_stopping_patience = args.patience

    cfg.train.device = str(device)


def main() -> None:
    args = parse_args()
    cfg_path = resolve_path("config/experiments") / f"{args.exp}.json"
    cfg = ExperimentConfig.from_file(cfg_path)
    device = choose_device(args.device)

    apply_overrides(cfg, args, device)
    set_random_seed(cfg.train.seed)

    save_dir = Path(cfg.output.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(save_dir / "finetune.log")
    logger.info(f"Experiment config: {cfg_path}")
    logger.info(f"Dataset directory: {cfg.data.dataset_dir}")
    logger.info(f"Save directory: {save_dir}")
    logger.info(f"Device: {device}")

    model = build_vit_locator(cfg)
    load_pretrained(model, resolve_path(args.pretrained_weight), device, strict=args.strict)
    configure_trainable_parameters(model, args.freeze)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable:,}/{total:,} ({args.freeze})")

    if args.compile:
        model = torch.compile(model)

    train_dataset = build_dataset(cfg, cfg.data.train_name, training=True)
    val_dataset = build_dataset(cfg, cfg.data.val_name, training=False)
    train_loader = build_loader(
        train_dataset,
        cfg.train.batch_size,
        cfg.train.seed,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = build_loader(
        val_dataset,
        cfg.train.batch_size,
        cfg.train.seed,
        shuffle=False,
        num_workers=args.num_workers,
    )

    finetune_loop(model, train_loader, val_loader, cfg, device, logger)


if __name__ == "__main__":
    main()
