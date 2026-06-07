from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from pyproj import Geod
from torch.utils.data import Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from my_module import create_logger
from my_module.config import ExperimentConfig
from my_module.models import build_vit_locator
from my_module.prediction import convert_relative_to_geo, flatten_prediction_row
from my_module.train import build_dataset, build_loader, load_pretrained, set_random_seed

geod = Geod(ellps="WGS84")


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def choose_device(device_name: str, cfg_device: str) -> torch.device:
    if device_name == "config":
        return torch.device(cfg_device)
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned TremorViT locator on a CSV/HDF5 split."
    )
    parser.add_argument(
        "--exp", default="vit_locator_v11", help="config/experiments/{exp}.json"
    )
    parser.add_argument("--dataset_dir", default=None, help="Override cfg.data.dataset_dir.")
    parser.add_argument("--target", default="test", help="Split name to evaluate.")
    parser.add_argument(
        "--weight",
        default=None,
        help="Weight path. Default: cfg.output.save_dir/best_weight.pth.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Default: reports/{experiment_name}/evaluate.",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Evaluate only the first N samples. Useful for quick smoke tests.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "config"],
        help="Evaluation device.",
    )
    parser.add_argument(
        "--jitter_mode",
        action="store_true",
        help="Use random locator-window jitter during evaluation.",
    )
    return parser.parse_args()


def apply_overrides(cfg: ExperimentConfig, args: argparse.Namespace) -> None:
    if args.dataset_dir is not None:
        cfg.data.dataset_dir = str(resolve_path(args.dataset_dir))
    else:
        cfg.data.dataset_dir = str(resolve_path(cfg.data.dataset_dir))

    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size

    cfg.output.save_dir = str(resolve_path(cfg.output.save_dir))


def unpack_eval_batch(
    batch: tuple[torch.Tensor, ...], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    if len(batch) == 5:
        waveform, _target, arrival, sta_lat, sta_lon = batch
    elif len(batch) == 4:
        waveform, _target, sta_lat, sta_lon = batch
        arrival = None
    else:
        raise ValueError(f"Evaluation batch must have 4 or 5 tensors, got {len(batch)}")

    return (
        waveform.to(device, non_blocking=True),
        arrival.to(device, non_blocking=True) if arrival is not None else None,
        sta_lat,
        sta_lon,
    )


def horizontal_error_km(row: pd.Series) -> float:
    _, _, dist_m = geod.inv(
        row["true_lon"], row["true_lat"], row["pred_lon"], row["pred_lat"]
    )
    return float(dist_m / 1000.0)


def build_summary(df: pd.DataFrame) -> dict[str, Any]:
    horizontal = df["horizontal_error_km"].to_numpy(float)
    depth_abs = df["depth_abs_error_km"].to_numpy(float)
    distance_3d = df["distance_3d_error_km"].to_numpy(float)
    return {
        "n": int(len(df)),
        "horizontal_mae_km": float(np.mean(horizontal)),
        "horizontal_median_km": float(np.median(horizontal)),
        "horizontal_p90_km": float(np.percentile(horizontal, 90)),
        "depth_mae_km": float(np.mean(depth_abs)),
        "distance_3d_mae_km": float(np.mean(distance_3d)),
        "distance_3d_median_km": float(np.median(distance_3d)),
        "distance_3d_p90_km": float(np.percentile(distance_3d, 90)),
    }


def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    target_df: pd.DataFrame,
    cfg: ExperimentConfig,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows = []
    offset = 0

    with torch.inference_mode():
        for batch in tqdm(loader, total=len(loader), desc="Evaluating"):
            waveform, _arrival, sta_lat, sta_lon = unpack_eval_batch(batch, device)
            pred = model(waveform)
            pred_hypo = pred[0] if isinstance(pred, tuple) else pred
            pred_np = pred_hypo.cpu().numpy()
            sta_lat_np = sta_lat.numpy()
            sta_lon_np = sta_lon.numpy()

            for b in range(pred_np.shape[0]):
                i = offset + b
                if i >= len(target_df):
                    break

                meta = target_df.iloc[i]
                true_lat = float(meta["lat"])
                true_lon = float(meta["lon"])
                true_dep = float(meta["dep"])

                geo_pred, mean_km, sigma_km2 = convert_relative_to_geo(
                    prediction=pred_np[b],
                    station_loc=(float(sta_lat_np[b]), float(sta_lon_np[b])),
                    scale_km=cfg.data.scale_km,
                )
                result = flatten_prediction_row(
                    true_lat=true_lat,
                    true_lon=true_lon,
                    true_dep=true_dep,
                    pred_hypo=geo_pred,
                    mean_km=mean_km,
                    Sigma_km2=sigma_km2,
                ).__dict__

                base_cols = {
                    col: meta[col]
                    for col in [
                        "origin_time",
                        "arrival_time",
                        "start_time_for_trainlocator",
                        "station",
                        "index",
                    ]
                    if col in meta.index
                }
                rows.append({**base_cols, **result})

            offset += pred_np.shape[0]

    df = pd.DataFrame(rows)
    df["horizontal_error_km"] = df.apply(horizontal_error_km, axis=1)
    df["depth_abs_error_km"] = (df["pred_dep"] - df["true_dep"]).abs()
    df["distance_3d_error_km"] = np.sqrt(
        df["horizontal_error_km"] ** 2 + df["depth_abs_error_km"] ** 2
    )
    return df


def main() -> None:
    args = parse_args()
    cfg_path = resolve_path("config/experiments") / f"{args.exp}.json"
    cfg = ExperimentConfig.from_file(cfg_path)
    apply_overrides(cfg, args)

    device = choose_device(args.device, cfg.train.device)
    cfg.train.device = str(device)
    set_random_seed(cfg.train.seed)

    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir is not None
        else ROOT / "reports" / cfg.experiment_name / "evaluate"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(output_dir / f"evaluate_{args.exp}_{args.target}.log")

    weight_path = (
        resolve_path(args.weight)
        if args.weight is not None
        else Path(cfg.output.save_dir) / "best_weight.pth"
    )
    logger.info(f"Experiment config: {cfg_path}")
    logger.info(f"Dataset directory: {cfg.data.dataset_dir}")
    logger.info(f"Weight: {weight_path}")
    logger.info(f"Device: {device}")

    model = build_vit_locator(cfg)
    load_pretrained(model, weight_path, device, strict=True)

    target_csv = Path(cfg.data.dataset_dir) / f"{args.target}.csv"
    target_df = pd.read_csv(target_csv)
    dataset = build_dataset(
        cfg,
        args.target,
        training=args.jitter_mode,
        include_station=True,
    )
    if args.max_samples is not None:
        n_samples = min(args.max_samples, len(target_df), len(dataset))
        target_df = target_df.iloc[:n_samples].reset_index(drop=True)
        dataset = Subset(dataset, range(n_samples))

    loader = build_loader(
        dataset,
        cfg.train.batch_size,
        cfg.train.seed,
        shuffle=False,
        num_workers=args.num_workers,
    )

    df = evaluate(model, loader, target_df, cfg, device)
    suffix = f"{args.exp}_{args.target}" + ("_jitter" if args.jitter_mode else "")
    pred_path = output_dir / f"locator_pred_{suffix}.csv"
    summary_path = output_dir / f"locator_metrics_{suffix}.json"

    df.to_csv(pred_path, index=False)
    summary = build_summary(df)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info(f"Saved predictions to: {pred_path}")
    logger.info(f"Saved metrics to: {summary_path}")
    logger.info(f"Summary: {summary}")


if __name__ == "__main__":
    main()
