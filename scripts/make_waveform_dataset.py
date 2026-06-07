from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from my_module import create_logger, get_station_catalog
from my_module.config import ProjectConfig
from my_module.sac import SacHandler
from my_module.train.dataset_builder import (
    build_station_catalog_dict,
    process_dataset_split,
)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create TremorViT CSV/HDF5 waveform datasets for locator training."
    )
    parser.add_argument(
        "--project_config",
        default="config/project_config.json",
        help="Path to project config JSON.",
    )
    parser.add_argument(
        "--catalog_dir",
        default=None,
        help=(
            "Directory containing tremor_catalog_{split}.csv. "
            "Default: data/{project_name}/catalog"
        ),
    )
    parser.add_argument(
        "--dataset_dir",
        default=None,
        help="Output dataset directory. Default: dataset_ssd/{project_name}.",
    )
    parser.add_argument(
        "--station_file",
        default=None,
        help="Station catalog file. Default: data/{project_name}/station/hinet_used.txt.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Dataset splits to create.",
    )
    parser.add_argument(
        "--input_pattern",
        default="tremor_catalog_{split}.csv",
        help="Input catalog filename pattern inside --catalog_dir.",
    )
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--win_sec", type=float, default=60.0)
    parser.add_argument("--jitter_sec", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_cfg = ProjectConfig.from_file(resolve_path(args.project_config))
    project_name = project_cfg.base.project_name

    catalog_dir = (
        resolve_path(args.catalog_dir)
        if args.catalog_dir is not None
        else ROOT / "data" / project_name / "catalog"
    )
    dataset_dir = (
        resolve_path(args.dataset_dir)
        if args.dataset_dir is not None
        else ROOT / "dataset_ssd" / project_name
    )
    station_file = (
        resolve_path(args.station_file)
        if args.station_file is not None
        else ROOT / "data" / project_name / "station" / "hinet_used.txt"
    )

    log_dir = ROOT / "logs" / project_name
    logger = create_logger(log_dir / "make_waveform_dataset.log")

    win_sec = float(args.win_sec)
    jitter_sec = float(args.jitter_sec)
    fs = int(project_cfg.waveform.sampling_rate)
    long_sec = win_sec + 2.0 * jitter_sec
    expected_len = int(round(long_sec * fs))

    logger.info(f"Catalog directory: {catalog_dir}")
    logger.info(f"Dataset directory: {dataset_dir}")
    logger.info(f"Station file: {station_file}")
    logger.info(f"Window: win_sec={win_sec}, jitter_sec={jitter_sec}, fs={fs}")

    station_df = get_station_catalog(station_file)
    station_catalog = build_station_catalog_dict(station_df)

    sac_handler = SacHandler(
        duration_seconds=float(long_sec - (1 / fs)),
        year_to_path=project_cfg.sac.year_to_path,
        component_channels=project_cfg.sac.component_channels,
        logger=logger,
    )

    dataset_dir.mkdir(parents=True, exist_ok=True)

    for split in tqdm(args.splits, desc="splits"):
        csv_input_path = catalog_dir / args.input_pattern.format(split=split)
        hdf_output_path = dataset_dir / "hdf" / f"{split}.h5"
        csv_output_path = dataset_dir / f"{split}.csv"

        if not csv_input_path.exists():
            logger.warning(f"Skip {split}: input catalog not found: {csv_input_path}")
            continue

        logger.info(f"Start processing {split}: {csv_input_path}")
        process_dataset_split(
            csv_input_path=csv_input_path,
            hdf_output_path=hdf_output_path,
            csv_output_path=csv_output_path,
            sac_handler=sac_handler,
            station_catalog=station_catalog,
            expected_len=expected_len,
            logger=logger,
            chunksize=args.chunksize,
            max_workers=args.max_workers,
        )


if __name__ == "__main__":
    main()
