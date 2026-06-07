from __future__ import annotations

import csv
import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from pyproj import Geod
from tqdm import tqdm

from ..sac import SacHandler

geod = Geod(ellps="WGS84")


def parse_answer(
    event_lat: Any, event_lon: Any, event_depth: Any, station_lat: Any, station_lon: Any
) -> tuple[float, float, float]:
    az12, _, dist_m = geod.inv(station_lon, station_lat, event_lon, event_lat)
    az = np.deg2rad(az12)
    east_km = (dist_m * np.sin(az)) / 1000.0
    north_km = (dist_m * np.cos(az)) / 1000.0
    return float(east_km), float(north_km), float(event_depth)


def validate_catalog_columns(csv_path: str | Path) -> None:
    required = {"start_time_for_trainlocator", "lat", "lon", "dep", "station"}
    columns = set(pd.read_csv(csv_path, nrows=0).columns)
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")


def process_record(
    record: pd.Series,
    sac_handler: SacHandler,
    station_catalog: dict[str, tuple[float, float]],
    expected_len: int,
    logger: logging.Logger | None = None,
) -> dict[str, Any] | None:
    start_time, event_lat, event_lon, event_depth, station_code = record[
        ["start_time_for_trainlocator", "lat", "lon", "dep", "station"]
    ]

    try:
        station_lat, station_lon = station_catalog[str(station_code)]
    except KeyError:
        if logger is not None:
            logger.warning(f"Station {station_code} not found in station catalog.")
        return None

    sac_traces = sac_handler.get_sac_traces(str(station_code), str(start_time))
    if sac_traces is None:
        if logger is not None:
            logger.info(f"No SAC traces: {start_time} {station_code}")
        return None

    waveforms = []
    for component_code in ["EW", "NS", "UD"]:
        trace = sac_traces.get(component_code)
        if trace is None:
            if logger is not None:
                logger.info(
                    f"Missing component {component_code}: {start_time} {station_code}"
                )
            return None

        data = trace.data
        if len(data) != expected_len:
            if logger is not None:
                logger.info(
                    f"Length mismatch: {start_time} {station_code} {component_code} "
                    f"len={len(data)} expected={expected_len}"
                )
            return None

        waveforms.append(data)

    waveform = np.asarray(waveforms, dtype="float32")
    if np.any(np.var(waveform, axis=1, keepdims=True) == 0):
        if logger is not None:
            logger.info(f"Zero variance detected: {start_time} {station_code}.")
        return None

    east_km, north_km, depth_km = parse_answer(
        event_lat=float(event_lat),
        event_lon=float(event_lon),
        event_depth=float(event_depth),
        station_lat=float(station_lat),
        station_lon=float(station_lon),
    )

    return {
        "waveform": waveform,
        "east_km": east_km,
        "north_km": north_km,
        "depth_km": depth_km,
        "sta_lat": float(station_lat),
        "sta_lon": float(station_lon),
        "meta": record.to_dict(),
    }


def append_batch(
    h5_datasets: dict[str, h5py.Dataset],
    writer: csv.DictWriter,
    batch_results: list[dict[str, Any]],
    record_index: int,
) -> int:
    n_new = len(batch_results)
    new_end = record_index + n_new

    for dataset in h5_datasets.values():
        dataset.resize((new_end, *dataset.shape[1:]))

    h5_datasets["waveforms"][record_index:new_end] = np.stack(
        [r["waveform"] for r in batch_results], axis=0
    )
    for key in ["east_km", "north_km", "depth_km", "sta_lat", "sta_lon"]:
        h5_datasets[key][record_index:new_end] = np.asarray(
            [r[key] for r in batch_results], dtype="float32"
        )

    for i, result in enumerate(batch_results):
        meta = result["meta"]
        meta["index"] = int(record_index + i)
        writer.writerow(meta)

    return new_end


def create_hdf5_datasets(h5f: h5py.File, expected_len: int) -> dict[str, h5py.Dataset]:
    return {
        "waveforms": h5f.create_dataset(
            "waveforms",
            shape=(0, 3, expected_len),
            maxshape=(None, 3, expected_len),
            dtype="float32",
            compression="lzf",
            chunks=(8, 3, expected_len),
            shuffle=True,
        ),
        "east_km": h5f.create_dataset(
            "east_km", shape=(0,), maxshape=(None,), dtype="float32", compression="lzf"
        ),
        "north_km": h5f.create_dataset(
            "north_km", shape=(0,), maxshape=(None,), dtype="float32", compression="lzf"
        ),
        "depth_km": h5f.create_dataset(
            "depth_km", shape=(0,), maxshape=(None,), dtype="float32", compression="lzf"
        ),
        "sta_lat": h5f.create_dataset(
            "sta_lat", shape=(0,), maxshape=(None,), dtype="float32", compression="lzf"
        ),
        "sta_lon": h5f.create_dataset(
            "sta_lon", shape=(0,), maxshape=(None,), dtype="float32", compression="lzf"
        ),
    }


def process_dataset_split(
    csv_input_path: str | Path,
    hdf_output_path: str | Path,
    csv_output_path: str | Path,
    sac_handler: SacHandler,
    station_catalog: dict[str, tuple[float, float]],
    expected_len: int,
    logger: logging.Logger | None = None,
    chunksize: int = 50_000,
    max_workers: int | None = None,
) -> int:
    csv_input_path = Path(csv_input_path)
    hdf_output_path = Path(hdf_output_path)
    csv_output_path = Path(csv_output_path)

    validate_catalog_columns(csv_input_path)
    hdf_output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)

    meta_columns = pd.read_csv(csv_input_path, nrows=0).columns.tolist()
    if "index" not in meta_columns:
        meta_columns.append("index")

    worker = partial(
        process_record,
        sac_handler=sac_handler,
        station_catalog=station_catalog,
        expected_len=expected_len,
        logger=logger,
    )

    record_index = 0
    if max_workers is None:
        max_workers = min(8, multiprocessing.cpu_count())

    with (
        h5py.File(hdf_output_path, "w") as h5f,
        csv_output_path.open("w", newline="") as csvfile,
    ):
        writer = csv.DictWriter(csvfile, fieldnames=meta_columns)
        writer.writeheader()
        h5_datasets = create_hdf5_datasets(h5f, expected_len)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for chunk in pd.read_csv(csv_input_path, chunksize=chunksize):
                futures = [executor.submit(worker, rec) for _, rec in chunk.iterrows()]
                batch_results = []

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"{csv_input_path.stem}",
                    leave=False,
                ):
                    try:
                        result = future.result()
                    except Exception:
                        if logger is not None:
                            logger.exception("Error processing record")
                        continue
                    if result is not None:
                        batch_results.append(result)

                if batch_results:
                    record_index = append_batch(
                        h5_datasets, writer, batch_results, record_index
                    )

    if logger is not None:
        logger.info(
            f"Finished {csv_input_path.name}: {record_index} records written to "
            f"{hdf_output_path}"
        )
    return record_index


def build_station_catalog_dict(station_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    return {
        str(row.station): (float(row.lat), float(row.lon))
        for row in station_df.itertuples(index=False)
    }
