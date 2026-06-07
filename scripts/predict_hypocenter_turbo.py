"""Run tremor detection and hypocenter localization over SAC archives."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import queue
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
from numpy.typing import NDArray
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from my_module import create_logger, get_station_catalog
from my_module.config import ExperimentConfig, ProjectConfig
from my_module.models import (
    build_vit_locator,
    load_cnn_detector,
    load_vit_locator_weights,
)
from my_module.prediction import convert_arrival_time, convert_relative_to_geo, infer
from my_module.process import (
    SpectrogramGenerator,
    convert_sactraces_to_waveform,
    make_detector_specs,
    normalize,
)
from my_module.sac import SacHandler

FloatArray = NDArray[np.float32 | np.float64]


class StationLocation(TypedDict):
    """Station latitude and longitude.

    Attributes
    ----------
    lat : float
        Station latitude in degrees.
    lon : float
        Station longitude in degrees.
    """

    lat: float
    lon: float


PredictionQueueItem = tuple[
    str,
    list[datetime],
    NDArray[np.float32],
    NDArray[np.float32],
    StationLocation,
]


BASE_CSV_HEADER: list[str] = [
    "origin_time",
    "pred_lat",
    "pred_lon",
    "pred_dep",
    "east_km",
    "north_km",
    "depth_km",
    "sigma11",
    "sigma12",
    "sigma13",
    "sigma21",
    "sigma22",
    "sigma23",
    "sigma31",
    "sigma32",
    "sigma33",
    "tremor_proba",
    "noise_proba",
    "eq_proba",
    "station",
    "station_lat",
    "station_lon",
]

ARRIVAL_CSV_HEADER: list[str] = ["pred_arrival_sec", "pred_arrival_sec_std"]


def set_time_list(start_time: str, end_time: str) -> list[datetime]:
    """Create an hourly datetime list.

    Parameters
    ----------
    start_time, end_time : str
        Inclusive start and end times formatted as ``"%Y-%m-%d-%H:%M:%S.%f"``.

    Returns
    -------
    list[datetime]
        Datetimes spaced by one hour.
    """
    st = datetime.strptime(start_time, "%Y-%m-%d-%H:%M:%S.%f")
    et = datetime.strptime(end_time, "%Y-%m-%d-%H:%M:%S.%f")
    out: list[datetime] = []
    cur = st
    while cur <= et:
        out.append(cur)
        cur += timedelta(hours=1)
    return out


def format_output_time(dt: datetime) -> str:
    """Format a datetime for output filenames.

    Parameters
    ----------
    dt : datetime
        Datetime to format.

    Returns
    -------
    str
        Timestamp with millisecond precision.
    """
    return dt.strftime("%Y-%m-%d-%H:%M:%S.%f")[:-3]


def fetch_spectrogram_and_waveform(
    sac_handler: SacHandler,
    spec_generator: SpectrogramGenerator,
    freqmin_for_detector: float,
    freqmax_for_detector: float,
    components: list[str],
    normalization_type: str,
    prediction_queue: queue.Queue[PredictionQueueItem | object],
    start_time: datetime,
    station: str,
    station_loc: StationLocation,
) -> None:
    """Fetch detector spectrograms and locator waveforms for one station-hour.

    Parameters
    ----------
    sac_handler : SacHandler
        SAC reader and filter helper.
    spec_generator : SpectrogramGenerator
        Spectrogram generator for detector input.
    freqmin_for_detector, freqmax_for_detector : float
        Detector filter frequency bounds.
    components : list[str]
        Waveform components used by the locator.
    normalization_type : str
        Locator waveform normalization mode.
    prediction_queue : queue.Queue
        Queue receiving ready-to-run model inputs.
    start_time : datetime
        Start of the hour to process.
    station : str
        Station code.
    station_loc : StationLocation
        Station latitude and longitude.
    """
    tr_all = sac_handler.get_sac_traces(
        station_code=station,
        start_time=start_time.strftime("%Y-%m-%d-%H:%M:%S.%f"),
        duration_seconds=3599.99,
    )
    if any(tr_all.get(c) is None for c in ["EW", "NS", "UD"]):
        return

    det_tr = sac_handler.filter_sac_traces(
        tr_all,
        freqmin=freqmin_for_detector,
        freqmax=freqmax_for_detector,
        corners=2,
    )

    det_tr_minutes, det_times = sac_handler.split_sac_traces_by_minute(
        det_tr,
        start_time,
    )
    all_specs, spec_times = make_detector_specs(
        spec_generator,
        det_tr_minutes,
        det_times,
    )
    if all_specs is None or len(spec_times) == 0:
        return

    loc_tr_minutes, loc_times = sac_handler.split_sac_traces_by_minute(
        tr_all,
        start_time,
    )
    seg_map = {ts: seg for seg, ts in zip(loc_tr_minutes, loc_times)}

    all_waves: list[NDArray[np.float32]] = []
    for ts in spec_times:
        seg = seg_map.get(ts)
        if seg is None:
            continue

        waveform = convert_sactraces_to_waveform(seg, components=components)
        normed_waveform = normalize(
            waveform, components=components, normalization_type=normalization_type
        )
        all_waves.append(normed_waveform)

    if not all_waves:
        return

    all_waves = np.stack(all_waves, axis=0).astype(np.float32, copy=False)
    prediction_queue.put(
        (station, spec_times, all_specs, all_waves, station_loc), block=True
    )


def process_predict(
    output_csv_path: Path,
    sac_handler: SacHandler,
    spec_generator: SpectrogramGenerator,
    detector_model: Any,
    locator_model: torch.nn.Module,
    time_station_list: list[tuple[datetime, str]],
    station_catalog_dict: dict[str, StationLocation],
    device: str,
    freqmin: float,
    freqmax: float,
    normalization_type: str,
    input_components: list[str],
    scale_km: float,
    win_sec: float,
    return_arrival: bool,
    logger: logging.Logger,
    n_producers: int = 8,
    write_batch_rows: int = 10_000,
) -> None:
    """Run threaded detector and locator inference and write predictions.

    Parameters
    ----------
    output_csv_path : Path
        Destination CSV path.
    sac_handler : SacHandler
        SAC reader and filter helper.
    spec_generator : SpectrogramGenerator
        Detector spectrogram generator.
    detector_model : Any
        Loaded Keras detector model.
    locator_model : torch.nn.Module
        Loaded locator model.
    time_station_list : list[tuple[datetime, str]]
        Station-hour tasks to process.
    station_catalog_dict : dict[str, StationLocation]
        Station metadata indexed by station code.
    device : str
        Inference device.
    freqmin, freqmax : float
        Detector filter frequency bounds.
    normalization_type : str
        Locator waveform normalization mode.
    input_components : list[str]
        Locator input components.
    scale_km : float
        Hypocenter scale factor in kilometers.
    win_sec : float
        Arrival-time window length in seconds.
    return_arrival : bool
        Whether to write arrival-time predictions.
    logger : logging.Logger
        Logger for producer failures.
    n_producers : int, default=8
        Number of producer threads.
    write_batch_rows : int, default=10000
        Number of rows buffered before writing.
    """
    n_producers = max(1, min(n_producers, len(time_station_list) or 1))
    chunks = np.array_split(time_station_list, n_producers)
    prediction_queue: queue.Queue[PredictionQueueItem | object] = queue.Queue(
        maxsize=4096
    )
    stop_item = object()

    total_tasks = len(time_station_list)
    pbar = tqdm(total=total_tasks, desc="Processing stations", ncols=100)

    det_device = next(detector_model.parameters()).device
    detector_model.eval()

    def producer_chunk(chunk: list[tuple[datetime, str]]) -> None:
        """Produce queued model inputs for a subset of station-hour tasks."""
        for start_time, station in chunk:
            try:
                station_loc = station_catalog_dict[station]
                fetch_spectrogram_and_waveform(
                    sac_handler=sac_handler,
                    spec_generator=spec_generator,
                    freqmin_for_detector=freqmin,
                    freqmax_for_detector=freqmax,
                    components=input_components,
                    normalization_type=normalization_type,
                    prediction_queue=prediction_queue,
                    start_time=start_time,
                    station=station,
                    station_loc=station_loc,
                )
            except Exception as e:
                logger.exception(f"[producer] {station} {start_time}: {e}")
            finally:
                pbar.update(1)
        prediction_queue.put(stop_item)

    with (
        ThreadPoolExecutor(max_workers=n_producers) as exe,
        open(output_csv_path, "w", newline="") as csvfile,
    ):
        writer = csv.writer(csvfile)
        csv_header = BASE_CSV_HEADER + (ARRIVAL_CSV_HEADER if return_arrival else [])
        writer.writerow(csv_header)

        for ch in chunks:
            exe.submit(producer_chunk, list(ch))

        out_rows: list[list[str | float]] = []
        flush = writer.writerows

        stops_seen = 0
        while stops_seen < n_producers:
            item = prediction_queue.get()

            if item is stop_item:
                stops_seen += 1
                continue

            item = item
            station, val_times, specs_np, waves_np, station_loc = item

            # detector
            specs_t = torch.from_numpy(specs_np).to(det_device, non_blocking=True)
            with torch.no_grad():
                probs = detector_model(specs_t).detach().cpu().numpy()

            locator_preds = infer(
                locator_model,
                waves_np,
                return_arrival=return_arrival,
            )
            hypo_preds = locator_preds["hypo"]
            arrival_preds = locator_preds["arrival_time"]

            for idx, (tt, proba, pred) in enumerate(zip(val_times, probs, hypo_preds)):
                sta_lat = float(station_loc["lat"])
                sta_lon = float(station_loc["lon"])

                pred_hypo, mean_km, Sigma_km2 = convert_relative_to_geo(
                    prediction=pred,
                    station_loc=(sta_lat, sta_lon),
                    scale_km=scale_km,
                )

                row = [
                    tt.strftime("%Y-%m-%d-%H:%M:%S.%f"),
                    f"{pred_hypo[0]:.4f}",
                    f"{pred_hypo[1]:.3f}",
                    f"{pred_hypo[2]:.3f}",
                    f"{mean_km[0]:.6f}",
                    f"{mean_km[1]:.6f}",
                    f"{mean_km[2]:.6f}",
                    f"{Sigma_km2[0, 0]:.6f}",
                    f"{Sigma_km2[0, 1]:.6f}",
                    f"{Sigma_km2[0, 2]:.6f}",
                    f"{Sigma_km2[1, 0]:.6f}",
                    f"{Sigma_km2[1, 1]:.6f}",
                    f"{Sigma_km2[1, 2]:.6f}",
                    f"{Sigma_km2[2, 0]:.6f}",
                    f"{Sigma_km2[2, 1]:.6f}",
                    f"{Sigma_km2[2, 2]:.6f}",
                    f"{proba[1]:.5f}",
                    f"{proba[0]:.5f}",
                    f"{proba[2]:.5f}",
                    station,
                    station_loc["lat"],
                    station_loc["lon"],
                ]

                if return_arrival:
                    if arrival_preds is not None:
                        pred_arrival_sec, pred_arrival_std = convert_arrival_time(
                            arrival_preds[idx],
                            win_sec=win_sec,
                        )
                    else:
                        pred_arrival_sec, pred_arrival_std = np.nan, np.nan
                    row.extend(
                        [
                            f"{pred_arrival_sec:.6f}",
                            f"{pred_arrival_std:.6f}",
                        ]
                    )

                out_rows.append(row)

                if len(out_rows) >= write_batch_rows:
                    flush(out_rows)
                    out_rows.clear()

        if out_rows:
            flush(out_rows)
            out_rows.clear()

    pbar.close()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run detector+locator over catalog.")
    parser.add_argument("--exp", type=str, default="vit_locator_v11")
    parser.add_argument(
        "--start_time", type=str, required=True, help="YYYY-mm-dd-HH:MM:SS.ffffff"
    )
    parser.add_argument(
        "--end_time", type=str, required=True, help="YYYY-mm-dd-HH:MM:SS.ffffff"
    )
    parser.add_argument("--n_producers", type=int, default=8)
    parser.add_argument(
        "--project_config",
        type=str,
        default="config/project_config.json",
        help="Path to project config JSON (relative to project root or absolute).",
    )
    parser.add_argument(
        "--station_file",
        type=str,
        default="data/version1.0/station/hinet_used.txt",
        help="Station catalog path (relative to project root or absolute).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for per-station predictions. Defaults to reports/<project>/hypocenter/org.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Inference device: "auto", "cpu", "cuda", or "cuda:<index>".',
    )
    return parser.parse_args()


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root.

    Parameters
    ----------
    path : str or Path
        Path to resolve.

    Returns
    -------
    Path
        Absolute path.
    """
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


def resolve_device(device: str) -> str:
    """Resolve an inference device string.

    Parameters
    ----------
    device : str
        Requested device, including ``"auto"``.

    Returns
    -------
    str
        Device string available on this machine.
    """
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA is not available; falling back to CPU.")
        return "cpu"
    return device


def main() -> None:
    """Run detector and hypocenter locator inference."""
    args = parse_args()
    time_list = set_time_list(args.start_time, args.end_time)
    start_dt = datetime.strptime(args.start_time, "%Y-%m-%d-%H:%M:%S.%f")
    end_dt = datetime.strptime(args.end_time, "%Y-%m-%d-%H:%M:%S.%f")

    project_cfg_path = resolve_project_path(args.project_config)
    project_cfg = ProjectConfig.from_file(project_cfg_path)
    project_name = project_cfg.base.project_name
    print(project_name)

    device = resolve_device(args.device)
    log_dir = ROOT / f"logs/{project_name}"
    experiments_dir = ROOT / "config/experiments"
    output_dir = (
        resolve_project_path(args.output_dir)
        if args.output_dir
        else ROOT / f"reports/{project_name}/hypocenter/org"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    detector_config_path = ROOT / "config/detector_configs.json"
    log_path = log_dir / "predict_hypocenter_turbo.log"
    config_path = (experiments_dir / f"{args.exp}.json").resolve()
    station_path = resolve_project_path(args.station_file)
    output_csv_path = (
        output_dir / f"{format_output_time(start_dt)}_{format_output_time(end_dt)}.csv"
    )

    logger = create_logger(log_path)

    with (detector_config_path).open("r", encoding="utf-8") as f:
        model_cfgs: dict[str, dict[str, Any]] = json.load(f)
    det_cfg = model_cfgs["tremor_detector"]
    det_model_path = ROOT / det_cfg["path"]

    detector_model = load_cnn_detector(det_model_path, device=device)

    experiment_cfg = ExperimentConfig.from_file(config_path)

    experiment_cfg.train.device = device
    locator_model = build_vit_locator(experiment_cfg, save_attention=True)
    weight_path = ROOT / Path(experiment_cfg.output.save_dir) / "best_weight.pth"
    locator_model = load_vit_locator_weights(
        locator_model, device=device, weight_path=weight_path
    )

    sac_params = {
        "duration_seconds": project_cfg.waveform.duration_sec,
        "year_to_path": project_cfg.sac.year_to_path,
        "component_channels": project_cfg.sac.component_channels,
    }
    spec_params = {
        "fft_window_sec": project_cfg.spectrogram.fft_window_sec,
        "overlap_rate": project_cfg.spectrogram.overlap_rate,
        "freqmin": project_cfg.spectrogram.freqmin,
        "freqmax": project_cfg.spectrogram.freqmax,
        "normalize_type": project_cfg.spectrogram.normalize_type,
    }
    sac_handler = SacHandler(**sac_params)
    spec_generator = SpectrogramGenerator(**spec_params)

    station_df = get_station_catalog(station_path)
    station_catalog_dict = station_df.set_index("station")[["lat", "lon"]].to_dict(
        "index"
    )

    time_station_list = [
        (tp, sc) for sc in station_catalog_dict.keys() for tp in time_list
    ]

    process_predict(
        output_csv_path=output_csv_path,
        sac_handler=sac_handler,
        spec_generator=spec_generator,
        detector_model=detector_model,
        locator_model=locator_model,
        time_station_list=time_station_list,
        station_catalog_dict=station_catalog_dict,
        device=device,
        freqmin=project_cfg.waveform.freqmin_for_detector,
        freqmax=project_cfg.waveform.freqmax_for_detector,
        normalization_type=experiment_cfg.data.normalization_type,
        input_components=experiment_cfg.data.input_components,
        scale_km=experiment_cfg.data.scale_km,
        win_sec=experiment_cfg.data.win_sec,
        return_arrival=experiment_cfg.data.arrival_time,
        logger=logger,
        n_producers=args.n_producers,
    )


if __name__ == "__main__":
    main()
