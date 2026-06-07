from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, get_worker_info

from ..config import ExperimentConfig
from ..process.waveform import normalize


class SeismicDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        hdf5_path: str | Path,
        input_components: list[str] | None,
        normalization_type: str,
        scale_km: float,
        fs: float,
        win_sec: float,
        jitter_sec: float,
        training: bool,
        arrival_time: bool,
        include_station: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.hdf5_path = Path(hdf5_path)
        self.catalog = pd.read_csv(self.csv_path)
        self.input_components = input_components
        self.normalization_type = normalization_type
        self.scale_km = scale_km
        self.fs = fs
        self.win_sec = win_sec
        self.win_len = int(round(win_sec * fs))
        self.jitter_len = int(round(jitter_sec * fs))
        self.training = training
        self.arrival_time = arrival_time
        self.include_station = include_station
        self._h5_file: h5py.File | None = None

        if "index" not in self.catalog.columns:
            raise ValueError(f"{self.csv_path} must contain an 'index' column.")

        self.indices = self.catalog["index"].to_numpy(np.int64)
        if self.arrival_time:
            self.arrival_times = pd.to_datetime(
                self.catalog["arrival_time"]
            ).to_numpy(dtype="datetime64[ns]")
            self.start_times = pd.to_datetime(
                self.catalog["start_time_for_trainlocator"]
            ).to_numpy(dtype="datetime64[ns]")

    def _get_h5(self) -> h5py.File:
        if self._h5_file is None:
            self._h5_file = h5py.File(self.hdf5_path, "r")
        return self._h5_file

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[Tensor, ...]:
        h5_index = int(self.indices[idx])
        h5 = self._get_h5()

        if self.training:
            off = np.random.randint(0, 2 * self.jitter_len + 1)
        else:
            off = self.jitter_len

        waveform = h5["waveforms"][h5_index, :, off : off + self.win_len]
        waveform = normalize(
            waveform,
            components=self.input_components,
            normalization_type=self.normalization_type,
        )

        target = torch.tensor(
            [
                float(h5["east_km"][h5_index]) / self.scale_km,
                float(h5["north_km"][h5_index]) / self.scale_km,
                float(h5["depth_km"][h5_index]) / self.scale_km,
            ],
            dtype=torch.float32,
        )
        waveform_tensor = torch.from_numpy(np.asarray(waveform, dtype=np.float32))

        if not self.arrival_time and not self.include_station:
            return waveform_tensor, target

        sta_lat_tensor = torch.tensor(float(h5["sta_lat"][h5_index]), dtype=torch.float32)
        sta_lon_tensor = torch.tensor(float(h5["sta_lon"][h5_index]), dtype=torch.float32)

        if not self.arrival_time:
            return waveform_tensor, target, sta_lat_tensor, sta_lon_tensor

        dt_ns = int(round(1e9 / self.fs))
        window_start_time = self.start_times[idx] + np.timedelta64(off * dt_ns, "ns")
        window_center_time = window_start_time + np.timedelta64(
            int(round(self.win_sec * 1e9 / 2)), "ns"
        )
        offset_sec = (self.arrival_times[idx] - window_center_time) / np.timedelta64(
            1, "s"
        )
        offset_tensor = torch.tensor(
            offset_sec / (self.win_sec / 2), dtype=torch.float32
        )
        if self.include_station:
            return waveform_tensor, target, offset_tensor, sta_lat_tensor, sta_lon_tensor
        return waveform_tensor, target, offset_tensor

    def close(self) -> None:
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def __del__(self) -> None:
        self.close()


def init_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    worker_info = get_worker_info()
    if worker_info is None:
        return

    dataset = worker_info.dataset
    if getattr(dataset, "_h5_file", None) is not None:
        dataset._h5_file.close()
    dataset._h5_file = None


def build_dataset(
    cfg: ExperimentConfig,
    name: str,
    training: bool,
    include_station: bool = False,
) -> SeismicDataset:
    dataset_dir = Path(cfg.data.dataset_dir)
    return SeismicDataset(
        csv_path=dataset_dir / f"{name}.csv",
        hdf5_path=dataset_dir / "hdf" / f"{name}.h5",
        input_components=cfg.data.input_components,
        normalization_type=cfg.data.normalization_type,
        scale_km=cfg.data.scale_km,
        fs=cfg.data.fs,
        win_sec=cfg.data.win_sec,
        jitter_sec=cfg.data.jitter_sec,
        training=training,
        arrival_time=cfg.data.arrival_time,
        include_station=include_station,
    )


def build_loader(
    dataset: Dataset,
    batch_size: int,
    seed: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs: dict[str, Any] = {}
    if num_workers > 0:
        kwargs.update(prefetch_factor=1, persistent_workers=False)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=init_worker,
        generator=generator if shuffle else None,
        **kwargs,
    )
