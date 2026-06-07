from __future__ import annotations

import math
from pathlib import Path
from typing import TypeVar

import numpy as np
import obspy
from obspy import Stream, Trace, UTCDateTime
from obspy.signal import filter as obspy_filter
from obspy.signal.invsim import cosine_taper
from scipy import signal

T = TypeVar("T", bound="SacTrace")


class SacStats:
    """SAC file metadata."""

    def __init__(
        self,
        *,
        station_code: str,
        channel_code: str,
        sampling_rate: float,
        npts: int,
        start_time: UTCDateTime,
        end_time: UTCDateTime,
    ) -> None:
        self.station_code = station_code
        self.channel_code = channel_code
        self.sampling_rate = float(sampling_rate)
        self.npts = int(npts)
        self.start_time = start_time
        self.end_time = end_time

    @property
    def station(self) -> str:
        return self.station_code

    @property
    def channel(self) -> str:
        return self.channel_code

    @property
    def starttime(self) -> UTCDateTime:
        return self.start_time

    @property
    def endtime(self) -> UTCDateTime:
        return self.end_time

    def copy(self) -> "SacStats":
        return SacStats(
            station_code=self.station_code,
            channel_code=self.channel_code,
            sampling_rate=self.sampling_rate,
            npts=self.npts,
            start_time=self.start_time,
            end_time=self.end_time,
        )

    def __str__(self) -> str:
        return (
            f"Station: {self.station_code}\n"
            f"Channel: {self.channel_code}\n"
            f"Start Time: {self.start_time}\n"
            f"End Time: {self.end_time}\n"
            f"Sampling Rate: {self.sampling_rate} Hz\n"
            f"Number of Points: {self.npts}"
        )


def calculate_taper_padding_npts(npts: int, taper_max_percentage: float) -> int:
    if npts <= 0 or taper_max_percentage <= 0:
        return 0
    if not taper_max_percentage < 1:
        raise ValueError("taper_max_percentage must be less than 1")

    # cosine_taper tapers p / 2 of both ends of the padded signal. Pad enough
    # samples so the returned center segment starts after that taper ramp.
    return min(
        int(math.ceil(npts * taper_max_percentage / (2 * (1 - taper_max_percentage)))),
        npts,
    )


class SacTrace:
    """SAC trace data and metadata."""

    def __init__(
        self, data: np.ndarray, stats: SacStats, *, copy: bool = False
    ) -> None:
        self.data = np.asarray(data).copy() if copy else np.asarray(data)
        self.stats = stats.copy() if copy else stats

    @property
    def sampling_rate(self) -> float:
        return self.stats.sampling_rate

    @property
    def npts(self) -> int:
        return self.stats.npts

    @classmethod
    def from_file(
        cls: type[T],
        sac_filepath: str | Path,
    ) -> T | None:
        try:
            sac_data_stream: Stream = obspy.read(
                str(sac_filepath),
                format="SAC",
                check_compression=False,
            )
        except Exception:
            return None

        trace: Trace = sac_data_stream[0]
        trace.data = np.asarray(trace.data, dtype=np.float32) * 1e-9

        stats = SacStats(
            station_code=trace.stats.station,
            channel_code=trace.stats.channel,
            sampling_rate=trace.stats.sampling_rate,
            npts=trace.stats.npts,
            start_time=trace.stats.starttime,
            end_time=trace.stats.endtime,
        )
        return cls(trace.data, stats)

    def _can_concatenate(self, other: T) -> bool:
        return (
            self.stats.station_code == other.stats.station_code
            and self.stats.channel_code == other.stats.channel_code
            and self.stats.sampling_rate == other.stats.sampling_rate
            and self.stats.end_time + (1 / self.stats.sampling_rate)
            == other.stats.start_time
        )

    def __add__(self: T, other_trace: T) -> T:
        if not self._can_concatenate(other_trace):
            raise ValueError("Cannot concatenate the given SacTraces.")

        new_data = np.concatenate((self.data, other_trace.data))
        new_stats = SacStats(
            station_code=self.stats.station_code,
            channel_code=self.stats.channel_code,
            sampling_rate=self.stats.sampling_rate,
            npts=new_data.size,
            start_time=self.stats.start_time,
            end_time=other_trace.stats.end_time,
        )
        return type(self)(new_data, new_stats)

    def __str__(self) -> str:
        return (
            f"{self.stats.station_code}.{self.stats.channel_code} | "
            f"{self.stats.start_time} - {self.stats.end_time} | "
            f"{self.stats.sampling_rate:.1f} Hz | "
            f"{self.stats.npts} samples"
        )

    def filtered(
        self: T,
        freqmin: float | None = None,
        freqmax: float | None = None,
        corners: int = 2,
        zerophase: bool = True,
        filter_type: str = "bandpass",
        taper_type: str = "cosine",
        taper_max_percentage: float = 0.05,
    ) -> T:
        if freqmin is None or freqmax is None:
            raise TypeError("freqmin and freqmax must be provided")
        if filter_type not in {"bandpass", "bandstop"}:
            raise ValueError(f"Unsupported filter_type: {filter_type}")
        if not 0 <= taper_max_percentage < 1:
            raise ValueError("taper_max_percentage must be in the range [0, 1)")

        data = np.asarray(self.data, dtype=np.float64).copy()
        sampling_rate = self.stats.sampling_rate
        padding_npts = calculate_taper_padding_npts(
            data.size,
            taper_max_percentage,
        )

        if padding_npts > 0:
            data = np.concatenate(
                [
                    data[:padding_npts][::-1],
                    data,
                    data[-padding_npts:][::-1],
                ]
            )

        data = signal.detrend(data, type="linear")
        data = data - np.mean(data)
        if taper_max_percentage > 0:
            if taper_type != "cosine":
                raise ValueError("Only cosine taper is supported")
            data = data * cosine_taper(data.size, p=taper_max_percentage)

        if filter_type == "bandpass":
            data = obspy_filter.bandpass(
                data,
                freqmin=freqmin,
                freqmax=freqmax,
                df=sampling_rate,
                corners=corners,
                zerophase=zerophase,
            )
        else:
            data = obspy_filter.bandstop(
                data,
                freqmin=freqmin,
                freqmax=freqmax,
                df=sampling_rate,
                corners=corners,
                zerophase=zerophase,
            )

        if padding_npts > 0:
            filtered_data = data[padding_npts : padding_npts + self.data.size]
        else:
            filtered_data = data

        return type(self)(np.asarray(filtered_data, dtype=np.float32), self.stats)

    def trim(
        self: T,
        start_time: UTCDateTime,
        end_time: UTCDateTime,
    ) -> T | None:
        if not (self.stats.start_time <= start_time < end_time <= self.stats.end_time):
            return None

        start_idx = round(
            (start_time - self.stats.start_time) * self.stats.sampling_rate
        )
        end_idx = (
            round((end_time - self.stats.start_time) * self.stats.sampling_rate) + 1
        )
        trimmed_data = self.data[start_idx:end_idx]
        new_stats = SacStats(
            station_code=self.stats.station_code,
            channel_code=self.stats.channel_code,
            sampling_rate=self.stats.sampling_rate,
            npts=trimmed_data.size,
            start_time=start_time,
            end_time=end_time,
        )
        return type(self)(trimmed_data, new_stats)
