from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path

from obspy import UTCDateTime

from .sac_trace import SacTrace

TIME_FORMATS = (
    "%Y-%m-%d-%H:%M:%S.%f",
    "%Y/%m/%d-%H:%M:%S.%f",
    "%Y-%m-%d-%H:%M:%S",
    "%Y/%m/%d-%H:%M:%S",
)

class SacHandler:
    """Read hourly SAC files and return raw trimmed traces."""

    def __init__(
        self,
        *,
        year_to_path: Mapping[int, str | Path],
        component_channels: Mapping[str, str],
        duration_seconds: float | None = None,
        taper_max_percentage: float = 0.05,
        logger: logging.Logger | None = None,
    ) -> None:
        if not year_to_path:
            raise ValueError("year_to_path must be provided")
        if not component_channels:
            raise ValueError("component_channels must be provided")
        if duration_seconds is not None and duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")

        self.year_to_path = {
            int(year): Path(path).expanduser()
            for year, path in year_to_path.items()
        }
        self.component_channels = dict(component_channels)
        self.duration_seconds = duration_seconds
        self.taper_max_percentage = float(taper_max_percentage)
        self.logger = logger or logging.getLogger(__name__)

    def get_sac_traces(
        self,
        station_code: str,
        start_time: str | datetime | UTCDateTime,
        duration_seconds: float | None = None,
    ) -> dict[str, SacTrace | None]:
        if duration_seconds is None:
            duration_seconds = self.duration_seconds
        if duration_seconds is None:
            raise ValueError("duration_seconds must be provided to read SAC traces")
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")

        start_time, end_time = self._calculate_time_range(start_time, duration_seconds)
        return {
            component_name: self.get_sac_trace(
                station_code,
                start_time,
                end_time,
                channel_code,
            )
            for component_name, channel_code in self.component_channels.items()
        }

    def get_sac_trace(
        self,
        station_code: str,
        start_time: str | datetime | UTCDateTime,
        end_time: str | datetime | UTCDateTime,
        channel_code: str,
    ) -> SacTrace | None:
        start = to_utc_datetime(start_time)
        end = to_utc_datetime(end_time)
        if not start < end:
            raise ValueError("start_time must be earlier than end_time")

        sac_filepaths = self._generate_sac_filepaths(
            start,
            end,
            station_code,
            channel_code,
        )
        sac_trace = self._read_and_concatenate_sac_traces(sac_filepaths)
        if sac_trace is None:
            return None
        trimmed = sac_trace.trim(start, end)
        if trimmed is None:
            self.logger.warning(
                "Requested SAC range is outside loaded trace; rejecting request: "
                "station=%s channel=%s request=%s - %s trace=%s - %s",
                station_code,
                channel_code,
                start,
                end,
                sac_trace.stats.start_time,
                sac_trace.stats.end_time,
            )
        return trimmed

    def filter_sac_traces(
        self,
        sac_traces: Mapping[str, SacTrace | None],
        freqmin: float,
        freqmax: float,
        filter_type: str = "bandpass",
        corners: int = 2,
        zerophase: bool = True,
        taper_max_percentage: float | None = None,
    ) -> dict[str, SacTrace | None]:
        if taper_max_percentage is None:
            taper_max_percentage = self.taper_max_percentage

        return {
            component_name: (
                trace.filtered(
                    freqmin=freqmin,
                    freqmax=freqmax,
                    filter_type=filter_type,
                    corners=corners,
                    zerophase=zerophase,
                    taper_max_percentage=taper_max_percentage,
                )
                if trace is not None
                else None
            )
            for component_name, trace in sac_traces.items()
        }

    def split_sac_traces_by_minute(
        self,
        sac_traces: Mapping[str, SacTrace | None],
        base_time: str | datetime | UTCDateTime,
        *,
        minutes: int = 60,
        duration_seconds: float = 59.99,
        components: tuple[str, ...] = ("EW", "NS", "UD"),
    ) -> tuple[list[dict[str, SacTrace]], list[datetime]]:
        if minutes <= 0:
            raise ValueError("minutes must be positive")
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")

        start = to_utc_datetime(base_time)
        minute_traces: list[dict[str, SacTrace]] = []
        times: list[datetime] = []

        for minute in range(minutes):
            t0 = start + 60.0 * minute
            t1 = t0 + duration_seconds
            trace_minute: dict[str, SacTrace] = {}
            for component in components:
                trace = sac_traces.get(component)
                if trace is None:
                    break
                trimmed = trace.trim(t0, t1)
                if trimmed is None:
                    break
                trace_minute[component] = trimmed
            else:
                minute_traces.append(trace_minute)
                times.append(t0.datetime)

        return minute_traces, times

    def _read_sac_trace(self, sac_filepath: str | Path) -> SacTrace | None:
        return SacTrace.from_file(sac_filepath)

    def _calculate_time_range(
        self,
        start_time: str | datetime | UTCDateTime,
        duration_seconds: float,
    ) -> tuple[UTCDateTime, UTCDateTime]:
        start = to_utc_datetime(start_time)
        return start, start + float(duration_seconds)

    def _generate_sac_filepaths(
        self,
        start_time: UTCDateTime,
        end_time: UTCDateTime,
        station_code: str,
        channel_code: str,
    ) -> list[Path]:
        sac_filepaths: list[Path] = []
        current_time = floor_to_hour(start_time)
        end_time_dt = floor_to_hour(end_time)

        while current_time <= end_time_dt:
            sac_filepaths.append(
                self._create_sac_filepath(current_time, station_code, channel_code)
            )
            current_time += timedelta(hours=1)

        return sac_filepaths

    def _read_and_concatenate_sac_traces(
        self,
        sac_filepaths: list[Path],
    ) -> SacTrace | None:
        if not sac_filepaths:
            return None

        sac_trace = self._read_sac_trace(sac_filepaths[0])
        if sac_trace is None:
            self._log_unreadable_sac_file(sac_filepaths[0])
            return None

        for sac_filepath in sac_filepaths[1:]:
            additional_sac_trace = self._read_sac_trace(sac_filepath)
            if additional_sac_trace is None:
                self._log_unreadable_sac_file(sac_filepath)
                return None

            try:
                sac_trace += additional_sac_trace
            except ValueError as e:
                self.logger.warning(
                    "SAC traces are not continuous; rejecting request: "
                    "station=%s channel=%s previous_end=%s next_start=%s file=%s "
                    "error=%s",
                    sac_trace.stats.station_code,
                    sac_trace.stats.channel_code,
                    sac_trace.stats.end_time,
                    additional_sac_trace.stats.start_time,
                    sac_filepath,
                    e,
                )
                return None

        return sac_trace

    def _log_unreadable_sac_file(self, sac_filepath: Path) -> None:
        self.logger.warning(
            "SAC file could not be read; rejecting requests that require it: file=%s",
            sac_filepath,
        )

    def _create_sac_filepath(
        self,
        hour_start: datetime,
        station_code: str,
        channel_code: str,
    ) -> Path:
        year = int(hour_start.year)
        try:
            base_path = self.year_to_path[year]
        except KeyError as exc:
            raise KeyError(
                f"Year {year} is not configured in the SAC data location settings"
            ) from exc

        sac_base_dir = hour_start.strftime("%Y%m%d%H")
        return Path(base_path) / sac_base_dir / f"{station_code}.{channel_code}.SAC"


def floor_to_hour(value: UTCDateTime) -> datetime:
    dt = value.datetime
    return dt.replace(minute=0, second=0, microsecond=0)


def to_utc_datetime(value: str | datetime | UTCDateTime) -> UTCDateTime:
    if isinstance(value, UTCDateTime):
        return value
    if isinstance(value, datetime):
        return UTCDateTime(value)
    return UTCDateTime(parse_time(str(value)))


def parse_time(value: str) -> datetime:
    for fmt in TIME_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(
        f"Unsupported time format: {value!r}. Expected one of {TIME_FORMATS}."
    )
