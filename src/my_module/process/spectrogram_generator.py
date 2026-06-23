from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import signal

from ..sac.sac_trace import SacTrace


@dataclass(frozen=True)
class StftParams:
    sampling_rate: float
    nperseg: int
    noverlap: int
    hop: int
    nfft: int
    window: np.ndarray
    transform: signal.ShortTimeFFT


class SpectrogramGenerator:
    def __init__(
        self,
        *,
        fft_window_sec: float = 5.0,
        overlap_rate: float = 0.762,
        freqmin: float = 0.0,
        freqmax: float = 10.0,
        normalize_type: str = "mean_std",
        eps: float = 1e-20,
        components: Sequence[str] = ("EW", "NS", "UD"),
        output_layout: str = "channels_last",
        align_legacy_stft_bins: bool = True,
        stft_padding: str = "zeros",
        logger: logging.Logger | None = None,
    ) -> None:
        self.fft_window_sec = float(fft_window_sec)
        self.overlap_rate = float(overlap_rate)
        self.freqmin = float(freqmin)
        self.freqmax = float(freqmax)
        self.normalize_type = self._normalize_type_name(normalize_type)
        self.eps = float(eps)
        self.components = tuple(components)
        self.output_layout = output_layout
        self.align_legacy_stft_bins = bool(align_legacy_stft_bins)
        self.stft_padding = stft_padding
        self.logger = logger or logging.getLogger(__name__)
        self.last_time_bins: np.ndarray | None = None
        self.last_freq_bins: np.ndarray | None = None
        self._stft_params_cache: dict[float, StftParams] = {}
        self._legacy_column_cache: dict[
            tuple[float, int, int, int], tuple[np.ndarray, np.ndarray]
        ] = {}

        self._validate_config()

    def _validate_config(self) -> None:
        if self.fft_window_sec <= 0:
            raise ValueError("fft_window_sec must be positive")
        if not 0 <= self.overlap_rate < 1:
            raise ValueError("overlap_rate must be in the range [0, 1)")
        if self.freqmin > self.freqmax:
            raise ValueError("freqmin must be less than or equal to freqmax")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.output_layout not in {"channels_first", "channels_last"}:
            raise ValueError(f"Unsupported output_layout: {self.output_layout}")
        if self.stft_padding not in {"zeros", "edge", "even", "odd"}:
            raise ValueError("stft_padding must be one of zeros, edge, even, or odd")

    @staticmethod
    def _normalize_type_name(value: str) -> str:
        normalized = str(value).lower()
        aliases = {
            "log_minmax": "log_min_to_one",
            "log_min_max": "log_min_to_one",
            "max_min": "min_max",
            "raw": "none",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"mean_std", "log_min_to_one", "min_max", "none"}:
            raise ValueError(f"normalize_type '{value}' is not supported.")
        return normalized

    def _get_stft_params(self, sampling_rate: float) -> StftParams:
        sampling_rate = float(sampling_rate)
        cached = self._stft_params_cache.get(sampling_rate)
        if cached is not None:
            return cached

        nperseg = int(round(sampling_rate * self.fft_window_sec))
        noverlap = int(round(nperseg * self.overlap_rate))
        hop = nperseg - noverlap
        if nperseg <= 0:
            raise ValueError("nperseg must be positive")
        if hop <= 0:
            raise ValueError("STFT hop must be positive")

        nfft = 2 ** int(nperseg).bit_length()
        window = signal.windows.tukey(nperseg, alpha=0.1)
        transform = signal.ShortTimeFFT(
            win=window,
            hop=hop,
            fs=sampling_rate,
            fft_mode="onesided",
            mfft=nfft,
            scale_to="magnitude",
        )
        params = StftParams(
            sampling_rate=sampling_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            hop=hop,
            nfft=nfft,
            window=window,
            transform=transform,
        )
        self._stft_params_cache[sampling_rate] = params
        return params

    def _get_trace(
        self,
        sac_traces: Mapping[str, SacTrace | None],
        component: str,
    ) -> SacTrace | None:
        return sac_traces.get(component)

    def generate_spectrograms(
        self,
        sac_traces: Mapping[str, SacTrace | None],
        normalize: bool = True,
    ) -> np.ndarray | None:
        spectrograms = []
        for component in self.components:
            sac_trace = self._get_trace(sac_traces, component)
            if sac_trace is None:
                return None

            spectrogram = self.generate_spectrogram(sac_trace, normalize=normalize)
            if spectrogram is None:
                return None
            spectrograms.append(spectrogram)

        stacked = np.stack(spectrograms, axis=0).astype(np.float32, copy=False)
        if self.output_layout == "channels_last":
            stacked = np.moveaxis(stacked, 0, -1)
        return stacked

    def generate_spectrogram_batch(
        self,
        sac_trace_batch: Sequence[Mapping[str, SacTrace | None]],
        normalize: bool = True,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Generate detector spectrograms for multiple windows at once.

        Parameters
        ----------
        sac_trace_batch : sequence of mappings
            Minute-length SAC traces. Each mapping must contain all configured
            components with the same sampling rate and sample count.
        normalize : bool, default=True
            Whether to normalize each component spectrogram independently.

        Returns
        -------
        tuple[np.ndarray, np.ndarray] or None
            Spectrograms with a leading batch axis and a boolean mask indicating
            which input windows produced valid spectrograms. The spectrogram
            layout follows ``output_layout``.
        """
        if not sac_trace_batch:
            return None

        params: StftParams | None = None
        npts: int | None = None
        component_waveforms: list[np.ndarray] = []

        for component in self.components:
            waveforms = []
            for sac_traces in sac_trace_batch:
                sac_trace = self._get_trace(sac_traces, component)
                if sac_trace is None:
                    return None

                trace_params = self._get_stft_params(sac_trace.stats.sampling_rate)
                if params is None:
                    params = trace_params
                    npts = sac_trace.data.size
                elif (
                    trace_params.sampling_rate != params.sampling_rate
                    or sac_trace.data.size != npts
                ):
                    raise ValueError(
                        "Batched spectrograms require equal sampling rates and "
                        "sample counts."
                    )

                waveforms.append(np.asarray(sac_trace.data, dtype=np.float64))

            component_waveforms.append(np.stack(waveforms, axis=0))

        if params is None or npts is None:
            return None

        waveform_batch = np.concatenate(component_waveforms, axis=0)
        freq, time_bins, zxx = self._run_stft_batch(waveform_batch, params)

        freq_mask = (self.freqmin <= freq) & (freq <= self.freqmax)
        if not np.any(freq_mask):
            self.logger.warning(
                "No STFT bins in frequency range %.3f-%.3f Hz",
                self.freqmin,
                self.freqmax,
            )
            return None

        self.last_time_bins = np.asarray(time_bins, dtype=np.float64)
        self.last_freq_bins = np.asarray(freq[freq_mask], dtype=np.float64)

        spec = self._to_spectral_values(zxx[:, freq_mask, :], params)
        valid_rows = np.all(np.isfinite(spec), axis=(1, 2))

        if normalize:
            spec, normalized_valid_rows = self._normalize_spectrogram_batch(spec)
            valid_rows &= normalized_valid_rows

        zero_rows = np.all(waveform_batch == 0, axis=1)
        spec[zero_rows] = 0.0

        n_components = len(self.components)
        n_windows = len(sac_trace_batch)
        component_first = spec.reshape(n_components, n_windows, *spec.shape[1:])
        valid_by_component = valid_rows.reshape(n_components, n_windows)
        valid_windows = np.all(valid_by_component, axis=0)

        stacked = np.moveaxis(component_first, 0, 1)
        if self.output_layout == "channels_last":
            stacked = np.moveaxis(stacked, 1, -1)

        return stacked.astype(np.float32, copy=False), valid_windows

    def generate_spectrogram(
        self,
        sac_trace: SacTrace,
        normalize: bool = True,
    ) -> np.ndarray | None:
        waveform = np.asarray(sac_trace.data, dtype=np.float64)
        params = self._get_stft_params(sac_trace.stats.sampling_rate)
        freq, time_bins, zxx = self._run_stft(waveform, params)

        freq_mask = (self.freqmin <= freq) & (freq <= self.freqmax)
        if not np.any(freq_mask):
            self.logger.warning(
                "No STFT bins in frequency range %.3f-%.3f Hz",
                self.freqmin,
                self.freqmax,
            )
            return None

        self.last_time_bins = np.asarray(time_bins, dtype=np.float64)
        self.last_freq_bins = np.asarray(freq[freq_mask], dtype=np.float64)
        spec = self._to_spectral_values(zxx[freq_mask, :], params)

        if np.all(waveform == 0):
            return np.zeros(spec.shape, dtype=np.float32)

        if normalize:
            spec = self._normalize_spectrogram(spec)
            if spec is None:
                return None

        return spec.astype(np.float32, copy=False)

    def _run_stft(
        self,
        waveform: np.ndarray,
        params: StftParams,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        zxx = params.transform.stft(waveform, padding=self.stft_padding)
        time_bins = params.transform.t(waveform.size)
        if self.align_legacy_stft_bins:
            time_bins, column_indices = self._legacy_time_bins(
                waveform.size,
                params,
                time_bins,
            )
            zxx = zxx[:, column_indices]
        return params.transform.f, time_bins, zxx

    def _run_stft_batch(
        self,
        waveforms: np.ndarray,
        params: StftParams,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        zxx = params.transform.stft(
            waveforms,
            padding=self.stft_padding,
            axis=-1,
        )
        time_bins = params.transform.t(waveforms.shape[-1])
        if self.align_legacy_stft_bins:
            time_bins, column_indices = self._legacy_time_bins(
                waveforms.shape[-1],
                params,
                time_bins,
            )
            zxx = zxx[..., column_indices]
        return params.transform.f, time_bins, zxx

    def _legacy_time_bins(
        self,
        npts: int,
        params: StftParams,
        short_time_bins: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        cache_key = (
            params.sampling_rate,
            int(npts),
            int(params.nperseg),
            int(params.hop),
        )
        cached = self._legacy_column_cache.get(cache_key)
        if cached is not None:
            return cached

        n_frames = self._legacy_frame_count(npts, params)
        legacy_time_bins = np.arange(n_frames, dtype=np.float64) * (
            params.hop / params.sampling_rate
        )
        column_indices = np.empty(n_frames, dtype=np.intp)
        for idx, time_bin in enumerate(legacy_time_bins):
            matches = np.where(np.isclose(short_time_bins, time_bin))[0]
            if matches.size != 1:
                raise RuntimeError(
                    "Could not align ShortTimeFFT time bin with legacy STFT: "
                    f"time={time_bin}"
                )
            column_indices[idx] = int(matches[0])

        cached = legacy_time_bins, column_indices
        self._legacy_column_cache[cache_key] = cached
        return cached

    @staticmethod
    def _legacy_frame_count(npts: int, params: StftParams) -> int:
        extended_npts = npts + params.nperseg
        padding_npts = (-(extended_npts - params.nperseg) % params.hop) % params.hop
        return 1 + (extended_npts + padding_npts - params.nperseg) // params.hop

    def _to_spectral_values(
        self,
        zxx: np.ndarray,
        params: StftParams,
    ) -> np.ndarray:
        magnitude = np.abs(zxx)
        if self.normalize_type == "mean_std":
            return magnitude
        return magnitude**2 / max(params.sampling_rate / params.nfft, self.eps)

    def _normalize_spectrogram(self, spec: np.ndarray) -> np.ndarray | None:
        if not np.all(np.isfinite(spec)):
            self.logger.warning("Spectrogram contains non-finite values.")
            return None

        if self.normalize_type == "log_min_to_one":
            if float(np.max(spec)) == float(np.min(spec)):
                self.logger.warning("Spectrogram is constant; returning zeros.")
                return np.zeros_like(spec, dtype=np.float64)
            min_value = float(np.min(spec))
            if min_value <= 0:
                min_value = self.eps
            scaled = np.log10(np.maximum(spec, self.eps) / min_value)
            max_value = float(np.max(scaled))
            if max_value <= 0 or not np.isfinite(max_value):
                self.logger.warning("Log-scaled spectrogram has invalid max value.")
                return None
            return scaled / max_value

        if self.normalize_type == "min_max":
            min_value = float(np.min(spec))
            max_value = float(np.max(spec))
            if max_value == min_value:
                self.logger.warning(
                    "Spectrogram min and max are identical; returning zeros."
                )
                return np.zeros_like(spec, dtype=np.float64)
            return (spec - min_value) / (max_value - min_value)

        if self.normalize_type == "mean_std":
            mean = float(np.mean(spec))
            std = float(np.std(spec))
            if std == 0:
                self.logger.warning("Spectrogram std is zero; returning zeros.")
                return np.zeros_like(spec, dtype=np.float64)
            return (spec - mean) / std

        if self.normalize_type == "none":
            return spec

        raise ValueError(f"normalize_type '{self.normalize_type}' is not supported.")

    def _normalize_spectrogram_batch(
        self,
        spec: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        valid_rows = np.all(np.isfinite(spec), axis=(1, 2))
        out = np.zeros_like(spec, dtype=np.float64)

        if self.normalize_type == "none":
            out[valid_rows] = spec[valid_rows]
            return out, valid_rows

        valid_spec = spec[valid_rows]
        if valid_spec.size == 0:
            return out, valid_rows

        if self.normalize_type == "log_min_to_one":
            min_value = np.min(valid_spec, axis=(1, 2), keepdims=True)
            max_value = np.max(valid_spec, axis=(1, 2), keepdims=True)
            constant = max_value == min_value
            safe_min = np.where(min_value <= 0, self.eps, min_value)
            scaled = np.log10(np.maximum(valid_spec, self.eps) / safe_min)
            scaled_max = np.max(scaled, axis=(1, 2), keepdims=True)
            scaled_valid = (scaled_max > 0) & np.isfinite(scaled_max)
            normalized = np.zeros_like(valid_spec, dtype=np.float64)
            use_scaled = (~constant) & scaled_valid
            np.divide(scaled, scaled_max, out=normalized, where=use_scaled)
            out[valid_rows] = normalized
            row_valid = np.ones(valid_spec.shape[0], dtype=bool)
            row_valid &= np.squeeze(constant | scaled_valid, axis=(1, 2))
            valid_rows[valid_rows] = row_valid
            return out, valid_rows

        if self.normalize_type == "min_max":
            min_value = np.min(valid_spec, axis=(1, 2), keepdims=True)
            max_value = np.max(valid_spec, axis=(1, 2), keepdims=True)
            denom = max_value - min_value
            normalized = np.zeros_like(valid_spec, dtype=np.float64)
            np.divide(valid_spec - min_value, denom, out=normalized, where=denom != 0)
            out[valid_rows] = normalized
            return out, valid_rows

        if self.normalize_type == "mean_std":
            mean = np.mean(valid_spec, axis=(1, 2), keepdims=True)
            std = np.std(valid_spec, axis=(1, 2), keepdims=True)
            normalized = np.zeros_like(valid_spec, dtype=np.float64)
            np.divide(valid_spec - mean, std, out=normalized, where=std != 0)
            out[valid_rows] = normalized
            return out, valid_rows

        raise ValueError(f"normalize_type '{self.normalize_type}' is not supported.")
