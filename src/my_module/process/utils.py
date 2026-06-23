from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from ..sac.sac_trace import SacTrace
from .spectrogram_generator import SpectrogramGenerator


def make_detector_specs(
    detector_spec_generator: SpectrogramGenerator,
    minute_slices: list[dict[str, SacTrace]],
    val_times: list[datetime],
) -> tuple[NDArray[np.float32] | None, list[datetime]]:
    """Create detector spectrogram batches from SAC traces.

    Parameters
    ----------
    detector_spec_generator : SpectrogramGenerator
        Spectrogram generator used for each minute window.
    minute_slices : list[dict[str, SacTrace]]
        Minute-length SAC traces.
    val_times : list[datetime]
        Timestamps corresponding to ``minute_slices``.

    Returns
    -------
    tuple[NDArray[np.float32] or None, list[datetime]]
        Spectrogram batch and sorted timestamps, or ``None`` and an empty list
        when no spectrograms can be generated.
    """
    specs_map: dict[datetime, NDArray[np.float32]] = {}
    complete_items = [
        (ts, tr)
        for tr, ts in zip(minute_slices, val_times)
        if all(
            tr.get(component) is not None
            for component in detector_spec_generator.components
        )
    ]

    if complete_items:
        keys, traces = zip(*sorted(complete_items, key=lambda item: item[0]))
        batch_result = detector_spec_generator.generate_spectrogram_batch(
            traces,
            normalize=True,
        )
        if batch_result is not None:
            batch_specs, valid_mask = batch_result
            for ts, spec, is_valid in zip(keys, batch_specs, valid_mask):
                if is_valid:
                    specs_map[ts] = spec

    if not specs_map:
        return None, []

    keys = sorted(specs_map.keys())
    specs = np.stack([specs_map[k] for k in keys], axis=0).astype(
        np.float32, copy=False
    )
    return specs, keys
