import numpy as np

from ..sac.sac_trace import SacTrace

COMP_DICT = {"EW": 0, "NS": 1, "UD": 2}


def normalize(
    raw_waveform: np.ndarray,
    components: list[str] | None,
    normalization_type: str,
) -> np.ndarray:
    """Normalize waveform channels.

    Parameters
    ----------
    raw_waveform : np.ndarray
        Waveform array with channel as the first axis.
    components : list[str] or None
        Optional component names to select before normalization.
    normalization_type : str
        Normalization mode. Supported values are ``"mean_std"`` and ``"max"``.

    Returns
    -------
    np.ndarray
        Normalized waveform as ``float32``.

    Raises
    ------
    ValueError
        If an unknown component or normalization type is provided.
    """
    if components is not None:
        unknown = [c for c in components if c not in COMP_DICT]
        if unknown:
            raise ValueError(f"Unknown components: {unknown}")
        if raw_waveform.shape[0] != len(components):
            idx = [COMP_DICT[c] for c in components]
            raw_waveform = raw_waveform[idx]

    raw_waveform = np.asarray(raw_waveform, dtype=np.float32)

    if normalization_type == "mean_std":
        mean = np.mean(raw_waveform, axis=1, keepdims=True)
        std = np.std(raw_waveform, axis=1, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        waveform = (raw_waveform - mean) / std
    elif normalization_type == "max":
        centered = raw_waveform - raw_waveform.mean(axis=1, keepdims=True)
        max_amp = np.max(np.abs(centered), axis=1, keepdims=True)
        max_amp = np.where(max_amp == 0, 1.0, max_amp)
        waveform = centered / max_amp
    else:
        raise ValueError(f"normalization_type '{normalization_type}' is not supported.")

    return waveform.astype(np.float32, copy=False)


def convert_sactraces_to_waveform(
    sac_traces: dict[str, SacTrace | None],
    components: list[str],
) -> np.ndarray:
    """Convert SAC traces into a waveform array.

    Parameters
    ----------
    sac_traces : dict[str, SacTrace]
        Mapping from component key to SAC trace.
    components : list[str]
        Component names to stack, such as ``["EW", "NS", "UD"]``.

    Returns
    -------
    np.ndarray
        Stacked waveform array.
    """
    missing = [
        component for component in components if sac_traces.get(component) is None
    ]
    if missing:
        raise ValueError(f"Missing SAC components: {missing}")

    waveform = np.array([sac_traces[component].data for component in components])
    return waveform
