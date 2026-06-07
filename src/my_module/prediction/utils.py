from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2


@dataclass
class HypocenterResult:
    """Flattened hypocenter prediction and uncertainty summary.

    Attributes
    ----------
    true_lat, true_lon, true_dep : float
        Reference hypocenter latitude, longitude, and depth.
    pred_lat, pred_lon, pred_dep : float
        Predicted hypocenter latitude, longitude, and depth.
    east_km, north_km, depth_km : float
        Relative prediction in local kilometers.
    east_std_km, north_std_km, depth_std_km : float
        Marginal standard deviations in kilometers.
    rho_en, rho_ed, rho_nd : float
        Pairwise correlations between local axes.
    volume_km3 : float
        Confidence ellipsoid volume in cubic kilometers.
    major_length_km : float
        Major-axis length of the confidence ellipsoid in kilometers.
    """

    true_lat: float
    true_lon: float
    true_dep: float
    pred_lat: float
    pred_lon: float
    pred_dep: float
    east_km: float
    north_km: float
    depth_km: float
    east_std_km: float
    north_std_km: float
    depth_std_km: float
    rho_en: float
    rho_ed: float
    rho_nd: float
    volume_km3: float
    major_length_km: float


def infer(
    model: torch.nn.Module,
    waveforms: torch.Tensor | ArrayLike,
    return_arrival: bool = False,
) -> dict[str, NDArray[np.float32] | None]:
    """Run model inference on waveform data.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to evaluate.
    waveforms : torch.Tensor or ArrayLike
        Waveform tensor with shape ``(channels, samples)`` or
        ``(batch, channels, samples)``.
    return_arrival : bool, default=False
        Whether to include arrival-time output when the model provides it.

    Returns
    -------
    dict[str, NDArray[np.float32] or None]
        Dictionary with ``hypo`` predictions and optional ``arrival_time``.
    """
    model.eval()
    device = next(model.parameters()).device

    if isinstance(waveforms, torch.Tensor):
        waveforms_tensor = waveforms.to(device=device, dtype=torch.float32)
    else:
        waveforms_tensor = torch.as_tensor(
            waveforms, dtype=torch.float32, device=device
        )

    if waveforms_tensor.ndim == 2:
        waveforms_tensor = waveforms_tensor.unsqueeze(0)

    with torch.inference_mode():
        pred = model(waveforms_tensor)

    if isinstance(pred, tuple):
        pred_hypo, pred_arrival = pred
    else:
        pred_hypo, pred_arrival = pred, None

    hypo_np = pred_hypo.cpu().numpy()
    arrival_np = (
        pred_arrival.cpu().numpy()
        if (return_arrival and pred_arrival is not None)
        else None
    )

    return {
        "hypo": hypo_np,
        "arrival_time": arrival_np,
    }


def compute_ellipse_params(sigma_km2: ArrayLike) -> tuple[float, float, float]:
    """Compute horizontal confidence ellipse parameters.

    Parameters
    ----------
    sigma_km2 : ArrayLike
        Two-dimensional covariance matrix in square kilometers.

    Returns
    -------
    tuple[float, float, float]
        Major-axis diameter, minor-axis diameter, and azimuth in degrees.
    """
    eigval, eigvec = np.linalg.eigh(sigma_km2)
    idx = eigval.argsort()[::-1]
    eigval = eigval[idx]
    eigvec = eigvec[:, idx]

    r95 = np.sqrt(chi2.ppf(0.95, 2))
    major_axis = 2 * r95 * np.sqrt(eigval[0])
    minor_axis = 2 * r95 * np.sqrt(eigval[1])

    v = eigvec[:, 0]
    azimuth_deg = np.degrees(np.arctan2(v[0], v[1])) % 360.0
    return major_axis, minor_axis, azimuth_deg


def compute_uncertainty(
    sigma_km2: ArrayLike,
    alpha: float = 0.95,
) -> tuple[NDArray[np.float64] | np.float64, NDArray[np.float64] | np.float64]:
    """Compute ellipsoid volume and major-axis length.

    Parameters
    ----------
    sigma_km2 : ArrayLike
        Three-dimensional covariance matrix or stack of matrices.
    alpha : float, default=0.95
        Confidence level.

    Returns
    -------
    tuple[NDArray[np.float64] or np.float64, NDArray[np.float64] or np.float64]
        Confidence volume and major-axis length.
    """
    C = chi2.ppf(alpha, 3)
    volume_km3 = (4.0 / 3.0) * np.pi * (C**1.5) * np.sqrt(np.linalg.det(sigma_km2))

    try:
        w = np.linalg.eigvalsh(sigma_km2)
        lam_max = w[..., -1]
    except Exception:
        lam_max = 1e30

    semi_major = np.sqrt(C * lam_max)
    major_length_km = 2.0 * semi_major
    return volume_km3, major_length_km


def flatten_prediction_row(
    true_lat: float,
    true_lon: float,
    true_dep: float,
    pred_hypo: NDArray[np.float64],
    mean_km: NDArray[np.float64],
    Sigma_km2: NDArray[np.float64],
) -> HypocenterResult:
    """Flatten a prediction and covariance into one result row.

    Parameters
    ----------
    true_lat, true_lon, true_dep : float
        Reference latitude, longitude, and depth.
    pred_hypo : NDArray[np.float64]
        Predicted ``[lat, lon, depth]``.
    mean_km : NDArray[np.float64]
        Relative local mean ``[east, north, depth]`` in kilometers.
    Sigma_km2 : NDArray[np.float64]
        Relative covariance matrix in square kilometers.

    Returns
    -------
    HypocenterResult
        Flattened prediction record.
    """
    std_km = np.sqrt(np.diag(Sigma_km2))

    denom_en = std_km[0] * std_km[1]
    denom_ed = std_km[0] * std_km[2]
    denom_nd = std_km[1] * std_km[2]

    rho_en = Sigma_km2[0, 1] / denom_en
    rho_ed = Sigma_km2[0, 2] / denom_ed
    rho_nd = Sigma_km2[1, 2] / denom_nd

    volume_km3, major_length_km = compute_uncertainty(Sigma_km2)

    return HypocenterResult(
        true_lat=float(true_lat),
        true_lon=float(true_lon),
        true_dep=float(true_dep),
        pred_lat=round(float(pred_hypo[0]), 4),
        pred_lon=round(float(pred_hypo[1]), 3),
        pred_dep=round(float(pred_hypo[2]), 3),
        east_km=float(mean_km[0]),
        north_km=float(mean_km[1]),
        depth_km=float(mean_km[2]),
        east_std_km=float(std_km[0]),
        north_std_km=float(std_km[1]),
        depth_std_km=float(std_km[2]),
        rho_en=float(rho_en),
        rho_ed=float(rho_ed),
        rho_nd=float(rho_nd),
        volume_km3=round(float(volume_km3), 3),
        major_length_km=round(float(major_length_km), 3),
    )
