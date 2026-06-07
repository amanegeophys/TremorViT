import sys
from pathlib import Path

import numpy as np
from pyproj import Geod

ROOT = Path(__file__).resolve().parents[1]  # project root
sys.path.append(str(ROOT / "src"))

geod = Geod(ellps="WGS84")


def softplus_stable(x: np.ndarray | float) -> np.ndarray | float:
    """Compute a numerically stable softplus.

    Parameters
    ----------
    x : np.ndarray or float
        Input value or array.

    Returns
    -------
    np.ndarray or float
        Softplus-transformed value.
    """
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def convert_arrival_time(
    prediction: np.ndarray, win_sec: float = 60.0, eps: float = 1e-4
) -> tuple[float, float]:
    """Convert a one-dimensional Gaussian arrival prediction to seconds.

    Parameters
    ----------
    prediction : np.ndarray
        Model output containing normalized mean and raw scale.
    win_sec : float, default=60.0
        Window length in seconds.
    eps : float, default=1e-4
        Small positive floor added to the standard deviation.

    Returns
    -------
    tuple[float, float]
        Predicted arrival mean and standard deviation in seconds from the
        window start.
    """
    prediction = np.asarray(prediction, dtype=float)
    mean_norm = float(np.tanh(prediction[0]))
    std_norm = float(softplus_stable(prediction[1]) + eps)
    half_win_sec = win_sec / 2.0
    pred_arrival_mean = (mean_norm + 1.0) * half_win_sec
    pred_arrival_std = std_norm * half_win_sec
    return pred_arrival_mean, pred_arrival_std


def convert_relative_to_geo(
    prediction: np.ndarray,
    station_loc: tuple[float, float],
    scale_km: float = 50.0,
    eps: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert relative hypocenter output to geographic coordinates.

    Parameters
    ----------
    prediction : np.ndarray
        Model output with shape ``(9,)`` containing
        ``[east, north, depth, l00, l11, l22, l10, l20, l21]``.
    station_loc : tuple[float, float]
        Station latitude and longitude in degrees.
    scale_km : float, default=50.0
        Scale factor from model units to kilometers.
    eps : float, default=1e-4
        Small positive floor added to Cholesky diagonal terms.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Geographic prediction ``[lat, lon, depth]``, relative mean in
        kilometers, and covariance in square kilometers.
    """
    prediction = np.asarray(prediction, dtype=float)
    s = float(scale_km)

    east_km = float(prediction[0]) * s
    north_km = float(prediction[1]) * s
    depth_km = float(prediction[2]) * s
    mean_km = np.array([east_km, north_km, depth_km], dtype=float)

    l00 = prediction[3]
    l11 = prediction[4]
    l22 = prediction[5]
    l10 = prediction[6]
    l20 = prediction[7]
    l21 = prediction[8]

    L_s = np.zeros((3, 3), dtype=float)
    L_s[0, 0] = softplus_stable(l00) + eps
    L_s[1, 0] = l10
    L_s[1, 1] = softplus_stable(l11) + eps
    L_s[2, 0] = l20
    L_s[2, 1] = l21
    L_s[2, 2] = softplus_stable(l22) + eps

    L_km = (s * np.eye(3)) @ L_s
    Sigma_km2 = L_km @ L_km.T
    Sigma_km2 = 0.5 * (Sigma_km2 + Sigma_km2.T)

    station_lat, station_lon = station_loc

    dist_m = float(np.hypot(east_km, north_km) * 1000.0)
    azimuth_deg = float(np.degrees(np.arctan2(east_km, north_km)))

    event_lon, event_lat, _ = geod.fwd(
        station_lon,
        station_lat,
        azimuth_deg,
        dist_m,
    )

    pred_hypo = np.array([event_lat, event_lon, depth_km], dtype=float)

    return pred_hypo, mean_km, Sigma_km2
