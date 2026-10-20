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


def relative_to_geographic(
    east_km: np.ndarray | float,
    north_km: np.ndarray | float,
    station_lon: np.ndarray | float,
    station_lat: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert the surface geodesic E/N coordinates used by the training labels.

    Parameters
    ----------
    east_km, north_km : np.ndarray or float
        Surface geodesic east and north coordinates in kilometers.
    station_lon, station_lat : np.ndarray or float
        Station longitude and latitude in degrees.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Source longitude and latitude in degrees.
    """
    distance_m = np.hypot(east_km, north_km) * 1000.0
    azimuth_deg = np.rad2deg(np.arctan2(east_km, north_km))
    lon, lat, _ = geod.fwd(station_lon, station_lat, azimuth_deg, distance_m)
    return np.asarray(lon), np.asarray(lat)


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

    event_lon, event_lat = relative_to_geographic(
        east_km, north_km, station_lon, station_lat,
    )

    pred_hypo = np.array([event_lat, event_lon, depth_km], dtype=float)

    return pred_hypo, mean_km, Sigma_km2


def covariance_at_epicenter(
    mean_km: np.ndarray,
    covariance_km2: np.ndarray,
    station_loc: tuple[float, float],
) -> np.ndarray:
    """Linearize station geodesic E/N(/depth) covariance at the predicted epicenter.

    ``station_loc`` is (latitude, longitude). Input and output use km²;
    depth remains positive downward. This is a direct first-order transform
    for plotting, not a change to the stored prediction or CI calculation.
    """
    mean_km = np.asarray(mean_km, dtype=float)
    covariance_km2 = np.asarray(covariance_km2, dtype=float)
    if covariance_km2.shape not in ((2, 2), (3, 3)):
        raise ValueError("Expected covariance shape (2, 2) or (3, 3)")
    station_lat, station_lon = station_loc
    center_lon, center_lat = relative_to_geographic(
        mean_km[0], mean_km[1], station_lon, station_lat,
    )
    # Central differences over 1 m, directly into epicenter-centered E/N km.
    step_km = 1e-3
    offsets = mean_km[:2] + step_km * np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    lon, lat = relative_to_geographic(
        offsets[:, 0], offsets[:, 1], np.full(4, station_lon), np.full(4, station_lat),
    )
    azimuth, _, distance_m = geod.inv(
        np.full(4, center_lon), np.full(4, center_lat), lon, lat,
    )
    azimuth = np.deg2rad(azimuth)
    local_km = (np.asarray(distance_m) / 1000.)[:, None] * np.column_stack(
        (np.sin(azimuth), np.cos(azimuth)),
    )
    jacobian = np.eye(covariance_km2.shape[0])
    jacobian[:2, :2] = ((local_km[::2] - local_km[1::2]) / (2 * step_km)).T
    covariance = jacobian @ covariance_km2 @ jacobian.T
    return 0.5 * (covariance + covariance.T)
