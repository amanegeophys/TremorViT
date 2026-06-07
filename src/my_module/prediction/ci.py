import numpy as np
import numpy.linalg as npl
from numpy.typing import ArrayLike, NDArray
from pyproj import Transformer
from scipy.optimize import Bounds, LinearConstraint, minimize, minimize_scalar

LLA_TO_ECEF = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
ECEF_TO_LLA = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)


FloatArray = NDArray[np.float64]


def enu_to_ecef_matrix(lat: ArrayLike, lon: ArrayLike) -> FloatArray:
    """Build local ENU-to-ECEF rotation matrices.

    Parameters
    ----------
    lat, lon : ArrayLike
        Latitude and longitude in degrees.

    Returns
    -------
    FloatArray
        Rotation matrices with shape ``(N, 3, 3)``.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    if lat.shape != lon.shape:
        raise ValueError(
            f"lat and lon must have same shape, got {lat.shape}, {lon.shape}"
        )

    n = lat.size
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    slat, clat = np.sin(lat_rad), np.cos(lat_rad)
    slon, clon = np.sin(lon_rad), np.cos(lon_rad)

    R = np.empty((n, 3, 3), dtype=np.float64)

    # East (col 0)
    R[:, 0, 0] = -slon
    R[:, 1, 0] = clon
    R[:, 2, 0] = 0.0

    # North (col 1)
    R[:, 0, 1] = -slat * clon
    R[:, 1, 1] = -slat * slon
    R[:, 2, 1] = clat

    # Up (col 2)
    R[:, 0, 2] = clat * clon
    R[:, 1, 2] = clat * slon
    R[:, 2, 2] = slat

    return R


def relative_to_ecef(
    sta_lat: ArrayLike,
    sta_lon: ArrayLike,
    east_km: ArrayLike,
    north_km: ArrayLike,
    depth_km: ArrayLike,
    sigma_km2: ArrayLike,
    lla_to_ecef: Transformer = LLA_TO_ECEF,
) -> tuple[FloatArray, FloatArray]:
    """Convert local relative END coordinates to ECEF coordinates.

    Parameters
    ----------
    sta_lat, sta_lon : ArrayLike
        Station latitude and longitude in degrees.
    east_km, north_km, depth_km : ArrayLike
        Local relative offsets in kilometers, with depth positive downward.
    sigma_km2 : ArrayLike
        Local END covariance matrix or stack of matrices in square kilometers.
    lla_to_ecef : Transformer, default=LLA_TO_ECEF
        Coordinate transformer from geodetic coordinates to ECEF.

    Returns
    -------
    tuple[FloatArray, FloatArray]
        Source ECEF coordinates and ECEF covariance matrices.
    """
    sta_lat = np.asarray(sta_lat, dtype=np.float64)
    sta_lon = np.asarray(sta_lon, dtype=np.float64)
    east_km = np.asarray(east_km, dtype=np.float64)
    north_km = np.asarray(north_km, dtype=np.float64)
    depth_km = np.asarray(depth_km, dtype=np.float64)
    sigma_km2 = np.asarray(sigma_km2, dtype=np.float64)

    sta_lat = np.atleast_1d(sta_lat)
    sta_lon = np.atleast_1d(sta_lon)
    east_km = np.atleast_1d(east_km)
    north_km = np.atleast_1d(north_km)
    depth_km = np.atleast_1d(depth_km)

    N = sta_lat.size
    if not (sta_lon.size == east_km.size == north_km.size == depth_km.size == N):
        raise ValueError("All input vectors must have same length.")

    if sigma_km2.ndim == 2:
        sigma_km2 = np.broadcast_to(sigma_km2, (N, 3, 3))
    elif sigma_km2.shape[0] != N:
        raise ValueError(
            f"sigma_km2.shape[0]={sigma_km2.shape[0]} does not match N={N}"
        )

    # ----- station LLA -> ECEF -----
    h0 = np.zeros_like(sta_lon, dtype=np.float64)
    X, Y, Z = lla_to_ecef.transform(sta_lon, sta_lat, h0)
    sta_ecef = np.stack([X, Y, Z], axis=-1)  # (N, 3)

    R = enu_to_ecef_matrix(sta_lat, sta_lon)  # (N, 3, 3)

    scale = np.array([1000.0, 1000.0, -1000.0], dtype=np.float64)
    T = R * scale[None, None, :]  # (N, 3, 3)

    enu_km = np.stack([east_km, north_km, depth_km], axis=-1)  # (N, 3)
    d_ecef = np.einsum("nij,nj->ni", T, enu_km, optimize=True)  # (N, 3)
    src_ecef = sta_ecef + d_ecef  # (N, 3)

    sigma_km2 = 0.5 * (sigma_km2 + np.swapaxes(sigma_km2, -1, -2))
    sigma_ecef = np.einsum(
        "nij,njk,nlk->nil", T, sigma_km2, T, optimize=True
    )  # (N, 3, 3)
    sigma_ecef = 0.5 * (sigma_ecef + np.swapaxes(sigma_ecef, -1, -2))

    return src_ecef, sigma_ecef


def ecef_to_geo_and_sigma(
    src_ecef: ArrayLike,
    sigma_ecef: ArrayLike,
    ecef_to_lla: Transformer = ECEF_TO_LLA,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Convert ECEF coordinates and covariance to geodetic END values.

    Parameters
    ----------
    src_ecef : ArrayLike
        Source ECEF coordinates with shape ``(N, 3)`` or ``(3,)``.
    sigma_ecef : ArrayLike
        ECEF covariance with shape ``(N, 3, 3)`` or ``(3, 3)``.
    ecef_to_lla : Transformer, default=ECEF_TO_LLA
        Coordinate transformer from ECEF to geodetic coordinates.

    Returns
    -------
    tuple[FloatArray, FloatArray, FloatArray, FloatArray]
        Source latitude, longitude, depth in kilometers, and local END
        covariance matrices in square kilometers.
    """
    src_ecef = np.asarray(src_ecef, dtype=np.float64)
    sigma_ecef = np.asarray(sigma_ecef, dtype=np.float64)

    if src_ecef.ndim == 1:
        src_ecef = src_ecef[None, :]  # (1, 3)
    if sigma_ecef.ndim == 2:
        sigma_ecef = sigma_ecef[None, :, :]  # (1, 3, 3)

    N = src_ecef.shape[0]
    if sigma_ecef.shape[0] != N:
        raise ValueError(
            f"sigma_ecef.shape[0]={sigma_ecef.shape[0]} does not match N={N}"
        )

    lon_src, lat_src, h_src_m = ecef_to_lla.transform(
        src_ecef[:, 0], src_ecef[:, 1], src_ecef[:, 2]
    )
    depth_src_km = -h_src_m / 1000.0

    R_src = enu_to_ecef_matrix(lat_src, lon_src)  # (N, 3, 3)
    R_T = np.swapaxes(R_src, -1, -2)

    Sigma_enu_m2 = np.einsum(
        "nij,njk,nkl->nil", R_T, sigma_ecef, R_src, optimize=True
    )  # (N, 3, 3)

    S_km = np.array([1e-3, 1e-3, -1e-3], dtype=np.float64)
    Sigma_END_km2 = Sigma_enu_m2 * S_km[None, :, None] * S_km[None, None, :]
    Sigma_END_km2 = 0.5 * (Sigma_END_km2 + np.swapaxes(Sigma_END_km2, -1, -2))

    return lat_src, lon_src, depth_src_km, Sigma_END_km2


def _logdet_from_chol(A: FloatArray) -> tuple[float, FloatArray]:
    """Compute log determinant and Cholesky factor.

    Parameters
    ----------
    A : FloatArray
        Symmetric positive-definite matrix.

    Returns
    -------
    tuple[float, FloatArray]
        Log determinant of ``A`` and its Cholesky factor.
    """
    L = npl.cholesky(A)
    return 2.0 * np.log(np.diag(L)).sum(), L


def ci_fuse_pair_scipy(
    x1: ArrayLike,
    P1: ArrayLike,
    x2: ArrayLike,
    P2: ArrayLike,
    objective: str = "logdet",
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Fuse two Gaussian estimates with covariance intersection.

    Parameters
    ----------
    x1, x2 : ArrayLike
        Mean vectors.
    P1, P2 : ArrayLike
        Covariance matrices.
    objective : {"logdet", "trace"}, default="logdet"
        Objective minimized during weight search.

    Returns
    -------
    tuple[FloatArray, FloatArray, FloatArray]
        Fused mean, covariance, and CI weights.
    """
    P1 = np.asarray(P1, dtype=np.float64)
    P2 = np.asarray(P2, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)

    J1 = npl.inv(P1)
    J2 = npl.inv(P2)

    def f(w: float) -> float:
        """Evaluate the scalar covariance-intersection objective."""
        Jsum = w * J1 + (1.0 - w) * J2

        if objective == "logdet":
            logdet, _ = _logdet_from_chol(Jsum)
            return -logdet
        elif objective == "trace":
            X = npl.solve(Jsum, np.eye(3))
            return np.trace(X)
        else:
            raise ValueError("objective must be 'logdet' or 'trace'")

    res = minimize_scalar(f, bounds=(0.0, 1.0), method="bounded")
    w = float(np.clip(res.x, 0.0, 1.0))

    Jsum = w * J1 + (1.0 - w) * J2
    Pc = npl.solve(Jsum, np.eye(3))
    Pc = 0.5 * (Pc + Pc.T)

    rhs = w * (J1 @ x1) + (1.0 - w) * (J2 @ x2)
    c = npl.solve(Jsum, rhs)

    return c, Pc, np.array([w, 1.0 - w])


def ci_fuse_many_sequential_scipy(
    means: ArrayLike,
    covs: ArrayLike,
    objective: str = "logdet",
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Sequentially fuse multiple Gaussian estimates.

    Parameters
    ----------
    means : ArrayLike
        Mean vectors with shape ``(M, 3)``.
    covs : ArrayLike
        Covariance matrices with shape ``(M, 3, 3)``.
    objective : {"logdet", "trace"}, default="logdet"
        Objective minimized during pairwise fusion.

    Returns
    -------
    tuple[FloatArray, FloatArray, FloatArray]
        Fused mean, covariance, and intermediate pairwise weights.
    """
    means = np.asarray(means, dtype=np.float64)
    covs = np.asarray(covs, dtype=np.float64)

    x, P = means[0], covs[0]
    ws = []
    for i in range(1, means.shape[0]):
        xi, Pi, wi = ci_fuse_pair_scipy(x, P, means[i], covs[i], objective=objective)
        x, P = xi, Pi
        ws.append(wi[0])
    return x, P, np.asarray(ws)


def ci_fuse_scipy(
    means: ArrayLike,
    covs: ArrayLike,
    objective: str = "logdet",
    verbose: bool = False,
) -> tuple[FloatArray, FloatArray, FloatArray | None]:
    """Fuse multiple Gaussian estimates with covariance intersection.

    Parameters
    ----------
    means : ArrayLike
        Mean vectors with shape ``(M, 3)``.
    covs : ArrayLike
        Covariance matrices with shape ``(M, 3, 3)``.
    objective : {"logdet", "trace"}, default="logdet"
        Objective minimized by SLSQP.
    verbose : bool, default=False
        Whether to print SLSQP diagnostics.

    Returns
    -------
    tuple[FloatArray, FloatArray, FloatArray or None]
        Fused mean, covariance, and optimized weights. Weights are ``None``
        when optimization fails and sequential fusion is used.
    """
    means = np.asarray(means, dtype=np.float64)
    covs = np.asarray(covs, dtype=np.float64)

    m = means.shape[0]
    if m == 2:
        return ci_fuse_pair_scipy(
            means[0], covs[0], means[1], covs[1], objective=objective
        )

    J = npl.inv(covs)  # (M, 3, 3)
    Jmu = np.einsum("mij,mj->mi", J, means)

    def obj(w: FloatArray) -> float:
        """Evaluate the multi-estimate covariance-intersection objective."""
        Jsum = (w[:, None, None] * J).sum(axis=0)
        try:
            if objective == "logdet":
                logdet, _ = _logdet_from_chol(Jsum)
                return -logdet
            elif objective == "trace":
                X = npl.solve(Jsum, np.eye(3))
                return np.trace(X)
            else:
                raise ValueError("objective must be 'logdet' or 'trace'")
        except npl.LinAlgError:
            return 1e30

    lc = LinearConstraint(np.ones((1, m)), 1.0, 1.0)
    bounds = Bounds(0.0, 1.0)

    w0 = np.trace(J, axis1=1, axis2=2)
    w0 = w0 / w0.sum()

    res = minimize(
        obj,
        w0,
        method="SLSQP",
        constraints=[lc],
        bounds=bounds,
        options=dict(maxiter=500, ftol=1e-12, disp=verbose),
    )

    if not res.success:
        print("CI optimization failed, falling back to sequential fusion.")
        c_fb, Pc_fb, _ = ci_fuse_many_sequential_scipy(means, covs, objective=objective)
        return c_fb, Pc_fb, None

    w = res.x
    Jsum = (w[:, None, None] * J).sum(axis=0)
    Pc = npl.solve(Jsum, np.eye(3))
    Pc = 0.5 * (Pc + Pc.T)
    rhs = (w[:, None] * Jmu).sum(axis=0)
    c = npl.solve(Jsum, rhs)

    return c, Pc, w
