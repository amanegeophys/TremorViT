import networkx as nx
import numpy as np
from numpy.typing import NDArray

from .ci import relative_to_ecef


def extract_kcore_components(G: nx.Graph, k: int = 2) -> list[list[int]]:
    """Extract connected components from the k-core of a graph.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    k : int, default=2
        Core number used by :func:`networkx.k_core`.

    Returns
    -------
    list[list[int]]
        Node lists for each connected component in the k-core.
    """
    Gc = nx.k_core(G, k=k)
    return [list(c) for c in nx.connected_components(Gc)]


def get_mahalanobis_matrix(
    sta_lat: NDArray[np.float64],
    sta_lon: NDArray[np.float64],
    east_km: NDArray[np.float64],
    north_km: NDArray[np.float64],
    depth_km: NDArray[np.float64],
    sigma_km2: NDArray[np.float64],
    c_thr: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], nx.Graph]:
    """Build a graph from pairwise Mahalanobis consistency.

    Parameters
    ----------
    sta_lat, sta_lon : NDArray[np.float64]
        Station latitude and longitude arrays.
    east_km, north_km, depth_km : NDArray[np.float64]
        Local relative hypocenter predictions in kilometers.
    sigma_km2 : NDArray[np.float64]
        Local covariance matrices in square kilometers.
    c_thr : float
        Squared Mahalanobis-distance threshold for adding edges.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64], nx.Graph]
        ECEF means, ECEF covariance matrices, and the consistency graph.
    """
    src_ecef, sigma_ecef = relative_to_ecef(
        sta_lat=sta_lat,
        sta_lon=sta_lon,
        east_km=east_km,
        north_km=north_km,
        depth_km=depth_km,
        sigma_km2=sigma_km2,
    )

    mu = src_ecef
    Sigma = sigma_ecef
    N = mu.shape[0]

    G = nx.Graph()
    G.add_nodes_from(range(N))

    for i in range(N - 1):
        D = mu[i] - mu[i + 1 :]  # (M,3)
        S = Sigma[i] + Sigma[i + 1 :]  # (M,3,3)

        L = np.linalg.cholesky(S)
        y = np.linalg.solve(L, D[..., None])  # (M,3,1)
        d2 = np.sum(y[..., 0] ** 2, axis=1)  # (M,)

        js = np.where(d2 <= c_thr)[0] + (i + 1)
        for j in js:
            G.add_edge(i, int(j))

    return src_ecef, sigma_ecef, G
