"""Fuse station-level hypocenter predictions into event-level estimates."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, TypedDict

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from numpy.typing import NDArray
from pyproj import Geod
from scipy.stats import chi2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))


from my_module.prediction import (
    ci_fuse_scipy,
    compute_uncertainty,
    ecef_to_geo_and_sigma,
    extract_kcore_components,
    get_mahalanobis_matrix,
)


FloatArray = NDArray[np.float64]


class FusedRow(TypedDict, total=False):
    """Output row for a fused hypocenter estimate."""

    origin_time: str
    lat: float
    lon: float
    dep: float
    n_in_comp: int
    stations: str
    sigma11: float
    sigma12: float
    sigma13: float
    sigma21: float
    sigma22: float
    sigma23: float
    sigma31: float
    sigma32: float
    sigma33: float
    volume_km3: float
    major_length_km: float
    k_core_k: int
    uncertainty_thr_single: float
    tremor_proba_thr: float
    epicentral_train_max_km: float
    n_candidates: int
    n_selected_sources: int | None
    n_conflict_group: int
    n_other_cluster_stations_within_50km: int
    has_other_cluster_station_within_50km: bool
    selected_by: str
    source_id: int
    possible_multiple_sources: bool


class Candidate(TypedDict):
    """Candidate fused source before conflict resolution."""

    comp: NDArray[np.int_]
    row: FusedRow
    fused_lat: float
    fused_lon: float
    fused_dep: float
    volume_km3: float
    major_length_km: float
    n_in_comp: int
    n_other_cluster_stations_within_50km: int
    has_other_cluster_station_within_50km: bool


def parse_dt(s: str) -> datetime:
    """Parse a timestamp string.

    Parameters
    ----------
    s : str
        Timestamp formatted as ``"%Y-%m-%d-%H:%M:%S.%f"``.

    Returns
    -------
    datetime
        Parsed datetime.
    """
    return datetime.strptime(s, "%Y-%m-%d-%H:%M:%S.%f")


def fmt_dt(dt: datetime) -> str:
    """Format a datetime with millisecond precision.

    Parameters
    ----------
    dt : datetime
        Datetime to format.

    Returns
    -------
    str
        Formatted timestamp.
    """
    return dt.strftime("%Y-%m-%d-%H:%M:%S.%f")[:-3]


def month_start(dt: datetime) -> datetime:
    """Return midnight on the first day of the same calendar month.

    Parameters
    ----------
    dt : datetime
        Datetime to normalize.

    Returns
    -------
    datetime
        Datetime on the first day of the same month at midnight.
    """
    return datetime(dt.year, dt.month, 1, 0, 0, 0, 0)


def next_month(dt: datetime) -> datetime:
    """Advance a datetime by one calendar month.

    Parameters
    ----------
    dt : datetime
        Input datetime.

    Returns
    -------
    datetime
        Datetime one month later.
    """
    dt = dt + relativedelta(months=1)
    return dt


def month_end(dt: datetime) -> datetime:
    """Return the final second of the month containing a datetime.

    Parameters
    ----------
    dt : datetime
        Datetime within the target month.

    Returns
    -------
    datetime
        Month end datetime.
    """
    nm = next_month(month_start(dt))
    return nm - timedelta(seconds=1)


def iter_months(start: datetime, end: datetime) -> Iterator[datetime]:
    """Iterate over monthly-spaced datetimes in a date range.

    Parameters
    ----------
    start, end : datetime
        Inclusive datetime range.

    Yields
    ------
    datetime
        Datetimes advanced by calendar month.
    """
    cur = month_start(start)
    end_m = month_start(end)
    while cur <= end_m:
        yield cur
        cur = next_month(cur)


def build_month_csv_path(org_dir: Path, any_dt_in_month: datetime) -> Path:
    """Build the monthly input CSV path for a datetime.

    Parameters
    ----------
    org_dir : Path
        Directory containing monthly prediction CSV files.
    any_dt_in_month : datetime
        Datetime inside the target month.

    Returns
    -------
    Path
        Expected monthly CSV path.
    """
    st = month_start(any_dt_in_month)
    ed = month_end(any_dt_in_month)
    return org_dir / f"{fmt_dt(st)}_{fmt_dt(ed)}.csv"


def build_range_csv_path(org_dir: Path, start: datetime, end: datetime) -> Path:
    """Build the prediction CSV path for an exact requested range."""
    return org_dir / f"{fmt_dt(start)}_{fmt_dt(end)}.csv"


def remove_isolated(
    df: pd.DataFrame,
    geod: Geod,
    time_h: int = 12,
    dist_km: float = 5.0,
) -> pd.DataFrame:
    """Remove fused events that are isolated in space and time.

    Parameters
    ----------
    df : pd.DataFrame
        Fused hypocenter table.
    geod : Geod
        Geodesic calculator.
    time_h : int, default=12
        Half-window in hours for neighbor search.
    dist_km : float, default=5.0
        Maximum epicentral distance for a neighboring event.

    Returns
    -------
    pd.DataFrame
        Filtered hypocenter table.
    """
    out = df.copy()

    out["origin_time"] = pd.to_datetime(
        out["origin_time"], format="%Y-%m-%d-%H:%M:%S.%f"
    )

    out = out.sort_values("origin_time").reset_index(drop=False)
    t = out["origin_time"].to_numpy()

    keep_idx: list[int] = []
    dt = np.timedelta64(time_h, "h")

    for i in tqdm(range(len(out)), total=len(out)):
        base_time = t[i]
        base_lat = float(out.loc[i, "lat"])
        base_lon = float(out.loc[i, "lon"])

        lo = np.searchsorted(t, base_time - dt, side="left")
        hi = np.searchsorted(t, base_time + dt, side="right")

        cand = out.iloc[lo:hi]
        if len(cand) <= 1:
            continue

        n = len(cand)
        lons0 = np.full(n, base_lon, dtype=float)
        lats0 = np.full(n, base_lat, dtype=float)
        _, _, dist_m = geod.inv(
            lons0, lats0, cand["lon"].to_numpy(), cand["lat"].to_numpy()
        )
        dist = dist_m / 1000.0

        has_neighbor = np.any((dist < dist_km) & (dist > 0.0))
        if has_neighbor:
            keep_idx.append(int(out.loc[i, "index"]))

    out2 = df.loc[keep_idx].copy()
    out2 = out2.sort_values("origin_time")
    return out2


def calc_station_distance_km(
    geod: Geod,
    src_lat: float,
    src_lon: float,
    sta_lat_arr: NDArray[np.float64],
    sta_lon_arr: NDArray[np.float64],
) -> FloatArray:
    """Calculate epicentral distances from a source to stations.

    Parameters
    ----------
    geod : Geod
        Geodesic calculator.
    src_lat, src_lon : float
        Source latitude and longitude in degrees.
    sta_lat_arr, sta_lon_arr : NDArray[np.float64]
        Station latitude and longitude arrays in degrees.

    Returns
    -------
    FloatArray
        Distances in kilometers.
    """
    sta_lat_arr = np.asarray(sta_lat_arr, dtype=float)
    sta_lon_arr = np.asarray(sta_lon_arr, dtype=float)

    _, _, d_m = geod.inv(
        np.full_like(sta_lon_arr, float(src_lon), dtype=float),
        np.full_like(sta_lat_arr, float(src_lat), dtype=float),
        sta_lon_arr,
        sta_lat_arr,
    )
    return d_m / 1000.0


def main() -> None:
    """Fuse station-level hypocenter predictions into event-level sources."""
    C95 = chi2.ppf(0.95, 3)
    geod = Geod(ellps="WGS84")

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument(
        "--org_dir",
        type=str,
        default="reports/hypocenter/org",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="reports/hypocenter",
    )
    parser.add_argument("--uncertainty_thr", type=float, default=7281 * 2)
    parser.add_argument("--uncertainty_mode", type=str, default="volume")
    parser.add_argument("--epicentral_train_max_km", type=float, default=50.0)
    parser.add_argument("--tremor_proba_thr", type=float, default=0.9)
    parser.add_argument("--k", type=int, default=2)
    args = parser.parse_args()

    start = parse_dt(args.start)
    end = parse_dt(args.end)

    org_dir = Path(args.org_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    usecols = [
        "origin_time",
        "station_lat",
        "station_lon",
        "east_km",
        "north_km",
        "depth_km",
        "sigma11",
        "sigma12",
        "sigma13",
        "sigma21",
        "sigma22",
        "sigma23",
        "sigma31",
        "sigma32",
        "sigma33",
        "tremor_proba",
        "station",
        "pred_lat",
        "pred_lon",
    ]

    parts: list[pd.DataFrame] = []
    exact_path = build_range_csv_path(org_dir, start, end)
    if exact_path.exists():
        df = pd.read_csv(exact_path, usecols=usecols)
        print(f"[read] {exact_path.name}, {len(df)}")
        parts.append(df)
    else:
        for m in iter_months(start, end):
            p = build_month_csv_path(org_dir, m)
            if not p.exists():
                print(f"[skip] missing: {p}")
                continue
            df = pd.read_csv(p, usecols=usecols)
            print(f"[read] {p.name}, {len(df)}")
            parts.append(df)

    if not parts:
        raise RuntimeError(
            "No prediction CSVs were found. Expected either "
            f"{exact_path} or monthly CSVs in the given range."
        )
    pdf = pd.concat(parts, ignore_index=True)

    pdf["origin_time"] = pd.to_datetime(
        pdf["origin_time"], format="%Y-%m-%d-%H:%M:%S.%f"
    )
    pdf = pdf[pdf["origin_time"].between(start, end)].copy()

    sig_cols = [
        "sigma11",
        "sigma12",
        "sigma13",
        "sigma21",
        "sigma22",
        "sigma23",
        "sigma31",
        "sigma32",
        "sigma33",
    ]

    volume_km3, major_length_km = compute_uncertainty(
        pdf[sig_cols].to_numpy().reshape(-1, 3, 3).astype(np.float64)
    )
    pdf["volume_km3"] = volume_km3
    pdf["major_length_km"] = major_length_km

    if args.uncertainty_mode == "volume":
        pdf = pdf[pdf["volume_km3"] <= args.uncertainty_thr]
    elif args.uncertainty_mode == "length":
        pdf = pdf[pdf["major_length_km"] <= args.uncertainty_thr]

    pdf = pdf[pdf["tremor_proba"] >= args.tremor_proba_thr]
    pdf = pdf.sort_values("origin_time")

    out_rows: list[FusedRow] = []
    n_groups = 0
    n_used = 0
    n_comps_total = 0
    comp_sizes = []

    pdf_g = pdf.groupby("origin_time", sort=False)
    for origin_time, g in tqdm(pdf_g, total=len(pdf_g)):
        n_groups += 1
        g = g.reset_index(drop=True)
        if len(g) < 3:
            continue

        n_used += 1

        src_ecef, sigma_ecef, G = get_mahalanobis_matrix(
            sta_lat=g["station_lat"].to_numpy(),
            sta_lon=g["station_lon"].to_numpy(),
            east_km=g["east_km"].to_numpy(),
            north_km=g["north_km"].to_numpy(),
            depth_km=g["depth_km"].to_numpy(),
            sigma_km2=g[sig_cols].to_numpy().reshape(-1, 3, 3),
            c_thr=C95,
        )

        comps = extract_kcore_components(G, k=args.k)
        n_comps_total += len(comps)
        comp_sizes.extend([len(c) for c in comps])

        candidates: list[Candidate] = []

        for comp in comps:
            idx = np.asarray(comp, dtype=int)

            src_ecef_comp = src_ecef[idx]
            sigma_ecef_comp = sigma_ecef[idx]

            c_ecef, Pc_ecef, _ = ci_fuse_scipy(
                src_ecef_comp,
                sigma_ecef_comp,
                objective="logdet",
                verbose=False,
            )  # m

            lat, lon, dep, Sigma_END_km2 = ecef_to_geo_and_sigma(
                c_ecef[None, :],
                Pc_ecef[None, :, :],
            )

            sig_km2 = Sigma_END_km2[0]  # (3,3) km^2

            volume_km3, major_length_km = compute_uncertainty(sig_km2)

            volume_km3 = float(np.asarray(volume_km3).squeeze())
            major_length_km = float(np.asarray(major_length_km).squeeze())

            stations = g.loc[idx, "station"].astype(str).tolist()
            station_str = ";".join(stations)

            row: FusedRow = {
                "origin_time": origin_time.strftime("%Y-%m-%d-%H:%M:%S.%f"),
                "lat": round(float(lat[0]), 4),
                "lon": round(float(lon[0]), 3),
                "dep": round(float(dep[0]), 3),
                "n_in_comp": int(len(idx)),
                "stations": station_str,
                "sigma11": float(sig_km2[0, 0]),
                "sigma12": float(sig_km2[0, 1]),
                "sigma13": float(sig_km2[0, 2]),
                "sigma21": float(sig_km2[1, 0]),
                "sigma22": float(sig_km2[1, 1]),
                "sigma23": float(sig_km2[1, 2]),
                "sigma31": float(sig_km2[2, 0]),
                "sigma32": float(sig_km2[2, 1]),
                "sigma33": float(sig_km2[2, 2]),
                "volume_km3": round(volume_km3, 3),
                "major_length_km": round(major_length_km, 3),
                "k_core_k": int(args.k),
                "uncertainty_thr_single": float(args.uncertainty_thr),
                "tremor_proba_thr": float(args.tremor_proba_thr),
                "epicentral_train_max_km": float(args.epicentral_train_max_km),
            }

            candidates.append(
                {
                    "comp": idx,
                    "row": row,
                    "fused_lat": float(lat[0]),
                    "fused_lon": float(lon[0]),
                    "fused_dep": float(dep[0]),
                    "volume_km3": volume_km3,
                    "major_length_km": major_length_km,
                    "n_in_comp": int(len(idx)),
                    "n_other_cluster_stations_within_50km": 0,
                    "has_other_cluster_station_within_50km": False,
                }
            )

        # ============================================================
        # Resolve multiple candidate sources using station-distance criterion.
        #
        # If source i has stations from cluster j within 50 km,
        # candidates i and j are regarded as conflicting.
        #
        # Non-conflicting candidates are retained as multiple sources.
        # Conflicting candidates are reduced to one representative.
        # ============================================================

        n_cand = len(candidates)
        conflict_graph: dict[int, set[int]] = {i: set() for i in range(n_cand)}

        for i, cand_i in enumerate(candidates):
            for j, cand_j in enumerate(candidates):
                if i >= j:
                    continue

                comp_i = cand_i["comp"]
                comp_j = cand_j["comp"]

                # Are stations of cluster j within 50 km of source i?
                d_i_to_sta_j_km = calc_station_distance_km(
                    geod=geod,
                    src_lat=cand_i["fused_lat"],
                    src_lon=cand_i["fused_lon"],
                    sta_lat_arr=g.loc[comp_j, "station_lat"].to_numpy(),
                    sta_lon_arr=g.loc[comp_j, "station_lon"].to_numpy(),
                )

                # Are stations of cluster i within 50 km of source j?
                d_j_to_sta_i_km = calc_station_distance_km(
                    geod=geod,
                    src_lat=cand_j["fused_lat"],
                    src_lon=cand_j["fused_lon"],
                    sta_lat_arr=g.loc[comp_i, "station_lat"].to_numpy(),
                    sta_lon_arr=g.loc[comp_i, "station_lon"].to_numpy(),
                )

                n_j_inside_i = int(
                    np.sum(d_i_to_sta_j_km <= args.epicentral_train_max_km)
                )
                n_i_inside_j = int(
                    np.sum(d_j_to_sta_i_km <= args.epicentral_train_max_km)
                )

                if (n_j_inside_i > 0) or (n_i_inside_j > 0):
                    conflict_graph[i].add(j)
                    conflict_graph[j].add(i)

                cand_i["n_other_cluster_stations_within_50km"] += n_j_inside_i
                cand_j["n_other_cluster_stations_within_50km"] += n_i_inside_j

        for cand in candidates:
            cand["has_other_cluster_station_within_50km"] = (
                cand["n_other_cluster_stations_within_50km"] > 0
            )

        # ============================================================
        # Extract connected components of the conflict graph.
        # Each group is a set of mutually related ambiguous candidates.
        # ============================================================

        visited: set[int] = set()
        conflict_groups: list[list[int]] = []

        for i in range(n_cand):
            if i in visited:
                continue

            stack = [i]
            group = []
            visited.add(i)

            while stack:
                u = stack.pop()
                group.append(u)

                for v in conflict_graph[u]:
                    if v not in visited:
                        visited.add(v)
                        stack.append(v)

            conflict_groups.append(group)

        # ============================================================
        # Select final candidates.
        # - If a group has one candidate: retain it.
        # - If a group has multiple candidates: select one representative.
        # ============================================================

        selected_candidates: list[Candidate] = []

        for group in conflict_groups:
            group_candidates = [candidates[idx] for idx in group]

            if len(group_candidates) == 1:
                selected = group_candidates[0]
                selected_by = "independent_50km"

            else:
                if args.uncertainty_mode == "volume":
                    selected = min(
                        group_candidates,
                        key=lambda cand: (
                            cand["volume_km3"],
                            -cand["n_in_comp"],
                            cand["n_other_cluster_stations_within_50km"],
                        ),
                    )
                    selected_by = "conflict_min_volume"

                else:
                    raise ValueError(
                        f"Unknown conflict_resolve_mode: {args.uncertainty_mode}"
                    )

            selected["row"]["n_candidates"] = int(n_cand)
            selected["row"]["n_selected_sources"] = None
            selected["row"]["n_conflict_group"] = int(len(group_candidates))
            selected["row"]["n_other_cluster_stations_within_50km"] = int(
                selected["n_other_cluster_stations_within_50km"]
            )
            selected["row"]["has_other_cluster_station_within_50km"] = bool(
                selected["has_other_cluster_station_within_50km"]
            )
            selected["row"]["selected_by"] = selected_by

            selected_candidates.append(selected)

        possible_multiple_sources = len(selected_candidates) >= 2

        for source_id, cand in enumerate(selected_candidates):
            cand["row"]["source_id"] = int(source_id)
            cand["row"]["n_selected_sources"] = int(len(selected_candidates))
            cand["row"]["possible_multiple_sources"] = bool(possible_multiple_sources)

        out_rows.extend([cand["row"] for cand in selected_candidates])

    out_df = pd.DataFrame(out_rows)

    out_path = out_dir / f"fused_{fmt_dt(start)}_{fmt_dt(end)}.csv"
    out_path2 = out_dir / f"fused_removed_{fmt_dt(start)}_{fmt_dt(end)}.csv"

    out_df.to_csv(out_path, index=False)

    out_df2 = remove_isolated(out_df, geod)
    out_df2.to_csv(out_path2, index=False)

    print(f"[saved] {out_path}")
    print("groups:", n_groups)
    print("used groups:", n_used)
    print("total comps:", n_comps_total)
    print("avg comps per used group:", n_comps_total / max(n_used, 1))
    print("median comp size:", float(np.median(comp_sizes)) if comp_sizes else None)
    print("mean comp size:", float(np.mean(comp_sizes)) if comp_sizes else None)
    print("N fused:", len(out_df))
    print("N fused rmoved:", len(out_df2))


if __name__ == "__main__":
    main()
