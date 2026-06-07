from pathlib import Path

import pandas as pd


def get_station_catalog(station_file: Path | str) -> pd.DataFrame:
    """Load station metadata from a whitespace-delimited file.

    Parameters
    ----------
    station_file : Path or str
        Path to the station catalog file.

    Returns
    -------
    pd.DataFrame
        Data frame with ``station``, ``lat``, and ``lon`` columns.

    Raises
    ------
    FileNotFoundError
        If ``station_file`` does not exist.
    """
    station_file = Path(station_file)

    if not station_file.exists():
        raise FileNotFoundError(f"⚠️ File not found: {station_file}")

    station_catalog = pd.read_csv(
        station_file,
        sep=r"\s+",
        engine="python",
    )
    return station_catalog[["station", "lat", "lon"]].copy()
