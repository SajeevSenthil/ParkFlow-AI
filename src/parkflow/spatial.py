"""Stage 2 -- spatial normalization (PRD section 5, step 6).

Junction is the primary zone. Rows whose junction is missing fall back to a
coordinate grid cell so they are never silently dropped. Output adds:
    zone, zone_kind, zone_lat, zone_lon
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema as S
from .config import Config
from .logging_utils import get_logger

log = get_logger("spatial")


def assign_zones(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    junction = df[S.JUNCTION_NAME].astype(str)
    is_missing = junction == cfg.spatial.missing_junction_label

    # Fallback grid id from rounded coordinates.
    dec = cfg.spatial.grid_decimals
    lat_r = df[S.LATITUDE].round(dec)
    lon_r = df[S.LONGITUDE].round(dec)
    grid_id = "GRID_" + lat_r.astype(str) + "_" + lon_r.astype(str)

    df[S.ZONE] = np.where(is_missing, grid_id, junction)
    df[S.ZONE_KIND] = np.where(is_missing, "grid", "junction")

    # Representative coordinate per zone = centroid of its points (for map placement).
    centroids = (
        df.groupby(S.ZONE)[[S.LATITUDE, S.LONGITUDE]]
        .mean()
        .rename(columns={S.LATITUDE: S.ZONE_LAT, S.LONGITUDE: S.ZONE_LON})
    )
    df = df.merge(centroids, on=S.ZONE, how="left")

    n_zones = df[S.ZONE].nunique()
    n_grid = int((df[S.ZONE_KIND] == "grid").sum())
    log.info("Assigned %d zones (%d rows fell back to grid cells)", n_zones, n_grid)
    return df


def zone_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """One row per zone: kind + representative coordinate. Used by the dashboard/map."""
    return (
        df.groupby(S.ZONE)
        .agg(
            zone_kind=(S.ZONE_KIND, "first"),
            zone_lat=(S.ZONE_LAT, "first"),
            zone_lon=(S.ZONE_LON, "first"),
        )
        .reset_index()
    )
