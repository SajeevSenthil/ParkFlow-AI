"""Shared test fixtures.

``make_raw`` builds a tiny DataFrame in the *real* raw schema (JSON-array
violation types, UTC-string timestamps, "No Junction" values, admin columns,
injected dirt) so tests exercise the actual parsing/cleaning paths -- this is a
unit-test fixture, not a product data source.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from parkflow import schema as S

_JUNCTIONS = [
    ("BTP051 - Safina Plaza Junction", "Commercial Street", 12.9820, 77.6090),
    ("BTP082 - KR Market Junction", "KR Market", 12.9619, 77.5751),
    ("BTP058 - Subbanna Junction", "Vijayanagar", 12.9719, 77.5360),
    ("No Junction", "Koramangala", 12.9352, 77.6245),  # forces grid fallback
]
_VEHICLES = ["CAR", "SCOOTER", "MOTOR CYCLE", "PASSENGER AUTO", "BUS (BMTC/KSRTC)"]


def make_raw(days: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-01-01", tz="UTC")
    rows: list[dict] = []
    for day in range(days):
        date = start + pd.Timedelta(days=day)
        dow = date.dayofweek
        for hour in range(24):
            intensity = (1.4 if 7 <= hour <= 10 or 17 <= hour <= 20 else 0.2)
            intensity *= 1.0 if dow < 5 else 0.5
            for jname, loc, lat, lon in _JUNCTIONS:
                for _ in range(rng.poisson(intensity)):
                    ts = date + pd.Timedelta(hours=hour, minutes=int(rng.integers(0, 60)))
                    rows.append(
                        {
                            "id": "FKID000000",
                            S.LATITUDE: round(lat + rng.normal(0, 0.0008), 6),
                            S.LONGITUDE: round(lon + rng.normal(0, 0.0008), 6),
                            S.LOCATION: loc,
                            S.VEHICLE_NUMBER: "FKN00GL0000",
                            S.VEHICLE_TYPE: rng.choice(_VEHICLES),
                            S.VIOLATION_TYPE: '["WRONG PARKING","NO PARKING"]',
                            S.OFFENCE_CODE: "[112,104]",
                            S.CREATED_DATETIME: ts.strftime("%Y-%m-%d %H:%M:%S+00"),
                            "device_id": "FKDEV00000",
                            S.POLICE_STATION: loc,
                            S.JUNCTION_NAME: jname,
                            S.VALIDATION_STATUS: rng.choice(["approved", "rejected"]),
                        }
                    )
    df = pd.DataFrame(rows)

    # Inject dirt: non-parking offences (must be filtered) + exact duplicates.
    n_np = max(1, len(df) // 10)
    extra = df.sample(n=n_np, random_state=seed).copy()
    extra[S.VIOLATION_TYPE] = '["DEFECTIVE NUMBER PLATE"]'
    dups = df.sample(n=max(1, len(df) // 20), random_state=seed + 1).copy()
    df = pd.concat([df, extra, dups], ignore_index=True)
    return df.sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_raw(days=30, seed=7)
