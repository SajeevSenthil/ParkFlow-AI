"""Typed configuration loaded from ``config/config.yaml``.

Dataclasses give us autocompletion, validation and a single source of truth for
every tunable. ``Config.load()`` is the only entry point the rest of the code uses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field  # noqa: F401 (field used in PreprocessingCfg)
from pathlib import Path
from typing import Any

import yaml


def _project_root() -> Path:
    # src/parkflow/config.py -> project root is three levels up.
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    raw_data: Path
    artifacts_dir: Path

    def ensure_dirs(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SpatialCfg:
    grid_decimals: int
    missing_junction_label: str
    missing_station_label: str


@dataclass(frozen=True)
class TemporalCfg:
    bin_hours: int
    lag_days: list[int]
    rolling_days: list[int]
    timezone: str = "Asia/Kolkata"

    @property
    def bins_per_day(self) -> int:
        if 24 % self.bin_hours != 0:
            raise ValueError("bin_hours must divide 24 evenly")
        return 24 // self.bin_hours


@dataclass(frozen=True)
class ModelCfg:
    test_fraction: float
    objective: str
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    min_child_weight: int
    random_state: int

    def xgb_params(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "random_state": self.random_state,
            "tree_method": "hist",
            "n_jobs": -1,
        }


@dataclass(frozen=True)
class RiskBand:
    name: str
    max: float


@dataclass(frozen=True)
class PriorityCfg:
    w_predicted: float
    w_historical: float
    w_junction: float


@dataclass(frozen=True)
class DisruptionCfg:
    vehicle_weights: dict[str, float]
    default_vehicle_weight: float
    junction_road_weight: float
    side_street_weight: float
    violation_severity_weights: dict[str, float]
    default_violation_severity: float
    peak_hours_morning: list[int]
    peak_hours_evening: list[int]
    peak_hour_multiplier: float
    repeat_offender_threshold: int


@dataclass(frozen=True)
class PatrolCfg:
    num_teams: int
    spatial_suppress_km: float


@dataclass(frozen=True)
class PreprocessingCfg:
    approved_only: bool = False
    exclude_statuses: list[str] = field(default_factory=lambda: ["rejected", "duplicate"])


@dataclass(frozen=True)
class Config:
    paths: Paths
    spatial: SpatialCfg
    temporal: TemporalCfg
    model: ModelCfg
    risk_bands: list[RiskBand]
    priority: PriorityCfg
    disruption: DisruptionCfg
    patrol: PatrolCfg
    preprocessing: PreprocessingCfg
    valid_violation_types: list[str]
    log_level: str

    @staticmethod
    def load(path: str | Path | None = None) -> "Config":
        root = _project_root()
        cfg_path = Path(path) if path else root / "config" / "config.yaml"
        with open(cfg_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        def abspath(p: str) -> Path:
            pp = Path(p)
            return pp if pp.is_absolute() else (root / pp)

        paths = Paths(
            raw_data=abspath(raw["paths"]["raw_data"]),
            artifacts_dir=abspath(raw["paths"]["artifacts_dir"]),
        )
        risk_bands = [
            RiskBand(name=b["name"], max=float(b["max"]) if b["max"] != ".inf" else math.inf)
            for b in raw["risk_bands"]
        ]
        pre_raw = raw.get("preprocessing", {})
        preprocessing = PreprocessingCfg(
            approved_only=bool(pre_raw.get("approved_only", False)),
            exclude_statuses=list(pre_raw.get("exclude_statuses", ["rejected", "duplicate"])),
        )
        return Config(
            paths=paths,
            spatial=SpatialCfg(**raw["spatial"]),
            temporal=TemporalCfg(**raw["temporal"]),
            model=ModelCfg(**raw["model"]),
            risk_bands=risk_bands,
            priority=PriorityCfg(**raw["priority"]),
            disruption=DisruptionCfg(**raw["disruption"]),
            patrol=PatrolCfg(**raw["patrol"]),
            preprocessing=preprocessing,
            valid_violation_types=list(raw["valid_violation_types"]),
            log_level=raw.get("logging", {}).get("level", "INFO"),
        )
