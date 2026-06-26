# ParkFlow-AI — Feature Reference

Every feature/signal, how it's computed, and where it's used (compute file → artifact → dashboard).

## A. Model input features (the 17 the model trains on)

All built in `src/parkflow/features.py`; the model reads them in `src/parkflow/model.py`.
Their impact is shown in the **Model tab** (feature importance + SHAP global) and per-zone in
**Prediction Center → "Why this zone?"**.

### Time features (5) — `add_time_features()`
Capture diurnal/weekly/seasonal rhythm of violations.

| Feature | Computation | Why / where |
|---|---|---|
| `hour` | hour of the bin (IST) | morning/evening commute peaks; Analytics Center hour chart |
| `dayofweek` | 0=Mon…6=Sun | weekday vs weekend pattern; Analytics day chart |
| `month` | 1–12 | seasonal drift |
| `weekofyear` | ISO week | finer seasonality |
| `is_weekend` | 1 if Sat/Sun | weekends are quieter |

### History / lag features (5) — `add_lag_features()`
Leakage-safe (only past values, `shift >= 1`). Capture momentum — a zone busy recently tends to stay busy.

| Feature | Computation | Why / where |
|---|---|---|
| `lag_1bin` | violations in the previous 3-h bin | short-term momentum |
| `lag_1d` | same bin yesterday | daily recurrence |
| `lag_7d` | same bin last week | weekly recurrence |
| `roll_7d_mean` | trailing 7-day mean | one of the strongest predictors (high SHAP) |
| `roll_30d_mean` | trailing 30-day mean | slow-moving base level |

### Zone-profile features (7) — `add_zone_statics()` (computed on training window only)
"What kind of place is this zone?" — the dominant signal group.

| Feature | Computation | Why / where |
|---|---|---|
| `zone_hist_mean` | mean violations/bin at the zone | the single strongest feature |
| `zone_freq_enc` | zone's share of all violations | frequency-encodes the high-cardinality junction id |
| `is_junction` | 1 named junction / 0 grid cell | junctions behave differently; also used in priority score |
| `zone_heavy_veh_share` | fraction of bus/LGV/van/tempo | heavy vehicles block more road |
| `carriageway_block_share` | fraction of lane-blocking violation types (double parking, main road, near signal/crossing, opposite-parked) | zones whose violations directly obstruct moving lanes |
| `repeat_offender_density` | `log1p(#vehicles seen > threshold times here)` | chronic blockage vs one-offs |
| `violation_growth_rate` | trailing 7-day mean ÷ 30-day mean (clipped 0–5) | is the zone trending worse or improving? |

## B. Congestion-impact signals (NOT model inputs — decision layer)

Computed in `src/parkflow/intelligence.py` `congestion_index()`, written to `future_forecast.csv`
by `src/parkflow/pipeline.py`, shown in **Prediction Center** and **Enforcement** tabs.

```
effective_load = predicted_violations × mean_PCU × road_factor × violation_severity × peak_hour_mult
est_capacity_reduction_% = max_cap × (1 − exp(−effective_load / saturation))
congestion_index (0–100)  = est_% / max_cap × 100
```

| Signal | Computation | Source |
|---|---|---|
| `mean_pcu` | mean Passenger-Car-Units of vehicles at zone (bus 3.5 … scooter 0.5) | Indo-HCM/IRC constants |
| `viol_severity` | mean carriageway-blocking severity of violation types | config weights |
| `road_factor` | 1.3 junction / 1.0 side street | side-friction principle |
| `peak_hour_mult` | 1.5 if forecast bin is in 7–11 or 17–20 IST | rush-hour amplification |

**Outputs used:** `congestion_index`, `est_capacity_reduction_pct`, `pcu_load` → forecast table + Enforcement ranking.

## C. Other decision outputs

| Output | Computation | File | Where used |
|---|---|---|---|
| `risk` band | predicted count → Low/Med/High/Critical | `risk_band()` | forecast color-coding, KPIs, patrol cards |
| `priority_score` | 0.6·pred + 0.3·hist + 0.1·junction (0–100) | `enforcement_priority()` | Enforcement ranking, patrol selection, Junction Risk |
| patrol plan | greedy top-N with haversine spatial spread | `allocate_patrols()` | `patrol_plan.csv` → Enforcement map + team cards |
| patrol **routes** | OR-Tools CVRP on a haversine matrix; ordered route per team, `zones_per_team` cap; greedy fallback | `route_patrols()` | `patrol_routes.csv` → Enforcement route map |

## C2. Judge-upgrade decision layers

| Output | Computation | File → artifact | Dashboard |
|---|---|---|---|
| **24h forecast timeline** | recursive multi-step: predict bin → feed back as lag → repeat for `forecast_horizon_bins` (leakage-safe); `--live` relabels the anchor to now | `build_multi_horizon_frames()` → `forecast_timeline.csv` | Prediction Center → "Next 24h timeline" |
| **Economic cost (₹)** | `pred × vehicles_blocked × (max_delay × cap_red%/100) × occupancy × value_of_time`; tied to the congestion layer. Value of time ≈ ₹164/commuter-hour derived from the Bengaluru ISEC study WP-554 (Vijayalakshmi & Krishna Raj, 2023): ₹11,45,568 ÷ 6,998 h. Per-commuter model (occupancy = 1.0); `vehicles_blocked`/`max_delay` are tunable assumptions | `economics.economic_impact()` → `economic_impact.csv` + `economic_summary` in `metrics.json` | Economic Impact tab + KPI |
| **Displacement** | covered zone sheds `displaced_fraction` to nearest uncovered zone within radius (else suppressed); naive-spread vs routed leakage | `displacement.simulate_displacement()` / `compare_layouts()` → `displacement.csv` + `displacement_summary` | Enforcement → displacement map + metrics |
| **Operator actions** | confirm / override / complete deployments persisted to SQLite (stdlib) | `operations.py` → `enforcement_log.db` | Enforcement → Operator console |

## D. Analytics features

Computed in `src/parkflow/analytics.py` and `src/parkflow/pipeline.py`.

| Feature | Computation | Artifact | Dashboard |
|---|---|---|---|
| Repeat offenders | per-vehicle count, unique zones, type, last seen, top zone | `repeat_offenders.csv` | Repeat Offenders tab (KPIs, table, density map) |
| Zone willful/infra signal | repeat-offender share per zone → "willful (towing)" vs "infrastructure" | `zone_repeat_offenders.csv` | artifact (signal) |
| Junction risk | per-junction total, avg/day, peak hour, dominant vehicle + predicted/risk/priority | `junction_risk.csv` | Junction Risk tab |
| Temporal trends | counts by hour/day/week + hour×day & type×hour heatmaps | from `events_analytics.parquet` | Analytics Center tab |
| Cleaned events | compact zone/station/type/time/lat-lon table | `events_analytics.parquet` | Hotspot Analysis filters + heatmap |

## E. Explainability & evaluation

| Feature | Computation | File | Where used |
|---|---|---|---|
| SHAP global | mean \|SHAP\| per feature | `src/parkflow/explain.py` | `shap_global.csv` → Model tab bar |
| SHAP per-zone | top signed feature contributions per zone | explain.py | `shap_reasons.csv` → Prediction Center "Why this zone?" |
| Top-K hit-rate | overlap of top-K predicted vs actual per time slice | `src/parkflow/evaluation.py` | `metrics.json` → Model tab |
| Hotspot PR-AUC | average precision for "is hotspot?" | evaluation.py | metrics.json → Model tab |
| Actual-vs-predicted | test predictions + error | pipeline.py | `test_predictions.csv` → Model tab diagnostics |

## F. The flow in one line

**preprocessing** (clean+filter) → **features** (17 inputs) → **model** (Tweedie, recursive 24h
forecast) → **intelligence** (congestion/risk/priority/patrol greedy+VRP) + **economics** (₹) +
**displacement** (blindspots) + **analytics** (offenders/junctions/trends) + **explain** (SHAP) →
**artifacts/** → **dashboard** (9 tabs) → **operations** (operator actions → `enforcement_log.db`).
