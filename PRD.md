# ParkFlow-AI — Product Requirements & Engineering Guide

**Predicting and Mitigating Parking-Induced Traffic Disruptions**

> This single document has two parts:
> **Part I — Product Requirements** (what the system does and why; §1–§13)
> **Part II — Engineering / Developer Guide** (how to run, the code map, where the model lives; §14–§22)

| Field | Value |
|---|---|
| Product | ParkFlow-AI |
| Version | 1.0 (Draft) |
| Date | 2026-06-19 |
| Status | Pre-implementation |
| Delivery | Interactive Streamlit dashboard + ML forecasting backend |

---

## 1. Problem Statement

Traffic authorities currently respond to illegal parking **reactively** — patrols are dispatched after congestion or complaints occur. Historical parking-violation records exist but are not converted into operational intelligence.

ParkFlow-AI transforms raw parking-violation records into **proactive enforcement intelligence**: it forecasts where and when violations will occur, ranks zones by enforcement urgency, and recommends patrol deployment — enabling a shift from reactive to **data-driven, proactive parking management**.

---

## 2. Goals & Non-Goals

### 2.1 Goals
1. Identify illegal-parking hotspots from historical violation density.
2. Forecast future violation intensity at the **junction × hour** level.
3. Classify locations into operational risk categories.
4. Rank enforcement zones by priority.
5. Recommend proactive patrol deployments.
6. Surface temporal and junction-level analytics for planning.
7. Deliver all of the above through a unified Streamlit dashboard.

### 2.2 Non-Goals (Out of Scope)
| Excluded Capability | Reason |
|---|---|
| Parking-induced congestion-impact analysis | Requires traffic-speed / lane / queue / road-capacity data not in dataset |
| Traffic-flow reduction estimation | Requires real-time/historical traffic-flow measurements not available |
| Real-time CV-based parking detection | Project focuses on historical analytics + prediction, not live detection |

---

## 3. Users & Use Cases

| User | Use Case |
|---|---|
| Traffic enforcement command | Decide where to send patrols tomorrow morning |
| Junction planners | Identify chronically problematic intersections |
| Operations analysts | Understand hourly/daily/weekly violation trends |
| Leadership | Monitor city-wide parking-enforcement KPIs |

**Primary scenario:** "It is 7 AM. Show me the highest-priority junctions for the next few hours and tell me where to deploy each patrol team."

---

## 4. Data Requirements

### 4.1 Raw Input Fields
- Latitude, Longitude
- Junction Name
- Police Station
- Vehicle Type
- Violation Type
- Created Datetime

### 4.2 Fields Removed From the *Model* (kept for analytics)
Administrative / workflow metadata with no predictive value, dropped from model features:
`id`, `device_id`, `created_by_id`, `center_code`, `closed_datetime`, `modified_datetime`, `action_taken_timestamp`, `data_sent_to_scita_timestamp`, `updated_vehicle_number`, `updated_vehicle_type`

**Retained for analytics** (not model features, but cheap high-value views):
- `vehicle_number` → repeat-offender detection.
- `validation_status` / `validation_timestamp` → a "confirmed violations" view (filter to *approved*) to cut false positives.

### 4.3 Relevant Violation Types (filter — keep only these)
`NO PARKING`, `WRONG PARKING`, `PARKING IN A MAIN ROAD`, `PARKING NEAR ROAD CROSSING`

---

## 5. Data Preprocessing Pipeline

| Step | Action | Notes |
|---|---|---|
| 1 | Remove irrelevant/admin columns | Reduce dimensionality |
| 2 | Handle missing values | junction→`No Junction`, station→`Unknown Station`, status→`Pending` (or drop if sparse) |
| 3 | Filter valid parking violations | Keep only the 4 relevant violation types |
| 4 | Remove duplicates | Key: `vehicle_number + latitude + longitude + created_datetime` |
| 5 | Extract temporal features | Hour, Day, Weekday, Month, Week No., Quarter, Weekend Flag |
| 6 | Spatial normalization | Junction-based (primary); coordinate grid (fallback) |
| 7 | Create violation counts | Aggregate to `Zone × time-bin → count` |
| 8 | **Build complete grid + zero-fill** | Full `zone × time-bin` Cartesian grid, absent cells = 0 (see §7.3) — **do before lag features** |
| 9 | Historical lag features | Prev bin/day/week, rolling 7-day & 30-day avg (computed on zero-filled, time-ordered series) |
| 10 | Location statistics | Total, avg daily, peak-hour, growth rate |
| 11 | Encode categoricals | Frequency/target encoding for high-cardinality `junction_name`; label encoding for the rest |

**Final modelling row:** `Location Zone × Time Window` → features + target (`Expected Violation Count`).

---

## 6. Feature Engineering

| Group | Features |
|---|---|
| Temporal | Hour, Day of Week, Month, Week Number, Weekend Flag |
| Historical Activity | Violations prev hour/day/week, Rolling 7-day avg, Rolling 30-day avg |
| Location | Junction, Police Station, Historical junction violation frequency |
| Vehicle | Vehicle Type distribution, Violation Type distribution |

---

## 7. Modelling

### 7.1 Problem Framing
Supervised **regression** — predict expected violation count for a junction during a future hourly window.

### 7.2 Regression, NOT classification (decision)
The model predicts a **count** (`Expected Violations = 23`), then risk bands are derived from that count (see §8.1). We do **not** output a "risk %" — a count regressor cannot produce a probability, and presenting one is incorrect. If a headline probability is ever wanted, it must be a *separate* classifier with its own metrics; we deliberately avoid that to keep one clean model.

### 7.3 Step 0 — Build a complete grid with zero-fill (critical)
Raw aggregation only yields rows where violations occurred, so the model would never see a "quiet" cell and could not predict lulls. Before feature engineering:
1. Build the full Cartesian grid of `junction × time-bin` across the date range.
2. **Fill absent combinations with 0.**
3. Engineer lag/rolling features on this dense, zero-filled series.

Get this wrong and MAE/R²/the whole model are meaningless. This is the first modelling step, not an afterthought.

### 7.4 Time granularity (decide empirically)
Hourly × junction will likely be extremely sparse (most cells 0 or 1) across hundreds of junctions. Evaluate **3–4 hour bins** (morning / midday / evening / night) or **daily-per-junction** against hourly once record volume is known. Pick the finest granularity that still carries signal.

### 7.5 Model
- **Primary:** XGBoost Regressor with **`objective="reg:tweedie"`** (`tweedie_variance_power≈1.3`). The Tweedie compound Poisson-Gamma distribution models non-negative, zero-heavy, overdispersed counts better than plain Poisson — a measurable MAE/PR-AUC gain on this data.
- **Rationale:** strong on tabular data, handles nonlinearity & mixed feature types, native SHAP via `pred_contribs`, performs well with temporal+spatial features.
- **Categorical encoding:** use **frequency / target encoding** for `junction_name` (high cardinality + a large `No Junction` bucket), not raw label encoding.

### 7.6 Inputs / Output
- **Inputs:** Junction, Police Station, Hour, Day of Week, Month, Weekend Flag, historical violation counts, rolling averages, vehicle distribution stats.
- **Output:** Predicted violation count for a junction-time-bin pair.

### 7.7 Evaluation
- **Strategy:** time-based split — train on earlier periods, test on later periods (simulates real forecasting).
- **Mandatory baseline:** compare XGBoost against a **seasonal-naive baseline** ("predict this junction's historical average for this hour-of-week"). For recurring temporal data this baseline is strong; if the model can't beat it, the ML adds nothing — and showing this comparison honestly is a credibility win.
- **Leakage guard:** split by time *first*, then compute lag/rolling features within each side respecting order. Features must never cross the train/test boundary.
- **Metrics:** MAE, RMSE, R², **Poisson deviance**, plus **ranking metrics** that match the operational question on sparse/imbalanced data: **Top-K hit-rate** (did we flag the right worst zones per window?) and **hotspot PR-AUC** (robust where accuracy is meaningless). Model wins on MAE/R²/Poisson/PR-AUC; the seasonal mean stays competitive on Top-K at fine granularity (reported honestly).
- **Explainability:** exact **TreeSHAP** (XGBoost native `pred_contribs`, no external dep) gives global feature importance and per-zone "why this zone?" reasons in the dashboard.

---

## 8. Intelligence Layers (Post-Prediction)

### 8.1 Risk Classification
| Predicted Count | Risk |
|---|---|
| 0 – 5 | Low |
| 6 – 15 | Medium |
| 16 – 25 | High |
| > 25 | Critical |

### 8.2 Parking Congestion Impact Index (answers the literal problem statement)
The problem statement asks to *quantify impact on traffic flow*. We have no traffic telemetry (no speed/queue/road-width — §2.2) **and the contest forbids external datasets**, so we estimate impact from **domain theory + the provided data only**:
```
pcu_load            = predicted_violations × mean_PCU(zone) × road_factor
est_capacity_red %  = max_cap × (1 − exp(−pcu_load / saturation_pcu))   # Indo-HCM-style
Congestion Index    = est_capacity_red% / max_cap × 100   (0–100)
```
**PCU** (Passenger Car Units, Indo-HCM/IRC) are standard traffic-engineering constants — a bus ≈ 3.5, a scooter ≈ 0.5 — not external data. The HCM principle that parked vehicles cut a road's *saturation flow* turns a violation forecast into an **estimated % of road capacity lost**, which is exactly what the brief asks for and is fully defensible under the no-external-data rule. (OSMnx road geometry / TomTom speeds would sharpen this but are **out of scope — external data → disqualification risk.**)

### 8.3 Hotspot Generation
- **Current hotspots:** from historical violation density.
- **Future hotspots:** from predicted violation counts.

### 8.4 Enforcement Priority Score
```
Priority = 0.6 × Predicted Violations
         + 0.3 × Historical Frequency
         + 0.1 × Junction Weight
```
Output: ranked enforcement zones. (`Junction Weight` defined in §13.)

### 8.5 Patrol Recommendation Engine
Allocate `N` available patrol teams (N is an input, not fixed) to priority zones via a **greedy allocation with spatial spread** — once a junction is assigned, nearby junctions are down-weighted so teams aren't all sent to one cluster. Output: recommended zones per team, time-of-day-aware deployment schedule, daily enforcement action list.

---

## 9. Deliverables

### Core
1. **Illegal Parking Hotspot Heatmap** — interactive city-wide heatmap, location & police-station density.
2. **Future Violation Hotspot Prediction** — predicted locations, expected counts/risk, future heatmap.
3. **Enforcement Priority Ranking** — ranked zones, priority scores, High/Medium/Low classes.
4. **Smart Patrol Recommendation Engine** — recommended zones, schedule, daily action list.

### Supporting Analytics
5. **Temporal Violation Analytics** — hour/day/week/month trends, peak periods.
6. **Junction Risk Assessment** — high-risk junction list, per-junction stats, severity ranking.

---

## 10. Dashboard Specification (Streamlit)

| Section | Contents |
|---|---|
| **Overview** | Total Violations, Active Hotspots, High-Priority Zones, Predicted Violations |
| **Hotspot Analysis** | Interactive heatmap; filters: location, police station, violation type |
| **Prediction Center** | Predicted hotspots, future risk levels, forecasted counts, future heatmap |
| **Enforcement Center** | Priority ranking, patrol recommendations, high-risk monitoring, junction risk dashboard |
| **Analytics Center** | Hourly, daily, weekly trends; monthly summaries; junction risk analysis |

---

## 11. End-to-End Workflow

```
Raw Parking Violation Dataset
        ↓
Data Preprocessing
        ↓
Spatial Aggregation (Junction-Level)
        ↓
Temporal Aggregation (time-bin) + Complete Grid Zero-Fill
        ↓
Historical Feature Engineering  (leakage-safe, time-ordered)
        ↓
Seasonal-Naive Baseline  ──►  benchmark
        ↓
Junction-Bin Violation Forecasting Model (XGBoost, count:poisson)
        ↓
Risk Banding (from predicted count)
        ↓
Hotspot Generation  +  Disruption Proxy
        ↓
Priority Scoring
        ↓
Patrol Recommendation Engine (greedy + spatial spread)
        ↓
Streamlit Dashboard (reads precomputed predictions)
```

---

## 12. Success Criteria

1. Model **beats the seasonal-naive baseline** on the held-out future test period (MAE/RMSE; Poisson deviance for count fit).
2. Dashboard renders all five sections with working filters and heatmaps.
3. Priority ranking and patrol recommendations are generated end-to-end from predictions.
4. Pipeline runs reproducibly from raw CSV → predictions file → dashboard (no retraining inside Streamlit).

---

## 13. Risks & Decisions

### Resolved decisions
| # | Decision |
|---|---|
| D1 | **Regression, not classification** — predict a count, derive risk bands; never output a fake "risk %" (§7.2). |
| D2 | **Zero-fill the complete grid before feature engineering** — non-negotiable step 0; otherwise the model never sees lulls (§7.3). |
| D3 | **Poisson objective + mandatory seasonal-naive baseline**; leakage-safe temporal split (§7.5, §7.7). |
| D4 | **Congestion = transparent Disruption Proxy**, explicitly *not* measured congestion; optional BTP-map correlation (§8.2). |
| D5 | **Patrol = greedy allocation with spatial spread**, N teams as input (§8.5). |
| D6 | **Keep `vehicle_number` / `validation_status` for analytics**, not as model features (§4.2). |

### Open questions
1. **Time granularity** — hourly vs 3–4h bins vs daily to be decided empirically once record volume is seen (§7.4). Currently 3h; daily/junction-only raises R² to ~0.59.
2. **Junction Weight** — the 0.1 priority term currently uses the junction flag; could be refined (road class, main-road flag, traffic importance).
3. **Grid fallback granularity** — coordinate-grid cell size for `No Junction` records is set to 2 dp (~1.1 km); revisit if needed.
4. **Risk thresholds** are fixed/global — may need per-junction calibration if violation scales vary widely across the city.
5. **Disruption Proxy weights** — vehicle and road/junction weights are heuristic; surface a "heuristic, not measured" disclaimer in the UI (done in the dashboard).

---

# Part II — Engineering / Developer Guide

> For someone new to the codebase who wants to run or improve it. Part I above is the
> *what & why*; this part is the *how*.

## 14. Running the project

### Prerequisites
- Python 3.10+ (developed on 3.12)
- The dataset CSV at the path set in `config/config.yaml` → `paths.raw_data`
  (currently `dataset/jan to may police violation_anonymized791b166.csv`)

### One-time setup
```bash
cd ParkFlow-AI
python -m venv .venv
.venv/Scripts/activate              # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -e ".[dashboard,dev]"   # package + streamlit + pytest
```

### The commands you'll actually use
```bash
parkflow info     # print resolved config (data path, bin size, model params)
parkflow run      # run the WHOLE pipeline: clean → train → test → write artifacts/
pytest -q         # run the test suite (validates pipeline logic, not model accuracy)
```
> If `parkflow` isn't on PATH, use `python -m parkflow run`.

### See the dashboard
```bash
streamlit run app/streamlit_app.py    # then open http://localhost:8501
```
The dashboard only **reads** `artifacts/` — run `parkflow run` at least once first.

### Typical dev loop
1. Edit code or `config/config.yaml`.
2. `parkflow run` (regenerates `artifacts/`).
3. Refresh / restart the dashboard.
4. `pytest -q` before committing.

---

## 15. Where the model lives

| Thing | Location |
|---|---|
| **Trained model file** | `artifacts/model.joblib` (written by `parkflow run`) |
| **Model code** (train/predict/save/load) | `src/parkflow/model.py` |
| **Model hyperparameters** | `config/config.yaml` → `model:` block |
| **What it trains on** | the zero-filled feature grid from `features.py` |
| **How it's scored** | `evaluation.py` → results in `artifacts/metrics.json` |

The model is an **XGBoost regressor** (`count:poisson`). It is **not** retrained by the
dashboard — training happens only in `parkflow run`; the dashboard loads precomputed outputs.

**Inputs (14 features):** time (hour, dayofweek, month, weekofyear, is_weekend); history
(lag_1bin, lag_1d, lag_7d, roll_7d_mean, roll_30d_mean); zone profile (zone_hist_mean,
zone_freq_enc, is_junction, zone_heavy_veh_share).
**Output:** `violation_count` — expected violations for one zone in one time window.

---

## 16. Repository structure

```
ParkFlow-AI/
├── config/config.yaml         ← ALL tunables (data path, bins, model params, weights, risk bands)
├── dataset/                   ← raw input CSV (gitignored; large)
├── src/parkflow/              ← the pipeline package (see §17)
├── app/streamlit_app.py       ← dashboard (reads artifacts/ only)
├── tests/                     ← pytest suite
├── artifacts/                 ← pipeline OUTPUTS: model, forecasts, metrics (gitignored)
├── PRD.md                     ← this file (product spec + engineering guide)
├── README.md                  ← short overview + quickstart
├── pyproject.toml             ← packaging, console script, pytest/ruff config
└── requirements.txt           ← dependency list
```

---

## 17. Module-by-module reference (`src/parkflow/`)

Listed in **pipeline order** — also the order data flows through them.

**Foundation**
| File | Responsibility |
|---|---|
| `schema.py` | Column-name constants (raw + engineered). One place to change on a schema change. |
| `config.py` | Loads `config.yaml` into typed, frozen dataclasses (`Config.load()`). |
| `logging_utils.py` | Consistent `parkflow.*` logging. |
| `io.py` | Read/write tables (Parquet→CSV fallback) and JSON metrics. |

**Pipeline stages**
| File | Stage | What it does |
|---|---|---|
| `preprocessing.py` | 1. Clean | Drop admin cols, **parse JSON-array `violation_type`**, **UTC→IST** timestamps, filter to parking types, impute junction/station, dedupe. Returns events + a removal-stats record. |
| `spatial.py` | 2. Zones | Assign each event to a **zone**: junction name, else a **coordinate-grid cell** fallback. Adds a representative lat/lon per zone. |
| `features.py` | 3. Features | **Critical module.** Aggregate to `zone × time-bin`, **build complete grid + zero-fill**, add **leakage-safe lags/rolling** + train-only zone statics, and build the **one-step-ahead future frame**. |
| `baseline.py` | 4a. Baseline | Seasonal-naive predictor — the bar the model must beat. |
| `model.py` | 4b. Model | XGBoost (`count:poisson`) wrapper: train, predict (clip ≥ 0), importance, save/load. |
| `evaluation.py` | 5. Evaluate | MAE, RMSE, R², Poisson deviance (model vs baseline). |

**Decision layers (business logic, not ML)**
| File | What it does |
|---|---|
| `intelligence.py` | `risk_band`, `disruption_proxy` (heuristic, not congestion), `enforcement_priority`, `allocate_patrols` (greedy + spatial spread). |
| `analytics.py` | Per-junction risk table + temporal trends; produces the compact `events_analytics` table the dashboard filters on. |

**Orchestration & interface**
| File | What it does |
|---|---|
| `pipeline.py` | **The conductor.** Runs every stage, compares model vs baseline, forecasts, applies decision layers, writes artifacts. Start at `run()`. |
| `cli.py` / `__main__.py` | CLI: `parkflow run`, `parkflow info`. |
| `__init__.py` | Package marker; exposes `Config`, `__version__`. |
| `app/streamlit_app.py` | 7-tab dashboard (Overview, Hotspot Analysis + filters/heatmap, Prediction, Enforcement, Analytics Center, Junction Risk, Model). Reads `artifacts/` only. |

---

## 18. What `parkflow run` produces (`artifacts/`)

| File | Contents | Consumed by |
|---|---|---|
| `model.joblib` | trained model + feature list | reuse / inference |
| `metrics.json` | baseline vs model scores, clean stats, config used | Model tab |
| `future_forecast.csv` | per-zone predicted count, risk, priority, proxy (next window) | Prediction / Enforcement |
| `patrol_plan.csv` | team → zone assignments (spatially spread) | Enforcement |
| `current_hotspots.csv` | historical violations per zone + coordinates | Overview |
| `junction_risk.csv` | per-junction severity assessment | Junction Risk |
| `events_analytics.parquet` | compact cleaned events (zone, station, type, time, lat/lon) | Hotspot Analysis + Analytics |
| `zone_metadata.csv` | zone kind + representative coordinate | mapping |
| `feature_importance.csv` | top model features | Model tab |

---

## 19. How to make common changes

| I want to… | Do this |
|---|---|
| Point at a different dataset | `config.yaml` → `paths.raw_data` |
| Forecast daily instead of 3-hourly | `config.yaml` → `temporal.bin_hours: 24` |
| Change risk thresholds | `config.yaml` → `risk_bands` |
| Tune the model | `config.yaml` → `model` block |
| Change patrol team count / spread | `config.yaml` → `patrol` block |
| Add a model feature | add it in `features.py` (append to `feature_cols`), re-run |
| Add a dashboard view | edit `app/streamlit_app.py` (read an artifact, add a tab) |
| Handle a new raw column | add the constant in `schema.py`, use it in `preprocessing.py` |
| Add a new pipeline output | write it in `pipeline.py` via `write_table`/`write_json` |

**Golden rule:** the dashboard stays read-only. Anything expensive belongs in `pipeline.py`,
written to `artifacts/`.

---

## 20. Testing

```bash
pytest -q
```
Tests live in `tests/` and assert **pipeline correctness**, not forecast accuracy:
- `test_preprocessing.py` — JSON-array parsing, non-parking/duplicate filtering, timezone conversion, required-column guard.
- `test_features.py` — grid completeness, zero-fill presence, lag non-leakage, future-frame shape.
- `conftest.py` — a tiny real-schema DataFrame fixture (unit-test data only).

Forecast quality is reported by `parkflow run` (the baseline-vs-model scorecard), not by pytest.

---

## 21. Engineering design decisions (don't "fix" these by accident)

1. **Zero-fill the grid before features** — else the model never sees quiet cells and can't predict lulls (`features.py`).
2. **Regression, then risk bands** — predict a *count*; never output a fake "risk %".
3. **Always beat the baseline** — `baseline.py` proves the ML adds value; losing to it is a regression.
4. **No future leakage** — lags use only past values; split is by time.
5. **Disruption proxy ≠ congestion** — transparent heuristic (no traffic-flow data exists). Keep it labeled.
6. **One model, not five** — complexity lives in the decision layer, not a model zoo.

---

## 22. Where to start reading the code

1. `config/config.yaml` — every knob.
2. `pipeline.py::run()` — the whole flow in ~80 lines.
3. `features.py` — the most subtle module (zero-fill + leakage-safe lags).
4. `app/streamlit_app.py` — how outputs become a product.

---

*Part I consolidates the original `pipeline.md`, `data_preprocessing.md`, and `final_deliverables.md` design notes; Part II replaces the standalone onboarding guide.*
