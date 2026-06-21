<h1 align="center">ParkFlow-AI</h1>
<p align="center"><b>Predicting & Mitigating Parking-Induced Traffic Disruptions</b></p>
<p align="center">
AI-driven parking enforcement intelligence for the Bengaluru Traffic Police —
forecast where violations will spike, quantify how much they choke traffic, and tell
patrols exactly where to go <i>before</i> congestion happens.
</p>

<p align="center">
  <em>Gridlock Hackathon 2.0 (Flipkart x BTP) — Problem Statement 1</em>
</p>

---

## 1. The Problem

On-street illegal parking near markets, metro stations and junctions chokes Bengaluru's
roads. Today enforcement is **reactive and patrol-based**:

- No city-wide view of *where* and *when* violations cluster
- No way to *quantify* how much a parking violation actually hurts traffic flow
- No data-driven way to *prioritize* which zones to enforce first

**Our question:** *Can AI detect illegal-parking hotspots and quantify their impact on
traffic flow, so enforcement becomes proactive and targeted?*

---

## 2. Our Solution — in one picture

```
   Raw BTP violations CSV  (≈298k records, Nov 2023 – Apr 2024)
              │
              ▼
   ┌─────────────────────────────────────────────────────────┐
   │  CLEAN  → parse violations, UTC→IST, drop false positives │
   │  ZONE   → junction (primary) / ~1km grid fallback         │
   │  FEATURES → zero-filled grid + 17 leakage-safe features    │
   └─────────────────────────────────────────────────────────┘
              │
              ▼
   ┌─────────────────────────────────────────────────────────┐
   │  FORECAST  → XGBoost (Tweedie) predicts violations/zone   │
   │  CONGESTION INDEX → % road capacity lost (PCU + Indo-HCM) │
   │  RISK · PRIORITY · PATROL PLAN                             │
   │  SHAP "why this zone" · repeat-offender intelligence       │
   └─────────────────────────────────────────────────────────┘
              │
              ▼
        Streamlit decision dashboard  (8 tabs, used by BTP)
```

**In one line:** historical violations → forecast → **congestion impact** → priority →
**"Deploy Team A to KR Market Junction"** — on a live dashboard.

---

## 3. What makes it strong

| Capability | Why it matters |
|---|---|
| **Violation forecasting** (XGBoost, Tweedie objective) | Predicts the *next* window per zone; handles the heavy zero-inflation of sparse parking data |
| **Parking Congestion Impact Index** | Estimates **% road capacity lost** using PCU + Indo-HCM saturation-flow principles — *answers the core "impact on traffic flow" ask* with **no external data** |
| **Hotspot heatmaps** (current + predicted) | Instant city-wide picture; filter by station / violation type / date |
| **Smart patrol allocation** | Assigns N teams to top zones with spatial spread — turns prediction into a daily action plan |
| **Repeat-offender intelligence** | Separates *willful* blockers (towing) from *infrastructure* problems (signage/space) |
| **SHAP "why this zone?"** | Every alert is explainable — builds trust with enforcement officers |
| **Honest evaluation** | Benchmarked against a seasonal-naive baseline on a held-out *future* period |

---

## 4. Results (held-out future test — real data)

Trained on the earliest 80% of the timeline, tested on the unseen later 20% (~169k zone-windows):

| Metric | Seasonal-naive baseline | **ParkFlow-AI** |
|---|---|---|
| MAE (lower better) | 0.378 | **0.376** |
| RMSE (lower better) | 2.039 | **1.966** |
| R² (higher better) | 0.294 | **0.344** |
| Poisson deviance (lower better) | 1.644 | **0.935** |
| Hotspot PR-AUC (higher better) | 0.354 | **0.389** |

**Beats the baseline on every error/calibration metric and on hotspot detection (PR-AUC).**
We report against a real baseline on a *future* window — no data leakage, no inflated numbers.

> Scale of the problem surfaced from the data: **243,405** confirmed parking violations across
> **701** enforcement zones; **8,089** repeat-offender vehicles accounting for **~30%** of all violations.

---

## 5. The Dashboard

A single decision cockpit for traffic command. Eight tabs:

| Tab | What an officer sees |
|---|---|
| **Overview** | City KPIs + live hotspot map |
| **Hotspot Analysis** | Graded violation heatmap with station / type / date filters |
| **Prediction Center** | Next-window forecast + congestion index + *"why this zone?"* |
| **Enforcement** | Today's patrol deployment plan + priority ranking |
| **Analytics Center** | Hour / day / week trends, peak-time heatmaps |
| **Junction Risk** | Per-junction severity, peak hour, dominant vehicle |
| **Repeat Offenders** | Chronic violators + willful-vs-infrastructure signal |
| **Model** | Baseline-vs-model metrics, SHAP, diagnostics |

<!-- SCREENSHOTS: drop images in docs/screenshots/ with these names -->
### Screens

| Overview & hotspot map | Predicted hotspots + Congestion Index |
|---|---|
| ![Overview](docs/screenshots/01-overview.png) | ![Prediction](docs/screenshots/03-prediction-congestion.png) |

| Patrol deployment plan | "Why this zone?" (SHAP) |
|---|---|
| ![Enforcement](docs/screenshots/04-enforcement-patrol.png) | ![SHAP](docs/screenshots/05-why-this-zone.png) |

| Violation heatmap (filters) | Model vs baseline + diagnostics |
|---|---|
| ![Heatmap](docs/screenshots/02-hotspot-heatmap.png) | ![Model](docs/screenshots/06-model-metrics.png) |

**More views**

| Junction risk assessment | Repeat-offender intelligence |
|---|---|
| ![Junction Risk](docs/screenshots/07-junction-risk.png) | ![Repeat Offenders](docs/screenshots/08-repeat-offenders.png) |

| Temporal analytics (peak hours / day×hour) | Feature importance + SHAP |
|---|---|
| ![Analytics](docs/screenshots/09-analytics.png) | ![Feature importance](docs/screenshots/10-feature-importance.png) |

---

## 6. How it works (technical)

**Forecasting target** — expected violation **count** per zone per 3-hour window (a supervised
regression). Risk bands (Low/Medium/High/Critical) are *derived* from the count — we never output
a fake "risk %".

**The 17 model features**
- **Time (5):** hour, day-of-week, month, week-of-year, weekend flag
- **History (5):** previous-bin / day / week lags + 7-day & 30-day rolling means *(leakage-safe — only past values)*
- **Zone profile (7):** historical mean, frequency encoding, junction flag, heavy-vehicle share, **carriageway-blocking share**, **repeat-offender density**, **violation growth rate**

**Why these choices**
- **Complete-grid zero-fill** before any feature — so the model learns *quiet* periods, not just busy ones (the single most important correctness step).
- **Tweedie objective** (compound Poisson-Gamma) — fits zero-inflated, overdispersed counts better than plain Poisson.
- **Time-based split** + past-only lags — evaluation simulates real forecasting with **zero leakage**.

**Parking Congestion Impact Index** *(the differentiator)*
```
pcu_load        = predicted_violations × mean_PCU × road_factor
capacity_lost % = max_cap × (1 − e^(−pcu_load / saturation))     <- Indo-HCM / HCM principle
congestion_index (0–100) = capacity_lost% / max_cap × 100
```
PCU = Passenger-Car-Units (a bus ≈ 3.5, a scooter ≈ 0.5). Built **only from the provided data +
standard traffic-engineering constants** — no external datasets or APIs.

---

## 7. Run it yourself

```bash
# 1. setup
python -m venv .venv && .venv/Scripts/activate         # Windows
pip install -e ".[dashboard,dev]"

# 2. run the full pipeline (clean -> train -> evaluate -> write artifacts)
parkflow run

# 3. launch the dashboard
streamlit run app/streamlit_app.py                      # http://localhost:8501
```
Dataset path is set in `config/config.yaml` (`paths.raw_data`). Tests: `pytest -q` (10 passing).

---

## 8. Project structure

```
ParkFlow-AI/
├── config/config.yaml      # every tunable (bins, model, risk bands, PCU weights)
├── dataset/                # raw BTP CSV (gitignored)
├── src/parkflow/           # pipeline package
│   ├── preprocessing.py    # clean, parse, UTC->IST, validation filter
│   ├── spatial.py          # junction / grid zoning
│   ├── features.py         # zero-fill grid + 17 features
│   ├── model.py            # XGBoost Tweedie forecaster
│   ├── baseline.py         # seasonal-naive benchmark
│   ├── evaluation.py       # MAE/RMSE/R2/Poisson + Top-K/PR-AUC
│   ├── intelligence.py     # congestion index, risk, priority, patrol
│   ├── analytics.py        # junction risk, repeat offenders, trends
│   ├── explain.py          # SHAP "why this zone"
│   └── pipeline.py         # orchestrates everything -> artifacts/
├── app/streamlit_app.py    # 8-tab dashboard (reads artifacts only)
└── tests/                  # correctness tests (zero-fill, no leakage, …)
```

---

## 9. Compliance & honesty

- **Uses only the HackerEarth-provided dataset** — no external datasets or APIs.
- **No data leakage** — strict time-based split, past-only features.
- **Benchmarked honestly** against a seasonal-naive baseline on a future window.
- **Scope:** congestion impact is *estimated* via PCU + HCM traffic-engineering principles
  (the dataset has no live speed/queue telemetry); it's a transparent, defensible model — not a
  fabricated number.

---

## 10. Tech stack

Python · pandas · XGBoost · scikit-learn · SHAP · Streamlit · Plotly

<p align="center"><b>From reactive patrols to proactive, data-driven parking enforcement.</b></p>
