<!-- Optional: drop a logo image at docs/logo.png and it will render here. -->
<h1 align="center">ParkFlow-AI</h1>

<p align="center">
<b>Predicting &amp; Mitigating Parking-Induced Traffic Disruptions</b><br/>
AI-driven parking-enforcement intelligence for the Bengaluru Traffic Police. It forecasts where
violations will spike, quantifies how much they choke traffic, and tells patrols exactly where to
go — turning reactive enforcement into proactive, data-driven deployment.
</p>

<p align="center"><em>Gridlock Hackathon 2.0 (Flipkart x BTP) — Problem Statement 1</em></p>

---

## 1. Problem Statement

On-street illegal parking near markets, metro stations and junctions chokes Bengaluru's roads.
Enforcement today is **reactive and patrol-based**:

- No city-wide view of *where* and *when* violations cluster.
- No way to *quantify* how much a parking violation hurts traffic flow.
- No data-driven way to *prioritize* which zones to enforce first.

> **The ask:** *Can AI detect illegal-parking hotspots and quantify their impact on traffic flow,
> so enforcement becomes proactive and targeted?*

---

## 2. Solution

ParkFlow-AI converts ~298k historical BTP violation records into forward-looking enforcement
intelligence. It aggregates violations to **junction × time-window** cells, forecasts the next
window's violation count with a gradient-boosted model, converts that into an estimated **road-capacity
loss** (the congestion link), and produces a ranked patrol plan — all served on a Streamlit dashboard
that reads only precomputed artifacts.

**Pipeline (high level)**

```mermaid
flowchart LR
  A[Raw BTP CSV] --> B[Preprocessing]
  B --> C[Spatial Zoning]
  C --> D["Feature Engineering<br/>(zero-fill + 17 features)"]
  D --> E["XGBoost Tweedie<br/>Forecast"]
  E --> F["Intelligence Layers<br/>congestion · risk · priority · patrol"]
  E --> G[SHAP Explainability]
  F --> H[(artifacts/)]
  G --> H
  H --> I[Streamlit Dashboard]
```

### 2.1 State of the art

Spatio-temporal violation forecasting is typically framed either as classical time-series
(ARIMA / exponential smoothing — poor with sparse, zero-heavy spatial grids) or as deep
spatio-temporal models (LSTM / STGCN — heavy and data-hungry). For tabular, sparse,
heterogeneous urban-violation data over a few months, **gradient-boosted trees with a
count-appropriate objective** are the practical state of the art: strong accuracy, fast to train,
interpretable via SHAP, and robust to mixed feature types. ParkFlow-AI uses an **XGBoost regressor
with a Tweedie objective** (compound Poisson-Gamma) to handle the zero-inflation and overdispersion
inherent to parking violations, benchmarked against a seasonal-naive baseline.

### 2.2 Architecture

```mermaid
flowchart TD
  subgraph Data
    R[Raw violations CSV]
  end
  subgraph Pipeline["parkflow run - offline"]
    P1[Clean and parse] --> P2[Spatial zoning]
    P2 --> P3[Zero-filled grid + features]
    P3 --> P4[Baseline + XGBoost Tweedie]
    P4 --> P5[Evaluate vs baseline]
    P4 --> P6[Forecast next window]
    P6 --> P7[Congestion · Risk · Priority · Patrol]
    P4 --> P8[SHAP explanations]
    P3 --> P9[Analytics: junctions, offenders, trends]
  end
  subgraph Artifacts["artifacts/"]
    AR[(CSV / Parquet / model.joblib / metrics.json)]
  end
  subgraph UI["Streamlit - online"]
    D[8-tab decision dashboard]
  end
  R --> P1
  P5 --> AR
  P7 --> AR
  P8 --> AR
  P9 --> AR
  AR --> D
```

### 2.3 Data preprocessing techniques

| Technique | Why / where it is used |
|---|---|
| Drop admin columns | Remove non-predictive workflow fields (device id, timestamps, etc.) |
| JSON-array violation parsing | `violation_type` is a JSON array per record; extract the parking tags |
| UTC → IST conversion | So hour-of-day features reflect **local** commute time, not UTC |
| Validation-status filter | Drop `rejected` / `duplicate` records to remove false positives |
| Deduplication | Remove repeated captures of the same event (vehicle+lat+lon+time) |
| Missing-value imputation | Fill missing junction / police-station labels with a sentinel |
| Spatial zoning (junction + grid fallback) | Map each event to an enforcement zone; junctions primary, ~1 km grid otherwise |
| Frequency encoding | Encode the high-cardinality **junction/zone name** as a numeric feature |
| Complete-grid zero-fill | Create rows for quiet zone-windows so the model learns lulls, not only spikes |
| Lag / rolling features | Capture temporal momentum from past windows (leakage-safe, past-only) |

### 2.4 Modelling equations

**Target** — expected violation count $\hat{y}$ for a zone in a future time window.

**Tweedie objective** (variance–mean relation, $1 < p < 2$ = compound Poisson-Gamma, matching
zero-inflated counts):

$$V(\mu) = \phi\,\mu^{p}, \qquad 1 < p < 2$$

**Parking Congestion Impact Index** (PCU = Passenger-Car-Units; $r$ = road factor; $C_{\max}$ = max
capacity-loss percentage; $S$ = saturation constant — Indo-HCM saturation-flow principle). Here
$\hat{\rho}$ is the estimated road-capacity reduction:

$$L = \hat{y}\cdot \overline{\text{PCU}}\cdot r$$

$$\hat{\rho} = C_{\max}\left(1 - e^{-L/S}\right)$$

$$\text{CongestionIndex} = 100 \cdot \frac{\hat{\rho}}{C_{\max}}$$

**Enforcement priority** (min-max normalised components $\tilde{y},\tilde{h}$; $j$ = junction flag):

$$P = 100\left(0.6\,\tilde{y} + 0.3\,\tilde{h} + 0.1\,j\right)$$

### 2.5 Evaluation and validation

- **Time-based split** — train on the earliest 80% of the timeline, test on the unseen latest 20%
  (simulates real forecasting; **no leakage** — all lags use past values only).
- **Baseline** — a seasonal-naive predictor (zone × hour-of-week mean) the model must beat.
- **Metrics** — MAE, RMSE, R², Poisson deviance (regression) + Top-K hit-rate and hotspot PR-AUC
  (ranking metrics, better suited to sparse, imbalanced hotspot data).

---

## 3. What makes it strong

| Strength | Why it matters |
|---|---|
| **Parking Congestion Impact Index** | Estimates **% road capacity lost** via PCU + Indo-HCM principles — directly answers the "impact on traffic flow" ask, with **no external data** |
| **Tweedie forecasting** | Handles zero-inflated, overdispersed counts far better than plain Poisson/Gaussian |
| **Zero-filled grid** | The model learns *quiet* windows, not just busy ones — the key correctness step |
| **SHAP explainability** | Every alert is explainable per zone — builds trust with enforcement officers |
| **Repeat-offender intelligence** | Separates willful blockers (towing) from infrastructure issues (signage/space) |
| **Smart patrol allocation** | Greedy + spatial-spread deployment plan — turns a forecast into a daily action list |
| **Honest evaluation** | Beats a real baseline on a held-out *future* window — no leakage, no inflated numbers |

---

## 4. How it works

```mermaid
flowchart TD
  E[Violation events] --> A["Aggregate to zone × 3h bin"]
  A --> Z["Zero-fill complete grid"]
  Z --> F["17 features:<br/>time · lags · zone-profile"]
  F --> S["Time split (80/20)"]
  S --> M["XGBoost Tweedie"]
  M --> Y["Predicted count ŷ"]
  Y --> R["Risk band"]
  Y --> C["Congestion index"]
  Y --> PR["Priority score"]
  PR --> PT["Patrol plan (top-N, spatial spread)"]
  M --> SH["SHAP why-this-zone"]
```

1. **Aggregate** events to `zone × 3-hour` counts, then **zero-fill** the full grid.
2. Engineer **17 leakage-safe features** (time, lags/rolling, zone profile).
3. Split by time, train the **Tweedie XGBoost** vs the **seasonal-naive baseline**.
4. Forecast the **next window**, then derive **risk band**, **congestion index**, **priority**, **patrol plan**.
5. Compute **SHAP** explanations; write everything to `artifacts/` for the dashboard.

---

## 5. Results

Held-out future test (~169k zone-windows; trained on the earliest 80%):

| Metric | Seasonal-naive baseline | **ParkFlow-AI** |
|---|---|---|
| MAE (lower better) | 0.378 | **0.376** |
| RMSE (lower better) | 2.039 | **1.966** |
| R² (higher better) | 0.294 | **0.344** |
| Poisson deviance (lower better) | 1.644 | **0.935** |
| Hotspot PR-AUC (higher better) | 0.354 | **0.389** |

**Beats the baseline on every error/calibration metric and on hotspot detection (PR-AUC).**

> From the data: **243,405** confirmed parking violations across **701** zones; **8,089**
> repeat-offender vehicles (~**30%** of all violations).

---

## 6. The Dashboard

A single decision cockpit (8 tabs): Overview, Hotspot Analysis, Prediction Center, Enforcement,
Analytics Center, Junction Risk, Repeat Offenders, Model.

**Overview — city KPIs + hotspot map**
![Overview](docs/screenshots/01-overview.png)

**Hotspot Analysis — violation heatmap with filters**
![Heatmap](docs/screenshots/02-hotspot-heatmap.png)

**Prediction Center — forecast + Parking Congestion Impact Index**
![Prediction](docs/screenshots/03-prediction-congestion.png)

**Enforcement — patrol deployment plan**
![Enforcement](docs/screenshots/04-enforcement-patrol.png)

**Why this zone? — SHAP explanation**
![SHAP](docs/screenshots/05-why-this-zone.png)

**Model — baseline vs model + diagnostics**
![Model](docs/screenshots/06-model-metrics.png)

**Junction Risk assessment**
![Junction Risk](docs/screenshots/07-junction-risk.png)

**Repeat-offender intelligence**
![Repeat Offenders](docs/screenshots/08-repeat-offenders.png)

**Analytics Center — temporal trends + intensity grid**
![Analytics](docs/screenshots/09-analytics.png)

**Feature importance + global SHAP**
![Feature importance](docs/screenshots/10-feature-importance.png)

---

## 7. Project structure

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
├── tests/                  # correctness tests (zero-fill, no leakage, …)
└── docs/screenshots/       # dashboard screenshots
```

---

## 8. Setup & run

```bash
# 1. setup
python -m venv .venv && .venv/Scripts/activate          # Windows
pip install -e ".[dashboard,dev]"

# 2. run the full pipeline (clean -> train -> evaluate -> write artifacts)
parkflow run

# 3. launch the dashboard
streamlit run app/streamlit_app.py                       # http://localhost:8501

# tests
pytest -q
```

Dataset path is set in `config/config.yaml` (`paths.raw_data`).
**Compliance:** uses only the provided dataset — no external datasets or APIs.

---

## 9. Links

| Resource | Link |
|---|---|
| Presentation (PPT) | _add link_ |
| Demo video | _add link_ |
| Live deployment | _add link_ |
| GitHub repository | https://github.com/SajeevSenthil/ParkFlow-AI |

---

## 10. Team

| Name | Role |
|---|---|
| _add name_ | _add role_ |
| _add name_ | _add role_ |
| _add name_ | _add role_ |

---

<p align="center"><b>From reactive patrols to proactive, data-driven parking enforcement.</b></p>
