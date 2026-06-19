# ParkFlow-AI
Predicting and Mitigating Parking-Induced Traffic Disruptions.

ParkFlow-AI turns historical parking-violation records into **proactive enforcement
intelligence**: it forecasts violation intensity per junction × time-window, derives
hotspots and risk bands, ranks enforcement priority, and recommends patrol deployment —
served through a Streamlit dashboard. See [PRD.md](PRD.md) for the full spec.

## Architecture

```
raw CSV ─► clean ─► spatial zones ─► features (ZERO-FILLED grid + leakage-safe lags)
        ─► seasonal-naive baseline + XGBoost (count:poisson) ─► evaluate
        ─► one-step forecast ─► risk band · disruption proxy · priority · patrol
        ─► artifacts/ ─► Streamlit dashboard (reads precomputed outputs)
```

Key correctness decisions (from the PRD):
- **Zero-filled complete grid** before any lag feature — the model must see quiet cells.
- **Regression** for a count, then risk bands — never a fake "risk %".
- **`count:poisson`** objective + a **mandatory seasonal-naive baseline** to beat.
- **Leakage-safe** temporal split; lags use only past values.
- **Disruption proxy** is a transparent heuristic, explicitly *not* measured congestion.

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/activate        # Windows
pip install -e ".[dashboard,dev]"                     # or: pip install -r requirements.txt

parkflow info                 # show resolved config (data path, bin size, ...)
parkflow run                  # full pipeline -> artifacts/
streamlit run app/streamlit_app.py
```

Data: the raw violations CSV path is set in `config/config.yaml` (`paths.raw_data`).
The loader handles the real BTP schema — JSON-array `violation_type`, UTC timestamps
(converted to `Asia/Kolkata`), and `No Junction` grid fallback.

## Layout

| Path | Purpose |
|---|---|
| `src/parkflow/` | pipeline package (config, preprocessing, spatial, features, model, intelligence, pipeline, cli) |
| `config/config.yaml` | every tunable (bin size, model params, risk bands, weights) |
| `app/streamlit_app.py` | dashboard (reads `artifacts/` only) |
| `tests/` | pytest suite (zero-fill completeness, lag leakage, cleaning) |
| `artifacts/` | pipeline outputs: forecast, patrol plan, metrics, model |

## Testing

```bash
pytest -q
```
