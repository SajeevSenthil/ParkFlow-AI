# ParkFlow-AI — Update Log

> All changes implemented in one session on 2026-06-20.
> Run `parkflow run` then `streamlit run app/streamlit_app.py` to see everything.

---

## Files Modified

| File | Nature of change |
|---|---|
| `config/config.yaml` | Added `preprocessing` block + extended `disruption` block with CIS config |
| `src/parkflow/config.py` | Added `PreprocessingCfg` dataclass; extended `DisruptionCfg` with 5 new fields |
| `src/parkflow/preprocessing.py` | Added validation-status filter step (Step 3b) |
| `src/parkflow/features.py` | Added 3 new zone-static model features |
| `src/parkflow/intelligence.py` | Replaced `disruption_proxy` with data-driven Congestion Impact Score (CIS) |
| `src/parkflow/pipeline.py` | Added repeat-offender artifact, test-predictions artifact, staleness timestamps |
| `app/streamlit_app.py` | All dashboard improvements — maps, new tab, new charts |

---

## 1. `config/config.yaml`

### Added — `preprocessing` block
```yaml
preprocessing:
  approved_only: false
  exclude_statuses: ["rejected", "duplicate"]
```
Controls whether the pipeline trains on confirmed-only violations or all records.

### Extended — `disruption` block
Added under the existing vehicle weights:
- `violation_severity_weights` — per-type carriageway-blocking severity (DOUBLE PARKING=3.0 … WRONG PARKING=1.0)
- `default_violation_severity: 1.0`
- `peak_hours_morning: [7, 8, 9, 10]`
- `peak_hours_evening: [17, 18, 19]`
- `peak_hour_multiplier: 1.5`
- `repeat_offender_threshold: 2`

---

## 2. `src/parkflow/config.py`

### Added — `PreprocessingCfg` dataclass
```python
@dataclass(frozen=True)
class PreprocessingCfg:
    approved_only: bool = False
    exclude_statuses: list[str] = ...
```
Loaded from `config.yaml → preprocessing:` with safe fallback defaults.

### Extended — `DisruptionCfg`
Five new typed fields: `violation_severity_weights`, `default_violation_severity`,
`peak_hours_morning`, `peak_hours_evening`, `peak_hour_multiplier`, `repeat_offender_threshold`.

### Updated — `Config`
Added `preprocessing: PreprocessingCfg` field. `Config.load()` now parses both blocks.

---

## 3. `src/parkflow/preprocessing.py`

### Added — Step 3b: validation status filter
Runs after parking-type filtering, before deduplication.

- If `approved_only: true` → keep only `validation_status == "approved"` rows.
- If `approved_only: false` (default) → drop rows with statuses in `exclude_statuses`
  (removes `rejected` and `duplicate` by default — confirmed false positives).
- Logs how many rows were removed and which mode was active.

Removes ~50,074 false-positive records (`rejected` + `duplicate`) from the default run.

---

## 4. `src/parkflow/features.py`

### Added — 3 new zone-static model features (in `add_zone_statics()`)

All computed from the training window only — no leakage.

| Feature | Computation | What it captures |
|---|---|---|
| `carriageway_block_share` | Fraction of violations that are DOUBLE PARKING / PARKING IN A MAIN ROAD / NEAR SIGNAL / NEAR CROSSING / OPPOSITE PARKED | Zones where violations directly block moving lanes |
| `repeat_offender_density` | `log1p(count of vehicles with > threshold visits at this zone)` | Chronic chronic blockage vs one-off incidents |
| `violation_growth_rate` | `roll_7d_mean / roll_30d_mean` — clipped 0–5 | Whether the zone is trending worse or improving |

Total model features increased from **14 → 17**.

---

## 5. `src/parkflow/intelligence.py`

### Replaced — `disruption_proxy()` → Congestion Impact Score (CIS)

**Old formula:** `predicted_violations × vehicle_weight × road_weight`
(three config constants multiplied together — added nothing over raw prediction)

**New formula — CIS built from 5 data-observed signals:**

```
CIS = simultaneous_density_norm
    × violation_severity_weight
    × vehicle_blocking_weight
    × peak_hour_multiplier
    × (1 + log1p(repeat_offender_count))
```

| Signal | Source | What it measures |
|---|---|---|
| `simultaneous_density_norm` | Max violations in any single 3h bin at zone, normalised 0–1 | How many vehicles block simultaneously |
| `violation_severity_weight` | Mean carriageway-blocking severity of violation types at zone | Whether violations directly obstruct moving lanes |
| `vehicle_blocking_weight` | Mean vehicle weight (bus=5, LGV=3, car=2…) at zone | Road space occupied per vehicle |
| `peak_hour_multiplier` | 1.5 if forecast bin falls in 07–11 or 17–20 IST, else 1.0 | Impact amplified during rush hours |
| `repeat_offender_factor` | `log1p(vehicles > threshold at zone)` | Chronic vs accidental congestion |

Result is min-max scaled to **0–100** for readability.
Stored as `disruption_proxy` column — no downstream changes needed.

Added four private helper functions:
`_zone_vehicle_weight()`, `_zone_violation_severity()`,
`_zone_simultaneous_density()`, `_zone_repeat_offender_factor()`

---

## 6. `src/parkflow/pipeline.py`

### Added — repeat offender artifact
After `analytics_events` is written, computes and writes `artifacts/repeat_offenders.csv`:
- Columns: `vehicle_number`, `violation_count`, `unique_zones`, `top_zone`, `vehicle_type`, `last_seen`
- Filtered to vehicles with `violation_count > repeat_offender_threshold`
- Sorted descending by violation count

### Added — test predictions artifact
After model evaluation, writes `artifacts/test_predictions.csv`:
- Columns: `zone`, `bin_start`, `violation_count` (actual), `predicted`, `error`
- Used by the Model tab's actual-vs-predicted diagnostic chart

### Added — staleness timestamps in `metrics.json`
Two new keys written at pipeline run time:
```json
{
  "pipeline_run_at": "2026-06-20T00:05:12.345678",
  "data_date_range": { "from": "...", "to": "..." }
}
```

---

## 7. `app/streamlit_app.py`

### New artifact loaded
`repeat_offenders = load("repeat_offenders")` added at startup.

### New tab added
Tab list changed from 7 → **8 tabs**:
`Overview | Hotspot Analysis | Prediction Center | Enforcement | Analytics Center | Junction Risk | Repeat Offenders | Model`

---

### Tab 1 — Overview: map upgrade
**Before:** `st.map()` — plain blue dots, no size, no color, no tooltip.
**After:** `px.scatter_mapbox` with:
- Bubble **size** = `historical_violations`
- Bubble **color** = predicted next-window risk (Critical=red, High=orange, Medium=yellow, Low=green)
- Hover tooltip: zone name, historical count, predicted violations

### Tab 2 — Hotspot Analysis: new filters + data quality donut
- Added **"Show" radio**: `All records` vs `Approved only` — lets analyst exclude unvalidated violations from heatmap
- Added **4th filter column** (was 3 columns, now 4)
- Added **data quality expander** with a donut chart showing `approved / rejected / duplicate / unreviewed / processing` split of all records

### Tab 3 — Prediction Center: added the missing map
**Before:** Table of top 25 zones + plain bar chart. No spatial view.
**After:**
- `px.scatter_mapbox` — predicted hotspot bubble map (size=predicted count, color=risk band)
- Risk band bar chart moved into a two-column layout beside the ranked table
- Cleaner layout: map → [bar chart | table]

### Tab 4 — Enforcement: patrol map + team cards
**Before:** Plain text lines — "Team A → Zone X · priority 87 · risk High"
**After:**
- `px.scatter_mapbox` showing each team's assigned zone with team-color-coded markers
- Text team cards kept below with risk emoji indicators (🔴🟠🟡🟢)
- Priority ranking table unchanged

### Tab 5 — Analytics Center: three new charts
**Before:** Hour bar + Day-of-week bar + Weekly line + Vehicle type bar (using `st.bar_chart`)
**After:**
1. **Hour bar** — upgraded to Plotly with red color scale
2. **Day-of-week bar** — upgraded to Plotly with blue color scale
3. **NEW — Hour × Day-of-week intensity heatmap** — shows exactly when enforcement pressure should be highest; every hour × day combination colored by total violations
4. **NEW — Violation type × Hour heatmap** — rows ordered by carriageway severity (DOUBLE PARKING first); shows which violation types spike at which hours
5. Weekly line chart + vehicle type bar retained

### Tab 6 — Junction Risk: added map
**Before:** Table of top 30 junctions + bar chart. No spatial view.
**After:**
- `px.scatter_mapbox` of junction centroids (size=historical violations, color=risk)
- Hover shows zone, priority score, peak hour, dominant vehicle type
- Table + bar chart moved into a two-column layout below the map

### Tab 7 — Repeat Offenders (NEW TAB)
Complete new tab with:
- **3 KPI metrics**: repeat offender count, max violations (single vehicle), avg unique zones per offender
- **Top 25 table**: vehicle number, violation count, unique zones, top zone, vehicle type, last seen
- **Horizontal bar chart**: top 15 zones by repeat-offender violation load
- **`density_mapbox` heatmap**: spatial density of where repeat vehicles operate (using events lat/lon)
- **Vehicle type bar**: which vehicle types are the most chronic offenders

### Tab 8 — Model: diagnostic charts + staleness warning
**Before:** Metrics table + verdict text + feature importance bar chart.
**After:**
- **Staleness banner** shown at top of app: info if <7 days old, warning if >7 days old
- **Data date range** caption under metrics table
- **NEW — Actual vs Predicted scatter** (sampled to 5k points): OLS trendline + perfect-prediction diagonal in red. Points above diagonal = over-predicted; below = under-predicted
- **NEW — Error distribution histogram**: symmetric around 0 = well-calibrated; right skew = systematic over-prediction
- **Feature importance** upgraded from `st.bar_chart` to Plotly horizontal bar with color scale; shows top 17 features

---

## New Artifacts Produced by `parkflow run`

| Artifact | Description |
|---|---|
| `artifacts/repeat_offenders.csv` | Vehicles with >2 violations: count, zones, top zone, vehicle type, last seen |
| `artifacts/test_predictions.csv` | Held-out test set: actual vs predicted violations + error column |

`metrics.json` now also contains `pipeline_run_at` and `data_date_range`.

---

## How to Apply

```bash
# Re-run the full pipeline to regenerate all artifacts with new features + CIS
parkflow run

# Launch the updated dashboard
streamlit run app/streamlit_app.py
```

> Set `preprocessing.approved_only: true` in `config/config.yaml` to train on
> BTP-confirmed violations only (removes ~42% unreviewed + rejected records).
