"""ParkFlow-AI Streamlit dashboard (reads precomputed artifacts only).

Run:  streamlit run app/streamlit_app.py
The pipeline must have been run first (`parkflow run`) so artifacts/ exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

st.set_page_config(page_title="ParkFlow-AI", page_icon="🅿️", layout="wide")


@st.cache_data
def load(name: str) -> pd.DataFrame:
    for ext in (".parquet", ".csv"):
        p = ART / f"{name}{ext}"
        if p.exists():
            return pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data
def load_metrics() -> dict:
    p = ART / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


if not (ART / "metrics.json").exists():
    st.error("No artifacts found. Run `parkflow run` first.")
    st.stop()

metrics = load_metrics()
forecast = load("future_forecast")
patrol = load("patrol_plan")
hotspots = load("current_hotspots")
junction_risk = load("junction_risk")
events = load("events_analytics")
shap_reasons = load("shap_reasons")
shap_global = load("shap_global")
top_offenders = load("top_offenders")
zone_repeat = load("zone_repeat_offenders")
if len(events) and "created_datetime" in events:
    events["created_datetime"] = pd.to_datetime(events["created_datetime"], errors="coerce")

st.title("🅿️ ParkFlow-AI — Parking Enforcement Intelligence")
st.caption("Spatial-temporal forecasting of parking violations → hotspots, priority, patrols.")

# --- KPI row ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total violations", f"{len(events):,}" if len(events) else int(hotspots['historical_violations'].sum()) if len(hotspots) else 0)
c2.metric("Active hotspots (zones)", len(hotspots))
high = int(forecast["risk"].isin(["High", "Critical"]).sum()) if "risk" in forecast else 0
c3.metric("High/Critical zones (next window)", high)
c4.metric("Predicted violations (next window)", round(float(forecast["predicted_violations"].sum()), 1) if len(forecast) else 0)

tabs = st.tabs(
    ["Overview", "Hotspot Analysis", "Prediction Center", "Enforcement",
     "Analytics Center", "Junction Risk", "Repeat Offenders", "Model"]
)

# ============================ Overview ============================
with tabs[0]:
    st.subheader("Current hotspots (historical density)")
    if len(hotspots):
        st.map(hotspots.rename(columns={"zone_lat": "lat", "zone_lon": "lon"})[["lat", "lon"]])
        st.dataframe(hotspots.head(20), use_container_width=True)

# ======================== Hotspot Analysis =======================
with tabs[1]:
    st.subheader("Violation heatmap with filters")
    if len(events):
        f = events.copy()
        cols = st.columns(3)
        stations = ["All"] + sorted(f["police_station"].dropna().astype(str).unique().tolist())
        vtypes = ["All"] + sorted(f["violation_type"].dropna().astype(str).unique().tolist())
        sel_station = cols[0].selectbox("Police station", stations)
        sel_vtype = cols[1].selectbox("Violation type", vtypes)
        dmin, dmax = f["created_datetime"].min(), f["created_datetime"].max()
        date_range = cols[2].date_input("Date range", value=(dmin.date(), dmax.date()))

        if sel_station != "All":
            f = f[f["police_station"].astype(str) == sel_station]
        if sel_vtype != "All":
            f = f[f["violation_type"].astype(str) == sel_vtype]
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            lo, hi = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
            f = f[(f["created_datetime"] >= lo) & (f["created_datetime"] < hi)]

        st.caption(f"{len(f):,} violations match the filter")
        if len(f):
            sample = f.sample(min(len(f), 20000), random_state=0)
            fig = px.density_mapbox(
                sample, lat="latitude", lon="longitude", radius=7,
                center=dict(lat=float(sample["latitude"].median()), lon=float(sample["longitude"].median())),
                zoom=10, mapbox_style="open-street-map", height=520,
            )
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

            top = (
                f.groupby("zone").size().rename("violations").reset_index()
                .sort_values("violations", ascending=False).head(15)
            )
            st.dataframe(top, use_container_width=True)

# ======================== Prediction Center ======================
with tabs[2]:
    st.subheader("Predicted hotspots — next window")
    if len(forecast):
        win = forecast["forecast_window_start"].iloc[0] if "forecast_window_start" in forecast else ""
        st.caption(f"Forecast window starting: {win}")
        st.dataframe(
            forecast[["zone", "predicted_violations", "risk", "congestion_index",
                      "est_capacity_reduction_pct", "priority_score"]]
            .sort_values("congestion_index", ascending=False).head(25),
            use_container_width=True,
        )
        risk_counts = forecast["risk"].value_counts().reindex(["Low", "Medium", "High", "Critical"]).fillna(0)
        st.bar_chart(risk_counts)
        st.caption(
            "**Parking Congestion Impact Index (0–100)** estimates lost road capacity from "
            "PCU (passenger-car-units) × HCM saturation-flow principles — derived only from the "
            "provided data + standard traffic-engineering constants (no external data)."
        )

        # --- Why this zone? (SHAP) ---
        if len(shap_reasons):
            st.markdown("#### Why is a zone flagged? (SHAP)")
            zsel = st.selectbox("Zone", forecast.sort_values("predicted_violations", ascending=False)["zone"].head(40).tolist())
            r = shap_reasons[shap_reasons["zone"] == zsel].sort_values("rank")
            for _, row in r.iterrows():
                arrow = "▲" if row["direction"] == "increases" else "▼"
                st.markdown(
                    f"{arrow} **{row['feature']}** = {row['feature_value']:.2f}  "
                    f"→ {row['direction']} the forecast  (SHAP {row['shap']:+.2f})"
                )

# =========================== Enforcement =========================
with tabs[3]:
    st.subheader("Today's patrol deployment plan")
    if len(patrol):
        for _, r in patrol.iterrows():
            st.markdown(
                f"**{r['team']} → {r['zone']}**  ·  priority {r['priority_score']}  ·  "
                f"risk {r.get('risk','')}  ·  predicted {r['predicted_violations']}"
            )
    st.subheader("Enforcement priority ranking")
    if len(forecast):
        st.dataframe(
            forecast[["zone", "priority_score", "risk", "predicted_violations",
                      "congestion_index", "est_capacity_reduction_pct"]].head(25),
            use_container_width=True,
        )

# ========================= Analytics Center ======================
with tabs[4]:
    st.subheader("Temporal violation trends")
    if len(events):
        ts = events["created_datetime"]
        by_hour = events.assign(hour=ts.dt.hour).groupby("hour").size()
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        by_dow = events.assign(dow=ts.dt.dayofweek).groupby("dow").size()
        by_dow.index = [dow_names[i] for i in by_dow.index]
        by_week = events.assign(week=ts.dt.to_period("W").astype(str)).groupby("week").size()

        a, b = st.columns(2)
        with a:
            st.caption("Violations by hour of day")
            st.bar_chart(by_hour)
        with b:
            st.caption("Violations by day of week")
            st.bar_chart(by_dow)
        st.caption("Weekly trend")
        st.line_chart(by_week)

        st.caption("Top vehicle types")
        st.bar_chart(events["vehicle_type"].astype(str).str.upper().value_counts().head(8))

# ========================== Junction Risk ========================
with tabs[5]:
    st.subheader("Junction risk assessment")
    if len(junction_risk):
        st.dataframe(junction_risk.head(30), use_container_width=True)
        topj = junction_risk.head(15).set_index("zone")
        metric = "priority_score" if "priority_score" in topj else "historical_violations"
        st.caption(f"Top junctions by {metric}")
        st.bar_chart(topj[metric])
    else:
        st.info("No junction-level rows available.")

# ========================== Repeat Offenders =====================
with tabs[6]:
    st.subheader("Repeat-offender intelligence")
    if len(zone_repeat):
        total = int(zone_repeat["total_violations"].sum())
        repeat = int(zone_repeat["repeat_violations"].sum())
        a, b = st.columns(2)
        a.metric("Violations from repeat offenders", f"{repeat:,}", f"{100*repeat/max(total,1):.1f}% of all")
        b.metric("Tracked vehicles (top list)", len(top_offenders) if len(top_offenders) else 0)
        st.markdown(
            "High **repeat** share → willful disregard (candidate for **towing**). "
            "High **unique** share → likely an **infrastructure** gap (signage / no legal parking)."
        )
        st.caption("Zones by repeat-offender share")
        st.dataframe(
            zone_repeat[["zone", "total_violations", "unique_vehicles",
                         "repeat_offender_share_pct", "signal"]].head(20),
            use_container_width=True,
        )
    if len(top_offenders):
        st.caption("Top repeat-offending vehicles (anonymized IDs)")
        st.dataframe(top_offenders, use_container_width=True)

# ============================== Model ============================
with tabs[7]:
    st.subheader("Model vs seasonal-naive baseline (held-out future)")
    if metrics:
        comp = pd.DataFrame({"baseline": metrics["baseline"], "model": metrics["model"]})
        st.dataframe(comp, use_container_width=True)
        verdict = "✅ Model beats baseline" if metrics.get("model_beats_baseline") else "⚠️ Model does NOT beat baseline"
        st.markdown(f"**{verdict}** (lower MAE / Poisson deviance is better; higher R² is better)")
        if "ranking" in metrics:
            st.caption("Ranking metrics (better for sparse, imbalanced hotspot data)")
            st.dataframe(
                pd.DataFrame({"baseline": metrics["ranking"]["baseline"], "model": metrics["ranking"]["model"]}),
                use_container_width=True,
            )
            st.caption("PR-AUC favours the model (calibration); Top-K is competitive for the seasonal mean at fine granularity.")
    if len(shap_global):
        st.subheader("Global feature importance (mean |SHAP|)")
        st.bar_chart(shap_global.set_index("feature")["mean_abs_shap"])
    else:
        fi = load("feature_importance")
        if len(fi):
            st.subheader("Top features")
            st.bar_chart(fi.set_index("feature")["importance"])
