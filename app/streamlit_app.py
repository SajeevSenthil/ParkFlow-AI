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
repeat_offenders = load("repeat_offenders")
if len(events) and "created_datetime" in events:
    events["created_datetime"] = pd.to_datetime(events["created_datetime"], errors="coerce")

st.title("🅿️ ParkFlow-AI — Parking Enforcement Intelligence")
st.caption("Spatial-temporal forecasting of parking violations → hotspots, priority, patrols.")

# --- Model staleness warning ---
if "pipeline_run_at" in metrics:
    try:
        run_at = pd.Timestamp(metrics["pipeline_run_at"])
        days_stale = (pd.Timestamp.now() - run_at).days
        data_to = metrics.get("data_date_range", {}).get("to", "unknown")
        if days_stale > 7:
            st.warning(
                f"⚠️ Model last trained **{days_stale} days ago** "
                f"(data up to {data_to}). Run `parkflow run` to refresh predictions."
            )
        elif days_stale > 0:
            st.info(f"ℹ️ Model trained **{days_stale} day(s) ago** · data up to {data_to}.")
    except Exception:
        pass

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
        map_hot = hotspots.dropna(subset=["zone_lat", "zone_lon"]).copy()

        # Join forecast risk so map dots are colored by predicted next-window risk.
        if len(forecast) and "risk" in forecast.columns:
            map_hot = map_hot.merge(
                forecast[["zone", "risk", "predicted_violations"]],
                on="zone", how="left",
            )
            map_hot["risk"] = map_hot["risk"].fillna("Low")
        else:
            map_hot["risk"] = "Low"
            map_hot["predicted_violations"] = 0

        RISK_COLORS = {"Low": "green", "Medium": "goldenrod", "High": "orange", "Critical": "red"}
        fig_ov = px.scatter_mapbox(
            map_hot.sort_values("historical_violations", ascending=False),
            lat="zone_lat",
            lon="zone_lon",
            size="historical_violations",
            color="risk",
            color_discrete_map=RISK_COLORS,
            hover_name="zone",
            hover_data={
                "historical_violations": True,
                "predicted_violations": True,
                "zone_lat": False,
                "zone_lon": False,
            },
            size_max=40,
            mapbox_style="open-street-map",
            zoom=11,
            height=520,
            title="Parking Violation Hotspots — bubble size = historical count, color = next-window risk",
            category_orders={"risk": ["Critical", "High", "Medium", "Low"]},
        )
        fig_ov.update_layout(margin=dict(l=0, r=0, t=30, b=0), legend_title_text="Next Risk")
        st.plotly_chart(fig_ov, use_container_width=True)
        st.dataframe(hotspots.head(20), use_container_width=True)

# ======================== Hotspot Analysis =======================
with tabs[1]:
    st.subheader("Violation heatmap with filters")
    if len(events):
        f = events.copy()

        # --- Data quality summary ---
        if "validation_status" in f.columns:
            with st.expander("Data quality — validation status breakdown", expanded=False):
                status_counts = (
                    f["validation_status"]
                    .fillna("unreviewed")
                    .astype(str)
                    .str.lower()
                    .value_counts()
                    .reset_index()
                )
                status_counts.columns = ["status", "count"]
                fig_donut = px.pie(
                    status_counts, names="status", values="count",
                    title="Violation Records by Validation Status",
                    color="status",
                    color_discrete_map={
                        "approved": "green", "rejected": "red",
                        "duplicate": "gray", "unreviewed": "lightblue",
                        "processing": "orange", "created1": "khaki",
                    },
                    hole=0.45,
                )
                fig_donut.update_traces(textposition="inside", textinfo="percent+label")
                fig_donut.update_layout(margin=dict(l=0, r=0, t=30, b=0), showlegend=True)
                st.plotly_chart(fig_donut, use_container_width=True)

        # --- Filters row ---
        col1, col2, col3, col4 = st.columns(4)
        stations = ["All"] + sorted(f["police_station"].dropna().astype(str).unique().tolist())
        vtypes = ["All"] + sorted(f["violation_type"].dropna().astype(str).unique().tolist())
        sel_station = col1.selectbox("Police station", stations)
        sel_vtype = col2.selectbox("Violation type", vtypes)

        val_options = ["All records", "Approved only"]
        sel_val = col3.radio("Show", val_options, horizontal=True)

        dmin, dmax = f["created_datetime"].min(), f["created_datetime"].max()
        date_range = col4.date_input("Date range", value=(dmin.date(), dmax.date()))

        if sel_station != "All":
            f = f[f["police_station"].astype(str) == sel_station]
        if sel_vtype != "All":
            f = f[f["violation_type"].astype(str) == sel_vtype]
        if sel_val == "Approved only" and "validation_status" in f.columns:
            f = f[f["validation_status"].astype(str).str.lower() == "approved"]
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

        # --- Predicted hotspot map ---
        map_df = forecast.dropna(subset=["zone_lat", "zone_lon"]).copy()
        map_df = map_df[map_df["predicted_violations"] > 0]
        if len(map_df):
            RISK_COLORS = {"Low": "green", "Medium": "goldenrod", "High": "orange", "Critical": "red"}
            fig_pred = px.scatter_mapbox(
                map_df.sort_values("predicted_violations", ascending=False),
                lat="zone_lat",
                lon="zone_lon",
                size="predicted_violations",
                color="risk",
                color_discrete_map=RISK_COLORS,
                hover_name="zone",
                hover_data={
                    "predicted_violations": True,
                    "priority_score": True,
                    "zone_lat": False,
                    "zone_lon": False,
                },
                size_max=35,
                mapbox_style="open-street-map",
                zoom=11,
                height=520,
                title="Predicted Violation Hotspots — Next Window",
                category_orders={"risk": ["Critical", "High", "Medium", "Low"]},
            )
            fig_pred.update_layout(margin=dict(l=0, r=0, t=30, b=0), legend_title_text="Risk")
            st.plotly_chart(fig_pred, use_container_width=True)
        else:
            st.info("No zones with predicted violations > 0.")

        # --- Risk band distribution + ranked table side by side ---
        col_chart, col_table = st.columns([1, 2])
        with col_chart:
            risk_counts = (
                forecast["risk"]
                .value_counts()
                .reindex(["Critical", "High", "Medium", "Low"])
                .fillna(0)
                .reset_index()
            )
            risk_counts.columns = ["risk", "zones"]
            fig_risk = px.bar(
                risk_counts,
                x="risk",
                y="zones",
                color="risk",
                color_discrete_map={"Low": "green", "Medium": "goldenrod", "High": "orange", "Critical": "red"},
                title="Zones by Risk Band",
            )
            fig_risk.update_layout(showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_risk, use_container_width=True)
        with col_table:
            st.caption("Top 25 zones — next window")
            display_cols = [c for c in ["zone", "predicted_violations", "risk", "priority_score", "disruption_proxy"] if c in forecast.columns]
            st.dataframe(
                forecast[display_cols]
                .sort_values("predicted_violations", ascending=False)
                .head(25)
                .reset_index(drop=True),
                use_container_width=True,
            )
        st.caption("ℹ️ Disruption proxy is a heuristic (vehicle mix × road weight), NOT measured congestion.")

# =========================== Enforcement =========================
with tabs[3]:
    st.subheader("Today's patrol deployment plan")
    if len(patrol):
        # --- Patrol deployment map ---
        patrol_map = patrol.dropna(subset=["zone_lat", "zone_lon"]).copy()
        if len(patrol_map):
            TEAM_COLORS = {
                "Team A": "#1f77b4", "Team B": "#2ca02c", "Team C": "#d62728",
                "Team D": "#9467bd", "Team E": "#8c564b",
            }
            fig_patrol = px.scatter_mapbox(
                patrol_map,
                lat="zone_lat",
                lon="zone_lon",
                color="team",
                color_discrete_map=TEAM_COLORS,
                hover_name="zone",
                hover_data={
                    "priority_score": True,
                    "predicted_violations": True,
                    "risk": True,
                    "zone_lat": False,
                    "zone_lon": False,
                },
                size_max=22,
                mapbox_style="open-street-map",
                zoom=11,
                height=460,
                title="Patrol Team Assignments",
            )
            # Make markers larger and add labels.
            fig_patrol.update_traces(marker=dict(size=18, opacity=0.85))
            fig_patrol.update_layout(margin=dict(l=0, r=0, t=30, b=0), legend_title_text="Team")
            st.plotly_chart(fig_patrol, use_container_width=True)

        # --- Text cards per team ---
        for _, r in patrol.iterrows():
            risk_emoji = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}.get(
                str(r.get("risk", "")), "⚪"
            )
            st.markdown(
                f"**{r['team']} → {r['zone']}** {risk_emoji}  "
                f"&nbsp;&nbsp;priority **{r['priority_score']}** · "
                f"risk **{r.get('risk', '—')}** · "
                f"predicted **{r['predicted_violations']}** violations"
            )

    st.divider()
    st.subheader("Enforcement priority ranking — top 25 zones")
    if len(forecast):
        disp = [c for c in ["zone", "priority_score", "risk", "predicted_violations", "disruption_proxy"] if c in forecast.columns]
        st.dataframe(forecast[disp].head(25), use_container_width=True)

# ========================= Analytics Center ======================
with tabs[4]:
    st.subheader("Temporal violation trends")
    if len(events):
        ts = events["created_datetime"]
        DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        # --- Row 1: Hour + Day-of-week bars ---
        by_hour = (
            events.assign(hour=ts.dt.hour)
            .groupby("hour").size().rename("violations").reset_index()
        )
        by_dow = (
            events.assign(dow=ts.dt.dayofweek)
            .groupby("dow").size().rename("violations").reset_index()
        )
        by_dow["day"] = by_dow["dow"].map(dict(enumerate(DOW_NAMES)))

        col_h, col_d = st.columns(2)
        with col_h:
            fig_h = px.bar(
                by_hour, x="hour", y="violations",
                labels={"hour": "Hour of Day (IST)", "violations": "Violations"},
                title="Violations by Hour of Day",
                color="violations", color_continuous_scale="Reds",
            )
            fig_h.update_layout(coloraxis_showscale=False, margin=dict(t=30, b=0))
            st.plotly_chart(fig_h, use_container_width=True)
        with col_d:
            fig_d = px.bar(
                by_dow, x="day", y="violations",
                labels={"day": "Day of Week", "violations": "Violations"},
                title="Violations by Day of Week",
                color="violations", color_continuous_scale="Blues",
                category_orders={"day": DOW_NAMES},
            )
            fig_d.update_layout(coloraxis_showscale=False, margin=dict(t=30, b=0))
            st.plotly_chart(fig_d, use_container_width=True)

        # --- Row 2: Hour × Day-of-Week intensity heatmap ---
        st.subheader("Enforcement intensity grid")
        st.caption("Where each cell = total violations at that hour × day combination — shows exactly when enforcement pressure should be highest.")
        pivot_hd = (
            events.assign(
                hour=ts.dt.hour,
                day=ts.dt.dayofweek.map(dict(enumerate(DOW_NAMES))),
            )
            .groupby(["day", "hour"]).size().rename("violations")
            .unstack(level="hour", fill_value=0)
        )
        pivot_hd = pivot_hd.reindex(DOW_NAMES)
        fig_heatmap = px.imshow(
            pivot_hd,
            labels=dict(x="Hour of Day (IST)", y="Day of Week", color="Violations"),
            color_continuous_scale="YlOrRd",
            title="Violations: Day × Hour",
            aspect="auto",
        )
        fig_heatmap.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # --- Row 3: Violation type × Hour heatmap ---
        if "violation_type" in events.columns:
            st.subheader("Carriageway impact by violation type and hour")
            st.caption("Shows which violation types spike at which hours — critical for targeted enforcement scheduling.")
            CARRIAGEWAY_ORDER = [
                "DOUBLE PARKING",
                "PARKING IN A MAIN ROAD",
                "PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS",
                "PARKING NEAR ROAD CROSSING",
                "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE",
                "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC",
                "NO PARKING",
                "WRONG PARKING",
                "PARKING ON FOOTPATH",
                "PARKING OTHER THAN BUS STOP",
            ]
            pivot_vt = (
                events.assign(
                    hour=ts.dt.hour,
                    vtype=events["violation_type"].astype(str).str.upper(),
                )
                .groupby(["vtype", "hour"]).size().rename("violations")
                .unstack(level="hour", fill_value=0)
            )
            # Reorder rows to prioritise high-impact types.
            ordered = [v for v in CARRIAGEWAY_ORDER if v in pivot_vt.index]
            rest = [v for v in pivot_vt.index if v not in ordered]
            pivot_vt = pivot_vt.reindex(ordered + rest)
            fig_vt = px.imshow(
                pivot_vt,
                labels=dict(x="Hour of Day (IST)", y="Violation Type", color="Violations"),
                color_continuous_scale="Reds",
                title="Violation Type × Hour of Day",
                aspect="auto",
            )
            fig_vt.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_vt, use_container_width=True)

        # --- Row 4: Weekly trend + vehicle types ---
        by_week = events.assign(week=ts.dt.to_period("W").astype(str)).groupby("week").size()
        st.caption("Weekly violation trend")
        st.line_chart(by_week)

        st.caption("Top vehicle types")
        st.bar_chart(events["vehicle_type"].astype(str).str.upper().value_counts().head(8))

# ========================== Junction Risk ========================
with tabs[5]:
    st.subheader("Junction risk assessment")
    if len(junction_risk):
        # --- Junction risk map ---
        jmap = junction_risk.dropna(subset=["zone_lat", "zone_lon"]).copy() if \
            {"zone_lat", "zone_lon"}.issubset(junction_risk.columns) else pd.DataFrame()

        if len(jmap):
            RISK_COLORS = {"Low": "green", "Medium": "goldenrod", "High": "orange", "Critical": "red"}
            jmap["risk"] = jmap["risk"].fillna("Low") if "risk" in jmap.columns else "Low"
            fig_jmap = px.scatter_mapbox(
                jmap.sort_values("historical_violations", ascending=False),
                lat="zone_lat",
                lon="zone_lon",
                size="historical_violations",
                color="risk",
                color_discrete_map=RISK_COLORS,
                hover_name="zone",
                hover_data={
                    "historical_violations": True,
                    "priority_score": "priority_score" in jmap.columns,
                    "peak_hour": "peak_hour" in jmap.columns,
                    "dominant_vehicle": "dominant_vehicle" in jmap.columns,
                    "zone_lat": False,
                    "zone_lon": False,
                },
                size_max=35,
                mapbox_style="open-street-map",
                zoom=11,
                height=500,
                title="Junction Risk Map — bubble size = historical violations, color = risk",
                category_orders={"risk": ["Critical", "High", "Medium", "Low"]},
            )
            fig_jmap.update_layout(margin=dict(l=0, r=0, t=30, b=0), legend_title_text="Risk")
            st.plotly_chart(fig_jmap, use_container_width=True)

        # --- Table + priority bar side by side ---
        col_jtbl, col_jbar = st.columns([3, 2])
        with col_jtbl:
            st.caption("Top 30 junctions")
            st.dataframe(junction_risk.head(30), use_container_width=True)
        with col_jbar:
            topj = junction_risk.head(15).set_index("zone")
            bar_metric = "priority_score" if "priority_score" in topj else "historical_violations"
            fig_jbar = px.bar(
                topj[bar_metric].sort_values().reset_index(),
                x=bar_metric,
                y="zone",
                orientation="h",
                color=bar_metric,
                color_continuous_scale="Reds",
                title=f"Top 15 Junctions by {bar_metric.replace('_', ' ').title()}",
            )
            fig_jbar.update_layout(
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig_jbar, use_container_width=True)
    else:
        st.info("No junction-level rows available.")

# ======================== Repeat Offenders =======================
with tabs[6]:
    st.subheader("Repeat Offender Analysis")
    st.caption(
        "Vehicles recorded at the same or multiple zones repeatedly — "
        "chronic blockers create sustained congestion, not just one-off incidents."
    )
    if len(repeat_offenders):
        # --- KPIs ---
        ro_k1, ro_k2, ro_k3 = st.columns(3)
        ro_k1.metric("Repeat offender vehicles", f"{len(repeat_offenders):,}")
        ro_k2.metric("Max violations (single vehicle)", int(repeat_offenders["violation_count"].max()))
        ro_k3.metric(
            "Avg unique zones per offender",
            round(float(repeat_offenders["unique_zones"].mean()), 1),
        )

        col_tbl, col_bar = st.columns([3, 2])
        with col_tbl:
            st.caption("Top 25 repeat offenders")
            display_ro = ["vehicle_number", "violation_count", "unique_zones",
                          "top_zone", "vehicle_type", "last_seen"]
            disp_cols = [c for c in display_ro if c in repeat_offenders.columns]
            st.dataframe(repeat_offenders[disp_cols].head(25), use_container_width=True)

        with col_bar:
            st.caption("Top 15 zones by repeat-offender vehicle density")
            zone_ro = (
                repeat_offenders.groupby("top_zone")["violation_count"]
                .sum()
                .sort_values(ascending=False)
                .head(15)
                .reset_index()
            )
            zone_ro.columns = ["zone", "total_violations"]
            fig_ro_bar = px.bar(
                zone_ro, x="total_violations", y="zone",
                orientation="h",
                color="total_violations",
                color_continuous_scale="Reds",
                title="Repeat Offender Violation Load by Zone",
                labels={"total_violations": "Total Violations", "zone": "Zone"},
            )
            fig_ro_bar.update_layout(
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig_ro_bar, use_container_width=True)

        # --- Spatial heatmap of repeat offenders ---
        if len(events) and "latitude" in events.columns:
            st.caption("Spatial density of repeat-offender activity")
            ro_vehicles = set(repeat_offenders["vehicle_number"].astype(str))
            if "vehicle_number" in events.columns:
                ro_events = events[events["vehicle_number"].astype(str).isin(ro_vehicles)]
            else:
                ro_events = pd.DataFrame()

            if len(ro_events):
                ro_sample = ro_events.sample(min(len(ro_events), 15000), random_state=1)
                fig_ro_map = px.density_mapbox(
                    ro_sample, lat="latitude", lon="longitude", radius=8,
                    center=dict(
                        lat=float(ro_sample["latitude"].median()),
                        lon=float(ro_sample["longitude"].median()),
                    ),
                    zoom=10, mapbox_style="open-street-map", height=480,
                    title="Where Repeat Offenders Operate",
                )
                fig_ro_map.update_layout(margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_ro_map, use_container_width=True)

        # --- Vehicle type breakdown of offenders ---
        if "vehicle_type" in repeat_offenders.columns:
            vt_ro = repeat_offenders["vehicle_type"].astype(str).str.upper().value_counts().head(8)
            st.caption("Repeat offenders by vehicle type")
            st.bar_chart(vt_ro)
    else:
        st.info("No repeat offender data found. Run `parkflow run` to generate the artifact.")

# ============================== Model ============================
with tabs[7]:
    st.subheader("Model vs seasonal-naive baseline (held-out future)")
    if metrics:
        comp = pd.DataFrame({"baseline": metrics["baseline"], "model": metrics["model"]})
        st.dataframe(comp, use_container_width=True)
        verdict = "✅ Model beats baseline" if metrics.get("model_beats_baseline") else "⚠️ Model does NOT beat baseline"
        st.markdown(f"**{verdict}** (lower MAE / Poisson deviance is better; higher R² is better)")

        if "data_date_range" in metrics:
            dr = metrics["data_date_range"]
            st.caption(f"Training data: {dr.get('from','?')} → {dr.get('to','?')}")

    # --- Actual vs Predicted diagnostic ---
    test_preds = load("test_predictions")
    if len(test_preds):
        st.subheader("Actual vs Predicted (held-out test set)")
        col_scatter, col_hist = st.columns(2)

        with col_scatter:
            max_val = float(test_preds[["violation_count", "predicted"]].max().max())
            fig_scatter = px.scatter(
                test_preds.sample(min(len(test_preds), 5000), random_state=0),
                x="violation_count",
                y="predicted",
                opacity=0.35,
                labels={"violation_count": "Actual violations", "predicted": "Predicted violations"},
                title="Actual vs Predicted",
            )
            fig_scatter.add_shape(
                type="line",
                x0=0, y0=0, x1=max_val, y1=max_val,
                line=dict(color="red", dash="dash", width=1.5),
            )
            fig_scatter.update_layout(margin=dict(t=40, b=0))
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption("Red dashed line = perfect prediction. Points above = over-predicted; below = under-predicted.")

        with col_hist:
            fig_err = px.histogram(
                test_preds,
                x="error",
                nbins=60,
                color_discrete_sequence=["steelblue"],
                labels={"error": "Prediction error (predicted − actual)"},
                title="Error Distribution",
            )
            fig_err.add_vline(x=0, line_dash="dash", line_color="red", line_width=1.5)
            fig_err.update_layout(margin=dict(t=40, b=0))
            st.plotly_chart(fig_err, use_container_width=True)
            st.caption("Symmetric distribution centred on 0 = well-calibrated. Right skew = systematic over-prediction.")

    # --- Feature importances ---
    fi = load("feature_importance")
    if len(fi):
        st.subheader("Top feature importances")
        fi_sorted = fi.sort_values("importance", ascending=True).tail(17)
        fig_fi = px.bar(
            fi_sorted,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Blues",
            title="XGBoost Feature Importances",
        )
        fig_fi.update_layout(
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig_fi, use_container_width=True)
