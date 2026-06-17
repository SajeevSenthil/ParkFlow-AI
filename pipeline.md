# pipeline.md

# Overview

The objective of ParkFlow AI is to transform historical parking violation records into actionable enforcement intelligence.

The system predicts future parking violations at junctions and locations across the city, identifies emerging hotspots, prioritizes enforcement zones, and recommends patrol deployment strategies.

The solution follows a spatial-temporal forecasting approach where historical parking violations are aggregated by junction and time window to predict future violation intensity.

---

# End-to-End Workflow

```text
Raw Parking Violation Dataset
            ↓
Data Preprocessing
            ↓
Spatial Aggregation (Junction-Level)
            ↓
Temporal Aggregation (Hourly)
            ↓
Historical Feature Engineering
            ↓
Junction-Hour Violation Forecasting Model
            ↓
Risk Classification
            ↓
Hotspot Generation
            ↓
Priority Scoring
            ↓
Patrol Recommendation Engine
            ↓
Streamlit Dashboard
```

---

# Stage 1: Data Preprocessing

The raw parking violation records are cleaned, validated, and transformed into structured data suitable for analysis.

## Inputs

* Latitude
* Longitude
* Junction Name
* Police Station
* Vehicle Type
* Violation Type
* Created Datetime

## Outputs

Cleaned parking violation records.

---

# Stage 2: Spatial Aggregation

Individual violation events are grouped into meaningful geographical regions.

## Primary Spatial Unit

```text
junction_name
```

Junctions represent real-world enforcement zones and are directly understandable by traffic authorities.

## Fallback Spatial Unit

```text
Geographical Grid Cell
```

Used when junction information is unavailable.

## Example

Raw Records:

```text
Vehicle A → Dairy Circle
Vehicle B → Dairy Circle
Vehicle C → Dairy Circle
```

Aggregated Record:

```text
Dairy Circle

Hour: 08:00

Violations: 12
```

---

# Stage 3: Temporal Aggregation

Violation events are grouped into fixed hourly windows.

## Aggregation Window

```text
1 Hour
```

## Example

```text
Dairy Circle

08:00 - 09:00

12 Violations
```

```text
Dairy Circle

09:00 - 10:00

18 Violations
```

## Reason

Hourly aggregation captures operational enforcement patterns while remaining granular enough for patrol planning.

---

# Stage 4: Historical Feature Engineering

Historical trends are converted into predictive features.

## Temporal Features

* Hour
* Day of Week
* Month
* Week Number
* Weekend Flag

---

## Historical Activity Features

* Violations Previous Hour
* Violations Previous Day
* Violations Previous Week
* Rolling 7-Day Average
* Rolling 30-Day Average

---

## Location Features

* Junction
* Police Station
* Historical Junction Violation Frequency

---

## Vehicle Features

* Vehicle Type Distribution
* Violation Type Distribution

---

# Stage 5: Prediction Target Definition

## Prediction Target

```text
Expected Violation Count
for a Junction
during a Future Hour
```

---

## Example

Input:

```text
Location: Dairy Circle

Date: Tomorrow

Hour: 08:00
```

Output:

```text
Predicted Violations = 23
```

This transforms the problem into a supervised regression task.

---

# Stage 6: Junction-Hour Violation Forecasting Model

## Objective

Predict the number of parking violations expected at a specific junction during a future hourly window.

---

## Model Type

Supervised Regression

---

## Recommended Model

### Primary Model

XGBoost Regressor

---

## Reasons

* Excellent performance on tabular datasets.
* Handles nonlinear relationships.
* Supports mixed feature types.
* Requires minimal preprocessing.
* Provides feature importance analysis.
* Performs well with temporal and spatial features.

---

## Model Inputs

* Junction
* Police Station
* Hour
* Day of Week
* Month
* Weekend Flag
* Historical Violation Counts
* Rolling Averages
* Vehicle Distribution Statistics

---

## Model Output

```text
Predicted Violation Count
```

for a Junction-Hour pair.

---

# Stage 7: Risk Classification Layer

Predicted counts are transformed into operational risk categories.

## Example

```text
0 - 5       Low Risk

6 - 15      Medium Risk

16 - 25     High Risk

>25         Critical Risk
```

## Output

Location-level risk assessment.

---

# Stage 8: Hotspot Generation

Hotspots are generated from predicted violation intensity.

## Current Hotspots

Derived from historical violation density.

---

## Future Hotspots

Derived from predicted violation counts.

---

## Example

```text
Dairy Circle

Predicted Violations: 23

Risk: High
```

---

# Stage 9: Enforcement Priority Scoring

Every junction receives a priority score.

## Inputs

* Predicted Violations
* Historical Violations
* Risk Category
* Junction Importance

---

## Example

```text
Priority Score

=
0.6 × Predicted Violations

+
0.3 × Historical Frequency

+
0.1 × Junction Weight
```

---

## Output

Ranked enforcement zones.

---

# Stage 10: Patrol Recommendation Engine

The highest-priority locations are converted into actionable deployment recommendations.

## Example

```text
Patrol Team A → Dairy Circle

Patrol Team B → KR Market

Patrol Team C → Subbanna Junction
```

---

# Model Evaluation

## Evaluation Strategy

The model will be evaluated on unseen future time periods to simulate real-world forecasting.

Training data should contain earlier periods while testing data should contain later periods.

---

## Metrics

### MAE

Measures average prediction error in violation counts.

---

### RMSE

Penalizes large prediction errors.

---

### R² Score

Measures how much variance in violation counts is explained by the model.

---

# Dashboard Outputs

## Hotspot Intelligence

* Current Violation Heatmap
* Hotspot Rankings
* Junction Analytics

---

## Prediction Center

* Future Hotspot Heatmap
* Predicted Violation Counts
* Risk Categories

---

## Enforcement Center

* Priority Rankings
* Patrol Recommendations
* High-Risk Junction Monitoring

---

## Analytics Center

* Hourly Trends
* Daily Trends
* Weekly Trends
* Junction Risk Analysis

---

# Final Outcome

The final system predicts parking violation intensity at the junction-hour level and converts those predictions into hotspot intelligence, enforcement priorities, and patrol recommendations.

This enables traffic authorities to shift from reactive enforcement to proactive, data-driven parking management.
