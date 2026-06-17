# modelling.md - Data Preprocessing Strategy

# Objective

The objective of preprocessing is to convert raw parking violation records into structured spatial-temporal data suitable for hotspot prediction and enforcement intelligence.

The raw dataset contains individual violation events. However, the prediction problem operates at the area level rather than the individual vehicle level. Therefore, preprocessing focuses on aggregating violations by location and time.

---

# Step 1: Remove Irrelevant Columns

Several columns are administrative fields that do not contribute to hotspot prediction.

## Columns to Remove

```text
id
vehicle_number
device_id
created_by_id
center_code
closed_datetime
modified_datetime
action_taken_timestamp
data_sent_to_scita_timestamp
updated_vehicle_number
updated_vehicle_type
validation_timestamp
```

## Reason

These fields:

* Do not influence future violation patterns.
* Act only as identifiers or workflow metadata.
* Introduce unnecessary dimensionality.
* Increase model complexity without predictive value.

---

# Step 2: Handle Missing Values

Several fields contain missing values.

## Important Columns

```text
junction_name
validation_status
police_station
```

## Strategy

### Junction Name

Replace missing values with:

```text
No Junction
```

### Police Station

Replace using:

```text
Unknown Station
```

or remove if extremely sparse.

### Validation Status

Treat missing records as:

```text
Pending
```

or exclude depending on business rules.

## Reason

Machine learning models cannot directly process missing categorical values.

---

# Step 3: Filter Valid Violations

Keep only genuine parking violations.

## Relevant Violation Types

```text
NO PARKING

WRONG PARKING

PARKING IN A MAIN ROAD

PARKING NEAR ROAD CROSSING
```

## Reason

The project focuses specifically on parking-related violations.

Other traffic offences should not influence parking hotspot analysis.

---

# Step 4: Remove Duplicate Records

Duplicate observations may occur due to:

* Multiple captures of the same vehicle.
* Multiple reporting attempts.
* Validation workflow duplication.

## Detection

Compare:

```text
vehicle_number
latitude
longitude
created_datetime
```

## Reason

Duplicate events artificially inflate hotspot severity.

---

# Step 5: Extract Temporal Features

The timestamp is one of the most important predictors.

## Source

```text
created_datetime
```

## Extract

```text
Hour

Day

Weekday

Month

Week Number

Quarter

Weekend Flag
```

Example:

```text
2024-03-14 21:20

Hour = 21

Weekday = Thursday

Month = March

Weekend = False
```

## Reason

Parking violations often follow recurring temporal patterns.

Examples:

* Office hours
* Weekend shopping traffic
* Evening congestion periods

The model cannot learn these patterns from raw timestamps.

---

# Step 6: Spatial Normalization

Latitude and longitude values are continuous.

Raw coordinates are unsuitable for hotspot prediction.

## Convert Coordinates into Zones

Possible approaches:

### Approach A: Grid-Based

Divide Bengaluru into fixed geographical cells.

Example:

```text
Grid_001

Grid_002

Grid_003
```

### Approach B: Junction-Based

Use:

```text
junction_name
```

as the spatial unit.

### Recommended

Use junctions whenever available.

Use coordinate grids for locations without junction information.

## Reason

Hotspots occur within regions, not at exact GPS coordinates.

Spatial grouping allows the model to learn area-level behavior.

---

# Step 7: Create Violation Counts

Aggregate individual records.

Example:

Before:

```text
Vehicle A

Vehicle B

Vehicle C
```

After:

```text
Dairy Circle

8 AM

12 Violations
```

## Aggregation Keys

```text
Location Zone

Hour
```

## Output

```text
Zone | Hour | Violation Count
```

## Reason

The prediction target is future violation volume.

The model should learn area-level patterns rather than individual vehicle behavior.

---

# Step 8: Historical Violation Features

Create lag features.

## Examples

```text
Violations Last Hour

Violations Last Day

Violations Last Week

Rolling 7-Day Average

Rolling 30-Day Average
```

## Reason

Future violations are strongly influenced by historical activity.

Locations with frequent violations tend to remain violation-prone.

These become the most important predictive features.

---

# Step 9: Create Location Statistics

For every location compute:

```text
Total Historical Violations

Average Daily Violations

Peak Hour Violations

Violation Growth Rate
```

## Reason

These metrics capture hotspot intensity and persistence.

---

# Step 10: Encode Categorical Variables

Categorical fields:

```text
police_station

junction_name

vehicle_type

violation_type
```

Convert using:

### Label Encoding

For tree-based models.

or

### One-Hot Encoding

For linear models.

## Reason

Machine learning algorithms require numerical inputs.

---

# Final Modelling Dataset

Each row should represent:

```text
Location Zone
+
Time Window
```

Example:

Location: Dairy Circle

Hour: 8 AM

Features:

* Day of Week
* Month
* Historical Violations
* Rolling Average
* Junction Type
* Police Station

Target:

Expected Violation Count

```

---

# Expected Benefits of Preprocessing

1. Removes noisy administrative information.
2. Converts raw events into hotspot intelligence.
3. Captures temporal patterns.
4. Captures spatial patterns.
5. Enables future hotspot prediction.
6. Improves model interpretability.
7. Produces features directly aligned with enforcement decision-making.
```
