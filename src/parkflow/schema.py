"""Canonical column names for the raw parking-violation dataset.

Centralising these as constants means a schema change touches one file, and the
rest of the codebase never hard-codes a string literal for a column.
"""

from __future__ import annotations

# --- Raw input columns (as delivered in the BTP export) ---
LATITUDE = "latitude"
LONGITUDE = "longitude"
LOCATION = "location"
VEHICLE_NUMBER = "vehicle_number"
VEHICLE_TYPE = "vehicle_type"
VIOLATION_TYPE = "violation_type"
OFFENCE_CODE = "offence_code"
CREATED_DATETIME = "created_datetime"
CLOSED_DATETIME = "closed_datetime"
DEVICE_ID = "device_id"
CENTER_CODE = "center_code"
POLICE_STATION = "police_station"
JUNCTION_NAME = "junction_name"
VALIDATION_STATUS = "validation_status"
VALIDATION_TIMESTAMP = "validation_timestamp"

# Minimum columns the pipeline needs to do anything useful.
REQUIRED_COLUMNS = [
    LATITUDE,
    LONGITUDE,
    VIOLATION_TYPE,
    CREATED_DATETIME,
]

# Admin / workflow columns dropped from the MODEL path (kept for analytics elsewhere).
ADMIN_COLUMNS = [
    "id",
    "description",
    DEVICE_ID,
    "created_by_id",
    CENTER_CODE,
    CLOSED_DATETIME,
    "modified_datetime",
    "data_sent_to_scita",
    "action_taken_timestamp",
    "data_sent_to_scita_timestamp",
    "updated_vehicle_number",
    "updated_vehicle_type",
]

# --- Engineered columns produced by the pipeline ---
ZONE = "zone"                 # unified spatial unit (junction or grid cell)
ZONE_KIND = "zone_kind"       # "junction" | "grid"
ZONE_LAT = "zone_lat"
ZONE_LON = "zone_lon"
BIN_START = "bin_start"       # floored timestamp of the aggregation window
TARGET = "violation_count"    # regression target
