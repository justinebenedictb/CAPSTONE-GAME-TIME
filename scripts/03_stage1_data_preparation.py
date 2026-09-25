"""
03_stage1_data_preparation.py

CAPSTONE
Stage 1 - Data Preparation

Pipeline:
1. Filter manifest to valid Run activities with the required data signature
2. Link manifest activities to granular second-by-second CSV files
3. Parse standardized-distance performance times:
      5.0 km
      10.0 km
      15.0 km
      21.1 km
      42.2 km
4. Build one-row-per-calendar-day athlete timelines
5. Zero-pad rest days
6. Build target-anchored 28-calendar-day sliding windows
7. Write audit/output files

Confirmed granular schema:
    secs
    km
    power
    hr
    cad
    alt

Project root:
    "C:\\Users\\admin\\Desktop\\CAPSTONE GAME TIME"
"""

from pathlib import Path
import math
import traceback

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(r"C:\Users\admin\Desktop\CAPSTONE GAME TIME")

MANIFEST_PATH = ROOT / "manifest file" / "run_activities_manifest.csv"
GRANULAR_ROOT = ROOT / "00_raw" / "granular"

OUTPUT_DIR = ROOT / "01_prepared"
LOG_DIR = ROOT / "logs"

# Confirmed analytical framework
REQUIRED_SPORT = "Run"
REQUIRED_SIGNATURE = "TDS-HC-AGL----R"

# Standard distances used as performance targets
STANDARD_DISTANCES = {
    "5.0": 5.0,
    "10.0": 10.0,
    "15.0": 15.0,
    "21.1": 21.1,
    "42.2": 42.2,
}

WINDOW_DAYS = 28

# Confirmed granular schema
GRANULAR_REQUIRED_COLUMNS = [
    "secs",
    "km",
    "power",
    "hr",
    "cad",
    "alt",
]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_filename(value):
    """Convert a value to a safe filename component."""
    return str(value).replace("/", "_").replace("\\", "_").replace(":", "_")


def format_seconds(seconds):
    """Convert seconds to HH:MM:SS or MM:SS."""
    if pd.isna(seconds):
        return np.nan

    seconds = float(seconds)

    if seconds < 0:
        return np.nan

    total_seconds = int(round(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    return f"{minutes:02d}:{secs:02d}"


def clean_numeric(series):
    """Convert a pandas Series to numeric, coercing invalid values to NaN."""
    return pd.to_numeric(series, errors="coerce")


# ============================================================
# 1. LOAD AND FILTER MANIFEST
# ============================================================

def load_and_filter_manifest():
    print("=" * 70)
    print("STEP 1 - LOADING AND FILTERING MANIFEST")
    print("=" * 70)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST_PATH}"
        )

    manifest = pd.read_csv(MANIFEST_PATH)

    print(f"Manifest rows loaded: {len(manifest):,}")
    print(f"Manifest columns: {len(manifest.columns)}")

    required_manifest_columns = [
        "id",
        "file",
        "date",
        "sport",
        "data",
    ]

    missing_columns = [
        col for col in required_manifest_columns
        if col not in manifest.columns
    ]

    if missing_columns:
        raise ValueError(
            "Manifest is missing required columns: "
            + ", ".join(missing_columns)
        )

    # --------------------------------------------------------
    # Filter sport
    # --------------------------------------------------------

    manifest["sport"] = manifest["sport"].astype(str).str.strip()

    run_mask = manifest["sport"].eq(REQUIRED_SPORT)

    # --------------------------------------------------------
    # Filter data signature
    # --------------------------------------------------------

    manifest["data"] = manifest["data"].astype(str).str.strip()

    signature_mask = manifest["data"].eq(REQUIRED_SIGNATURE)

    filtered = manifest.loc[
        run_mask & signature_mask
    ].copy()

    print(f"Run activities: {run_mask.sum():,}")
    print(f"Required signature: {signature_mask.sum():,}")
    print(f"Final Stage 1 activities: {len(filtered):,}")

    if filtered.empty:
        raise ValueError(
            "No activities remained after applying the Run + "
            f"{REQUIRED_SIGNATURE} filters."
        )

    # --------------------------------------------------------
    # Parse datetime
    # --------------------------------------------------------

    filtered["datetime_utc"] = pd.to_datetime(
        filtered["date"],
        errors="coerce",
        utc=True
    )

    invalid_dates = filtered["datetime_utc"].isna().sum()

    print(f"Invalid activity dates: {invalid_dates:,}")

    if invalid_dates > 0:
        filtered = filtered.loc[
            filtered["datetime_utc"].notna()
        ].copy()

    filtered["calendar_date"] = (
        filtered["datetime_utc"]
        .dt.floor("D")
        .dt.date
    )

    # --------------------------------------------------------
    # Clean identifiers
    # --------------------------------------------------------

    filtered["id"] = filtered["id"].astype(str).str.strip()
    filtered["file"] = filtered["file"].astype(str).str.strip()

    # Remove exact duplicate activity mappings while retaining
    # the first manifest record.
    before_duplicates = len(filtered)

    filtered = filtered.drop_duplicates(
        subset=["id", "file"],
        keep="first"
    ).copy()

    duplicate_rows_removed = (
        before_duplicates - len(filtered)
    )

    print(
        f"Duplicate (id, file) rows removed: "
        f"{duplicate_rows_removed:,}"
    )

    # Chronological ordering
    filtered = filtered.sort_values(
        ["id", "datetime_utc", "file"]
    ).reset_index(drop=True)

    # Sequence number within athlete
    filtered["activity_sequence"] = (
        filtered.groupby("id").cumcount() + 1
    )

    print(
        f"Unique athletes: "
        f"{filtered['id'].nunique():,}"
    )

    print(
        f"Date range: "
        f"{filtered['calendar_date'].min()} "
        f"to "
        f"{filtered['calendar_date'].max()}"
    )

    return filtered


# ============================================================
# 2. PARSE STANDARD-DISTANCE TARGETS
# ============================================================

def interpolate_target_time(granular, target_km):
    """
    Find elapsed time at a target distance.

    Uses:
        secs = elapsed seconds
        km   = cumulative distance in kilometers

    If the target distance falls between two observations,
    linearly interpolate elapsed time.

    Returns:
        float seconds, or NaN if target distance was not reached.
    """

    work = granular[["secs", "km"]].copy()

    work["secs"] = clean_numeric(work["secs"])
    work["km"] = clean_numeric(work["km"])

    work = work.dropna(subset=["secs", "km"])

    if work.empty:
        return np.nan

    work = work.sort_values("km").reset_index(drop=True)

    # Remove impossible negative values
    work = work[
        (work["secs"] >= 0) &
        (work["km"] >= 0)
    ].copy()

    if work.empty:
        return np.nan

    # Collapse duplicate distance values by keeping the
    # earliest elapsed time.
    work = (
        work.groupby("km", as_index=False)["secs"]
        .min()
        .sort_values("km")
        .reset_index(drop=True)
    )

    max_distance = work["km"].max()

    if max_distance < target_km:
        return np.nan

    # Exact observation exists
    exact = work.loc[
        np.isclose(work["km"], target_km, atol=1e-9)
    ]

    if not exact.empty:
        return float(exact.iloc[0]["secs"])

    # First observation beyond target
    upper_idx = work["km"].searchsorted(
        target_km,
        side="left"
    )

    if upper_idx <= 0:
        return np.nan

    if upper_idx >= len(work):
        return np.nan

    lower = work.iloc[upper_idx - 1]
    upper = work.iloc[upper_idx]

    x1 = float(lower["km"])
    y1 = float(lower["secs"])

    x2 = float(upper["km"])
    y2 = float(upper["secs"])

    if x2 <= x1:
        return np.nan

    # Linear interpolation
    fraction = (
        (target_km - x1) /
        (x2 - x1)
    )

    target_seconds = (
        y1 +
        fraction * (y2 - y1)
    )

    return float(target_seconds)


def parse_standard_distance_targets(activity_row):
    """
    Read one granular activity and extract standard-distance
    performance targets.
    """

    athlete_id = str(activity_row["id"])
    filename = str(activity_row["file"])

    granular_path = (
        GRANULAR_ROOT /
        athlete_id /
        filename
    )

    result = {
        "id": athlete_id,
        "file": filename,
        "granular_path": str(granular_path),
        "granular_status": "OK",
        "granular_rows": np.nan,
        "granular_max_km": np.nan,
    }

    # Initialize target fields
    for distance_label in STANDARD_DISTANCES:
        result[
            f"target_{distance_label.replace('.', '_')}km_sec"
        ] = np.nan

        result[
            f"target_{distance_label.replace('.', '_')}km_time"
        ] = np.nan

    if not granular_path.exists():
        result["granular_status"] = "MISSING_FILE"
        return result

    try:
        granular = pd.read_csv(granular_path)

    except Exception as exc:
        result["granular_status"] = (
            f"READ_ERROR: {type(exc).__name__}"
        )
        return result

    result["granular_rows"] = len(granular)

    # --------------------------------------------------------
    # Strict schema validation
    # --------------------------------------------------------

    missing_columns = [
        col
        for col in GRANULAR_REQUIRED_COLUMNS
        if col not in granular.columns
    ]

    if missing_columns:
        result["granular_status"] = (
            "MISSING_COLUMNS: "
            + ",".join(missing_columns)
        )
        return result

    # --------------------------------------------------------
    # Clean numeric fields
    # --------------------------------------------------------

    for col in GRANULAR_REQUIRED_COLUMNS:
        granular[col] = clean_numeric(granular[col])

    granular = granular.sort_values(
        ["secs", "km"]
    ).reset_index(drop=True)

    valid_distance = granular["km"].dropna()

    if valid_distance.empty:
        result["granular_status"] = "NO_VALID_DISTANCE"
        return result

    result["granular_max_km"] = valid_distance.max()

    # --------------------------------------------------------
    # Extract standard-distance targets
    # --------------------------------------------------------

    for distance_label, target_km in STANDARD_DISTANCES.items():

        target_seconds = interpolate_target_time(
            granular,
            target_km
        )

        key = (
            f"target_{distance_label.replace('.', '_')}km"
        )

        result[f"{key}_sec"] = target_seconds
        result[f"{key}_time"] = (
            format_seconds(target_seconds)
            if not pd.isna(target_seconds)
            else np.nan
        )

    return result


def parse_all_targets(manifest):
    print()
    print("=" * 70)
    print("STEP 2 - PARSING STANDARD-DISTANCE TARGETS")
    print("=" * 70)

    records = []

    total = len(manifest)

    for i, (_, row) in enumerate(manifest.iterrows(), start=1):

        if i == 1 or i % 500 == 0 or i == total:
            print(
                f"Processing granular files: "
                f"{i:,}/{total:,}"
            )

        records.append(
            parse_standard_distance_targets(row)
        )

    target_df = pd.DataFrame(records)

    status_counts = (
        target_df["granular_status"]
        .value_counts(dropna=False)
    )

    print()
    print("Granular file status:")
    print(status_counts.to_string())

    # Target availability
    print()
    print("Standard-distance target availability:")

    for distance_label in STANDARD_DISTANCES:

        col = (
            f"target_{distance_label.replace('.', '_')}km_sec"
        )

        available = target_df[col].notna().sum()

        print(
            f"  {distance_label:>4} km: "
            f"{available:,} activities"
        )

    return target_df


# ============================================================
# 3. COMBINE MANIFEST + TARGET DATA
# ============================================================

def build_activity_index(manifest, target_df):
    print()
    print("=" * 70)
    print("STEP 3 - BUILDING ACTIVITY INDEX")
    print("=" * 70)

    # Keep manifest metadata
    activity = manifest.merge(
        target_df,
        on=["id", "file"],
        how="left",
        validate="one_to_one"
    )

    # Ensure chronological ordering
    activity = activity.sort_values(
        ["id", "datetime_utc", "file"]
    ).reset_index(drop=True)

    # Recalculate sequence after merge
    activity["activity_sequence"] = (
        activity.groupby("id").cumcount() + 1
    )

    print(
        f"Activity index rows: {len(activity):,}"
    )

    print(
        f"Unique athletes: "
        f"{activity['id'].nunique():,}"
    )

    return activity


# ============================================================
# 4. BUILD DAILY TIMELINE
# ============================================================

def build_daily_timeline(activity):
    print()
    print("=" * 70)
    print("STEP 4 - BUILDING DAILY TIMELINE")
    print("=" * 70)

    daily_records = []

    # Numeric manifest columns that may represent
    # activity-level training measurements.
    sum_candidates = [
        "workout_time",
        "total_distance",
        "elevation_gain",
        "total_work",
        "coggan_tss",
    ]

    mean_candidates = [
        "average_speed",
        "average_hr",
        "average_cadence",
        "average_power",
        "calories",
    ]

    available_sum = [
        c for c in sum_candidates
        if c in activity.columns
    ]

    available_mean = [
        c for c in mean_candidates
        if c in activity.columns
    ]

    # Convert available numerical columns
    for col in available_sum + available_mean:
        activity[col] = clean_numeric(activity[col])

    # --------------------------------------------------------
    # Process athlete by athlete
    # --------------------------------------------------------

    for athlete_id, athlete_df in activity.groupby(
        "id",
        sort=False
    ):

        athlete_df = athlete_df.sort_values(
            "calendar_date"
        ).copy()

        start_date = pd.Timestamp(
            athlete_df["calendar_date"].min()
        )

        end_date = pd.Timestamp(
            athlete_df["calendar_date"].max()
        )

        all_dates = pd.date_range(
            start=start_date,
            end=end_date,
            freq="D"
        )

        activity_dates = set(
            athlete_df["calendar_date"]
        )

        # Track last actual activity date
        last_activity_date = None

        for current_timestamp in all_dates:

            current_date = current_timestamp.date()

            day_activities = athlete_df[
                athlete_df["calendar_date"] == current_date
            ]

            is_activity_day = len(day_activities) > 0

            row = {
                "id": athlete_id,
                "calendar_date": current_date,
                "is_activity_day": int(is_activity_day),
                "activity_count": len(day_activities),
            }

            # ------------------------------------------------
            # Training metadata
            # ------------------------------------------------

            for col in available_sum:
                if is_activity_day:
                    row[col] = day_activities[col].sum(
                        skipna=True
                    )
                else:
                    row[col] = 0.0

            for col in available_mean:
                if is_activity_day:
                    row[col] = day_activities[col].mean(
                        skipna=True
                    )
                else:
                    row[col] = 0.0

            # ------------------------------------------------
            # Aggregate target times
            #
            # If multiple activities on the same day reach
            # the same standard distance, retain the fastest
            # valid time.
            # ------------------------------------------------

            for distance_label in STANDARD_DISTANCES:

                target_col = (
                    f"target_"
                    f"{distance_label.replace('.', '_')}"
                    f"km_sec"
                )

                time_col = (
                    f"target_"
                    f"{distance_label.replace('.', '_')}"
                    f"km_time"
                )

                if is_activity_day:
                    valid_targets = (
                        day_activities[target_col]
                        .dropna()
                    )

                    if not valid_targets.empty:
                        fastest = valid_targets.min()

                        row[target_col] = fastest
                        row[time_col] = (
                            format_seconds(fastest)
                        )
                    else:
                        row[target_col] = np.nan
                        row[time_col] = np.nan

                else:
                    row[target_col] = np.nan
                    row[time_col] = np.nan

            # ------------------------------------------------
            # Correct days-since-previous-activity
            # ------------------------------------------------

            if is_activity_day:
                if last_activity_date is None:
                    row["days_since_previous_activity"] = np.nan
                else:
                    row["days_since_previous_activity"] = (
                        current_date - last_activity_date
                    ).days

                last_activity_date = current_date

            else:
                if last_activity_date is None:
                    row["days_since_previous_activity"] = np.nan
                else:
                    row["days_since_previous_activity"] = (
                        current_date - last_activity_date
                    ).days

            daily_records.append(row)

    daily = pd.DataFrame(daily_records)

    # --------------------------------------------------------
    # Final timeline ordering
    # --------------------------------------------------------

    daily = daily.sort_values(
        ["id", "calendar_date"]
    ).reset_index(drop=True)

    # Sequential calendar index within athlete
    daily["day_index"] = (
        daily.groupby("id").cumcount()
    )

    # Ensure activity indicator is integer
    daily["is_activity_day"] = (
        daily["is_activity_day"]
        .astype(int)
    )

    print(
        f"Daily timeline rows: {len(daily):,}"
    )

    print(
        f"Unique athletes: "
        f"{daily['id'].nunique():,}"
    )

    print(
        f"Activity days: "
        f"{daily['is_activity_day'].sum():,}"
    )

    print(
        f"Rest days: "
        f"{(daily['is_activity_day'] == 0).sum():,}"
    )

    return daily


# ============================================================
# 5. BUILD 28-DAY TARGET-ANCHORED WINDOWS
# ============================================================

def build_28day_windows(daily):
    """
    Build 28-calendar-day windows.

    A window is retained only when its END DATE contains a
    standard-distance target observation.

    Therefore:

        window_start = target_date - 27 days
        window_end   = target_date

    This ensures each future modeling sample has a defined
    28-day history ending on the target observation date.

    The actual target value is recorded in the index, while
    the full daily timeline remains in stage1_daily_timeline.csv.
    """

    print()
    print("=" * 70)
    print("STEP 5 - BUILDING 28-DAY TARGET-ANCHORED WINDOWS")
    print("=" * 70)

    windows = []

    target_columns = {
        distance_label: (
            f"target_"
            f"{distance_label.replace('.', '_')}"
            f"km_sec"
        )
        for distance_label in STANDARD_DISTANCES
    }

    for athlete_id, athlete_daily in daily.groupby(
        "id",
        sort=False
    ):

        athlete_daily = athlete_daily.sort_values(
            "calendar_date"
        ).reset_index(drop=True)

        # Fast lookup by date
        date_to_row = {
            row["calendar_date"]: row
            for _, row in athlete_daily.iterrows()
        }

        # ----------------------------------------------------
        # Each target observation becomes a potential window
        # ----------------------------------------------------

        for _, target_row in athlete_daily.iterrows():

            target_date = target_row["calendar_date"]

            target_distances = []

            for distance_label, target_col in target_columns.items():

                target_value = target_row[target_col]

                if pd.notna(target_value):
                    target_distances.append(
                        distance_label
                    )

            if not target_distances:
                continue

            target_timestamp = pd.Timestamp(
                target_date
            )

            window_start = (
                target_timestamp -
                pd.Timedelta(days=WINDOW_DAYS - 1)
            ).date()

            window_end = target_date

            # Need the full 28 calendar days available
            expected_days = pd.date_range(
                start=pd.Timestamp(window_start),
                end=pd.Timestamp(window_end),
                freq="D"
            )

            # ------------------------------------------------
            # Do not construct incomplete windows.
            # ------------------------------------------------

            if len(expected_days) != WINDOW_DAYS:
                continue

            first_available_date = (
                athlete_daily["calendar_date"].min()
            )

            if first_available_date > window_start:
                continue

            window_mask = (
                (athlete_daily["calendar_date"] >= window_start) &
                (athlete_daily["calendar_date"] <= window_end)
            )

            window_daily = athlete_daily.loc[
                window_mask
            ]

            # Safety check: must have exactly 28 calendar rows
            if len(window_daily) != WINDOW_DAYS:
                continue

            actual_start = window_daily[
                "calendar_date"
            ].min()

            actual_end = window_daily[
                "calendar_date"
            ].max()

            actual_span = (
                pd.Timestamp(actual_end) -
                pd.Timestamp(actual_start)
            ).days

            if actual_span != WINDOW_DAYS - 1:
                raise ValueError(
                    f"28-day window validation failed for "
                    f"athlete {athlete_id}: "
                    f"{actual_start} to {actual_end}"
                )

            activity_days = int(
                window_daily["is_activity_day"].sum()
            )

            rest_days = (
                WINDOW_DAYS - activity_days
            )

            # ------------------------------------------------
            # One window record per target distance
            # ------------------------------------------------

            for distance_label in target_distances:

                target_col = target_columns[
                    distance_label
                ]

                target_seconds = target_row[
                    target_col
                ]

                windows.append({
                    "id": athlete_id,
                    "window_start": window_start,
                    "window_end": window_end,
                    "target_date": target_date,
                    "target_distance_km": float(
                        STANDARD_DISTANCES[
                            distance_label
                        ]
                    ),
                    "target_distance_label": (
                        f"{distance_label} km"
                    ),
                    "target_seconds": float(
                        target_seconds
                    ),
                    "target_time": format_seconds(
                        target_seconds
                    ),
                    "window_days": WINDOW_DAYS,
                    "activity_days": activity_days,
                    "rest_days": rest_days,
                })

    windows_df = pd.DataFrame(windows)

    if windows_df.empty:
        print(
            "WARNING: No complete target-anchored "
            "28-day windows were created."
        )
        return windows_df

    windows_df = windows_df.sort_values(
        [
            "id",
            "target_date",
            "target_distance_km",
        ]
    ).reset_index(drop=True)

    # Unique window ID
    windows_df.insert(
        0,
        "window_id",
        [
            f"{athlete}_{date}_{distance}"
            for athlete, date, distance in zip(
                windows_df["id"],
                windows_df["target_date"],
                windows_df["target_distance_km"]
            )
        ]
    )

    print(
        f"28-day target windows: "
        f"{len(windows_df):,}"
    )

    print(
        f"Unique athletes with windows: "
        f"{windows_df['id'].nunique():,}"
    )

    print()
    print("Windows by target distance:")

    print(
        windows_df["target_distance_label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    return windows_df


# ============================================================
# 6. IDENTIFY MISSING GRANULAR ACTIVITIES
# ============================================================

def build_missing_activity_audit(activity):
    print()
    print("=" * 70)
    print("STEP 6 - BUILDING GRANULAR FILE AUDIT")
    print("=" * 70)

    missing = activity[
        activity["granular_status"] != "OK"
    ].copy()

    print(
        f"Activities with granular issues: "
        f"{len(missing):,}"
    )

    if not missing.empty:
        print()
        print("Issue breakdown:")

        print(
            missing["granular_status"]
            .value_counts(dropna=False)
            .to_string()
        )

    return missing


# ============================================================
# 7. SUMMARY
# ============================================================

def write_summary(
    activity,
    daily,
    windows,
    missing
):
    print()
    print("=" * 70)
    print("STEP 7 - WRITING SUMMARY")
    print("=" * 70)

    summary_path = (
        OUTPUT_DIR /
        "stage1_summary.txt"
    )

    total_activities = len(activity)
    total_athletes = activity["id"].nunique()

    valid_granular = (
        activity["granular_status"] == "OK"
    ).sum()

    summary_lines = [
        "CAPSTONE - STAGE 1 DATA PREPARATION SUMMARY",
        "=" * 60,
        "",
        f"Manifest source: {MANIFEST_PATH}",
        f"Granular root: {GRANULAR_ROOT}",
        "",
        "FILTERING",
        "-" * 60,
        f"Required sport: {REQUIRED_SPORT}",
        f"Required signature: {REQUIRED_SIGNATURE}",
        f"Filtered activities: {total_activities:,}",
        f"Unique athletes: {total_athletes:,}",
        "",
        "GRANULAR FILES",
        "-" * 60,
        f"Valid granular files: {valid_granular:,}",
        f"Granular issues: {len(missing):,}",
        "",
        "DAILY TIMELINE",
        "-" * 60,
        f"Daily rows: {len(daily):,}",
        f"Activity days: {int(daily['is_activity_day'].sum()):,}",
        f"Rest days: {int((daily['is_activity_day'] == 0).sum()):,}",
        "",
        "STANDARD-DISTANCE TARGETS",
        "-" * 60,
    ]

    for distance_label in STANDARD_DISTANCES:

        col = (
            f"target_"
            f"{distance_label.replace('.', '_')}"
            f"km_sec"
        )

        available = activity[col].notna().sum()

        summary_lines.append(
            f"{distance_label} km targets: "
            f"{available:,}"
        )

    summary_lines.extend([
        "",
        "28-DAY WINDOWS",
        "-" * 60,
        f"Complete target-anchored windows: {len(windows):,}",
        "",
        "Window definition:",
        "28 consecutive calendar days ending on a",
        "standard-distance target observation.",
        "",
        "TARGET DISTANCES",
        "-" * 60,
        "5.0 km",
        "10.0 km",
        "15.0 km",
        "21.1 km",
        "42.2 km",
        "",
        "GRANULAR SCHEMA",
        "-" * 60,
        "secs = elapsed seconds",
        "km = cumulative distance in kilometers",
        "power = power",
        "hr = heart rate",
        "cad = cadence",
        "alt = altitude",
        "",
        "STATUS",
        "-" * 60,
        "Stage 1 preparation completed.",
    ])

    summary_path.write_text(
        "\n".join(summary_lines),
        encoding="utf-8"
    )

    print(
        f"Summary written to:\n{summary_path}"
    )


# ============================================================
# 8. MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("CAPSTONE - STAGE 1 DATA PREPARATION")
    print("=" * 70)
    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    try:

        # ----------------------------------------------------
        # Step 1
        # ----------------------------------------------------

        manifest = load_and_filter_manifest()

        # ----------------------------------------------------
        # Step 2
        # ----------------------------------------------------

        target_df = parse_all_targets(
            manifest
        )

        # ----------------------------------------------------
        # Step 3
        # ----------------------------------------------------

        activity = build_activity_index(
            manifest,
            target_df
        )

        activity_output = (
            OUTPUT_DIR /
            "stage1_activity_index.csv"
        )

        activity.to_csv(
            activity_output,
            index=False
        )

        print(
            f"\nSaved:\n{activity_output}"
        )

        # ----------------------------------------------------
        # Step 4
        # ----------------------------------------------------

        daily = build_daily_timeline(
            activity
        )

        daily_output = (
            OUTPUT_DIR /
            "stage1_daily_timeline.csv"
        )

        daily.to_csv(
            daily_output,
            index=False
        )

        print(
            f"\nSaved:\n{daily_output}"
        )

        # ----------------------------------------------------
        # Step 5
        # ----------------------------------------------------

        windows = build_28day_windows(
            daily
        )

        windows_output = (
            OUTPUT_DIR /
            "stage1_28day_window_index.csv"
        )

        windows.to_csv(
            windows_output,
            index=False
        )

        print(
            f"\nSaved:\n{windows_output}"
        )

        # ----------------------------------------------------
        # Step 6
        # ----------------------------------------------------

        missing = build_missing_activity_audit(
            activity
        )

        missing_output = (
            OUTPUT_DIR /
            "stage1_missing_activities.csv"
        )

        missing.to_csv(
            missing_output,
            index=False
        )

        print(
            f"\nSaved:\n{missing_output}"
        )

        # ----------------------------------------------------
        # Step 7
        # ----------------------------------------------------

        write_summary(
            activity,
            daily,
            windows,
            missing
        )

        # ----------------------------------------------------
        # FINAL VALIDATION
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("FINAL VALIDATION")
        print("=" * 70)

        print(
            f"Activities: "
            f"{len(activity):,}"
        )

        print(
            f"Athletes: "
            f"{activity['id'].nunique():,}"
        )

        print(
            f"Daily rows: "
            f"{len(daily):,}"
        )

        print(
            f"28-day target windows: "
            f"{len(windows):,}"
        )

        print(
            f"Granular issues: "
            f"{len(missing):,}"
        )

        # ----------------------------------------------------
        # Validate every target-anchored window
        # ----------------------------------------------------

        if not windows.empty:

            invalid_window_lengths = (
                windows["window_days"] != WINDOW_DAYS
            ).sum()

            invalid_rest_activity = (
                windows["activity_days"] +
                windows["rest_days"] != WINDOW_DAYS
            ).sum()

            print(
                f"Invalid window lengths: "
                f"{invalid_window_lengths:,}"
            )

            print(
                f"Invalid activity/rest totals: "
                f"{invalid_rest_activity:,}"
            )

            if (
                invalid_window_lengths > 0 or
                invalid_rest_activity > 0
            ):
                raise ValueError(
                    "Window validation failed."
                )

        print()
        print("=" * 70)
        print("STAGE 1 COMPLETE")
        print("=" * 70)

        print()
        print("Outputs:")
        print(f"1. {activity_output}")
        print(f"2. {daily_output}")
        print(f"3. {windows_output}")
        print(f"4. {missing_output}")
        print(
            f"5. {OUTPUT_DIR / 'stage1_summary.txt'}"
        )

        print()
        print(
            "Do NOT proceed to EDA until the output counts "
            "and target extraction have been checked."
        )

    except Exception as exc:

        error_log = (
            LOG_DIR /
            "stage1_error.log"
        )

        error_text = (
            "STAGE 1 ERROR\n"
            "=" * 60
            + "\n\n"
            + str(exc)
            + "\n\n"
            + traceback.format_exc()
        )

        error_log.write_text(
            error_text,
            encoding="utf-8"
        )

        print()
        print("=" * 70)
        print("STAGE 1 FAILED")
        print("=" * 70)

        print()
        print(str(exc))

        print()
        print(
            f"Full traceback saved to:\n{error_log}"
        )

        raise


if __name__ == "__main__":
    main()