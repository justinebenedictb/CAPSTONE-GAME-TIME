r"""
04_stage2_eda.py

CAPSTONE
Stage 2 - Exploratory Data Analysis (EDA)

Purpose
-------
Explore the prepared running dataset before Feature Engineering.

EDA sections:
    1. Feature variance and temporal distribution
    2. Autocorrelation and lag analysis
    3. Target variable distribution
    4. EDA findings relevant to Feature Engineering

IMPORTANT
---------
This stage intentionally DOES NOT calculate:
    - TSS
    - ATL
    - CTL
    - ACWR
    - GAP
    - ITRA Equivalent Distance

Those belong to Stage 3 Feature Engineering.

The purpose of EDA is to identify empirical patterns in the
raw/prepared data that inform those subsequent transformations.

Input:
    01_prepared/stage1_activity_index.csv
    01_prepared/stage1_daily_timeline.csv
    01_prepared/stage1_28day_window_index.csv

Output:
    02_eda/tables/
    02_eda/plots/
    02_eda/eda_summary.txt
"""

from pathlib import Path
import math
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(
    r"C:\Users\admin\Desktop\CAPSTONE GAME TIME"
)

INPUT_DIR = ROOT / "01_prepared"

OUTPUT_DIR = ROOT / "02_eda"
TABLE_DIR = OUTPUT_DIR / "tables"
PLOT_DIR = OUTPUT_DIR / "plots"

LOG_DIR = ROOT / "logs"

ACTIVITY_FILE = (
    INPUT_DIR /
    "stage1_activity_index.csv"
)

DAILY_FILE = (
    INPUT_DIR /
    "stage1_daily_timeline.csv"
)

WINDOW_FILE = (
    INPUT_DIR /
    "stage1_28day_window_index.csv"
)


# Standard performance targets
TARGET_DISTANCES = [
    5.0,
    10.0,
    15.0,
    21.1,
    42.2,
]


# ============================================================
# GENERAL UTILITIES
# ============================================================

def save_figure(filename):
    """
    Save the current matplotlib figure.
    """
    path = PLOT_DIR / filename

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return path


def numeric_summary(df, columns):
    """
    Generate descriptive statistics for numeric variables.
    """
    records = []

    for col in columns:

        if col not in df.columns:
            continue

        series = pd.to_numeric(
            df[col],
            errors="coerce"
        ).dropna()

        if series.empty:
            continue

        records.append({
            "feature": col,
            "n": len(series),
            "missing": int(
                df[col].isna().sum()
            ),
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "variance": series.var(),
            "min": series.min(),
            "q1": series.quantile(0.25),
            "q3": series.quantile(0.75),
            "max": series.max(),
            "skewness": series.skew(),
        })

    return pd.DataFrame(records)


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_stage1_data():

    print("=" * 70)
    print("LOADING STAGE 1 DATA")
    print("=" * 70)

    required_files = [
        ACTIVITY_FILE,
        DAILY_FILE,
        WINDOW_FILE,
    ]

    for path in required_files:

        if not path.exists():
            raise FileNotFoundError(
                f"Required Stage 1 file not found:\n{path}"
            )

    activity = pd.read_csv(
        ACTIVITY_FILE
    )

    # Stage 1 keeps mislabelled rides for audit only; EDA covers real runs.
    if "suspected_non_run" in activity.columns:
        activity = activity[~activity["suspected_non_run"].astype(bool)].copy()

    daily = pd.read_csv(
        DAILY_FILE
    )

    windows = pd.read_csv(
        WINDOW_FILE
    )

    # --------------------------------------------------------
    # Parse dates
    # --------------------------------------------------------

    if "datetime_utc" in activity.columns:
        activity["datetime_utc"] = pd.to_datetime(
            activity["datetime_utc"],
            errors="coerce",
            utc=True
        )

    if "calendar_date" in activity.columns:
        activity["calendar_date"] = pd.to_datetime(
            activity["calendar_date"],
            errors="coerce"
        )

    if "calendar_date" in daily.columns:
        daily["calendar_date"] = pd.to_datetime(
            daily["calendar_date"],
            errors="coerce"
        )

    if "window_start" in windows.columns:
        windows["window_start"] = pd.to_datetime(
            windows["window_start"],
            errors="coerce"
        )

    if "window_end" in windows.columns:
        windows["window_end"] = pd.to_datetime(
            windows["window_end"],
            errors="coerce"
        )

    if "target_date" in windows.columns:
        windows["target_date"] = pd.to_datetime(
            windows["target_date"],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Convert IDs to strings
    # --------------------------------------------------------

    for df in [activity, daily, windows]:

        if "id" in df.columns:
            df["id"] = (
                df["id"]
                .astype(str)
                .str.strip()
            )

    print(
        f"Activity rows: {len(activity):,}"
    )

    print(
        f"Daily rows: {len(daily):,}"
    )

    print(
        f"Window rows: {len(windows):,}"
    )

    print(
        f"Activity athletes: "
        f"{activity['id'].nunique():,}"
    )

    print(
        f"Daily athletes: "
        f"{daily['id'].nunique():,}"
    )

    return activity, daily, windows


# ============================================================
# 2. FEATURE VARIANCE
# ============================================================

def analyze_feature_variance(activity, daily):

    print()
    print("=" * 70)
    print("SECTION 1A - FEATURE VARIANCE")
    print("=" * 70)

    # Variables available at activity level
    activity_candidates = [
        "workout_time",
        "total_distance",
        "elevation_gain",
        "total_work",
        "coggan_tss",
        "average_speed",
        "average_hr",
        "average_cadence",
        "average_power",
        "calories",
    ]

    # Variables available in daily timeline
    daily_candidates = [
        "workout_time",
        "total_distance",
        "elevation_gain",
        "total_work",
        "coggan_tss",
        "average_speed",
        "average_hr",
        "average_cadence",
        "average_power",
        "calories",
    ]

    activity_summary = numeric_summary(
        activity,
        activity_candidates
    )

    daily_summary = numeric_summary(
        daily,
        daily_candidates
    )

    activity_summary.to_csv(
        TABLE_DIR /
        "eda_activity_feature_variance.csv",
        index=False
    )

    daily_summary.to_csv(
        TABLE_DIR /
        "eda_daily_feature_variance.csv",
        index=False
    )

    print()
    print("Activity-level variance:")

    if not activity_summary.empty:
        print(
            activity_summary[
                [
                    "feature",
                    "n",
                    "mean",
                    "std",
                    "variance",
                    "min",
                    "median",
                    "max",
                ]
            ].to_string(index=False)
        )

    print()
    print("Daily-level variance:")

    if not daily_summary.empty:
        print(
            daily_summary[
                [
                    "feature",
                    "n",
                    "mean",
                    "std",
                    "variance",
                    "min",
                    "median",
                    "max",
                ]
            ].to_string(index=False)
        )

    return activity_summary, daily_summary


# ============================================================
# 3. FEATURE DISTRIBUTIONS
# ============================================================

def plot_feature_distributions(activity):

    print()
    print("=" * 70)
    print("SECTION 1B - FEATURE DISTRIBUTIONS")
    print("=" * 70)

    features = [
        "total_distance",
        "workout_time",
        "elevation_gain",
        "average_speed",
        "average_hr",
        "average_power",
        "average_cadence",
    ]

    for feature in features:

        if feature not in activity.columns:
            continue

        values = pd.to_numeric(
            activity[feature],
            errors="coerce"
        ).dropna()

        if values.empty:
            continue

        plt.figure(
            figsize=(8, 5)
        )

        plt.hist(
            values,
            bins=50
        )

        plt.xlabel(feature)
        plt.ylabel("Number of Activities")
        plt.title(
            f"Distribution of {feature}"
        )

        save_figure(
            f"distribution_{feature}.png"
        )

    print(
        "Feature distribution plots generated."
    )


# ============================================================
# 4. TEMPORAL DISTRIBUTION
# ============================================================

def analyze_temporal_distribution(
    activity,
    daily
):

    print()
    print("=" * 70)
    print("SECTION 1C - TEMPORAL DISTRIBUTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Monthly activity frequency
    # --------------------------------------------------------

    activity = activity.copy()

    activity["month"] = (
        activity["datetime_utc"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly = (
        activity
        .groupby("month")
        .size()
        .reset_index(
            name="activity_count"
        )
    )

    monthly.to_csv(
        TABLE_DIR /
        "eda_monthly_activity_frequency.csv",
        index=False
    )

    plt.figure(
        figsize=(14, 5)
    )

    plt.plot(
        monthly["month"],
        monthly["activity_count"]
    )

    plt.xlabel("Month")
    plt.ylabel("Number of Activities")
    plt.title(
        "Monthly Running Activity Frequency"
    )

    plt.xticks(
        rotation=90
    )

    save_figure(
        "temporal_monthly_activity_frequency.png"
    )

    # --------------------------------------------------------
    # Activity vs rest days
    # --------------------------------------------------------

    if "is_activity_day" in daily.columns:

        activity_rest = (
            daily["is_activity_day"]
            .value_counts()
            .rename(
                index={
                    0: "Rest Day",
                    1: "Activity Day"
                }
            )
        )

        activity_rest.to_csv(
            TABLE_DIR /
            "eda_activity_vs_rest_days.csv"
        )

        plt.figure(
            figsize=(7, 5)
        )

        plt.bar(
            activity_rest.index.astype(str),
            activity_rest.values
        )

        plt.xlabel("Day Type")
        plt.ylabel("Number of Days")
        plt.title(
            "Activity Days vs Rest Days"
        )

        save_figure(
            "temporal_activity_vs_rest_days.png"
        )

    # --------------------------------------------------------
    # Activities per athlete
    # --------------------------------------------------------

    athlete_activity_counts = (
        activity
        .groupby("id")
        .size()
        .reset_index(
            name="activity_count"
        )
    )

    athlete_activity_counts.to_csv(
        TABLE_DIR /
        "eda_athlete_activity_counts.csv",
        index=False
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        athlete_activity_counts[
            "activity_count"
        ],
        bins=50
    )

    plt.xlabel(
        "Number of Activities per Athlete"
    )

    plt.ylabel(
        "Number of Athletes"
    )

    plt.title(
        "Distribution of Activities per Athlete"
    )

    save_figure(
        "temporal_activities_per_athlete.png"
    )

    # --------------------------------------------------------
    # Daily training frequency per athlete
    # --------------------------------------------------------

    if "is_activity_day" in daily.columns:

        athlete_training_days = (
            daily
            .groupby("id")["is_activity_day"]
            .sum()
            .reset_index(
                name="activity_days"
            )
        )

        athlete_training_days.to_csv(
            TABLE_DIR /
            "eda_athlete_training_days.csv",
            index=False
        )

    print(
        "Temporal distribution analysis complete."
    )


# ============================================================
# 5. AUTOCORRELATION
# ============================================================

def calculate_autocorrelation(
    daily,
    feature,
    max_lag=28
):

    """
    Calculate average athlete-level autocorrelation.

    For each athlete:
        - sort chronologically
        - calculate correlation between X_t and X_(t-lag)

    The result is averaged across athletes with sufficient
    observations.
    """

    if feature not in daily.columns:
        return pd.DataFrame()

    results = []

    for athlete_id, athlete_df in daily.groupby(
        "id",
        sort=False
    ):

        series = pd.to_numeric(
            athlete_df[feature],
            errors="coerce"
        )

        series = series.reset_index(
            drop=True
        )

        if series.notna().sum() < (
            max_lag + 5
        ):
            continue

        athlete_results = []

        for lag in range(
            1,
            max_lag + 1
        ):

            lagged = pd.DataFrame({
                "current": series,
                "lagged": series.shift(lag)
            }).dropna()

            if len(lagged) < 5:
                continue

            correlation = (
                lagged["current"]
                .corr(
                    lagged["lagged"]
                )
            )

            if pd.notna(correlation):

                athlete_results.append(
                    correlation
                )

        for lag, correlation in enumerate(
            athlete_results,
            start=1
        ):

            results.append({
                "id": athlete_id,
                "feature": feature,
                "lag_days": lag,
                "autocorrelation": correlation,
            })

    if not results:
        return pd.DataFrame()

    result = pd.DataFrame(
        results
    )

    summary = (
        result
        .groupby(
            ["feature", "lag_days"]
        )["autocorrelation"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std",
            ]
        )
        .reset_index()
    )

    return summary


def analyze_autocorrelation(daily):

    print()
    print("=" * 70)
    print("SECTION 2A - AUTOCORRELATION")
    print("=" * 70)

    features = [
        "total_distance",
        "workout_time",
        "elevation_gain",
        "average_hr",
        "average_speed",
    ]

    all_results = []

    for feature in features:

        print(
            f"Calculating autocorrelation: "
            f"{feature}"
        )

        result = calculate_autocorrelation(
            daily,
            feature,
            max_lag=28
        )

        if not result.empty:
            all_results.append(
                result
            )

            plt.figure(
                figsize=(9, 5)
            )

            plt.plot(
                result["lag_days"],
                result["mean"],
                marker="o"
            )

            plt.axhline(
                0,
                linestyle="--"
            )

            plt.xlabel(
                "Lag (days)"
            )

            plt.ylabel(
                "Mean Autocorrelation"
            )

            plt.title(
                f"Autocorrelation of {feature}"
            )

            save_figure(
                f"autocorrelation_{feature}.png"
            )

    if all_results:

        combined = pd.concat(
            all_results,
            ignore_index=True
        )

        combined.to_csv(
            TABLE_DIR /
            "eda_autocorrelation.csv",
            index=False
        )

        print()
        print(
            combined.to_string(
                index=False
            )
        )

        return combined

    print(
        "No autocorrelation results generated."
    )

    return pd.DataFrame()


# ============================================================
# 6. LAGGED TRAINING LOAD PROXIES
# ============================================================

def calculate_lagged_load_relationships(
    daily,
    windows
):

    """
    Exploratory analysis of whether recent raw training
    variables are associated with target performance.

    This is intentionally NOT TSS/ATL/CTL/ACWR.

    We use raw workload proxies:
        - total_distance
        - workout_time
        - elevation_gain

    For each target observation, calculate prior:
        1-day
        3-day
        7-day
        14-day
        28-day

    cumulative values.

    Correlations are then calculated against standardized
    target time.

    NOTE:
        Lower target time = better performance.
    """

    print()
    print("=" * 70)
    print("SECTION 2B - LAGGED LOAD / PERFORMANCE ANALYSIS")
    print("=" * 70)

    load_features = [
        "total_distance",
        "workout_time",
        "elevation_gain",
    ]

    horizons = [
        1,
        3,
        7,
        14,
        28,
    ]

    if windows.empty:
        print(
            "No target windows available."
        )
        return pd.DataFrame()

    # --------------------------------------------------------
    # Build lookup
    # --------------------------------------------------------

    daily = daily.copy()

    daily["calendar_date"] = pd.to_datetime(
        daily["calendar_date"]
    )

    records = []

    for _, target in windows.iterrows():

        athlete_id = str(
            target["id"]
        )

        target_date = pd.Timestamp(
            target["target_date"]
        )

        target_seconds = pd.to_numeric(
            target["target_seconds"],
            errors="coerce"
        )

        if pd.isna(target_seconds):
            continue

        athlete_daily = daily[
            daily["id"] == athlete_id
        ].copy()

        athlete_daily = athlete_daily[
            athlete_daily["calendar_date"]
            < target_date
        ]

        if athlete_daily.empty:
            continue

        for horizon in horizons:

            start_date = (
                target_date -
                pd.Timedelta(
                    days=horizon
                )
            )

            history = athlete_daily[
                (
                    athlete_daily["calendar_date"]
                    >= start_date
                )
                &
                (
                    athlete_daily["calendar_date"]
                    < target_date
                )
            ]

            if history.empty:
                continue

            record = {
                "id": athlete_id,
                "target_date": target_date,
                "target_distance_km": target[
                    "target_distance_km"
                ],
                "target_seconds": target_seconds,
                "horizon_days": horizon,
            }

            for feature in load_features:

                if feature not in history.columns:
                    continue

                values = pd.to_numeric(
                    history[feature],
                    errors="coerce"
                )

                record[
                    f"{feature}_{horizon}d_sum"
                ] = values.sum(
                    skipna=True
                )

            records.append(record)

    lag_df = pd.DataFrame(
        records
    )

    if lag_df.empty:
        return lag_df

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    correlation_records = []

    for distance in TARGET_DISTANCES:

        distance_df = lag_df[
            np.isclose(
                lag_df["target_distance_km"],
                distance
            )
        ]

        if distance_df.empty:
            continue

        for horizon in horizons:

            for feature in load_features:

                column = (
                    f"{feature}_{horizon}d_sum"
                )

                if column not in distance_df.columns:
                    continue

                valid = distance_df[
                    [
                        column,
                        "target_seconds"
                    ]
                ].dropna()

                if len(valid) < 10:
                    correlation = np.nan
                else:
                    correlation = (
                        valid[column]
                        .corr(
                            valid["target_seconds"]
                        )
                    )

                correlation_records.append({
                    "target_distance_km": distance,
                    "horizon_days": horizon,
                    "load_proxy": feature,
                    "n": len(valid),
                    "pearson_r": correlation,
                })

    correlations = pd.DataFrame(
        correlation_records
    )

    lag_df.to_csv(
        TABLE_DIR /
        "eda_lagged_load_observations.csv",
        index=False
    )

    correlations.to_csv(
        TABLE_DIR /
        "eda_lagged_load_correlations.csv",
        index=False
    )

    # --------------------------------------------------------
    # Plot correlations
    # --------------------------------------------------------

    for distance in TARGET_DISTANCES:

        distance_corr = correlations[
            np.isclose(
                correlations[
                    "target_distance_km"
                ],
                distance
            )
        ]

        if distance_corr.empty:
            continue

        for feature in load_features:

            feature_corr = distance_corr[
                distance_corr[
                    "load_proxy"
                ] == feature
            ]

            if feature_corr.empty:
                continue

            plt.figure(
                figsize=(8, 5)
            )

            plt.plot(
                feature_corr[
                    "horizon_days"
                ],
                feature_corr[
                    "pearson_r"
                ],
                marker="o"
            )

            plt.axhline(
                0,
                linestyle="--"
            )

            plt.xlabel(
                "Prior Training Window (days)"
            )

            plt.ylabel(
                "Pearson Correlation"
            )

            plt.title(
                f"{feature} vs {distance} km "
                "Target Time"
            )

            save_figure(
                f"lag_correlation_"
                f"{feature}_"
                f"{str(distance).replace('.', '_')}km.png"
            )

    print()
    print(
        correlations.to_string(
            index=False
        )
    )

    return correlations


# ============================================================
# 7. TARGET VARIABLE DISTRIBUTION
# ============================================================

def analyze_target_distribution(
    activity,
    windows
):

    print()
    print("=" * 70)
    print("SECTION 3 - TARGET VARIABLE DISTRIBUTION")
    print("=" * 70)

    target_records = []

    for distance_label, distance in [
        ("5.0", 5.0),
        ("10.0", 10.0),
        ("15.0", 15.0),
        ("21.1", 21.1),
        ("42.2", 42.2),
    ]:

        column = (
            f"target_"
            f"{distance_label.replace('.', '_')}"
            f"km_sec"
        )

        if column not in activity.columns:
            continue

        values = pd.to_numeric(
            activity[column],
            errors="coerce"
        ).dropna()

        for value in values:

            target_records.append({
                "target_distance_km": distance,
                "target_seconds": value,
                "target_time": (
                    f"{int(value // 3600):02d}:"
                    f"{int((value % 3600) // 60):02d}:"
                    f"{int(value % 60):02d}"
                ),
            })

    targets = pd.DataFrame(
        target_records
    )

    if targets.empty:
        print(
            "No target observations found."
        )
        return pd.DataFrame()

    # --------------------------------------------------------
    # Descriptive statistics
    # --------------------------------------------------------

    records = []

    for distance, group in targets.groupby(
        "target_distance_km"
    ):

        values = group[
            "target_seconds"
        ]

        records.append({
            "target_distance_km": distance,
            "n": len(values),
            "mean_seconds": values.mean(),
            "median_seconds": values.median(),
            "std_seconds": values.std(),
            "variance_seconds": values.var(),
            "min_seconds": values.min(),
            "q1_seconds": values.quantile(0.25),
            "q3_seconds": values.quantile(0.75),
            "max_seconds": values.max(),
            "skewness": values.skew(),
        })

    target_summary = pd.DataFrame(
        records
    )

    target_summary.to_csv(
        TABLE_DIR /
        "eda_target_distribution_summary.csv",
        index=False
    )

    targets.to_csv(
        TABLE_DIR /
        "eda_target_observations.csv",
        index=False
    )

    print()
    print(
        target_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Histograms
    # --------------------------------------------------------

    for distance, group in targets.groupby(
        "target_distance_km"
    ):

        plt.figure(
            figsize=(8, 5)
        )

        plt.hist(
            group["target_seconds"],
            bins=40
        )

        plt.xlabel(
            "Target Time (seconds)"
        )

        plt.ylabel(
            "Number of Observations"
        )

        plt.title(
            f"Distribution of {distance} km "
            "Target Times"
        )

        save_figure(
            f"target_distribution_"
            f"{str(distance).replace('.', '_')}km.png"
        )

    # --------------------------------------------------------
    # Boxplot across distances
    # --------------------------------------------------------

    box_data = []

    labels = []

    for distance in TARGET_DISTANCES:

        values = targets.loc[
            np.isclose(
                targets[
                    "target_distance_km"
                ],
                distance
            ),
            "target_seconds"
        ].dropna()

        if not values.empty:

            box_data.append(
                values.values
            )

            labels.append(
                f"{distance} km"
            )

    if box_data:

        plt.figure(
            figsize=(10, 6)
        )

        plt.boxplot(
            box_data,
            labels=labels
        )

        plt.xlabel(
            "Target Distance"
        )

        plt.ylabel(
            "Target Time (seconds)"
        )

        plt.title(
            "Target Performance Distributions"
        )

        save_figure(
            "target_distributions_boxplot.png"
        )

    # --------------------------------------------------------
    # Target observations by athlete
    # --------------------------------------------------------

    if "id" in activity.columns:

        target_counts = []

        for athlete_id, group in activity.groupby(
            "id"
        ):

            count = 0

            for distance_label in [
                "5.0",
                "10.0",
                "15.0",
                "21.1",
                "42.2",
            ]:

                column = (
                    f"target_"
                    f"{distance_label.replace('.', '_')}"
                    f"km_sec"
                )

                if column in group.columns:
                    count += group[
                        column
                    ].notna().sum()

            target_counts.append({
                "id": athlete_id,
                "target_observation_count": count,
            })

        target_counts = pd.DataFrame(
            target_counts
        )

        target_counts.to_csv(
            TABLE_DIR /
            "eda_target_observations_per_athlete.csv",
            index=False
        )

    return target_summary


# ============================================================
# 8. TARGET OUTLIER SCREENING
# ============================================================

def screen_target_outliers(activity):

    print()
    print("=" * 70)
    print("TARGET OUTLIER SCREENING")
    print("=" * 70)

    records = []

    target_columns = {
        "5.0": "target_5_0km_sec",
        "10.0": "target_10_0km_sec",
        "15.0": "target_15_0km_sec",
        "21.1": "target_21_1km_sec",
        "42.2": "target_42_2km_sec",
    }

    for distance_label, column in target_columns.items():

        if column not in activity.columns:
            continue

        values = pd.to_numeric(
            activity[column],
            errors="coerce"
        )

        valid = values.dropna()

        if len(valid) < 4:
            continue

        q1 = valid.quantile(0.25)
        q3 = valid.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_mask = (
            (values < lower) |
            (values > upper)
        )

        records.append({
            "target_distance_km": float(
                distance_label
            ),
            "n": len(valid),
            "q1_seconds": q1,
            "q3_seconds": q3,
            "iqr_seconds": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": int(
                outlier_mask.sum()
            ),
            "outlier_percentage": (
                outlier_mask.sum()
                / len(valid)
                * 100
            ),
        })

    result = pd.DataFrame(
        records
    )

    result.to_csv(
        TABLE_DIR /
        "eda_target_outlier_screening.csv",
        index=False
    )

    if not result.empty:
        print(
            result.to_string(
                index=False
            )
        )

    return result


# ============================================================
# 9. EDA SUMMARY
# ============================================================

def write_eda_summary(
    activity,
    daily,
    windows,
    variance_activity,
    variance_daily,
    autocorrelation,
    lag_correlations,
    target_summary,
    outlier_summary,
):

    print()
    print("=" * 70)
    print("WRITING EDA SUMMARY")
    print("=" * 70)

    summary = []

    summary.append(
        "CAPSTONE - STAGE 2 EXPLORATORY DATA ANALYSIS"
    )

    summary.append("=" * 70)
    summary.append("")

    summary.append(
        "DATASET OVERVIEW"
    )

    summary.append("-" * 70)

    summary.append(
        f"Activity observations: "
        f"{len(activity):,}"
    )

    summary.append(
        f"Unique athletes: "
        f"{activity['id'].nunique():,}"
    )

    summary.append(
        f"Daily observations: "
        f"{len(daily):,}"
    )

    summary.append(
        f"28-day target windows: "
        f"{len(windows):,}"
    )

    summary.append("")

    summary.append(
        "1. FEATURE VARIANCE AND TEMPORAL DISTRIBUTION"
    )

    summary.append("-" * 70)

    if not variance_activity.empty:

        highest_variance = (
            variance_activity
            .sort_values(
                "variance",
                ascending=False
            )
            .head(10)
        )

        summary.append(
            "Highest raw activity-level variances:"
        )

        for _, row in highest_variance.iterrows():

            summary.append(
                f"  - {row['feature']}: "
                f"variance={row['variance']:.4f}, "
                f"std={row['std']:.4f}"
            )

    summary.append("")

    summary.append(
        "2. AUTOCORRELATION AND LAG ANALYSIS"
    )

    summary.append("-" * 70)

    if not autocorrelation.empty:

        for feature in autocorrelation[
            "feature"
        ].unique():

            feature_acf = autocorrelation[
                autocorrelation[
                    "feature"
                ] == feature
            ]

            strongest = (
                feature_acf
                .loc[
                    feature_acf[
                        "mean"
                    ].abs().idxmax()
                ]
            )

            summary.append(
                f"{feature}: strongest observed "
                f"absolute mean autocorrelation "
                f"at lag {int(strongest['lag_days'])} "
                f"days "
                f"(r={strongest['mean']:.3f})"
            )

    summary.append("")

    if not lag_correlations.empty:

        summary.append(
            "Lagged raw workload-performance "
            "relationships were explored using "
            "1, 3, 7, 14, and 28-day horizons."
        )

        summary.append(
            "These are exploratory associations and "
            "should not be interpreted as causal effects."
        )

    summary.append("")

    summary.append(
        "3. TARGET VARIABLE DISTRIBUTION"
    )

    summary.append("-" * 70)

    if not target_summary.empty:

        for _, row in target_summary.iterrows():

            summary.append(
                f"{row['target_distance_km']} km: "
                f"n={int(row['n'])}, "
                f"median={row['median_seconds']:.2f}s, "
                f"mean={row['mean_seconds']:.2f}s, "
                f"SD={row['std_seconds']:.2f}s, "
                f"skew={row['skewness']:.3f}"
            )

    summary.append("")

    summary.append(
        "OUTLIER SCREENING"
    )

    summary.append("-" * 70)

    if not outlier_summary.empty:

        for _, row in outlier_summary.iterrows():

            summary.append(
                f"{row['target_distance_km']} km: "
                f"{int(row['outlier_count'])} "
                f"potential IQR outliers "
                f"({row['outlier_percentage']:.2f}%)"
            )

    summary.append("")

    summary.append(
        "IMPORTANT METHODOLOGICAL NOTE"
    )

    summary.append("-" * 70)

    summary.append(
        "EDA findings should inform Feature Engineering, "
        "but should not be treated as proof that a particular "
        "feature is causally necessary."
    )

    summary.append(
        "The next stage will use the observed temporal "
        "patterns, variability, and workload-performance "
        "relationships to motivate terrain normalization "
        "and cumulative training-load features."
    )

    summary_path = (
        OUTPUT_DIR /
        "eda_summary.txt"
    )

    summary_path.write_text(
        "\n".join(summary),
        encoding="utf-8"
    )

    print(
        f"EDA summary saved to:\n{summary_path}"
    )


# ============================================================
# 10. MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("CAPSTONE - STAGE 2 EDA")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    try:

        # ----------------------------------------------------
        # Load Stage 1 outputs
        # ----------------------------------------------------

        (
            activity,
            daily,
            windows
        ) = load_stage1_data()

        # ----------------------------------------------------
        # SECTION 1
        # ----------------------------------------------------

        (
            variance_activity,
            variance_daily
        ) = analyze_feature_variance(
            activity,
            daily
        )

        plot_feature_distributions(
            activity
        )

        analyze_temporal_distribution(
            activity,
            daily
        )

        # ----------------------------------------------------
        # SECTION 2
        # ----------------------------------------------------

        autocorrelation = (
            analyze_autocorrelation(
                daily
            )
        )

        lag_correlations = (
            calculate_lagged_load_relationships(
                daily,
                windows
            )
        )

        # ----------------------------------------------------
        # SECTION 3
        # ----------------------------------------------------

        target_summary = (
            analyze_target_distribution(
                activity,
                windows
            )
        )

        outlier_summary = (
            screen_target_outliers(
                activity
            )
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        write_eda_summary(
            activity,
            daily,
            windows,
            variance_activity,
            variance_daily,
            autocorrelation,
            lag_correlations,
            target_summary,
            outlier_summary,
        )

        # ----------------------------------------------------
        # Final validation
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("EDA COMPLETE")
        print("=" * 70)

        print()
        print(
            f"Tables written to:\n{TABLE_DIR}"
        )

        print()
        print(
            f"Plots written to:\n{PLOT_DIR}"
        )

        print()
        print(
            f"Summary written to:\n"
            f"{OUTPUT_DIR / 'eda_summary.txt'}"
        )

        print()
        print(
            "IMPORTANT:"
        )

        print(
            "Do not begin Feature Engineering yet."
        )

        print(
            "First inspect the generated EDA tables and "
            "plots so the Stage 3 feature decisions can "
            "be based on the actual observed dataset."
        )

    except Exception as exc:

        error_log = (
            LOG_DIR /
            "stage2_eda_error.log"
        )

        error_text = (
            "STAGE 2 EDA ERROR\n"
            "=" * 70
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
        print("STAGE 2 EDA FAILED")
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