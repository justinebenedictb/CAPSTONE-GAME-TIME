from pathlib import Path
import csv
import os
import sys
import time
import zipfile
import shutil
from datetime import datetime

import pandas as pd
import requests


# ============================================================
# CAPSTONE
# SCRIPT 02 - OSF DATASET DOWNLOADER
#
# PURPOSE
# ------------------------------------------------------------
# 1. Read the verified OSF audit from Script 01
# 2. Download each required athlete ZIP only once
# 3. Extract ONLY the activity CSVs required by the manifest
# 4. Track every download and extraction
# 5. Resume safely if the script is interrupted
#
# IMPORTANT
# ------------------------------------------------------------
# This script DOES NOT perform data preparation yet.
#
# Output structure:
#
# CAPSTONE GAME TIME
# |
# +-- 00_raw
# |   |
# |   +-- granular
# |       |
# |       +-- ATHLETE_ID_1
# |       |   +-- activity.csv
# |       |   +-- activity.csv
# |       |
# |       +-- ATHLETE_ID_2
# |
# +-- logs
#     |
#     +-- osf_dataset_audit.csv
#     +-- download_audit.csv
#     +-- download_summary.txt
#
# ============================================================


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(
    r"c:\Users\admin\Desktop\CAPSTONE GAME TIME"
)

AUDIT_FILE = (
    ROOT
    / "logs"
    / "osf_dataset_audit.csv"
)

RAW_DIR = (
    ROOT
    / "00_raw"
)

GRANULAR_DIR = (
    RAW_DIR
    / "granular"
)

ZIP_DIR = (
    RAW_DIR
    / "athlete_zips"
)

LOG_DIR = (
    ROOT
    / "logs"
)

DOWNLOAD_AUDIT_FILE = (
    LOG_DIR
    / "download_audit.csv"
)

DOWNLOAD_SUMMARY_FILE = (
    LOG_DIR
    / "download_summary.txt"
)


# ============================================================
# 2. DOWNLOAD SETTINGS
# ============================================================

# Keep downloaded ZIP files after extraction?
#
# False = ZIP is deleted after successful extraction.
# This saves disk space.
#
# True = ZIP files remain in:
#        00_raw\athlete_zips
#
KEEP_ZIPS = False


# Number of times to retry failed HTTP requests.
MAX_RETRIES = 5


# Timeout for individual HTTP requests.
REQUEST_TIMEOUT = 120


# How many files to process before printing a progress message.
PROGRESS_EVERY = 10


# ============================================================
# 3. CREATE DIRECTORIES
# ============================================================

RAW_DIR.mkdir(
    parents=True,
    exist_ok=True
)

GRANULAR_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ZIP_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 4. HTTP SESSION
# ============================================================

session = requests.Session()

session.headers.update(
    {
        "User-Agent":
            "CAPSTONE-OSF-Dataset-Downloader/1.0"
    }
)

# Optional OSF token.
#
# Public OSF projects normally do not require this.
#
# If needed:
#
# PowerShell:
#
# $env:OSF_TOKEN="YOUR_TOKEN"
#
# Do NOT put the token directly into this script.

OSF_TOKEN = os.environ.get(
    "OSF_TOKEN",
    ""
).strip()

if OSF_TOKEN:

    session.headers.update(
        {
            "Authorization":
                f"Bearer {OSF_TOKEN}"
        }
    )


# ============================================================
# 5. TIMESTAMP
# ============================================================

def timestamp():

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# ============================================================
# 6. LOAD AUDIT
# ============================================================

def load_audit():

    print()
    print("=" * 70)
    print("LOADING OSF AUDIT")
    print("=" * 70)

    if not AUDIT_FILE.exists():

        print()
        print(
            "ERROR: Audit file was not found:"
        )

        print(
            AUDIT_FILE
        )

        print()
        print(
            "Run Script 01 first."
        )

        sys.exit(1)

    audit = pd.read_csv(
        AUDIT_FILE,
        dtype=str,
        keep_default_na=False
    )

    required_columns = {
        "athlete_id",
        "activity_file",
        "zip_status",
        "osf_zip_name",
        "osf_zip_path",
        "osf_download_url"
    }

    missing = (
        required_columns
        - set(audit.columns)
    )

    if missing:

        print()
        print(
            "ERROR: The audit is missing "
            "required columns:"
        )

        for column in sorted(missing):

            print(
                f"  - {column}"
            )

        sys.exit(1)

    print()
    print(
        f"Audit loaded: {AUDIT_FILE}"
    )

    print(
        f"Total audit rows: "
        f"{len(audit):,}"
    )

    matched = (
        audit["zip_status"]
        == "MATCHED"
    ).sum()

    missing_zip = (
        audit["zip_status"]
        == "MISSING_ZIP"
    ).sum()

    ambiguous = (
        audit["zip_status"]
        == "AMBIGUOUS_MULTIPLE_ZIPS"
    ).sum()

    print(
        f"Matched activity pairs: "
        f"{matched:,}"
    )

    print(
        f"Missing ZIP activity pairs: "
        f"{missing_zip:,}"
    )

    print(
        f"Ambiguous ZIP activity pairs: "
        f"{ambiguous:,}"
    )

    return audit


# ============================================================
# 7. LOAD EXISTING DOWNLOAD TRACKING
# ============================================================

def load_existing_tracking():

    if not DOWNLOAD_AUDIT_FILE.exists():

        return {}

    try:

        existing = pd.read_csv(
            DOWNLOAD_AUDIT_FILE,
            dtype=str,
            keep_default_na=False
        )

    except Exception as exc:

        print()
        print(
            "WARNING: Could not read existing "
            "download audit."
        )

        print(exc)

        return {}

    tracking = {}

    for row in existing.to_dict(
        orient="records"
    ):

        key = (
            row.get(
                "athlete_id",
                ""
            ),
            row.get(
                "activity_file",
                ""
            )
        )

        tracking[key] = row

    return tracking


# ============================================================
# 8. SAVE TRACKING
# ============================================================

def save_tracking(
    tracking
):

    if not tracking:

        return

    rows = list(
        tracking.values()
    )

    columns = [
        "athlete_id",
        "activity_file",
        "zip_status",
        "download_status",
        "extraction_status",
        "local_path",
        "zip_size_bytes",
        "activity_size_bytes",
        "timestamp",
        "error"
    ]

    df = pd.DataFrame(
        rows
    )

    for column in columns:

        if column not in df.columns:

            df[column] = ""

    df = df[
        columns
    ]

    df = df.sort_values(
        [
            "athlete_id",
            "activity_file"
        ]
    )

    df.to_csv(
        DOWNLOAD_AUDIT_FILE,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 9. DOWNLOAD ZIP
# ============================================================

def download_zip(
    url,
    destination
):

    temp_destination = Path(
        str(destination)
        + ".part"
    )

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            # Remove incomplete previous attempt.
            if temp_destination.exists():

                temp_destination.unlink()

            print(
                f"      Download attempt "
                f"{attempt}/{MAX_RETRIES}"
            )

            with session.get(
                url,
                stream=True,
                timeout=REQUEST_TIMEOUT
            ) as response:

                response.raise_for_status()

                with open(
                    temp_destination,
                    "wb"
                ) as output:

                    for chunk in response.iter_content(
                        chunk_size=1024 * 1024
                    ):

                        if chunk:

                            output.write(
                                chunk
                            )

            # Make sure something was downloaded.
            if not temp_destination.exists():

                raise RuntimeError(
                    "Temporary ZIP file was not created."
                )

            if temp_destination.stat().st_size == 0:

                raise RuntimeError(
                    "Downloaded ZIP file is empty."
                )

            # Rename completed download.
            temp_destination.replace(
                destination
            )

            return True, ""

        except Exception as exc:

            last_error = str(exc)

            print(
                f"      ERROR: {last_error}"
            )

            if attempt < MAX_RETRIES:

                wait = min(
                    2 ** (attempt - 1),
                    30
                )

                print(
                    f"      Retrying in "
                    f"{wait} seconds..."
                )

                time.sleep(
                    wait
                )

    return False, last_error


# ============================================================
# 10. FIND REQUIRED FILE INSIDE ZIP
# ============================================================

def find_zip_member(
    zip_file,
    activity_filename
):

    target = Path(
        activity_filename
    ).name.lower()

    matches = []

    for member in zip_file.infolist():

        # Ignore directories.
        if member.is_dir():

            continue

        member_basename = (
            Path(
                member.filename
            )
            .name
            .lower()
        )

        if member_basename == target:

            matches.append(
                member
            )

    if len(matches) == 0:

        return None, "NOT_FOUND"

    if len(matches) > 1:

        return None, "MULTIPLE_MATCHES"

    return matches[0], "FOUND"


# ============================================================
# 11. EXTRACT ONE ACTIVITY SAFELY
# ============================================================

def extract_activity(
    zip_file,
    member,
    destination
):

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Extract manually rather than using ZipFile.extract()
    # so the destination is explicitly controlled.

    with zip_file.open(
        member,
        "r"
    ) as source:

        with open(
            destination,
            "wb"
        ) as target:

            shutil.copyfileobj(
                source,
                target,
                length=1024 * 1024
            )

    if not destination.exists():

        raise RuntimeError(
            "Extraction completed but "
            "destination file does not exist."
        )

    size = (
        destination.stat()
        .st_size
    )

    if size == 0:

        raise RuntimeError(
            "Extracted activity CSV is empty."
        )

    return size


# ============================================================
# 12. PROCESS ONE ATHLETE
# ============================================================

def process_athlete(
    athlete_id,
    athlete_rows,
    tracking
):

    print()
    print("-" * 70)
    print(
        f"ATHLETE: {athlete_id}"
    )

    first_row = (
        athlete_rows.iloc[0]
    )

    zip_name = (
        first_row["osf_zip_name"]
    )

    download_url = (
        first_row["osf_download_url"]
    )

    if not download_url:

        print(
            "  ERROR: No OSF download URL."
        )

        for row in athlete_rows.itertuples(
            index=False
        ):

            key = (
                row.athlete_id,
                row.activity_file
            )

            tracking[key] = {
                "athlete_id":
                    row.athlete_id,

                "activity_file":
                    row.activity_file,

                "zip_status":
                    row.zip_status,

                "download_status":
                    "FAILED",

                "extraction_status":
                    "NOT_ATTEMPTED",

                "local_path":
                    "",

                "zip_size_bytes":
                    "",

                "activity_size_bytes":
                    "",

                "timestamp":
                    timestamp(),

                "error":
                    "Missing OSF download URL"
            }

        return 0, 0

    # --------------------------------------------------------
    # Local ZIP path
    # --------------------------------------------------------

    zip_path = (
        ZIP_DIR
        / f"{athlete_id}.zip"
    )

    # --------------------------------------------------------
    # Determine whether ZIP already exists
    # --------------------------------------------------------

    if zip_path.exists() and zip_path.stat().st_size > 0:

        print(
            f"  ZIP already exists: "
            f"{zip_path.name}"
        )

        download_status = (
            "ALREADY_PRESENT"
        )

    else:

        print(
            f"  ZIP: {zip_name}"
        )

        print(
            "  Downloading..."
        )

        success, error = download_zip(
            download_url,
            zip_path
        )

        if not success:

            print(
                "  ZIP DOWNLOAD FAILED."
            )

            for row in athlete_rows.itertuples(
                index=False
            ):

                key = (
                    row.athlete_id,
                    row.activity_file
                )

                tracking[key] = {
                    "athlete_id":
                        row.athlete_id,

                    "activity_file":
                        row.activity_file,

                    "zip_status":
                        row.zip_status,

                    "download_status":
                        "FAILED",

                    "extraction_status":
                        "NOT_ATTEMPTED",

                    "local_path":
                        "",

                    "zip_size_bytes":
                        "",

                    "activity_size_bytes":
                        "",

                    "timestamp":
                        timestamp(),

                    "error":
                        error
                }

            return 0, 0

        download_status = (
            "DOWNLOADED"
        )

        print(
            "  ZIP download complete."
        )

    zip_size = (
        zip_path.stat()
        .st_size
    )

    extracted_count = 0
    skipped_count = 0

    # --------------------------------------------------------
    # Open ZIP
    # --------------------------------------------------------

    try:

        with zipfile.ZipFile(
            zip_path,
            "r"
        ) as archive:

            bad_file = (
                archive.testzip()
            )

            if bad_file:

                raise RuntimeError(
                    "ZIP integrity check failed "
                    f"at: {bad_file}"
                )

            # ------------------------------------------------
            # Process required activities
            # ------------------------------------------------

            for row in athlete_rows.itertuples(
                index=False
            ):

                activity_file = (
                    row.activity_file
                )

                key = (
                    row.athlete_id,
                    row.activity_file
                )

                destination = (
                    GRANULAR_DIR
                    / athlete_id
                    / activity_file
                )

                # --------------------------------------------
                # Already extracted
                # --------------------------------------------

                if (
                    destination.exists()
                    and
                    destination.stat().st_size > 0
                ):

                    tracking[key] = {
                        "athlete_id":
                            athlete_id,

                        "activity_file":
                            activity_file,

                        "zip_status":
                            "MATCHED",

                        "download_status":
                            download_status,

                        "extraction_status":
                            "ALREADY_PRESENT",

                        "local_path":
                            str(destination),

                        "zip_size_bytes":
                            zip_size,

                        "activity_size_bytes":
                            destination.stat().st_size,

                        "timestamp":
                            timestamp(),

                        "error":
                            ""
                    }

                    skipped_count += 1

                    continue

                # --------------------------------------------
                # Find member
                # --------------------------------------------

                member, status = (
                    find_zip_member(
                        archive,
                        activity_file
                    )
                )

                if status != "FOUND":

                    error = (
                        f"Activity file "
                        f"{activity_file} "
                        f"was {status} inside ZIP."
                    )

                    print(
                        f"      {error}"
                    )

                    tracking[key] = {
                        "athlete_id":
                            athlete_id,

                        "activity_file":
                            activity_file,

                        "zip_status":
                            "MATCHED",

                        "download_status":
                            download_status,

                        "extraction_status":
                            status,

                        "local_path":
                            "",

                        "zip_size_bytes":
                            zip_size,

                        "activity_size_bytes":
                            "",

                        "timestamp":
                            timestamp(),

                        "error":
                            error
                    }

                    continue

                # --------------------------------------------
                # Extract
                # --------------------------------------------

                try:

                    activity_size = (
                        extract_activity(
                            archive,
                            member,
                            destination
                        )
                    )

                    tracking[key] = {
                        "athlete_id":
                            athlete_id,

                        "activity_file":
                            activity_file,

                        "zip_status":
                            "MATCHED",

                        "download_status":
                            download_status,

                        "extraction_status":
                            "EXTRACTED",

                        "local_path":
                            str(destination),

                        "zip_size_bytes":
                            zip_size,

                        "activity_size_bytes":
                            activity_size,

                        "timestamp":
                            timestamp(),

                        "error":
                            ""
                    }

                    extracted_count += 1

                except Exception as exc:

                    error = str(exc)

                    print(
                        f"      EXTRACTION ERROR: "
                        f"{error}"
                    )

                    tracking[key] = {
                        "athlete_id":
                            athlete_id,

                        "activity_file":
                            activity_file,

                        "zip_status":
                            "MATCHED",

                        "download_status":
                            download_status,

                        "extraction_status":
                            "FAILED",

                        "local_path":
                            "",

                        "zip_size_bytes":
                            zip_size,

                        "activity_size_bytes":
                            "",

                        "timestamp":
                            timestamp(),

                        "error":
                            error
                    }

    except zipfile.BadZipFile:

        error = (
            "Downloaded file is not a valid ZIP."
        )

        print(
            f"  ERROR: {error}"
        )

        for row in athlete_rows.itertuples(
            index=False
        ):

            key = (
                row.athlete_id,
                row.activity_file
            )

            tracking[key] = {
                "athlete_id":
                    row.athlete_id,

                "activity_file":
                    row.activity_file,

                "zip_status":
                    "MATCHED",

                "download_status":
                    download_status,

                "extraction_status":
                    "FAILED",

                "local_path":
                    "",

                "zip_size_bytes":
                    zip_size,

                "activity_size_bytes":
                    "",

                "timestamp":
                    timestamp(),

                "error":
                    error
            }

        return 0, 0

    except Exception as exc:

        error = str(exc)

        print(
            f"  ERROR opening ZIP: "
            f"{error}"
        )

        for row in athlete_rows.itertuples(
            index=False
        ):

            key = (
                row.athlete_id,
                row.activity_file
            )

            tracking[key] = {
                "athlete_id":
                    row.athlete_id,

                "activity_file":
                    row.activity_file,

                "zip_status":
                    "MATCHED",

                "download_status":
                    download_status,

                "extraction_status":
                    "FAILED",

                "local_path":
                    "",

                "zip_size_bytes":
                    zip_size,

                "activity_size_bytes":
                    "",

                "timestamp":
                    timestamp(),

                "error":
                    error
            }

        return 0, 0

    # --------------------------------------------------------
    # Delete ZIP if configured
    # --------------------------------------------------------

    if not KEEP_ZIPS:

        try:

            zip_path.unlink()

            print(
                "  Temporary ZIP deleted."
            )

        except Exception as exc:

            print(
                f"  WARNING: Could not delete ZIP: "
                f"{exc}"
            )

    else:

        print(
            f"  ZIP retained: {zip_path}"
        )

    return (
        extracted_count,
        skipped_count
    )


# ============================================================
# 13. WRITE FINAL SUMMARY
# ============================================================

def write_summary(
    audit,
    tracking
):

    total_pairs = len(
        audit
    )

    matched_pairs = (
        audit["zip_status"]
        == "MATCHED"
    ).sum()

    missing_pairs = (
        audit["zip_status"]
        == "MISSING_ZIP"
    ).sum()

    extracted = 0
    already_present = 0
    failed = 0
    not_attempted = 0

    for row in tracking.values():

        status = row.get(
            "extraction_status",
            ""
        )

        if status == "EXTRACTED":

            extracted += 1

        elif status == "ALREADY_PRESENT":

            already_present += 1

        elif status in (
            "FAILED",
            "NOT_FOUND",
            "MULTIPLE_MATCHES"
        ):

            failed += 1

        elif status == "NOT_ATTEMPTED":

            not_attempted += 1

    summary = [
        "CAPSTONE OSF DATASET DOWNLOAD SUMMARY",
        "=" * 70,
        "",
        f"Total manifest activity pairs: {total_pairs:,}",
        f"Pairs with matched ZIP: {matched_pairs:,}",
        f"Pairs with missing ZIP: {missing_pairs:,}",
        "",
        "EXTRACTION",
        f"Successfully extracted: {extracted:,}",
        f"Already present / skipped: {already_present:,}",
        f"Failed extraction/download: {failed:,}",
        f"Not attempted: {not_attempted:,}",
        "",
        f"Granular output directory:",
        f"{GRANULAR_DIR}",
        "",
        f"Detailed tracking:",
        f"{DOWNLOAD_AUDIT_FILE}",
        "",
        "NOTE:",
        "Missing ZIPs are retained in the audit and are NOT silently discarded.",
        "They will need to be investigated separately before final dataset preparation."
    ]

    text = "\n".join(
        summary
    )

    DOWNLOAD_SUMMARY_FILE.write_text(
        text,
        encoding="utf-8"
    )

    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


# ============================================================
# 14. MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("CAPSTONE - SCRIPT 02")
    print("OSF DATASET DOWNLOADER")
    print("=" * 70)

    print()
    print(
        f"Project root:"
    )

    print(
        ROOT
    )

    print()
    print(
        f"Granular output:"
    )

    print(
        GRANULAR_DIR
    )

    print()
    print(
        f"Keep ZIP files: "
        f"{KEEP_ZIPS}"
    )

    # --------------------------------------------------------
    # Load audit
    # --------------------------------------------------------

    audit = load_audit()

    # --------------------------------------------------------
    # Only matched ZIPs can be downloaded.
    # --------------------------------------------------------

    matched = audit[
        audit["zip_status"]
        == "MATCHED"
    ].copy()

    missing = audit[
        audit["zip_status"]
        != "MATCHED"
    ].copy()

    print()
    print(
        f"Activity pairs eligible for download: "
        f"{len(matched):,}"
    )

    print(
        f"Activity pairs currently unavailable: "
        f"{len(missing):,}"
    )

    if len(matched) == 0:

        print()
        print(
            "ERROR: No matched ZIPs available."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Existing tracking
    # --------------------------------------------------------

    tracking = (
        load_existing_tracking()
    )

    print()
    print(
        f"Existing tracking records: "
        f"{len(tracking):,}"
    )

    # --------------------------------------------------------
    # Group by athlete
    #
    # THIS IS IMPORTANT:
    #
    # We download one ZIP per athlete, not one ZIP
    # for every activity.
    # --------------------------------------------------------

    athlete_groups = (
        matched
        .groupby(
            "athlete_id",
            sort=True
        )
    )

    total_athletes = (
        matched["athlete_id"]
        .nunique()
    )

    print()
    print(
        f"Unique athlete ZIPs to process: "
        f"{total_athletes:,}"
    )

    print()
    print(
        "Starting download..."
    )

    print()

    processed_athletes = 0
    total_extracted = 0
    total_skipped = 0

    # --------------------------------------------------------
    # Process athlete by athlete
    # --------------------------------------------------------

    for athlete_id, athlete_rows in athlete_groups:

        processed_athletes += 1

        print()
        print(
            f"[{processed_athletes:,}/"
            f"{total_athletes:,}]"
        )

        extracted, skipped = (
            process_athlete(
                athlete_id,
                athlete_rows,
                tracking
            )
        )

        total_extracted += extracted
        total_skipped += skipped

        # Save tracking after EVERY athlete.
        #
        # This makes the process resumable if:
        # - internet drops
        # - PC crashes
        # - OSF throttles
        # - Python stops
        #
        save_tracking(
            tracking
        )

        if (
            processed_athletes
            % PROGRESS_EVERY
            == 0
        ):

            print()
            print(
                "PROGRESS"
            )

            print(
                f"  Athletes processed: "
                f"{processed_athletes:,}/"
                f"{total_athletes:,}"
            )

            print(
                f"  Extracted this run: "
                f"{total_extracted:,}"
            )

            print(
                f"  Already present: "
                f"{total_skipped:,}"
            )

            print(
                f"  Tracking records: "
                f"{len(tracking):,}"
            )

    # --------------------------------------------------------
    # Final save
    # --------------------------------------------------------

    save_tracking(
        tracking
    )

    write_summary(
        audit,
        tracking
    )

    print()
    print(
        "DOWNLOAD PROCESS COMPLETE."
    )

    print()
    print(
        "Next step is NOT data preparation yet."
    )

    print(
        "First inspect download_summary.txt and "
        "download_audit.csv."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()