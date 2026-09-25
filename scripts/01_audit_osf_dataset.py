from pathlib import Path
from collections import defaultdict
from urllib.parse import unquote
import html
import os
import sys
import time

import pandas as pd
import requests


# ============================================================
# CAPSTONE
# SCRIPT 01 - OSF DATASET AUDIT
#
# PURPOSE:
#   - Read the filtered run_activities_manifest.csv
#   - identify required (athlete id, activity file) pairs
#   - Scan the OSF project for athlete ZIP files
#   - Match each athlete id to its corresponding ZIP
#   - Produce an audit CSV
#
# IMPORTANT:
#   THIS SCRIPT DOES NOT DOWNLOAD ANY DATA.
# ============================================================


# ------------------------------------------------------------
# 1. LOCAL PATHS
# ------------------------------------------------------------

ROOT = Path(
    r"c:\Users\admin\Desktop\CAPSTONE GAME TIME"
)

# Change this if your manifest is somewhere else.
MANIFEST = (
    ROOT
    / "manifest file"
    / "run_activities_manifest.csv"
)

LOG_DIR = ROOT / "logs"

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

AUDIT_FILE = (
    LOG_DIR
    / "osf_dataset_audit.csv"
)

SUMMARY_FILE = (
    LOG_DIR
    / "osf_dataset_audit_summary.txt"
)


# ------------------------------------------------------------
# 2. OSF PROJECT
# ------------------------------------------------------------

PROJECT_id = "6hfpz"

PROVidER = "osfstorage"

OSF_API_ROOT = (
    f"https://api.osf.io/v2/"
    f"nodes/{PROJECT_id}/"
    f"files/{PROVidER}/"
)


# ------------------------------------------------------------
# 3. HTTP SESSION
# ------------------------------------------------------------

session = requests.Session()

session.headers.update(
    {
        "User-Agent":
            "CAPSTONE-OSF-Dataset-Audit/1.0"
    }
)

# Public OSF project should not require a token.
#
# If OSF ever requires authentication, you can set:
#
#   $env:OSF_TOKEN="your_token_here"
#
# We do NOT hard-code tokens into this script.

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


# ------------------------------------------------------------
# 4. NORMALIZATION
# ------------------------------------------------------------

def normalize(value):
    """
    Clean values from the manifest without changing
    their substantive content.
    """

    if pd.isna(value):
        return ""

    value = html.unescape(
        str(value)
    )

    return value.strip()


# ------------------------------------------------------------
# 5. LOAD MANIFEST
# ------------------------------------------------------------

def load_manifest():

    print()
    print("=" * 70)
    print("LOADING MANIFEST")
    print("=" * 70)

    if not MANIFEST.exists():

        print()
        print("ERROR: Manifest not found.")
        print()
        print(MANIFEST)
        print()

        print(
            "Change the MANIFEST path near the top "
            "of this script to your actual CSV location."
        )

        sys.exit(1)

    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1252"
    ]

    df = None

    for encoding in encodings:

        try:

            df = pd.read_csv(
                MANIFEST,
                encoding=encoding,
                low_memory=False
            )

            break

        except UnicodeDecodeError:

            continue

    if df is None:

        raise RuntimeError(
            "Could not decode the manifest CSV."
        )

    print()
    print(
        f"Manifest loaded: {MANIFEST}"
    )

    print()
    print("Columns found:")

    for column in df.columns:

        print(
            f"  - {column}"
        )

    # Required columns
    required_columns = {
        "id",
        "file"
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        print()
        print(
            "ERROR: Required column(s) missing:"
        )

        for column in missing_columns:

            print(
                f"  - {column}"
            )

        sys.exit(1)

    # Normalize
    df["id"] = (
        df["id"]
        .map(normalize)
    )

    df["file"] = (
        df["file"]
        .map(normalize)
    )

    # Remove rows without id/file
    before = len(df)

    df = df[
        (df["id"] != "")
        &
        (df["file"] != "")
    ].copy()

    removed = (
        before - len(df)
    )

    if removed > 0:

        print()
        print(
            f"WARNING: Removed {removed:,} "
            "rows with missing id/file."
        )

    return df


# ------------------------------------------------------------
# 6. REQUEST JSON FROM OSF
# ------------------------------------------------------------

def get_json(
    url,
    retries=4
):

    last_error = None

    for attempt in range(
        1,
        retries + 1
    ):

        try:

            response = session.get(
                url,
                timeout=60
            )

            response.raise_for_status()

            return response.json()

        except requests.RequestException as exc:

            last_error = exc

            print()
            print(
                f"OSF request failed "
                f"(attempt {attempt}/{retries})"
            )

            print(
                f"URL: {url}"
            )

            if attempt < retries:

                wait = 2 ** (
                    attempt - 1
                )

                print(
                    f"Retrying in {wait} seconds..."
                )

                time.sleep(wait)

    raise RuntimeError(
        f"OSF request failed after "
        f"{retries} attempts:\n"
        f"{last_error}"
    )


# ------------------------------------------------------------
# 7. GET CHILDREN URL FOR AN OSF FOLDER
# ------------------------------------------------------------

def get_folder_children_url(
    item
):

    relationships = (
        item.get(
            "relationships",
            {}
        )
    )

    files_relationship = (
        relationships.get(
            "files",
            {}
        )
    )

    links = (
        files_relationship.get(
            "links",
            {}
        )
    )

    related = (
        links.get(
            "related",
            {}
        )
    )

    return related.get(
        "href"
    )


# ------------------------------------------------------------
# 8. SCAN OSF
# ------------------------------------------------------------

def scan_osf_storage():

    print()
    print("=" * 70)
    print("SCANNING OSF STORAGE")
    print("=" * 70)

    print()
    print(
        "OSF API root:"
    )

    print(
        OSF_API_ROOT
    )

    print()
    print(
        "No files will be downloaded."
    )

    zip_map = defaultdict(list)

    visited_urls = set()

    pages_scanned = 0

    def walk(url):

        nonlocal pages_scanned

        if not url:
            return

        if url in visited_urls:
            return

        visited_urls.add(url)

        data = get_json(
            url
        )

        pages_scanned += 1

        if pages_scanned % 10 == 0:

            print(
                f"  OSF pages scanned: "
                f"{pages_scanned:,}"
            )

        items = data.get(
            "data",
            []
        )

        for item in items:

            attributes = (
                item.get(
                    "attributes",
                    {}
                )
            )

            kind = attributes.get(
                "kind"
            )

            name = attributes.get(
                "name",
                ""
            )

            materialized_path = (
                attributes.get(
                    "materialized_path",
                    ""
                )
            )

            # --------------------------------------------
            # FILE
            # --------------------------------------------

            if kind == "file":

                candidate_name = (
                    unquote(
                        name
                        or
                        Path(
                            materialized_path
                        ).name
                    )
                )

                if candidate_name.lower().endswith(
                    ".zip"
                ):

                    athlete_id = (
                        Path(
                            candidate_name
                        )
                        .stem
                        .strip()
                        .lower()
                    )

                    zip_map[
                        athlete_id
                    ].append(
                        {
                            "name":
                                candidate_name,

                            "path":
                                materialized_path,

                            "download_url":
                                item
                                .get(
                                    "links",
                                    {}
                                )
                                .get(
                                    "download",
                                    ""
                                ),

                            "api_url":
                                item
                                .get(
                                    "links",
                                    {}
                                )
                                .get(
                                    "self",
                                    ""
                                ),
                        }
                    )

            # --------------------------------------------
            # FOLDER
            # --------------------------------------------

            elif kind == "folder":

                child_url = (
                    get_folder_children_url(
                        item
                    )
                )

                if child_url:

                    walk(
                        child_url
                    )

        # --------------------------------------------
        # NEXT PAGE
        # --------------------------------------------

        next_url = (
            data
            .get(
                "links",
                {}
            )
            .get(
                "next"
            )
        )

        if next_url:

            walk(
                next_url
            )

    walk(
        OSF_API_ROOT
    )

    total_zip_records = sum(
        len(values)
        for values in zip_map.values()
    )

    print()
    print(
        f"OSF pages scanned: "
        f"{pages_scanned:,}"
    )

    print(
        f"ZIP records discovered: "
        f"{total_zip_records:,}"
    )

    print(
        f"Unique ZIP names / athlete ids: "
        f"{len(zip_map):,}"
    )

    return zip_map


# ------------------------------------------------------------
# 9. BUILD AUDIT
# ------------------------------------------------------------

def build_audit(
    manifest,
    zip_map
):

    print()
    print("=" * 70)
    print("MATCHING MANIFEST TO OSF ZIP FILES")
    print("=" * 70)

    # We care about unique id + file pairs.
    pairs = (
        manifest[
            [
                "id",
                "file"
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    audit_rows = []

    for row in pairs.itertuples(
        index=False
    ):

        athlete_id = row.id

        activity_file = row.file

        candidates = zip_map.get(
            athlete_id.lower(),
            []
        )

        # --------------------------------------------
        # NO ZIP
        # --------------------------------------------

        if len(candidates) == 0:

            zip_status = (
                "MISSING_ZIP"
            )

            zip_name = ""
            zip_path = ""
            download_url = ""
            api_url = ""

        # --------------------------------------------
        # EXACTLY ONE ZIP
        # --------------------------------------------

        elif len(candidates) == 1:

            zip_status = (
                "MATCHED"
            )

            zip_name = (
                candidates[0]["name"]
            )

            zip_path = (
                candidates[0]["path"]
            )

            download_url = (
                candidates[0]["download_url"]
            )

            api_url = (
                candidates[0]["api_url"]
            )

        # --------------------------------------------
        # MULTIPLE ZIPS
        # --------------------------------------------

        else:

            zip_status = (
                "AMBIGUOUS_MULTIPLE_ZIPS"
            )

            zip_name = " | ".join(
                item["name"]
                for item in candidates
            )

            zip_path = " | ".join(
                item["path"]
                for item in candidates
            )

            download_url = " | ".join(
                item["download_url"]
                for item in candidates
            )

            api_url = " | ".join(
                item["api_url"]
                for item in candidates
            )

        audit_rows.append(
            {
                "athlete_id":
                    athlete_id,

                "activity_file":
                    activity_file,

                "zip_status":
                    zip_status,

                "osf_zip_name":
                    zip_name,

                "osf_zip_path":
                    zip_path,

                "osf_download_url":
                    download_url,

                "osf_api_url":
                    api_url,

                # We haven't opened the ZIP yet.
                "activity_status":
                    "NOT_CHECKED_YET",
            }
        )

    audit_df = pd.DataFrame(
        audit_rows
    )

    audit_df = (
        audit_df
        .sort_values(
            [
                "athlete_id",
                "activity_file"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    audit_df.to_csv(
        AUDIT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    return audit_df


# ------------------------------------------------------------
# 10. SUMMARY
# ------------------------------------------------------------

def write_summary(
    manifest,
    audit,
    zip_map
):

    total_rows = len(
        manifest
    )

    unique_pairs = len(
        manifest[
            [
                "id",
                "file"
            ]
        ]
        .drop_duplicates()
    )

    duplicate_rows = (
        total_rows
        - unique_pairs
    )

    unique_athletes = (
        manifest["id"]
        .nunique()
    )

    unique_files = (
        manifest["file"]
        .nunique()
    )

    matched = (
        audit["zip_status"]
        == "MATCHED"
    ).sum()

    missing = (
        audit["zip_status"]
        == "MISSING_ZIP"
    ).sum()

    ambiguous = (
        audit["zip_status"]
        == "AMBIGUOUS_MULTIPLE_ZIPS"
    ).sum()

    matched_athletes = (
        audit.loc[
            audit["zip_status"]
            == "MATCHED",
            "athlete_id"
        ]
        .nunique()
    )

    summary = []

    summary.append(
        "CAPSTONE OSF DATASET AUDIT"
    )

    summary.append(
        "=" * 70
    )

    summary.append(
        f"Manifest: {MANIFEST}"
    )

    summary.append(
        f"OSF Project: {PROJECT_id}"
    )

    summary.append(
        f"Provider: {PROVidER}"
    )

    summary.append("")

    summary.append(
        "MANIFEST"
    )

    summary.append(
        f"Total manifest rows: "
        f"{total_rows:,}"
    )

    summary.append(
        f"Unique (id, file) pairs: "
        f"{unique_pairs:,}"
    )

    summary.append(
        f"Duplicate manifest rows: "
        f"{duplicate_rows:,}"
    )

    summary.append(
        f"Unique athlete ids: "
        f"{unique_athletes:,}"
    )

    summary.append(
        f"Unique activity filenames: "
        f"{unique_files:,}"
    )

    summary.append("")

    summary.append(
        "OSF ZIP MATCHING"
    )

    summary.append(
        f"ZIP records discovered: "
        f"{sum(len(v) for v in zip_map.values()):,}"
    )

    summary.append(
        f"Required pairs with MATCHED ZIP: "
        f"{matched:,}"
    )

    summary.append(
        f"Required pairs with MISSING ZIP: "
        f"{missing:,}"
    )

    summary.append(
        f"Required pairs with AMBIGUOUS ZIP: "
        f"{ambiguous:,}"
    )

    summary.append(
        f"Unique athletes with MATCHED ZIP: "
        f"{matched_athletes:,}"
    )

    summary.append("")

    summary.append(
        "DOWNLOAD STATUS"
    )

    summary.append(
        "NO ZIP FILES WERE DOWNLOADED."
    )

    summary.append(
        "NO ACTIVITY CSV FILES WERE EXTRACTED."
    )

    summary.append("")

    summary.append(
        "NEXT STEP:"
    )

    summary.append(
        "Review osf_dataset_audit.csv before "
        "starting the downloader."
    )

    summary_text = "\n".join(
        summary
    )

    SUMMARY_FILE.write_text(
        summary_text,
        encoding="utf-8"
    )

    print()
    print("=" * 70)
    print(summary_text)
    print("=" * 70)


# ------------------------------------------------------------
# 11. MAIN
# ------------------------------------------------------------

def main():

    print()
    print("=" * 70)
    print("CAPSTONE - SCRIPT 01")
    print("OSF DATASET AUDIT")
    print("=" * 70)

    manifest = (
        load_manifest()
    )

    print()
    print(
        f"Total usable manifest rows: "
        f"{len(manifest):,}"
    )

    print(
        f"Unique athlete ids: "
        f"{manifest['id'].nunique():,}"
    )

    print(
        "Unique (id, file) pairs: "
        f"{len(manifest[['id', 'file']].drop_duplicates()):,}"
    )

    zip_map = (
        scan_osf_storage()
    )

    audit = (
        build_audit(
            manifest,
            zip_map
        )
    )

    write_summary(
        manifest,
        audit,
        zip_map
    )

    print()
    print(
        "AUDIT COMPLETE."
    )

    print()
    print(
        "Detailed audit:"
    )

    print(
        AUDIT_FILE
    )

    print()
    print(
        "Summary:"
    )

    print(
        SUMMARY_FILE
    )

    print()


if __name__ == "__main__":
    main()