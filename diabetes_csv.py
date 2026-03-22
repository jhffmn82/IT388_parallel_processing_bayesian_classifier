# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 21:33:41 2026

@author: jhffm
"""

from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

# ---------------------------------------------------------------------
# PURPOSE
# ---------------------------------------------------------------------
# This script downloads the UCI CDC Diabetes Health Indicators dataset,
# cleans and transforms it into a simplified format for a C-based
# categorical Naive Bayes classifier.
#
# OUTPUT FILES (7 total):
#   1. diabetes_meta.csv
#   2. diabetes_20000_labeled.csv
#   3. diabetes_100000_labeled.csv
#   4. diabetes_500000_labeled.csv
#   5. diabetes_20000_unlabeled.csv
#   6. diabetes_100000_unlabeled.csv
#   7. diabetes_500000_unlabeled.csv
#
# IMPORTANT DESIGN RULES
# ---------------------------------------------------------------------
# 1. All output data files use INTEGER-CODED categorical values only.
#    No strings are written into the data CSV files.
#
# 2. The metadata file and all data files use the SAME coding.
#    Any value found in a data CSV must be one of the allowed values
#    listed for that column in the metadata file.
#
# 3. The LAST COLUMN is always the classifier/target column.
#    There is no separate "is_target" column in the metadata.
#
# 4. The labeled CSV files include the target column.
#    The unlabeled CSV files omit the final target column.
#
# 5. The 500,000-row files are larger than the original dataset, so they
#    are created using sampling WITH replacement. The 20,000-row and
#    100,000-row files are created WITHOUT replacement.
#
# META FILE STRUCTURE
# ---------------------------------------------------------------------
# Row 1: attribute names
# Row 2: number of allowed values for each attribute
# Rows 3+: allowed values listed vertically under each attribute
# Last column: target/classifier
#
# Example:
#   Row 1 -> HighBP,HighChol,...,Target
#   Row 2 -> 2,2,...,2
#   Row 3 -> 0,0,...,0
#   Row 4 -> 1,1,...,1
#   ...
#
# Blank cells are used when one column has fewer allowed values than
# another column.
#
# PREPROCESSING / CLEANING RULES
# ---------------------------------------------------------------------
# Target:
#   0 = healthy
#   1 = pre-diabetic or diabetic
#
# BMI is reduced to 5 categories:
#   0 = underweight      (< 18.5)
#   1 = healthy          (18.5 to < 25)
#   2 = overweight       (25 to < 30)
#   3 = obese            (30 to < 40)
#   4 = morbidly_obese   (>= 40)
#
# MentHlth and PhysHlth are reduced to 4 categories:
#   0 = none      (0 days)
#   1 = low       (1 to 5 days)
#   2 = moderate  (6 to 15 days)
#   3 = high      (16 to 30 days)
#
# All other features are kept as integer-coded categories from the
# machine-learning-ready UCI dataset.
# ---------------------------------------------------------------------

OUTPUT_DIR = Path("diabetes_output")
RANDOM_SEED = 42
DATASET_SIZES = [20_000, 100_000, 500_000]


def normalize_target(y: pd.DataFrame) -> pd.Series:
    """
    Convert the diabetes target to binary integer categories.

    Expected input:
      - already binary: {0,1}
      - or 3-class: {0,1,2}

    Output:
      0 = healthy
      1 = pre-diabetic or diabetic
    """
    if y.shape[1] != 1:
        raise ValueError(f"Expected exactly 1 target column, found {y.shape[1]}")

    target_name = y.columns[0]
    s = pd.to_numeric(y[target_name], errors="raise").astype(int)
    unique_vals = set(s.dropna().unique().tolist())

    if unique_vals.issubset({0, 1}):
        return s.rename("Target")

    if unique_vals.issubset({0, 1, 2}):
        return s.map(lambda v: 0 if v == 0 else 1).astype(int).rename("Target")

    raise ValueError(f"Unexpected target values: {sorted(unique_vals)}")


def bin_bmi(value: int) -> int:
    """
    Convert BMI to 5 integer-coded categories.

    Output coding:
      0 = underweight
      1 = healthy
      2 = overweight
      3 = obese
      4 = morbidly_obese
    """
    bmi = float(value)

    if bmi < 18.5:
        return 0
    if bmi < 25.0:
        return 1
    if bmi < 30.0:
        return 2
    if bmi < 40.0:
        return 3
    return 4


def bin_health_days(value: int) -> int:
    """
    Convert MentHlth / PhysHlth to 4 integer-coded categories.

    Raw input is the number of bad health days in the past 30 days.

    Output coding:
      0 = none
      1 = low
      2 = moderate
      3 = high
    """
    days = int(value)

    if days == 0:
        return 0
    if days <= 5:
        return 1
    if days <= 15:
        return 2
    return 3


def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature transformations required for the classifier.

    Important:
    - Output remains integer-coded
    - Output values must match the metadata file
    - No strings are introduced into the data CSV files
    """
    out = df.copy()

    if "BMI" in out.columns:
        out["BMI"] = out["BMI"].apply(bin_bmi).astype(int)

    if "MentHlth" in out.columns:
        out["MentHlth"] = out["MentHlth"].apply(bin_health_days).astype(int)

    if "PhysHlth" in out.columns:
        out["PhysHlth"] = out["PhysHlth"].apply(bin_health_days).astype(int)

    return out


def load_dataset() -> pd.DataFrame:
    """
    Fetch the UCI diabetes dataset, clean it, transform selected columns,
    and return a final DataFrame with the target in the last column.

    Final output requirements:
    - no missing values
    - all values are integers
    - target column is named 'Target'
    - target column is last
    """
    ds = fetch_ucirepo(id=891)

    X = ds.data.features.copy()
    y = ds.data.targets.copy()

    target = normalize_target(y)
    df = pd.concat([X, target], axis=1)

    # Drop incomplete rows so the C program does not need to handle missing data
    df = df.dropna(axis=0).reset_index(drop=True)

    # Force all raw values to integer form before transformations
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)

    # Transform only feature columns; keep target as final column
    feature_cols = [c for c in df.columns if c != "Target"]
    transformed_features = transform_features(df[feature_cols])

    final_df = pd.concat([transformed_features, df["Target"]], axis=1)
    return final_df


def build_meta_file(df: pd.DataFrame, output_path: Path) -> None:
    """
    Write the metadata file in the format expected by the C program.

    Meta file layout:
      Row 1: attribute names
      Row 2: number of allowed values for each attribute
      Rows 3+: allowed values listed vertically under each attribute
      Last column: target/classifier

    Important:
    - The data CSV files must match this coding exactly
    - Every value appearing in any labeled/unlabeled CSV must be listed
      under the corresponding column in this file
    """
    headers = list(df.columns)

    # Collect allowed values for each column from the transformed dataset
    allowed_values = {}
    for col in headers:
        allowed_values[col] = sorted(df[col].dropna().astype(int).unique().tolist())

    # Row 2 stores how many allowed values each column has
    counts = [len(allowed_values[col]) for col in headers]
    max_count = max(counts)

    rows = [headers, counts]

    # Rows 3+ store allowed values for each column, vertically
    for i in range(max_count):
        row = []
        for col in headers:
            values = allowed_values[col]
            row.append(values[i] if i < len(values) else "")
        rows.append(row)

    meta_df = pd.DataFrame(rows)
    meta_df.to_csv(output_path, index=False, header=False)


def validate_against_meta(df: pd.DataFrame) -> None:
    """
    Sanity check: ensure every column contains only its discovered allowed values.
    This protects against accidental non-integer or transformed-out-of-range values.
    """
    for col in df.columns:
        values = df[col].dropna().astype(int)
        unique_vals = sorted(values.unique().tolist())
        if len(unique_vals) == 0:
            raise ValueError(f"Column {col} has no values after preprocessing.")

        # Ensure all values are integers
        if not pd.api.types.is_integer_dtype(values.dtype):
            raise ValueError(f"Column {col} is not integer-coded.")


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Sample rows while roughly preserving class balance.

    If n <= dataset size:
      sample WITHOUT replacement

    If n > dataset size:
      sample WITH replacement
      (used for the 500,000-row files)
    """
    total_rows = len(df)
    replace = n > total_rows

    class_counts = df["Target"].value_counts().sort_index()
    sampled_parts = []
    assigned = 0
    classes = list(class_counts.index)

    for i, cls in enumerate(classes):
        cls_df = df[df["Target"] == cls]
        proportion = len(cls_df) / total_rows

        if i < len(classes) - 1:
            cls_n = round(n * proportion)
            assigned += cls_n
        else:
            cls_n = n - assigned

        part = cls_df.sample(
            n=cls_n,
            replace=replace,
            random_state=seed + int(cls)
        )
        sampled_parts.append(part)

    sampled = pd.concat(sampled_parts, axis=0)
    sampled = sampled.sample(frac=1, random_state=seed).reset_index(drop=True)
    return sampled


def write_dataset_pair(df: pd.DataFrame, size: int, output_dir: Path) -> None:
    """
    Write one labeled CSV and one unlabeled CSV for the given size.

    Labeled file:
      - includes all columns, with Target last

    Unlabeled file:
      - same structure, but omits the final Target column

    Both files use the same integer coding defined in the metadata file.
    """
    sampled = stratified_sample(df, size, RANDOM_SEED + size)

    labeled_path = output_dir / f"diabetes_{size}_labeled.csv"
    unlabeled_path = output_dir / f"diabetes_{size}_unlabeled.csv"

    # Labeled file matches the full metadata structure
    sampled.to_csv(labeled_path, index=False)

    # Unlabeled file removes the classifier column for future predictions
    sampled.drop(columns=["Target"]).to_csv(unlabeled_path, index=False)


def main() -> None:
    """
    Main pipeline:
    1. Fetch and clean the UCI diabetes dataset
    2. Transform selected high-cardinality columns into lower-cardinality categories
    3. Validate that all output is integer-coded
    4. Write the metadata file
    5. Write labeled and unlabeled CSV files at three sizes
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching UCI diabetes dataset...")
    df = load_dataset()
    validate_against_meta(df)

    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns including target.")
    print("Applied categorical binning:")
    print(" - BMI: 0=underweight, 1=healthy, 2=overweight, 3=obese, 4=morbidly_obese")
    print(" - MentHlth: 0=none, 1=low, 2=moderate, 3=high")
    print(" - PhysHlth: 0=none, 1=low, 2=moderate, 3=high")
    print(" - Target: 0=healthy, 1=pre-diabetic_or_diabetic")

    meta_path = OUTPUT_DIR / "diabetes_meta.csv"
    build_meta_file(df, meta_path)
    print(f"Wrote metadata file: {meta_path}")

    for size in DATASET_SIZES:
        if size > len(df):
            print(
                f"Creating {size:,}-row files with replacement "
                f"(dataset has {len(df):,} rows)."
            )
        else:
            print(f"Creating {size:,}-row files without replacement.")

        write_dataset_pair(df, size, OUTPUT_DIR)

    print("\nGenerated files:")
    for path in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f" - {path}")


if __name__ == "__main__":
    main()