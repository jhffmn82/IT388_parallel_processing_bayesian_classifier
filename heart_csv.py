# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 22:01:51 2026

@author: jhffm
"""

from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------
# PURPOSE
# ---------------------------------------------------------------------
# This script reads the heart disease CSV file and transforms it into a
# simplified format for a C-based categorical Naive Bayes classifier.
#
# INPUT FILE (assumed to be in the working directory):
#   heart_disease_health_indicators_BRFSS2015.csv
#
# OUTPUT FILES (3 total):
#   1. heart_meta.csv
#   2. heart_full_labeled.csv
#   3. heart_full_unlabeled.csv
#
# IMPORTANT DESIGN RULES
# ---------------------------------------------------------------------
# 1. All output data files use INTEGER-CODED categorical values only.
#    No strings are written into the data CSV files.
#
# 2. The metadata file and both data files use the SAME coding.
#    Any value found in a data CSV must be one of the allowed values
#    listed for that column in the metadata file.
#
# 3. The LAST COLUMN is always the classifier/target column.
#    There is no separate "is_target" column in the metadata.
#
# 4. The labeled CSV file includes the target column.
#    The unlabeled CSV file omits the final target column.
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
#   HeartDiseaseorAttack is renamed to Target and moved to the last column.
#   0 = no heart disease / attack
#   1 = heart disease / attack
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
# Diabetes is reduced to binary:
#   0 = not diabetic
#   1 = pre-diabetic or diabetic
#
# All other features are kept as integer-coded categories from the CSV.
# ---------------------------------------------------------------------

INPUT_FILE = Path("heart_disease_health_indicators_BRFSS2015.csv")
META_FILE = Path("heart_meta.csv")
LABELED_FILE = Path("heart_full_labeled.csv")
UNLABELED_FILE = Path("heart_full_unlabeled.csv")


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


def bin_diabetes(value: int) -> int:
    """
    Convert Diabetes to binary integer-coded categories.

    Input values in this dataset:
      0 = not diabetic
      1 = pre-diabetic
      2 = diabetic

    Output coding:
      0 = not diabetic
      1 = pre-diabetic or diabetic
    """
    value = int(value)

    if value == 0:
        return 0
    if value in (1, 2):
        return 1

    raise ValueError(f"Unexpected Diabetes value: {value}")


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

    if "Diabetes" in out.columns:
        out["Diabetes"] = out["Diabetes"].apply(bin_diabetes).astype(int)

    return out


def load_dataset() -> pd.DataFrame:
    """
    Load the heart dataset from the working directory, clean it,
    transform selected columns, rename the classifier to Target,
    and place the target in the last column.

    Final output requirements:
    - no missing values
    - all values are integers
    - target column is named 'Target'
    - target column is last
    """
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find input file: {INPUT_FILE.resolve()}"
        )

    df = pd.read_csv(INPUT_FILE)

    # Drop incomplete rows so the C program does not need to handle missing data
    df = df.dropna(axis=0).reset_index(drop=True)

    # Force all raw values to integer form before transformations
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)

    target_col = "HeartDiseaseorAttack"
    if target_col not in df.columns:
        raise ValueError(f"Expected target column '{target_col}' not found.")

    # Separate target, transform features, then place target last
    target = df[target_col].astype(int).rename("Target")
    feature_df = df.drop(columns=[target_col])
    feature_df = transform_features(feature_df)

    final_df = pd.concat([feature_df, target], axis=1)
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
    - Every value appearing in the labeled/unlabeled CSV must be listed
      under the corresponding column in this file
    """
    headers = list(df.columns)

    allowed_values = {}
    for col in headers:
        allowed_values[col] = sorted(df[col].dropna().astype(int).unique().tolist())

    counts = [len(allowed_values[col]) for col in headers]
    max_count = max(counts)

    rows = [headers, counts]

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
    Sanity check: ensure every column is integer-coded and non-empty
    after preprocessing.
    """
    for col in df.columns:
        values = df[col].dropna().astype(int)
        unique_vals = sorted(values.unique().tolist())

        if len(unique_vals) == 0:
            raise ValueError(f"Column {col} has no values after preprocessing.")

        if not pd.api.types.is_integer_dtype(values.dtype):
            raise ValueError(f"Column {col} is not integer-coded.")


def write_output_files(df: pd.DataFrame) -> None:
    """
    Write one labeled CSV and one unlabeled CSV.

    Labeled file:
      - includes all columns, with Target last

    Unlabeled file:
      - same structure, but omits the final Target column

    Both files use the same integer coding defined in the metadata file.
    """
    df.to_csv(LABELED_FILE, index=False)
    df.drop(columns=["Target"]).to_csv(UNLABELED_FILE, index=False)


def main() -> None:
    """
    Main pipeline:
    1. Load and clean the heart disease CSV
    2. Transform selected high-cardinality columns into lower-cardinality categories
    3. Validate that all output is integer-coded
    4. Write the metadata file
    5. Write labeled and unlabeled CSV files
    """
    print(f"Reading input file: {INPUT_FILE}")
    df = load_dataset()
    validate_against_meta(df)

    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns including target.")
    print("Applied categorical binning:")
    print(" - BMI: 0=underweight, 1=healthy, 2=overweight, 3=obese, 4=morbidly_obese")
    print(" - MentHlth: 0=none, 1=low, 2=moderate, 3=high")
    print(" - PhysHlth: 0=none, 1=low, 2=moderate, 3=high")
    print(" - Diabetes: 0=not_diabetic, 1=pre_diabetic_or_diabetic")
    print(" - Target: 0=no_heart_disease_or_attack, 1=heart_disease_or_attack")

    build_meta_file(df, META_FILE)
    print(f"Wrote metadata file: {META_FILE}")

    write_output_files(df)
    print(f"Wrote labeled data file: {LABELED_FILE}")
    print(f"Wrote unlabeled data file: {UNLABELED_FILE}")

    print("\nGenerated files:")
    print(f" - {META_FILE}")
    print(f" - {LABELED_FILE}")
    print(f" - {UNLABELED_FILE}")


if __name__ == "__main__":
    main()