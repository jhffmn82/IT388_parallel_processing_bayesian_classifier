# -*- coding: utf-8 -*-
"""
heart_csv.py

Reads the heart disease CSV and converts it into integer-coded files
for use with our C Naive Bayes classifier.

Input:
  heart_disease_health_indicators_BRFSS2015.csv  (must be in working directory)

Outputs:
  heart_meta.csv
  heart_full_labeled.csv
  heart_full_unlabeled.csv

Preprocessing decisions:
  Target (HeartDiseaseorAttack): renamed to Target, moved to last column
    0 = no heart disease/attack, 1 = heart disease/attack
  BMI:      0=underweight, 1=healthy, 2=overweight, 3=obese, 4=morbidly obese
  MentHlth: 0=none (0 days), 1=low (1-5), 2=moderate (6-15), 3=high (16-30)
  PhysHlth: same binning as MentHlth
  Diabetes: 0=not diabetic, 1=pre-diabetic or diabetic

@author: jhffm
Created: Sat Mar 21 22:01:51 2026
"""

from pathlib import Path
import pandas as pd

INPUT_FILE = Path("heart_disease_health_indicators_BRFSS2015.csv")
META_FILE = Path("heart_meta.csv")
LABELED_FILE = Path("heart_full_labeled.csv")
UNLABELED_FILE = Path("heart_full_unlabeled.csv")


def bin_bmi(value: int) -> int:
    """Bin a raw BMI value into 5 categories (0=underweight ... 4=morbidly obese)."""
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
    """Bin bad-health days (0-30) into 4 categories (0=none, 1=low, 2=moderate, 3=high)."""
    days = int(value)
    if days == 0:
        return 0
    if days <= 5:
        return 1
    if days <= 15:
        return 2
    return 3


def bin_diabetes(value: int) -> int:
    """Collapse the 3-class Diabetes column into binary (0=not diabetic, 1=pre/diabetic)."""
    value = int(value)
    if value == 0:
        return 0
    if value in (1, 2):
        return 1
    raise ValueError(f"Unexpected Diabetes value: {value}")


def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply binning to BMI, MentHlth, PhysHlth, and Diabetes. Everything else unchanged."""
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
    Load the heart CSV, clean it, and return a ready-to-use DataFrame.
    Target column is renamed and moved to the end; everything is integer-coded.
    """
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Could not find input file: {INPUT_FILE.resolve()}")

    df = pd.read_csv(INPUT_FILE)

    # Drop rows with missing values — the C side doesn't handle them
    df = df.dropna(axis=0).reset_index(drop=True)

    # Cast everything to int before binning
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)

    target_col = "HeartDiseaseorAttack"
    if target_col not in df.columns:
        raise ValueError(f"Expected target column '{target_col}' not found.")

    # Pull out the target, transform features, then stick target back on at the end
    target = df[target_col].astype(int).rename("Target")
    feature_df = transform_features(df.drop(columns=[target_col]))

    return pd.concat([feature_df, target], axis=1)


def build_meta_file(df: pd.DataFrame, output_path: Path) -> None:
    """
    Write the metadata CSV that the C classifier reads at startup.

    Layout:
      Row 1: column names
      Row 2: number of allowed values per column
      Row 3+: allowed values listed vertically, one per row
    """
    headers = list(df.columns)

    allowed_values = {}
    for col in headers:
        allowed_values[col] = sorted(df[col].dropna().astype(int).unique().tolist())

    counts = [len(allowed_values[col]) for col in headers]
    max_count = max(counts)

    rows = [headers, counts]

    # Stack values vertically — blank cells where a column runs out
    for i in range(max_count):
        row = []
        for col in headers:
            values = allowed_values[col]
            row.append(values[i] if i < len(values) else "")
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_path, index=False, header=False)


def validate_against_meta(df: pd.DataFrame) -> None:
    """Quick sanity check — makes sure every column is integer-coded and non-empty."""
    for col in df.columns:
        values = df[col].dropna().astype(int)
        if len(values.unique()) == 0:
            raise ValueError(f"Column {col} has no values after preprocessing.")
        if not pd.api.types.is_integer_dtype(values.dtype):
            raise ValueError(f"Column {col} is not integer-coded.")


def write_output_files(df: pd.DataFrame) -> None:
    """Write the labeled CSV (with target) and unlabeled CSV (without target)."""
    df.to_csv(LABELED_FILE, index=False)
    df.drop(columns=["Target"]).to_csv(UNLABELED_FILE, index=False)


def main() -> None:
    print(f"Reading input file: {INPUT_FILE}")
    df = load_dataset()
    validate_against_meta(df)

    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns (including target).")
    print("Binning applied:")
    print("  BMI       -> 0=underweight, 1=healthy, 2=overweight, 3=obese, 4=morbidly_obese")
    print("  MentHlth  -> 0=none, 1=low, 2=moderate, 3=high")
    print("  PhysHlth  -> 0=none, 1=low, 2=moderate, 3=high")
    print("  Diabetes  -> 0=not_diabetic, 1=pre_diabetic_or_diabetic")
    print("  Target    -> 0=no_heart_disease_or_attack, 1=heart_disease_or_attack")

    build_meta_file(df, META_FILE)
    print(f"Wrote metadata: {META_FILE}")

    write_output_files(df)
    print(f"Wrote labeled data: {LABELED_FILE}")
    print(f"Wrote unlabeled data: {UNLABELED_FILE}")


if __name__ == "__main__":
    main()