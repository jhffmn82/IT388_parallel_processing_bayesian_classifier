# -*- coding: utf-8 -*-
"""
diabetes_csv.py

Downloads the UCI CDC Diabetes Health Indicators dataset and converts it
into integer-coded CSV files for use with our C Naive Bayes classifier.

Outputs (7 files total):
  diabetes_meta.csv
  diabetes_20000_labeled.csv   / diabetes_20000_unlabeled.csv
  diabetes_100000_labeled.csv  / diabetes_100000_unlabeled.csv
  diabetes_500000_labeled.csv  / diabetes_500000_unlabeled.csv

Quick notes on the format:
  - Every value in every data file is an integer — no strings.
  - The metadata file lists allowed values for each column.
  - The last column is always the target/class label.
  - Labeled files include the target; unlabeled files don't.
  - The 500k files are larger than the dataset so they use sampling
    with replacement. The 20k and 100k files sample without replacement.

Preprocessing decisions:
  Target:     0 = healthy, 1 = pre-diabetic or diabetic
  BMI:        0=underweight, 1=healthy, 2=overweight, 3=obese, 4=morbidly obese
  MentHlth:   0=none (0 days), 1=low (1-5), 2=moderate (6-15), 3=high (16-30)
  PhysHlth:   same binning as MentHlth

@author: jhffm
Created: Sat Mar 21 21:33:41 2026
"""

from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

OUTPUT_DIR = Path("diabetes_output")
RANDOM_SEED = 42
DATASET_SIZES = [20_000, 100_000, 500_000]


def normalize_target(y: pd.DataFrame) -> pd.Series:
    """Convert the diabetes target column to binary (0 = healthy, 1 = diabetic/pre)."""
    if y.shape[1] != 1:
        raise ValueError(f"Expected exactly 1 target column, found {y.shape[1]}")

    target_name = y.columns[0]
    s = pd.to_numeric(y[target_name], errors="raise").astype(int)
    unique_vals = set(s.dropna().unique().tolist())

    if unique_vals.issubset({0, 1}):
        return s.rename("Target")

    # 3-class version of the dataset — collapse 1 and 2 into the same category
    if unique_vals.issubset({0, 1, 2}):
        return s.map(lambda v: 0 if v == 0 else 1).astype(int).rename("Target")

    raise ValueError(f"Unexpected target values: {sorted(unique_vals)}")


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


def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply binning transforms to BMI, MentHlth, and PhysHlth. All other columns unchanged."""
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
    Fetch the UCI dataset, clean it, and return a ready-to-use DataFrame.
    Target column is last, everything is integer-coded, no missing values.
    """
    ds = fetch_ucirepo(id=891)

    X = ds.data.features.copy()
    y = ds.data.targets.copy()

    target = normalize_target(y)
    df = pd.concat([X, target], axis=1)

    # Drop any rows with missing values — the C side doesn't handle them
    df = df.dropna(axis=0).reset_index(drop=True)

    # Make sure everything is an integer before binning
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)

    feature_cols = [c for c in df.columns if c != "Target"]
    transformed_features = transform_features(df[feature_cols])

    return pd.concat([transformed_features, df["Target"]], axis=1)


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

    # Stack the allowed values vertically — blank cells where a column runs out
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


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Sample n rows while keeping roughly the same class balance as the original.
    Uses replacement when n is larger than the dataset (i.e. the 500k case).
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

        # Divide proportionally; give the remainder to the last class
        if i < len(classes) - 1:
            cls_n = round(n * proportion)
            assigned += cls_n
        else:
            cls_n = n - assigned

        part = cls_df.sample(n=cls_n, replace=replace, random_state=seed + int(cls))
        sampled_parts.append(part)

    sampled = pd.concat(sampled_parts, axis=0)
    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)


def write_dataset_pair(df: pd.DataFrame, size: int, output_dir: Path) -> None:
    """Write one labeled and one unlabeled CSV for the given row count."""
    sampled = stratified_sample(df, size, RANDOM_SEED + size)

    labeled_path = output_dir / f"diabetes_{size}_labeled.csv"
    unlabeled_path = output_dir / f"diabetes_{size}_unlabeled.csv"

    sampled.to_csv(labeled_path, index=False)
    sampled.drop(columns=["Target"]).to_csv(unlabeled_path, index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching UCI diabetes dataset...")
    df = load_dataset()
    validate_against_meta(df)

    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns (including target).")
    print("Binning applied:")
    print("  BMI       -> 0=underweight, 1=healthy, 2=overweight, 3=obese, 4=morbidly_obese")
    print("  MentHlth  -> 0=none, 1=low, 2=moderate, 3=high")
    print("  PhysHlth  -> 0=none, 1=low, 2=moderate, 3=high")
    print("  Target    -> 0=healthy, 1=pre-diabetic_or_diabetic")

    meta_path = OUTPUT_DIR / "diabetes_meta.csv"
    build_meta_file(df, meta_path)
    print(f"Wrote metadata: {meta_path}")

    for size in DATASET_SIZES:
        if size > len(df):
            print(f"Creating {size:,}-row files with replacement (dataset only has {len(df):,} rows).")
        else:
            print(f"Creating {size:,}-row files without replacement.")
        write_dataset_pair(df, size, OUTPUT_DIR)

    print("\nGenerated files:")
    for path in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f"  {path}")


if __name__ == "__main__":
    main()