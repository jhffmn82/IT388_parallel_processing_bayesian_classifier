# Parallel Naive Bayes - Preprocessing Data

This folder contains the preprocessed datasets and metadata files for our C-based Naive Bayes classifier. Everything is converted to integer-coded categories so the C program doesn't have to deal with string parsing.

## Data Sources

Diabetes: UCI Machine Learning Repository - CDC Diabetes Health Indicators
https://archive.ics.uci.edu/dataset/891/cdc-diabetes-health-indicators

Heart Disease: Kaggle - Heart Disease Health Indicators (Alex Teboul)
https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset

Both datasets come from the CDC BRFSS survey.

## Files

diabetes_output/ - diabetes labeled/unlabeled CSVs and meta file
heart_output/ - heart disease labeled/unlabeled CSVs and meta file
diabetes_csv.py - downloads and preprocesses the diabetes dataset
heart_csv.py - preprocesses the heart dataset from the provided CSV
heart_disease_health_indicators_BRFSS2015.csv - raw heart dataset input

## Meta File Format

Each dataset has a *_meta.csv file that the C program reads first to understand the shape of the data.

Row 1: column names
Row 2: number of allowed values per column
Row 3+: allowed values listed vertically under each column

The last column is always the target. All values are integers.

## Preprocessing

All values are converted to integer-coded categories.

Target: 0 = negative (no disease), 1 = positive (disease)

BMI: 0=underweight, 1=healthy, 2=overweight, 3=obese, 4=morbidly obese

MentHlth / PhysHlth: 0=none (0 days), 1=low (1-5), 2=moderate (6-15), 3=high (16-30)

Diabetes (heart dataset only): 0=not diabetic, 1=pre-diabetic or diabetic

All other features keep their original integer values from the source dataset.

## Dataset Sizes

The diabetes dataset comes in three sizes: 20k, 100k, and 500k rows. The 20k and 100k are sampled without replacement. The 500k is sampled with replacement since the original dataset only has ~253k rows. The heart dataset uses the full dataset as-is.