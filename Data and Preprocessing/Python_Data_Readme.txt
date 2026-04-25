Parallel Naive Bayes Preprocessing Data

This folder contains preprocessed datasets and metadata files used for a Naive Bayes classifier implemented in C. The data has been converted into a simplified integer-coded categorical format so that it can be easily read and processed in C without additional parsing.

DATA SOURCES

Diabetes Dataset
Source: UCI Machine Learning Repository
Dataset: CDC Diabetes Health Indicators
Link: https://archive.ics.uci.edu/dataset/891/cdc-diabetes-health-indicators

Heart Disease Dataset
Source: Kaggle
Dataset: Heart Disease Health Indicators
Author: Alex Teboul
Link: https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset

Both datasets are derived from the CDC BRFSS (Behavioral Risk Factor Surveillance System) survey data.

FILES

diabetes_output/
Contains multiple versions of the diabetes dataset:
labeled datasets (include target column)
unlabeled datasets (target column removed)
meta file describing structure

heart_output/
Contains:
full labeled dataset
full unlabeled dataset
meta file describing structure

diabetes_csv.py
Script used to download and preprocess the diabetes dataset.

heart_csv.py
Script used to preprocess the heart dataset from the provided CSV file.

heart_disease_health_indicators_BRFSS2015.csv
Raw heart dataset used as input.

META FILE FORMAT
Each dataset has a corresponding meta file ( *_meta.csv ).

Structure:
Row 1: attribute (column) names
Row 2: number of allowed values for each attribute
Rows 3+: allowed values listed vertically under each column
Last column: target (classifier)

Important rules:
The number of attributes is the number of columns
The last column is always the target
All values are integer-coded categories
Every value in the data files must appear in the meta file
This format allows the C program to:
determine number of features
determine number of values per feature
identify the target column (last column)

DATA FORMAT
Labeled files:
include all columns, including target (last column)
one row per record
all values are integers

Unlabeled files:
same as labeled files, but without the target column
used for prediction input

PREPROCESSING
All values are converted to integer-coded categories.

Target:
0 = negative (no disease)
1 = positive (disease)

BMI is converted to 5 categories:
0 = underweight
1 = healthy
2 = overweight
3 = obese
4 = morbidly obese

Mental and Physical Health (0–30 days) are converted to 4 categories:
0 = none (0 days)
1 = low (1–5 days)
2 = moderate (6–15 days)
3 = high (16–30 days)

Diabetes (in heart dataset):
0 = not diabetic
1 = pre-diabetic or diabetic

Other features remain as integer categorical values.

DATASET SIZES
The diabetes dataset includes:
20,000 row sample
100,000 row sample
500,000 row sample

The original dataset contains ~253,000 rows.
The 500,000 row dataset is created using sampling WITH replacement, meaning rows are reused to create a larger dataset. This is done to test performance of the parallel implementation on larger inputs.
The smaller datasets (20k and 100k) are sampled WITHOUT replacement.

PURPOSE
These files exist to:
provide clean input for a C-based Naive Bayes classifier
avoid string parsing by using integer-coded values
standardize dataset structure using metadata
support testing of serial, OpenMP, and MPI implementations