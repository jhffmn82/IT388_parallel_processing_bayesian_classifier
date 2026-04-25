# IT388 Parallel Naive Bayes Classifier

Group project for IT 388 Parallel Processing. Implements a Naive Bayes classifier in C across four versions: serial, OpenMP, MPI, and a hybrid MPI+OpenMP. Tested on public health datasets from the CDC BRFSS survey.

Group: Justin Hoffman, Nathan Wolniak, Brady Davidson, Daniel Sevik

---

## C Classifier Files

nb_classifier.c — serial baseline

omp_nb_classifier.c — OpenMP parallel version

mpi_nb_classifier.c — MPI parallel version

nb_classifier_hybrid.c — hybrid MPI + OpenMP version

All four take the same arguments:
```
<meta.csv> <labeled.csv> <unlabeled.csv> <output.csv> <k> <num_processes/threads>
```

## Python Preprocessing

diabetes_csv.py — downloads and preprocesses the UCI diabetes dataset

heart_csv.py — preprocesses the heart disease dataset from the provided CSV

parse_output.py — parses timing output from cluster runs and builds summary tables

## Data

diabetes_output/ — preprocessed diabetes CSVs and metadata file

heart_output/ — preprocessed heart disease CSVs and metadata file

heart_disease_health_indicators_BRFSS2015.csv — raw heart dataset input file

See Python_Data_Readme.txt for details on the data format and preprocessing decisions.

## Cluster Scripts

runscript_final.sb — SLURM job script used on the Expanse cluster

cluster_trial.sh — shell script used for local cluster testing

## Output Files

expanse_out.txt — raw timing output from Expanse runs

expanse_out_parsed.txt — parsed summary tables from Expanse runs

cluster_output.txt — raw timing output from local cluster runs

cluster_output_parsed.txt — parsed summary tables from local cluster runs

## Notebooks

bayesian-classifier-example.ipynb — reference implementation used to verify classifier correctness

DataVis.ipynb — plots and visualizations for the timing results
