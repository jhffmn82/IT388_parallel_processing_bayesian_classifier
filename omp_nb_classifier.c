/* File:     omp_nb_classifier.c
 *
 * Purpose:  OpenMP parallel Naive Bayesian classifier for our group project.
 *           Forked from the serial baseline. Parallelism is added to the
 *           training count accumulation, log-probability conversion,
 *           classification, accuracy, and confusion matrix steps.
 *           The sixth CLI argument (num_processes) is used here as the
 *           number of OpenMP threads.
 *
 * Course:   IT 388 Parallel Processing
 * Group:    Justin Hoffman, Nathan Wolniak, Brady Davidson
 *
 * Compile:  gcc -O2 -Wall -fopenmp omp_nb_classifier.c -lm -o nb_omp
 * Run:      ./nb_omp <meta.csv> <labeled.csv> <unlabeled.csv> <output.csv> <k> <num_threads>
 *
 * Notes:
 *   1. Laplace smoothing is fixed at 1.0 in this version.
 *   2. The sixth argument sets the OpenMP thread count via omp_set_num_threads.
 *   3. Parallel regions use schedule(static) since per-row work is uniform.
 *   4. The fold loop in cross_validate is left serial; parallelism comes
 *      from the inner train/classify calls to avoid nested parallelism.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

//needed for reading in CSV, defines max rows to read in at a time
#define MAX_LINE_LEN 8192

/* Print the expected command line format and quit. */
void Usage(char* prog_name) {
    fprintf(stderr,
        "usage: %s <meta.csv> <labeled.csv> <unlabeled.csv> <output.csv> <k> <num_processes>\n",
        prog_name);
    exit(0);
}

/* Count the number of data rows in a CSV file.
 * We start at -1 because the first row is the header.
 */
int count_rows(const char* filename) {
    FILE* fp = fopen(filename, "r");
    char line[MAX_LINE_LEN];
    int rows = -1;

    while (fgets(line, sizeof(line), fp) != NULL) rows++;
    fclose(fp);
    return rows;
}

/* Read the last column name from the metadata header.
 * That becomes the target column name in the output file.
 */
void get_target_name(const char* meta_file, char* target_name) {
    FILE* fp = fopen(meta_file, "r");
    char line[MAX_LINE_LEN];
    char* token;

    fgets(line, sizeof(line), fp);
    fclose(fp);

    token = strtok(line, ",\r\n");
    while (token != NULL) {
        strcpy(target_name, token);
        token = strtok(NULL, ",\r\n");
    }
}

/* Read the metadata file.
 *
 * From the metadata we get:
 *   number of features
 *   number of classes
 *   number of possible values for each feature
 *   minimum value for each feature
 *   actual class labels
 *   offsets for flattening the probability tables
 */
void read_metadata(const char* meta_file,
                   int* num_features,
                   int* num_classes,
                   int** feature_num_values,
                   int** feature_min_values,
                   int** class_values,
                   int** feature_offsets,
                   int* total_prob_size,
                   char* target_name) {
    FILE* fp = fopen(meta_file, "r");
    char line[MAX_LINE_LEN];
    char copy[MAX_LINE_LEN];
    char* token;
    int total_cols = 0;
    int i, r;

    /* Count how many columns are in the metadata header. */
    fgets(line, sizeof(line), fp);
    strcpy(copy, line);
    token = strtok(copy, ",\r\n");
    while (token != NULL) {
        total_cols++;
        token = strtok(NULL, ",\r\n");
    }

    *num_features = total_cols - 1;
    get_target_name(meta_file, target_name);

    /* Allocate the main metadata arrays. */
    *feature_num_values = (int*) malloc((*num_features) * sizeof(int));
    *feature_min_values = (int*) malloc((*num_features) * sizeof(int));
    *feature_offsets = (int*) malloc((*num_features) * sizeof(int));

    /* The second row tells us how many values each column can take. */
    fgets(line, sizeof(line), fp);
    token = strtok(line, ",\r\n");
    for (i = 0; i < *num_features; i++) {
        (*feature_num_values)[i] = atoi(token);
        token = strtok(NULL, ",\r\n");
    }
    *num_classes = atoi(token);
    *class_values = (int*) malloc((*num_classes) * sizeof(int));

    /* Precompute offsets so all feature/class/value counts can live
     * in one flat array instead of a 3D structure.
     */
    *total_prob_size = 0;
    for (i = 0; i < *num_features; i++) {
        (*feature_offsets)[i] = *total_prob_size;
        *total_prob_size += (*num_classes) * (*feature_num_values)[i];
    }

    /* Read the allowed values rows.
     * We only really need the minimum feature value and the class labels.
     */
    for (r = 0; r < *num_classes; r++) {
        fgets(line, sizeof(line), fp);
        token = strtok(line, ",\r\n");
        for (i = 0; i < total_cols; i++) {
            int value = atoi(token);
            if (r == 0 && i < *num_features) (*feature_min_values)[i] = value;
            if (i == total_cols - 1) (*class_values)[r] = value;
            token = strtok(NULL, ",\r\n");
        }
    }

    fclose(fp);
}

/* Read either the labeled or unlabeled CSV data into one flat array.
 * We skip the header and then store everything row by row.
 */
void read_csv_data(const char* filename, int cols, int rows, int* data) {
    FILE* fp = fopen(filename, "r");
    char line[MAX_LINE_LEN];
    char* token;
    int i, j;

    fgets(line, sizeof(line), fp);   /* skip header */

    for (i = 0; i < rows; i++) {
        fgets(line, sizeof(line), fp);
        token = strtok(line, ",\r\n");
        for (j = 0; j < cols; j++) {
            data[i * cols + j] = atoi(token);
            token = strtok(NULL, ",\r\n");
        }
    }

    fclose(fp);
}

/* Convert an actual class label into a class index 0..C-1. */
int class_label_to_index(int class_label, int* class_values, int num_classes) {
    int c;
    for (c = 0; c < num_classes; c++) {
        if (class_values[c] == class_label) return c;
    }
    return 0;
}

/* Convert a feature value into an index using the minimum value from metadata.
 * Example: if a feature ranges from 1..5, then value 1 maps to index 0.
 */
int feature_value_to_index(int feature_j, int value, int* feature_min_values) {
    return value - feature_min_values[feature_j];
}

/* Set all class counts and feature counts back to zero before training. */
void zero_arrays(int num_classes,
                 int total_prob_size,
                 long long* class_counts,
                 long long* feature_counts) {
    int i;
    for (i = 0; i < num_classes; i++) class_counts[i] = 0;
    for (i = 0; i < total_prob_size; i++) feature_counts[i] = 0;
}


/* Count how often each class appears and how often each feature value
 * appears inside each class.
 *
 * This is the main training loop.
 *
 * OpenMP strategy:
 *   Use array reductions on the two count arrays. The runtime gives each
 *   thread private copies, accumulates into them during the loop, then
 *   merges them into the shared global counts (typically via a tree
 *   reduction) at the end of the parallel region.
 *
 *   The MPI version uses the same logical pattern with MPI_Allreduce on
 *   class_counts and feature_counts; the hybrid version combines both.
 */
void accumulate_counts_range(int* labeled_data,
                             int start_row,
                             int end_row,
                             int labeled_cols,
                             int num_features,
                             int num_classes,
                             int* feature_num_values,
                             int* feature_offsets,
                             int* class_values,
                             int* feature_min_values,
                             long long* class_counts,
                             long long* feature_counts) {
    int i, j;

    /* Compute total_prob_size so we can size the reduction clause. */
    int total_prob_size = 0;
    for (j = 0; j < num_features; j++) {
        total_prob_size += num_classes * feature_num_values[j];
    }

    #pragma omp parallel for schedule(static) reduction(+:class_counts[:num_classes]) reduction(+:feature_counts[:total_prob_size])
    for (i = start_row; i < end_row; i++) {
        int class_label = labeled_data[i * labeled_cols + labeled_cols - 1];
        int class_idx = class_label_to_index(class_label, class_values, num_classes);
        class_counts[class_idx]++;

        for (j = 0; j < num_features; j++) {
            int value = labeled_data[i * labeled_cols + j];
            int value_idx = feature_value_to_index(j, value, feature_min_values);
            int idx = feature_offsets[j] + class_idx * feature_num_values[j] + value_idx;
            feature_counts[idx]++;
        }
    }
}

/* Convert the raw counts into log probabilities.
 * We use Laplace smoothing with alpha fixed at 1.0.
 */
void counts_to_log_probs(int num_features,
                         int num_classes,
                         int* feature_num_values,
                         int* feature_offsets,
                         long long* class_counts,
                         long long* feature_counts,
                         int total_rows,
                         double* log_class_priors,
                         double* log_probs) {
    const double alpha = 1.0;
    int c, j, v;
    double prior_denom = total_rows + alpha * num_classes;

    /* Compute the prior probability for each class. */
    for (c = 0; c < num_classes; c++) {
        log_class_priors[c] = log((class_counts[c] + alpha) / prior_denom);
    }

    /* Compute the conditional probability tables for each feature. */
    for (j = 0; j < num_features; j++) {
        for (c = 0; c < num_classes; c++) {
            double denom = class_counts[c] + alpha * feature_num_values[j];
            for (v = 0; v < feature_num_values[j]; v++) {
                int idx = feature_offsets[j] + c * feature_num_values[j] + v;
                log_probs[idx] = log((feature_counts[idx] + alpha) / denom);
            }
        }
    }
}

/* Train the model by clearing the old counts, collecting new counts,
 * and then converting those counts into log probabilities.
 */
void train_model(int* labeled_data,
                 int rows,
                 int labeled_cols,
                 int num_features,
                 int num_classes,
                 int* feature_num_values,
                 int* feature_offsets,
                 int* class_values,
                 int* feature_min_values,
                 int total_prob_size,
                 long long* class_counts,
                 long long* feature_counts,
                 double* log_class_priors,
                 double* log_probs) {
    
    zero_arrays(num_classes, total_prob_size, class_counts, feature_counts);
    accumulate_counts_range(labeled_data, 0, rows, labeled_cols, num_features, num_classes,
                            feature_num_values, feature_offsets, class_values,
                            feature_min_values, class_counts, feature_counts);
    counts_to_log_probs(num_features, num_classes, feature_num_values, feature_offsets,
                        class_counts, feature_counts, rows, log_class_priors, log_probs);
}

/* Classify one row by computing the log score for each class
 * and returning the class with the best score.
 */
int classify_row(int* row,
                 int num_features,
                 int num_classes,
                 int* feature_num_values,
                 int* feature_offsets,
                 int* class_values,
                 int* feature_min_values,
                 double* log_class_priors,
                 double* log_probs) {
    int c, j;
    int best_class = 0;
    double best_score = -1e300;

    for (c = 0; c < num_classes; c++) {
        double score = log_class_priors[c];
        for (j = 0; j < num_features; j++) {
            int value_idx = feature_value_to_index(j, row[j], feature_min_values);
            int idx = feature_offsets[j] + c * feature_num_values[j] + value_idx;
            score += log_probs[idx];
        }
        if (score > best_score) {
            best_score = score;
            best_class = c;
        }
    }

    return class_values[best_class];
}

/* Classify every row in a dataset.
 *
 * OpenMP approach:
 *   Each row is classified independently from
 *   the others, and each thread writes to its own predictions[i] slot -
 *   so no contention, no reduction needed. Just split the rows across
 *   threads with schedule(static) since every row does the same amount
 *   of work (same number of features, same arithmetic).
 */
void classify_dataset(int* data,
                      int rows,
                      int cols,
                      int num_features,
                      int num_classes,
                      int* feature_num_values,
                      int* feature_offsets,
                      int* class_values,
                      int* feature_min_values,
                      double* log_class_priors,
                      double* log_probs,
                      int* predictions) {
    int i;

    #pragma omp parallel for
    for (i = 0; i < rows; i++) {
        predictions[i] = classify_row(&data[i * cols], num_features, num_classes,
                                      feature_num_values, feature_offsets,
                                      class_values, feature_min_values,
                                      log_class_priors, log_probs);
    }
}

/* Pull the true class labels out of the last column of the labeled data.
 *
 * OpenMP: parallelize the row loop.
 */
void build_truth(int* labeled_data, int rows, int labeled_cols, int* truth) {
    int i;

    #pragma omp parallel for
    for (i = 0; i < rows; i++) {
        truth[i] = labeled_data[i * labeled_cols + labeled_cols - 1];
    }
}
/* Compute simple accuracy = correct / total.
 *
 * OpenMP: standard reduction on the correct count.
 */
double compute_accuracy(int* truth, int* pred, int n) {
    int i, correct = 0;

    #pragma omp parallel for reduction(+:correct)
    for (i = 0; i < n; i++) {
        if (truth[i] == pred[i]) correct++;
    }

    return (double) correct / n;
}

/* Build a binary confusion matrix.
 * This version assumes the class labels are 0 and 1.
 *
 * OpenMP: four-way reduction on local accumulators, then written back
 * through the output pointers at the end.
 */
void confusion_matrix_binary(int* truth, int* pred, int n,
                             int* tn, int* fp, int* fn, int* tp) {
    int i;
    int tn_local = 0, fp_local = 0, fn_local = 0, tp_local = 0;

    #pragma omp parallel for schedule(static) \
        reduction(+:tn_local, fp_local, fn_local, tp_local)
    for (i = 0; i < n; i++) {
        if      (truth[i] == 0 && pred[i] == 0) tn_local++;
        else if (truth[i] == 0 && pred[i] == 1) fp_local++;
        else if (truth[i] == 1 && pred[i] == 0) fn_local++;
        else if (truth[i] == 1 && pred[i] == 1) tp_local++;
    }

    *tn = tn_local;
    *fp = fp_local;
    *fn = fn_local;
    *tp = tp_local;
}


/* Copy selected rows from one flat matrix into another.
 * We use this when building train/test sets for k-fold validation.
 *
 * OpenMP: parallelize the outer row loop. Each iteration writes to a
 * distinct row in dest, so no contention.
 */
void copy_rows(int* src, int src_cols, int* row_indices, int n_rows, int* dest) {
    int i, j;

    #pragma omp parallel for private(j)
    for (i = 0; i < n_rows; i++) {
        for (j = 0; j < src_cols; j++) {
            dest[i * src_cols + j] = src[row_indices[i] * src_cols + j];
        }
    }
}

/* Run k-fold cross validation.
 * This version uses simple contiguous folds instead of shuffling.
 *
 * Later on, folds could also be split across processes or threads.
 */
void cross_validate(int* labeled_data,
                    int labeled_rows,
                    int labeled_cols,
                    int num_features,
                    int num_classes,
                    int* feature_num_values,
                    int* feature_min_values,
                    int* feature_offsets,
                    int* class_values,
                    int total_prob_size,
                    int k,
                    double* avg_train_acc,
                    double* avg_test_acc,
                    int* total_tn,
                    int* total_fp,
                    int* total_fn,
                    int* total_tp) {
    int fold, i;
    double train_sum = 0.0, test_sum = 0.0;

    /* Allocate one model for reuse across all folds. */
    long long* class_counts = (long long*) malloc(num_classes * sizeof(long long));
    long long* feature_counts = (long long*) malloc(total_prob_size * sizeof(long long));
    double* log_class_priors = (double*) malloc(num_classes * sizeof(double));
    double* log_probs = (double*) malloc(total_prob_size * sizeof(double));

    *total_tn = *total_fp = *total_fn = *total_tp = 0;

    for (fold = 0; fold < k; fold++) {
        int start = (fold * labeled_rows) / k;
        int end = ((fold + 1) * labeled_rows) / k;
        int test_size = end - start;
        int train_size = labeled_rows - test_size;

        /* Build row index lists for this fold. */
        int* train_idx = (int*) malloc(train_size * sizeof(int));
        int* test_idx = (int*) malloc(test_size * sizeof(int));

        /* Allocate the fold-specific train/test data and predictions. */
        int* train_data = (int*) malloc(train_size * labeled_cols * sizeof(int));
        int* test_data = (int*) malloc(test_size * labeled_cols * sizeof(int));
        int* y_train = (int*) malloc(train_size * sizeof(int));
        int* y_test = (int*) malloc(test_size * sizeof(int));
        int* pred_train = (int*) malloc(train_size * sizeof(int));
        int* pred_test = (int*) malloc(test_size * sizeof(int));

        int train_pos = 0, test_pos = 0;
        int tn, fp, fn, tp;

        /* Split rows into this fold's train set and test set. */
        for (i = 0; i < labeled_rows; i++) {
            if (i >= start && i < end) test_idx[test_pos++] = i;
            else train_idx[train_pos++] = i;
        }

        copy_rows(labeled_data, labeled_cols, train_idx, train_size, train_data);
        copy_rows(labeled_data, labeled_cols, test_idx, test_size, test_data);
        build_truth(train_data, train_size, labeled_cols, y_train);
        build_truth(test_data, test_size, labeled_cols, y_test);

        /* Train on the training fold. */
        train_model(train_data, train_size, labeled_cols, num_features, num_classes,
                    feature_num_values, feature_offsets, class_values,
                    feature_min_values, total_prob_size,
                    class_counts, feature_counts, log_class_priors, log_probs);

        /* Score both train and test so we can report both averages. */
        classify_dataset(train_data, train_size, labeled_cols, num_features, num_classes,
                         feature_num_values, feature_offsets, class_values,
                         feature_min_values, log_class_priors, log_probs, pred_train);

        classify_dataset(test_data, test_size, labeled_cols, num_features, num_classes,
                         feature_num_values, feature_offsets, class_values,
                         feature_min_values, log_class_priors, log_probs, pred_test);

        train_sum += compute_accuracy(y_train, pred_train, train_size);
        test_sum += compute_accuracy(y_test, pred_test, test_size);

        confusion_matrix_binary(y_test, pred_test, test_size, &tn, &fp, &fn, &tp);
        *total_tn += tn;
        *total_fp += fp;
        *total_fn += fn;
        *total_tp += tp;

        free(train_idx);
        free(test_idx);
        free(train_data);
        free(test_data);
        free(y_train);
        free(y_test);
        free(pred_train);
        free(pred_test);
    }

    *avg_train_acc = train_sum / k;
    *avg_test_acc = test_sum / k;

    free(class_counts);
    free(feature_counts);
    free(log_class_priors);
    free(log_probs);
}

/* Write the unlabeled dataset back out with the predicted class added
 * as the final column.
 */
void write_predictions_csv(const char* filename,
                           int* unlabeled_data,
                           int unlabeled_rows,
                           int num_features,
                           const char* target_name,
                           int* predictions) {
    FILE* fp = fopen(filename, "w");
    int i, j;

    /* Write a simple header. */
    for (j = 0; j < num_features; j++) {
        fprintf(fp, "X%d,", j + 1);
    }
    fprintf(fp, "%s\n", target_name);

    /* Write each unlabeled row followed by its prediction. */
    for (i = 0; i < unlabeled_rows; i++) {
        for (j = 0; j < num_features; j++) {
            fprintf(fp, "%d,", unlabeled_data[i * num_features + j]);
        }
        fprintf(fp, "%d\n", predictions[i]);
    }

    fclose(fp);
}

/* Main driver for the whole program.
 * This reads the files, trains the model, runs cross validation,
 * classifies the unlabeled data, and prints the results.
 */
int main(int argc, char* argv[]) {
    char* meta_file;
    char* labeled_file;
    char* unlabeled_file;
    char* output_file;
    int k, num_processes;

    int num_features, num_classes, total_prob_size;
    int* feature_num_values;
    int* feature_min_values;
    int* class_values;
    int* feature_offsets;
    char target_name[256];

    int labeled_rows, labeled_cols, unlabeled_rows;
    int* labeled_data;
    int* unlabeled_data;

    long long* class_counts;
    long long* feature_counts;
    double* log_class_priors;
    double* log_probs;

    int* truth;
    int* train_predictions;
    int* unlabeled_predictions;

    double t0, t1;
    double train_time, classify_time, cv_time, total_time;
    double train_accuracy, avg_train_acc, avg_test_acc;
    int tn, fp, fn, tp;

    if (argc != 7) Usage(argv[0]);

    /* Read command line arguments. */
    meta_file = argv[1];
    labeled_file = argv[2];
    unlabeled_file = argv[3];
    output_file = argv[4];
    k = atoi(argv[5]);
    num_processes = atoi(argv[6]);

    if (k < 2) Usage(argv[0]);

    // set num_threads
    omp_set_num_threads(num_processes);

    /* Confirm the thread count for our log files. */
    #pragma omp parallel
    {
        #pragma omp single
        printf("OpenMP running with %d threads\n", omp_get_num_threads());
    }

    /* Read metadata and figure out the problem dimensions. */
    read_metadata(meta_file, &num_features, &num_classes,
                  &feature_num_values, &feature_min_values,
                  &class_values, &feature_offsets,
                  &total_prob_size, target_name);

    labeled_cols = num_features + 1;
    labeled_rows = count_rows(labeled_file);
    unlabeled_rows = count_rows(unlabeled_file);

    /* Allocate the labeled and unlabeled datasets. */
    labeled_data = (int*) malloc(labeled_rows * labeled_cols * sizeof(int));
    unlabeled_data = (int*) malloc(unlabeled_rows * num_features * sizeof(int));

    /* Read the actual CSV values into memory. */
    read_csv_data(labeled_file, labeled_cols, labeled_rows, labeled_data);
    read_csv_data(unlabeled_file, num_features, unlabeled_rows, unlabeled_data);

    /* Allocate the model arrays. */
    class_counts = (long long*) malloc(num_classes * sizeof(long long));
    feature_counts = (long long*) malloc(total_prob_size * sizeof(long long));
    log_class_priors = (double*) malloc(num_classes * sizeof(double));
    log_probs = (double*) malloc(total_prob_size * sizeof(double));

    /* Allocate arrays for labels and predictions. */
    truth = (int*) malloc(labeled_rows * sizeof(int));
    train_predictions = (int*) malloc(labeled_rows * sizeof(int));
    unlabeled_predictions = (int*) malloc(unlabeled_rows * sizeof(int));

    build_truth(labeled_data, labeled_rows, labeled_cols, truth);

    /* Time the training step on the full labeled dataset. */
    t0 = omp_get_wtime();
    train_model(labeled_data, labeled_rows, labeled_cols, num_features, num_classes,
                feature_num_values, feature_offsets, class_values,
                feature_min_values, total_prob_size,
                class_counts, feature_counts, log_class_priors, log_probs);
    t1 = omp_get_wtime();
    train_time = t1 - t0;

    /* Time classification on both the labeled and unlabeled datasets. */
    t0 = omp_get_wtime();
    classify_dataset(labeled_data, labeled_rows, labeled_cols, num_features, num_classes,
                     feature_num_values, feature_offsets, class_values,
                     feature_min_values, log_class_priors, log_probs, train_predictions);

    classify_dataset(unlabeled_data, unlabeled_rows, num_features, num_features, num_classes,
                     feature_num_values, feature_offsets, class_values,
                     feature_min_values, log_class_priors, log_probs, unlabeled_predictions);
    t1 = omp_get_wtime();
    classify_time = t1 - t0;

    train_accuracy = compute_accuracy(truth, train_predictions, labeled_rows);

    /* Time k-fold cross validation separately. */
    t0 = omp_get_wtime();
    cross_validate(labeled_data, labeled_rows, labeled_cols, num_features, num_classes,
                   feature_num_values, feature_min_values, feature_offsets,
                   class_values, total_prob_size, k,
                   &avg_train_acc, &avg_test_acc, &tn, &fp, &fn, &tp);
    t1 = omp_get_wtime();
    cv_time = t1 - t0;

    total_time = train_time + classify_time + cv_time;

    /* Write predictions for the unlabeled dataset. */
    write_predictions_csv(output_file, unlabeled_data, unlabeled_rows,
                          num_features, target_name, unlabeled_predictions);

    /* Print a simple summary of results. */
    printf("\n=== Naive Bayesian Classification Results ===\n");
    printf("Training rows:   %d\n", labeled_rows);
    printf("Unlabeled rows:  %d\n", unlabeled_rows);
    printf("Features:        %d\n", num_features);
    printf("Classes:         %d\n", num_classes);
    printf("k-folds:         %d\n", k);
    printf("Processes:       %d\n", num_processes);

    printf("\nTraining accuracy:         %.6f\n", train_accuracy);
    printf("Average CV train accuracy: %.6f\n", avg_train_acc);
    printf("Average CV test accuracy:  %.6f\n", avg_test_acc);

    printf("\nConfusion Matrix from CV test folds\n");
    printf("TN: %d  FP: %d\n", tn, fp);
    printf("FN: %d  TP: %d\n", fn, tp);

    printf("\nTimings (excluding file I/O)\n");
    printf("Train time:    %.6f sec\n", train_time);
    printf("Classify time: %.6f sec\n", classify_time);
    printf("CV time:       %.6f sec\n", cv_time);
    printf("Total time:    %.6f sec\n", total_time);
    printf("\nPredictions written to: %s\n", output_file);

    /* Free all heap memory before exiting. */
    free(feature_num_values);
    free(feature_min_values);
    free(class_values);
    free(feature_offsets);
    free(labeled_data);
    free(unlabeled_data);
    free(class_counts);
    free(feature_counts);
    free(log_class_priors);
    free(log_probs);
    free(truth);
    free(train_predictions);
    free(unlabeled_predictions);

    return 0;
}

