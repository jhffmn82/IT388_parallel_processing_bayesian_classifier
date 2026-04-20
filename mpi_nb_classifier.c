/* File:     mpi_nb_classifier.c
 *
 * Purpose:  MPI Naive Bayesian classifier for our group project.
 *           
 *
 * Course:   IT 388 Parallel Processing
 * Group:    Justin Hoffman, Nathan Wolniak, Brady Davidson
 *
 * Compile:  mpicc mpi_nb_classifier.c -o mpi_nb -lm
 * Run:      mpiexec ./mpi_nb <meta.csv> <labeled.csv> <unlabeled.csv> <output.csv> <k> 		<num_processes>
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

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
 * Later OpenMP: split rows across threads and reduce local counts.
 * Later MPI: each process handles part of the rows and reduces counts.
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
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Calculate the local slice for this process
    int total_rows_to_process = end_row - start_row;
    int rows_per_proc = total_rows_to_process / size;
    int remainder = total_rows_to_process % size;

    int local_start = start_row + rank * rows_per_proc + (rank < remainder ? rank : remainder);
    int local_end = local_start + rows_per_proc + (rank < remainder ? 1 : 0);

    // Allocate and zero local buffers for this rank's partial work
    // Determine the size of feature_counts from the offsets and the last feature's range
    int total_feature_elements = feature_offsets[num_features - 1] + 
                                 (num_classes * feature_num_values[num_features - 1]);

    long long* local_class_counts = (long long*)calloc(num_classes, sizeof(long long));
    long long* local_feature_counts = (long long*)calloc(total_feature_elements, sizeof(long long));

    // Local accumulation loop
    for (int i = local_start; i < local_end; i++) {
        int class_label = labeled_data[i * labeled_cols + labeled_cols - 1];
        int class_idx = class_label_to_index(class_label, class_values, num_classes);
        local_class_counts[class_idx]++;

        for (int j = 0; j < num_features; j++) {
            int value = labeled_data[i * labeled_cols + j];
            int value_idx = feature_value_to_index(j, value, feature_min_values);
            int idx = feature_offsets[j] + class_idx * feature_num_values[j] + value_idx;
            local_feature_counts[idx]++;
        }
    }

    // Reduce local results into the global arrays provided in arguments
    // Use Allreduce so all processes have the finished model
    MPI_Allreduce(local_class_counts, class_counts, num_classes, 
                  MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    
    MPI_Allreduce(local_feature_counts, feature_counts, total_feature_elements, 
                  MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    // Cleanup local memory
    free(local_class_counts);
    free(local_feature_counts);
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
 * Later OpenMP: this loop is easy to parallelize because each row
 * can be classified independently.
 */


void classify_dataset(int* data,
                          int total_rows,
                          int cols,
                          int num_features,
                          int num_classes,
                          int* feature_num_values,
                          int* feature_offsets,
                          int* class_values,
                          int* feature_min_values,
                          double* log_class_priors,
                          double* log_probs,
                          int* predictions) { // Final results gathered here on root
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Calculate row distribution (same logic as before)
    int rows_per_proc = total_rows / size;
    int remainder = total_rows % size;
    
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    int start_row = rank * rows_per_proc + (rank < remainder ? rank : remainder);

    // Local classification
    // Each process only needs a local buffer for its share of predictions
    int* local_predictions = (int*)malloc(local_rows * sizeof(int));

    for (int i = 0; i < local_rows; i++) {
        // Point to the specific row in the (potentially shared) data array
        int global_row_idx = start_row + i;
        local_predictions[i] = classify_row(&data[global_row_idx * cols], 
                                           num_features, num_classes,
                                           feature_num_values, feature_offsets,
                                           class_values, feature_min_values,
                                           log_class_priors, log_probs);
    }

    // Prepare for Gather
    // MPI_Gatherv is needed because the number of rows might differ by 1 per process
    // All ranks need recv_counts and displacements for Allgatherv ~ updated Justin
    int* recv_counts = (int*)malloc(size * sizeof(int));
    int* displacements = (int*)malloc(size * sizeof(int));

    for (int i = 0; i < size; i++) {
        recv_counts[i] = rows_per_proc + (i < remainder ? 1 : 0);
        displacements[i] = i * rows_per_proc + (i < remainder ? i : remainder);
    }
    // Changed to Allgather ~ Justin
    // Collect all local predictions into the root's predictions array
    MPI_Allgatherv(local_predictions, local_rows, MPI_INT,
                   predictions, recv_counts, displacements, MPI_INT,
                   MPI_COMM_WORLD);

    // Cleanup
    free(local_predictions);
    free(recv_counts);
    free(displacements);
}


/* Pull the true class labels out of the last column of the labeled data. */
void build_truth(int* labeled_data, int total_rows, int labeled_cols, int* truth) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Calculate row distribution
    int rows_per_proc = total_rows / size;
    int remainder = total_rows % size;

    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    int start_row = rank * rows_per_proc + (rank < remainder ? rank : remainder);

    // Local extraction
    int* local_truth = (int*)malloc(local_rows * sizeof(int));
    for (int i = 0; i < local_rows; i++) {
        int global_row_idx = start_row + i;
        local_truth[i] = labeled_data[global_row_idx * labeled_cols + labeled_cols - 1];
    }

    // All ranks need recv_counts and displacements for Allgatherv ~ updated Justin
    int* recv_counts = (int*)malloc(size * sizeof(int));
    int* displacements = (int*)malloc(size * sizeof(int));

    for (int i = 0; i < size; i++) {
        recv_counts[i] = rows_per_proc + (i < remainder ? 1 : 0);
        displacements[i] = i * rows_per_proc + (i < remainder ? i : remainder);
    }

    // Changed to Allgatherv ~ Justin
    MPI_Allgatherv(local_truth, local_rows, MPI_INT,
                   truth, recv_counts, displacements, MPI_INT,
                   MPI_COMM_WORLD);

    // Cleanup
    free(local_truth);
    free(recv_counts);
    free(displacements);
}


/* Compute simple accuracy = correct / total. */
double compute_accuracy(int* truth, int* pred, int total_n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Calculate the local range for this process
    int n_per_proc = total_n / size;
    int remainder = total_n % size;
    
    int local_n = n_per_proc + (rank < remainder ? 1 : 0);
    int start_idx = rank * n_per_proc + (rank < remainder ? rank : remainder);

    // Count correct predictions locally
    int local_correct = 0;
    for (int i = 0; i < local_n; i++) {
        if (truth[start_idx + i] == pred[start_idx + i]) {
            local_correct++;
        }
    }

    // Sum the local counts into a global total
    int global_correct = 0;
    MPI_Reduce(&local_correct, &global_correct, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Root process calculates and returns the final ratio
    if (rank == 0) {
        return (double)global_correct / total_n;
    } else {
        return 0.0; // Non-root processes don't typically need the result
    }
}


/* Build a binary confusion matrix.
 * This version assumes the class labels are 0 and 1.
 */
void confusion_matrix_binary(int* truth, int* pred, int total_n,
                                 int* tn, int* fp, int* fn, int* tp) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Calculate the local range for this process
    int n_per_proc = total_n / size;
    int remainder = total_n % size;
    int local_n = n_per_proc + (rank < remainder ? 1 : 0);
    int start_idx = rank * n_per_proc + (rank < remainder ? rank : remainder);

    // Tally local results in an array to simplify reduction
    int local_counts[4] = {0, 0, 0, 0}; // [TN, FP, FN, TP]

    for (int i = 0; i < local_n; i++) {
        int idx = start_idx + i;
        if (truth[idx] == 0 && pred[idx] == 0)      local_counts[0]++; // TN
        else if (truth[idx] == 0 && pred[idx] == 1) local_counts[1]++; // FP
        else if (truth[idx] == 1 && pred[idx] == 0) local_counts[2]++; // FN
        else if (truth[idx] == 1 && pred[idx] == 1) local_counts[3]++; // TP
    }

    // Single reduction of the 4-element array to Rank 0
    int global_counts[4] = {0, 0, 0, 0};
    MPI_Reduce(local_counts, global_counts, 4, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Assign results to output pointers ONLY on root
    if (rank == 0) {
        *tn = global_counts[0];
        *fp = global_counts[1];
        *fn = global_counts[2];
        *tp = global_counts[3];
    }
}



/* Copy selected rows from one flat matrix into another.
 * We use this when building train/test sets for k-fold validation.
 */
void copy_rows(int* src, int src_cols, int* row_indices, int total_n_rows, int* dest) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Calculate row distribution
    int rows_per_proc = total_n_rows / size;
    int remainder = total_n_rows % size;

    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    int start_row_idx = rank * rows_per_proc + (rank < remainder ? rank : remainder);

    // Perform local copy
    // We need a local buffer to store the rows this process is responsible for
    int* local_dest = (int*)malloc(local_rows * src_cols * sizeof(int));

    for (int i = 0; i < local_rows; i++) {
        int global_row_to_copy = row_indices[start_row_idx + i];
        // Use memcpy for a faster row-wise copy than a nested loop
        memcpy(&local_dest[i * src_cols], 
               &src[global_row_to_copy * src_cols], 
               src_cols * sizeof(int));
    }

    // Prepare Gathering metadata (only needed by root)
    int* recv_counts = NULL;
    int* displacements = NULL;

    if (rank == 0) {
        recv_counts = (int*)malloc(size * sizeof(int));
        displacements = (int*)malloc(size * sizeof(int));

        for (int i = 0; i < size; i++) {
            int proc_rows = rows_per_proc + (i < remainder ? 1 : 0);
            int proc_start = i * rows_per_proc + (i < remainder ? i : remainder);
            
            // Counts and displacements are in units of MPI_INT (total elements)
            recv_counts[i] = proc_rows * src_cols;
            displacements[i] = proc_start * src_cols;
        }
    }

    // Gather local row blocks into the global dest array on Rank 0
    MPI_Gatherv(local_dest, local_rows * src_cols, MPI_INT,
                dest, recv_counts, displacements, MPI_INT,
                0, MPI_COMM_WORLD);

    // Cleanup
    free(local_dest);
    if (rank == 0) {
        free(recv_counts);
        free(displacements);
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

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 7) {
        if (rank == 0) Usage(argv[0]);
        MPI_Finalize();
        return 0;
    }

    /* Read command line arguments. */
    meta_file = argv[1];
    labeled_file = argv[2];
    unlabeled_file = argv[3];
    output_file = argv[4];
    k = atoi(argv[5]);
    num_processes = atoi(argv[6]);

    if (k < 2) Usage(argv[0]);

    meta_file = argv[1];
    labeled_file = argv[2];
    unlabeled_file = argv[3];
    output_file = argv[4];
    k = atoi(argv[5]);
    // num_processes = atoi(argv[6]); // In MPI, we usually use the 'size' from MPI_Comm_size

    /* Metadata and Data Loading */
    // For simplicity, all processes read the metadata/CSV or Rank 0 reads and broadcasts.
    // Reading on all processes is often faster than broadcasting massive arrays.
    read_metadata(meta_file, &num_features, &num_classes,
                  &feature_num_values, &feature_min_values,
                  &class_values, &feature_offsets,
                  &total_prob_size, target_name);

    labeled_cols = num_features + 1;
    labeled_rows = count_rows(labeled_file);
    unlabeled_rows = count_rows(unlabeled_file);

    labeled_data = (int*) malloc(labeled_rows * labeled_cols * sizeof(int));
    unlabeled_data = (int*) malloc(unlabeled_rows * num_features * sizeof(int));

    read_csv_data(labeled_file, labeled_cols, labeled_rows, labeled_data);
    read_csv_data(unlabeled_file, num_features, unlabeled_rows, unlabeled_data);

    /* Model Allocation */
    class_counts = (long long*) calloc(num_classes, sizeof(long long));
    feature_counts = (long long*) calloc(total_prob_size, sizeof(long long));
    log_class_priors = (double*) malloc(num_classes * sizeof(double));
    log_probs = (double*) malloc(total_prob_size * sizeof(double));

    truth = (int*) malloc(labeled_rows * sizeof(int));
    train_predictions = (int*) malloc(labeled_rows * sizeof(int));
    unlabeled_predictions = (int*) malloc(unlabeled_rows * sizeof(int));

    /* Parallel Execution using MPI functions */
    build_truth(labeled_data, labeled_rows, labeled_cols, truth);

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    
    // Note: train_model internally calls accumulate_counts logic
    // Ensure train_model is updated to reduce counts and then compute priors/probs
    train_model(labeled_data, labeled_rows, labeled_cols, num_features, num_classes,
                    feature_num_values, feature_offsets, class_values,
                    feature_min_values, total_prob_size,
                    class_counts, feature_counts, log_class_priors, log_probs);
    
    // Broadcast the trained model from Rank 0 to all workers for classification
    MPI_Bcast(log_class_priors, num_classes, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(log_probs, total_prob_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    t1 = MPI_Wtime();
    train_time = t1 - t0;

    t0 = MPI_Wtime();
    classify_dataset(labeled_data, labeled_rows, labeled_cols, num_features, num_classes,
                         feature_num_values, feature_offsets, class_values,
                         feature_min_values, log_class_priors, log_probs, train_predictions);

    classify_dataset(unlabeled_data, unlabeled_rows, num_features, num_features, num_classes,
                         feature_num_values, feature_offsets, class_values,
                         feature_min_values, log_class_priors, log_probs, unlabeled_predictions);
    t1 = MPI_Wtime();
    classify_time = t1 - t0;

    train_accuracy = compute_accuracy(truth, train_predictions, labeled_rows);

    /* Cross Validation */
    t0 = MPI_Wtime();
    cross_validate(labeled_data, labeled_rows, labeled_cols, num_features, num_classes,
                       feature_num_values, feature_min_values, feature_offsets,
                       class_values, total_prob_size, k,
                       &avg_train_acc, &avg_test_acc, &tn, &fp, &fn, &tp);
    t1 = MPI_Wtime();
    cv_time = t1 - t0;

    /* Output and Reporting (Rank 0 only) */
    if (rank == 0) {
        total_time = train_time + classify_time + cv_time;
        write_predictions_csv(output_file, unlabeled_data, unlabeled_rows,
                              num_features, target_name, unlabeled_predictions);

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

    }

    /* Cleanup */
    free(feature_num_values); free(feature_min_values); free(class_values);
    free(feature_offsets); free(labeled_data); free(unlabeled_data);
    free(class_counts); free(feature_counts); free(log_class_priors);
    free(log_probs); free(truth); free(train_predictions);
    free(unlabeled_predictions);

    MPI_Finalize();
    return 0;
}
