#!/bin/bash

# All output goes to cluster_output.txt
exec > cluster_output.txt 2>&1

echo "============================================"
echo " CLUSTER TRIAL RUN"
echo " $(date)"
echo "============================================"

# ── Phase 0: Compile ─────────────────────────────
echo ""
echo "=== Compiling binaries ==="
gcc  -O2 -Wall -fopenmp nb_classifier.c       -lm -o nb_serial
gcc  -O2 -Wall -fopenmp omp_nb_classifier.c   -lm -o nb_omp
mpicc -O2 -Wall -fopenmp mpi_nb_classifier.c  -lm -o nb_mpi
mpicc -O2 -Wall -fopenmp nb_classifier_hybrid.c -lm -o nb_hybrid
echo "Done compiling."

# ── Phase 1: Heart correctness ───────────────────
echo ""
echo "=== Phase 1: Heart correctness ==="

echo "-- Serial | heart | 1 thread --"
./nb_serial heart_meta.csv heart_full_labeled.csv heart_full_unlabeled.csv temp.csv 5 1

echo "-- OMP | heart | 8 threads --"
./nb_omp heart_meta.csv heart_full_labeled.csv heart_full_unlabeled.csv temp.csv 5 8

echo "-- MPI | heart | 8 procs --"
mpiexec -n 8 ./nb_mpi heart_meta.csv heart_full_labeled.csv heart_full_unlabeled.csv temp.csv 5 8

echo "-- Hybrid | heart | 4 procs x 2 threads --"
mpiexec -n 4 ./nb_hybrid heart_meta.csv heart_full_labeled.csv heart_full_unlabeled.csv temp.csv 5 2

# ── Phase 2: Data-size scaling ───────────────────
echo ""
echo "=== Phase 2: Data-size scaling ==="

for dataset in 20000 100000 500000; do
    echo "-- Serial | ${dataset} --"
    ./nb_serial diabetes_meta.csv diabetes_${dataset}_labeled.csv diabetes_${dataset}_unlabeled.csv temp.csv 5 1

    echo "-- OMP | ${dataset} | 8 threads --"
    ./nb_omp diabetes_meta.csv diabetes_${dataset}_labeled.csv diabetes_${dataset}_unlabeled.csv temp.csv 5 8

    echo "-- MPI | ${dataset} | 8 procs --"
    mpiexec -n 8 ./nb_mpi diabetes_meta.csv diabetes_${dataset}_labeled.csv diabetes_${dataset}_unlabeled.csv temp.csv 5 8

    echo "-- Hybrid | ${dataset} | 2 procs x 4 threads --"
    mpiexec -n 2 ./nb_hybrid diabetes_meta.csv diabetes_${dataset}_labeled.csv diabetes_${dataset}_unlabeled.csv temp.csv 5 4
done

# ── Phase 3: Serial baseline ─────────────────────
echo ""
echo "=== Phase 3: Serial baseline | 1M ==="

for run in 1 2 3 4 5; do
    echo "-- Run ${run} --"
    ./nb_serial diabetes_meta.csv diabetes_1000000_labeled.csv diabetes_1000000_unlabeled.csv temp.csv 5 1
done

# ── Phase 4: OMP scaling ─────────────────────────
echo ""
echo "=== Phase 4: OMP scaling | 1M ==="

for threads in 1 2 4 8; do
    for run in 1 2 3 4 5; do
        echo "-- ${threads} threads | Run ${run} --"
        ./nb_omp diabetes_meta.csv diabetes_1000000_labeled.csv diabetes_1000000_unlabeled.csv temp.csv 5 ${threads}
    done
done

# ── Phase 5: MPI scaling ─────────────────────────
echo ""
echo "=== Phase 5: MPI scaling | 1M ==="

for procs in 1 2 4 8; do
    for run in 1 2 3 4 5; do
        echo "-- ${procs} procs | Run ${run} --"
        mpiexec -n ${procs} ./nb_mpi diabetes_meta.csv diabetes_1000000_labeled.csv diabetes_1000000_unlabeled.csv temp.csv 5 ${procs}
    done
done

# ── Phase 6: Hybrid sweep ─────────────────────────
echo ""
echo "=== Phase 6: Hybrid sweep | 1M ==="

for procs in 1 2 4 8; do
    for threads in 1 2 4 8; do
        for run in 1 2 3 4 5; do
            echo "-- ${procs} procs x ${threads} threads | Run ${run} --"
            mpiexec -n ${procs} ./nb_hybrid diabetes_meta.csv diabetes_1000000_labeled.csv diabetes_1000000_unlabeled.csv temp.csv 5 ${threads}
        done
    done
done

echo ""
echo "============================================"
echo " TRIAL COMPLETE: $(date)"
echo "============================================"
