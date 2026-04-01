#!/bin/bash

# 2D benchmark: sweeps N, B, P configurations

p_values=(2 4)
N_values=(64 128 256 512 1024)

for p in "${p_values[@]}"; do
  num_procs=$((p * p))
  for N in "${N_values[@]}"; do
    max_b=$((N / (p * p)))

    b=1
    while (( b <= max_b )); do
      echo "=== N=$N B=$b P=$p (procs=$num_procs) ==="
      make -s build N_DIM=$N B_DIM=$b P_DIM=$p
      mpiexec -n $num_procs ./zmodel.x
      echo "------------------------------------------------"
      b=$((b * 2))
    done
  done
done
