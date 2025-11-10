#!/bin/bash

p_values=(2 4)
N_values=(64 128 256 512 1024)

CUDA_HOME="/usr/local/cuda"

for p in "${p_values[@]}"; do
  for N in "${N_values[@]}"; do
    # Base cap: powers of two up to N / (2*p)
    max_b=$(( N / (2 * p) ))

    # For p=4, cap one power-of-two smaller to avoid segfault cases
    if (( p == 4 )); then
      max_b=$(( max_b / 2 ))   # e.g., 64/8=8 -> 4; 128/8=16 -> 8
    fi

    b=1
    while (( b <= max_b )); do
      echo "N_DIM=$N  B_DIM=$b  P_DIM=$p"
      nvcc -ccbin=mpicxx \
       -I"$CUDA_HOME/include" zmodel_fft.cu -o zmodel_fft.x \
       -L"$CUDA_HOME/lib64" -lcufft -lcudart \
       -UP_DIM -DP_DIM="$p" \
       -UN_DIM -DN_DIM="$N" \
       -UB_DIM -DB_DIM="$b"

      # total MPI ranks is p*p (since P_DIM is processors per row/col)
      mpiexec -n $((p*p)) ./zmodel_fft.x
      echo "------------------------------------------------"

      b=$(( b * 2 ))
    done
  done
done