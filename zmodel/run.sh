#!/bin/bash

p_values=(4)
N_values=(64 128 256 512 1024)
b_values=(1 2 4 8 16 32)

CUDA_HOME = "/usr/local/cuda"

for p in "${p_values[@]}"; do
    for N in "${N_values[@]}"; do
        for b in "${b_values[@]}"; do
            echo $p $N $b 
            nvcc -ccbin=mpicxx -I$CUDA_HOME/include zmodel_fft.cu -o zmodel_fft.x -L$CUDA_HOME/lib64 -lcufft -lcudart -UP_DIM -DP_DIM=$p -UN_DIM -DN_DIM=$N -UB_DIM -DB_DIM=$b
            mpiexec -n 4 ./zmodel_fft.x
            echo "------------------------------------------------"
        done
    done
done