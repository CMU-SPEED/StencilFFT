#!/bin/bash

p_values=(8 64) # NOTE: this is P_TOTAL = P_DIM^3
N_values=(16 32 64 128 256)

#FIXME: Run heffte make command to get executable heffte3d.x

for p in "${p_values[@]}"; do
  for N in "${N_values[@]}"; do
    echo "N_DIM=$N P_TOTAL=$p"
    mpiexec -n $p ./heffte3d.x $N
    echo "------------------------------------------------"
  done
done
