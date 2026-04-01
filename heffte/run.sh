#!/bin/bash

p_values=(4 16) # NOTE: this only use P_TOTAl not P_DIM
N_values=(64 128 256 512 1024)

#FIXME: Run heffte make command to get executable heffte.x

for p in "${p_values[@]}"; do
  for N in "${N_values[@]}"; do
    echo "N_DIM=$N P_TOTAL=$p"
    mpiexec -n $p ./heffte.x $N
    echo "------------------------------------------------"
  done
done