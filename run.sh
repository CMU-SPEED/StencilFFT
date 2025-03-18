#!/bin/bash

p_values=(8)
N_values=(64 128 256 512 1024)
b_values=(1 2 4 8 16 32)

for p in "${p_values[@]}"; do
    for N in "${N_values[@]}"; do
        for b in "${b_values[@]}"; do
            echo "mpiexec --mca btl_ofi_provider_exclude psm3 --hostfile hostname -n "$p" ./stencil.x "$N" "$b""
            mpiexec --mca btl_ofi_provider_exclude psm3 --hostfile hostname -n "$p" ./stencil.x "$N" "$b"
            echo "------------------------------------------------"
        done
    done
done