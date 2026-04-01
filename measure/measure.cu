#include <stdio.h>
#include <string.h>
#include <mpi.h>

#define N 100
#define SIZE (N*sizeof(float))

__global__ void VecAdd(float* A, float* B, float* C) {
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

// Check for CUDA aware MPI
int main(int argc, char *argv[]) {
    int myrank, tag=99;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // Host side memory
    float *A_host = (float*)malloc(SIZE);
    float *B_host = (float*)malloc(SIZE);
    float *C_host = (float*)malloc(SIZE);

    for (int i = 0; i < N; i++) {
        if (myrank == 0) {
            A_host[i] = 1.0;
        } else {
            A_host[i] = 0.0;
        }
        B_host[i] = 0.0;
        C_host[i] = 0.0;
    }

    // Device side memory
    float *A_device, *B_device, *C_device;
    cudaMalloc((void**)&A_device, SIZE);
    cudaMalloc((void**)&B_device, SIZE);
    cudaMalloc((void**)&C_device, SIZE);

    // Copy host data to device
    cudaMemcpy(A_device, A_host, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B_host, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(C_device, C_host, SIZE, cudaMemcpyHostToDevice);

    VecAdd<<<1, N>>>(A_device, B_device, C_device);
    cudaDeviceSynchronize();

    if (myrank == 0) {
        // cudaMemcpy(A_host, A_device, SIZE, cudaMemcpyDeviceToHost);
        // MPI_Send(A_host, N, MPI_FLOAT, 1, tag, MPI_COMM_WORLD);
        MPI_Send(A_device, N, MPI_FLOAT, 1, tag, MPI_COMM_WORLD);
    } else {
        MPI_Recv(A_device, N, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &status);
        // MPI_Recv(A_host, N, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &status);
        // cudaMemcpy(A_device, A_host, SIZE, cudaMemcpyHostToDevice);
    }

    VecAdd<<<1, N>>>(A_device, B_device, C_device);
    cudaDeviceSynchronize();

    cudaMemcpy(C_host, C_device, SIZE, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 2; i++) {
        if (myrank == i) {
            for (int j = 0; j < N; j++) {
                std::cout<<C_host[j]<<" ";
            }
            std::cout<<std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}