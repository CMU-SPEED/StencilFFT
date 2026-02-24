#include <mpi.h>
#include <stdio.h>
#include <complex>
#include <iostream>
#include <chrono>

using Complex = std::complex<double>;

#define P_DIM (2)
#define LOCAL_DIM (4096)

#define N (LOCAL_DIM*LOCAL_DIM*P_DIM)
#define SIZE (N*sizeof(Complex))

__global__ void VecAdd(Complex* A, Complex* B, Complex* C) {
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

// FIXME: this test will still run all the computation on the same GPU
// Everything is mapped to device 0 by default, see test_fft() for fix
int main(int argc, char *argv[]) {
    int id;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    MPI_Comm row_comm, col_comm;

    int rid = id / P_DIM;
    MPI_Comm_split(MPI_COMM_WORLD, rid, id, &row_comm);

    int cid = id % P_DIM;
    MPI_Comm_split(MPI_COMM_WORLD, cid, id, &col_comm);

    Complex *A_host, *B_host, *C_host;
    cudaHostAlloc((void**)&A_host, SIZE, cudaHostAllocDefault);
    cudaHostAlloc((void**)&B_host, SIZE, cudaHostAllocDefault);
    cudaHostAlloc((void**)&C_host, SIZE, cudaHostAllocDefault);

    for (int i = 0; i < N; i++) {
        A_host[i] = 1.0;
        B_host[i] = 2.0;
        C_host[i] = 3.0;
    }

    Complex *A_device, *B_device, *C_device;
    cudaMalloc((void**)&A_device, SIZE);
    cudaMalloc((void**)&B_device, SIZE);
    cudaMalloc((void**)&C_device, SIZE);
    cudaMemcpy(A_device, A_host, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B_host, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(C_device, B_host, SIZE, cudaMemcpyHostToDevice);

    VecAdd<<<1, 1024>>>(A_device, B_device, C_device);
    cudaDeviceSynchronize();

    for (int i = 0; i < 10; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::high_resolution_clock::now();

        MPI_Alltoall(A_device,
                        (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                        MPI_C_DOUBLE_COMPLEX,
                        B_device,
                        (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                        MPI_C_DOUBLE_COMPLEX,
                        row_comm);

        MPI_Barrier(MPI_COMM_WORLD);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        double ns = duration.count();
        std::cout << "id: " << id << " A2A: " << ns << " ns" << std::endl;
    }
    
    VecAdd<<<1, 1024>>>(A_device, B_device, C_device);
    cudaDeviceSynchronize();

    cudaMemcpy(C_host, C_device, SIZE, cudaMemcpyDeviceToHost);

    std::cout << C_host[0].real() << std::endl;

    cudaFreeHost(A_host);
    cudaFreeHost(B_host);
    cudaFreeHost(C_host);
    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);

    MPI_Finalize();
    return 0;
}