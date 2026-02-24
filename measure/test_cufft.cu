#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cufft.h>

int main() {
    constexpr int N = 128;
    constexpr int BATCH = 1;

    cufftDoubleComplex* data = nullptr;
    size_t size_bytes = sizeof(cufftDoubleComplex) * N * BATCH;

    cudaMallocManaged(&data, size_bytes);

    for (int i = 0; i < N * BATCH; i++) {
        data[i].x = double(i);
        data[i].y = -double(i);
    }

    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_Z2Z, BATCH);

    for (int i = 0; i < 5; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        for (int j = 0; j < 1000; j++) {
            cufftExecZ2Z(plan, data, data, CUFFT_FORWARD);
        }
        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        std::cout << "TOTAL " << duration.count() << " ns\n" << std::endl;
    }

    for (int i = 0; i < N; i++) {
        std::cout << "(" << data[i].x << ", " << data[i].y << ") ";
    }
    std::cout << "\n";

    cufftDestroy(plan);
    cudaFree(data);

    return 0;
}
