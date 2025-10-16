#include <mpi.h>
#include <iostream>
#include <vector>
#include <chrono>

#define N 1000
#define ITER 100
#define REPEAT 1000

int main(int argc, char *argv[]) {
    int tag = 99;
    int myrank, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    std::vector<float> buf(N * ITER, 42.0f);

    // Test baseline size
    // Warm up
    if (myrank == 0) {
        MPI_Send(buf.data(), 1, MPI_FLOAT, 1, tag, MPI_COMM_WORLD);
        MPI_Recv(buf.data(), 1, MPI_FLOAT, 1, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        MPI_Recv(buf.data(), 1, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(buf.data(), 1, MPI_FLOAT, 0, tag, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Benchmark
    long long total_ns = 0;
    for (int r = 0; r < REPEAT; r++) {
        if (myrank == 0) {
            auto start = std::chrono::high_resolution_clock::now();
            MPI_Send(buf.data(), 1, MPI_FLOAT, 1, tag, MPI_COMM_WORLD);
            MPI_Recv(buf.data(), 1, MPI_FLOAT, 1, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            auto end = std::chrono::high_resolution_clock::now();
            total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        } else {
            MPI_Recv(buf.data(), 1, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(buf.data(), 1, MPI_FLOAT, 0, tag, MPI_COMM_WORLD);
        }
    }

    if (myrank == 0) {
        double latency_ns = (double)total_ns / REPEAT / 2.0;
        std::cout << "Size " << 4 << " bytes: "
                  << latency_ns << " ns (avg)" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Test multiple data sizes
    for (int i = 1; i <= ITER; i++) {
        int count = i * N;
        int bytes = count * sizeof(float);

        // Warm up
        if (myrank == 0) {
            MPI_Send(buf.data(), count, MPI_FLOAT, 1, tag, MPI_COMM_WORLD);
            MPI_Recv(buf.data(), count, MPI_FLOAT, 1, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(buf.data(), count, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(buf.data(), count, MPI_FLOAT, 0, tag, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // Benchmark
        long long total_ns = 0;
        for (int r = 0; r < REPEAT; r++) {
            if (myrank == 0) {
                auto start = std::chrono::high_resolution_clock::now();
                MPI_Send(buf.data(), count, MPI_FLOAT, 1, tag, MPI_COMM_WORLD);
                MPI_Recv(buf.data(), count, MPI_FLOAT, 1, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                auto end = std::chrono::high_resolution_clock::now();
                total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            } else {
                MPI_Recv(buf.data(), count, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buf.data(), count, MPI_FLOAT, 0, tag, MPI_COMM_WORLD);
            }
        }

        if (myrank == 0) {
            double latency_ns = (double)total_ns / REPEAT / 2.0;
            std::cout << "Size " << bytes << " bytes: "
                      << latency_ns << " ns (avg)" << std::endl;
            // std::cout << bytes << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
