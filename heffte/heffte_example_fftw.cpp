#include "heffte.h"
#include <chrono>

// Adapted from HeFFTe examples (https://github.com/icl-utk-edu/heffte)                                                 
// Copyright (c) 2020, University of Tennessee. Licensed under BSD 3-Clause License.

/*!
 * \brief HeFFTe example 1, simple DFT using two MPI ranks and FFTW backend.
 *
 * Performing DFT on three dimensional data in a box of 4 by 4 by 4 split
 * across the third dimension between two MPI ranks.
 */
void compute_dft(MPI_Comm comm){

    int me; // this process rank within the comm
    MPI_Comm_rank(comm, &me);

    int num_ranks; // total number of ranks in the comm
    MPI_Comm_size(comm, &num_ranks);

    long N = 64;
    long sqrtP = (long)std::sqrt(num_ranks);

    long px = me / sqrtP;
    long py = me % sqrtP;

    long block = N / sqrtP;

    //FIXME: need to update this to work with more processors
    //heffte::box3d<> const boxes[4] = {
    //    {{0, 0, 0}, {31, 31, 0}}, // rank 0: lower left
    //    {{32, 0, 0}, {63, 31, 0}}, // rank 1: lower right
    //    {{0, 32, 0}, {31, 63, 0}}, // rank 2: upper left
    //    {{32, 32, 0}, {63, 63, 0}}, // rank 3: upper right
    //};
    //heffte::box3d<> const boxes[4] = {
    //    {{0, 0, 0}, {N/2 - 1, N/2 - 1, 0}}, // rank 0: lower left
    //    {{N/2, 0, 0}, {N - 1, N/2 - 1, 0}}, // rank 1: lower right
    //    {{0, N/2, 0}, {N / 2 - 1, N - 1, 0}}, // rank 2: upper left
    //    {{N/2, N/2, 0}, {N - 1, N - 1, 0}}, // rank 3: upper right
    //};

    //heffte::box3d<> const boxes[9] = {
    //    {{0, 0, 0}, {N/2 - 1, N/2 - 1, 0}}, // rank 0: lower left
    //    {{N/2, 0, 0}, {N - 1, N/2 - 1, 0}}, // rank 1: lower right
    //    {{0, N/2, 0}, {N / 2 - 1, N - 1, 0}}, // rank 2: upper left
    //    {{N/2, N/2, 0}, {N - 1, N - 1, 0}}, // rank 3: upper right
    //};

    // the box associated with this MPI rank
    //heffte::box3d<> const my_box = boxes[me];
    heffte::box3d<> const my_box(
            {px * block, py * block, 0},
            {(px+1) * block - 1, (py+1) * block - 1, 0}
    );

    // define the heffte class and the input and output geometry
    heffte::fft3d<heffte::backend::fftw> fft(my_box, my_box, comm);

    // vectors with the correct sizes to store the input and output data
    // taking the size of the input and output boxes
    std::vector<std::complex<double>> input(fft.size_inbox());
    std::vector<std::complex<double>> output(fft.size_outbox());
    std::vector<std::complex<double>> reference(fft.size_outbox());

    // FIXME: need to update this to work with more processors
    for (long i = 0; i < N / sqrtP; i++) {
        for (long j = 0; j < N / sqrtP; j++) {
            long index = i * (N / sqrtP) + j;
            long offset = ((me/sqrtP) * N * (N/sqrtP)) + ((me%sqrtP) * (N/sqrtP));
            input[index] = {(double)(offset + (i * N) + j), 0.0};
            reference[index] = {(double)(offset + (i * N) + j), 0.0};
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();

    // perform a forward DFT
    fft.forward(input.data(), output.data());

    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    if (me == 0) std::cout << "FFT: " << duration.count() << " ns" << std::endl;

    // check the accuracy
    //if (me == 1){
        // given the data, the solution on MPI rank 1 is very simple
    //    std::vector<std::complex<double>> reference_output(fft.size_outbox());
    //    reference_output[0] = {-512.0, 0.0};

        // compare the computed to the actual and throw error if there is a mismatch
    //    for(int i=0; i<fft.size_outbox(); i++)
    //        if (std::abs(reference_output[i] - output[i]) > 1.E-14)
    //            throw std::runtime_error("discrepancy between the reference and actual output");
    //}

    if (me == 0) {
        std::cout << output[0] << std::endl;
    }

    // reset the input to zero
    std::fill(input.begin(), input.end(), std::complex<double>(0.0, 0.0));

    // perform a backward DFT
    fft.backward(output.data(), input.data());

    // rescale the result
    //for(auto &i : input) i /= 64.0;
    //for(auto &i : input) i /= 16.0;
    for(auto &i : input) i /= N * N;

    // compare the computed entries to the original input data
    double err = 0.0;
    for(int i=0; i<fft.size_inbox(); i++)
        err = std::max(err, std::abs(input[i] - reference[i]));

    // print the error for each MPI rank
    if (me == 1) std::cout << "rank 1 computed error: " << err << std::endl;
    MPI_Barrier(comm);
    std::cout << "Rank " << me << " max error = " << err << std::endl;
}

int main(int argc, char** argv){

    MPI_Init(&argc, &argv);

    compute_dft(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}