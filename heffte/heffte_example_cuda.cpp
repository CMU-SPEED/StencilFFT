#include "heffte.h"
#include <chrono>
#include <iostream>
#include "heffte_plan_logic.h"

// Adapted from HeFFTe examples (https://github.com/icl-utk-edu/heffte)                                                 
// Copyright (c) 2020, University of Tennessee. Licensed under BSD 3-Clause License.

/*!
 * \brief HeFFTe example 5, using the cuFFT backend.
 *
 * This example is near identical to the first (fftw) example,
 * the main difference is the use of the cufft backend.
 * The interface and types for the cufft backend work the same,
 * with the exception that the array must sit on the GPU device
 * and if the optional vector interface is used then the vector
 * containers are of type heffte::cuda::vector.
 */
void compute_dft(MPI_Comm comm, int N){
    int me;
    MPI_Comm_rank(comm, &me);

    int num_ranks;
    MPI_Comm_size(comm, &num_ranks);

    long sqrtP = (long)std::sqrt(num_ranks);

    long px = me / sqrtP;
    long py = me % sqrtP;

    long block = N / sqrtP;

    heffte::box3d<> const my_box(
            {px * block, py * block, 0},
            {(px+1)* block - 1, (py+1) * block - 1, 0}
    );

    if (heffte::gpu::device_count() > 1){
        // on a multi-gpu system, distribute the devices across the mpi ranks
        heffte::gpu::device_set(heffte::mpi::comm_rank(comm) % heffte::gpu::device_count());
    }

    // define the heffte class and the input and output geometry
    // heffte::plan_options can be specified just as in the backend::fftw
    //heffte::fft3d<heffte::backend::cufft> fft(my_box, my_box, comm);

    //heffte::plan_options options;
    heffte::plan_options options = heffte::default_options<heffte::backend::cufft>();
    options.algorithm = heffte::reshape_algorithm::alltoall;
    options.use_gpu_aware = true;
    heffte::fft3d<heffte::backend::cufft> fft(my_box, my_box, comm, options);

    // create some input on the CPU
    std::vector<std::complex<double>> input(fft.size_inbox());

    // Initialize the input
    for (long i = 0; i < N / sqrtP; i++) {
        for (long j = 0; j < N / sqrtP; j++) {
            long index = i * (N / sqrtP) + j;
            long offset = ((me /sqrtP) * N * (N/sqrtP)) + ((me%sqrtP) * (N/sqrtP));
            input[index] = {(double)(offset + (i * N) + j), 0.0};
        }
    }

    // load the input into the GPU memory
    // this is equivalent to cudaMalloc() followed by cudaMemcpy()
    // the destructor of heffte::gpu::vector will call cudaFree()
    heffte::gpu::vector<std::complex<double>> gpu_input = heffte::gpu::transfer::load(input);

    // allocate memory on the device for the output
    heffte::gpu::vector<std::complex<double>> gpu_output(fft.size_outbox());

    // allocate scratch space, this is using the public type alias buffer_container
    // and for the cufft backend this is heffte::gpu::vector
    // for the CPU backends (fftw and mkl) the buffer_container is std::vector
    heffte::fft3d<heffte::backend::cufft>::buffer_container<std::complex<double>> workspace(fft.size_workspace());
    static_assert(std::is_same<decltype(gpu_output), decltype(workspace)>::value,
                  "the containers for the output and workspace have different types");

    for (int i = 0; i < 10; i++) {
        MPI_Barrier(MPI_COMM_WORLD);

        auto start = std::chrono::high_resolution_clock::now();

        // perform forward fft using arrays and the user-created workspace
        fft.forward(gpu_input.data(), gpu_output.data(), workspace.data(), heffte::scale::full);

	cudaDeviceSynchronize();

        MPI_Barrier(MPI_COMM_WORLD);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
	//std::cout << "FFT " << duration.count() << " ns, me:" << me << std::endl;
	
	long long time = duration.count();

	long long max_time = 0;
    	MPI_Reduce(&time, &max_time, 1, MPI_LONG_LONG, MPI_MAX, 0, comm);
    	if (me == 0) std::cout << "iter=" << i << " max_time=" << max_time << " ns\n";
    }

    

    // optional step, free the workspace since the inverse will use the vector API
    workspace = heffte::gpu::vector<std::complex<double>>();

    // compute the inverse FFT transform using the container API
    //heffte::gpu::vector<std::complex<double>> gpu_inverse = fft.backward(gpu_output);

    // move the result back to the CPU for comparison purposes
    //std::vector<std::complex<double>> inverse = heffte::gpu::transfer::unload(gpu_inverse);

    // compute the error between the input and the inverse
    //double err = 0.0;
    //for(size_t i=0; i<input.size(); i++)
    //    err = std::max(err, std::abs(inverse[i] - input[i]));

    // print the error for each MPI rank
    //std::cout << std::scientific;
    //for(int i=0; i<num_ranks; i++){
    //    MPI_Barrier(comm);
    //    if (me == i) std::cout << "rank " << i << " computed error: " << err << std::endl;
    //}
}

int main(int argc, char** argv){
    if (argc < 2) {
	std::cout<<"Please provide N, b, and p"<<std::endl;
        return 1;
    }

    int N = atoi(argv[1]);
    
    MPI_Init(NULL, NULL);

    compute_dft(MPI_COMM_WORLD, N);

    MPI_Finalize();

    return 0;
}