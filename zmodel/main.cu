#include "./transforms/fft.h"
#include "./stencils/laplace.h"

// End to end 2D example.  
// To run 3D basically just add _3d to every function and change data transfer sizes
// Ensure the Makefile launches with the correct number of processors, this is set independently from the P_DIM Macro
// Run with __ZMODEL__INCLUDE__INVERSE__ 
void example(void) {
    // Set to false to only do data movement, helpful for debug
    constexpr bool do_compute = true;
    constexpr bool include_inverse = true;

    // Initialize 2D communicators
    MPI_Init(NULL, NULL);

    int P, id;
    P = P_DIM * P_DIM;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    MPI_Comm row_comm, col_comm;

    int rid = id / P_DIM;
    MPI_Comm_split(MPI_COMM_WORLD, rid, id, &row_comm);

    int cid = id % P_DIM;
    MPI_Comm_split(MPI_COMM_WORLD, cid, id, &col_comm);

    // Initialize Laplace data structures
    LaplaceBuffers laplace_buffers;
    init_laplace_buffers(laplace_buffers);

    // Initialize FFT data structures
    FftHostPlans host_plans;
    init_plans(host_plans);

    // Toggle this Macro to enable fast device side kernel for plan1 (need cufftdx installed)
    FftdxFft1Ctx ctx1;
    #ifdef __USE__FFTDX__
        init_fftdx_fft1(ctx1);
    #endif

    FftBuffers buffers;
    init_buffers<include_inverse>(buffers, rid, cid);

    // This loads test data into the host, replace this with the hdf5 version in utils.cu
    init_host(rid, cid, buffers.host_buf0);
    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_buf0, buffers.host_buf0, LOCAL_COMPLEX_BYTES, MEM_COPY_HOST_TO_DEVICE));

    // Laplace Routine
    pack_laplace<<<LOCAL_DIM, LOCAL_DIM>>>(buffers.device_buf0, laplace_buffers);
    cudaDeviceSynchronize();

    communicate_laplace(laplace_buffers, row_comm, col_comm, rid, cid);
    MPI_Barrier(MPI_COMM_WORLD);

    laplace<<<VEC_TOTAL, VEC>>>(buffers.device_buf0, buffers.device_buf1, laplace_buffers, rid, cid);
    cudaDeviceSynchronize();

    // Copy result back to buf0 because FFT required buf0 to be input
    // This can be avoided by swapping the pointers but this is done for clarity
    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_buf0, buffers.device_buf1, LOCAL_COMPLEX_BYTES, MEM_COPY_DEVICE_TO_DEVICE));

    forward_fft<do_compute>(buffers, row_comm, col_comm, id, host_plans, ctx1);

    // The distributed FFT permutes the output relative to a standard DFT.
    // reciprocal_to_global maps a local position in the FFT output to the
    // corresponding global frequency index: buf1[local_row * L + local_col]
    // on processor (rid, cid) holds the same value as numpy.fft.fft2(data)[ky][kx].
    // int ky, kx;
    // reciprocal_to_global(rid, cid, local_row, local_col, ky, kx);

    // FFT uses buf1 as output.  IFFT uses buf0 as input
    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_buf0, buffers.device_buf1, LOCAL_COMPLEX_BYTES, MEM_COPY_DEVICE_TO_DEVICE));

    // IFFT normalization factor
    double s = 1.0 / (double(N_DIM) * double(N_DIM));
    Complex scale = DEVICE_FFT_DOUBLECOMPLEX_CONSTRUCTOR(s, 0.0);

    inverse_fft<do_compute>(buffers, row_comm, col_comm, id, scale, host_plans);

    // Copy results back to host
    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.host_buf1, buffers.device_buf1, LOCAL_COMPLEX_BYTES, MEM_COPY_DEVICE_TO_HOST));

    #ifdef __PRINT__RESULTS__
        MPI_Barrier(MPI_COMM_WORLD);
        if (id == 0) {
            std::cout << "Host out" << std::endl;
            for (int i = 0; i < LOCAL_DIM; i++) {
                for (int j = 0; j < LOCAL_DIM; j++) {
                    std::cout << buffers.host_buf1[i * LOCAL_DIM + j].x << " ";
                }
                std::cout << std::endl;
            }
        }
    #endif

    destroy_plans(host_plans);
    destroy_buffers<include_inverse>(buffers);

    MPI_Finalize();
}

int main(void) {
    // test_fft();
    // test_fft_3d();
    // test_laplace();
    test_laplace_3d();
    // example();
    return 0;
}