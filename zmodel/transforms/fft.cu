#include "fft.h"
#include <chrono>
#include <complex>
#include <fstream>
#include "fft_host_plans.h"

__global__ void pack_forward_all_to_all_row(Complex *device_in, Complex *device_out) {
    const int row = blockIdx.x / VEC_COL;
    const int col = blockIdx.x % VEC_COL;
    const int in_offset = (row * LOCAL_DIM) + (col * VEC);

    const int vec_index = threadIdx.x;                      // Index into the vector

    const int vec_split = VEC / P_DIM;                      // The size of a split vector
    const int split = vec_index / vec_split;                // Which section of the split vector
    const int index_in_split = vec_index % vec_split;       // Index into the split vector

    // Note these are both measured in element space
    const int slab_size = (LOCAL_DIM*LOCAL_DIM) / P_DIM;    // Num elements in a slab of splits
    const int slab_vec_row = LOCAL_DIM / P_DIM;             // Number of split vectors in a col of a slab
    const int slab_vec_col = LOCAL_DIM / vec_split;         // Number of split vectors in a row of a slab/local

    const int slab_offset = split * slab_size;
    const int out_row = (blockIdx.x / slab_vec_col) % slab_vec_row;
    const int out_col = blockIdx.x % slab_vec_col;
    const int out_offset = slab_offset + (out_row * LOCAL_DIM) + (out_col * vec_split);

    device_out[out_offset + index_in_split] = device_in[in_offset + vec_index];
}

template<bool do_compute>
__global__ void pack_forward_fft0(Complex *device_in, Complex *device_out, Complex *forward_twiddles0) {
    const int split_size = LOCAL_DIM / P_DIM;                // The size of the split

    const int index_in_row_split = blockIdx.x % split_size;
    const int index_in_vec_split = threadIdx.x % split_size;
    
    const int row_split = blockIdx.x / split_size;          // Which section of the row split
    const int vec_split = threadIdx.x / split_size;         // Which section of the split local vector

    // Swap the vec and row splits
    const int row = vec_split + (index_in_row_split * P_DIM);
    const int col = (row_split * split_size) + index_in_vec_split;

    if constexpr (do_compute) {
        device_out[row * LOCAL_DIM + col] = DEVICE_FFT_DOUBLECOMPLEX_MUL(device_in[blockIdx.x * LOCAL_DIM + threadIdx.x],
                                                                         forward_twiddles0[row * LOCAL_DIM + col]);
    } else {
        device_out[row * LOCAL_DIM + col] = device_in[blockIdx.x * LOCAL_DIM + threadIdx.x];
    }
}

template<bool do_compute>
__global__ void pack_forward_fft1(Complex *device_in, Complex *device_out, Complex *forward_twiddles1) {
    const int split_size = LOCAL_DIM / P_DIM;

    const int which_split = blockIdx.x / split_size;
    const int index_in_split = blockIdx.x % split_size;

    const int which_block = index_in_split / B_DIM;
    const int index_in_block = index_in_split % B_DIM;

    const int row = (which_block * P_DIM * B_DIM) + (which_split * B_DIM) + index_in_block;
    
    if constexpr (do_compute) {
        device_out[row * LOCAL_DIM + threadIdx.x] = DEVICE_FFT_DOUBLECOMPLEX_MUL(device_in[blockIdx.x * LOCAL_DIM + threadIdx.x],
                                                                            forward_twiddles1[row * LOCAL_DIM + threadIdx.x]);
    } else {
        device_out[row * LOCAL_DIM + threadIdx.x] = device_in[blockIdx.x * LOCAL_DIM + threadIdx.x];
    }
}

void init_forward_twiddles0(Complex *in, int cid) {
    int stride = VEC/P_DIM; // This is the processor offset
    for (int local_row = 0; local_row < LOCAL_DIM; local_row++) {
        for (int fft_row = 0; fft_row < stride; fft_row++) {
            for (int fft_col = 0; fft_col < LOCAL_DIM/stride; fft_col++) {
                int fft_row_offset = cid * stride;
                double k = (double)(fft_row + fft_row_offset);
                double l = (double)fft_col;
                #ifdef __PRINT__TWIDDLES__
                    in[(local_row * LOCAL_DIM) + fft_row + (fft_col * stride)] = {k, l};
                #else
                    std::complex<double> twiddle = std::exp(std::complex<double>(0.0, -2*M_PI*k*l/N_DIM));
                    in[(local_row * LOCAL_DIM) + fft_row + (fft_col * stride)] = DEVICE_FFT_DOUBLECOMPLEX_CONSTRUCTOR(twiddle.real(), twiddle.imag());
                #endif
            }
        }
    }
}

void init_forward_twiddles1(Complex *in, int rid) {
    int stride = (N_DIM/(B_DIM*P_DIM))/P_DIM; // This is the processor offset
    for (int fft_row = 0; fft_row < (LOCAL_DIM*LOCAL_DIM) / (LOCAL_DIM*B_DIM*P_DIM); fft_row++) {
        for (int fft_col = 0; fft_col < B_DIM*P_DIM; fft_col++) {
            for (int local_col = 0; local_col < LOCAL_DIM; local_col++) {
                int fft_row_offset = rid * stride;
                double k = (double)(fft_row + fft_row_offset);
                double l = (double)fft_col;
                #ifdef __PRINT__TWIDDLES__
                    in[fft_row * (LOCAL_DIM*B_DIM*P_DIM) + fft_col * LOCAL_DIM + local_col] = {k, l};
                #else
                    std::complex<double> twiddle = std::exp(std::complex<double>(0.0, -2*M_PI*k*l/N_DIM));
                    in[fft_row * (LOCAL_DIM*B_DIM*P_DIM) + fft_col * LOCAL_DIM + local_col] = DEVICE_FFT_DOUBLECOMPLEX_CONSTRUCTOR(twiddle.real(), twiddle.imag());
                #endif
            }
        }
    }
}

// NOTE: This is used for debugging --> can comment out FFTs and replace with this
__global__ void placeholder(Complex *device_in, Complex *device_out) {
    device_out[blockIdx.x * LOCAL_DIM + threadIdx.x] = device_in[blockIdx.x * LOCAL_DIM + threadIdx.x];
}

/**
 * @brief 
 * Data starts in device_buf0
 * Data ends in device_buf1
 */
template<bool do_compute>
void forward_fft(Complex *host_buf0, Complex *host_buf1, Complex *device_buf0, Complex *device_buf1,
                 Complex *device_twiddles0, Complex *device_twiddles1, MPI_Comm *row_comm, MPI_Comm *col_comm,
                 cufftHandle *plan0,  Vector<cufftHandle> &plan1, cufftHandle *plan2,  Vector<cufftHandle> &plan3,
                 int id, Vector<cudaStream_t> &streams, FftdxFft1Ctx& ctx1) {
    #ifdef __PRINT__TIMING__
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::high_resolution_clock::now();
    #endif // __PRINT__TIMING__

    // #ifdef __PRINT__DETAILED__TIMING__
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     auto start_fft0 = std::chrono::high_resolution_clock::now();
    // #endif // __PRINT__DETAILED__TIMING__

    // if constexpr (do_compute) {
    //     DEVICE_FFT_SAFE_CALL(DEVICE_FFT_EXECZ2Z(*plan0, device_buf0, device_buf1, DEVICE_FFT_FORWARD));
    //     DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
    // } else {
    //     placeholder<<<LOCAL_DIM, LOCAL_DIM>>>(device_buf0, device_buf1);        
    // }

    // #ifdef __PRINT__DETAILED__TIMING__
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     auto end_fft0 = std::chrono::high_resolution_clock::now();
    //     auto duration_fft0 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_fft0 - start_fft0); 
    //     long long ns_fft0 = duration_fft0.count(); 
    //     long long max_time_fft0 = 0; 
    //     MPI_Reduce(&ns_fft0, &max_time_fft0, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD); 
    //     if (id == 0) std::cout << "FFT0 " << max_time_fft0 << " ns" << std::endl;
    // #endif // __PRINT__DETAILED__TIMING__

    // #ifdef __PRINT__DETAILED__TIMING__
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     auto start_pack0 = std::chrono::high_resolution_clock::now();
    // #endif // __PRINT__DETAILED__TIMING__

    // pack_forward_all_to_all_row<<<VEC_TOTAL, VEC>>>(device_buf1, device_buf0);
    // DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());

    // #ifdef __PRINT__DETAILED__TIMING__
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     auto end_pack0 = std::chrono::high_resolution_clock::now();
    //     auto duration_pack0 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_pack0 - start_pack0); 
    //     long long ns_pack0 = duration_pack0.count(); 
    //     long long max_time_pack0 = 0; 
    //     MPI_Reduce(&ns_pack0, &max_time_pack0, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD); 
    //     if (id == 0) std::cout << "PACK0 " << max_time_pack0 << " ns" << std::endl;
    // #endif // __PRINT__DETAILED__TIMING__

    // #ifdef __PRINT__DETAILED__TIMING__
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     auto start_comms0 = std::chrono::high_resolution_clock::now();
    // #endif // __PRINT__DETAILED__TIMING__

    // #ifdef __GPU__AWARE__MPI__
    //     MPI_Alltoall(device_buf0,
    //                 (LOCAL_DIM*LOCAL_DIM)/P_DIM,
    //                 MPI_C_DOUBLE_COMPLEX,
    //                 device_buf1,
    //                 (LOCAL_DIM*LOCAL_DIM)/P_DIM,
    //                 MPI_C_DOUBLE_COMPLEX,
    //                 *row_comm);
    // #else
    //     DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(host_buf0, device_buf0, LOCAL_COMPLEX_BYTES, MEM_COPY_DEVICE_TO_HOST));
    //     MPI_Alltoall(host_buf0,
    //                 (LOCAL_DIM*LOCAL_DIM)/P_DIM,
    //                 MPI_C_DOUBLE_COMPLEX,
    //                 host_buf1, 
    //                 (LOCAL_DIM*LOCAL_DIM)/P_DIM,
    //                 MPI_C_DOUBLE_COMPLEX,
    //                 *row_comm);
    //     DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(device_buf1, host_buf1, LOCAL_COMPLEX_BYTES, MEM_COPY_HOST_TO_DEVICE));
    // #endif // __GPU__AWARE__MPI__

    // #ifdef __PRINT__DETAILED__TIMING__
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     auto end_comms0 = std::chrono::high_resolution_clock::now();
    //     auto duration_comms0 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_comms0 - start_comms0); 
    //     long long ns_comms0 = duration_comms0.count(); 
    //     long long max_time_comms0 = 0; 
    //     MPI_Reduce(&ns_comms0, &max_time_comms0, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD); 
    //     if (id == 0) std::cout << "COMMS0 " << max_time_comms0 << " ns" << std::endl;
    // #endif // __PRINT__DETAILED__TIMING__

    // #ifdef __PRINT__DETAILED__TIMING__
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     auto start_pack1 = std::chrono::high_resolution_clock::now();
    // #endif // __PRINT__DETAILED__TIMING__
    
    // pack_forward_fft0<do_compute><<<LOCAL_DIM, LOCAL_DIM>>>(device_buf1, device_buf0, device_twiddles0);
    // DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());

    // #ifdef __PRINT__DETAILED__TIMING__
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     auto end_pack1 = std::chrono::high_resolution_clock::now();
    //     auto duration_pack1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_pack1 - start_pack1); 
    //     long long ns_pack1 = duration_pack1.count(); 
    //     long long max_time_pack1 = 0; 
    //     MPI_Reduce(&ns_pack1, &max_time_pack1, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD); 
    //     if (id == 0) std::cout << "PACK1 " << max_time_pack1 << " ns" << std::endl;
    // #endif // __PRINT__DETAILED__TIMING__

    // #ifdef __PRINT__DETAILED__TIMING__
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     auto start_fft1 = std::chrono::high_resolution_clock::now();
    // #endif // __PRINT__DETAILED__TIMING__

    if constexpr (do_compute) {
        #ifdef __USE__FFTDX__
            fft1_row_grouped_cufftdx_kernel<typename FftdxFft1Ctx::FFT, FftdxFft1Ctx::VEC_SPLIT>
            <<<ctx1.grid, ctx1.block, ctx1.shmem>>>(device_buf0, device_buf1, ctx1.ws);
        #else
            for (int row = 0; row < LOCAL_DIM; row++) {
                Complex* row_ptr0 = device_buf0 + row * LOCAL_DIM;
                Complex* row_ptr1 = device_buf1 + row * LOCAL_DIM;
                DEVICE_FFT_SAFE_CALL(DEVICE_FFT_EXECZ2Z(plan1[row % NUM_STREAMS], row_ptr0, row_ptr1, DEVICE_FFT_FORWARD));
            }
        #endif
        DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
    } else {
        placeholder<<<LOCAL_DIM, LOCAL_DIM>>>(device_buf0, device_buf1);
    }

    // #ifdef __PRINT__DETAILED__TIMING__
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     auto end_fft1 = std::chrono::high_resolution_clock::now();
    //     auto duration_fft1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_fft1 - start_fft1); 
    //     long long ns_fft1 = duration_fft1.count(); 
    //     long long max_time_fft1 = 0; 
    //     MPI_Reduce(&ns_fft1, &max_time_fft1, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    //     if (id == 0) std::cout << "FFT1 " << max_time_fft1 << " ns" << std::endl;
    // #endif // __PRINT__DETAILED__TIMING__

    // #ifdef __PRINT__DETAILED__TIMING__
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     auto start_fft2 = std::chrono::high_resolution_clock::now();
    // #endif // __PRINT__DETAILED__TIMING__

    // if constexpr (do_compute) {
    //     DEVICE_FFT_SAFE_CALL(DEVICE_FFT_EXECZ2Z(*plan2, device_buf1, device_buf0, DEVICE_FFT_FORWARD));
    //     DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
    // } else {
    //     placeholder<<<LOCAL_DIM, LOCAL_DIM>>>(device_buf1, device_buf0);
    // }

    // #ifdef __PRINT__DETAILED__TIMING__
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     auto end_fft2 = std::chrono::high_resolution_clock::now();
    //     auto duration_fft2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_fft2 - start_fft2); 
    //     long long ns_fft2 = duration_fft2.count(); 
    //     long long max_time_fft2 = 0; 
    //     MPI_Reduce(&ns_fft2, &max_time_fft2, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    //     if (id == 0) std::cout << "FFT2 " << max_time_fft2 << " ns" << std::endl;
    // #endif // __PRINT__DETAILED__TIMING__

    // #ifdef __PRINT__DETAILED__TIMING__
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     auto start_comms1 = std::chrono::high_resolution_clock::now();
    // #endif // __PRINT__DETAILED__TIMING__

    // #ifdef __GPU__AWARE__MPI__
    //     MPI_Alltoall(device_buf0,
    //                 (LOCAL_DIM*LOCAL_DIM)/P_DIM,
    //                 MPI_C_DOUBLE_COMPLEX,
    //                 device_buf1,
    //                 (LOCAL_DIM*LOCAL_DIM)/P_DIM,
    //                 MPI_C_DOUBLE_COMPLEX,
    //                 *col_comm);
    // #else
    //     DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(host_buf0, device_buf0, LOCAL_COMPLEX_BYTES, MEM_COPY_DEVICE_TO_HOST));
    //     MPI_Alltoall(host_buf0,
    //                 (LOCAL_DIM*LOCAL_DIM)/P_DIM,
    //                 MPI_C_DOUBLE_COMPLEX,
    //                 host_buf1,
    //                 (LOCAL_DIM*LOCAL_DIM)/P_DIM,
    //                 MPI_C_DOUBLE_COMPLEX,
    //                 *col_comm);
    //     DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(device_buf1, host_buf1, LOCAL_COMPLEX_BYTES, MEM_COPY_HOST_TO_DEVICE));
    // #endif // __GPU__AWARE__MPI__

    // #ifdef __PRINT__DETAILED__TIMING__
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     auto end_comms1 = std::chrono::high_resolution_clock::now();
    //     auto duration_comms1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_comms1 - start_comms1);
    //     long long ns_comms1 = duration_comms1.count();
    //     long long max_time_comms1 = 0;
    //     MPI_Reduce(&ns_comms1, &max_time_comms1, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    //     if (id == 0) std::cout << "COMMS1 " << max_time_comms1 << " ns" << std::endl;
    // #endif // __PRINT__DETAILED__TIMING__

    // #ifdef __PRINT__DETAILED__TIMING__
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     auto start_pack2 = std::chrono::high_resolution_clock::now();
    // #endif // __PRINT__DETAILED__TIMING__

    // pack_forward_fft1<do_compute><<<LOCAL_DIM, LOCAL_DIM>>>(device_buf1, device_buf0, device_twiddles1);
    // DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());

    // #ifdef __PRINT__DETAILED__TIMING__
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     auto end_pack2 = std::chrono::high_resolution_clock::now();
    //     auto duration_pack2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_pack2 - start_pack2); 
    //     long long ns_pack2 = duration_pack2.count(); 
    //     long long max_time_pack2 = 0; 
    //     MPI_Reduce(&ns_pack2, &max_time_pack2, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    //     if (id == 0) std::cout << "PACK2 " << max_time_pack2 << " ns" << std::endl; 
    // #endif // __PRINT__DETAILED__TIMING__

    // #ifdef __PRINT__DETAILED__TIMING__
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     auto start_fft3 = std::chrono::high_resolution_clock::now();
    // #endif // __PRINT__DETAILED__TIMING__

    // if constexpr (do_compute) {
    //     for (int row = 0; row < LOCAL_DIM/(P_DIM*B_DIM); row++) {
    //         Complex* row_ptr0 = device_buf0 + row * LOCAL_DIM * P_DIM * B_DIM;
    //         Complex* row_ptr1 = device_buf1 + row * LOCAL_DIM * P_DIM * B_DIM;
    //         DEVICE_FFT_SAFE_CALL(DEVICE_FFT_EXECZ2Z(plan3[row % NUM_STREAMS], row_ptr0, row_ptr1, DEVICE_FFT_FORWARD));
    //     }
    //     DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
    // } else {
    //     placeholder<<<LOCAL_DIM, LOCAL_DIM>>>(device_buf0, device_buf1);
    // }

    // #ifdef __PRINT__DETAILED__TIMING__
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     auto end_fft3 = std::chrono::high_resolution_clock::now();
    //     auto duration_fft3 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_fft3 - start_fft3); 
    //     long long ns_fft3 = duration_fft3.count();
    //     long long max_time_fft3 = 0; 
    //     MPI_Reduce(&ns_fft3, &max_time_fft3, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    //     if (id == 0) std::cout << "FFT3 " << max_time_fft3 << " ns" << std::endl;
    // #endif // __PRINT__DETAILED__TIMING__

    #ifdef __PRINT__TIMING__
        MPI_Barrier(MPI_COMM_WORLD);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        long long ns = duration.count();
        long long max_time = 0;
        MPI_Reduce(&ns, &max_time, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
        if (id == 0) std::cout << "TOTAL " << max_time << " ns\n" << std::endl;
    #endif // __PRINT__TIMING__
}

template void forward_fft<false>(Complex*, Complex*, Complex*, Complex*,
                                 Complex*, Complex*, MPI_Comm*, MPI_Comm*,
                                 cufftHandle*, Vector<cufftHandle>&, cufftHandle*, Vector<cufftHandle>&, 
                                 int, Vector<cudaStream_t>&, FftdxFft1Ctx&);

template void forward_fft<true>(Complex*, Complex*, Complex*, Complex*,
                                Complex*, Complex*, MPI_Comm*, MPI_Comm*,
                                cufftHandle*, Vector<cufftHandle>&, cufftHandle*, Vector<cufftHandle>&,
                                int, Vector<cudaStream_t>&, FftdxFft1Ctx&);

template<bool do_compute>
__global__ void pack_inverse_all_to_all_row(Complex *device_in, Complex *device_out, Complex scale) {
    const int row = blockIdx.x / VEC_COL;
    const int col = blockIdx.x % VEC_COL;
    const int in_offset = (row * LOCAL_DIM) + (col * VEC);

    const int vec_index = threadIdx.x;                      // Index into the vector

    const int vec_split = VEC / P_DIM;                      // The size of a split vector
    const int split = vec_index / vec_split;                // Which section of the split vector
    const int index_in_split = vec_index % vec_split;       // Index into the split vector

    // Note these are both measured in element space
    const int slab_size = (LOCAL_DIM*LOCAL_DIM) / P_DIM;    // Num elements in a slab of splits
    const int slab_vec_row = LOCAL_DIM / P_DIM;             // Number of split vectors in a col of a slab
    const int slab_vec_col = LOCAL_DIM / vec_split;         // Number of split vectors in a row of a slab/local

    const int slab_offset = split * slab_size;
    const int out_row = (blockIdx.x / slab_vec_col) % slab_vec_row;
    const int out_col = blockIdx.x % slab_vec_col;
    const int out_offset = slab_offset + (out_row * LOCAL_DIM) + (out_col * vec_split);

    if constexpr (do_compute) {
        device_out[in_offset + vec_index] = DEVICE_FFT_DOUBLECOMPLEX_MUL(device_in[out_offset + index_in_split], scale);
    } else {
        device_out[in_offset + vec_index] = device_in[out_offset + index_in_split];
    }
}

template<bool do_compute>
__global__ void pack_inverse_fft0(Complex *device_in, Complex *device_out, Complex *inverse_twiddles0) {
    const int split_size = LOCAL_DIM / P_DIM;                // The size of the split

    const int index_in_row_split = blockIdx.x % split_size;
    const int index_in_vec_split = threadIdx.x % split_size;
    
    const int row_split = blockIdx.x / split_size;          // Which section of the row split
    const int vec_split = threadIdx.x / split_size;         // Which section of the split local vector

    // Swap the vec and row splits
    const int row = vec_split + (index_in_row_split * P_DIM);
    const int col = (row_split * split_size) + index_in_vec_split;

    if constexpr (do_compute) {
        device_out[blockIdx.x * LOCAL_DIM + threadIdx.x] = DEVICE_FFT_DOUBLECOMPLEX_MUL(device_in[row * LOCAL_DIM + col],
                                                                         inverse_twiddles0[row * LOCAL_DIM + col]);
    } else {
        device_out[blockIdx.x * LOCAL_DIM + threadIdx.x] = device_in[row * LOCAL_DIM + col];
    }
}

template<bool do_compute>
__global__ void pack_inverse_fft1(Complex *device_in, Complex *device_out, Complex *inverse_twiddles1) {
    const int split_size = LOCAL_DIM / P_DIM;

    const int which_split = blockIdx.x / split_size;
    const int index_in_split = blockIdx.x % split_size;

    const int which_block = index_in_split / B_DIM;
    const int index_in_block = index_in_split % B_DIM;

    const int row = (which_block * P_DIM * B_DIM) + (which_split * B_DIM) + index_in_block;

    if constexpr (do_compute) {
        device_out[blockIdx.x * LOCAL_DIM + threadIdx.x] = DEVICE_FFT_DOUBLECOMPLEX_MUL(device_in[row * LOCAL_DIM + threadIdx.x],
                                                                                    inverse_twiddles1[row * LOCAL_DIM + threadIdx.x]);
    } else {
        device_out[blockIdx.x * LOCAL_DIM + threadIdx.x] = device_in[row * LOCAL_DIM + threadIdx.x];
    }
}

void init_inverse_twiddles0(Complex *in, int cid) {
    int stride = VEC/P_DIM; // This is the processor offset
    for (int local_row = 0; local_row < LOCAL_DIM; local_row++) {
        for (int fft_row = 0; fft_row < stride; fft_row++) {
            for (int fft_col = 0; fft_col < LOCAL_DIM/stride; fft_col++) {
                int fft_row_offset = cid * stride;
                double k = (double)(fft_row + fft_row_offset);
                double l = (double)fft_col;
                #ifdef __PRINT__TWIDDLES__
                    in[(local_row * LOCAL_DIM) + fft_row + (fft_col * stride)] = {k, l};
                #else
                    std::complex<double> twiddle = std::exp(std::complex<double>(0.0, 2*M_PI*k*l/N_DIM));
                    in[(local_row * LOCAL_DIM) + fft_row + (fft_col * stride)] = DEVICE_FFT_DOUBLECOMPLEX_CONSTRUCTOR(twiddle.real(), twiddle.imag());
                #endif
            }
        }
    }
}

void init_inverse_twiddles1(Complex *in, int rid) {
    int stride = (N_DIM/(B_DIM*P_DIM))/P_DIM; // This is the processor offset
    for (int fft_row = 0; fft_row < (LOCAL_DIM*LOCAL_DIM) / (LOCAL_DIM*B_DIM*P_DIM); fft_row++) {
        for (int fft_col = 0; fft_col < B_DIM*P_DIM; fft_col++) {
            for (int local_col = 0; local_col < LOCAL_DIM; local_col++) {
                int fft_row_offset = rid * stride;
                double k = (double)(fft_row + fft_row_offset);
                double l = (double)fft_col;
                #ifdef __PRINT__TWIDDLES__
                    in[fft_row * (LOCAL_DIM*B_DIM*P_DIM) + fft_col * LOCAL_DIM + local_col] = {k, l};
                #else
                    std::complex<double> twiddle = std::exp(std::complex<double>(0.0, 2*M_PI*k*l/N_DIM));
                    in[fft_row * (LOCAL_DIM*B_DIM*P_DIM) + fft_col * LOCAL_DIM + local_col] = DEVICE_FFT_DOUBLECOMPLEX_CONSTRUCTOR(twiddle.real(), twiddle.imag());
                #endif
            }
        }
    }
}

/**
 * @brief 
 * Data starts in device_buf0
 * Data ends in device_buf1
 */
template<bool do_compute>
void inverse_fft(Complex *host_buf0, Complex *host_buf1, Complex *device_buf0, Complex *device_buf1,
                 Complex *device_twiddles0, Complex *device_twiddles1, MPI_Comm *row_comm, MPI_Comm *col_comm,
                 cufftHandle *plan0, Vector<cufftHandle> &plan1, cufftHandle *plan2, Vector<cufftHandle> &plan3,
                 int id, Complex scale, Vector<cudaStream_t> &streams) {
    #ifdef __PRINT__TIMING__
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::high_resolution_clock::now();
    #endif // __PRINT__TIMING__
    
    if constexpr (do_compute) {
        for (int row = 0; row < LOCAL_DIM/(P_DIM*B_DIM); row++) {
            Complex* row_ptr0 = device_buf0 + row * LOCAL_DIM * P_DIM * B_DIM;
            Complex* row_ptr1 = device_buf1 + row * LOCAL_DIM * P_DIM * B_DIM;
            DEVICE_FFT_SAFE_CALL(DEVICE_FFT_EXECZ2Z(plan3[row % NUM_STREAMS], row_ptr0, row_ptr1, DEVICE_FFT_INVERSE));
        }
    } else {
        placeholder<<<LOCAL_DIM, LOCAL_DIM>>>(device_buf0, device_buf1);
    }

    pack_inverse_fft1<do_compute><<<LOCAL_DIM, LOCAL_DIM>>>(device_buf1, device_buf0, device_twiddles1);
    DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());

    #ifdef __GPU__AWARE__MPI__
        MPI_Alltoall(device_buf0,
                    (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                    MPI_C_DOUBLE_COMPLEX,
                    device_buf1,
                    (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                    MPI_C_DOUBLE_COMPLEX,
                    *col_comm);
    #else
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(host_buf0, device_buf0, LOCAL_COMPLEX_BYTES, MEM_COPY_DEVICE_TO_HOST));
        MPI_Alltoall(host_buf0,
                    (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                    MPI_C_DOUBLE_COMPLEX,
                    host_buf1,
                    (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                    MPI_C_DOUBLE_COMPLEX,
                    *col_comm);
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(device_buf1, host_buf1, LOCAL_COMPLEX_BYTES, MEM_COPY_HOST_TO_DEVICE));
    #endif // __GPU__AWARE__MPI__

    if constexpr (do_compute) {
        DEVICE_FFT_SAFE_CALL(DEVICE_FFT_EXECZ2Z(*plan2, device_buf1, device_buf0, DEVICE_FFT_INVERSE));
    } else {
        placeholder<<<LOCAL_DIM, LOCAL_DIM>>>(device_buf1, device_buf0);
    }

    if constexpr (do_compute) {
        for (int row = 0; row < LOCAL_DIM; row++) {
            Complex* row_ptr0 = device_buf0 + row * LOCAL_DIM;
            Complex* row_ptr1 = device_buf1 + row * LOCAL_DIM;
            DEVICE_FFT_SAFE_CALL(DEVICE_FFT_EXECZ2Z(plan1[row % NUM_STREAMS], row_ptr0, row_ptr1, DEVICE_FFT_INVERSE));
        }
    } else {
        placeholder<<<LOCAL_DIM, LOCAL_DIM>>>(device_buf0, device_buf1);
    }

    pack_inverse_fft0<do_compute><<<LOCAL_DIM, LOCAL_DIM>>>(device_buf1, device_buf0, device_twiddles0);
    DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());

    #ifdef __GPU__AWARE__MPI__
        MPI_Alltoall(device_buf0,
                    (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                    MPI_C_DOUBLE_COMPLEX,
                    device_buf1,
                    (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                    MPI_C_DOUBLE_COMPLEX,
                    *row_comm);
    #else
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(host_buf0, device_buf0, LOCAL_COMPLEX_BYTES, MEM_COPY_DEVICE_TO_HOST));
        MPI_Alltoall(host_buf0,
                    (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                    MPI_C_DOUBLE_COMPLEX,
                    host_buf1, 
                    (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                    MPI_C_DOUBLE_COMPLEX,
                    *row_comm);
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(device_buf1, host_buf1, LOCAL_COMPLEX_BYTES, MEM_COPY_HOST_TO_DEVICE));
    #endif // __GPU__AWARE__MPI__

    pack_inverse_all_to_all_row<do_compute><<<VEC_TOTAL, VEC>>>(device_buf1, device_buf0, scale);
    DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());

    if constexpr (do_compute) {
        DEVICE_FFT_SAFE_CALL(DEVICE_FFT_EXECZ2Z(*plan0, device_buf0, device_buf1, DEVICE_FFT_INVERSE));
    } else {
        placeholder<<<LOCAL_DIM, LOCAL_DIM>>>(device_buf0, device_buf1);
    }

    #ifdef __PRINT__TIMING__
        MPI_Barrier(MPI_COMM_WORLD);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        double ns = duration.count();
        if (id == 0) std::cout << "IFFT: " << ns << " ns" << std::endl;
    #endif // __PRINT__TIMING__
}

template void inverse_fft<false>(Complex*, Complex*, Complex*, Complex*,
                                 Complex*, Complex*, MPI_Comm*, MPI_Comm*,
                                 cufftHandle*, Vector<cufftHandle>&, cufftHandle*, Vector<cufftHandle>&,
                                 int, Complex, Vector<cudaStream_t>&);

template void inverse_fft<true>(Complex*, Complex*, Complex*, Complex*,
                                Complex*, Complex*, MPI_Comm*, MPI_Comm*,
                                cufftHandle*, Vector<cufftHandle>&, cufftHandle*, Vector<cufftHandle>&,
                                int, Complex, Vector<cudaStream_t>&);

void test_fft() {
    MPI_Init(NULL, NULL);

    #ifdef __GPU__SET__
        MPI_Comm local_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);

        int local_rank;
        MPI_Comm_rank(local_comm, &local_rank);

        DEVICE_RT_SAFE_CALL(DEVICE_SET(local_rank));

        int num_devices;
        DEVICE_COUNT(&num_devices);

        for (int i = 0; i < num_devices; i++) {
            if (i != local_rank) {
                DEVICE_ENABLE_PA(i, 0); 
            }
        }
    #endif // __GPU__SET__

    int P, id;
    P = P_DIM * P_DIM;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    #ifdef __PRINT__SANITY__
        if (id == 0) std::cout << "N_DIM: " << N_DIM << ", B_DIM: " << B_DIM << " P_DIM: " << P_DIM << std::endl;
    #endif

    MPI_Comm row_comm, col_comm;

    int rid = id / P_DIM;
    MPI_Comm_split(MPI_COMM_WORLD, rid, id, &row_comm);

    int cid = id % P_DIM;
    MPI_Comm_split(MPI_COMM_WORLD, cid, id, &col_comm);

    cufftHandle plan0, plan2;
    Vector<cufftHandle> plan1(NUM_STREAMS), plan3(NUM_STREAMS);
    Vector<cudaStream_t> streams(NUM_STREAMS);
    init_plans(&plan0, plan1, &plan2, plan3, streams);

    FftdxFft1Ctx ctx1;
    init_fftdx_fft1(ctx1);

    Complex *host_buf0, *host_buf1, *host_forward_twiddles0, *host_forward_twiddles1, *host_inverse_twiddles0, *host_inverse_twiddles1;
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&host_buf0, LOCAL_COMPLEX_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&host_buf1, LOCAL_COMPLEX_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&host_forward_twiddles0, LOCAL_COMPLEX_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&host_forward_twiddles1, LOCAL_COMPLEX_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&host_inverse_twiddles0, LOCAL_COMPLEX_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&host_inverse_twiddles1, LOCAL_COMPLEX_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
    init_host(rid, cid, host_buf0);
    init_forward_twiddles0(host_forward_twiddles0, cid);
    init_forward_twiddles1(host_forward_twiddles1, rid);
    init_inverse_twiddles0(host_inverse_twiddles0, cid);
    init_inverse_twiddles1(host_inverse_twiddles1, rid);

    #ifdef __PRINT__TWIDDLES__
    if (id == 0) {
        std::cout << "Twiddles" << std::endl;
        std::cout << "rid: " << rid << " " << "cid: " << cid << std::endl;
        for (int i = 0; i < LOCAL_DIM; i++) {
            for (int j = 0; j < LOCAL_DIM; j++) {
                std::cout << "(" << host_forward_twiddles1[i * LOCAL_DIM + j].x << ", " << host_forward_twiddles1[i * LOCAL_DIM + j].y << ") ";
            }
            std::cout << std::endl;
        }
    }
    #endif

    Complex *device_buf0, *device_buf1, *device_forward_twiddles0, *device_forward_twiddles1, *device_inverse_twiddles0, *device_inverse_twiddles1;
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&device_buf0, LOCAL_COMPLEX_BYTES));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&device_buf1, LOCAL_COMPLEX_BYTES));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&device_forward_twiddles0, LOCAL_COMPLEX_BYTES));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&device_forward_twiddles1, LOCAL_COMPLEX_BYTES));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&device_inverse_twiddles0, LOCAL_COMPLEX_BYTES));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&device_inverse_twiddles1, LOCAL_COMPLEX_BYTES));
    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(device_buf0, host_buf0, LOCAL_COMPLEX_BYTES, MEM_COPY_HOST_TO_DEVICE));
    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(device_buf1, host_buf1, LOCAL_COMPLEX_BYTES, MEM_COPY_HOST_TO_DEVICE));
    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(device_forward_twiddles0, host_forward_twiddles0, LOCAL_COMPLEX_BYTES, MEM_COPY_HOST_TO_DEVICE));
    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(device_forward_twiddles1, host_forward_twiddles1, LOCAL_COMPLEX_BYTES, MEM_COPY_HOST_TO_DEVICE));
    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(device_inverse_twiddles0, host_inverse_twiddles0, LOCAL_COMPLEX_BYTES, MEM_COPY_HOST_TO_DEVICE));
    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(device_inverse_twiddles1, host_inverse_twiddles1, LOCAL_COMPLEX_BYTES, MEM_COPY_HOST_TO_DEVICE));

    // Used to scale the inverse fft
    double s = 1.0 / (double(N_DIM) * double(N_DIM));
    Complex scale = DEVICE_FFT_DOUBLECOMPLEX_CONSTRUCTOR(s, 0.0);

    #ifdef __ZMODEL__COMPUTE__
        constexpr bool do_compute = true;
    #else
        constexpr bool do_compute = false;
    #endif

    for (int i = 0; i < RUNS; i++) {
        forward_fft<do_compute>(host_buf0, host_buf1, device_buf0, device_buf1,
                                device_forward_twiddles0, device_forward_twiddles1, 
                                &row_comm, &col_comm, &plan0, plan1, &plan2, plan3, id, streams, ctx1);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // // for (int i = 0; i < RUNS; i++) {
    //     // Swapped device buffer because taking output from forward as input
        //inverse_fft<do_compute>(host_buf0, host_buf1, device_buf1, device_buf0,
                                //device_inverse_twiddles0, device_inverse_twiddles1,
                                //&row_comm, &col_comm, &plan0, plan1, &plan2, plan3, id, scale, streams);
    // // }
    
    // // FIXME: Swap device_buf1 to device_buf0 if you are not using inverse
    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(host_buf1, device_buf1, LOCAL_COMPLEX_BYTES, MEM_COPY_DEVICE_TO_HOST));

    // store_block_cyclic(host_buf1);

    #ifdef __PRINT__RESULTS__
    MPI_Barrier(MPI_COMM_WORLD);
    if (id == 0) {
        std::cout << "Host out" << std::endl;
        for (int i = 0; i < LOCAL_DIM; i++) {
            for (int j = 0; j < LOCAL_DIM; j++) {
                std::cout << host_buf1[i * LOCAL_DIM + j].x << " ";
                // std::cout << "(" << host_buf1[i * LOCAL_DIM + j].x << ", " << host_buf1[i * LOCAL_DIM + j].y << ") ";
            }
            std::cout << std::endl;
        }
    }
    #endif

    #ifdef __PRINT__RESULTS__TO__FILE__
        std::string filename = "output_r" + std::to_string(rid)
                            + "_c" + std::to_string(cid)
                            + ".txt";
        std::ofstream file(filename);
        for (int i = 0; i < LOCAL_DIM; i++) {
            for (int j = 0; j < LOCAL_DIM; j++) {
                file << "(" << host_buf1[i * LOCAL_DIM + j].x << ", " << host_buf1[i * LOCAL_DIM + j].y << ") ";
            }
            file << "\n";
        }
    #endif

    // MPI_Barrier(MPI_COMM_WORLD);
    // if (id == 0) {
        // print_block_cyclic(host_buf1);
    // }

    DEVICE_FFT_SAFE_CALL(DEVICE_FFT_DESTROY(plan0));
    DEVICE_FFT_SAFE_CALL(DEVICE_FFT_DESTROY(plan2));
    for (int i = 0; i < NUM_STREAMS; i++) {
        DEVICE_FFT_SAFE_CALL(DEVICE_FFT_DESTROY(plan1[i]));
        DEVICE_FFT_SAFE_CALL(DEVICE_FFT_DESTROY(plan3[i]));
        DEVICE_RT_SAFE_CALL(DEVICE_STREAM_DESTROY(streams[i]));
    }
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(host_buf0));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(host_buf1));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(host_forward_twiddles0));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(host_forward_twiddles1));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(host_inverse_twiddles0));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(host_inverse_twiddles1));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(device_buf0));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(device_buf1));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(device_forward_twiddles0));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(device_forward_twiddles1));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(device_inverse_twiddles0));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(device_inverse_twiddles1));

    MPI_Finalize();
}