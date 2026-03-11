#include "fft.h"
#include <chrono>
#include <complex>
#include <fstream>

/******************** MACROS & Function Declarations ********************/

#ifdef __PRINT__DETAILED__TIMING__
    #define TIME_MPI_MAX_NS(TAG, ID, ...) do { \
        MPI_Barrier(MPI_COMM_WORLD); \
        auto _t0 = std::chrono::high_resolution_clock::now(); \
        __VA_ARGS__ \
        auto _t1 = std::chrono::high_resolution_clock::now(); \
        long long _ns = std::chrono::duration_cast<std::chrono::nanoseconds>(_t1 - _t0).count(); \
        long long _max = 0; \
        MPI_Reduce(&_ns, &_max, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD); \
        if ((ID) == 0) std::cout << (TAG) << " " << _max << " ns\n"; \
    } while(0)
#else
    #define TIME_MPI_MAX_NS(TAG, ID, ...) do { __VA_ARGS__ } while(0)
#endif

template void forward_fft<true>(FftBuffers&, MPI_Comm*, MPI_Comm*, int, FftHostPlans&, FftdxFft1Ctx&);

template void forward_fft<false>(FftBuffers&, MPI_Comm*, MPI_Comm*, int, FftHostPlans&, FftdxFft1Ctx&);

template void inverse_fft<true>(FftBuffers&, MPI_Comm*, MPI_Comm*, int, Complex, FftHostPlans&);

template void inverse_fft<false>(FftBuffers&, MPI_Comm*, MPI_Comm*, int, Complex, FftHostPlans&);

/******************** Init ********************/

void init_forward_twiddles(Complex *in, int id) {
    int stride = VEC/P_DIM; // This is the processor offset
    for (int local_row = 0; local_row < LOCAL_DIM; local_row++) {
        for (int fft_row = 0; fft_row < stride; fft_row++) {
            for (int fft_col = 0; fft_col < LOCAL_DIM/stride; fft_col++) {
                int fft_row_offset = id * stride;
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

void init_inverse_twiddles(Complex *in, int id) {
    int stride = VEC/P_DIM; // This is the processor offset
    for (int local_row = 0; local_row < LOCAL_DIM; local_row++) {
        for (int fft_row = 0; fft_row < stride; fft_row++) {
            for (int fft_col = 0; fft_col < LOCAL_DIM/stride; fft_col++) {
                int fft_row_offset = id * stride;
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

template<bool include_inverse>
void init_buffers(FftBuffers& buffers, int rid, int cid) {
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&buffers.host_buf0, LOCAL_COMPLEX_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&buffers.host_buf1, LOCAL_COMPLEX_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&buffers.host_forward_twiddles0, LOCAL_COMPLEX_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&buffers.host_forward_twiddles1, LOCAL_COMPLEX_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
    if constexpr (include_inverse) {
        DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&buffers.host_inverse_twiddles0, LOCAL_COMPLEX_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
        DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&buffers.host_inverse_twiddles1, LOCAL_COMPLEX_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
    }

    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&buffers.device_buf0, LOCAL_COMPLEX_BYTES));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&buffers.device_buf1, LOCAL_COMPLEX_BYTES));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&buffers.device_forward_twiddles0, LOCAL_COMPLEX_BYTES));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&buffers.device_forward_twiddles1, LOCAL_COMPLEX_BYTES));
    if constexpr (include_inverse) {
        DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&buffers.device_inverse_twiddles0, LOCAL_COMPLEX_BYTES));
        DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&buffers.device_inverse_twiddles1, LOCAL_COMPLEX_BYTES));
    }

    init_host(rid, cid, buffers.host_buf0);
    init_forward_twiddles(buffers.host_forward_twiddles0, cid);
    init_forward_twiddles(buffers.host_forward_twiddles1, rid);
    if constexpr (include_inverse) {
        init_inverse_twiddles(buffers.host_inverse_twiddles0, cid);
        init_inverse_twiddles(buffers.host_inverse_twiddles1, rid);
    }

    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_buf0, buffers.host_buf0, LOCAL_COMPLEX_BYTES, MEM_COPY_HOST_TO_DEVICE));
    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_buf1, buffers.host_buf1, LOCAL_COMPLEX_BYTES, MEM_COPY_HOST_TO_DEVICE));
    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_forward_twiddles0, buffers.host_forward_twiddles0, LOCAL_COMPLEX_BYTES, MEM_COPY_HOST_TO_DEVICE));
    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_forward_twiddles1, buffers.host_forward_twiddles1, LOCAL_COMPLEX_BYTES, MEM_COPY_HOST_TO_DEVICE));
    if constexpr (include_inverse) {
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_inverse_twiddles0, buffers.host_inverse_twiddles0, LOCAL_COMPLEX_BYTES, MEM_COPY_HOST_TO_DEVICE));
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_inverse_twiddles1, buffers.host_inverse_twiddles1, LOCAL_COMPLEX_BYTES, MEM_COPY_HOST_TO_DEVICE));
    }
}

template<bool include_inverse>
void destroy_buffers(FftBuffers& buffers) {
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(buffers.host_buf0));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(buffers.host_buf1));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(buffers.host_forward_twiddles0));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(buffers.host_forward_twiddles1));
    if constexpr (include_inverse) {
        DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(buffers.host_inverse_twiddles0));
        DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(buffers.host_inverse_twiddles1));
    }

    DEVICE_RT_SAFE_CALL(DEVICE_FREE(buffers.device_buf0));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(buffers.device_buf1));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(buffers.device_forward_twiddles0));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(buffers.device_forward_twiddles1));
    if constexpr (include_inverse) {
        DEVICE_RT_SAFE_CALL(DEVICE_FREE(buffers.device_inverse_twiddles0));
        DEVICE_RT_SAFE_CALL(DEVICE_FREE(buffers.device_inverse_twiddles1));
    }
}

// NOTE: This is used for debugging --> can comment out FFTs and replace with this
__global__ void placeholder(Complex *device_in, Complex *device_out) {
    device_out[blockIdx.x * LOCAL_DIM + threadIdx.x] = device_in[blockIdx.x * LOCAL_DIM + threadIdx.x];
}

/******************** Forward ********************/

__global__ void pack_forward_fft0(Complex *device_in, Complex *device_out) {
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
__global__ void pack_forward_fft1(Complex *device_in, Complex *device_out, Complex *forward_twiddles) {
    const int split_size = LOCAL_DIM / P_DIM;                // The size of the split

    const int index_in_row_split = blockIdx.x % split_size;
    const int index_in_vec_split = threadIdx.x % split_size;

    const int row_split = blockIdx.x / split_size;          // Which section of the row split
    const int vec_split = threadIdx.x / split_size;         // Which section of the split local vector

    // Swap the vec and row splits
    const int row = vec_split + (index_in_row_split * P_DIM);
    const int col = (row_split * split_size) + index_in_vec_split;

    if constexpr (do_compute) {
        device_out[row * LOCAL_DIM + col] = device_in[blockIdx.x * LOCAL_DIM + threadIdx.x] * forward_twiddles[row * LOCAL_DIM + col];
    } else {
        device_out[row * LOCAL_DIM + col] = device_in[blockIdx.x * LOCAL_DIM + threadIdx.x];
    }
}

__global__ void repack_transpose(Complex *device_in, Complex *device_out) {
    const int i = blockIdx.x;
    const int j = threadIdx.x;
    device_out[j * LOCAL_DIM + (i % B_DIM) * VEC + (i / B_DIM)] = device_in[i * LOCAL_DIM + j];
}

/**
 * @brief
 * Data starts in device_buf0
 * Data ends in device_buf1
 *
 * Pipeline (repack-transpose method):
 *   x-direction: Plan_A -> pack -> alltoall(row) -> unpack+twiddle -> Plan_B
 *   repack-transpose
 *   y-direction: Plan_A -> pack -> alltoall(col) -> unpack+twiddle -> Plan_B
 */
template<bool do_compute>
void forward_fft(FftBuffers& buffers, MPI_Comm *row_comm, MPI_Comm *col_comm,
                 int id, FftHostPlans& host_plans, FftdxFft1Ctx& ctx1) {
    #ifdef __PRINT__TIMING__
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::high_resolution_clock::now();
    #endif // __PRINT__TIMING__

    TIME_MPI_MAX_NS("FFT0", id,
        if constexpr (do_compute) {
            DEVICE_FFT_SAFE_CALL(DEVICE_FFT_EXECZ2Z(host_plans.plan0, buffers.device_buf0, buffers.device_buf1, DEVICE_FFT_FORWARD));
            DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
        } else {
            placeholder<<<LOCAL_DIM, LOCAL_DIM>>>(buffers.device_buf0, buffers.device_buf1);
        }
    );

    TIME_MPI_MAX_NS("PACK0", id,
        pack_forward_fft0<<<VEC_TOTAL, VEC>>>(buffers.device_buf1, buffers.device_buf0);
        DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
    );

    TIME_MPI_MAX_NS("COMMS0", id,
        #ifdef __GPU__AWARE__MPI__
            MPI_Alltoall(buffers.device_buf0,
                        (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                        MPI_C_DOUBLE_COMPLEX,
                        buffers.device_buf1,
                        (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                        MPI_C_DOUBLE_COMPLEX,
                        *row_comm);
        #else
            DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.host_buf0, buffers.device_buf0, LOCAL_COMPLEX_BYTES, MEM_COPY_DEVICE_TO_HOST));
            MPI_Alltoall(buffers.host_buf0,
                        (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                        MPI_C_DOUBLE_COMPLEX,
                        buffers.host_buf1,
                        (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                        MPI_C_DOUBLE_COMPLEX,
                        *row_comm);
            DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_buf1, buffers.host_buf1, LOCAL_COMPLEX_BYTES, MEM_COPY_HOST_TO_DEVICE));
        #endif // __GPU__AWARE__MPI__
    );

    TIME_MPI_MAX_NS("PACK1", id,
        pack_forward_fft1<do_compute><<<LOCAL_DIM, LOCAL_DIM>>>(buffers.device_buf1, buffers.device_buf0, buffers.device_forward_twiddles0);
        DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
    );

    TIME_MPI_MAX_NS("FFT1", id,
        if constexpr (do_compute) {
            #ifdef __USE__FFTDX__
                fft1_kernel<typename FftdxFft1Ctx::FFT, FftdxFft1Ctx::VEC_SPLIT>
                <<<ctx1.grid, ctx1.block, ctx1.shmem>>>(buffers.device_buf0, buffers.device_buf1, ctx1.ws);
            #else
                for (int row = 0; row < LOCAL_DIM; row++) {
                    Complex* row_ptr0 = buffers.device_buf0 + row * LOCAL_DIM;
                    Complex* row_ptr1 = buffers.device_buf1 + row * LOCAL_DIM;
                    DEVICE_FFT_SAFE_CALL(DEVICE_FFT_EXECZ2Z(host_plans.plan1[row % NUM_STREAMS], row_ptr0, row_ptr1, DEVICE_FFT_FORWARD));
                }
            #endif
            DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
        } else {
            placeholder<<<LOCAL_DIM, LOCAL_DIM>>>(buffers.device_buf0, buffers.device_buf1);
        }
    );

    TIME_MPI_MAX_NS("REPACK", id,
        repack_transpose<<<LOCAL_DIM, LOCAL_DIM>>>(buffers.device_buf1, buffers.device_buf0);
        DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
    );

    TIME_MPI_MAX_NS("FFT2", id,
        if constexpr (do_compute) {
            DEVICE_FFT_SAFE_CALL(DEVICE_FFT_EXECZ2Z(host_plans.plan0, buffers.device_buf0, buffers.device_buf1, DEVICE_FFT_FORWARD));
            DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
        } else {
            placeholder<<<LOCAL_DIM, LOCAL_DIM>>>(buffers.device_buf0, buffers.device_buf1);
        }
    );

    TIME_MPI_MAX_NS("PACK2", id,
        pack_forward_fft0<<<VEC_TOTAL, VEC>>>(buffers.device_buf1, buffers.device_buf0);
        DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
    );

    TIME_MPI_MAX_NS("COMMS1", id,
        #ifdef __GPU__AWARE__MPI__
            MPI_Alltoall(buffers.device_buf0,
                        (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                        MPI_C_DOUBLE_COMPLEX,
                        buffers.device_buf1,
                        (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                        MPI_C_DOUBLE_COMPLEX,
                        *col_comm);
        #else
            DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.host_buf0, buffers.device_buf0, LOCAL_COMPLEX_BYTES, MEM_COPY_DEVICE_TO_HOST));
            MPI_Alltoall(buffers.host_buf0,
                        (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                        MPI_C_DOUBLE_COMPLEX,
                        buffers.host_buf1,
                        (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                        MPI_C_DOUBLE_COMPLEX,
                        *col_comm);
            DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_buf1, buffers.host_buf1, LOCAL_COMPLEX_BYTES, MEM_COPY_HOST_TO_DEVICE));
        #endif // __GPU__AWARE__MPI__
    );

    TIME_MPI_MAX_NS("PACK3", id,
        pack_forward_fft1<do_compute><<<LOCAL_DIM, LOCAL_DIM>>>(buffers.device_buf1, buffers.device_buf0, buffers.device_forward_twiddles1);
        DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
    );

    TIME_MPI_MAX_NS("FFT3", id,
        if constexpr (do_compute) {
            #ifdef __USE__FFTDX__
                fft1_kernel<typename FftdxFft1Ctx::FFT, FftdxFft1Ctx::VEC_SPLIT>
                <<<ctx1.grid, ctx1.block, ctx1.shmem>>>(buffers.device_buf0, buffers.device_buf1, ctx1.ws);
            #else
                for (int row = 0; row < LOCAL_DIM; row++) {
                    Complex* row_ptr0 = buffers.device_buf0 + row * LOCAL_DIM;
                    Complex* row_ptr1 = buffers.device_buf1 + row * LOCAL_DIM;
                    DEVICE_FFT_SAFE_CALL(DEVICE_FFT_EXECZ2Z(host_plans.plan1[row % NUM_STREAMS], row_ptr0, row_ptr1, DEVICE_FFT_FORWARD));
                }
            #endif
            DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
        } else {
            placeholder<<<LOCAL_DIM, LOCAL_DIM>>>(buffers.device_buf0, buffers.device_buf1);
        }
    );

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

/******************** Inverse ********************/

template<bool do_compute>
__global__ void pack_inverse_fft0(Complex *device_in, Complex *device_out, Complex scale) {
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
__global__ void pack_inverse_fft1(Complex *device_in, Complex *device_out, Complex *inverse_twiddles) {
    const int split_size = LOCAL_DIM / P_DIM;                // The size of the split

    const int index_in_row_split = blockIdx.x % split_size;
    const int index_in_vec_split = threadIdx.x % split_size;

    const int row_split = blockIdx.x / split_size;          // Which section of the row split
    const int vec_split = threadIdx.x / split_size;         // Which section of the split local vector

    // Swap the vec and row splits
    const int row = vec_split + (index_in_row_split * P_DIM);
    const int col = (row_split * split_size) + index_in_vec_split;

    if constexpr (do_compute) {
        device_out[blockIdx.x * LOCAL_DIM + threadIdx.x] = device_in[row * LOCAL_DIM + col] * inverse_twiddles[row * LOCAL_DIM + col];
    } else {
        device_out[blockIdx.x * LOCAL_DIM + threadIdx.x] = device_in[row * LOCAL_DIM + col];
    }
}

__global__ void inverse_repack_transpose(Complex *device_in, Complex *device_out) {
    const int i = blockIdx.x;
    const int j = threadIdx.x;
    device_out[i * LOCAL_DIM + j] = device_in[j * LOCAL_DIM + (i % B_DIM) * VEC + (i / B_DIM)];
}

/**
 * @brief
 * Data starts in device_buf0
 * Data ends in device_buf1
 *
 * Pipeline (inverse of repack-transpose method):
 *   y-direction: Plan_B^-1 -> inv_unpack+inv_twiddle -> alltoall(col) -> inv_pack -> Plan_A^-1
 *   inverse repack-transpose
 *   x-direction: Plan_B^-1 -> inv_unpack+inv_twiddle -> alltoall(row) -> inv_pack+scale -> Plan_A^-1
 */
template<bool do_compute>
void inverse_fft(FftBuffers& buffers, MPI_Comm *row_comm, MPI_Comm *col_comm,
                 int id, Complex scale, FftHostPlans& host_plans) {
    #ifdef __PRINT__TIMING__
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::high_resolution_clock::now();
    #endif // __PRINT__TIMING__

    // Only apply the scaling factor once
    Complex unity = DEVICE_FFT_DOUBLECOMPLEX_CONSTRUCTOR(1.0, 0.0);

    if constexpr (do_compute) {
        for (int row = 0; row < LOCAL_DIM; row++) {
            Complex* row_ptr0 = buffers.device_buf0 + row * LOCAL_DIM;
            Complex* row_ptr1 = buffers.device_buf1 + row * LOCAL_DIM;
            DEVICE_FFT_SAFE_CALL(DEVICE_FFT_EXECZ2Z(host_plans.plan1[row % NUM_STREAMS], row_ptr0, row_ptr1, DEVICE_FFT_INVERSE));
        }
    } else {
        placeholder<<<LOCAL_DIM, LOCAL_DIM>>>(buffers.device_buf0, buffers.device_buf1);
    }

    pack_inverse_fft1<do_compute><<<LOCAL_DIM, LOCAL_DIM>>>(buffers.device_buf1, buffers.device_buf0, buffers.device_inverse_twiddles1);
    DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());

    #ifdef __GPU__AWARE__MPI__
        MPI_Alltoall(buffers.device_buf0,
                    (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                    MPI_C_DOUBLE_COMPLEX,
                    buffers.device_buf1,
                    (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                    MPI_C_DOUBLE_COMPLEX,
                    *col_comm);
    #else
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.host_buf0, buffers.device_buf0, LOCAL_COMPLEX_BYTES, MEM_COPY_DEVICE_TO_HOST));
        MPI_Alltoall(buffers.host_buf0,
                    (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                    MPI_C_DOUBLE_COMPLEX,
                    buffers.host_buf1,
                    (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                    MPI_C_DOUBLE_COMPLEX,
                    *col_comm);
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_buf1, buffers.host_buf1, LOCAL_COMPLEX_BYTES, MEM_COPY_HOST_TO_DEVICE));
    #endif // __GPU__AWARE__MPI__

    pack_inverse_fft0<do_compute><<<VEC_TOTAL, VEC>>>(buffers.device_buf1, buffers.device_buf0, unity);
    DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());

    if constexpr (do_compute) {
        DEVICE_FFT_SAFE_CALL(DEVICE_FFT_EXECZ2Z(host_plans.plan0, buffers.device_buf0, buffers.device_buf1, DEVICE_FFT_INVERSE));
    } else {
        placeholder<<<LOCAL_DIM, LOCAL_DIM>>>(buffers.device_buf0, buffers.device_buf1);
    }

    inverse_repack_transpose<<<LOCAL_DIM, LOCAL_DIM>>>(buffers.device_buf1, buffers.device_buf0);
    DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());

    if constexpr (do_compute) {
        for (int row = 0; row < LOCAL_DIM; row++) {
            Complex* row_ptr0 = buffers.device_buf0 + row * LOCAL_DIM;
            Complex* row_ptr1 = buffers.device_buf1 + row * LOCAL_DIM;
            DEVICE_FFT_SAFE_CALL(DEVICE_FFT_EXECZ2Z(host_plans.plan1[row % NUM_STREAMS], row_ptr0, row_ptr1, DEVICE_FFT_INVERSE));
        }
    } else {
        placeholder<<<LOCAL_DIM, LOCAL_DIM>>>(buffers.device_buf0, buffers.device_buf1);
    }

    pack_inverse_fft1<do_compute><<<LOCAL_DIM, LOCAL_DIM>>>(buffers.device_buf1, buffers.device_buf0, buffers.device_inverse_twiddles0);
    DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());

    #ifdef __GPU__AWARE__MPI__
        MPI_Alltoall(buffers.device_buf0,
                    (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                    MPI_C_DOUBLE_COMPLEX,
                    buffers.device_buf1,
                    (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                    MPI_C_DOUBLE_COMPLEX,
                    *row_comm);
    #else
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.host_buf0, buffers.device_buf0, LOCAL_COMPLEX_BYTES, MEM_COPY_DEVICE_TO_HOST));
        MPI_Alltoall(buffers.host_buf0,
                    (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                    MPI_C_DOUBLE_COMPLEX,
                    buffers.host_buf1,
                    (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                    MPI_C_DOUBLE_COMPLEX,
                    *row_comm);
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_buf1, buffers.host_buf1, LOCAL_COMPLEX_BYTES, MEM_COPY_HOST_TO_DEVICE));
    #endif // __GPU__AWARE__MPI__

    pack_inverse_fft0<do_compute><<<VEC_TOTAL, VEC>>>(buffers.device_buf1, buffers.device_buf0, scale);
    DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());

    if constexpr (do_compute) {
        DEVICE_FFT_SAFE_CALL(DEVICE_FFT_EXECZ2Z(host_plans.plan0, buffers.device_buf0, buffers.device_buf1, DEVICE_FFT_INVERSE));
    } else {
        placeholder<<<LOCAL_DIM, LOCAL_DIM>>>(buffers.device_buf0, buffers.device_buf1);
    }

    #ifdef __PRINT__TIMING__
        MPI_Barrier(MPI_COMM_WORLD);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        double ns = duration.count();
        if (id == 0) std::cout << "IFFT: " << ns << " ns" << std::endl;
    #endif // __PRINT__TIMING__
}

/******************** Test ********************/

void test_fft() {
    #ifdef __ZMODEL__INCLUDE__INVERSE__
        constexpr bool include_inverse = true;
    #else
        constexpr bool include_inverse = false;
    #endif

    #ifdef __ZMODEL__COMPUTE__
        constexpr bool do_compute = true;
    #else
        constexpr bool do_compute = false;
    #endif

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

    FftHostPlans host_plans;
    init_plans(host_plans);

    FftdxFft1Ctx ctx1;
    init_fftdx_fft1(ctx1);

    FftBuffers buffers;
    init_buffers<include_inverse>(buffers, rid, cid);

    #ifdef __PRINT__TWIDDLES__
        if (id == 0) {
            std::cout << "Twiddles" << std::endl;
            std::cout << "rid: " << rid << " " << "cid: " << cid << std::endl;
            for (int i = 0; i < LOCAL_DIM; i++) {
                for (int j = 0; j < LOCAL_DIM; j++) {
                    std::cout << "(" << buffers.host_forward_twiddles1[i * LOCAL_DIM + j].x \
                    << ", " << buffers.host_forward_twiddles1[i * LOCAL_DIM + j].y << ") ";
                }
                std::cout << std::endl;
            }
        }
    #endif

    // Used to scale the inverse fft
    double s = 1.0 / (double(N_DIM) * double(N_DIM));
    Complex scale = DEVICE_FFT_DOUBLECOMPLEX_CONSTRUCTOR(s, 0.0);

    for (int i = 0; i < RUNS; i++) {
        forward_fft<do_compute>(buffers, &row_comm, &col_comm, id, host_plans, ctx1);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    #ifdef __ZMODEL__INCLUDE__INVERSE__
        for (int i = 0; i < RUNS; i++) {
            inverse_fft<do_compute>(buffers, &row_comm, &col_comm, id, scale, host_plans);
        }
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.host_buf1, buffers.device_buf1, LOCAL_COMPLEX_BYTES, MEM_COPY_DEVICE_TO_HOST));
    #else
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.host_buf1, buffers.device_buf1, LOCAL_COMPLEX_BYTES, MEM_COPY_DEVICE_TO_HOST));
    #endif // __ZMODEL__INCLUDE__INVERSE__

    #ifdef __PRINT__RESULTS__
    MPI_Barrier(MPI_COMM_WORLD);
    if (id == 0) {
        std::cout << "Host out" << std::endl;
        for (int i = 0; i < LOCAL_DIM; i++) {
            for (int j = 0; j < LOCAL_DIM; j++) {
                std::cout << buffers.host_buf1[i * LOCAL_DIM + j].x << " ";
                // std::cout << "(" << buffers.host_buf1[i * LOCAL_DIM + j].x << ", " << buffers.host_buf1[i * LOCAL_DIM + j].y << ") ";
            }
            std::cout << std::endl;
        }
    }
    #endif

    destroy_plans(host_plans);
    destroy_buffers<include_inverse>(buffers);

    MPI_Finalize();
}
