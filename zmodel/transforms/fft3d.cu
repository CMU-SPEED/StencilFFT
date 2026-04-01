#include "fft.h"
#include <chrono>
#include <complex>

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

#define COMM_SIZE_3D ((LOCAL_DIM*LOCAL_DIM*LOCAL_DIM)/P_DIM)

template void forward_fft_3d<true>(FftBuffers&, MPI_Comm&, MPI_Comm&, MPI_Comm&, int, FftHostPlans&, FftdxFft1Ctx&);
template void forward_fft_3d<false>(FftBuffers&, MPI_Comm&, MPI_Comm&, MPI_Comm&, int, FftHostPlans&, FftdxFft1Ctx&);
template void inverse_fft_3d<true>(FftBuffers&, MPI_Comm&, MPI_Comm&, MPI_Comm&, int, Complex, FftHostPlans&);
template void inverse_fft_3d<false>(FftBuffers&, MPI_Comm&, MPI_Comm&, MPI_Comm&, int, Complex, FftHostPlans&);

template void init_buffers_3d<true>(FftBuffers&, int, int, int);
template void init_buffers_3d<false>(FftBuffers&, int, int, int);
template void destroy_buffers_3d<true>(FftBuffers&);
template void destroy_buffers_3d<false>(FftBuffers&);

/******************** Init ********************/

void init_forward_twiddles_3d(Complex *in, int id) {
    int stride = VEC/P_DIM; // This is the processor offset
    for (int dep = 0; dep < LOCAL_DIM; dep++) {
        for (int local_row = 0; local_row < LOCAL_DIM; local_row++) {
            for (int fft_row = 0; fft_row < stride; fft_row++) {
                for (int fft_col = 0; fft_col < LOCAL_DIM/stride; fft_col++) {
                    int fft_row_offset = id * stride;
                    double k = (double)(fft_row + fft_row_offset);
                    double l = (double)fft_col;
                    std::complex<double> twiddle = std::exp(std::complex<double>(0.0, -2*M_PI*k*l/N_DIM));
                    in[(dep * LOCAL_DIM * LOCAL_DIM) + (local_row * LOCAL_DIM) + fft_row + (fft_col * stride)] = DEVICE_FFT_DOUBLECOMPLEX_CONSTRUCTOR(twiddle.real(), twiddle.imag());
                }
            }
        }
    }
}

void init_inverse_twiddles_3d(Complex *in, int id) {
    int stride = VEC/P_DIM; // This is the processor offset
    for (int dep = 0; dep < LOCAL_DIM; dep++) {
        for (int local_row = 0; local_row < LOCAL_DIM; local_row++) {
            for (int fft_row = 0; fft_row < stride; fft_row++) {
                for (int fft_col = 0; fft_col < LOCAL_DIM/stride; fft_col++) {
                    int fft_row_offset = id * stride;
                    double k = (double)(fft_row + fft_row_offset);
                    double l = (double)fft_col;
                    std::complex<double> twiddle = std::exp(std::complex<double>(0.0, 2*M_PI*k*l/N_DIM));
                    in[(dep * LOCAL_DIM * LOCAL_DIM) + (local_row * LOCAL_DIM) + fft_row + (fft_col * stride)] = DEVICE_FFT_DOUBLECOMPLEX_CONSTRUCTOR(twiddle.real(), twiddle.imag());
                }
            }
        }
    }
}

template<bool include_inverse>
void init_buffers_3d(FftBuffers& buffers, int rid, int cid, int did) {
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&buffers.host_buf0, LOCAL_COMPLEX_BYTES_3D, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&buffers.host_buf1, LOCAL_COMPLEX_BYTES_3D, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&buffers.host_forward_twiddles0, LOCAL_COMPLEX_BYTES_3D, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&buffers.host_forward_twiddles1, LOCAL_COMPLEX_BYTES_3D, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&buffers.host_forward_twiddles2, LOCAL_COMPLEX_BYTES_3D, DEVICE_HOST_ALLOC_DEFAULT));
    if constexpr (include_inverse) {
        DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&buffers.host_inverse_twiddles0, LOCAL_COMPLEX_BYTES_3D, DEVICE_HOST_ALLOC_DEFAULT));
        DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&buffers.host_inverse_twiddles1, LOCAL_COMPLEX_BYTES_3D, DEVICE_HOST_ALLOC_DEFAULT));
        DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&buffers.host_inverse_twiddles2, LOCAL_COMPLEX_BYTES_3D, DEVICE_HOST_ALLOC_DEFAULT));
    }

    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&buffers.device_buf0, LOCAL_COMPLEX_BYTES_3D));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&buffers.device_buf1, LOCAL_COMPLEX_BYTES_3D));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&buffers.device_forward_twiddles0, LOCAL_COMPLEX_BYTES_3D));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&buffers.device_forward_twiddles1, LOCAL_COMPLEX_BYTES_3D));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&buffers.device_forward_twiddles2, LOCAL_COMPLEX_BYTES_3D));
    if constexpr (include_inverse) {
        DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&buffers.device_inverse_twiddles0, LOCAL_COMPLEX_BYTES_3D));
        DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&buffers.device_inverse_twiddles1, LOCAL_COMPLEX_BYTES_3D));
        DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&buffers.device_inverse_twiddles2, LOCAL_COMPLEX_BYTES_3D));
    }

    init_forward_twiddles_3d(buffers.host_forward_twiddles0, cid);
    init_forward_twiddles_3d(buffers.host_forward_twiddles1, rid);
    init_forward_twiddles_3d(buffers.host_forward_twiddles2, did);
    if constexpr (include_inverse) {
        init_inverse_twiddles_3d(buffers.host_inverse_twiddles0, cid);
        init_inverse_twiddles_3d(buffers.host_inverse_twiddles1, rid);
        init_inverse_twiddles_3d(buffers.host_inverse_twiddles2, did);
    }

    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_forward_twiddles0, buffers.host_forward_twiddles0, LOCAL_COMPLEX_BYTES_3D, MEM_COPY_HOST_TO_DEVICE));
    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_forward_twiddles1, buffers.host_forward_twiddles1, LOCAL_COMPLEX_BYTES_3D, MEM_COPY_HOST_TO_DEVICE));
    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_forward_twiddles2, buffers.host_forward_twiddles2, LOCAL_COMPLEX_BYTES_3D, MEM_COPY_HOST_TO_DEVICE));
    if constexpr (include_inverse) {
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_inverse_twiddles0, buffers.host_inverse_twiddles0, LOCAL_COMPLEX_BYTES_3D, MEM_COPY_HOST_TO_DEVICE));
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_inverse_twiddles1, buffers.host_inverse_twiddles1, LOCAL_COMPLEX_BYTES_3D, MEM_COPY_HOST_TO_DEVICE));
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_inverse_twiddles2, buffers.host_inverse_twiddles2, LOCAL_COMPLEX_BYTES_3D, MEM_COPY_HOST_TO_DEVICE));
    }
}

template<bool include_inverse>
void destroy_buffers_3d(FftBuffers& buffers) {
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(buffers.host_buf0));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(buffers.host_buf1));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(buffers.host_forward_twiddles0));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(buffers.host_forward_twiddles1));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(buffers.host_forward_twiddles2));
    if constexpr (include_inverse) {
        DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(buffers.host_inverse_twiddles0));
        DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(buffers.host_inverse_twiddles1));
        DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(buffers.host_inverse_twiddles2));
    }

    DEVICE_RT_SAFE_CALL(DEVICE_FREE(buffers.device_buf0));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(buffers.device_buf1));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(buffers.device_forward_twiddles0));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(buffers.device_forward_twiddles1));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(buffers.device_forward_twiddles2));
    if constexpr (include_inverse) {
        DEVICE_RT_SAFE_CALL(DEVICE_FREE(buffers.device_inverse_twiddles0));
        DEVICE_RT_SAFE_CALL(DEVICE_FREE(buffers.device_inverse_twiddles1));
        DEVICE_RT_SAFE_CALL(DEVICE_FREE(buffers.device_inverse_twiddles2));
    }
}

__global__ void placeholder_3d(Complex *device_in, Complex *device_out) {
    device_out[blockIdx.x * LOCAL_DIM + threadIdx.x] = device_in[blockIdx.x * LOCAL_DIM + threadIdx.x];
}

/******************** Forward ********************/

// Pack for alltoall: splits each vector into P sub-vectors and packs into P chunks of L^3/P
// Grid: dim3(LOCAL_DIM, VEC_TOTAL), Threads: VEC
__global__ void pack_forward_fft_3d(Complex *device_in, Complex *device_out) {
    const int row = blockIdx.y / VEC_COL;
    const int col = blockIdx.y % VEC_COL;
    const int dep = blockIdx.x;
    const int in_offset = (row * LOCAL_DIM) + (col * VEC) + (dep * LOCAL_DIM * LOCAL_DIM);

    const int vec_index = threadIdx.x;                      // Index into the vector

    const int vec_split = VEC / P_DIM;                      // The size of a split vector
    const int split = vec_index / vec_split;                // Which section of the split vector
    const int index_in_split = vec_index % vec_split;       // Index into the split vector

    // Note these are both measured in element space
    const int slab_size = (LOCAL_DIM*LOCAL_DIM) / P_DIM;    // Num elements in a slab of splits
    const int slab_vec_row = LOCAL_DIM / P_DIM;             // Number of split vectors in a col of a slab
    const int slab_vec_col = LOCAL_DIM / vec_split;         // Number of split vectors in a row of a slab/local

    const int slab_offset = (split * slab_size * LOCAL_DIM) + (dep * slab_size);
    const int out_row = (blockIdx.y / slab_vec_col) % slab_vec_row;
    const int out_col = blockIdx.y % slab_vec_col;
    const int out_offset = slab_offset + (out_row * LOCAL_DIM) + (out_col * vec_split);

    device_out[out_offset + index_in_split] = device_in[in_offset + vec_index];
}

// Unpack after alltoall: reassembles P received chunks into local array + optional twiddle multiply
// Grid: dim3(LOCAL_DIM, LOCAL_DIM), Threads: LOCAL_DIM
template<bool do_compute>
__global__ void unpack_forward_fft_3d(Complex *device_in, Complex *device_out, Complex *forward_twiddles) {
    const int split_size = LOCAL_DIM / P_DIM;                // The size of the split
    const int slab_size = (LOCAL_DIM * LOCAL_DIM) / P_DIM;   // Num elements in a depth page

    const int index_in_row_split = blockIdx.y % split_size;
    const int index_in_vec_split = threadIdx.x % split_size;

    const int row_split = blockIdx.y / split_size;          // Which section of the row split
    const int vec_split = threadIdx.x / split_size;         // Which section of the split local vector

    // Input: jump to proc chunk, then depth page, then row
    const int in_offset = (row_split * slab_size * LOCAL_DIM) + (blockIdx.x * slab_size) + (index_in_row_split * LOCAL_DIM) + threadIdx.x;

    // Swap the vec and row splits
    const int row = vec_split + (index_in_row_split * P_DIM);
    const int col = (row_split * split_size) + index_in_vec_split;

    const int out_idx = (blockIdx.x * LOCAL_DIM * LOCAL_DIM) + (row * LOCAL_DIM) + col;

    if constexpr (do_compute) {
        device_out[out_idx] = device_in[in_offset] * forward_twiddles[out_idx];
    } else {
        device_out[out_idx] = device_in[in_offset];
    }
}

// Repack-transpose xy: swap x (current rows) <-> y (current cols) within each depth slice
// out[d*L^2 + j*L + (i%B)*VEC + i//B] = in[d*L^2 + i*L + j]
// Grid: dim3(LOCAL_DIM, LOCAL_DIM), Threads: LOCAL_DIM
__global__ void repack_transpose_xy_3d(Complex *device_in, Complex *device_out) {
    const int d = blockIdx.x;
    const int i = blockIdx.y;
    const int j = threadIdx.x;
    device_out[d * LOCAL_DIM * LOCAL_DIM + j * LOCAL_DIM + (i % B_DIM) * VEC + (i / B_DIM)] =
        device_in[d * LOCAL_DIM * LOCAL_DIM + i * LOCAL_DIM + j];
}

// Repack-transpose yz: swap y (current cols, original rows) <-> z (current depth, original depth)
// out[c*L^2 + r*L + (d%B)*VEC + d//B] = in[d*L^2 + r*L + c]
// Grid: dim3(LOCAL_DIM, LOCAL_DIM), Threads: LOCAL_DIM
__global__ void repack_transpose_yz_3d(Complex *device_in, Complex *device_out) {
    const int d = blockIdx.x;
    const int r = blockIdx.y;
    const int c = threadIdx.x;
    device_out[c * LOCAL_DIM * LOCAL_DIM + r * LOCAL_DIM + (d % B_DIM) * VEC + (d / B_DIM)] =
        device_in[d * LOCAL_DIM * LOCAL_DIM + r * LOCAL_DIM + c];
}

/**
 * @brief
 * Data starts in device_buf0
 * Data ends in device_buf1
 *
 * Pipeline (repack-transpose method, 3 directions):
 *   x-direction: Plan_A -> pack -> alltoall(row) -> unpack+twiddle(cid) -> Plan_B
 *   repack-transpose-xy
 *   y-direction: Plan_A -> pack -> alltoall(col) -> unpack+twiddle(rid) -> Plan_B
 *   repack-transpose-yz
 *   z-direction: Plan_A -> pack -> alltoall(dep) -> unpack+twiddle(did) -> Plan_B
 */
template<bool do_compute>
void forward_fft_3d(FftBuffers& buffers, MPI_Comm& row_comm, MPI_Comm& col_comm, MPI_Comm& dep_comm,
                    int id, FftHostPlans& host_plans, FftdxFft1Ctx& ctx1) {
    #ifdef __PRINT__TIMING__
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::high_resolution_clock::now();
    #endif // __PRINT__TIMING__

    dim3 grid_pack(LOCAL_DIM, VEC_TOTAL);    // for pack kernel: depth x vec_total
    dim3 grid_unpack(LOCAL_DIM, LOCAL_DIM);  // for unpack kernel: depth x row
    dim3 grid_repack(LOCAL_DIM, LOCAL_DIM);  // for repack kernels: depth x row

    TIME_MPI_MAX_NS("FFT0", id,
        if constexpr (do_compute) {
            DEVICE_FFT_SAFE_CALL(DEVICE_FFT_EXECZ2Z(host_plans.plan0, buffers.device_buf0, buffers.device_buf1, DEVICE_FFT_FORWARD));
            DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
        } else {
            placeholder_3d<<<LOCAL_DIM * LOCAL_DIM, LOCAL_DIM>>>(buffers.device_buf0, buffers.device_buf1);
        }
    );

    TIME_MPI_MAX_NS("PACK0", id,
        pack_forward_fft_3d<<<grid_pack, VEC>>>(buffers.device_buf1, buffers.device_buf0);
        DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
    );

    TIME_MPI_MAX_NS("COMMS0", id,
        #ifdef __GPU__AWARE__MPI__
            MPI_Alltoall(buffers.device_buf0, COMM_SIZE_3D, MPI_C_DOUBLE_COMPLEX,
                         buffers.device_buf1, COMM_SIZE_3D, MPI_C_DOUBLE_COMPLEX, row_comm);
        #else
            DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.host_buf0, buffers.device_buf0, LOCAL_COMPLEX_BYTES_3D, MEM_COPY_DEVICE_TO_HOST));
            MPI_Alltoall(buffers.host_buf0, COMM_SIZE_3D, MPI_C_DOUBLE_COMPLEX,
                         buffers.host_buf1, COMM_SIZE_3D, MPI_C_DOUBLE_COMPLEX, row_comm);
            DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_buf1, buffers.host_buf1, LOCAL_COMPLEX_BYTES_3D, MEM_COPY_HOST_TO_DEVICE));
        #endif // __GPU__AWARE__MPI__
    );

    TIME_MPI_MAX_NS("PACK1", id,
        unpack_forward_fft_3d<do_compute><<<grid_unpack, LOCAL_DIM>>>(buffers.device_buf1, buffers.device_buf0, buffers.device_forward_twiddles0);
        DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
    );

    TIME_MPI_MAX_NS("FFT1", id,
        if constexpr (do_compute) {
            #ifdef __USE__FFTDX__
                fft1_3d_kernel<typename FftdxFft1Ctx::FFT, FftdxFft1Ctx::VEC_SPLIT>
                <<<ctx1.grid, ctx1.block, ctx1.shmem>>>(buffers.device_buf0, buffers.device_buf1, ctx1.ws);
            #else
                for (int i = 0; i < LOCAL_DIM * LOCAL_DIM; i++) {
                    Complex* row_ptr0 = buffers.device_buf0 + i * LOCAL_DIM;
                    Complex* row_ptr1 = buffers.device_buf1 + i * LOCAL_DIM;
                    DEVICE_FFT_SAFE_CALL(DEVICE_FFT_EXECZ2Z(host_plans.plan1[i % NUM_STREAMS], row_ptr0, row_ptr1, DEVICE_FFT_FORWARD));
                }
            #endif
            DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
        } else {
            placeholder_3d<<<LOCAL_DIM * LOCAL_DIM, LOCAL_DIM>>>(buffers.device_buf0, buffers.device_buf1);
        }
    );

    TIME_MPI_MAX_NS("REPACK_XY", id,
        repack_transpose_xy_3d<<<grid_repack, LOCAL_DIM>>>(buffers.device_buf1, buffers.device_buf0);
        DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
    );

    TIME_MPI_MAX_NS("FFT2", id,
        if constexpr (do_compute) {
            DEVICE_FFT_SAFE_CALL(DEVICE_FFT_EXECZ2Z(host_plans.plan0, buffers.device_buf0, buffers.device_buf1, DEVICE_FFT_FORWARD));
            DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
        } else {
            placeholder_3d<<<LOCAL_DIM * LOCAL_DIM, LOCAL_DIM>>>(buffers.device_buf0, buffers.device_buf1);
        }
    );

    TIME_MPI_MAX_NS("PACK2", id,
        pack_forward_fft_3d<<<grid_pack, VEC>>>(buffers.device_buf1, buffers.device_buf0);
        DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
    );

    TIME_MPI_MAX_NS("COMMS1", id,
        #ifdef __GPU__AWARE__MPI__
            MPI_Alltoall(buffers.device_buf0, COMM_SIZE_3D, MPI_C_DOUBLE_COMPLEX,
                         buffers.device_buf1, COMM_SIZE_3D, MPI_C_DOUBLE_COMPLEX, col_comm);
        #else
            DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.host_buf0, buffers.device_buf0, LOCAL_COMPLEX_BYTES_3D, MEM_COPY_DEVICE_TO_HOST));
            MPI_Alltoall(buffers.host_buf0, COMM_SIZE_3D, MPI_C_DOUBLE_COMPLEX,
                         buffers.host_buf1, COMM_SIZE_3D, MPI_C_DOUBLE_COMPLEX, col_comm);
            DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_buf1, buffers.host_buf1, LOCAL_COMPLEX_BYTES_3D, MEM_COPY_HOST_TO_DEVICE));
        #endif // __GPU__AWARE__MPI__
    );

    TIME_MPI_MAX_NS("PACK3", id,
        unpack_forward_fft_3d<do_compute><<<grid_unpack, LOCAL_DIM>>>(buffers.device_buf1, buffers.device_buf0, buffers.device_forward_twiddles1);
        DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
    );

    TIME_MPI_MAX_NS("FFT3", id,
        if constexpr (do_compute) {
            #ifdef __USE__FFTDX__
                fft1_3d_kernel<typename FftdxFft1Ctx::FFT, FftdxFft1Ctx::VEC_SPLIT>
                <<<ctx1.grid, ctx1.block, ctx1.shmem>>>(buffers.device_buf0, buffers.device_buf1, ctx1.ws);
            #else
                for (int i = 0; i < LOCAL_DIM * LOCAL_DIM; i++) {
                    Complex* row_ptr0 = buffers.device_buf0 + i * LOCAL_DIM;
                    Complex* row_ptr1 = buffers.device_buf1 + i * LOCAL_DIM;
                    DEVICE_FFT_SAFE_CALL(DEVICE_FFT_EXECZ2Z(host_plans.plan1[i % NUM_STREAMS], row_ptr0, row_ptr1, DEVICE_FFT_FORWARD));
                }
            #endif
            DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
        } else {
            placeholder_3d<<<LOCAL_DIM * LOCAL_DIM, LOCAL_DIM>>>(buffers.device_buf0, buffers.device_buf1);
        }
    );

    TIME_MPI_MAX_NS("REPACK_YZ", id,
        repack_transpose_yz_3d<<<grid_repack, LOCAL_DIM>>>(buffers.device_buf1, buffers.device_buf0);
        DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
    );

    TIME_MPI_MAX_NS("FFT4", id,
        if constexpr (do_compute) {
            DEVICE_FFT_SAFE_CALL(DEVICE_FFT_EXECZ2Z(host_plans.plan0, buffers.device_buf0, buffers.device_buf1, DEVICE_FFT_FORWARD));
            DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
        } else {
            placeholder_3d<<<LOCAL_DIM * LOCAL_DIM, LOCAL_DIM>>>(buffers.device_buf0, buffers.device_buf1);
        }
    );

    TIME_MPI_MAX_NS("PACK4", id,
        pack_forward_fft_3d<<<grid_pack, VEC>>>(buffers.device_buf1, buffers.device_buf0);
        DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
    );

    TIME_MPI_MAX_NS("COMMS2", id,
        #ifdef __GPU__AWARE__MPI__
            MPI_Alltoall(buffers.device_buf0, COMM_SIZE_3D, MPI_C_DOUBLE_COMPLEX,
                         buffers.device_buf1, COMM_SIZE_3D, MPI_C_DOUBLE_COMPLEX, dep_comm);
        #else
            DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.host_buf0, buffers.device_buf0, LOCAL_COMPLEX_BYTES_3D, MEM_COPY_DEVICE_TO_HOST));
            MPI_Alltoall(buffers.host_buf0, COMM_SIZE_3D, MPI_C_DOUBLE_COMPLEX,
                         buffers.host_buf1, COMM_SIZE_3D, MPI_C_DOUBLE_COMPLEX, dep_comm);
            DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_buf1, buffers.host_buf1, LOCAL_COMPLEX_BYTES_3D, MEM_COPY_HOST_TO_DEVICE));
        #endif // __GPU__AWARE__MPI__
    );

    TIME_MPI_MAX_NS("PACK5", id,
        unpack_forward_fft_3d<do_compute><<<grid_unpack, LOCAL_DIM>>>(buffers.device_buf1, buffers.device_buf0, buffers.device_forward_twiddles2);
        DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
    );

    TIME_MPI_MAX_NS("FFT5", id,
        if constexpr (do_compute) {
            #ifdef __USE__FFTDX__
                fft1_3d_kernel<typename FftdxFft1Ctx::FFT, FftdxFft1Ctx::VEC_SPLIT>
                <<<ctx1.grid, ctx1.block, ctx1.shmem>>>(buffers.device_buf0, buffers.device_buf1, ctx1.ws);
            #else
                for (int i = 0; i < LOCAL_DIM * LOCAL_DIM; i++) {
                    Complex* row_ptr0 = buffers.device_buf0 + i * LOCAL_DIM;
                    Complex* row_ptr1 = buffers.device_buf1 + i * LOCAL_DIM;
                    DEVICE_FFT_SAFE_CALL(DEVICE_FFT_EXECZ2Z(host_plans.plan1[i % NUM_STREAMS], row_ptr0, row_ptr1, DEVICE_FFT_FORWARD));
                }
            #endif
            DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
        } else {
            placeholder_3d<<<LOCAL_DIM * LOCAL_DIM, LOCAL_DIM>>>(buffers.device_buf0, buffers.device_buf1);
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

// Inverse of pack_forward_fft_3d
// Grid: dim3(LOCAL_DIM, VEC_TOTAL), Threads: VEC
template<bool do_compute>
__global__ void pack_inverse_fft_3d(Complex *device_in, Complex *device_out, Complex scale) {
    const int row = blockIdx.y / VEC_COL;
    const int col = blockIdx.y % VEC_COL;
    const int dep = blockIdx.x;
    const int natural_offset = (row * LOCAL_DIM) + (col * VEC) + (dep * LOCAL_DIM * LOCAL_DIM);

    const int vec_index = threadIdx.x;

    const int vec_split = VEC / P_DIM;
    const int split = vec_index / vec_split;
    const int index_in_split = vec_index % vec_split;

    const int slab_size = (LOCAL_DIM*LOCAL_DIM) / P_DIM;
    const int slab_vec_row = LOCAL_DIM / P_DIM;
    const int slab_vec_col = LOCAL_DIM / vec_split;

    const int slab_offset = (split * slab_size * LOCAL_DIM) + (dep * slab_size);
    const int out_row = (blockIdx.y / slab_vec_col) % slab_vec_row;
    const int out_col = blockIdx.y % slab_vec_col;
    const int packed_offset = slab_offset + (out_row * LOCAL_DIM) + (out_col * vec_split);

    if constexpr (do_compute) {
        device_out[natural_offset + vec_index] = scale * device_in[packed_offset + index_in_split];
    } else {
        device_out[natural_offset + vec_index] = device_in[packed_offset + index_in_split];
    }
}

// Inverse of unpack_forward_fft_3d
// Grid: dim3(LOCAL_DIM, LOCAL_DIM), Threads: LOCAL_DIM
template<bool do_compute>
__global__ void unpack_inverse_fft_3d(Complex *device_in, Complex *device_out, Complex *inverse_twiddles) {
    const int split_size = LOCAL_DIM / P_DIM;
    const int slab_size = (LOCAL_DIM * LOCAL_DIM) / P_DIM;

    const int index_in_row_split = blockIdx.y % split_size;
    const int index_in_vec_split = threadIdx.x % split_size;

    const int row_split = blockIdx.y / split_size;
    const int vec_split = threadIdx.x / split_size;

    const int packed_offset = (row_split * slab_size * LOCAL_DIM) + (blockIdx.x * slab_size) + (index_in_row_split * LOCAL_DIM) + threadIdx.x;

    const int row = vec_split + (index_in_row_split * P_DIM);
    const int col = (row_split * split_size) + index_in_vec_split;

    const int natural_idx = (blockIdx.x * LOCAL_DIM * LOCAL_DIM) + (row * LOCAL_DIM) + col;

    if constexpr (do_compute) {
        device_out[packed_offset] = device_in[natural_idx] * inverse_twiddles[natural_idx];
    } else {
        device_out[packed_offset] = device_in[natural_idx];
    }
}

// Inverse of repack_transpose_xy_3d
// out[d*L^2 + i*L + j] = in[d*L^2 + j*L + (i%B)*VEC + i/B]
// Grid: dim3(LOCAL_DIM, LOCAL_DIM), Threads: LOCAL_DIM
__global__ void inverse_repack_transpose_xy_3d(Complex *device_in, Complex *device_out) {
    const int d = blockIdx.x;
    const int i = blockIdx.y;
    const int j = threadIdx.x;
    device_out[d * LOCAL_DIM * LOCAL_DIM + i * LOCAL_DIM + j] =
        device_in[d * LOCAL_DIM * LOCAL_DIM + j * LOCAL_DIM + (i % B_DIM) * VEC + (i / B_DIM)];
}

// Inverse of repack_transpose_yz_3d
// out[d*L^2 + r*L + c] = in[c*L^2 + r*L + (d%B)*VEC + d/B]
// Grid: dim3(LOCAL_DIM, LOCAL_DIM), Threads: LOCAL_DIM
__global__ void inverse_repack_transpose_yz_3d(Complex *device_in, Complex *device_out) {
    const int d = blockIdx.x;
    const int r = blockIdx.y;
    const int c = threadIdx.x;
    device_out[d * LOCAL_DIM * LOCAL_DIM + r * LOCAL_DIM + c] =
        device_in[c * LOCAL_DIM * LOCAL_DIM + r * LOCAL_DIM + (d % B_DIM) * VEC + (d / B_DIM)];
}

/**
 * @brief
 * Data starts in device_buf0
 * Data ends in device_buf1
 *
 * Pipeline (inverse of repack-transpose method, 3 directions):
 *   z-direction: Plan_B^-1 -> inv_unpack+inv_twiddle(did) -> alltoall(dep) -> inv_pack -> Plan_A^-1
 *   inverse repack-transpose-yz
 *   y-direction: Plan_B^-1 -> inv_unpack+inv_twiddle(rid) -> alltoall(col) -> inv_pack -> Plan_A^-1
 *   inverse repack-transpose-xy
 *   x-direction: Plan_B^-1 -> inv_unpack+inv_twiddle(cid) -> alltoall(row) -> inv_pack+scale -> Plan_A^-1
 */
template<bool do_compute>
void inverse_fft_3d(FftBuffers& buffers, MPI_Comm& row_comm, MPI_Comm& col_comm, MPI_Comm& dep_comm,
                    int id, Complex scale, FftHostPlans& host_plans) {
    #ifdef __PRINT__TIMING__
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::high_resolution_clock::now();
    #endif // __PRINT__TIMING__

    Complex unity = DEVICE_FFT_DOUBLECOMPLEX_CONSTRUCTOR(1.0, 0.0);

    dim3 grid_pack(LOCAL_DIM, VEC_TOTAL);
    dim3 grid_unpack(LOCAL_DIM, LOCAL_DIM);
    dim3 grid_repack(LOCAL_DIM, LOCAL_DIM);

    if constexpr (do_compute) {
        for (int i = 0; i < LOCAL_DIM * LOCAL_DIM; i++) {
            Complex* row_ptr0 = buffers.device_buf0 + i * LOCAL_DIM;
            Complex* row_ptr1 = buffers.device_buf1 + i * LOCAL_DIM;
            DEVICE_FFT_SAFE_CALL(DEVICE_FFT_EXECZ2Z(host_plans.plan1[i % NUM_STREAMS], row_ptr0, row_ptr1, DEVICE_FFT_INVERSE));
        }
    } else {
        placeholder_3d<<<LOCAL_DIM * LOCAL_DIM, LOCAL_DIM>>>(buffers.device_buf0, buffers.device_buf1);
    }

    unpack_inverse_fft_3d<do_compute><<<grid_unpack, LOCAL_DIM>>>(buffers.device_buf1, buffers.device_buf0, buffers.device_inverse_twiddles2);
    DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());

    #ifdef __GPU__AWARE__MPI__
        MPI_Alltoall(buffers.device_buf0, COMM_SIZE_3D, MPI_C_DOUBLE_COMPLEX,
                     buffers.device_buf1, COMM_SIZE_3D, MPI_C_DOUBLE_COMPLEX, dep_comm);
    #else
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.host_buf0, buffers.device_buf0, LOCAL_COMPLEX_BYTES_3D, MEM_COPY_DEVICE_TO_HOST));
        MPI_Alltoall(buffers.host_buf0, COMM_SIZE_3D, MPI_C_DOUBLE_COMPLEX,
                     buffers.host_buf1, COMM_SIZE_3D, MPI_C_DOUBLE_COMPLEX, dep_comm);
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_buf1, buffers.host_buf1, LOCAL_COMPLEX_BYTES_3D, MEM_COPY_HOST_TO_DEVICE));
    #endif // __GPU__AWARE__MPI__

    pack_inverse_fft_3d<do_compute><<<grid_pack, VEC>>>(buffers.device_buf1, buffers.device_buf0, unity);
    DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());

    if constexpr (do_compute) {
        DEVICE_FFT_SAFE_CALL(DEVICE_FFT_EXECZ2Z(host_plans.plan0, buffers.device_buf0, buffers.device_buf1, DEVICE_FFT_INVERSE));
    } else {
        placeholder_3d<<<LOCAL_DIM * LOCAL_DIM, LOCAL_DIM>>>(buffers.device_buf0, buffers.device_buf1);
    }

    inverse_repack_transpose_yz_3d<<<grid_repack, LOCAL_DIM>>>(buffers.device_buf1, buffers.device_buf0);
    DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());

    if constexpr (do_compute) {
        for (int i = 0; i < LOCAL_DIM * LOCAL_DIM; i++) {
            Complex* row_ptr0 = buffers.device_buf0 + i * LOCAL_DIM;
            Complex* row_ptr1 = buffers.device_buf1 + i * LOCAL_DIM;
            DEVICE_FFT_SAFE_CALL(DEVICE_FFT_EXECZ2Z(host_plans.plan1[i % NUM_STREAMS], row_ptr0, row_ptr1, DEVICE_FFT_INVERSE));
        }
    } else {
        placeholder_3d<<<LOCAL_DIM * LOCAL_DIM, LOCAL_DIM>>>(buffers.device_buf0, buffers.device_buf1);
    }

    unpack_inverse_fft_3d<do_compute><<<grid_unpack, LOCAL_DIM>>>(buffers.device_buf1, buffers.device_buf0, buffers.device_inverse_twiddles1);
    DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());

    #ifdef __GPU__AWARE__MPI__
        MPI_Alltoall(buffers.device_buf0, COMM_SIZE_3D, MPI_C_DOUBLE_COMPLEX,
                     buffers.device_buf1, COMM_SIZE_3D, MPI_C_DOUBLE_COMPLEX, col_comm);
    #else
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.host_buf0, buffers.device_buf0, LOCAL_COMPLEX_BYTES_3D, MEM_COPY_DEVICE_TO_HOST));
        MPI_Alltoall(buffers.host_buf0, COMM_SIZE_3D, MPI_C_DOUBLE_COMPLEX,
                     buffers.host_buf1, COMM_SIZE_3D, MPI_C_DOUBLE_COMPLEX, col_comm);
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_buf1, buffers.host_buf1, LOCAL_COMPLEX_BYTES_3D, MEM_COPY_HOST_TO_DEVICE));
    #endif // __GPU__AWARE__MPI__

    pack_inverse_fft_3d<do_compute><<<grid_pack, VEC>>>(buffers.device_buf1, buffers.device_buf0, unity);
    DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());

    if constexpr (do_compute) {
        DEVICE_FFT_SAFE_CALL(DEVICE_FFT_EXECZ2Z(host_plans.plan0, buffers.device_buf0, buffers.device_buf1, DEVICE_FFT_INVERSE));
    } else {
        placeholder_3d<<<LOCAL_DIM * LOCAL_DIM, LOCAL_DIM>>>(buffers.device_buf0, buffers.device_buf1);
    }

    inverse_repack_transpose_xy_3d<<<grid_repack, LOCAL_DIM>>>(buffers.device_buf1, buffers.device_buf0);
    DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());

    if constexpr (do_compute) {
        for (int i = 0; i < LOCAL_DIM * LOCAL_DIM; i++) {
            Complex* row_ptr0 = buffers.device_buf0 + i * LOCAL_DIM;
            Complex* row_ptr1 = buffers.device_buf1 + i * LOCAL_DIM;
            DEVICE_FFT_SAFE_CALL(DEVICE_FFT_EXECZ2Z(host_plans.plan1[i % NUM_STREAMS], row_ptr0, row_ptr1, DEVICE_FFT_INVERSE));
        }
    } else {
        placeholder_3d<<<LOCAL_DIM * LOCAL_DIM, LOCAL_DIM>>>(buffers.device_buf0, buffers.device_buf1);
    }

    unpack_inverse_fft_3d<do_compute><<<grid_unpack, LOCAL_DIM>>>(buffers.device_buf1, buffers.device_buf0, buffers.device_inverse_twiddles0);
    DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());

    #ifdef __GPU__AWARE__MPI__
        MPI_Alltoall(buffers.device_buf0, COMM_SIZE_3D, MPI_C_DOUBLE_COMPLEX,
                     buffers.device_buf1, COMM_SIZE_3D, MPI_C_DOUBLE_COMPLEX, row_comm);
    #else
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.host_buf0, buffers.device_buf0, LOCAL_COMPLEX_BYTES_3D, MEM_COPY_DEVICE_TO_HOST));
        MPI_Alltoall(buffers.host_buf0, COMM_SIZE_3D, MPI_C_DOUBLE_COMPLEX,
                     buffers.host_buf1, COMM_SIZE_3D, MPI_C_DOUBLE_COMPLEX, row_comm);
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_buf1, buffers.host_buf1, LOCAL_COMPLEX_BYTES_3D, MEM_COPY_HOST_TO_DEVICE));
    #endif // __GPU__AWARE__MPI__

    pack_inverse_fft_3d<do_compute><<<grid_pack, VEC>>>(buffers.device_buf1, buffers.device_buf0, scale);
    DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());

    if constexpr (do_compute) {
        DEVICE_FFT_SAFE_CALL(DEVICE_FFT_EXECZ2Z(host_plans.plan0, buffers.device_buf0, buffers.device_buf1, DEVICE_FFT_INVERSE));
    } else {
        placeholder_3d<<<LOCAL_DIM * LOCAL_DIM, LOCAL_DIM>>>(buffers.device_buf0, buffers.device_buf1);
    }

    #ifdef __PRINT__TIMING__
        MPI_Barrier(MPI_COMM_WORLD);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        long long ns = duration.count();
        long long max_time = 0;
        MPI_Reduce(&ns, &max_time, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
        if (id == 0) std::cout << "IFFT3D TOTAL " << max_time << " ns\n" << std::endl;
    #endif // __PRINT__TIMING__
}

/******************** Test ********************/

void test_fft_3d() {
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

    int P, id;
    P = P_DIM * P_DIM * P_DIM;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    int rid, cid, did;
    did = id / (P_DIM * P_DIM);
    rid = (id % (P_DIM * P_DIM)) / P_DIM;
    cid = (id % (P_DIM * P_DIM)) % P_DIM;

    #ifdef __PRINT__SANITY__
        if (id == 0) std::cout << "N_DIM: " << N_DIM << ", B_DIM: " << B_DIM << " P_DIM: " << P_DIM << std::endl;
    #endif

    MPI_Comm row_comm, col_comm, dep_comm;

    // The processors in a dep_grp should all share the same rid and cid
    int dep_grp = rid * P_DIM + cid;
    MPI_Comm_split(MPI_COMM_WORLD, dep_grp, id, &dep_comm);

    // The processors in a row_grp should all share the same did and rid
    int row_grp = did * P_DIM + rid;
    MPI_Comm_split(MPI_COMM_WORLD, row_grp, id, &row_comm);

    // The processors in a col_grp should all share the same did and cid
    int col_grp = did * P_DIM + cid;
    MPI_Comm_split(MPI_COMM_WORLD, col_grp, id, &col_comm);

    FftHostPlans host_plans;
    init_plans_3d(host_plans);

    FftdxFft1Ctx ctx1;
    #ifdef __USE__FFTDX__
        init_fftdx_fft1_3d(ctx1);
    #endif

    FftBuffers buffers;
    init_buffers_3d<include_inverse>(buffers, rid, cid, did);
    init_host_3d(rid, cid, did, buffers.host_buf0);
    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_buf0, buffers.host_buf0, LOCAL_COMPLEX_BYTES_3D, MEM_COPY_HOST_TO_DEVICE));

    for (int i = 0; i < RUNS; i++) {
        forward_fft_3d<do_compute>(buffers, row_comm, col_comm, dep_comm, id, host_plans, ctx1);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    #ifdef __ZMODEL__INCLUDE__INVERSE__
        double s = 1.0 / (double(N_DIM) * double(N_DIM) * double(N_DIM));
        Complex scale = DEVICE_FFT_DOUBLECOMPLEX_CONSTRUCTOR(s, 0.0);

        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_buf0, buffers.device_buf1, LOCAL_COMPLEX_BYTES_3D, MEM_COPY_DEVICE_TO_DEVICE));
        for (int i = 0; i < RUNS; i++) {
            inverse_fft_3d<do_compute>(buffers, row_comm, col_comm, dep_comm, id, scale, host_plans);
        }
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.host_buf1, buffers.device_buf1, LOCAL_COMPLEX_BYTES_3D, MEM_COPY_DEVICE_TO_HOST));
    #else
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.host_buf1, buffers.device_buf1, LOCAL_COMPLEX_BYTES_3D, MEM_COPY_DEVICE_TO_HOST));
    #endif // __ZMODEL__INCLUDE__INVERSE__

    #ifdef __PRINT__RESULTS__
        MPI_Barrier(MPI_COMM_WORLD);
        if (id == 0) {
            std::cout << "Host out" << std::endl;
            for (int d = 0; d < LOCAL_DIM; d++) {
                std::cout << "depth " << d << std::endl;
                for (int i = 0; i < LOCAL_DIM; i++) {
                    for (int j = 0; j < LOCAL_DIM; j++) {
                        std::cout << buffers.host_buf1[(d * LOCAL_DIM * LOCAL_DIM) + (i * LOCAL_DIM) + j].x << " ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
        }
    #endif

    destroy_plans_3d(host_plans);
    destroy_buffers_3d<include_inverse>(buffers);

    MPI_Finalize();
}

/******************** Benchmark ********************/

void benchmark_kernels_3d() {
    Complex *device_buf0, *device_buf1, *device_twiddles;
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&device_buf0, LOCAL_COMPLEX_BYTES_3D));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&device_buf1, LOCAL_COMPLEX_BYTES_3D));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&device_twiddles, LOCAL_COMPLEX_BYTES_3D));
    DEVICE_RT_SAFE_CALL(DEVICE_MEM_SET(device_buf0, 0, LOCAL_COMPLEX_BYTES_3D));
    DEVICE_RT_SAFE_CALL(DEVICE_MEM_SET(device_buf1, 0, LOCAL_COMPLEX_BYTES_3D));
    DEVICE_RT_SAFE_CALL(DEVICE_MEM_SET(device_twiddles, 0, LOCAL_COMPLEX_BYTES_3D));

    FftHostPlans host_plans;
    init_plans_3d(host_plans);

    #ifdef __USE__FFTDX__
        FftdxFft1Ctx ctx1;
        init_fftdx_fft1_3d(ctx1);
    #endif

    dim3 grid_pack(LOCAL_DIM, VEC_TOTAL);
    dim3 grid_unpack(LOCAL_DIM, LOCAL_DIM);
    dim3 grid_repack(LOCAL_DIM, LOCAL_DIM);

    // Warm up
    pack_forward_fft_3d<<<grid_pack, VEC>>>(device_buf0, device_buf1);
    DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());

    auto time_kernel = [](const char* name, int runs, auto kernel_fn) {
        DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < runs; i++) {
            kernel_fn();
        }
        DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
        auto t1 = std::chrono::high_resolution_clock::now();
        long long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        std::cout << name << " " << ns / runs << " ns (avg over " << runs << " runs)\n";
    };

    time_kernel("FFT_PLAN0", RUNS, [&]() {
        DEVICE_FFT_SAFE_CALL(DEVICE_FFT_EXECZ2Z(host_plans.plan0, device_buf0, device_buf1, DEVICE_FFT_FORWARD));
        DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
    });

    time_kernel("FFT_PLAN1", RUNS, [&]() {
        #ifdef __USE__FFTDX__
            fft1_3d_kernel<typename FftdxFft1Ctx::FFT, FftdxFft1Ctx::VEC_SPLIT>
            <<<ctx1.grid, ctx1.block, ctx1.shmem>>>(device_buf0, device_buf1, ctx1.ws);
        #else
            for (int i = 0; i < LOCAL_DIM * LOCAL_DIM; i++) {
                Complex* row_ptr0 = device_buf0 + i * LOCAL_DIM;
                Complex* row_ptr1 = device_buf1 + i * LOCAL_DIM;
                DEVICE_FFT_SAFE_CALL(DEVICE_FFT_EXECZ2Z(host_plans.plan1[i % NUM_STREAMS], row_ptr0, row_ptr1, DEVICE_FFT_FORWARD));
            }
        #endif
        DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
    });

    time_kernel("PACK", RUNS, [&]() {
        pack_forward_fft_3d<<<grid_pack, VEC>>>(device_buf0, device_buf1);
        DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
    });

    time_kernel("UNPACK_TWIDDLE", RUNS, [&]() {
        unpack_forward_fft_3d<true><<<grid_unpack, LOCAL_DIM>>>(device_buf0, device_buf1, device_twiddles);
        DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
    });

    time_kernel("REPACK_XY", RUNS, [&]() {
        repack_transpose_xy_3d<<<grid_repack, LOCAL_DIM>>>(device_buf0, device_buf1);
        DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
    });

    time_kernel("REPACK_YZ", RUNS, [&]() {
        repack_transpose_yz_3d<<<grid_repack, LOCAL_DIM>>>(device_buf0, device_buf1);
        DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
    });

    time_kernel("INV_UNPACK_TWIDDLE", RUNS, [&]() {
        unpack_inverse_fft_3d<true><<<grid_unpack, LOCAL_DIM>>>(device_buf0, device_buf1, device_twiddles);
        DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
    });

    Complex unity = DEVICE_FFT_DOUBLECOMPLEX_CONSTRUCTOR(1.0, 0.0);
    time_kernel("INV_PACK", RUNS, [&]() {
        pack_inverse_fft_3d<true><<<grid_pack, VEC>>>(device_buf0, device_buf1, unity);
        DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
    });

    time_kernel("INV_REPACK_XY", RUNS, [&]() {
        inverse_repack_transpose_xy_3d<<<grid_repack, LOCAL_DIM>>>(device_buf0, device_buf1);
        DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
    });

    time_kernel("INV_REPACK_YZ", RUNS, [&]() {
        inverse_repack_transpose_yz_3d<<<grid_repack, LOCAL_DIM>>>(device_buf0, device_buf1);
        DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());
    });

    destroy_plans_3d(host_plans);
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(device_buf0));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(device_buf1));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(device_twiddles));
}
