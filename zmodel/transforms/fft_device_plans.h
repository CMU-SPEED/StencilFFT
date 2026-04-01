#ifndef __ZMODEL__FFT__DEVICE__PLANS__
#define __ZMODEL__FFT__DEVICE__PLANS__

#include "../utils.h"

#ifdef __USE__FFTDX__

#include <cufftdx.hpp>
using namespace cufftdx;

struct FftdxFft1Ctx {
    static constexpr int VEC_SPLIT = VEC / P_DIM;
    static constexpr int FFT_SIZE  = LOCAL_DIM / VEC_SPLIT;

    // Tuneable parameter
    static constexpr int FFTS_PER_BLOCK = (VEC_SPLIT >= 8 ? 8 : VEC_SPLIT);

    using FFT = decltype(
        cufftdx::Size<FFT_SIZE>() +
        cufftdx::Precision<double>() +
        cufftdx::Type<cufftdx::fft_type::c2c>() +
        cufftdx::Direction<cufftdx::fft_direction::forward>() +
        cufftdx::FFTsPerBlock<FFTS_PER_BLOCK>() +
        cufftdx::SM<CUFFTDX_TARGET_SM>() +
        cufftdx::Block()
    );

    static_assert(FFT::block_dim.x * FFT::block_dim.y <= FFT::max_threads_per_block, "Too many threads per block");

    typename FFT::workspace_type ws;
    dim3 block, grid;
    size_t shmem;
};

/******************** 2D Plans ********************/

inline void init_fftdx_fft1(FftdxFft1Ctx& ctx1) {
    ctx1.block  = FftdxFft1Ctx::FFT::block_dim;
    ctx1.grid   = dim3(LOCAL_DIM, (FftdxFft1Ctx::VEC_SPLIT + FftdxFft1Ctx::FFT::ffts_per_block - 1) /
                               FftdxFft1Ctx::FFT::ffts_per_block);
    ctx1.shmem  = FftdxFft1Ctx::FFT::shared_memory_size;

    cudaError_t err = cudaSuccess;
    ctx1.ws = cufftdx::make_workspace<typename FftdxFft1Ctx::FFT>(err);
    if (err != cudaSuccess) std::abort();
}

template<class FFT, int VEC_SPLIT>
__global__ void fft1_kernel(const Complex* __restrict__ in, Complex* __restrict__ out,
                            typename FFT::workspace_type workspace) {
    using complex_type = typename FFT::value_type;

    const complex_type* in_c  = reinterpret_cast<const complex_type*>(in);
    complex_type*       out_c = reinterpret_cast<complex_type*>(out);

    const unsigned int row = blockIdx.x;

    // Which FFT-within-row this threadblock instance handles
    const unsigned int t = blockIdx.y * FFT::ffts_per_block + threadIdx.y;
    if (t >= (unsigned) VEC_SPLIT) return;

    constexpr unsigned int FFT_SIZE = cufftdx::size_of<FFT>::value;

    // Base for this FFT inside the row (interleaved by VEC_SPLIT)
    unsigned int index = row * LOCAL_DIM + t + threadIdx.x * VEC_SPLIT;
    const unsigned int stride = FFT::stride * VEC_SPLIT;

    complex_type thread_data[FFT::storage_size];

    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];

    #pragma unroll
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
        if ((i * FFT::stride + threadIdx.x) < FFT_SIZE) {
            thread_data[i] = in_c[index];
            index += stride;
        }
    }

    FFT().execute(thread_data, shared_mem, workspace);

    index = row * LOCAL_DIM + t + threadIdx.x * VEC_SPLIT;
    #pragma unroll
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
        if ((i * FFT::stride + threadIdx.x) < FFT_SIZE) {
            out_c[index] = thread_data[i];
            index += stride;
        }
    }
}

/******************** 3D Plans ********************/

inline void init_fftdx_fft1_3d(FftdxFft1Ctx& ctx1) {
    ctx1.block  = FftdxFft1Ctx::FFT::block_dim;
    ctx1.grid   = dim3(LOCAL_DIM * LOCAL_DIM, (FftdxFft1Ctx::VEC_SPLIT + FftdxFft1Ctx::FFT::ffts_per_block - 1) /
                               FftdxFft1Ctx::FFT::ffts_per_block);
    ctx1.shmem  = FftdxFft1Ctx::FFT::shared_memory_size;

    cudaError_t err = cudaSuccess;
    ctx1.ws = cufftdx::make_workspace<typename FftdxFft1Ctx::FFT>(err);
    if (err != cudaSuccess) std::abort();
}

template<class FFT, int VEC_SPLIT>
__global__ void fft1_3d_kernel(const Complex* __restrict__ in, Complex* __restrict__ out,
                               typename FFT::workspace_type workspace) {
    using complex_type = typename FFT::value_type;

    const complex_type* in_c  = reinterpret_cast<const complex_type*>(in);
    complex_type*       out_c = reinterpret_cast<complex_type*>(out);

    // blockIdx.x ranges over LOCAL_DIM * LOCAL_DIM (depth * row)
    const unsigned int row = blockIdx.x;

    const unsigned int t = blockIdx.y * FFT::ffts_per_block + threadIdx.y;
    if (t >= (unsigned) VEC_SPLIT) return;

    constexpr unsigned int FFT_SIZE = cufftdx::size_of<FFT>::value;

    unsigned int index = row * LOCAL_DIM + t + threadIdx.x * VEC_SPLIT;
    const unsigned int stride = FFT::stride * VEC_SPLIT;

    complex_type thread_data[FFT::storage_size];

    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];

    #pragma unroll
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
        if ((i * FFT::stride + threadIdx.x) < FFT_SIZE) {
            thread_data[i] = in_c[index];
            index += stride;
        }
    }

    FFT().execute(thread_data, shared_mem, workspace);

    index = row * LOCAL_DIM + t + threadIdx.x * VEC_SPLIT;
    #pragma unroll
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
        if ((i * FFT::stride + threadIdx.x) < FFT_SIZE) {
            out_c[index] = thread_data[i];
            index += stride;
        }
    }
}

#else // !__USE__FFTDX__

// Stub for when cuFFTDx is disabled
struct FftdxFft1Ctx {
    dim3 block, grid;
    size_t shmem;
};

#endif // __USE__FFTDX__

#endif // __ZMODEL__FFT__DEVICE__PLANS__
