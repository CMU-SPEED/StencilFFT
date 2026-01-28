#ifndef __ZMODEL__FFT__DEVICE__PLANS__
#define __ZMODEL__FFT__DEVICE__PLANS__

#include "../utils.h"
#include <cufftdx.hpp>
using namespace cufftdx;

#define CUFFTDX_TARGET_SM 750

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

struct FftdxFft3Ctx {
    static constexpr int FFT_SIZE   = B_DIM * P_DIM;                 // n3
    static constexpr int BATCH      = LOCAL_DIM;                     // batch3
    static constexpr int STRIDE     = LOCAL_DIM;                     // istride3 / ostride3
    static constexpr int ROW_BLOCKS = LOCAL_DIM / (P_DIM * B_DIM);   // your loop bound

    // Tuneable parameter: how many of the BATCH FFTs a CTA handles at once (threadIdx.y)
    static constexpr int FFTS_PER_BLOCK = (BATCH >= 8 ? 8 : BATCH);

    using FFT = decltype(
        cufftdx::Size<FFT_SIZE>() +
        cufftdx::Precision<double>() +
        cufftdx::Type<cufftdx::fft_type::c2c>() +
        cufftdx::Direction<cufftdx::fft_direction::forward>() +
        cufftdx::FFTsPerBlock<FFTS_PER_BLOCK>() +
        cufftdx::SM<CUFFTDX_TARGET_SM>() +
        cufftdx::Block()
    );

    static_assert(FFT::block_dim.x * FFT::block_dim.y <= FFT::max_threads_per_block,
                  "Too many threads per block");

    typename FFT::workspace_type ws;
    dim3 block, grid;
    size_t shmem;
};

inline void init_fftdx_fft3(FftdxFft3Ctx& ctx3) {
    ctx3.block = FftdxFft3Ctx::FFT::block_dim;
    ctx3.grid  = dim3(
        FftdxFft3Ctx::ROW_BLOCKS,
        (FftdxFft3Ctx::BATCH + FftdxFft3Ctx::FFT::ffts_per_block - 1) / FftdxFft3Ctx::FFT::ffts_per_block
    );
    ctx3.shmem = FftdxFft3Ctx::FFT::shared_memory_size;

    cudaError_t err = cudaSuccess;
    ctx3.ws = cufftdx::make_workspace<typename FftdxFft3Ctx::FFT>(err);
    if (err != cudaSuccess) std::abort();
}

template<class FFT, int BATCH, int STRIDE, int ROW_BLOCKS>
__global__ void fft3_kernel(const Complex* __restrict__ in, Complex* __restrict__ out,
                            typename FFT::workspace_type workspace) {
    using complex_type = typename FFT::value_type;

    const complex_type* in_c  = reinterpret_cast<const complex_type*>(in);
    complex_type*       out_c = reinterpret_cast<complex_type*>(out);

    // This is your "row" loop index in the original code
    const unsigned int row_blk = blockIdx.x;
    if (row_blk >= (unsigned)ROW_BLOCKS) return;

    // Which batched FFT within this row-block (this corresponds to "b" in idist=1 layout)
    const unsigned int b = blockIdx.y * FFT::ffts_per_block + threadIdx.y;
    if (b >= (unsigned)BATCH) return;

    constexpr unsigned int FFT_SIZE = cufftdx::size_of<FFT>::value;

    // Each row_blk chunk is a (BATCH x FFT_SIZE) column-major matrix with leading dim STRIDE:
    // element k of this FFT is at: base + b + k*STRIDE
    const unsigned int chunk_base = row_blk * (BATCH * FFT_SIZE);

    unsigned int index = chunk_base + b + threadIdx.x * STRIDE;
    const unsigned int stride = FFT::stride * STRIDE;

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

    index = chunk_base + b + threadIdx.x * STRIDE;
    #pragma unroll
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
        if ((i * FFT::stride + threadIdx.x) < FFT_SIZE) {
            out_c[index] = thread_data[i];
            index += stride;
        }
    }
}


#endif // __ZMODEL__FFT__DEVICE__PLANS__