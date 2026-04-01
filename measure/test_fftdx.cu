#include <iostream>
#include <cufftdx.hpp>
using namespace cufftdx;

template<class FFT>
__global__ void block_fft_kernel(typename FFT::value_type* data,
                                 typename FFT::workspace_type workspace) {
    
    for (int i = 0; i < 1000; i++) {

        using complex_type = typename FFT::value_type;

        complex_type thread_data[FFT::storage_size];

        const unsigned int local_fft_id  = threadIdx.y;
        const unsigned int global_fft_id = (blockIdx.x * FFT::ffts_per_block) + local_fft_id;

        const unsigned int offset = cufftdx::size_of<FFT>::value * global_fft_id;
        const unsigned int stride = FFT::stride;

        

        unsigned int index = offset + threadIdx.x;
        for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
            if ((i * stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
                thread_data[i] = data[index];
                index += stride;
            }
        }

        extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];

        
        FFT().execute(thread_data, shared_mem, workspace);
        

        index = offset + threadIdx.x;
        for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
            if ((i * stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
                data[index] = thread_data[i];
                index += stride;
            }
        }

    }
}

int main(void) {
    using FFT = decltype(Size<128>() + Precision<double>() +
                        Type<fft_type::c2c>() +
                        Direction<fft_direction::forward>() +
                        FFTsPerBlock<1>() + ElementsPerThread<8>() +
                        SM<750>() + Block());

    using complex_type = typename FFT::value_type;

    complex_type* data;
    auto size       = FFT::ffts_per_block * cufftdx::size_of<FFT>::value;
    auto size_bytes = size * sizeof(complex_type);

    cudaMallocManaged(&data, size_bytes);

    for (size_t i = 0; i < size; i++) {
        data[i] = complex_type{double(i), -double(i)};
    }

    cudaError_t error_code = cudaSuccess;
    auto workspace = make_workspace<FFT>(error_code);

    for (int i = 0; i < 5; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        block_fft_kernel<FFT><<<1, FFT::block_dim, FFT::shared_memory_size>>>(data, workspace);
        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        std::cout << "TOTAL " << duration.count() << " ns\n" << std::endl;
    }
    
    for (int i = 0; i < 128; i++) {
        std::cout << "(" << data[i].x << ", " << data[i].y << ") ";
    }
    std::cout << std::endl;

    cudaFree(data);

    return 0;
}
