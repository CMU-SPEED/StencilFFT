#include <mpi.h>
#include "utils.h"
#include <chrono>
#include <complex>

// FIXME: Inverse FFT
// Header file + finish cleanup
// FIXME: NULL pointer checks for inputs
// FIXME: Check pointer function declaration convention

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

__global__ void pack_forward_fft0(Complex *device_in, Complex *device_out, Complex *forward_twiddles0) {
    const int split_size = LOCAL_DIM / P_DIM;                // The size of the split

    const int index_in_row_split = blockIdx.x % split_size;
    const int index_in_vec_split = threadIdx.x % split_size;
    
    const int row_split = blockIdx.x / split_size;          // Which section of the row split
    const int vec_split = threadIdx.x / split_size;         // Which section of the split local vector

    // Swap the vec and row splits
    const int row = vec_split + (index_in_row_split * P_DIM);
    const int col = (row_split * split_size) + index_in_vec_split;

    // FIXME: THIS if statement breaks things

    // #if defined(__PRINT__TWIDDLES__) || !defined(__ZMODEL__COMPUTE___)
    // device_out[row * LOCAL_DIM + col] = device_in[blockIdx.x * LOCAL_DIM + threadIdx.x];
    // #else
    // FIXME: not sure if I should use index from input or output?
    // FIMXE: I think this should use the output index
    device_out[row * LOCAL_DIM + col] = DEVICE_FFT_DOUBLECOMPLEX_MUL(device_in[blockIdx.x * LOCAL_DIM + threadIdx.x],
                                                                     forward_twiddles0[row * LOCAL_DIM + col]);
    // #endif
}

__global__ void pack_forward_fft1(Complex *device_in, Complex *device_out, Complex *forward_twiddles1) {
    const int split_size = LOCAL_DIM / P_DIM;

    const int which_split = blockIdx.x / split_size;
    const int index_in_split = blockIdx.x % split_size;

    const int which_block = index_in_split / B_DIM;
    const int index_in_block = index_in_split % B_DIM;

    const int row = (which_block * P_DIM * B_DIM) + (which_split * B_DIM) + index_in_block;

    // #if defined(__PRINT__TWIDDLES__) || !defined(__ZMODEL__COMPUTE___)
    // device_out[row * LOCAL_DIM + threadIdx.x] = device_in[blockIdx.x * LOCAL_DIM + threadIdx.x];
    // #else
    // FIXME: I think this is the right way to apply it --> same index in input and twiddles?
    device_out[row * LOCAL_DIM + threadIdx.x] = DEVICE_FFT_DOUBLECOMPLEX_MUL(device_in[blockIdx.x * LOCAL_DIM + threadIdx.x],
                                                                            forward_twiddles1[row * LOCAL_DIM + threadIdx.x]);
    // #endif
}

void init_forward_twiddles0(Complex *in, int cid) {
    int stride = VEC/P_DIM; // This is the processor offset
    for (int local_row = 0; local_row < LOCAL_DIM; local_row++) {
        for (int fft_row = 0; fft_row < stride; fft_row++) {
            for (int fft_col = 0; fft_col < LOCAL_DIM/stride; fft_col++) {
                int fft_row_offset = cid * stride;
                double k = (double)(fft_row + fft_row_offset);
                double l = (double)fft_col;
                #if defined(__PRINT__TWIDDLES__)
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
                #if defined(__PRINT__TWIDDLES__)
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
 * 
 * Data starts in device_buf0
 * Data ends in device_buf0
 *
 * Host buffers and memcopy are needed because we dont have GPU aware MPI
 */
void forward_fft(Complex *host_buf0, Complex *host_buf1, Complex *device_buf0, Complex *device_buf1,
                 Complex *device_twiddles0, Complex *device_twiddles1, MPI_Comm *row_comm, MPI_Comm *col_comm,
                 cufftHandle *plan0, cufftHandle *plan1, cufftHandle *plan2, cufftHandle *plan3) {
    #ifdef __ZMODEL__COMPUTE__

    #ifdef __PRINT__TIMING__
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    #endif

    DEVICE_FFT_SAFE_CALL(DEVICE_FFT_EXECZ2Z(*plan0, device_buf0, device_buf1, DEVICE_FFT_FORWARD));
    // placeholder<<<LOCAL_DIM, LOCAL_DIM>>>(device_buf0, device_buf1);
    // DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());

    pack_forward_all_to_all_row<<<VEC_TOTAL, VEC>>>(device_buf1, device_buf0);
    DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());

    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(host_buf1, device_buf0, LOCAL_BYTES, MEM_COPY_DEVICE_TO_HOST));

    MPI_Alltoall(host_buf1,
                (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                MPI_C_DOUBLE_COMPLEX,
                host_buf0, 
                (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                MPI_C_DOUBLE_COMPLEX,
                *row_comm);

    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(device_buf0, host_buf0, LOCAL_BYTES, MEM_COPY_HOST_TO_DEVICE));
    
    pack_forward_fft0<<<LOCAL_DIM, LOCAL_DIM>>>(device_buf0, device_buf1, device_twiddles0);
    DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());

    // FIXME: can we parallelize this with streams --> benchmark to see if this is a bottleneck first
    for (int row = 0; row < LOCAL_DIM; row++) {
        Complex* row_ptr0 = device_buf1 + row * LOCAL_DIM;
        Complex* row_ptr1 = device_buf0 + row * LOCAL_DIM;
        DEVICE_FFT_SAFE_CALL(DEVICE_FFT_EXECZ2Z(*plan1, row_ptr0, row_ptr1, DEVICE_FFT_FORWARD));
    }
    // placeholder<<<LOCAL_DIM, LOCAL_DIM>>>(device_buf1, device_buf0);
    // DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());

    DEVICE_FFT_SAFE_CALL(DEVICE_FFT_EXECZ2Z(*plan2, device_buf0, device_buf1, DEVICE_FFT_FORWARD));
    // placeholder<<<LOCAL_DIM, LOCAL_DIM>>>(device_buf0, device_buf1);
    // DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());

    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(host_buf1, device_buf1, LOCAL_BYTES, MEM_COPY_DEVICE_TO_HOST));

    MPI_Alltoall(host_buf1,
                (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                MPI_C_DOUBLE_COMPLEX,
                host_buf0,
                (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                MPI_C_DOUBLE_COMPLEX,
                *col_comm);

    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(device_buf0, host_buf0, LOCAL_BYTES, MEM_COPY_HOST_TO_DEVICE));

    pack_forward_fft1<<<LOCAL_DIM, LOCAL_DIM>>>(device_buf0, device_buf1, device_twiddles1);
    DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());

    for (int row = 0; row < LOCAL_DIM/(P_DIM*B_DIM); row++) {
        Complex* row_ptr0 = device_buf1 + row * LOCAL_DIM * P_DIM * B_DIM;
        Complex* row_ptr1 = device_buf0 + row * LOCAL_DIM * P_DIM * B_DIM;
        DEVICE_FFT_SAFE_CALL(DEVICE_FFT_EXECZ2Z(*plan3, row_ptr0, row_ptr1, DEVICE_FFT_FORWARD));
    }
    // placeholder<<<LOCAL_DIM, LOCAL_DIM>>>(device_buf1, device_buf0);
    // DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());

    #ifdef __PRINT__TIMING__
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    double ns = duration.count();
    std::cout << "FFT: " << ns << " ns" << std::endl;
    #endif // __PRINT__TIMING__

    #else // __ZMODEL__COMPUTE__

    pack_forward_all_to_all_row<<<VEC_TOTAL, VEC>>>(device_buf0, device_buf1);
    DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());

    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(host_buf1, device_buf1, LOCAL_BYTES, MEM_COPY_DEVICE_TO_HOST));

    MPI_Alltoall(host_buf1,
                (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                MPI_C_DOUBLE_COMPLEX,
                host_buf0, 
                (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                MPI_C_DOUBLE_COMPLEX,
                *row_comm);

    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(device_buf0, host_buf0, LOCAL_BYTES, MEM_COPY_HOST_TO_DEVICE));

    pack_forward_fft0<<<LOCAL_DIM, LOCAL_DIM>>>(device_buf0, device_buf1, device_twiddles0);
    DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());

    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(host_buf1, device_buf1, LOCAL_BYTES, MEM_COPY_DEVICE_TO_HOST));

    MPI_Alltoall(host_buf1,
                (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                MPI_C_DOUBLE_COMPLEX,
                host_buf0,
                (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                MPI_C_DOUBLE_COMPLEX,
                *col_comm);

    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(device_buf1, host_buf0, LOCAL_BYTES, MEM_COPY_HOST_TO_DEVICE));

    pack_forward_fft1<<<LOCAL_DIM, LOCAL_DIM>>>(device_buf1, device_buf0, device_twiddles1);
    DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());

    #endif // __ZMODEL__COMPUTE__
}

__global__ void pack_inverse_all_to_all_row(Complex *device_in, Complex *device_out) {
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

    device_out[in_offset + vec_index] = device_in[out_offset + index_in_split];
}

__global__ void pack_inverse_fft0(Complex *device_in, Complex *device_out, Complex *inverse_twiddles0) {
    const int split_size = LOCAL_DIM / P_DIM;                // The size of the split

    const int index_in_row_split = blockIdx.x % split_size;
    const int index_in_vec_split = threadIdx.x % split_size;
    
    const int row_split = blockIdx.x / split_size;          // Which section of the row split
    const int vec_split = threadIdx.x / split_size;         // Which section of the split local vector

    // Swap the vec and row splits
    const int row = vec_split + (index_in_row_split * P_DIM);
    const int col = (row_split * split_size) + index_in_vec_split;

    device_out[blockIdx.x * LOCAL_DIM + threadIdx.x] = device_in[row * LOCAL_DIM + col];
}

__global__ void pack_inverse_fft1(Complex *device_in, Complex *device_out, Complex *inverse_twiddles1) {
    const int split_size = LOCAL_DIM / P_DIM;

    const int which_split = blockIdx.x / split_size;
    const int index_in_split = blockIdx.x % split_size;

    const int which_block = index_in_split / B_DIM;
    const int index_in_block = index_in_split % B_DIM;

    const int row = (which_block * P_DIM * B_DIM) + (which_split * B_DIM) + index_in_block;

    device_out[blockIdx.x * LOCAL_DIM + threadIdx.x] = device_in[row * LOCAL_DIM + threadIdx.x];
}

void inverse_fft(Complex *host_buf0, Complex *host_buf1, Complex *device_buf0, Complex *device_buf1,
                 Complex *device_twiddles0, Complex *device_twiddles1, MPI_Comm *row_comm, MPI_Comm *col_comm,
                 cufftHandle *plan0, cufftHandle *plan1, cufftHandle *plan2, cufftHandle *plan3) {

    // #ifdef __ZMODEL__COMPUTE__

    // TODO!

    // #else

    pack_inverse_fft1<<<LOCAL_DIM, LOCAL_DIM>>>(device_buf0, device_buf1, device_twiddles1);

    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(host_buf1, device_buf1, LOCAL_BYTES, MEM_COPY_DEVICE_TO_HOST));

    MPI_Alltoall(host_buf1,
                (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                MPI_C_DOUBLE_COMPLEX,
                host_buf0,
                (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                MPI_C_DOUBLE_COMPLEX,
                *col_comm);

    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(device_buf0, host_buf0, LOCAL_BYTES, MEM_COPY_HOST_TO_DEVICE));

    pack_inverse_fft0<<<LOCAL_DIM, LOCAL_DIM>>>(device_buf0, device_buf1, device_twiddles0);

    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(host_buf1, device_buf1, LOCAL_BYTES, MEM_COPY_DEVICE_TO_HOST));

    MPI_Alltoall(host_buf1,
                (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                MPI_C_DOUBLE_COMPLEX,
                host_buf0, 
                (LOCAL_DIM*LOCAL_DIM)/P_DIM,
                MPI_C_DOUBLE_COMPLEX,
                *row_comm);
    
    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(device_buf1, host_buf0, LOCAL_BYTES, MEM_COPY_HOST_TO_DEVICE));

    pack_inverse_all_to_all_row<<<VEC_TOTAL, VEC>>>(device_buf1, device_buf0);

    // #endif // __ZMODEL__COMPUTE__
}

int main(void) {
    MPI_Init(NULL, NULL);

    int P, id;
    P = P_DIM * P_DIM;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    #ifdef __PRINT__SANITY__
    if (id == 0) std::cout << "N_DIM:" << N_DIM << ", B_DIM:" << B_DIM << " P_DIM:" << P_DIM << std::endl;
    #endif

    MPI_Comm row_comm, col_comm;

    int rid = id / P_DIM;
    MPI_Comm_split(MPI_COMM_WORLD, rid, id, &row_comm);

    int cid = id % P_DIM;
    MPI_Comm_split(MPI_COMM_WORLD, cid, id, &col_comm);

    cufftHandle plan0, plan1, plan2, plan3;
    init_plans(&plan0, &plan1, &plan2, &plan3);

    Complex *host_buf0, *host_buf1, *host_twiddles0, *host_twiddles1;
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&host_buf0, LOCAL_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&host_buf1, LOCAL_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&host_twiddles0, LOCAL_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&host_twiddles1, LOCAL_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
    init_host(rid, cid, host_buf0);
    init_forward_twiddles0(host_twiddles0, cid);
    init_forward_twiddles1(host_twiddles1, rid);

    #ifdef __PRINT__TWIDDLES__
    if (id == 0) {
        std::cout << "Twiddles" << std::endl;
        std::cout << "rid: " << rid << " " << "cid: " << cid << std::endl;
        for (int i = 0; i < LOCAL_DIM; i++) {
            for (int j = 0; j < LOCAL_DIM; j++) {
                std::cout << "(" << host_twiddles1[i * LOCAL_DIM + j].x << ", " << host_twiddles1[i * LOCAL_DIM + j].y << ") ";
            }
            std::cout << std::endl;
        }
    }
    #endif

    Complex *device_buf0, *device_buf1, *device_twiddles0, *device_twiddles1;
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&device_buf0, LOCAL_BYTES));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&device_buf1, LOCAL_BYTES));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&device_twiddles0, LOCAL_BYTES));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&device_twiddles1, LOCAL_BYTES));
    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(device_buf0, host_buf0, LOCAL_BYTES, MEM_COPY_HOST_TO_DEVICE));
    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(device_buf1, host_buf1, LOCAL_BYTES, MEM_COPY_HOST_TO_DEVICE));
    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(device_twiddles0, host_twiddles0, LOCAL_BYTES, MEM_COPY_HOST_TO_DEVICE));
    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(device_twiddles1, host_twiddles1, LOCAL_BYTES, MEM_COPY_HOST_TO_DEVICE));

    forward_fft(host_buf0, host_buf1, device_buf0, device_buf1,
                 device_twiddles0, device_twiddles1, 
                 &row_comm, &col_comm, &plan0, &plan1, &plan2, &plan3);

    // inverse_fft(host_buf0, host_buf1, device_buf0, device_buf1,
    //              NULL, NULL, 
    //              &row_comm, &col_comm, NULL, NULL, NULL, NULL);

    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(host_buf1, device_buf0, LOCAL_BYTES, MEM_COPY_DEVICE_TO_HOST));

    #ifdef __PRINT__RESULTS__
    MPI_Barrier(MPI_COMM_WORLD);
    if (id == 0) {
        std::cout << "Host out" << std::endl;
        for (int i = 0; i < LOCAL_DIM; i++) {
            for (int j = 0; j < LOCAL_DIM; j++) {
                std::cout << host_buf1[i * LOCAL_DIM + j].x << " ";
            }
            std::cout << std::endl;
        }
    }
    #endif

    // MPI_Barrier(MPI_COMM_WORLD);
    // if (id == 0) {
        // print_block_cyclic(host_buf1);
    // }

    DEVICE_FFT_SAFE_CALL(DEVICE_FFT_DESTROY(plan0));
    DEVICE_FFT_SAFE_CALL(DEVICE_FFT_DESTROY(plan1));
    DEVICE_FFT_SAFE_CALL(DEVICE_FFT_DESTROY(plan2));
    DEVICE_FFT_SAFE_CALL(DEVICE_FFT_DESTROY(plan3));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(host_buf0));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(host_buf1));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(host_twiddles0));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(host_twiddles1));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(device_buf0));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(device_buf1));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(device_twiddles0));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(device_twiddles1));

    MPI_Finalize();
	
    return 0;
}