#include <mpi.h>
#include "utils.h"
#include <iostream>
#include <chrono>

// FIXME: use device macros

__global__ void pack_all_to_all_row(Complex *device_in, Complex *device_out) {
    const int row = blockIdx.x / vec_col;
    const int col = blockIdx.x % vec_col;
    const int in_offset = (row * local) + (col * vec);

    const int vec_index = threadIdx.x;        // Index into the vector

    const int vec_split = vec / p;            // The size of a split vector
    const int split = vec_index / vec_split;  // Which section of the split vector
    const int index_in_split = vec_index % vec_split; // Index into the split vector

    // Note these are both measured in element space
    const int slab_size = (local*local) / p;      // Num elements in a slab of splits
    const int slab_vec_row = local / p;           // Number of split vectors in a col of a slab
    const int slab_vec_col = local / vec_split;   // Number of split vectors in a row of a slab/local

    const int slab_offset = split * slab_size;
    const int out_row = (blockIdx.x / slab_vec_col) % slab_vec_row;
    const int out_col = blockIdx.x % slab_vec_col;
    const int out_offset = slab_offset + (out_row * local) + (out_col * vec_split);

    device_out[out_offset + index_in_split] = device_in[in_offset + vec_index];
}

__global__ void pack_fft_1(Complex *device_in, Complex *device_out) {
    const int split_size = local / p;                            // The size of the split

    const int index_in_row_split = blockIdx.x % split_size;
    const int index_in_vec_split = threadIdx.x % split_size;
    
    const int row_split = blockIdx.x / split_size;       // Which section of the row split
    const int vec_split = threadIdx.x / split_size;      // Which section of the split local vector

    // Swap the vec and row splits
    const int row = vec_split + (index_in_row_split * p);
    const int col = (row_split * split_size) + index_in_vec_split;

    device_out[row * local + col] = device_in[blockIdx.x * local + threadIdx.x];
}

__global__ void pack_fft_2(Complex *device_in, Complex *device_out) {
    const int split_size = local / p;

    const int which_split = blockIdx.x / split_size;
    const int index_in_split = blockIdx.x % split_size;

    const int which_block = index_in_split / b;
    const int index_in_block = index_in_split % b;

    const int row = (which_block * p * b) + (which_split * b) + index_in_block;

    device_out[row * local + threadIdx.x] = device_in[blockIdx.x * local + threadIdx.x];
}

// FIXME: apply the twiddles in the packing routines

void init_twiddles0(Complex *in, int cid) {
    for (int local_row = 0; local_row < local; local_row++) {
        for (int fft_row = 0; fft_row < b; fft_row++) {
            for (int fft_col = 0; fft_col < local/b; fft_col++) {
                int fft_row_offset = cid * b;                           // FIXME: is b the correct offet here?
                double k = (double)(fft_row + fft_row_offset);
                double l = (double)fft_col;
                in[(local_row * local) + fft_row + (fft_col * b)] = {k, l};
                // in[(local_row * local) + fft_row + (fft_col * b)] = std::exp(Complex(0.0, -2*M_PI*k*l/local));
            }
        }
    }
}

void init_twiddles1(Complex *in, int rid) {
    for (int fft_row = 0; fft_row < (local*local) / (local*b*p); fft_row++) {
        for (int fft_col = 0; fft_col < b*p; fft_col++) {
            for (int local_col = 0; local_col < local; local_col++) {
                int fft_row_offset = rid * b;                       // FIXME: i dont think this is correct
                double k = (double)(fft_row + fft_row_offset);
                double l = (double)fft_col;
                in[fft_row * (local*b*p) + fft_col * local + local_col] = {k , l};
            }
        }
    }
}

int main(void) {
    MPI_Init(NULL, NULL);

    int P, id;
    P = p * p;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    MPI_Comm row_comm, col_comm;

    int rid = id / p;
    MPI_Comm_split(MPI_COMM_WORLD, rid, id, &row_comm);

    int cid = id % p;
    MPI_Comm_split(MPI_COMM_WORLD, cid, id, &col_comm);

    cufftHandle plan0, plan1, plan2, plan3;
    init_plans(&plan0, &plan1, &plan2, &plan3);

    int local_size = local*local*sizeof(Complex);

    // FIXME: make these cuda malloc
    Complex *host_in = (Complex*)malloc(local_size);
    Complex *host_out = (Complex*)calloc(local*local, sizeof(Complex));
    Complex *twiddles0 = (Complex*)calloc(local*local, sizeof(Complex));
    Complex *twiddles1 = (Complex*)calloc(local*local, sizeof(Complex));

    init_host(cid, rid, local_size, host_in);   // FIXME: the rid and cid args are swapped in this
    init_twiddles0(twiddles0, cid);
    init_twiddles1(twiddles1, rid);

    if (id == 0) {
        std::cout << "Twiddles" << std::endl;
        std::cout << "rid: " << rid << " " << "cid: " << cid << std::endl;
        for (int i = 0; i < local; i++) {
            for (int j = 0; j < local; j++) {
                std::cout << "(" << twiddles1[i * local + j].x << ", " << twiddles1[i * local + j].y << ") ";
            }
            std::cout << std::endl;
        }
    }

    Complex *device_in, *device_out;
    cudaMalloc((void**)&device_in, local_size);
    cudaMalloc((void**)&device_out, local_size);

    auto start = std::chrono::high_resolution_clock::now();

    cudaMemcpy(device_in, host_in, local_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_out, host_out, local_size, cudaMemcpyHostToDevice);

    // FIXME: Doing this in place might be slower?
    // cufftExecZ2Z(plan0, device_in, device_in, CUFFT_FORWARD);

    pack_all_to_all_row<<<vec_total, vec>>>(device_in, device_out);
    cudaDeviceSynchronize();

    cudaMemcpy(host_out, device_out, local_size, cudaMemcpyDeviceToHost);

    // Twiddles

    MPI_Alltoall(host_out,
                (local * local) / p,
                MPI_C_DOUBLE_COMPLEX,
                host_in, 
                (local * local) / p,
                MPI_C_DOUBLE_COMPLEX,
                row_comm);

    // for (int i = 0; i < local; i++)
    //     for (int j = 0; j< local; j++)
    //         host_out[i * local + j] = 0.0;
    
    cudaMemcpy(device_in, host_in, local_size, cudaMemcpyHostToDevice);
    // cudaMemcpy(device_out, host_out, local_size, cudaMemcpyHostToDevice); // FIXME: remove, this is for debug

    pack_fft_1<<<local, local>>>(device_in, device_out);
    cudaDeviceSynchronize();

    // FIXME: can we parallelize this
    // for (int row = 0; row < local; row++) {
    //     Complex* row_ptr = device_out + row * local;
    //     cufftExecZ2Z(plan1, row_ptr, row_ptr, CUFFT_FORWARD);
    // }

    // cufftExecZ2Z(plan2, device_out, device_out, CUFFT_FORWARD);

    cudaMemcpy(host_out, device_out, local_size, cudaMemcpyDeviceToHost);

    MPI_Alltoall(host_out,
                (local * local) / p,
                MPI_C_DOUBLE_COMPLEX,
                host_in,
                (local * local) / p,
                MPI_C_DOUBLE_COMPLEX,
                col_comm);

    // for (int i = 0; i < local; i++)
    //     for (int j = 0; j< local; j++)
    //         host_out[i * local + j] = 0.0;
    
    cudaMemcpy(device_in, host_in, local_size, cudaMemcpyHostToDevice);
    // cudaMemcpy(device_out, host_out, local_size, cudaMemcpyHostToDevice); // FIXME: remove, this is for debug

    pack_fft_2<<<local, local>>>(device_in, device_out);
    cudaDeviceSynchronize();

    // for (int row = 0; row < local/(p*b); row++) {
    //     Complex* row_ptr = device_out + row * local * p * b;
    //     cufftExecZ2Z(plan3, row_ptr, row_ptr, CUFFT_FORWARD);
    // }

    cudaMemcpy(host_out, device_out, local_size, cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    double ns = duration.count();
    if (id == 0) std::cout << "FFT: " << ns << " ns" << std::endl;

    if (id == 0) {
        std::cout << "Host out" << std::endl;
        for (int i = 0; i < local; i++) {
            for (int j = 0; j < local; j++) {
                std::cout << host_out[i * local + j].x << " ";
            }
            std::cout << std::endl;
        }
    }    

    free(host_in);
    free(host_out);
    free(twiddles0);
    free(twiddles1);
    cudaFree(device_in);
    cudaFree(device_out);

    cufftDestroy(plan0);
    cufftDestroy(plan1);
    cufftDestroy(plan2);

    MPI_Finalize();
	
    return 0;
}