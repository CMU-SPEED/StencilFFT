#include "fft.h"

__global__ void pack_forward_all_to_all_row_3d(Complex *device_in, Complex *device_out) {
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

__global__ void pack_forward_fft0_3d(Complex *device_in, Complex *device_out) {
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

    device_out[(blockIdx.x * LOCAL_DIM * LOCAL_DIM) + (row * LOCAL_DIM) + col] = device_in[in_offset];
}

__global__ void pack_forward_all_to_all_col_3d(Complex *device_in, Complex *device_out) {
    const int split_size = LOCAL_DIM / P_DIM;
    const int slab_size = (LOCAL_DIM * LOCAL_DIM) / P_DIM;

    const int dest = blockIdx.y / split_size;
    const int row_in_chunk = blockIdx.y % split_size;

    const int in_offset = (blockIdx.x * LOCAL_DIM * LOCAL_DIM) + (blockIdx.y * LOCAL_DIM) + threadIdx.x;
    const int out_offset = (dest * slab_size * LOCAL_DIM) + (blockIdx.x * slab_size) + (row_in_chunk * LOCAL_DIM) + threadIdx.x;

    device_out[out_offset] = device_in[in_offset];
}

__global__ void pack_forward_fft1_3d(Complex *device_in, Complex *device_out) {
    const int split_size = LOCAL_DIM / P_DIM;
    const int slab_size = (LOCAL_DIM * LOCAL_DIM) / P_DIM;

    const int which_split = blockIdx.y / split_size;
    const int index_in_split = blockIdx.y % split_size;

    const int which_block = index_in_split / B_DIM;
    const int index_in_block = index_in_split % B_DIM;

    const int row = (which_block * P_DIM * B_DIM) + (which_split * B_DIM) + index_in_block;

    // Input: jump to proc chunk, then depth page, then row
    const int in_offset = (which_split * slab_size * LOCAL_DIM) + (blockIdx.x * slab_size) + (index_in_split * LOCAL_DIM) + threadIdx.x;

    device_out[(blockIdx.x * LOCAL_DIM * LOCAL_DIM) + (row * LOCAL_DIM) + threadIdx.x] = device_in[in_offset];
}

void test_fft_3d() {
    MPI_Init(NULL, NULL);

    int P, id;
    P = P_DIM * P_DIM * P_DIM;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    int rid, cid, did;
    did = id / (P_DIM * P_DIM);
    rid = (id % (P_DIM * P_DIM)) / P_DIM;
    cid = (id % (P_DIM * P_DIM)) % P_DIM;

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

    // FIXME: swap this out for modified init_buffers
    FftBuffers buffers;
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&buffers.host_buf0, LOCAL_COMPLEX_BYTES_3D, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&buffers.host_buf1, LOCAL_COMPLEX_BYTES_3D, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&buffers.device_buf0, LOCAL_COMPLEX_BYTES_3D));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&buffers.device_buf1, LOCAL_COMPLEX_BYTES_3D));

    // FIXME: integrate this into init_hosts
    size_t row_offset = cid * B_DIM;                    // offset based on which processor in the row
    size_t col_offset = rid * (N_DIM*B_DIM);            // offset based on which processor in the col
    size_t dep_offset = did * (N_DIM * N_DIM * B_DIM);  // offset based on which processor in depth
    size_t offset = row_offset + col_offset + dep_offset;

    for (size_t i = 0; i < LOCAL_DIM/B_DIM; i++) {
        for (size_t ii = 0; ii < B_DIM; ii++) {
            for (size_t j = 0; j < LOCAL_DIM/B_DIM; j++) {
                for (size_t jj = 0; jj < B_DIM; jj++) {
                    for (size_t k = 0; k < LOCAL_DIM/B_DIM; k++) {
                        for (size_t kk = 0; kk < B_DIM; kk++) {
                            size_t array_index = (i * B_DIM * LOCAL_DIM * LOCAL_DIM) + (ii * LOCAL_DIM * LOCAL_DIM) +
                                                 (j * B_DIM * LOCAL_DIM) + (jj * LOCAL_DIM) + (k * B_DIM) + kk;
                            double real = (i * N_DIM * N_DIM * B_DIM * P_DIM) + (ii * N_DIM * N_DIM) +
                                          (j * N_DIM * B_DIM * P_DIM) + (jj * N_DIM) +
                                          (k * B_DIM * P_DIM) + kk + offset;
                            buffers.host_buf0[array_index] = {real, 0.0};
                        }
                    }
                }
            }
        }
    }

    if (id == 0) {
        std::cout << "Block cyclic" << std::endl;
        for (int i = 0; i < LOCAL_DIM; i++) {
            for (int j = 0; j < LOCAL_DIM; j++) {
                for (int k = 0; k < LOCAL_DIM; k++) {
                    std::cout << buffers.host_buf0[(i * LOCAL_DIM * LOCAL_DIM)+ (j * LOCAL_DIM) + k].x << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            std::cout << std::endl;
        }
    }

    for (size_t i = 0; i < LOCAL_DIM; i++) {
        size_t depth = i * LOCAL_DIM * LOCAL_DIM;
        for (size_t ii = 0; ii < LOCAL_DIM; ii++) {
            size_t row = ii * LOCAL_DIM;
            for (size_t j = 0; j < B_DIM; j++) {
                for (size_t jj = 0; jj < LOCAL_DIM/B_DIM; jj++) {
                    buffers.host_buf1[depth + row + (j * (LOCAL_DIM/B_DIM))+ jj] = buffers.host_buf0[depth + row + j + (jj * B_DIM)];
                }
            }
        }
    }

    if (id == 0) {
        std::cout << "Initial packing" << std::endl;
        for (int i = 0; i < LOCAL_DIM; i++) {
            for (int j = 0; j < LOCAL_DIM; j++) {
                for (int k = 0; k < LOCAL_DIM; k++) {
                    std::cout << buffers.host_buf1[(i * LOCAL_DIM * LOCAL_DIM)+ (j * LOCAL_DIM) + k].x << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            std::cout << std::endl;
        }
    }

    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_buf0, buffers.host_buf0, LOCAL_COMPLEX_BYTES_3D, MEM_COPY_HOST_TO_DEVICE));
    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_buf1, buffers.host_buf1, LOCAL_COMPLEX_BYTES_3D, MEM_COPY_HOST_TO_DEVICE));

    dim3 grid0(LOCAL_DIM, VEC_TOTAL);
    pack_forward_all_to_all_row_3d<<<grid0, VEC>>>(buffers.device_buf1, buffers.device_buf0);
    DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());

    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.host_buf0, buffers.device_buf0, LOCAL_COMPLEX_BYTES_3D, MEM_COPY_DEVICE_TO_HOST));

    if (id == 0) {
        std::cout << "After pack_forward_all_to_all_row_3d" << std::endl;
        for (int i = 0; i < LOCAL_DIM; i++) {
            for (int j = 0; j < LOCAL_DIM; j++) {
                for (int k = 0; k < LOCAL_DIM; k++) {
                    std::cout << buffers.host_buf0[(i * LOCAL_DIM * LOCAL_DIM)+ (j * LOCAL_DIM) + k].x << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            std::cout << std::endl;
        }
    }

    MPI_Alltoall(buffers.host_buf0,
                 (LOCAL_DIM*LOCAL_DIM*LOCAL_DIM)/P_DIM,
                  MPI_C_DOUBLE_COMPLEX,
                  buffers.host_buf1, 
                 (LOCAL_DIM*LOCAL_DIM*LOCAL_DIM)/P_DIM,
                  MPI_C_DOUBLE_COMPLEX,
                  row_comm);

    if (id == 0) {
        std::cout << "After All to All Rows" << std::endl;
        for (int i = 0; i < LOCAL_DIM; i++) {
            for (int j = 0; j < LOCAL_DIM; j++) {
                for (int k = 0; k < LOCAL_DIM; k++) {
                    std::cout << buffers.host_buf1[(i * LOCAL_DIM * LOCAL_DIM)+ (j * LOCAL_DIM) + k].x << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            std::cout << std::endl;
        }
    }

    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_buf1, buffers.host_buf1, LOCAL_COMPLEX_BYTES_3D, MEM_COPY_HOST_TO_DEVICE));

    dim3 grid1(LOCAL_DIM, LOCAL_DIM);
    pack_forward_fft0_3d<<<grid1, LOCAL_DIM>>>(buffers.device_buf1, buffers.device_buf0);
    DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());

    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.host_buf0, buffers.device_buf0, LOCAL_COMPLEX_BYTES_3D, MEM_COPY_DEVICE_TO_HOST));

    if (id == 0) {
        std::cout << "After pack_forward_fft0_3d" << std::endl;
        for (int i = 0; i < LOCAL_DIM; i++) {
            for (int j = 0; j < LOCAL_DIM; j++) {
                for (int k = 0; k < LOCAL_DIM; k++) {
                    std::cout << buffers.host_buf0[(i * LOCAL_DIM * LOCAL_DIM)+ (j * LOCAL_DIM) + k].x << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            std::cout << std::endl;
        }
    }

    dim3 grid2(LOCAL_DIM, LOCAL_DIM);
    pack_forward_all_to_all_col_3d<<<grid2, LOCAL_DIM>>>(buffers.device_buf0, buffers.device_buf1);
    DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());

    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.host_buf0, buffers.device_buf1, LOCAL_COMPLEX_BYTES_3D, MEM_COPY_DEVICE_TO_HOST));

    if (id == 0) {
        std::cout << "After pack_forward_all_to_all_col_3d" << std::endl;
        for (int i = 0; i < LOCAL_DIM; i++) {
            for (int j = 0; j < LOCAL_DIM; j++) {
                for (int k = 0; k < LOCAL_DIM; k++) {
                    std::cout << buffers.host_buf0[(i * LOCAL_DIM * LOCAL_DIM)+ (j * LOCAL_DIM) + k].x << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            std::cout << std::endl;
        }
    }

    MPI_Alltoall(buffers.host_buf0,
                 (LOCAL_DIM*LOCAL_DIM*LOCAL_DIM)/P_DIM,
                  MPI_C_DOUBLE_COMPLEX,
                  buffers.host_buf1,
                 (LOCAL_DIM*LOCAL_DIM*LOCAL_DIM)/P_DIM,
                  MPI_C_DOUBLE_COMPLEX,
                  col_comm);

    if (id == 0) {
        std::cout << "After All to All Cols" << std::endl;
        for (int i = 0; i < LOCAL_DIM; i++) {
            for (int j = 0; j < LOCAL_DIM; j++) {
                for (int k = 0; k < LOCAL_DIM; k++) {
                    std::cout << buffers.host_buf1[(i * LOCAL_DIM * LOCAL_DIM)+ (j * LOCAL_DIM) + k].x << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            std::cout << std::endl;
        }
    }

    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_buf1, buffers.host_buf1, LOCAL_COMPLEX_BYTES_3D, MEM_COPY_HOST_TO_DEVICE));

    dim3 grid3(LOCAL_DIM, LOCAL_DIM);
    pack_forward_fft1_3d<<<grid3, LOCAL_DIM>>>(buffers.device_buf1, buffers.device_buf0);
    DEVICE_RT_SAFE_CALL(DEVICE_SYNCHRONIZE());

    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.host_buf0, buffers.device_buf0, LOCAL_COMPLEX_BYTES_3D, MEM_COPY_DEVICE_TO_HOST));

    if (id == 0) {
        std::cout << "After pack_forward_fft1_3d" << std::endl;
        for (int i = 0; i < LOCAL_DIM; i++) {
            for (int j = 0; j < LOCAL_DIM; j++) {
                for (int k = 0; k < LOCAL_DIM; k++) {
                    std::cout << buffers.host_buf0[(i * LOCAL_DIM * LOCAL_DIM)+ (j * LOCAL_DIM) + k].x << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            std::cout << std::endl;
        }
    }

    // Depth all-to-all: no packing needed since depth is outermost
    MPI_Alltoall(buffers.host_buf0,
                 (LOCAL_DIM*LOCAL_DIM*LOCAL_DIM)/P_DIM,
                  MPI_C_DOUBLE_COMPLEX,
                  buffers.host_buf1,
                 (LOCAL_DIM*LOCAL_DIM*LOCAL_DIM)/P_DIM,
                  MPI_C_DOUBLE_COMPLEX,
                  dep_comm);

    if (id == 0) {
        std::cout << "After All to All Depth" << std::endl;
        for (int i = 0; i < LOCAL_DIM; i++) {
            for (int j = 0; j < LOCAL_DIM; j++) {
                for (int k = 0; k < LOCAL_DIM; k++) {
                    std::cout << buffers.host_buf1[(i * LOCAL_DIM * LOCAL_DIM)+ (j * LOCAL_DIM) + k].x << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            std::cout << std::endl;
        }
    }

    // FIXME: swap this out for modified destroy buffers
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(buffers.host_buf0));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(buffers.host_buf1));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(buffers.device_buf0));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(buffers.device_buf1));

    MPI_Finalize();
}