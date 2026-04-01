#include <iostream>
#include "laplace.h"
#include "../transforms/fft.h"
#include <chrono>

/******************** MACROS ********************/

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

/******************** Init ********************/


void init_laplace_buffers(LaplaceBuffers& laplace_buffers) {
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&laplace_buffers.host_send_above, PACKED_ROW_COMPLEX_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&laplace_buffers.host_send_below, PACKED_ROW_COMPLEX_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&laplace_buffers.host_send_left, PACKED_COL_COMPLEX_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&laplace_buffers.host_send_right, PACKED_COL_COMPLEX_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&laplace_buffers.host_recv_above, PACKED_ROW_COMPLEX_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&laplace_buffers.host_recv_below, PACKED_ROW_COMPLEX_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&laplace_buffers.host_recv_left, PACKED_COL_COMPLEX_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&laplace_buffers.host_recv_right, PACKED_COL_COMPLEX_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&laplace_buffers.device_send_above, PACKED_ROW_COMPLEX_BYTES));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&laplace_buffers.device_send_below, PACKED_ROW_COMPLEX_BYTES));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&laplace_buffers.device_send_left, PACKED_COL_COMPLEX_BYTES));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&laplace_buffers.device_send_right, PACKED_COL_COMPLEX_BYTES));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&laplace_buffers.device_recv_above, PACKED_ROW_COMPLEX_BYTES));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&laplace_buffers.device_recv_below, PACKED_ROW_COMPLEX_BYTES));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&laplace_buffers.device_recv_left, PACKED_COL_COMPLEX_BYTES));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&laplace_buffers.device_recv_right, PACKED_COL_COMPLEX_BYTES));
}


void destroy_laplace_buffers(LaplaceBuffers& laplace_buffers) {
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(laplace_buffers.host_send_above));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(laplace_buffers.host_send_below));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(laplace_buffers.host_send_left));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(laplace_buffers.host_send_right));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(laplace_buffers.host_recv_above));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(laplace_buffers.host_recv_below));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(laplace_buffers.host_recv_left));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(laplace_buffers.host_recv_right));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(laplace_buffers.device_send_above));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(laplace_buffers.device_send_below));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(laplace_buffers.device_send_left));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(laplace_buffers.device_send_right));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(laplace_buffers.device_recv_above));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(laplace_buffers.device_recv_below));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(laplace_buffers.device_recv_left));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(laplace_buffers.device_recv_right));
} 

/******************** Laplace ********************/

/**
 * 2D matrix
 * 
 * 5 point laplacian stencil on complex data
 * 
 */
__global__ void laplace(Complex *device_in, Complex *device_out, 
                        LaplaceBuffers laplace_buffers, int rid, int cid) {
    int row = blockIdx.x / VEC_COL;
    int col = blockIdx.x % VEC_COL;
    int my_offset = (row * LOCAL_DIM) + (col * VEC);

    int vec_index = threadIdx.x;

    Complex scratch = {0.0, 0.0};

    // The vector sits at the top of the block it is in
    bool top_of_block = ((row % B_DIM) == 0);
    // The block the vector is in is the top row of blocks on the processor
    bool top_block = (row == 0);
    // The processor is in the top row of the grid of processors 
    bool top_p = (rid == 0);

    bool bottom_of_block = ((row % B_DIM) == B_DIM - 1);
    bool bottom_block = (row == LOCAL_DIM - 1);
    bool bottom_p = (rid == P_DIM - 1);

    bool rightmost_col = (col == VEC_COL - 1);
    bool rightmost_p = (cid == P_DIM - 1);

    bool leftmost_col = (col == 0);
    bool leftmost_p = (cid == 0);

    // Get row above
    // Assumes you are working with the packed bottom row of the block data from the processor above you
    int above_offset;
    if (top_of_block) {
        if (top_p) {
            // If you are the top processor then you must grab elements from the PACKED row above you
            if (top_block) {
                // If you are at the very top you need to wrap around because of boundary conditions
                above_offset = ((LOCAL_DIM/B_DIM - 1)*LOCAL_DIM) + (col * VEC);
            } else {
                above_offset = ((row/B_DIM - 1)*LOCAL_DIM) + (col * VEC);
            }
        } else {
            // If you are not the top processor then you are actually grabbing elements from the same PACKED row
            // Therefore there is no need to consider the case of wrap around
            above_offset = ((row/B_DIM)*LOCAL_DIM) + (col * VEC);
        }
        scratch += laplace_buffers.device_recv_above[above_offset + vec_index];
    } else {
        above_offset = ((row - 1) * LOCAL_DIM) + (col * VEC);
        scratch += device_in[above_offset + vec_index];
    }

    // Get row below
    // Assumes you are working with the packed top row of the block data from the processor below you
    int below_offset;
    if (bottom_of_block) {
        if (bottom_p) {
            if (bottom_block) {
                below_offset = (col * VEC);
            } else {
                below_offset = ((row/B_DIM + 1) * LOCAL_DIM) + (col * VEC);
            }
        } else {
            below_offset = ((row/B_DIM) * LOCAL_DIM) + (col * VEC);
        }
        scratch += laplace_buffers.device_recv_below[below_offset + vec_index];
    } else {
        below_offset = ((row + 1) * LOCAL_DIM) + (col * VEC);
        scratch += device_in[below_offset + vec_index];
    }

    // Get col to left
    // Assumes you are working with the packed rightmost col of the vec data from the processor to the left of you
    int left_offset;
    if (leftmost_col) {
        // Periodic boundary conditions force you to shift the vec to align elements when on leftmost processor
        left_offset = (row * VEC);
        int new_vec_index = leftmost_p ? (vec_index - 1 + VEC) % VEC : vec_index;
        scratch += laplace_buffers.device_recv_left[left_offset + new_vec_index];
    } else {
        left_offset = (row * LOCAL_DIM) + ((col - 1) * VEC);
        scratch += device_in[left_offset + vec_index];
    }

    // Get col to right
    // Assumes you are working with the packed leftmost col of the vec data from the processor to the right of you
    int right_offset;
    if (rightmost_col) {
        right_offset = (row * VEC);
        int new_vec_index = rightmost_p ? (vec_index + 1) % VEC : vec_index;
        scratch += laplace_buffers.device_recv_right[right_offset + new_vec_index];
    } else {
        right_offset = (row * LOCAL_DIM) + ((col + 1) * VEC);
        scratch += device_in[right_offset + vec_index];
    }

    scratch -= 4*device_in[my_offset + vec_index];
    device_out[my_offset + vec_index] = scratch;
}


__global__ void pack_laplace(Complex *device_in, LaplaceBuffers laplace_buffers) {
    int i = blockIdx.x;
    int j = threadIdx.x;

    // dy packing: only first LOCAL_DIM/B_DIM blocks, all LOCAL_DIM threads
    if (i < LOCAL_DIM / B_DIM) {
        laplace_buffers.device_send_above[i * LOCAL_DIM + j] = device_in[i * LOCAL_DIM * B_DIM + j];

        int bottom_offset = LOCAL_DIM * (B_DIM - 1);
        laplace_buffers.device_send_below[i * LOCAL_DIM + j] = device_in[bottom_offset + i * LOCAL_DIM * B_DIM + j];
    }

    // dx packing: all LOCAL_DIM blocks, only first VEC threads
    if (j < VEC) {
        laplace_buffers.device_send_left[i * VEC + j] = device_in[i * LOCAL_DIM + j];

        int right_offset = LOCAL_DIM - VEC;
        laplace_buffers.device_send_right[i * VEC + j] = device_in[right_offset + i * LOCAL_DIM + j];
    }
}


void communicate_laplace(LaplaceBuffers& laplace_buffers,
                         MPI_Comm& row_comm, MPI_Comm& col_comm, int rid, int cid) {
    int left  = (cid - 1 + P_DIM) % P_DIM;
    int right = (cid + 1)         % P_DIM;
    int up    = (rid - 1 + P_DIM) % P_DIM;
    int down  = (rid + 1)         % P_DIM;

    #ifdef __GPU__AWARE__MPI__
        MPI_Sendrecv(laplace_buffers.device_send_right, PACKED_COL_COMPLEX, MPI_C_DOUBLE_COMPLEX, right, 0,
                     laplace_buffers.device_recv_left, PACKED_COL_COMPLEX, MPI_C_DOUBLE_COMPLEX, left, 0,
                     row_comm, MPI_STATUS_IGNORE);
    #else
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(laplace_buffers.host_send_right, laplace_buffers.device_send_right, 
                                            PACKED_COL_COMPLEX_BYTES, MEM_COPY_DEVICE_TO_HOST));
        MPI_Sendrecv(laplace_buffers.host_send_right, PACKED_COL_COMPLEX, MPI_C_DOUBLE_COMPLEX, right, 0,
                     laplace_buffers.host_recv_left, PACKED_COL_COMPLEX, MPI_C_DOUBLE_COMPLEX, left, 0,
                     row_comm, MPI_STATUS_IGNORE);
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(laplace_buffers.device_recv_left, laplace_buffers.host_recv_left, 
                                            PACKED_COL_COMPLEX_BYTES, MEM_COPY_HOST_TO_DEVICE));
    #endif

    #ifdef __GPU__AWARE__MPI__
        MPI_Sendrecv(laplace_buffers.device_send_left, PACKED_COL_COMPLEX, MPI_C_DOUBLE_COMPLEX, left, 1,
                     laplace_buffers.device_recv_right, PACKED_COL_COMPLEX, MPI_C_DOUBLE_COMPLEX, right, 1,
                     row_comm, MPI_STATUS_IGNORE);
    #else
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(laplace_buffers.host_send_left, laplace_buffers.device_send_left, 
                                            PACKED_COL_COMPLEX_BYTES, MEM_COPY_DEVICE_TO_HOST));
        MPI_Sendrecv(laplace_buffers.host_send_left, PACKED_COL_COMPLEX, MPI_C_DOUBLE_COMPLEX, left, 1,
                     laplace_buffers.host_recv_right, PACKED_COL_COMPLEX, MPI_C_DOUBLE_COMPLEX, right, 1,
                     row_comm, MPI_STATUS_IGNORE);
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(laplace_buffers.device_recv_right, laplace_buffers.host_recv_right, 
                                            PACKED_COL_COMPLEX_BYTES, MEM_COPY_HOST_TO_DEVICE));
    #endif

    #ifdef __GPU__AWARE__MPI__
        MPI_Sendrecv(laplace_buffers.device_send_above, PACKED_ROW_COMPLEX, MPI_C_DOUBLE_COMPLEX, up, 2,
                     laplace_buffers.device_recv_below, PACKED_ROW_COMPLEX, MPI_C_DOUBLE_COMPLEX, down, 2,
                     col_comm, MPI_STATUS_IGNORE);
    #else
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(laplace_buffers.host_send_above, laplace_buffers.device_send_above,
                                            PACKED_ROW_COMPLEX_BYTES, MEM_COPY_DEVICE_TO_HOST));
        MPI_Sendrecv(laplace_buffers.host_send_above, PACKED_ROW_COMPLEX, MPI_C_DOUBLE_COMPLEX, up, 2,
                     laplace_buffers.host_recv_below, PACKED_ROW_COMPLEX, MPI_C_DOUBLE_COMPLEX, down, 2,
                     col_comm, MPI_STATUS_IGNORE);
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(laplace_buffers.device_recv_below, laplace_buffers.host_recv_below,
                                            PACKED_ROW_COMPLEX_BYTES, MEM_COPY_HOST_TO_DEVICE));
    #endif

    #ifdef __GPU__AWARE__MPI__
        MPI_Sendrecv(laplace_buffers.device_send_below, PACKED_ROW_COMPLEX, MPI_C_DOUBLE_COMPLEX, down, 3,
                     laplace_buffers.device_recv_above, PACKED_ROW_COMPLEX, MPI_C_DOUBLE_COMPLEX, up, 3,
                     col_comm, MPI_STATUS_IGNORE);
    #else
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(laplace_buffers.host_send_below, laplace_buffers.device_send_below, 
                                            PACKED_ROW_COMPLEX_BYTES, MEM_COPY_DEVICE_TO_HOST));
        MPI_Sendrecv(laplace_buffers.host_send_below, PACKED_ROW_COMPLEX, MPI_C_DOUBLE_COMPLEX, down, 3,
                     laplace_buffers.host_recv_above, PACKED_ROW_COMPLEX, MPI_C_DOUBLE_COMPLEX, up, 3,
                     col_comm, MPI_STATUS_IGNORE);
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(laplace_buffers.device_recv_above, laplace_buffers.host_recv_above,
                                            PACKED_ROW_COMPLEX_BYTES, MEM_COPY_HOST_TO_DEVICE));
    #endif
}

/******************** Test ********************/

void test_laplace() {
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
    
    FftBuffers fft_buffers;
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&fft_buffers.host_buf0, LOCAL_COMPLEX_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&fft_buffers.host_buf1, LOCAL_COMPLEX_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
    init_host(rid, cid, fft_buffers.host_buf0);
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&fft_buffers.device_buf0, LOCAL_COMPLEX_BYTES));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&fft_buffers.device_buf1, LOCAL_COMPLEX_BYTES));
    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(fft_buffers.device_buf0, fft_buffers.host_buf0, LOCAL_COMPLEX_BYTES, MEM_COPY_HOST_TO_DEVICE));
    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(fft_buffers.device_buf1, fft_buffers.host_buf1, LOCAL_COMPLEX_BYTES, MEM_COPY_HOST_TO_DEVICE));

    LaplaceBuffers laplace_buffers;
    init_laplace_buffers(laplace_buffers);

    for (int i = 0; i < RUNS; i++) {
        #ifdef __PRINT__TIMING__
            MPI_Barrier(MPI_COMM_WORLD);
            auto start = std::chrono::high_resolution_clock::now();
        #endif // __PRINT__TIMING__

        TIME_MPI_MAX_NS("PACK", id,
            pack_laplace<<<LOCAL_DIM, LOCAL_DIM>>>(fft_buffers.device_buf0, laplace_buffers);
            cudaDeviceSynchronize();
        );

        TIME_MPI_MAX_NS("COMMS", id,
            communicate_laplace(laplace_buffers, row_comm, col_comm, rid, cid);
            MPI_Barrier(MPI_COMM_WORLD);
        );
    
        TIME_MPI_MAX_NS("KERNEL", id,
            laplace<<<VEC_TOTAL, VEC>>>(fft_buffers.device_buf0, fft_buffers.device_buf1, laplace_buffers, rid, cid);
            cudaDeviceSynchronize();
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

    cudaMemcpy(fft_buffers.host_buf1, fft_buffers.device_buf1, LOCAL_COMPLEX_BYTES, cudaMemcpyDeviceToHost);

    #ifdef __PRINT__RESULTS__
        std::cout << "Host out" << std::endl;
        for (int i = 0; i < LOCAL_DIM; i++) {
            for (int j = 0; j < LOCAL_DIM; j++) {
                std::cout << fft_buffers.host_buf1[i * LOCAL_DIM + j].x << " ";
            }
            std::cout << std::endl;
        }
    #endif

    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(fft_buffers.host_buf0));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(fft_buffers.host_buf1));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(fft_buffers.device_buf0));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(fft_buffers.device_buf1));
    destroy_laplace_buffers(laplace_buffers);

    MPI_Finalize();
}