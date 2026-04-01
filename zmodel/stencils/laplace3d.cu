#include <iostream>
#include "laplace.h"
#include "../transforms/fft.h"

void init_laplace_buffers_3d(LaplaceBuffers3d& laplace_buffers) {
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&laplace_buffers.host_send_above, PACKED_FACE_COMPLEX_BYTES_3D, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&laplace_buffers.host_send_below, PACKED_FACE_COMPLEX_BYTES_3D, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&laplace_buffers.host_send_left, PACKED_FACE_COMPLEX_BYTES_3D, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&laplace_buffers.host_send_right, PACKED_FACE_COMPLEX_BYTES_3D, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&laplace_buffers.host_send_front, PACKED_FACE_COMPLEX_BYTES_3D, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&laplace_buffers.host_send_back, PACKED_FACE_COMPLEX_BYTES_3D, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&laplace_buffers.host_recv_above, PACKED_FACE_COMPLEX_BYTES_3D, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&laplace_buffers.host_recv_below, PACKED_FACE_COMPLEX_BYTES_3D, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&laplace_buffers.host_recv_left, PACKED_FACE_COMPLEX_BYTES_3D, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&laplace_buffers.host_recv_right, PACKED_FACE_COMPLEX_BYTES_3D, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&laplace_buffers.host_recv_front, PACKED_FACE_COMPLEX_BYTES_3D, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&laplace_buffers.host_recv_back, PACKED_FACE_COMPLEX_BYTES_3D, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&laplace_buffers.device_send_above, PACKED_FACE_COMPLEX_BYTES_3D));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&laplace_buffers.device_send_below, PACKED_FACE_COMPLEX_BYTES_3D));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&laplace_buffers.device_send_left, PACKED_FACE_COMPLEX_BYTES_3D));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&laplace_buffers.device_send_right, PACKED_FACE_COMPLEX_BYTES_3D));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&laplace_buffers.device_send_front, PACKED_FACE_COMPLEX_BYTES_3D));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&laplace_buffers.device_send_back, PACKED_FACE_COMPLEX_BYTES_3D));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&laplace_buffers.device_recv_above, PACKED_FACE_COMPLEX_BYTES_3D));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&laplace_buffers.device_recv_below, PACKED_FACE_COMPLEX_BYTES_3D));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&laplace_buffers.device_recv_left, PACKED_FACE_COMPLEX_BYTES_3D));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&laplace_buffers.device_recv_right, PACKED_FACE_COMPLEX_BYTES_3D));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&laplace_buffers.device_recv_front, PACKED_FACE_COMPLEX_BYTES_3D));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&laplace_buffers.device_recv_back, PACKED_FACE_COMPLEX_BYTES_3D));
}


void destroy_laplace_buffers_3d(LaplaceBuffers3d& laplace_buffers) {
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(laplace_buffers.host_send_above));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(laplace_buffers.host_send_below));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(laplace_buffers.host_send_left));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(laplace_buffers.host_send_right));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(laplace_buffers.host_send_front));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(laplace_buffers.host_send_back));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(laplace_buffers.host_recv_above));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(laplace_buffers.host_recv_below));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(laplace_buffers.host_recv_left));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(laplace_buffers.host_recv_right));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(laplace_buffers.host_recv_front));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(laplace_buffers.host_recv_back));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(laplace_buffers.device_send_above));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(laplace_buffers.device_send_below));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(laplace_buffers.device_send_left));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(laplace_buffers.device_send_right));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(laplace_buffers.device_send_front));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(laplace_buffers.device_send_back));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(laplace_buffers.device_recv_above));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(laplace_buffers.device_recv_below));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(laplace_buffers.device_recv_left));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(laplace_buffers.device_recv_right));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(laplace_buffers.device_recv_front));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(laplace_buffers.device_recv_back));
}


/**
 * 3D volume
 *
 * 7 point laplacian stencil on complex data
 *
 * Grid: dim3(LOCAL_DIM, VEC_TOTAL), Threads: VEC
 * blockIdx.x = depth, blockIdx.y = vector index, threadIdx.x = element within vector
 */
__global__ void laplace_3d(Complex *device_in, Complex *device_out,
                           LaplaceBuffers3d laplace_buffers, int rid, int cid, int did) {
    int dep = blockIdx.x;
    int row = blockIdx.y / VEC_COL;
    int col = blockIdx.y % VEC_COL;
    int my_offset = (dep * LOCAL_DIM * LOCAL_DIM) + (row * LOCAL_DIM) + (col * VEC);

    int vec_index = threadIdx.x;

    Complex scratch = {0.0, 0.0};

    // ---- Y-direction (above/below) ----

    bool top_of_block = ((row % B_DIM) == 0);
    bool top_block = (row == 0);
    bool top_p = (rid == 0);

    bool bottom_of_block = ((row % B_DIM) == B_DIM - 1);
    bool bottom_block = (row == LOCAL_DIM - 1);
    bool bottom_p = (rid == P_DIM - 1);

    // Get row above
    int above_offset;
    if (top_of_block) {
        if (top_p) {
            if (top_block) {
                above_offset = (dep * PACKED_ROW_COMPLEX) + ((LOCAL_DIM/B_DIM - 1)*LOCAL_DIM) + (col * VEC);
            } else {
                above_offset = (dep * PACKED_ROW_COMPLEX) + ((row/B_DIM - 1)*LOCAL_DIM) + (col * VEC);
            }
        } else {
            above_offset = (dep * PACKED_ROW_COMPLEX) + ((row/B_DIM)*LOCAL_DIM) + (col * VEC);
        }
        scratch += laplace_buffers.device_recv_above[above_offset + vec_index];
    } else {
        above_offset = (dep * LOCAL_DIM * LOCAL_DIM) + ((row - 1) * LOCAL_DIM) + (col * VEC);
        scratch += device_in[above_offset + vec_index];
    }

    // Get row below
    int below_offset;
    if (bottom_of_block) {
        if (bottom_p) {
            if (bottom_block) {
                below_offset = (dep * PACKED_ROW_COMPLEX) + (col * VEC);
            } else {
                below_offset = (dep * PACKED_ROW_COMPLEX) + ((row/B_DIM + 1) * LOCAL_DIM) + (col * VEC);
            }
        } else {
            below_offset = (dep * PACKED_ROW_COMPLEX) + ((row/B_DIM) * LOCAL_DIM) + (col * VEC);
        }
        scratch += laplace_buffers.device_recv_below[below_offset + vec_index];
    } else {
        below_offset = (dep * LOCAL_DIM * LOCAL_DIM) + ((row + 1) * LOCAL_DIM) + (col * VEC);
        scratch += device_in[below_offset + vec_index];
    }

    // ---- X-direction (left/right) ----

    bool rightmost_col = (col == VEC_COL - 1);
    bool rightmost_p = (cid == P_DIM - 1);

    bool leftmost_col = (col == 0);
    bool leftmost_p = (cid == 0);

    // Get col to left
    int left_offset;
    if (leftmost_col) {
        left_offset = (dep * PACKED_COL_COMPLEX) + (row * VEC);
        int new_vec_index = leftmost_p ? (vec_index - 1 + VEC) % VEC : vec_index;
        scratch += laplace_buffers.device_recv_left[left_offset + new_vec_index];
    } else {
        left_offset = (dep * LOCAL_DIM * LOCAL_DIM) + (row * LOCAL_DIM) + ((col - 1) * VEC);
        scratch += device_in[left_offset + vec_index];
    }

    // Get col to right
    int right_offset;
    if (rightmost_col) {
        right_offset = (dep * PACKED_COL_COMPLEX) + (row * VEC);
        int new_vec_index = rightmost_p ? (vec_index + 1) % VEC : vec_index;
        scratch += laplace_buffers.device_recv_right[right_offset + new_vec_index];
    } else {
        right_offset = (dep * LOCAL_DIM * LOCAL_DIM) + (row * LOCAL_DIM) + ((col + 1) * VEC);
        scratch += device_in[right_offset + vec_index];
    }

    // ---- Z-direction (front/back) ----

    bool front_of_dblock = ((dep % B_DIM) == 0);
    bool front_dblock = (dep == 0);
    bool front_p = (did == 0);

    bool back_of_dblock = ((dep % B_DIM) == B_DIM - 1);
    bool back_dblock = (dep == LOCAL_DIM - 1);
    bool back_p = (did == P_DIM - 1);

    // Get depth in front (dep - 1)
    int front_offset;
    if (front_of_dblock) {
        if (front_p) {
            if (front_dblock) {
                front_offset = ((LOCAL_DIM/B_DIM - 1) * LOCAL_DIM * LOCAL_DIM) + (row * LOCAL_DIM) + (col * VEC);
            } else {
                front_offset = ((dep/B_DIM - 1) * LOCAL_DIM * LOCAL_DIM) + (row * LOCAL_DIM) + (col * VEC);
            }
        } else {
            front_offset = ((dep/B_DIM) * LOCAL_DIM * LOCAL_DIM) + (row * LOCAL_DIM) + (col * VEC);
        }
        scratch += laplace_buffers.device_recv_front[front_offset + vec_index];
    } else {
        front_offset = ((dep - 1) * LOCAL_DIM * LOCAL_DIM) + (row * LOCAL_DIM) + (col * VEC);
        scratch += device_in[front_offset + vec_index];
    }

    // Get depth behind (dep + 1)
    int back_offset;
    if (back_of_dblock) {
        if (back_p) {
            if (back_dblock) {
                back_offset = (row * LOCAL_DIM) + (col * VEC);
            } else {
                back_offset = ((dep/B_DIM + 1) * LOCAL_DIM * LOCAL_DIM) + (row * LOCAL_DIM) + (col * VEC);
            }
        } else {
            back_offset = ((dep/B_DIM) * LOCAL_DIM * LOCAL_DIM) + (row * LOCAL_DIM) + (col * VEC);
        }
        scratch += laplace_buffers.device_recv_back[back_offset + vec_index];
    } else {
        back_offset = ((dep + 1) * LOCAL_DIM * LOCAL_DIM) + (row * LOCAL_DIM) + (col * VEC);
        scratch += device_in[back_offset + vec_index];
    }

    scratch -= 6*device_in[my_offset + vec_index];
    device_out[my_offset + vec_index] = scratch;
}


__global__ void pack_laplace_3d(Complex *device_in, LaplaceBuffers3d laplace_buffers) {
    int dep = blockIdx.x;
    int i = blockIdx.y;
    int j = threadIdx.x;

    // y-packing: above/below (i < L/B blocks, all L columns)
    if (i < LOCAL_DIM / B_DIM) {
        laplace_buffers.device_send_above[dep * PACKED_ROW_COMPLEX + i * LOCAL_DIM + j] =
            device_in[dep * LOCAL_DIM * LOCAL_DIM + i * LOCAL_DIM * B_DIM + j];

        int bottom_offset = LOCAL_DIM * (B_DIM - 1);
        laplace_buffers.device_send_below[dep * PACKED_ROW_COMPLEX + i * LOCAL_DIM + j] =
            device_in[dep * LOCAL_DIM * LOCAL_DIM + bottom_offset + i * LOCAL_DIM * B_DIM + j];
    }

    // x-packing: left/right (all L rows, first VEC columns)
    if (j < VEC) {
        laplace_buffers.device_send_left[dep * PACKED_COL_COMPLEX + i * VEC + j] =
            device_in[dep * LOCAL_DIM * LOCAL_DIM + i * LOCAL_DIM + j];

        int right_offset = LOCAL_DIM - VEC;
        laplace_buffers.device_send_right[dep * PACKED_COL_COMPLEX + i * VEC + j] =
            device_in[dep * LOCAL_DIM * LOCAL_DIM + right_offset + i * LOCAL_DIM + j];
    }

    // z-packing: front/back (dep < L/B depth blocks, all L rows, all L columns)
    if (dep < LOCAL_DIM / B_DIM) {
        laplace_buffers.device_send_front[dep * LOCAL_DIM * LOCAL_DIM + i * LOCAL_DIM + j] =
            device_in[dep * LOCAL_DIM * LOCAL_DIM * B_DIM + i * LOCAL_DIM + j];

        int back_depth_offset = LOCAL_DIM * LOCAL_DIM * (B_DIM - 1);
        laplace_buffers.device_send_back[dep * LOCAL_DIM * LOCAL_DIM + i * LOCAL_DIM + j] =
            device_in[back_depth_offset + dep * LOCAL_DIM * LOCAL_DIM * B_DIM + i * LOCAL_DIM + j];
    }
}


void communicate_laplace_3d(LaplaceBuffers3d& laplace_buffers,
                            MPI_Comm& row_comm, MPI_Comm& col_comm, MPI_Comm& dep_comm,
                            int rid, int cid, int did) {
    int left  = (cid - 1 + P_DIM) % P_DIM;
    int right = (cid + 1)         % P_DIM;
    int up    = (rid - 1 + P_DIM) % P_DIM;
    int down  = (rid + 1)         % P_DIM;
    int front = (did - 1 + P_DIM) % P_DIM;
    int back  = (did + 1)         % P_DIM;

    // x-direction: left/right across row_comm
    #ifdef __GPU__AWARE__MPI__
        MPI_Sendrecv(laplace_buffers.device_send_right, PACKED_FACE_COMPLEX_3D, MPI_C_DOUBLE_COMPLEX, right, 0,
                     laplace_buffers.device_recv_left, PACKED_FACE_COMPLEX_3D, MPI_C_DOUBLE_COMPLEX, left, 0,
                     row_comm, MPI_STATUS_IGNORE);
    #else
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(laplace_buffers.host_send_right, laplace_buffers.device_send_right,
                                            PACKED_FACE_COMPLEX_BYTES_3D, MEM_COPY_DEVICE_TO_HOST));
        MPI_Sendrecv(laplace_buffers.host_send_right, PACKED_FACE_COMPLEX_3D, MPI_C_DOUBLE_COMPLEX, right, 0,
                     laplace_buffers.host_recv_left, PACKED_FACE_COMPLEX_3D, MPI_C_DOUBLE_COMPLEX, left, 0,
                     row_comm, MPI_STATUS_IGNORE);
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(laplace_buffers.device_recv_left, laplace_buffers.host_recv_left,
                                            PACKED_FACE_COMPLEX_BYTES_3D, MEM_COPY_HOST_TO_DEVICE));
    #endif

    #ifdef __GPU__AWARE__MPI__
        MPI_Sendrecv(laplace_buffers.device_send_left, PACKED_FACE_COMPLEX_3D, MPI_C_DOUBLE_COMPLEX, left, 1,
                     laplace_buffers.device_recv_right, PACKED_FACE_COMPLEX_3D, MPI_C_DOUBLE_COMPLEX, right, 1,
                     row_comm, MPI_STATUS_IGNORE);
    #else
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(laplace_buffers.host_send_left, laplace_buffers.device_send_left,
                                            PACKED_FACE_COMPLEX_BYTES_3D, MEM_COPY_DEVICE_TO_HOST));
        MPI_Sendrecv(laplace_buffers.host_send_left, PACKED_FACE_COMPLEX_3D, MPI_C_DOUBLE_COMPLEX, left, 1,
                     laplace_buffers.host_recv_right, PACKED_FACE_COMPLEX_3D, MPI_C_DOUBLE_COMPLEX, right, 1,
                     row_comm, MPI_STATUS_IGNORE);
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(laplace_buffers.device_recv_right, laplace_buffers.host_recv_right,
                                            PACKED_FACE_COMPLEX_BYTES_3D, MEM_COPY_HOST_TO_DEVICE));
    #endif

    // y-direction: above/below across col_comm
    #ifdef __GPU__AWARE__MPI__
        MPI_Sendrecv(laplace_buffers.device_send_above, PACKED_FACE_COMPLEX_3D, MPI_C_DOUBLE_COMPLEX, up, 2,
                     laplace_buffers.device_recv_below, PACKED_FACE_COMPLEX_3D, MPI_C_DOUBLE_COMPLEX, down, 2,
                     col_comm, MPI_STATUS_IGNORE);
    #else
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(laplace_buffers.host_send_above, laplace_buffers.device_send_above,
                                            PACKED_FACE_COMPLEX_BYTES_3D, MEM_COPY_DEVICE_TO_HOST));
        MPI_Sendrecv(laplace_buffers.host_send_above, PACKED_FACE_COMPLEX_3D, MPI_C_DOUBLE_COMPLEX, up, 2,
                     laplace_buffers.host_recv_below, PACKED_FACE_COMPLEX_3D, MPI_C_DOUBLE_COMPLEX, down, 2,
                     col_comm, MPI_STATUS_IGNORE);
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(laplace_buffers.device_recv_below, laplace_buffers.host_recv_below,
                                            PACKED_FACE_COMPLEX_BYTES_3D, MEM_COPY_HOST_TO_DEVICE));
    #endif

    #ifdef __GPU__AWARE__MPI__
        MPI_Sendrecv(laplace_buffers.device_send_below, PACKED_FACE_COMPLEX_3D, MPI_C_DOUBLE_COMPLEX, down, 3,
                     laplace_buffers.device_recv_above, PACKED_FACE_COMPLEX_3D, MPI_C_DOUBLE_COMPLEX, up, 3,
                     col_comm, MPI_STATUS_IGNORE);
    #else
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(laplace_buffers.host_send_below, laplace_buffers.device_send_below,
                                            PACKED_FACE_COMPLEX_BYTES_3D, MEM_COPY_DEVICE_TO_HOST));
        MPI_Sendrecv(laplace_buffers.host_send_below, PACKED_FACE_COMPLEX_3D, MPI_C_DOUBLE_COMPLEX, down, 3,
                     laplace_buffers.host_recv_above, PACKED_FACE_COMPLEX_3D, MPI_C_DOUBLE_COMPLEX, up, 3,
                     col_comm, MPI_STATUS_IGNORE);
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(laplace_buffers.device_recv_above, laplace_buffers.host_recv_above,
                                            PACKED_FACE_COMPLEX_BYTES_3D, MEM_COPY_HOST_TO_DEVICE));
    #endif

    // z-direction: front/back across dep_comm
    #ifdef __GPU__AWARE__MPI__
        MPI_Sendrecv(laplace_buffers.device_send_back, PACKED_FACE_COMPLEX_3D, MPI_C_DOUBLE_COMPLEX, back, 4,
                     laplace_buffers.device_recv_front, PACKED_FACE_COMPLEX_3D, MPI_C_DOUBLE_COMPLEX, front, 4,
                     dep_comm, MPI_STATUS_IGNORE);
    #else
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(laplace_buffers.host_send_back, laplace_buffers.device_send_back,
                                            PACKED_FACE_COMPLEX_BYTES_3D, MEM_COPY_DEVICE_TO_HOST));
        MPI_Sendrecv(laplace_buffers.host_send_back, PACKED_FACE_COMPLEX_3D, MPI_C_DOUBLE_COMPLEX, back, 4,
                     laplace_buffers.host_recv_front, PACKED_FACE_COMPLEX_3D, MPI_C_DOUBLE_COMPLEX, front, 4,
                     dep_comm, MPI_STATUS_IGNORE);
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(laplace_buffers.device_recv_front, laplace_buffers.host_recv_front,
                                            PACKED_FACE_COMPLEX_BYTES_3D, MEM_COPY_HOST_TO_DEVICE));
    #endif

    #ifdef __GPU__AWARE__MPI__
        MPI_Sendrecv(laplace_buffers.device_send_front, PACKED_FACE_COMPLEX_3D, MPI_C_DOUBLE_COMPLEX, front, 5,
                     laplace_buffers.device_recv_back, PACKED_FACE_COMPLEX_3D, MPI_C_DOUBLE_COMPLEX, back, 5,
                     dep_comm, MPI_STATUS_IGNORE);
    #else
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(laplace_buffers.host_send_front, laplace_buffers.device_send_front,
                                            PACKED_FACE_COMPLEX_BYTES_3D, MEM_COPY_DEVICE_TO_HOST));
        MPI_Sendrecv(laplace_buffers.host_send_front, PACKED_FACE_COMPLEX_3D, MPI_C_DOUBLE_COMPLEX, front, 5,
                     laplace_buffers.host_recv_back, PACKED_FACE_COMPLEX_3D, MPI_C_DOUBLE_COMPLEX, back, 5,
                     dep_comm, MPI_STATUS_IGNORE);
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(laplace_buffers.device_recv_back, laplace_buffers.host_recv_back,
                                            PACKED_FACE_COMPLEX_BYTES_3D, MEM_COPY_HOST_TO_DEVICE));
    #endif
}


void test_laplace_3d() {
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

    int dep_grp = rid * P_DIM + cid;
    MPI_Comm_split(MPI_COMM_WORLD, dep_grp, id, &dep_comm);

    int row_grp = did * P_DIM + rid;
    MPI_Comm_split(MPI_COMM_WORLD, row_grp, id, &row_comm);

    int col_grp = did * P_DIM + cid;
    MPI_Comm_split(MPI_COMM_WORLD, col_grp, id, &col_comm);

    FftBuffers fft_buffers;
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&fft_buffers.host_buf0, LOCAL_COMPLEX_BYTES_3D, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&fft_buffers.host_buf1, LOCAL_COMPLEX_BYTES_3D, DEVICE_HOST_ALLOC_DEFAULT));
    init_host_3d(rid, cid, did, fft_buffers.host_buf0);
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&fft_buffers.device_buf0, LOCAL_COMPLEX_BYTES_3D));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&fft_buffers.device_buf1, LOCAL_COMPLEX_BYTES_3D));
    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(fft_buffers.device_buf0, fft_buffers.host_buf0, LOCAL_COMPLEX_BYTES_3D, MEM_COPY_HOST_TO_DEVICE));
    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(fft_buffers.device_buf1, fft_buffers.host_buf1, LOCAL_COMPLEX_BYTES_3D, MEM_COPY_HOST_TO_DEVICE));

    LaplaceBuffers3d laplace_buffers;
    init_laplace_buffers_3d(laplace_buffers);

    dim3 grid_pack(LOCAL_DIM, LOCAL_DIM);
    pack_laplace_3d<<<grid_pack, LOCAL_DIM>>>(fft_buffers.device_buf0, laplace_buffers);
    cudaDeviceSynchronize();

    communicate_laplace_3d(laplace_buffers, row_comm, col_comm, dep_comm, rid, cid, did);
    MPI_Barrier(MPI_COMM_WORLD);

    dim3 grid_stencil(LOCAL_DIM, VEC_TOTAL);
    laplace_3d<<<grid_stencil, VEC>>>(fft_buffers.device_buf0, fft_buffers.device_buf1, laplace_buffers, rid, cid, did);
    cudaDeviceSynchronize();

    cudaMemcpy(fft_buffers.host_buf1, fft_buffers.device_buf1, LOCAL_COMPLEX_BYTES_3D, cudaMemcpyDeviceToHost);

    #ifdef __PRINT__RESULTS__
        MPI_Barrier(MPI_COMM_WORLD);
        if (id == 0) {
            std::cout << "Host out (3D Laplace)" << std::endl;
            for (int d = 0; d < LOCAL_DIM; d++) {
                std::cout << "depth " << d << std::endl;
                for (int i = 0; i < LOCAL_DIM; i++) {
                    for (int j = 0; j < LOCAL_DIM; j++) {
                        std::cout << fft_buffers.host_buf1[(d * LOCAL_DIM * LOCAL_DIM) + (i * LOCAL_DIM) + j].x << " ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
        }
    #endif

    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(fft_buffers.host_buf0));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(fft_buffers.host_buf1));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(fft_buffers.device_buf0));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(fft_buffers.device_buf1));
    destroy_laplace_buffers_3d(laplace_buffers);

    MPI_Finalize();
}
