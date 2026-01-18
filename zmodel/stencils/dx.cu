#include "dx.h"
#include <iostream>
#include "../utils.h"

/**
 * Compute the 4th order finite difference in the x dimension
 * Apply the filter [1, −8, 0, 8, −1] / (12 * delta) on the grid [i-1, i-1, i, i+1, i+2]
 */
__global__ void dx(Scalar *device_in, Scalar *device_out,
                   int rid, int cid, Scalar delta,
                   Scalar *device_packed_left0, Scalar *device_packed_left1,
                   Scalar *device_packed_right0, Scalar *device_packed_right1) {
    int row = blockIdx.x / VEC_COL;
    int col = blockIdx.x % VEC_COL;
    int my_offset = (row * LOCAL_DIM) + (col * VEC);
    int vec_index = threadIdx.x;

    Scalar scratch = 0.0;

    bool left0_col = (col == 0);
    bool left1_col = (col == 1);
    bool right0_col = (col == VEC_COL - 1);
    bool right1_col = (col == VEC_COL - 2);

    bool left_p = (cid == 0);
    bool right_p = (cid == P_DIM - 1);

    // Get col to left
    // Assumes you are working with the packed rightmost col of the vec data from the processor to the left of you
    int left0_offset;
    int left_vec_index = left_p ? (vec_index - 1 + VEC) % VEC : vec_index;
    if (left0_col) {
        // Periodic boundary conditions force you to shift the vec to align elements when on leftmost processor
        left0_offset = (row * VEC);
        scratch += (device_packed_left0[left0_offset + left_vec_index] * -8);
    } else {
        left0_offset = (row * LOCAL_DIM) + ((col - 1) * VEC);
        scratch += (device_in[left0_offset + vec_index] * -8);
    }

    // Get col one step from the left
    int left1_offset;
    if (left0_col) {
        left1_offset = (row * VEC);
        scratch += device_packed_left1[left1_offset + left_vec_index];
    } else if (left1_col) {
        left1_offset = (row * VEC);
        scratch += device_packed_left0[left1_offset + left_vec_index];
    } else {
        left1_offset = (row * LOCAL_DIM) + ((col - 2) * VEC);
        scratch += device_in[left1_offset + vec_index];
    }

    // Get row directly to the right
    int right0_offset;
    int right_vec_index = right_p ? (vec_index + 1) % VEC : vec_index;
    if (right0_col) {
        right0_offset = (row * VEC);
        scratch += (device_packed_right0[right0_offset + right_vec_index] * 8);
    } else {
        right0_offset = (row * LOCAL_DIM) + ((col + 1) * VEC);
        scratch += (device_in[right0_offset + vec_index] * 8);
    }

    // Get row one step from the right
    int right1_offset;
    if (right0_col) {
        right1_offset = (row * VEC);
        scratch -= device_packed_right1[right1_offset + right_vec_index];
    } else if (right1_col) {
        right1_offset = (row * VEC);
        scratch -= device_packed_right0[right1_offset + right_vec_index];
    } else {
        right1_offset = (row * LOCAL_DIM) + ((col + 2) * VEC);
        scratch -= device_in[right1_offset + vec_index];
    }

    scratch /= (12 * delta);

    device_out[my_offset + vec_index] = scratch;
}

// FIXME: This needs integration and testing
// Launch with <<<LOCAL_DIM, VEC>>>
__global__ void pack_dx(Scalar *device_in,
                        Scalar *device_packed_left0, Scalar *device_packed_left1,
                        Scalar *device_packed_right0, Scalar *device_packed_right1) {
        int i = blockIdx.x;
        int j = threadIdx.x;

        device_packed_left0[i * VEC + j] = device_in[i * LOCAL_DIM + j];

        int left1_offset = VEC;
        device_packed_left1[i * VEC + j] = device_in[left1_offset + i * LOCAL_DIM + j];
    
        int right0_offset = LOCAL_DIM - VEC;
        device_packed_right0[i * VEC + j] = device_in[right0_offset + i * LOCAL_DIM + j];

        int right1_offset = LOCAL_DIM - (2 * VEC);
        device_packed_right1[i * VEC + j] = device_in[right1_offset + i * LOCAL_DIM + j];
}

// Packs the leftmost col of the vec in local
void pack_dx_left0(Scalar *in, Scalar *packed) {
    for (int i = 0; i < LOCAL_DIM; i++) {
        for (int j = 0; j < VEC; j++) {
            packed[i * VEC + j] = in[i * LOCAL_DIM + j];
        }
    }
}

// Packs the second from the leftmost col of the vec in local
void pack_dx_left1(Scalar *in, Scalar *packed) {
    int offset = VEC;
    for (int i = 0; i < LOCAL_DIM; i++) {
        for (int j = 0; j < VEC; j++) {
            packed[i * VEC + j] = in[offset + i * LOCAL_DIM + j];
        }
    }
}

// Packs the rightmost col of the vec in local
void pack_dx_right0(Scalar *in, Scalar *packed) {
    int offset = LOCAL_DIM - VEC;
    for (int i = 0; i < LOCAL_DIM; i++) {
        for (int j = 0; j < VEC; j++) {
            packed[i * VEC + j] = in[offset + i * LOCAL_DIM + j];
        }
    }
}

// Packs the second from the rightmost col of the vec in local
void pack_dx_right1(Scalar *in, Scalar *packed) {
    int offset = LOCAL_DIM - (2 * VEC);
    for (int i = 0; i < LOCAL_DIM; i++) {
        for (int j = 0; j < VEC; j++) {
            packed[i * VEC + j] = in[offset + i * LOCAL_DIM + j];
        }
    }
}

void init_packed_dx(int rid, int cid,
                 Scalar *host_packed_left0, Scalar *host_packed_left1,
                 Scalar *host_packed_right0, Scalar *host_packed_right1) {
    
    Scalar *host_in = (Scalar*)malloc(LOCAL_SCALAR_BYTES);

    int left_cid = (cid == 0) ? P_DIM - 1 : cid - 1;
    int right_cid = (cid + 1) % P_DIM;

    init_host_scalar(rid, right_cid, host_in);
    pack_dx_left0(host_in, host_packed_right0);
    pack_dx_left1(host_in, host_packed_right1);

    init_host_scalar(rid, left_cid, host_in);
    pack_dx_right0(host_in, host_packed_left0);
    pack_dx_right1(host_in, host_packed_left1);

    free(host_in);
}

void test_dx() {
    Scalar delta = 1; // FIXME: find out what a realistic value is

    int rid = 0;
    int cid = 1;

    Scalar *host_in = (Scalar*)malloc(LOCAL_SCALAR_BYTES);
    Scalar *host_out = (Scalar*)malloc(LOCAL_SCALAR_BYTES);
    Scalar *host_packed_left0 = (Scalar*)malloc(PACKED_COL_SCALAR_BYTES);
    Scalar *host_packed_left1 = (Scalar*)malloc(PACKED_COL_SCALAR_BYTES);
    Scalar *host_packed_right0 = (Scalar*)malloc(PACKED_COL_SCALAR_BYTES);
    Scalar *host_packed_right1 = (Scalar*)malloc(PACKED_COL_SCALAR_BYTES);

    init_host_scalar(rid, cid, host_in);

    // Simulate communication
    init_packed_dx(rid, cid,
                   host_packed_left0, host_packed_left1,
                   host_packed_right0, host_packed_right1);

    // for (int i = 0; i < local; i++) {
    //     for (int j = 0; j < vec; j++) {
    //         std::cout << "(" << host_packed_right1[i * vec + j].x << "," << host_packed_right1[i * vec + j].y << ") ";
    //     }
    //     std::cout << std::endl;
    // }

    Scalar *device_in, *device_out;
    Scalar *device_packed_left0, *device_packed_left1;
    Scalar *device_packed_right0, *device_packed_right1;
    cudaMalloc((void**)&device_in, LOCAL_SCALAR_BYTES);
    cudaMalloc((void**)&device_out, LOCAL_SCALAR_BYTES);
    cudaMalloc((void**)&device_packed_left0, PACKED_COL_SCALAR_BYTES);
    cudaMalloc((void**)&device_packed_left1, PACKED_COL_SCALAR_BYTES);
    cudaMalloc((void**)&device_packed_right0, PACKED_COL_SCALAR_BYTES);
    cudaMalloc((void**)&device_packed_right1, PACKED_COL_SCALAR_BYTES);

    cudaMemcpy(device_in, host_in, LOCAL_SCALAR_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(device_out, host_out, LOCAL_SCALAR_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(device_packed_left0, host_packed_left0, PACKED_COL_SCALAR_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(device_packed_left1, host_packed_left1, PACKED_COL_SCALAR_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(device_packed_right0, host_packed_right0, PACKED_COL_SCALAR_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(device_packed_right1, host_packed_right1, PACKED_COL_SCALAR_BYTES, cudaMemcpyHostToDevice);

    dx<<<VEC_TOTAL, VEC>>>(device_in, device_out, rid, cid, delta,
                           device_packed_left0, device_packed_left1,
                           device_packed_right0, device_packed_right1);
    cudaDeviceSynchronize();

    cudaMemcpy(host_out, device_out, LOCAL_SCALAR_BYTES, cudaMemcpyDeviceToHost);

    std::cout << "Host in" << std::endl;
    for (int i = 0; i < LOCAL_DIM; i++) {
        for (int j = 0; j < LOCAL_DIM; j++) {
            std::cout << "(" << host_in[i * LOCAL_DIM + j] << ") ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Host out" << std::endl;
    for (int i = 0; i < LOCAL_DIM; i++) {
        for (int j = 0; j < LOCAL_DIM; j++) {
            std::cout << "(" << host_out[i * LOCAL_DIM + j] << ") ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "host_packed_right0" << std::endl;
    for (int i = 0; i < LOCAL_DIM; i++) {
        for (int j = 0; j < VEC; j++) {
            std::cout << "(" << host_packed_right0[i * VEC + j] << ") ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "host_packed_right1" << std::endl;
    for (int i = 0; i < LOCAL_DIM; i++) {
        for (int j = 0; j < VEC; j++) {
            std::cout << "(" << host_packed_right1[i * VEC + j] << ") ";
        }
        std::cout << std::endl;
    }

    free(host_in);
    free(host_out);
    free(host_packed_left0);
    free(host_packed_left1);
    free(host_packed_right0);
    free(host_packed_right1);
    cudaFree(device_in);
    cudaFree(device_out);
    cudaFree(device_packed_left0);
    cudaFree(device_packed_left1);
    cudaFree(device_packed_right0);
    cudaFree(device_packed_right1);
}