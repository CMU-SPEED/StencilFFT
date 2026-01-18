#include "dy.h"
#include <iostream>
#include <assert.h>
#include "../utils.h"

/**
 * Compute the 4th order finite difference in the y dimension
 */
__global__ void dy(Scalar *device_in, Scalar *device_out, 
                   int rid, int cid, Scalar delta,
                   Scalar *device_packed_above0, Scalar *device_packed_above1,
                   Scalar *device_packed_below0, Scalar *device_packed_below1) {
    int row = blockIdx.x / VEC_COL;
    int col = blockIdx.x % VEC_COL;
    int col_offset = (col * VEC) + threadIdx.x;
    int my_offset = (row * LOCAL_DIM) + col_offset;

    Scalar scratch = 0.0;

    bool top0_of_block = ((row % B_DIM) == 0);
    bool top1_of_block = ((row % B_DIM) == 1);
    bool bottom0_of_block = ((row % B_DIM) == B_DIM - 1);
    bool bottom1_of_block = ((row % B_DIM) == B_DIM - 2);

    bool top0_row = (row == 0);
    bool top1_row = (row == 1);
    bool bottom0_row = (row == LOCAL_DIM - 1);
    bool bottom1_row = (row == LOCAL_DIM - 2);

    bool top_p = (rid == 0);
    bool bottom_p = (rid == P_DIM - 1);

    int one_above_offset;
    if (top0_of_block) {
        if (top_p) {
            one_above_offset = top0_row ? ((LOCAL_DIM/B_DIM - 1)*LOCAL_DIM) : ((row/B_DIM - 1)*LOCAL_DIM);
        } else {
            one_above_offset = ((row/B_DIM)*LOCAL_DIM);
        }
        scratch += (device_packed_above0[one_above_offset + col_offset] * -8);
    } else {
        one_above_offset = ((row - 1) * LOCAL_DIM);
        scratch += (device_in[one_above_offset + col_offset] * -8);
    }

    int two_above_offset;
    if (top0_of_block) {
        if (top_p) {
            two_above_offset = top0_row ? ((LOCAL_DIM/B_DIM - 1)*LOCAL_DIM) : ((row/B_DIM - 1)*LOCAL_DIM);
        } else {
            two_above_offset = ((row/B_DIM)*LOCAL_DIM);
        }
        scratch += device_packed_above1[two_above_offset + col_offset];
    } else if (top1_of_block) {
        if (top_p) {
            two_above_offset = top1_row ? ((LOCAL_DIM/B_DIM - 1)*LOCAL_DIM) : ((row/B_DIM - 1)*LOCAL_DIM);
        } else {
            two_above_offset = ((row/B_DIM)*LOCAL_DIM);
        }
        scratch += device_packed_above0[two_above_offset + col_offset];
    } else {
        two_above_offset = ((row - 2) * LOCAL_DIM);
        scratch += device_in[two_above_offset + col_offset];
    }

    int one_below_offset;
    if (bottom0_of_block) {
        if (bottom_p) {
            one_below_offset = bottom0_row ? 0 : ((row/B_DIM + 1) * LOCAL_DIM);
        } else {
            one_below_offset = ((row/B_DIM) * LOCAL_DIM);
        }
        scratch += (device_packed_below0[one_below_offset + col_offset] * 8);
    } else {
        one_below_offset = ((row + 1) * LOCAL_DIM);
        scratch += (device_in[one_below_offset + col_offset] * 8);
    }

    int two_below_offset;
    if (bottom0_of_block) {
        if (bottom_p) {            
            two_below_offset = bottom0_row ? 0 : ((row/B_DIM + 1) * LOCAL_DIM);
        } else {
            two_below_offset = ((row/B_DIM) * LOCAL_DIM);
        }
        scratch -= device_packed_below1[two_below_offset + col_offset];
    } else if (bottom1_of_block) {
        if (bottom_p) {
            two_below_offset = bottom1_row ? 0 : ((row/B_DIM + 1) * LOCAL_DIM);
        } else {
            two_below_offset = ((row/B_DIM) * LOCAL_DIM);
        }
        scratch -= device_packed_below0[two_below_offset + col_offset];
    } else {
        two_below_offset = ((row + 2) * LOCAL_DIM);
        scratch -= device_in[two_below_offset + col_offset];
    }

    scratch /= (12 * delta);
    
    device_out[my_offset] = scratch;
}

// FIXME: This needs integration and testing
// Launch with <<<LOCAL_DIM/B_DIM, LOCAL_DIM>>>
// NOTE: might need to swap these if the problem size gets very big due to max # threads in CTA
__global__ void pack_dy(Scalar *device_in,
                        Scalar *device_packed_top0, Scalar *device_packed_top1,
                        Scalar *device_packed_bottom0, Scalar *device_packed_bottom1) {
    
    int i = blockIdx.x;
    int j = threadIdx.x;
    
    device_packed_top0[i * LOCAL_DIM + j] = device_in[i * LOCAL_DIM * B_DIM + j];

    int top1_offset = LOCAL_DIM;
    device_packed_top1[i * LOCAL_DIM + j] = device_in[top1_offset + i * LOCAL_DIM * B_DIM + j];

    int bottom0_offset = LOCAL_DIM * (B_DIM - 1);
    device_packed_bottom0[i * LOCAL_DIM + j] = device_in[bottom0_offset + i * LOCAL_DIM * B_DIM + j];

    int bottom1_offset = LOCAL_DIM * (B_DIM - 2);
    device_packed_bottom1[i * LOCAL_DIM + j] = device_in[bottom1_offset + i * LOCAL_DIM * B_DIM + j];
}

// Packs the top row of all blocks
void pack_dy_top0(Scalar *in, Scalar *packed) {
    for (int i = 0; i < LOCAL_DIM/B_DIM; i++) {
        for (int j = 0; j < LOCAL_DIM; j++) {
            packed[i * LOCAL_DIM + j] = in[i * LOCAL_DIM * B_DIM + j];
        }
    }
}

// Packs the second from top row of all blocks
void pack_dy_top1(Scalar *in, Scalar *packed) {
    int offset = LOCAL_DIM;
    for (int i = 0; i < LOCAL_DIM/B_DIM; i++) {
        for (int j = 0; j < LOCAL_DIM; j++) {
            packed[i * LOCAL_DIM + j] = in[offset + i * LOCAL_DIM * B_DIM + j];
        }
    }
}

// Packs the bottom row of all blocks
void pack_dy_bottom0(Scalar *in, Scalar *packed) {
    int offset = LOCAL_DIM * (B_DIM - 1);
    for (int i = 0; i < LOCAL_DIM/B_DIM; i++) {
        for (int j = 0; j < LOCAL_DIM; j++) {
            packed[i * LOCAL_DIM + j] = in[offset + i * LOCAL_DIM * B_DIM + j];
        }
    }
}

// Packs the second from bottom row of all blocks
void pack_dy_bottom1(Scalar *in, Scalar *packed) {
    int offset = LOCAL_DIM * (B_DIM - 2);
    for (int i = 0; i < LOCAL_DIM/B_DIM; i++) {
        for (int j = 0; j < LOCAL_DIM; j++) {
            packed[i * LOCAL_DIM + j] = in[offset + i * LOCAL_DIM * B_DIM + j];
        }
    }
}

// NOTE: This is just for testing
void init_packed_dy(int rid, int cid,
                 Scalar *host_packed_above0, Scalar *host_packed_above1,
                 Scalar *host_packed_below0, Scalar *host_packed_below1) {

    Scalar *host_in = (Scalar*)malloc(LOCAL_SCALAR_BYTES);

    int above_rid = (rid == 0) ? P_DIM - 1 : rid - 1;
    int below_rid = (rid + 1) % P_DIM;

    init_host_scalar(above_rid, cid, host_in);
    pack_dy_bottom0(host_in, host_packed_above0);
    pack_dy_bottom1(host_in, host_packed_above1);

    init_host_scalar(below_rid, cid, host_in);
    pack_dy_top0(host_in, host_packed_below0);
    pack_dy_top1(host_in, host_packed_below1);

    free(host_in);
}

void test_dy() {
    Scalar delta = 1; // FIXME: find out what a realistic value is

    int rid = 0;
    int cid = 0;

    Scalar *host_in = (Scalar*)malloc(LOCAL_SCALAR_BYTES);
    Scalar *host_out = (Scalar*)malloc(LOCAL_SCALAR_BYTES);
    Scalar *host_packed_above0 = (Scalar*)malloc(PACKED_ROW_SCALAR_BYTES);
    Scalar *host_packed_above1 = (Scalar*)malloc(PACKED_ROW_SCALAR_BYTES);
    Scalar *host_packed_below0 = (Scalar*)malloc(PACKED_ROW_SCALAR_BYTES);
    Scalar *host_packed_below1 = (Scalar*)malloc(PACKED_ROW_SCALAR_BYTES);

    init_host_scalar(rid, cid, host_in);

    // Simulate communication
    init_packed_dy(rid, cid,
                   host_packed_above0, host_packed_above1,
                   host_packed_below0, host_packed_below1);

    // for (int i = 0; i < LOCAL_DIM/b; i++) {
    //     for (int j = 0; j < LOCAL_DIM; j++) {
    //         std::cout << "(" << host_packed_below1[i * LOCAL_DIM + j].x << "," << host_packed_below1[i * LOCAL_DIM + j].y << ") ";
    //     }
    //     std::cout << std::endl;
    // }

    Scalar *device_in, *device_out;
    Scalar *device_packed_above0, *device_packed_above1;
    Scalar *device_packed_below0, *device_packed_below1;
    cudaMalloc((void**)&device_in, LOCAL_SCALAR_BYTES);
    cudaMalloc((void**)&device_out, LOCAL_SCALAR_BYTES);
    cudaMalloc((void**)&device_packed_above0, PACKED_ROW_SCALAR_BYTES);
    cudaMalloc((void**)&device_packed_above1, PACKED_ROW_SCALAR_BYTES);
    cudaMalloc((void**)&device_packed_below0, PACKED_ROW_SCALAR_BYTES);
    cudaMalloc((void**)&device_packed_below1, PACKED_ROW_SCALAR_BYTES);

    cudaMemcpy(device_in, host_in, LOCAL_SCALAR_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(device_out, host_out, LOCAL_SCALAR_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(device_packed_above0, host_packed_above0, PACKED_ROW_SCALAR_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(device_packed_above1, host_packed_above1, PACKED_ROW_SCALAR_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(device_packed_below0, host_packed_below0, PACKED_ROW_SCALAR_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(device_packed_below1, host_packed_below1, PACKED_ROW_SCALAR_BYTES, cudaMemcpyHostToDevice);

    dy<<<VEC_TOTAL, VEC>>>(device_in, device_out, rid, cid, delta,
                           device_packed_above0, device_packed_above1,
                           device_packed_below0, device_packed_below1);
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

    std::cout << "host_packed_below0" << std::endl;
    for (int i = 0; i < (LOCAL_DIM/B_DIM); i++) {
        for (int j = 0; j < LOCAL_DIM; j++) {
            std::cout << "(" << host_packed_below0[i * LOCAL_DIM + j] << ") ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "host_packed_below1" << std::endl;
    for (int i = 0; i < (LOCAL_DIM/B_DIM); i++) {
        for (int j = 0; j < LOCAL_DIM; j++) {
            std::cout << "(" << host_packed_below1[i * LOCAL_DIM + j] << ") ";
        }
        std::cout << std::endl;
    }

    free(host_in);
    free(host_out);
    free(host_packed_above0);
    free(host_packed_above1);
    free(host_packed_below0);
    free(host_packed_below1);
    cudaFree(device_in);
    cudaFree(device_out);
    cudaFree(device_packed_above0);
    cudaFree(device_packed_above1);
    cudaFree(device_packed_below0);
    cudaFree(device_packed_below1);
}
