#include <iostream>
#include <assert.h>

// int N = 16;
#define N (64)
// int b = 2;
#define b (4)
// int p = 2; // Number of processors in the rows or columns. ex: p = 2 --> 4 total processors
#define p (2)
// int local = N/p;
#define local (N/p)

// This is the size of the vector the stencil will be batched with
// int vec = local/b;
#define vec (local/b)
// The total number of vectors on the local processor
// int vec_total = local/vec;
#define vec_total ((local*local)/vec)
// The number of vectors in a row / col respectively 
#define vec_row (local)
#define vec_col (b)

typedef struct interleaved {
    float x;
    float y;
} interleaved_t;

/**
 * Compute the 4th order finite difference in the y dimension
 */
__global__ void dy(interleaved_t *device_in, interleaved_t *device_out, float delta,
                   interleaved_t *device_packed_above0, interleaved_t *device_packed_above1,
                   interleaved_t *device_packed_below0, interleaved_t *device_packed_below1) {
    int row = blockIdx.x / vec_col;
    int col = blockIdx.x % vec_col;
    int my_offset = (row * local) + (col * vec);

    int vec_index = threadIdx.x;

    interleaved_t scratch = {0.0, 0.0};

    // Get row directly above
    if (0 < (row % b)) {
        int above_offset = ((row - 1) * local) + (col * vec);
        scratch.x += (device_in[above_offset + vec_index].x * -8);
    } else {
        if (row == 0) {
            int above_offset = ((local/b - 1)*local) + (col * vec);
            scratch.x += (device_packed_above0[above_offset + vec_index].x * -8);
        } else {
            int above_offset = ((row/b - 1)*local) + (col * vec);
            scratch.x += (device_packed_above0[above_offset + vec_index].x * -8);
        }
    }

    // Get row two steps above
    if (1 < (row % b)) {
        int above_offset = ((row - 2) * local) + (col * vec);
        scratch.x += device_in[above_offset + vec_index].x;
    } else {
        if (row == 0) {
            int above_offset = ((local/b - 1)*local) + (col * vec);
            scratch.x += device_packed_above1[above_offset + vec_index].x;
        } else if (row == 1) {
            int above_offset = ((local/b - 1)*local) + (col * vec);
            scratch.x += device_packed_above0[above_offset + vec_index].x;
        } else if ((row % b) == 0) {
            int above_offset = ((row/b - 1)*local) + (col * vec);
            scratch.x += device_packed_above1[above_offset + vec_index].x;
        } else if ((row % b) == 1) {
            int above_offset = ((row/b - 1)*local) + (col * vec);
            scratch.x += device_packed_above0[above_offset + vec_index].x;
        } else {
            assert(false);
        }
    }

    // Get row directly below
    if ((row % b) < b - 1) {
        int below_offset = ((row + 1) * local) + (col * vec);
        scratch.x += (device_in[below_offset + vec_index].x * 8);
    } else {
        if (row == local - 1) {
            int below_offset = (col * vec);
            scratch.x += (device_packed_below0[below_offset + vec_index].x * 8);
        } else {
            int below_offset = ((row/b + 1) * local) + (col * vec);
            scratch.x += (device_packed_below0[below_offset + vec_index].x * 8);
        }
    }

    // Get row two steps below
    if ((row % b) < b - 2) {
        int below_offset = ((row + 2) * local) + (col * vec);
        scratch.x -= device_in[below_offset + vec_index].x;
    } else {
        if (row == local - 1) {
            int below_offset = (col * vec);
            scratch.x -= device_packed_below1[below_offset + vec_index].x;
        } else if (row == local - 2) {
            int below_offset = (col * vec);
            scratch.x -= device_packed_below0[below_offset + vec_index].x;
        } else if ((row % b) == b - 1) {
            int below_offset = ((row/b + 1) * local) + (col * vec);
            scratch.x -= device_packed_below1[below_offset + vec_index].x;
        } else if ((row % b) == b - 2) {
            int below_offset = ((row/b + 1) * local) + (col * vec);
            scratch.x -= device_packed_below0[below_offset + vec_index].x;
        } else {
            assert(false);
        }
    }

    scratch.x /= (12 * delta);
    
    device_out[my_offset + vec_index] = scratch;
}

// Packs the top row of all blocks
void pack_dy_top0(interleaved_t *in, interleaved_t *packed) {
    for (int i = 0; i < local/b; i++) {
        for (int j = 0; j < local; j++) {
            packed[i * local + j] = in[i * local * b + j];
        }
    }
}

// Packs the second from top row of all blocks
void pack_dy_top1(interleaved_t *in, interleaved_t *packed) {
    int offset = local;
    for (int i = 0; i < local/b; i++) {
        for (int j = 0; j < local; j++) {
            packed[i * local + j] = in[offset + i * local * b + j];
        }
    }
}

// Packs the bottom row of all blocks
void pack_dy_bottom0(interleaved_t *in, interleaved_t *packed) {
    int offset = local * (b - 1);
    for (int i = 0; i < local/b; i++) {
        for (int j = 0; j < local; j++) {
            packed[i * local + j] = in[offset + i * local * b + j];
        }
    }
}

// Packs the second from bottom row of all blocks
void pack_dy_bottom1(interleaved_t *in, interleaved_t *packed) {
    int offset = local * (b - 2);
    for (int i = 0; i < local/b; i++) {
        for (int j = 0; j < local; j++) {
            packed[i * local + j] = in[offset + i * local * b + j];
        }
    }
}

int main(int argc, char *argv[]) {
    float delta = 1; // FIXME: find out what a realistic value is
    int local_size = local*local*sizeof(interleaved_t);
    int packed_row_size = local*(local/b)*sizeof(interleaved_t);

    interleaved_t *init = (interleaved_t*)malloc(local_size);
    interleaved_t *host_in = (interleaved_t*)malloc(local_size);
    interleaved_t *host_out = (interleaved_t*)malloc(local_size);
    interleaved_t *host_packed_above0 = (interleaved_t*)malloc(packed_row_size);
    interleaved_t *host_packed_above1 = (interleaved_t*)malloc(packed_row_size);
    interleaved_t *host_packed_below0 = (interleaved_t*)malloc(packed_row_size);
    interleaved_t *host_packed_below1 = (interleaved_t*)malloc(packed_row_size);

    // Init block cyclic data as if you are the first processor in the rows and columns --> code taken from zmodel
    for (int i = 0; i < local/b; i++) {
        for (int ii = 0; ii < b; ii++) {
            for (int j = 0; j < local/b; j++) {
                for (int jj = 0; jj < b; jj++) {
                    int array_index = (i * b * (local/b) * b) + (ii * (local/b) * b) + (j * b) + jj;
                    float real = (i * N * b * p) + (ii * N) + (j * b * p) + jj;
                    init[array_index] = {real, real};
                    host_out[array_index] = {0.0, 0.0};
                }
            }
        }
    }

    // Pack data for the stencils --> code taken from zmodel
    for (int i = 0; i < local; i++) {
        for (int j = 0; j < b; j++) {
            for (int jj = 0; jj < local/b; jj++) {
                int row = i * local;
                host_in[row + (j * (local/b))+ jj] = init[row + j + (jj * b)];
            }
        }
    }

    // Simulate communication
    pack_dy_top0(host_in, host_packed_below0);
    pack_dy_top1(host_in, host_packed_below1);
    pack_dy_bottom0(host_in, host_packed_above0);
    pack_dy_bottom1(host_in, host_packed_above1);

    // for (int i = 0; i < local/b; i++) {
    //     for (int j = 0; j < local; j++) {
    //         std::cout << "(" << host_packed_below1[i * local + j].x << "," << host_packed_below1[i * local + j].y << ") ";
    //     }
    //     std::cout << std::endl;
    // }

    interleaved_t *device_in, *device_out;
    interleaved_t *device_packed_above0, *device_packed_above1;
    interleaved_t *device_packed_below0, *device_packed_below1;
    cudaMalloc((void**)&device_in, local_size);
    cudaMalloc((void**)&device_out, local_size);
    cudaMalloc((void**)&device_packed_above0, packed_row_size);
    cudaMalloc((void**)&device_packed_above1, packed_row_size);
    cudaMalloc((void**)&device_packed_below0, packed_row_size);
    cudaMalloc((void**)&device_packed_below1, packed_row_size);

    cudaMemcpy(device_in, host_in, local_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_out, host_out, local_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_packed_above0, host_packed_above0, packed_row_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_packed_above1, host_packed_above1, packed_row_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_packed_below0, host_packed_below0, packed_row_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_packed_below1, host_packed_below1, packed_row_size, cudaMemcpyHostToDevice);

    dy<<<vec_total, vec>>>(device_in, device_out, delta,
                           device_packed_above0, device_packed_above1,
                           device_packed_below0, device_packed_below1);
    cudaDeviceSynchronize();

    cudaMemcpy(host_out, device_out, local_size, cudaMemcpyDeviceToHost);

    std::cout << "Host out" << std::endl;
    for (int i = 0; i < local; i++) {
        for (int j = 0; j < local; j++) {
            std::cout << "(" << host_out[i * local + j].x << "," << host_out[i * local + j].y << ") ";
        }
        std::cout << std::endl;
    }

    free(init);
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
