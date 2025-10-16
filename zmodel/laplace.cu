#include <iostream>

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

// TODO: Cast interleaved to float and then double/triple the number of threads

__global__ void laplace(interleaved_t *device_in, interleaved_t *device_out,
                        interleaved_t *device_packed_above, interleaved_t *device_packed_below,
                        interleaved_t *device_packed_left, interleaved_t *device_packed_right,
                        interleaved_t *device_packed_above_left, interleaved_t *device_packed_above_right,
                        interleaved_t *device_packed_below_left, interleaved_t *device_packed_below_right) {
    int row = blockIdx.x / vec_col;
    int col = blockIdx.x % vec_col;
    int my_offset = (row * local) + (col * vec);

    int vec_index = threadIdx.x;

    // TODO: NEED TO ADD SCALING FACTORS!

    interleaved_t scratch = {0.0, 0.0};

    // Get row above
    if (0 < (row % b)) {
        int above_offset = ((row - 1) * local) + (col * vec);
        // scratch.x += device_in[above_offset + vec_index].x;
        // scratch.y += device_in[above_offset + vec_index].y;
    } else {
        // The vecs in the row are at the top of a block
        // Need to get the vec at the bottom of the block above
        // Go up a row.  If at the top row then go to the end.  Stay in the same column
        if (row == 0) {
            int above_offset = ((local/b - 1)*local) + (col * vec);
            // scratch.x += device_packed_above[above_offset + vec_index].x;
            // scratch.y += device_packed_above[above_offset + vec_index].y;
        } else {
            int above_offset = ((row/b - 1)*local) + (col * vec);
            // scratch.x += device_packed_above[above_offset + vec_index].x;
            // scratch.y += device_packed_above[above_offset + vec_index].y;
        }
    }

    // Get row below
    if ((row % b) < b - 1) {
        int below_offset = ((row + 1) * local) + (col * vec);
        // scratch.x += device_in[below_offset + vec_index].x;
        // scratch.y += device_in[below_offset + vec_index].y;
    } else {
        // The vecs in the row are at the bottom of a block
        // Need to get the vec at the top of the block below
        // Go down a row.  If the bottom row then go to the first row.
        if (row == local - 1) {
            int below_offset = (col * vec);
            // scratch.x += device_packed_below[below_offset + vec_index].x;
            // scratch.y += device_packed_below[below_offset + vec_index].y;
        } else {
            int below_offset = ((row/b + 1) * local) + (col * vec);
            // scratch.x += device_packed_below[below_offset + vec_index].x;
            // scratch.y += device_packed_below[below_offset + vec_index].y;
        }
    }

    // Get col to left
    if (0 < col) {
        int left_offset = (row * local) + ((col - 1) * vec);
        // scratch.x += device_in[left_offset + vec_index].x;
        // scratch.y += device_in[left_offset + vec_index].y;
    } else {
        // The only case is the wrap around case which is handled by the collective communcations
        int left_offset = (row * vec);
        // scratch.x += device_packed_left[left_offset + vec_index].x;
        // scratch.y += device_packed_left[left_offset + vec_index].y;
    }

    // Get col to right
    if (col < vec_col - 1) {
        int right_offset = (row * local) + ((col + 1) * vec);
        // scratch.x += device_in[right_offset + vec_index].x;
        // scratch.y += device_in[right_offset + vec_index].y;
    } else {
        int right_offset = (row * vec);
        // scratch.x += device_packed_right[right_offset + vec_index].x;
        // scratch.y += device_packed_right[right_offset + vec_index].y;
    }

    // TODO: corners

    // FIXME: my earlier attempt at corners is wrong --> I should just be using the above/below and left/right corner buffers?
    // Actually im not sure i have to think about this more

    // device_out[my_offset + vec_index] = scratch - 3 * device_out[my_offset + vec_index];
    device_out[my_offset + vec_index] = scratch;
}

// Packs the top row of all blocks
void pack_laplace_top(interleaved_t *in, interleaved_t *packed) {
    for (int i = 0; i < local/b; i++) {
        for (int j = 0; j < local; j++) {
            packed[i * local + j] = in[i * local * b + j];
        }
    }
}

// Packs the bottom row of all blocks
void pack_laplace_bottom(interleaved_t *in, interleaved_t *packed) {
    int offset = local * (b - 1);
    for (int i = 0; i < local/b; i++) {
        for (int j = 0; j < local; j++) {
            packed[i * local + j] = in[offset + i * local * b + j];
        }
    }
}

// Packs the leftmost col of vec in local
void pack_laplace_left(interleaved_t *in, interleaved_t *packed) {
    for (int i = 0; i < local; i++) {
        for (int j = 0; j < vec; j++) {
            packed[i * vec + j] = in[i * local + j];
        }
    }
}

// Packs the rightmost col of vec in local
void pack_laplace_right(interleaved_t *in, interleaved_t *packed) {
    int offset = local - vec;
    for (int i = 0; i < local; i++) {
        for (int j = 0; j < vec; j++) {
            packed[i * vec + j] = in[offset + i * local + j];
        }
    }
}

// Packs the top left element
// This corresponds to the top vec in each block of the leftmost col in local
void pack_laplace_top_left(interleaved_t *in, interleaved_t *packed) {
    for (int i = 0; i < local/b; i++) {
        for (int j = 0; j < vec; j++) {
            packed[i * vec + j] = in[i * local * b + j];
        }
    }
}

// Packs the top right element
// This corresponds to the top vec in each block of the rightmost col in local
void pack_laplace_top_right(interleaved_t *in, interleaved_t *packed) {
    int offset = local - vec;
    for (int i = 0; i < local/b; i++) {
        for (int j = 0; j < vec; j++) {
            packed[i * vec + j] = in[offset + i * local * b + j];
        }
    }
}

// Pack the bottom left element
// This corresponds to the bottom vec in each block of the leftmost col in local
void pack_laplace_bottom_left(interleaved_t *in, interleaved_t *packed) {
    int offset = local * (b - 1);
    for (int i = 0; i < local/b; i++) {
        for (int j = 0; j < vec; j++) {
            packed[i * vec + j] = in[offset + i * local * b + j];
        }
    }
}

// Pack the bottom right element
// This corresponds to the bottom vec in each block of the rightmost col in local
void pack_laplace_bottom_right(interleaved_t *in, interleaved_t *packed) {
    int offset = (local * (b - 1)) + (local - vec);
    for (int i = 0; i < local/b; i++) {
        for (int j = 0; j < vec; j++) {
            packed[i * vec + j] = in[offset + i * local * b + j];
        }
    }
}

int main(int argc, char *argv[]) {
    int local_size = local*local*sizeof(interleaved_t);
    int packed_row_size = local*(local/b)*sizeof(interleaved_t);
    int packed_col_size = local*vec*sizeof(interleaved_t);
    int packed_corner_size = (local/b)*vec*sizeof(interleaved_t);

    interleaved_t *init = (interleaved_t*)malloc(local_size);
    interleaved_t *host_in = (interleaved_t*)malloc(local_size);
    interleaved_t *host_out = (interleaved_t*)malloc(local_size);
    interleaved_t *host_packed_above = (interleaved_t*)malloc(packed_row_size);
    interleaved_t *host_packed_below = (interleaved_t*)malloc(packed_row_size);
    interleaved_t *host_packed_left = (interleaved_t*)malloc(packed_col_size);
    interleaved_t *host_packed_right = (interleaved_t*)malloc(packed_col_size);
    interleaved_t *host_packed_above_left = (interleaved_t*)malloc(packed_corner_size);
    interleaved_t *host_packed_above_right = (interleaved_t*)malloc(packed_corner_size);
    interleaved_t *host_packed_below_left = (interleaved_t*)malloc(packed_corner_size);
    interleaved_t *host_packed_below_right = (interleaved_t*)malloc(packed_corner_size);

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

    // Pack for laplace
    /* 
        This packing is done to simulate the communcation

        In the real thing the data we pack on the left will be sent to the processor 
        on the left who will refer to the data as if it was on the right.
    */
    pack_laplace_top(host_in, host_packed_below);
    pack_laplace_bottom(host_in, host_packed_above);
    pack_laplace_left(host_in, host_packed_right);
    pack_laplace_right(host_in, host_packed_left);
    // pack_laplace_top_left(host_in, host_packed_above_left);
    // pack_laplace_top_right(host_in, host_packed_above_right);
    // pack_laplace_bottom_left(host_in, host_packed_below_left);
    // pack_laplace_bottom_right(host_in, host_packed_below_right);

    // for (int i = 0; i < local/b; i++) {
    //     for (int j = 0; j < vec; j++) {
    //         std::cout << "(" << host_packed_below_right[i * vec + j].x << "," << host_packed_below_right[i * vec + j].y << ") ";
    //     }
    //     std::cout << std::endl;
    // }

    interleaved_t *device_in, *device_out;
    interleaved_t *device_packed_above, *device_packed_below, *device_packed_left, *device_packed_right;
    interleaved_t *device_packed_above_left, *device_packed_above_right, *device_packed_below_left, *device_packed_below_right;
    cudaMalloc((void**)&device_in, local_size);
    cudaMalloc((void**)&device_out, local_size);
    cudaMalloc((void**)&device_packed_above, packed_row_size);
    cudaMalloc((void**)&device_packed_below, packed_row_size);
    cudaMalloc((void**)&device_packed_left, packed_col_size);
    cudaMalloc((void**)&device_packed_right, packed_col_size);
    cudaMalloc((void**)&device_packed_above_left, packed_corner_size);
    cudaMalloc((void**)&device_packed_above_right, packed_corner_size);
    cudaMalloc((void**)&device_packed_below_left, packed_corner_size);
    cudaMalloc((void**)&device_packed_below_right, packed_corner_size);

    cudaMemcpy(device_in, host_in, local_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_out, host_out, local_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_packed_above, host_packed_above, packed_row_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_packed_below, host_packed_below, packed_row_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_packed_left, host_packed_left, packed_col_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_packed_right, host_packed_right, packed_col_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_packed_above_left, host_packed_above_left, packed_corner_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_packed_above_right, host_packed_above_right, packed_corner_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_packed_below_left, host_packed_below_left, packed_corner_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_packed_below_right, host_packed_below_right, packed_corner_size, cudaMemcpyHostToDevice);

    laplace<<<vec_total, vec>>>(device_in, device_out, 
                                device_packed_above, device_packed_below,
                                device_packed_left, device_packed_right,
                                device_packed_above_left, device_packed_above_right,
                                device_packed_below_left, device_packed_below_right);
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
    free(host_packed_above);
    free(host_packed_below);
    free(host_packed_left);
    free(host_packed_right);
    free(host_packed_above_left);
    free(host_packed_above_right);
    free(host_packed_below_left);
    free(host_packed_below_right);
    cudaFree(device_in);
    cudaFree(device_out);
    cudaFree(device_packed_above);
    cudaFree(device_packed_below);
    cudaFree(device_packed_left);
    cudaFree(device_packed_right);
    cudaFree(device_packed_above_left);
    cudaFree(device_packed_above_right);
    cudaFree(device_packed_below_left);
    cudaFree(device_packed_below_right);
}