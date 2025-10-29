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

/**
 * Compute the 4th order finite difference in the x dimension
 * Apply the filter [1, −8, 0, 8, −1] / (12 * delta) on the grid [i-1, i-1, i, i+1, i+2]
 */
__global__ void dx(float *device_in, float *device_out, float delta, int rid, int cid,
                   float *device_packed_left0, float *device_packed_left1,
                   float *device_packed_right0, float *device_packed_right1) {
    int row = blockIdx.x / vec_col;
    int col = blockIdx.x % vec_col;
    int my_offset = (row * local) + (col * vec);

    int vec_index = threadIdx.x;

    float scratch = 0.0;

    bool rightmost_col = (col == vec_col - 1);
    bool rightmost_p = (cid == p - 1);

    bool leftmost_col = (col == 0);
    bool leftmost2_col = (col == 1);
    bool leftmost_p = (cid == 0);

    // Get col directly to the left
    // if (col == 0) {
        // int left_offset = (row * vec);
        // scratch += (device_packed_left0[left_offset + vec_index] * -8);
    // } else {
        // int left_offset = (row * local) + ((col - 1) * vec);
        // scratch += (device_in[left_offset + vec_index] * -8);
    // }

    // Get col to left
    // Assumes you are working with the packed rightmost col of the vec data from the processor to the left of you
    int left_offset;
    if (leftmost_col) {
        // Periodic boundary conditions force you to shift the vec to align elements when on leftmost processor
        left_offset = (row * vec);
        int new_vec_index = leftmost_p ? (vec_index - 1 + vec) % vec : vec_index;
        // scratch += device_packed_left0[left_offset + new_vec_index];
    } else {
        left_offset = (row * local) + ((col - 1) * vec);
        // scratch += device_in[left_offset + vec_index];
    }

    // Get col one step from the left
    if (leftmost_col) {
        int left_offset = (row * vec);
        int new_vec_index = leftmost_p ? (vec_index - 1 + vec) % vec : vec_index;
        // scratch += device_packed_left1[left_offset + new_vec_index];
    } else if (leftmost2_col) {
        int left_offset = (row * vec);
        int new_vec_index = leftmost_p ? (vec_index - 1 + vec) % vec : vec_index;
        // scratch += device_packed_left0[left_offset + new_vec_index];
    } else {
        int left_offset = (row * local) + ((col - 2) * vec);
        // scratch += device_in[left_offset + vec_index];
    }

    // Get row directly to the right
    // if (col == vec_col - 1) {
        // int right_offset = (row * vec);
        // scratch += (device_packed_right0[right_offset + vec_index] * 8);
    // } else {
        // int right_offset = (row * local) + ((col + 1) * vec);
        // scratch += (device_in[right_offset + vec_index] * 8);
    // }

    // Get row one step from the right
    // if (col == vec_col - 1) {
        // int right_offset = (row * vec);
        // scratch -= device_packed_right1[right_offset + vec_index];
    // } else if (col == vec_col - 2) {
        // int right_offset = (row * vec);
        // scratch -= device_packed_right0[right_offset + vec_index];
    // } else {
        // int right_offset = (row * local) + ((col + 2) * vec);
        // scratch -= device_in[right_offset + vec_index];
    // }

    // scratch /= (12 * delta);

    device_out[my_offset + vec_index] = scratch;
}

// Packs the leftmost col of the vec in local
void pack_dx_left0(float *in, float *packed) {
    for (int i = 0; i < local; i++) {
        for (int j = 0; j < vec; j++) {
            packed[i * vec + j] = in[i * local + j];
        }
    }
}

// Packs the second from the leftmost col of the vec in local
void pack_dx_left1(float *in, float *packed) {
    int offset = vec;
    for (int i = 0; i < local; i++) {
        for (int j = 0; j < vec; j++) {
            packed[i * vec + j] = in[offset + i * local + j];
        }
    }
}

// Packs the rightmost col of the vec in local
void pack_dx_right0(float *in, float *packed) {
    int offset = local - vec;
    for (int i = 0; i < local; i++) {
        for (int j = 0; j < vec; j++) {
            packed[i * vec + j] = in[offset + i * local + j];
        }
    }
}

// Packs the second from the rightmost col of the vec in local
void pack_dx_right1(float *in, float *packed) {
    int offset = local - (2 * vec);
    for (int i = 0; i < local; i++) {
        for (int j = 0; j < vec; j++) {
            packed[i * vec + j] = in[offset + i * local + j];
        }
    }
}

int main(int argc, char *argv[]) {
    float delta = 1; // FIXME: find out what a realistic value is
    int local_size = local*local*sizeof(float);
    int packed_col_size = local*vec*sizeof(float);

    float *init = (float*)malloc(local_size);
    float *host_in = (float*)malloc(local_size);
    float *host_out = (float*)malloc(local_size);
    float *host_packed_left0 = (float*)malloc(packed_col_size);
    float *host_packed_left1 = (float*)malloc(packed_col_size);
    float *host_packed_right0 = (float*)malloc(packed_col_size);
    float *host_packed_right1 = (float*)malloc(packed_col_size);

    int rid = 0;
    int cid = 0;

    // Init block cyclic data as if you are the first processor in the rows and columns --> code taken from zmodel
    for (int i = 0; i < local/b; i++) {
        for (int ii = 0; ii < b; ii++) {
            for (int j = 0; j < local/b; j++) {
                for (int jj = 0; jj < b; jj++) {
                    int array_index = (i * b * (local/b) * b) + (ii * (local/b) * b) + (j * b) + jj;
                    float real = (i * N * b * p) + (ii * N) + (j * b * p) + jj;
                    init[array_index] = real;
                    host_out[array_index] = 0.0;
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
    pack_dx_left0(host_in, host_packed_right0);
    pack_dx_left1(host_in, host_packed_right1);
    pack_dx_right0(host_in, host_packed_left0);
    pack_dx_right1(host_in, host_packed_left1);

    // for (int i = 0; i < local; i++) {
    //     for (int j = 0; j < vec; j++) {
    //         std::cout << "(" << host_packed_right1[i * vec + j].x << "," << host_packed_right1[i * vec + j].y << ") ";
    //     }
    //     std::cout << std::endl;
    // }

    float *device_in, *device_out;
    float *device_packed_left0, *device_packed_left1;
    float *device_packed_right0, *device_packed_right1;
    cudaMalloc((void**)&device_in, local_size);
    cudaMalloc((void**)&device_out, local_size);
    cudaMalloc((void**)&device_packed_left0, packed_col_size);
    cudaMalloc((void**)&device_packed_left1, packed_col_size);
    cudaMalloc((void**)&device_packed_right0, packed_col_size);
    cudaMalloc((void**)&device_packed_right1, packed_col_size);

    cudaMemcpy(device_in, host_in, local_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_out, host_out, local_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_packed_left0, host_packed_left0, packed_col_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_packed_left1, host_packed_left1, packed_col_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_packed_right0, host_packed_right0, packed_col_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_packed_right1, host_packed_right1, packed_col_size, cudaMemcpyHostToDevice);

    dx<<<vec_total, vec>>>(device_in, device_out, delta,
                           rid, cid,
                           device_packed_left0, device_packed_left1,
                           device_packed_right0, device_packed_right1);
    cudaDeviceSynchronize();

    cudaMemcpy(host_out, device_out, local_size, cudaMemcpyDeviceToHost);

    std::cout << "Host out" << std::endl;
    for (int i = 0; i < local; i++) {
        for (int j = 0; j < local; j++) {
            std::cout << host_out[i * local + j] << " ";
        }
        std::cout << std::endl;
    }

    free(init);
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