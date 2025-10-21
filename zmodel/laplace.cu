#include <iostream>

#define N (64)
#define b (4)
#define p (2) // Number of processors in the rows or columns. ex: p = 2 --> 4 total processors
#define local (N/p)

// This is the size of the vector the stencil will be batched with
#define vec (local/b)
// The total number of vectors on the local processor
#define vec_total ((local*local)/vec)
// The number of vectors in a row / col respectively 
#define vec_row (local)
#define vec_col (b)

// FIXME: Check that the column alignment vec shift isnt parameterized by the number of processors or something else
// FIXME: I wonder if we can make some of these if else checks happen at compile time
// FIXME: the corners should be broken out into helper functions in order to cleanup redundant code
// FIXME: Need to add scaling factor
__global__ void laplace(float *device_in, float *device_out,
                        int rid, int cid,
                        float *device_packed_above, float *device_packed_below,
                        float *device_packed_left, float *device_packed_right,
                        float *device_packed_above_left, float *device_packed_above_right,
                        float *device_packed_below_left, float *device_packed_below_right) {
    int row = blockIdx.x / vec_col;
    int col = blockIdx.x % vec_col;
    int my_offset = (row * local) + (col * vec);

    int vec_index = threadIdx.x;

    float scratch = 0.0;

    // The vector sits at the top of the block it is in
    bool top_of_block = ((row % b) == 0);
    // The block the vector is in is the top row of blocks on the processor
    bool top_block = (row == 0);
    // The processor is in the top row of the grid of processors 
    bool top_p = (rid == 0);

    bool bottom_of_block = ((row % b) == b - 1);
    bool bottom_block = (row == local - 1);
    bool bottom_p = (rid == p - 1);

    bool rightmost_col = (col == vec_col - 1);
    bool rightmost_p = (cid == p - 1);

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
                above_offset = ((local/b - 1)*local) + (col * vec);
            } else {
                above_offset = ((row/b - 1)*local) + (col * vec);
            }
        } else {
            // If you are not the top processor then you are actually grabbing elements from the same PACKED row
            // Therefore there is no need to consider the case of wrap around
            above_offset = ((row/b)*local) + (col * vec);
        }
        // scratch += device_packed_above[above_offset + vec_index];
    } else {
        above_offset = ((row - 1) * local) + (col * vec);
        // scratch += device_in[above_offset + vec_index];
    }

    // Get row below
    // Assumes you are working with the packed top row of the block data from the processor below you
    int below_offset;
    if (bottom_of_block) {
        if (bottom_p) {
            if (bottom_block) {
                below_offset = (col * vec);
            } else {
                below_offset = ((row/b + 1) * local) + (col * vec);
            }
        } else {
            below_offset = ((row/b) * local) + (col * vec);
        }
        // scratch += device_packed_below[below_offset + vec_index];
    } else {
        below_offset = ((row + 1) * local) + (col * vec);
        // scratch += device_in[below_offset + vec_index];
    }

    // Get col to left
    // Assumes you are working with the packed rightmost col of the vec data from the processor to the left of you
    int left_offset;
    if (leftmost_col) {
        // Periodic boundary conditions force you to shift the vec to align elements when on leftmost processor
        left_offset = (row * vec);
        int new_vec_index = leftmost_p ? (vec_index - 1 + vec) % vec : vec_index;
        // scratch += device_packed_left[left_offset + new_vec_index];
    } else {
        left_offset = (row * local) + ((col - 1) * vec);
        // scratch += device_in[left_offset + vec_index];
    }

    // Get col to right
    // Assumes you are working with the packed leftmost col of the vec data from the processor to the right of you
    int right_offset;
    if (rightmost_col) {
        right_offset = (row * vec);
        int new_vec_index = rightmost_p ? (vec_index + 1) % vec : vec_index;
        // scratch += device_packed_right[right_offset + new_vec_index];
    } else {
        right_offset = (row * local) + ((col + 1) * vec);
        // scratch += device_in[right_offset + vec_index];
    }

    // Top left corner
    int top_left_offset;
    if (top_of_block) {
        if (top_p) {
            // Calculate the offset
            if (top_block) {
                if (leftmost_col) {
                    // Data comes from corner packing so no col offset
                    top_left_offset = ((local/b - 1) * vec);
                } else {
                    // Data comes from above packing so the whole row is stored
                    top_left_offset = ((local/b - 1) * local) + ((col - 1) * vec);
                }
            } else {
                if (leftmost_col) {
                    top_left_offset = ((row/b - 1) * vec);
                } else {
                    top_left_offset = ((row/b - 1) * local) + ((col - 1) * vec);
                }
            }

            // Calculate the shift
            if (leftmost_col) {
                // leftmost gets data from corner packing
                int new_vec_index = leftmost_p ? (vec_index - 1 + vec) % vec : vec_index;
                // scratch += device_packed_above_left[top_left_offset + new_vec_index];
            } else {
                // Non leftmost gets data from above
                // scratch += device_packed_above[top_left_offset + vec_index];
            }
        } else {
            // Use the same row of the packed data because it is not the top p
            if (leftmost_col) {
                top_left_offset = ((row/b) * vec);
                int new_vec_index = leftmost_p ? (vec_index - 1 + vec) % vec : vec_index;
                // scratch += device_packed_above_left[top_left_offset + new_vec_index];
            } else {
                top_left_offset = ((row/b)*local) + ((col - 1) * vec);
                // scratch += device_packed_above[top_left_offset + vec_index];
            }
        }
    } else {
        // This is the base case
        if (leftmost_col) {
            top_left_offset = ((row - 1) * vec);
            int new_vec_index = leftmost_p ? (vec_index - 1 + vec) % vec : vec_index;
            // scratch += device_packed_left[top_left_offset + new_vec_index];
        } else {
            top_left_offset = ((row - 1) * local) + ((col - 1) * vec);
            // scratch += device_in[top_left_offset + vec_index];
        }
    }

    // Top right corner
    int top_right_offset;
    if (top_of_block) {
        if (top_p) {
            if (top_block) {
                if (rightmost_col) {
                    top_right_offset = ((local/b - 1) * vec);
                } else {
                    top_right_offset = ((local/b - 1) * local) + ((col + 1) * vec);
                }
            } else {
                if (rightmost_col) {
                    top_right_offset = ((row/b - 1) * vec);
                } else {
                    top_right_offset = ((row/b - 1) * local) + ((col + 1) * vec);
                }
            }

            if (rightmost_col) {
                int new_vec_index = rightmost_p ? (vec_index + 1) % vec : vec_index;
                // scratch += device_packed_above_right[top_right_offset + new_vec_index];
            } else {
                // scratch += device_packed_above[top_right_offset + vec_index];
            }
        } else {
            if (rightmost_col) {
                top_right_offset = ((row/b) * vec);
                int new_vec_index = rightmost_p ? (vec_index + 1) % vec : vec_index;
                // scratch += device_packed_above_right[top_right_offset + new_vec_index];
            } else {
                top_right_offset = ((row/b) * local) + ((col + 1) * vec);
                // scratch += device_packed_above[top_right_offset + vec_index];
            }
        }
    } else {
        if (rightmost_col) {
            top_right_offset = ((row - 1) * vec);
            int new_vec_index = rightmost_p ? (vec_index + 1) % vec : vec_index;
            // scratch += device_packed_right[top_right_offset + new_vec_index];
        } else {
            top_right_offset = ((row - 1) * local) + ((col + 1) * vec);
            // scratch += device_in[top_right_offset + vec_index];
        }
    }

    // Bottom left corner
    int bottom_left_offset;
    if (bottom_of_block) {
        if (bottom_p) {
            if (bottom_block) {
                if (leftmost_col) {
                    bottom_left_offset = (0 * vec);
                } else {
                    bottom_left_offset = (0 * local) + ((col - 1) * vec);
                }
            } else {
                if (leftmost_col) {
                    bottom_left_offset = ((row/b + 1) * vec);
                } else {
                    bottom_left_offset = ((row/b + 1) * local) + ((col - 1) * vec);
                }
            }

            if (leftmost_col) {
                int new_vec_index = leftmost_p ? (vec_index - 1 + vec) % vec : vec_index;
                // scratch += device_packed_below_left[bottom_left_offset + new_vec_index];
            } else {
                // scratch += device_packed_below[bottom_left_offset + vec_index];
            }

        } else {
            if (leftmost_col) {
                bottom_left_offset = ((row/b) * vec);
                int new_vec_index = leftmost_p ? (vec_index - 1 + vec) % vec : vec_index;
                // scratch += device_packed_below_left[bottom_left_offset + new_vec_index];
            } else {
                bottom_left_offset = ((row/b) * local) + ((col - 1) * vec);
                // scratch += device_packed_below[bottom_left_offset + vec_index];
            }
        }
    } else {
        if (leftmost_col) {
            bottom_left_offset = ((row + 1) * vec);
            int new_vec_index = leftmost_p ? (vec_index - 1 + vec) % vec : vec_index;
            // scratch += device_packed_left[bottom_left_offset + new_vec_index];
        } else {
            bottom_left_offset = ((row + 1) * local) + ((col - 1) * vec);
            // scratch += device_in[bottom_left_offset + vec_index];
        }
    }

    // Bottom right corner
    int bottom_right_offset;
    if (bottom_of_block) {
        if (bottom_p) {
            if (bottom_block) {
                if (rightmost_col) {
                    bottom_right_offset = (0 * vec);
                } else {
                    bottom_right_offset = (0 * local) + ((col + 1) * vec);
                }
            } else {
                if (rightmost_col) {
                    bottom_right_offset = ((row/b + 1) * vec);
                } else {
                    bottom_right_offset = ((row/b + 1) * local) + ((col + 1) * vec);
                }
            }

            if (rightmost_col) {
                int new_vec_index = rightmost_p ? (vec_index + 1) % vec : vec_index;
                // scratch += device_packed_below_right[bottom_right_offset + new_vec_index];
            } else {
                // scratch += device_packed_below[bottom_right_offset + vec_index];
            }

        } else {
            if (rightmost_col) {
                bottom_right_offset = ((row/b) * vec);
                int new_vec_index = rightmost_p ? (vec_index + 1) % vec : vec_index;
                // scratch += device_packed_below_right[bottom_right_offset + new_vec_index];
            } else {
                bottom_right_offset = ((row/b) * local) + ((col + 1) * vec);
                // scratch += device_packed_below[bottom_right_offset + vec_index];
            }
        }
    } else {
        if (rightmost_col) {
            bottom_right_offset = ((row + 1) * vec);
            int new_vec_index = rightmost_p ? (vec_index + 1) % vec : vec_index;
            // scratch += device_packed_right[bottom_right_offset + new_vec_index];
        } else {
            bottom_right_offset = ((row + 1) * local) + ((col + 1) * vec);
            // scratch += device_in[bottom_right_offset + vec_index];
        }
    }

    device_out[my_offset + vec_index] = scratch;
}

// Packs the top row of all blocks
void pack_laplace_top(float *in, float *packed) {
    for (int i = 0; i < local/b; i++) {
        for (int j = 0; j < local; j++) {
            packed[i * local + j] = in[i * local * b + j];
        }
    }
}

// Packs the bottom row of all blocks
void pack_laplace_bottom(float *in, float *packed) {
    int offset = local * (b - 1);
    for (int i = 0; i < local/b; i++) {
        for (int j = 0; j < local; j++) {
            packed[i * local + j] = in[offset + i * local * b + j];
        }
    }
}

// Packs the leftmost col of vec in local
void pack_laplace_left(float *in, float *packed) {
    for (int i = 0; i < local; i++) {
        for (int j = 0; j < vec; j++) {
            packed[i * vec + j] = in[i * local + j];
        }
    }
}

// Packs the rightmost col of vec in local
void pack_laplace_right(float *in, float *packed) {
    int offset = local - vec;
    for (int i = 0; i < local; i++) {
        for (int j = 0; j < vec; j++) {
            packed[i * vec + j] = in[offset + i * local + j];
        }
    }
}

// Packs the top left element
// This corresponds to the top vec in each block of the leftmost col in local
void pack_laplace_top_left(float *in, float *packed) {
    for (int i = 0; i < local/b; i++) {
        for (int j = 0; j < vec; j++) {
            packed[i * vec + j] = in[i * local * b + j];
        }
    }
}

// Packs the top right element
// This corresponds to the top vec in each block of the rightmost col in local
void pack_laplace_top_right(float *in, float *packed) {
    int offset = local - vec;
    for (int i = 0; i < local/b; i++) {
        for (int j = 0; j < vec; j++) {
            packed[i * vec + j] = in[offset + i * local * b + j];
        }
    }
}

// Pack the bottom left element
// This corresponds to the bottom vec in each block of the leftmost col in local
void pack_laplace_bottom_left(float *in, float *packed) {
    int offset = local * (b - 1);
    for (int i = 0; i < local/b; i++) {
        for (int j = 0; j < vec; j++) {
            packed[i * vec + j] = in[offset + i * local * b + j];
        }
    }
}

// Pack the bottom right element
// This corresponds to the bottom vec in each block of the rightmost col in local
void pack_laplace_bottom_right(float *in, float *packed) {
    int offset = (local * (b - 1)) + (local - vec);
    for (int i = 0; i < local/b; i++) {
        for (int j = 0; j < vec; j++) {
            packed[i * vec + j] = in[offset + i * local * b + j];
        }
    }
}

void init_host(int rid, int cid, int local_size, float *host_in) {
    int row_offset = rid * b;       // offset based on which processor in the row
    int col_offset = cid * (N*b);   // offset based on which processor in the col
    int offset = row_offset + col_offset;

    float *init = (float*)malloc(local_size);
    
    for (int i = 0; i < local/b; i++) {
        for (int ii = 0; ii < b; ii++) {
            for (int j = 0; j < local/b; j++) {
                for (int jj = 0; jj < b; jj++) {
                    int array_index = (i * b * (local/b) * b) + (ii * (local/b) * b) + (j * b) + jj;
                    float real = (i * N * b * p) + (ii * N) + (j * b * p) + jj + offset;
                    init[array_index] = real;
                }
            }
        }
    }

    for (int i = 0; i < local; i++) {
        for (int j = 0; j < b; j++) {
            for (int jj = 0; jj < local/b; jj++) {
                int row = i * local;
                host_in[row + (j * (local/b))+ jj] = init[row + j + (jj * b)];
            }
        }
    }

    free(init);
}

// FIXME: RID AND CID ARE SWAPPED??? --> need to double check init function
// This is just for testing
void init_packed(int rid, int cid, int local_size,
                 float *host_packed_above, float *host_packed_below,
                 float *host_packed_left, float *host_packed_right,
                 float *host_packed_above_left, float *host_packed_above_right,
                 float *host_packed_below_left, float *host_packed_below_right) {

    float *host_in = (float*)malloc(local_size);

    int above_rid = (rid == 0) ? p - 1 : rid - 1;
    int below_rid = (rid + 1) % p;
    int left_cid = (cid == 0) ? p - 1 : cid - 1;
    int right_cid = (cid + 1) % p;

    //FIXME: For some reason flipping the rid and cid values gives the corrent values?

    init_host(rid, above_rid, local_size, host_in);
    pack_laplace_bottom(host_in, host_packed_above);

    init_host(rid, below_rid, local_size, host_in);
    pack_laplace_top(host_in, host_packed_below);

    init_host(left_cid, cid, local_size, host_in);
    pack_laplace_right(host_in, host_packed_left);

    init_host(right_cid, cid, local_size, host_in);
    pack_laplace_left(host_in, host_packed_right);

    init_host(left_cid, above_rid, local_size, host_in);
    pack_laplace_bottom_right(host_in, host_packed_above_left);

    init_host(right_cid, above_rid, local_size, host_in);
    pack_laplace_bottom_left(host_in, host_packed_above_right);

    init_host(left_cid, below_rid, local_size, host_in);
    pack_laplace_top_right(host_in, host_packed_below_left);

    init_host(right_cid, below_rid, local_size, host_in);
    pack_laplace_top_left(host_in, host_packed_below_right);
}

int main(int argc, char *argv[]) {
    int local_size = local*local*sizeof(float);
    int packed_row_size = local*(local/b)*sizeof(float);
    int packed_col_size = local*vec*sizeof(float);
    int packed_corner_size = (local/b)*vec*sizeof(float);

    float *host_in = (float*)malloc(local_size);
    float *host_out = (float*)calloc(local*local, sizeof(float));
    float *host_packed_above = (float*)malloc(packed_row_size);
    float *host_packed_below = (float*)malloc(packed_row_size);
    float *host_packed_left = (float*)malloc(packed_col_size);
    float *host_packed_right = (float*)malloc(packed_col_size);
    float *host_packed_above_left = (float*)malloc(packed_corner_size);
    float *host_packed_above_right = (float*)malloc(packed_corner_size);
    float *host_packed_below_left = (float*)malloc(packed_corner_size);
    float *host_packed_below_right = (float*)malloc(packed_corner_size);

    int rid = 0;
    int cid = 0;

    init_host(rid, cid, local_size, host_in);

    init_packed(rid, cid, local_size,
                host_packed_above, host_packed_below,
                host_packed_left, host_packed_right,
                host_packed_above_left, host_packed_above_right,
                host_packed_below_left, host_packed_below_right);

    float *device_in, *device_out;
    float *device_packed_above, *device_packed_below, *device_packed_left, *device_packed_right;
    float *device_packed_above_left, *device_packed_above_right, *device_packed_below_left, *device_packed_below_right;
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
                                rid, cid,
                                device_packed_above, device_packed_below,
                                device_packed_left, device_packed_right,
                                device_packed_above_left, device_packed_above_right,
                                device_packed_below_left, device_packed_below_right);
    cudaDeviceSynchronize();

    cudaMemcpy(host_out, device_out, local_size, cudaMemcpyDeviceToHost);

    std::cout << "Host out" << std::endl;
    for (int i = 0; i < local; i++) {
        for (int j = 0; j < local; j++) {
            std::cout << host_out[i * local + j] << " ";
        }
        std::cout << std::endl;
    }

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