#include <iostream>
#include "../utils.h"

// FIXME: Check that the column alignment vec shift isnt parameterized by the number of processors or something else
// FIXME: I wonder if we can make some of these if else checks happen at compile time
// FIXME: the corners should be broken out into helper functions in order to cleanup redundant code
// FIXME: Need to add scaling factor
__global__ void laplace(Scalar *device_in, Scalar *device_out,
                        int rid, int cid,
                        Scalar *device_packed_above, Scalar *device_packed_below,
                        Scalar *device_packed_left, Scalar *device_packed_right,
                        Scalar *device_packed_above_left, Scalar *device_packed_above_right,
                        Scalar *device_packed_below_left, Scalar *device_packed_below_right) {
    int row = blockIdx.x / VEC_COL;
    int col = blockIdx.x % VEC_COL;
    int my_offset = (row * LOCAL_DIM) + (col * VEC);

    int vec_index = threadIdx.x;

    float scratch = 0.0;

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
        // scratch += device_packed_above[above_offset + vec_index];
    } else {
        above_offset = ((row - 1) * LOCAL_DIM) + (col * VEC);
        // scratch += device_in[above_offset + vec_index];
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
        // scratch += device_packed_below[below_offset + vec_index];
    } else {
        below_offset = ((row + 1) * LOCAL_DIM) + (col * VEC);
        // scratch += device_in[below_offset + vec_index];
    }

    // Get col to left
    // Assumes you are working with the packed rightmost col of the vec data from the processor to the left of you
    int left_offset;
    if (leftmost_col) {
        // Periodic boundary conditions force you to shift the vec to align elements when on leftmost processor
        left_offset = (row * VEC);
        int new_vec_index = leftmost_p ? (vec_index - 1 + VEC) % VEC : vec_index;
        // scratch += device_packed_left[left_offset + new_vec_index];
    } else {
        left_offset = (row * LOCAL_DIM) + ((col - 1) * VEC);
        // scratch += device_in[left_offset + vec_index];
    }

    // Get col to right
    // Assumes you are working with the packed leftmost col of the vec data from the processor to the right of you
    int right_offset;
    if (rightmost_col) {
        right_offset = (row * VEC);
        int new_vec_index = rightmost_p ? (vec_index + 1) % VEC : vec_index;
        // scratch += device_packed_right[right_offset + new_vec_index];
    } else {
        right_offset = (row * LOCAL_DIM) + ((col + 1) * VEC);
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
                    top_left_offset = ((LOCAL_DIM/B_DIM - 1) * VEC);
                } else {
                    // Data comes from above packing so the whole row is stored
                    top_left_offset = ((LOCAL_DIM/B_DIM - 1) * LOCAL_DIM) + ((col - 1) * VEC);
                }
            } else {
                if (leftmost_col) {
                    top_left_offset = ((row/B_DIM - 1) * VEC);
                } else {
                    top_left_offset = ((row/B_DIM - 1) * LOCAL_DIM) + ((col - 1) * VEC);
                }
            }

            // Calculate the shift
            if (leftmost_col) {
                // leftmost gets data from corner packing
                int new_vec_index = leftmost_p ? (vec_index - 1 + VEC) % VEC : vec_index;
                // scratch += device_packed_above_left[top_left_offset + new_vec_index];
            } else {
                // Non leftmost gets data from above
                // scratch += device_packed_above[top_left_offset + vec_index];
            }
        } else {
            // Use the same row of the packed data because it is not the top p
            if (leftmost_col) {
                top_left_offset = ((row/B_DIM) * VEC);
                int new_vec_index = leftmost_p ? (vec_index - 1 + VEC) % VEC : vec_index;
                // scratch += device_packed_above_left[top_left_offset + new_vec_index];
            } else {
                top_left_offset = ((row/B_DIM)*LOCAL_DIM) + ((col - 1) * VEC);
                // scratch += device_packed_above[top_left_offset + vec_index];
            }
        }
    } else {
        // This is the base case
        if (leftmost_col) {
            top_left_offset = ((row - 1) * VEC);
            int new_vec_index = leftmost_p ? (vec_index - 1 + VEC) % VEC : vec_index;
            // scratch += device_packed_left[top_left_offset + new_vec_index];
        } else {
            top_left_offset = ((row - 1) * LOCAL_DIM) + ((col - 1) * VEC);
            // scratch += device_in[top_left_offset + vec_index];
        }
    }

    // Top right corner
    int top_right_offset;
    if (top_of_block) {
        if (top_p) {
            if (top_block) {
                if (rightmost_col) {
                    top_right_offset = ((LOCAL_DIM/B_DIM - 1) * VEC);
                } else {
                    top_right_offset = ((LOCAL_DIM/B_DIM - 1) * LOCAL_DIM) + ((col + 1) * VEC);
                }
            } else {
                if (rightmost_col) {
                    top_right_offset = ((row/B_DIM - 1) * VEC);
                } else {
                    top_right_offset = ((row/B_DIM - 1) * LOCAL_DIM) + ((col + 1) * VEC);
                }
            }

            if (rightmost_col) {
                int new_vec_index = rightmost_p ? (vec_index + 1) % VEC : vec_index;
                // scratch += device_packed_above_right[top_right_offset + new_vec_index];
            } else {
                // scratch += device_packed_above[top_right_offset + vec_index];
            }
        } else {
            if (rightmost_col) {
                top_right_offset = ((row/B_DIM) * VEC);
                int new_vec_index = rightmost_p ? (vec_index + 1) % VEC : vec_index;
                // scratch += device_packed_above_right[top_right_offset + new_vec_index];
            } else {
                top_right_offset = ((row/B_DIM) * LOCAL_DIM) + ((col + 1) * VEC);
                // scratch += device_packed_above[top_right_offset + vec_index];
            }
        }
    } else {
        if (rightmost_col) {
            top_right_offset = ((row - 1) * VEC);
            int new_vec_index = rightmost_p ? (vec_index + 1) % VEC : vec_index;
            // scratch += device_packed_right[top_right_offset + new_vec_index];
        } else {
            top_right_offset = ((row - 1) * LOCAL_DIM) + ((col + 1) * VEC);
            // scratch += device_in[top_right_offset + vec_index];
        }
    }

    // Bottom left corner
    int bottom_left_offset;
    if (bottom_of_block) {
        if (bottom_p) {
            if (bottom_block) {
                if (leftmost_col) {
                    bottom_left_offset = (0 * VEC);
                } else {
                    bottom_left_offset = (0 * LOCAL_DIM) + ((col - 1) * VEC);
                }
            } else {
                if (leftmost_col) {
                    bottom_left_offset = ((row/B_DIM + 1) * VEC);
                } else {
                    bottom_left_offset = ((row/B_DIM + 1) * LOCAL_DIM) + ((col - 1) * VEC);
                }
            }

            if (leftmost_col) {
                int new_vec_index = leftmost_p ? (vec_index - 1 + VEC) % VEC : vec_index;
                // scratch += device_packed_below_left[bottom_left_offset + new_vec_index];
            } else {
                // scratch += device_packed_below[bottom_left_offset + vec_index];
            }

        } else {
            if (leftmost_col) {
                bottom_left_offset = ((row/B_DIM) * VEC);
                int new_vec_index = leftmost_p ? (vec_index - 1 + VEC) % VEC : vec_index;
                // scratch += device_packed_below_left[bottom_left_offset + new_vec_index];
            } else {
                bottom_left_offset = ((row/B_DIM) * LOCAL_DIM) + ((col - 1) * VEC);
                // scratch += device_packed_below[bottom_left_offset + vec_index];
            }
        }
    } else {
        if (leftmost_col) {
            bottom_left_offset = ((row + 1) * VEC);
            int new_vec_index = leftmost_p ? (vec_index - 1 + VEC) % VEC : vec_index;
            // scratch += device_packed_left[bottom_left_offset + new_vec_index];
        } else {
            bottom_left_offset = ((row + 1) * LOCAL_DIM) + ((col - 1) * VEC);
            // scratch += device_in[bottom_left_offset + vec_index];
        }
    }

    // Bottom right corner
    int bottom_right_offset;
    if (bottom_of_block) {
        if (bottom_p) {
            if (bottom_block) {
                if (rightmost_col) {
                    bottom_right_offset = (0 * VEC);
                } else {
                    bottom_right_offset = (0 * LOCAL_DIM) + ((col + 1) * VEC);
                }
            } else {
                if (rightmost_col) {
                    bottom_right_offset = ((row/B_DIM + 1) * VEC);
                } else {
                    bottom_right_offset = ((row/B_DIM + 1) * LOCAL_DIM) + ((col + 1) * VEC);
                }
            }

            if (rightmost_col) {
                int new_vec_index = rightmost_p ? (vec_index + 1) % VEC : vec_index;
                // scratch += device_packed_below_right[bottom_right_offset + new_vec_index];
            } else {
                // scratch += device_packed_below[bottom_right_offset + vec_index];
            }

        } else {
            if (rightmost_col) {
                bottom_right_offset = ((row/B_DIM) * VEC);
                int new_vec_index = rightmost_p ? (vec_index + 1) % VEC : vec_index;
                // scratch += device_packed_below_right[bottom_right_offset + new_vec_index];
            } else {
                bottom_right_offset = ((row/B_DIM) * LOCAL_DIM) + ((col + 1) * VEC);
                // scratch += device_packed_below[bottom_right_offset + vec_index];
            }
        }
    } else {
        if (rightmost_col) {
            bottom_right_offset = ((row + 1) * VEC);
            int new_vec_index = rightmost_p ? (vec_index + 1) % VEC : vec_index;
            // scratch += device_packed_right[bottom_right_offset + new_vec_index];
        } else {
            bottom_right_offset = ((row + 1) * LOCAL_DIM) + ((col + 1) * VEC);
            // scratch += device_in[bottom_right_offset + vec_index];
        }
    }

    device_out[my_offset + vec_index] = scratch;
}

// Packs the top row of all blocks
void pack_laplace_top(Scalar *in, Scalar *packed) {
    for (int i = 0; i < LOCAL_DIM/B_DIM; i++) {
        for (int j = 0; j < LOCAL_DIM; j++) {
            packed[i * LOCAL_DIM + j] = in[i * LOCAL_DIM * B_DIM + j];
        }
    }
}

// Packs the bottom row of all blocks
void pack_laplace_bottom(Scalar *in, Scalar *packed) {
    int offset = LOCAL_DIM * (B_DIM - 1);
    for (int i = 0; i < LOCAL_DIM/B_DIM; i++) {
        for (int j = 0; j < LOCAL_DIM; j++) {
            packed[i * LOCAL_DIM + j] = in[offset + i * LOCAL_DIM * B_DIM + j];
        }
    }
}

// Packs the leftmost col of vec in local
void pack_laplace_left(Scalar *in, Scalar *packed) {
    for (int i = 0; i < LOCAL_DIM; i++) {
        for (int j = 0; j < VEC; j++) {
            packed[i * VEC + j] = in[i * LOCAL_DIM + j];
        }
    }
}

// Packs the rightmost col of vec in local
void pack_laplace_right(Scalar *in, Scalar *packed) {
    int offset = LOCAL_DIM - VEC;
    for (int i = 0; i < LOCAL_DIM; i++) {
        for (int j = 0; j < VEC; j++) {
            packed[i * VEC + j] = in[offset + i * LOCAL_DIM + j];
        }
    }
}

// Packs the top left element
// This corresponds to the top vec in each block of the leftmost col in local
void pack_laplace_top_left(Scalar *in, Scalar *packed) {
    for (int i = 0; i < LOCAL_DIM/B_DIM; i++) {
        for (int j = 0; j < VEC; j++) {
            packed[i * VEC + j] = in[i * LOCAL_DIM * B_DIM + j];
        }
    }
}

// Packs the top right element
// This corresponds to the top vec in each block of the rightmost col in local
void pack_laplace_top_right(Scalar *in, Scalar *packed) {
    int offset = LOCAL_DIM - VEC;
    for (int i = 0; i < LOCAL_DIM/B_DIM; i++) {
        for (int j = 0; j < VEC; j++) {
            packed[i * VEC + j] = in[offset + i * LOCAL_DIM * B_DIM + j];
        }
    }
}

// Pack the bottom left element
// This corresponds to the bottom vec in each block of the leftmost col in local
void pack_laplace_bottom_left(Scalar *in, Scalar *packed) {
    int offset = LOCAL_DIM * (B_DIM - 1);
    for (int i = 0; i < LOCAL_DIM/B_DIM; i++) {
        for (int j = 0; j < VEC; j++) {
            packed[i * VEC + j] = in[offset + i * LOCAL_DIM * B_DIM + j];
        }
    }
}

// Pack the bottom right element
// This corresponds to the bottom vec in each block of the rightmost col in local
void pack_laplace_bottom_right(Scalar *in, Scalar *packed) {
    int offset = (LOCAL_DIM * (B_DIM - 1)) + (LOCAL_DIM - VEC);
    for (int i = 0; i < LOCAL_DIM/B_DIM; i++) {
        for (int j = 0; j < VEC; j++) {
            packed[i * VEC + j] = in[offset + i * LOCAL_DIM * B_DIM + j];
        }
    }
}

// NOTE: This is just for testing
void init_packed(int rid, int cid,
                 Scalar *host_packed_above, Scalar *host_packed_below,
                 Scalar *host_packed_left, Scalar *host_packed_right,
                 Scalar *host_packed_above_left, Scalar *host_packed_above_right,
                 Scalar *host_packed_below_left, Scalar *host_packed_below_right) {

    Scalar *host_in = (Scalar*)malloc(LOCAL_SCALAR_BYTES);

    int above_rid = (rid == 0) ? P_DIM - 1 : rid - 1;
    int below_rid = (rid + 1) % P_DIM;
    int left_cid = (cid == 0) ? P_DIM - 1 : cid - 1;
    int right_cid = (cid + 1) % P_DIM;

    init_host_scalar(above_rid, cid, host_in);
    pack_laplace_bottom(host_in, host_packed_above);

    init_host_scalar(below_rid, cid, host_in);
    pack_laplace_top(host_in, host_packed_below);

    init_host_scalar(rid, left_cid, host_in);
    pack_laplace_right(host_in, host_packed_left);

    init_host_scalar(rid, right_cid, host_in);
    pack_laplace_left(host_in, host_packed_right);

    init_host_scalar(above_rid, left_cid, host_in);
    pack_laplace_bottom_right(host_in, host_packed_above_left);

    init_host_scalar(above_rid, right_cid, host_in);
    pack_laplace_bottom_left(host_in, host_packed_above_right);

    init_host_scalar(below_rid, left_cid, host_in);
    pack_laplace_top_right(host_in, host_packed_below_left);

    init_host_scalar(below_rid, right_cid, host_in);
    pack_laplace_top_left(host_in, host_packed_below_right);
}

void test_laplace() {
    Scalar *host_in = (Scalar*)malloc(LOCAL_SCALAR_BYTES);
    Scalar *host_out = (Scalar*)calloc(LOCAL_DIM*LOCAL_DIM, sizeof(Scalar));
    Scalar *host_packed_above = (Scalar*)malloc(PACKED_ROW_SCALAR_BYTES);
    Scalar *host_packed_below = (Scalar*)malloc(PACKED_ROW_SCALAR_BYTES);
    Scalar *host_packed_left = (Scalar*)malloc(PACKED_COL_SCALAR_BYTES);
    Scalar *host_packed_right = (Scalar*)malloc(PACKED_COL_SCALAR_BYTES);
    Scalar *host_packed_above_left = (Scalar*)malloc(PACKED_CORNER_SCALAR_BYTES);
    Scalar *host_packed_above_right = (Scalar*)malloc(PACKED_CORNER_SCALAR_BYTES);
    Scalar *host_packed_below_left = (Scalar*)malloc(PACKED_CORNER_SCALAR_BYTES);
    Scalar *host_packed_below_right = (Scalar*)malloc(PACKED_CORNER_SCALAR_BYTES);

    int rid = 0;
    int cid = 0;

    init_host_scalar(rid, cid, host_in);

    init_packed(rid, cid,
                host_packed_above, host_packed_below,
                host_packed_left, host_packed_right,
                host_packed_above_left, host_packed_above_right,
                host_packed_below_left, host_packed_below_right);

    Scalar *device_in, *device_out;
    Scalar *device_packed_above, *device_packed_below, *device_packed_left, *device_packed_right;
    Scalar *device_packed_above_left, *device_packed_above_right, *device_packed_below_left, *device_packed_below_right;
    cudaMalloc((void**)&device_in, LOCAL_SCALAR_BYTES);
    cudaMalloc((void**)&device_out, LOCAL_SCALAR_BYTES);
    cudaMalloc((void**)&device_packed_above, PACKED_ROW_SCALAR_BYTES);
    cudaMalloc((void**)&device_packed_below, PACKED_ROW_SCALAR_BYTES);
    cudaMalloc((void**)&device_packed_left, PACKED_COL_SCALAR_BYTES);
    cudaMalloc((void**)&device_packed_right, PACKED_COL_SCALAR_BYTES);
    cudaMalloc((void**)&device_packed_above_left, PACKED_CORNER_SCALAR_BYTES);
    cudaMalloc((void**)&device_packed_above_right, PACKED_CORNER_SCALAR_BYTES);
    cudaMalloc((void**)&device_packed_below_left, PACKED_CORNER_SCALAR_BYTES);
    cudaMalloc((void**)&device_packed_below_right, PACKED_CORNER_SCALAR_BYTES);

    cudaMemcpy(device_in, host_in, LOCAL_SCALAR_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(device_out, host_out, LOCAL_SCALAR_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(device_packed_above, host_packed_above, PACKED_ROW_SCALAR_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(device_packed_below, host_packed_below, PACKED_ROW_SCALAR_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(device_packed_left, host_packed_left, PACKED_COL_SCALAR_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(device_packed_right, host_packed_right, PACKED_COL_SCALAR_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(device_packed_above_left, host_packed_above_left, PACKED_CORNER_SCALAR_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(device_packed_above_right, host_packed_above_right, PACKED_CORNER_SCALAR_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(device_packed_below_left, host_packed_below_left, PACKED_CORNER_SCALAR_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(device_packed_below_right, host_packed_below_right, PACKED_CORNER_SCALAR_BYTES, cudaMemcpyHostToDevice);

    laplace<<<VEC_TOTAL, VEC>>>(device_in, device_out,
                                rid, cid,
                                device_packed_above, device_packed_below,
                                device_packed_left, device_packed_right,
                                device_packed_above_left, device_packed_above_right,
                                device_packed_below_left, device_packed_below_right);
    cudaDeviceSynchronize();

    cudaMemcpy(host_out, device_out, LOCAL_SCALAR_BYTES, cudaMemcpyDeviceToHost);

    std::cout << "Host out" << std::endl;
    for (int i = 0; i < LOCAL_DIM; i++) {
        for (int j = 0; j < LOCAL_DIM; j++) {
            std::cout << host_out[i * LOCAL_DIM + j] << " ";
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