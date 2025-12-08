#ifndef __ZMODEL__UTILS__
#define __ZMODEL__UTILS__

#define __ZMODEL__CUDA__
// #define __ZMODEL__HIP__
#include "device_macros.h"
#include <vector>

#define __PRINT__TIMING__
#define __PRINT__DETAILED__TIMMING__
// #define __PRINT__RESULTS__
// #define __PRINT__TWIDDLES__
#define __PRINT__SANITY__
#define __ZMODEL__COMPUTE__
// #define __GPU__AWARE__MPI__
// #define __GPU__SET__
// #define __PRINT__RESULTS__TO__FILE___

// NOTE: these macros can be set on the commandline using: -UB_DIM -DB_DIM=8
#ifndef N_DIM
    #define N_DIM (64)
#endif
#ifndef B_DIM
    #define B_DIM (4)
#endif
#ifndef P_DIM
    // Number of processors in the rows or columns. ex: p = 2 --> 4 total processors
    // This is really sqrt(P)
    #define P_DIM (2)
#endif

#define NUM_STREAMS (4)

using Complex = DEVICE_FFT_DOUBLECOMPLEX;
using Scalar = double;

template <typename T>
using Vector = std::vector<T>;

#define LOCAL_DIM (N_DIM/P_DIM)
#define LOCAL_COMPLEX_BYTES (LOCAL_DIM*LOCAL_DIM*sizeof(Complex))
#define LOCAL_SCALAR_BYTES (LOCAL_DIM*LOCAL_DIM*sizeof(Scalar))
#define PACKED_ROW_SCALAR_BYTES (LOCAL_DIM*(LOCAL_DIM/B_DIM)*sizeof(Scalar))
#define PACKED_COL_SCALAR_BYTES (LOCAL_DIM*VEC*sizeof(Scalar))
#define PACKED_CORNER_SCALAR_BYTES ((LOCAL_DIM/B_DIM)*VEC*sizeof(Scalar))

// This is the size of the vector the stencil will be batched with
#define VEC (LOCAL_DIM/B_DIM)
// The total number of vectors on the local processor
#define VEC_TOTAL ((LOCAL_DIM*LOCAL_DIM)/VEC)
 
#define VEC_ROW (LOCAL_DIM) // Number of vectors in a col
#define VEC_COL (B_DIM)     // Number of vectors in a row

static inline void init_host(int rid, int cid, Complex *host_in) {
    int row_offset = cid * B_DIM;           // offset based on which processor in the row
    int col_offset = rid * (N_DIM*B_DIM);   // offset based on which processor in the col
    int offset = row_offset + col_offset;

    Complex *init = (Complex*)malloc(LOCAL_COMPLEX_BYTES);
    
    for (int i = 0; i < LOCAL_DIM/B_DIM; i++) {
        for (int ii = 0; ii < B_DIM; ii++) {
            for (int j = 0; j < LOCAL_DIM/B_DIM; j++) {
                for (int jj = 0; jj < B_DIM; jj++) {
                    int array_index = (i * B_DIM * (LOCAL_DIM/B_DIM) * B_DIM) + (ii * (LOCAL_DIM/B_DIM) * B_DIM) + (j * B_DIM) + jj;
                    double real = (i * N_DIM * B_DIM * P_DIM) + (ii * N_DIM) + (j * B_DIM * P_DIM) + jj + offset;
                    init[array_index] = {real, 0.0};
                }
            }
        }
    }

    for (int i = 0; i < LOCAL_DIM; i++) {
        for (int j = 0; j < B_DIM; j++) {
            for (int jj = 0; jj < LOCAL_DIM/B_DIM; jj++) {
                int row = i * LOCAL_DIM;
                host_in[row + (j * (LOCAL_DIM/B_DIM))+ jj] = init[row + j + (jj * B_DIM)];
            }
        }
    }

    free(init);
}

// FIXME: USE TEMPLATE TO COMBINE THIS WITH THE COMPLEX VERSION
static inline void init_host_scalar(int rid, int cid, Scalar *host_in) {
    int row_offset = cid * B_DIM;           // offset based on which processor in the row
    int col_offset = rid * (N_DIM*B_DIM);   // offset based on which processor in the col
    int offset = row_offset + col_offset;

    Scalar *init = (Scalar*)malloc(LOCAL_COMPLEX_BYTES);
    
    for (int i = 0; i < LOCAL_DIM/B_DIM; i++) {
        for (int ii = 0; ii < B_DIM; ii++) {
            for (int j = 0; j < LOCAL_DIM/B_DIM; j++) {
                for (int jj = 0; jj < B_DIM; jj++) {
                    int array_index = (i * B_DIM * (LOCAL_DIM/B_DIM) * B_DIM) + (ii * (LOCAL_DIM/B_DIM) * B_DIM) + (j * B_DIM) + jj;
                    double real = (i * N_DIM * B_DIM * P_DIM) + (ii * N_DIM) + (j * B_DIM * P_DIM) + jj + offset;
                    init[array_index] = real;
                }
            }
        }
    }

    for (int i = 0; i < LOCAL_DIM; i++) {
        for (int j = 0; j < B_DIM; j++) {
            for (int jj = 0; jj < LOCAL_DIM/B_DIM; jj++) {
                int row = i * LOCAL_DIM;
                host_in[row + (j * (LOCAL_DIM/B_DIM))+ jj] = init[row + j + (jj * B_DIM)];
            }
        }
    }

    free(init);
}

static inline void print_block_cyclic(Complex *host_in) {
    Complex *init = (Complex*)malloc(LOCAL_COMPLEX_BYTES);

    for (int i = 0; i < LOCAL_DIM; i++) {
        for (int j = 0; j < B_DIM; j++) {
            for (int jj = 0; jj < LOCAL_DIM/B_DIM; jj++) {
                int row = i * LOCAL_DIM;
                init[row + j + (jj * B_DIM)] = host_in[row + (j * (LOCAL_DIM/B_DIM))+ jj];
            }
        }
    }

    std::cout << "Block Cyclic" << std::endl;
    for (int i = 0; i < LOCAL_DIM; i++) {
        for (int j = 0; j < LOCAL_DIM; j++) {
            std::cout << "(" << init[i * LOCAL_DIM + j].x << "," << init[i * LOCAL_DIM + j].y << ") ";
        }
        std::cout << std::endl;
    }

    free(init);
}

static inline void store_block_cyclic(Complex *host_in) {
    Complex *init = (Complex*)malloc(LOCAL_COMPLEX_BYTES);

    for (int i = 0; i < LOCAL_DIM; i++) {
        for (int j = 0; j < B_DIM; j++) {
            for (int jj = 0; jj < LOCAL_DIM/B_DIM; jj++) {
                int row = i * LOCAL_DIM;
                init[row + j + (jj * B_DIM)] = host_in[row + (j * (LOCAL_DIM/B_DIM))+ jj];
            }
        }
    }

    for (int i = 0; i < LOCAL_DIM; i++) {
        for (int j = 0; j < LOCAL_DIM; j++) {
            host_in[i * LOCAL_DIM + j] = init[i * LOCAL_DIM + j];
        }
    }

    free(init);
}

#endif // __ZMODEL__UTILS__