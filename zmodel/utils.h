#ifndef __ZMODEL__UTILS__
#define __ZMODEL__UTILS__

#define __ZMODEL__CUDA__
// #define __ZMODEL__HIP__
#include "device_macros.h"
#include <vector>

using Complex = DEVICE_FFT_DOUBLECOMPLEX;
using Scalar = double;

template <typename T>
using Vector = std::vector<T>;

#define __PRINT__TIMING__
#define __PRINT__DETAILED__TIMING__ 
#define __PRINT__RESULTS__
// #define __PRINT__TWIDDLES__
#define __PRINT__SANITY__
#define __ZMODEL__COMPUTE__
// #define __GPU__AWARE__MPI__
// #define __GPU__SET__
// #define __PRINT__RESULTS__TO__FILE__
// #define __USE__FFTDX__
// #define  __ZMODEL__INCLUDE__INVERSE__

// NOTE: these macros can be set on the commandline using: -UB_DIM -DB_DIM=8
#ifndef N_DIM
    #define N_DIM (16)
#endif
#ifndef B_DIM
    #define B_DIM (2)
#endif
#ifndef P_DIM
    // Number of processors in the rows or columns. ex: p = 2 --> 4 total processors
    // This is really sqrt(P)
    #define P_DIM (2)
#endif

#define RUNS (1)
#define NUM_STREAMS (4)

#define CUFFTDX_TARGET_SM 750

/******************** 2D MACROS ********************/

#define LOCAL_DIM (N_DIM/P_DIM)

#define LOCAL_COMPLEX_BYTES (LOCAL_DIM*LOCAL_DIM*sizeof(Complex))
#define PACKED_ROW_COMPLEX (LOCAL_DIM*(LOCAL_DIM/B_DIM))
#define PACKED_ROW_COMPLEX_BYTES (PACKED_ROW_COMPLEX*sizeof(Complex))
#define PACKED_COL_COMPLEX (LOCAL_DIM*VEC)
#define PACKED_COL_COMPLEX_BYTES (PACKED_COL_COMPLEX*sizeof(Complex))

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

/******************** 3D MACROS ********************/

#define LOCAL_COMPLEX_BYTES_3D (LOCAL_DIM*LOCAL_DIM*LOCAL_DIM*sizeof(Complex))

/******************** Operators ********************/

__host__ __device__ static inline Complex operator*(double a, const Complex& b) {
    return {a * b.x, a * b.y};
}

__host__ __device__ static inline Complex& operator+=(Complex& a, const Complex& b) {
    a.x += b.x;
    a.y += b.y;
    return a;
}

__host__ __device__ static inline Complex& operator-=(Complex& a, const Complex& b) {
    a.x -= b.x;
    a.y -= b.y;
    return a;
}

/******************** 2D Init ********************/

template<typename T>
inline void init_host(int rid, int cid, T *host_in) {
    int row_offset = cid * B_DIM;           // offset based on which processor in the row
    int col_offset = rid * (N_DIM*B_DIM);   // offset based on which processor in the col
    int offset = row_offset + col_offset;

    T*init = (T*)malloc(LOCAL_COMPLEX_BYTES);
    
    for (int i = 0; i < LOCAL_DIM/B_DIM; i++) {
        for (int ii = 0; ii < B_DIM; ii++) {
            for (int j = 0; j < LOCAL_DIM/B_DIM; j++) {
                for (int jj = 0; jj < B_DIM; jj++) {
                    int array_index = (i * B_DIM * (LOCAL_DIM/B_DIM) * B_DIM) + (ii * (LOCAL_DIM/B_DIM) * B_DIM) + (j * B_DIM) + jj;
                    double real = (i * N_DIM * B_DIM * P_DIM) + (ii * N_DIM) + (j * B_DIM * P_DIM) + jj + offset;

                    if constexpr (std::is_same_v<T, Complex>) {
                        init[array_index] = {real, 0.0};
                    } else {
                        init[array_index] = real;
                    }
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

inline void print_block_cyclic(Complex *host_in) {
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

inline void store_block_cyclic(Complex *host_in) {
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

/******************** 3D Init ********************/

#endif // __ZMODEL__UTILS__