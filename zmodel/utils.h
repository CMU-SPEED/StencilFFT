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

__host__ __device__ static inline Complex operator*(const Complex& a, const Complex& b) {
    return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

/******************** 2D Helpers ********************/

template<typename T>
inline void init_host(int rid, int cid, T *host_in) {
    size_t row_offset = cid * B_DIM;           // offset based on which processor in the row
    size_t col_offset = rid * (N_DIM*B_DIM);   // offset based on which processor in the col
    size_t offset = row_offset + col_offset;

    T *init = (T*)malloc(LOCAL_COMPLEX_BYTES);
    
    for (size_t i = 0; i < LOCAL_DIM/B_DIM; i++) {
        for (size_t ii = 0; ii < B_DIM; ii++) {
            for (size_t j = 0; j < LOCAL_DIM/B_DIM; j++) {
                for (size_t jj = 0; jj < B_DIM; jj++) {
                    size_t array_index = (i * B_DIM * (LOCAL_DIM/B_DIM) * B_DIM) + (ii * (LOCAL_DIM/B_DIM) * B_DIM) + (j * B_DIM) + jj;
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

    for (size_t i = 0; i < LOCAL_DIM; i++) {
        for (size_t j = 0; j < B_DIM; j++) {
            for (size_t jj = 0; jj < LOCAL_DIM/B_DIM; jj++) {
                size_t row = i * LOCAL_DIM;
                host_in[row + (j * (LOCAL_DIM/B_DIM))+ jj] = init[row + j + (jj * B_DIM)];
            }
        }
    }

    free(init);
}

/******************** 3D Helpers ********************/

template<typename T>
inline void init_host_3d(int rid, int cid, int did, T *host_in) {
    size_t row_offset = cid * B_DIM;                    // offset based on which processor in the row
    size_t col_offset = rid * (N_DIM*B_DIM);            // offset based on which processor in the col
    size_t dep_offset = did * (N_DIM * N_DIM * B_DIM);  // offset based on which processor in depth
    size_t offset = row_offset + col_offset + dep_offset;

    T *init = (T*)malloc(LOCAL_COMPLEX_BYTES_3D);

    for (size_t i = 0; i < LOCAL_DIM/B_DIM; i++) {
        for (size_t ii = 0; ii < B_DIM; ii++) {
            for (size_t j = 0; j < LOCAL_DIM/B_DIM; j++) {
                for (size_t jj = 0; jj < B_DIM; jj++) {
                    for (size_t k = 0; k < LOCAL_DIM/B_DIM; k++) {
                        for (size_t kk = 0; kk < B_DIM; kk++) {
                            size_t array_index = (i * B_DIM * LOCAL_DIM * LOCAL_DIM) + (ii * LOCAL_DIM * LOCAL_DIM) +
                                                 (j * B_DIM * LOCAL_DIM) + (jj * LOCAL_DIM) + (k * B_DIM) + kk;
                            double real = (i * N_DIM * N_DIM * B_DIM * P_DIM) + (ii * N_DIM * N_DIM) +
                                          (j * N_DIM * B_DIM * P_DIM) + (jj * N_DIM) +
                                          (k * B_DIM * P_DIM) + kk + offset;

                            if constexpr (std::is_same_v<T, Complex>) {
                               init[array_index] = {real, 0.0};
                            } else {
                                init[array_index] = real;
                            }
                        }
                    }
                }
            }
        }
    }

    for (size_t i = 0; i < LOCAL_DIM; i++) {
        size_t depth = i * LOCAL_DIM * LOCAL_DIM;
        for (size_t ii = 0; ii < LOCAL_DIM; ii++) {
            size_t row = ii * LOCAL_DIM;
            for (size_t j = 0; j < B_DIM; j++) {
                for (size_t jj = 0; jj < LOCAL_DIM/B_DIM; jj++) {
                    host_in[depth + row + (j * (LOCAL_DIM/B_DIM))+ jj] = init[depth + row + j + (jj * B_DIM)];
                }
            }
        }
    }

    free(init);
}

#endif // __ZMODEL__UTILS__