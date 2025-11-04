#ifndef __ZMODEL__UTILS__
#define __ZMODEL__UTILS__

#define __ZMODEL__CUDA__
// #define __ZMODEL__HIP__
#include "device_macros.h"

#define __PRINT__TIMING__
#define __PRINT__RESULTS__
// #define __PRINT__TWIDDLES__
// #define __PRINT__SANITY__
#define __ZMODEL__COMPUTE__

// NOTE: these macros can be set on the commandline
#define N_DIM (64)
#define B_DIM (4)
#define P_DIM (2) // Number of processors in the rows or columns. ex: p = 2 --> 4 total processors
#define LOCAL_DIM (N_DIM/P_DIM)
#define LOCAL_BYTES (LOCAL_DIM*LOCAL_DIM*sizeof(Complex))

// This is the size of the vector the stencil will be batched with
#define VEC (LOCAL_DIM/B_DIM)
// The total number of vectors on the local processor
#define VEC_TOTAL ((LOCAL_DIM*LOCAL_DIM)/VEC)
 
#define VEC_ROW (LOCAL_DIM) // Number of vectors in a col
#define VEC_COL (B_DIM)     // Number of vectors in a row

using Complex = DEVICE_FFT_DOUBLECOMPLEX;

void init_host(int rid, int cid, Complex *host_in) {
    int row_offset = cid * B_DIM;           // offset based on which processor in the row
    int col_offset = rid * (N_DIM*B_DIM);   // offset based on which processor in the col
    int offset = row_offset + col_offset;

    Complex *init = (Complex*)malloc(LOCAL_BYTES);
    
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

void print_block_cyclic(Complex *host_in) {
    Complex *init = (Complex*)malloc(LOCAL_BYTES);

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

void init_plans(cufftHandle *plan0, cufftHandle *plan1, cufftHandle *plan2, cufftHandle *plan3) {
    DEVICE_FFT_SAFE_CALL(DEVICE_FFT_CREATE(plan0));
    DEVICE_FFT_SAFE_CALL(DEVICE_FFT_CREATE(plan1));
    DEVICE_FFT_SAFE_CALL(DEVICE_FFT_CREATE(plan2));
    DEVICE_FFT_SAFE_CALL(DEVICE_FFT_CREATE(plan3));

    const int rank0 = 1;
    int n0[1] = { VEC };
    int inembed0[1] = { VEC };
    int onembed0[1] = { VEC };
    const int istride0 = 1;
    const int ostride0 = 1;
    const int idist0 = VEC;
    const int odist0 = VEC;
    const int batch0 = VEC_TOTAL;
    size_t workSize0 = 0;
    DEVICE_FFT_SAFE_CALL(DEVICE_FFT_MAKE_PLAN_MANY(
      *plan0, rank0, n0,
      inembed0,  istride0, idist0,
      onembed0,  ostride0, odist0,
      DEVICE_FFT_Z2Z, batch0, &workSize0));

    const int rank1 = 1;
    int n1[1] = { LOCAL_DIM/(VEC/P_DIM) };
    int inembed1[1] = { LOCAL_DIM };
    int onembed1[1] = { LOCAL_DIM };
    const int istride1 = VEC/P_DIM;
    const int ostride1 = VEC/P_DIM;
    const int idist1 = 1;
    const int odist1 = 1;
    const int batch1 = VEC/P_DIM;
    size_t workSize1 = 0;
    DEVICE_FFT_SAFE_CALL(DEVICE_FFT_MAKE_PLAN_MANY(
      *plan1, rank1, n1,
      inembed1,  istride1, idist1,
      onembed1,  ostride1, odist1,
      DEVICE_FFT_Z2Z, batch1, &workSize1));

    const int rank2 = 1;
    int n2[1] = { VEC };
    int inembed2[1] = { LOCAL_DIM*LOCAL_DIM };
    int onembed2[1] = { LOCAL_DIM*LOCAL_DIM };
    const int istride2 = LOCAL_DIM*B_DIM;
    const int ostride2 = LOCAL_DIM*B_DIM;
    const int idist2 = 1;
    const int odist2 = 1;
    const int batch2 = LOCAL_DIM*B_DIM;
    size_t workSize2 = 0;
    DEVICE_FFT_SAFE_CALL(DEVICE_FFT_MAKE_PLAN_MANY(
      *plan2, rank2, n2,
      inembed2,  istride2, idist2,
      onembed2,  ostride2, odist2,
      DEVICE_FFT_Z2Z, batch2, &workSize2));

    const int rank3 = 1;
    int n3[1] = { B_DIM*P_DIM };
    int inembed3[1] = { LOCAL_DIM*B_DIM*P_DIM };
    int onembed3[1] = { LOCAL_DIM*B_DIM*P_DIM };
    const int istride3 = LOCAL_DIM;
    const int ostride3 = LOCAL_DIM;
    const int idist3 = 1;
    const int odist3 = 1;
    const int batch3 = LOCAL_DIM;
    size_t workSize3 = 0;
    DEVICE_FFT_SAFE_CALL(DEVICE_FFT_MAKE_PLAN_MANY(
      *plan3, rank3, n3,
      inembed3,  istride3, idist3,
      onembed3,  ostride3, odist3,
      DEVICE_FFT_Z2Z, batch3, &workSize3));
}

#endif