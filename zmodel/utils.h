#include <complex>
#include <cufft.h>
#include <cuda_runtime.h>

// FIXME: need to revist the use of macros --> we want to be able to pass things in on commandline
#define N (64)
#define b (4)
#define p (2) // Number of processors in the rows or columns. ex: p = 2 --> 4 total processors
#define local (N/p)

// This is the size of the vector the stencil will be batched with
#define vec (local/b)
// The total number of vectors on the local processor
#define vec_total ((local*local)/vec)
 
#define vec_row (local) // Number of vectors in a col
#define vec_col (b)     // Number of vectors in a row

// using Complex = std::complex<double>;
using Complex = cufftDoubleComplex;

// FIXME: this need to be revisted --> I think cid and rid may be swapped
void init_host(int rid, int cid, int local_size, Complex *host_in) {
    int row_offset = rid * b;       // offset based on which processor in the row
    int col_offset = cid * (N*b);   // offset based on which processor in the col
    int offset = row_offset + col_offset;

    Complex *init = (Complex*)malloc(local_size);
    
    for (int i = 0; i < local/b; i++) {
        for (int ii = 0; ii < b; ii++) {
            for (int j = 0; j < local/b; j++) {
                for (int jj = 0; jj < b; jj++) {
                    int array_index = (i * b * (local/b) * b) + (ii * (local/b) * b) + (j * b) + jj;
                    double real = (i * N * b * p) + (ii * N) + (j * b * p) + jj + offset;
                    init[array_index] = {real, 0.0};
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


void init_plans(cufftHandle *plan0, cufftHandle *plan1, cufftHandle *plan2, cufftHandle *plan3) {
    cufftCreate(plan0); // FIXME: need to wrap this in a check
    cufftCreate(plan1);
    cufftCreate(plan2);
    cufftCreate(plan3);

    const int rank0 = 1;
    int n0[1] = { vec };
    int inembed0[1] = { vec };
    int onembed0[1] = { vec };
    const int istride0 = 1;
    const int ostride0 = 1;
    const int idist0 = vec;
    const int odist0 = vec;
    const int batch0 = vec_total;
    size_t workSize0 = 0;
    cufftMakePlanMany(
      *plan0, rank0, n0,
      inembed0,  istride0, idist0,
      onembed0,  ostride0, odist0,
      CUFFT_Z2Z, batch0, &workSize0);

    const int rank1 = 1;
    int n1[1] = { local/(vec/p) };
    int inembed1[1] = { local };
    int onembed1[1] = { local };
    const int istride1 = vec/p;
    const int ostride1 = vec/p;
    const int idist1 = 1;
    const int odist1 = 1;
    const int batch1 = vec/p;
    size_t workSize1 = 0;
    cufftMakePlanMany(
      *plan1, rank1, n1,
      inembed1,  istride1, idist1,
      onembed1,  ostride1, odist1,
      CUFFT_Z2Z, batch1, &workSize1);

    const int rank2 = 1;
    int n2[1] = { vec };
    int inembed2[1] = { local*local };
    int onembed2[1] = { local*local };
    const int istride2 = local*b;
    const int ostride2 = local*b;
    const int idist2 = 1;
    const int odist2 = 1;
    const int batch2 = local*b;
    size_t workSize2 = 0;
    cufftMakePlanMany(
      *plan2, rank2, n2,
      inembed2,  istride2, idist2,
      onembed2,  ostride2, odist2,
      CUFFT_Z2Z, batch2, &workSize2);

    const int rank3 = 1;
    int n3[1] = { b*p };
    int inembed3[1] = { local*b*p };
    int onembed3[1] = { local*b*p };
    const int istride3 = local;
    const int ostride3 = local;
    const int idist3 = 1;
    const int odist3 = 1;
    const int batch3 = local;
    size_t workSize3 = 0;
    cufftMakePlanMany(
      *plan3, rank3, n3,
      inembed3,  istride3, idist3,
      onembed3,  ostride3, odist3,
      CUFFT_Z2Z, batch3, &workSize3);
}