#ifndef __ZMODEL__FFT__
#define __ZMODEL__FFT__

#include <mpi.h>
#include "../utils.h"
#include "fft_device_plans.h"

template<bool do_compute>
void forward_fft(Complex *host_buf0, Complex *host_buf1, Complex *device_buf0, Complex *device_buf1,
                 Complex *device_twiddles0, Complex *device_twiddles1, MPI_Comm *row_comm, MPI_Comm *col_comm,
                 cufftHandle *plan0,  Vector<cufftHandle> &plan1, cufftHandle *plan2,  Vector<cufftHandle> &plan3,
                 int id, Vector<cudaStream_t> &streams, FftdxFft1Ctx& ctx1);

template<bool do_compute>
void inverse_fft(Complex *host_buf0, Complex *host_buf1, Complex *device_buf0, Complex *device_buf1,
                 Complex *device_twiddles0, Complex *device_twiddles1, MPI_Comm *row_comm, MPI_Comm *col_comm,
                 cufftHandle *plan0, Vector<cufftHandle> &plan1, cufftHandle *plan2, Vector<cufftHandle> &plan3,
                 int id, Complex scale, Vector<cudaStream_t> &streams);

void test_fft();

#endif // __ZMODEL__FFT__