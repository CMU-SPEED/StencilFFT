#ifndef __ZMODEL__FFT__
#define __ZMODEL__FFT__

#include <mpi.h>
#include "../utils.h"
#include "fft_device_plans.h"
#include "fft_host_plans.h"

struct FftBuffers {
    Complex *host_buf0;
    Complex *host_buf1;
    Complex *host_forward_twiddles0; 
    Complex *host_forward_twiddles1; 
    Complex *host_forward_twiddles2;
    Complex *host_inverse_twiddles0;
    Complex *host_inverse_twiddles1;
    Complex *host_inverse_twiddles2;
    Complex *device_buf0;
    Complex *device_buf1;
    Complex *device_forward_twiddles0; 
    Complex *device_forward_twiddles1; 
    Complex *device_forward_twiddles2;
    Complex *device_inverse_twiddles0; 
    Complex *device_inverse_twiddles1;
    Complex *device_inverse_twiddles2;
};

template<bool include_inverse>
void init_buffers(FftBuffers& buffers, int rid, int cid);

template<bool include_inverse>
void destroy_buffers(FftBuffers& buffers);

template<bool do_compute>
void forward_fft(FftBuffers& buffers, MPI_Comm& row_comm, MPI_Comm& col_comm,
                 int id, FftHostPlans& host_plans, FftdxFft1Ctx& ctx1);

template<bool do_compute>
void inverse_fft(FftBuffers& buffers, MPI_Comm& row_comm, MPI_Comm& col_comm,
                 int id, Complex scale, FftHostPlans& host_plans);

void test_fft();

template<bool include_inverse>
void init_buffers_3d(FftBuffers& buffers, int rid, int cid, int did);

template<bool include_inverse>
void destroy_buffers_3d(FftBuffers& buffers);

template<bool do_compute>
void forward_fft_3d(FftBuffers& buffers, MPI_Comm& row_comm, MPI_Comm& col_comm, MPI_Comm& dep_comm,
                    int id, FftHostPlans& host_plans, FftdxFft1Ctx& ctx1);

template<bool do_compute>
void inverse_fft_3d(FftBuffers& buffers, MPI_Comm& row_comm, MPI_Comm& col_comm, MPI_Comm& dep_comm,
                    int id, Complex scale, FftHostPlans& host_plans);

void test_fft_3d();

#endif // __ZMODEL__FFT__