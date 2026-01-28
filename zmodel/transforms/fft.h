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
    Complex *host_inverse_twiddles0;
    Complex *host_inverse_twiddles1;
    Complex *device_buf0;
    Complex *device_buf1;
    Complex *device_forward_twiddles0; 
    Complex *device_forward_twiddles1; 
    Complex *device_inverse_twiddles0; 
    Complex *device_inverse_twiddles1;
};

// These are needed here because of init_buffers
void init_forward_twiddles0(Complex* in, int cid);

void init_forward_twiddles1(Complex* in, int rid);

void init_inverse_twiddles0(Complex* in, int cid);

void init_inverse_twiddles1(Complex* in, int rid);

template<bool do_compute>
void forward_fft(FftBuffers& buffers, MPI_Comm *row_comm, MPI_Comm *col_comm,
                 int id, FftHostPlans& host_plans, FftdxFft1Ctx& ctx1, FftdxFft3Ctx& ctx3);

template<bool do_compute>
void inverse_fft(FftBuffers& buffers, MPI_Comm *row_comm, MPI_Comm *col_comm,
                 int id, Complex scale, FftHostPlans& host_plans);

void test_fft();

template<bool include_inverse>
void init_buffers(FftBuffers& buffers, int rid, int cid) {
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&buffers.host_buf0, LOCAL_COMPLEX_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&buffers.host_buf1, LOCAL_COMPLEX_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&buffers.host_forward_twiddles0, LOCAL_COMPLEX_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&buffers.host_forward_twiddles1, LOCAL_COMPLEX_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
    if constexpr (include_inverse) {
        DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&buffers.host_inverse_twiddles0, LOCAL_COMPLEX_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
        DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&buffers.host_inverse_twiddles1, LOCAL_COMPLEX_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
    }

    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&buffers.device_buf0, LOCAL_COMPLEX_BYTES));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&buffers.device_buf1, LOCAL_COMPLEX_BYTES));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&buffers.device_forward_twiddles0, LOCAL_COMPLEX_BYTES));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&buffers.device_forward_twiddles1, LOCAL_COMPLEX_BYTES));
    if constexpr (include_inverse) {
        DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&buffers.device_inverse_twiddles0, LOCAL_COMPLEX_BYTES));
        DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&buffers.device_inverse_twiddles1, LOCAL_COMPLEX_BYTES));
    }

    init_host(rid, cid, buffers.host_buf0);
    init_forward_twiddles0(buffers.host_forward_twiddles0, cid);
    init_forward_twiddles1(buffers.host_forward_twiddles1, rid);
    if constexpr (include_inverse) {
        init_inverse_twiddles0(buffers.host_inverse_twiddles0, cid);
        init_inverse_twiddles1(buffers.host_inverse_twiddles1, rid);
    }

    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_buf0, buffers.host_buf0, LOCAL_COMPLEX_BYTES, MEM_COPY_HOST_TO_DEVICE));
    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_buf1, buffers.host_buf1, LOCAL_COMPLEX_BYTES, MEM_COPY_HOST_TO_DEVICE));
    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_forward_twiddles0, buffers.host_forward_twiddles0, LOCAL_COMPLEX_BYTES, MEM_COPY_HOST_TO_DEVICE));
    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_forward_twiddles1, buffers.host_forward_twiddles1, LOCAL_COMPLEX_BYTES, MEM_COPY_HOST_TO_DEVICE));
    if constexpr (include_inverse) {
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_inverse_twiddles0, buffers.host_inverse_twiddles0, LOCAL_COMPLEX_BYTES, MEM_COPY_HOST_TO_DEVICE));
        DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(buffers.device_inverse_twiddles1, buffers.host_inverse_twiddles1, LOCAL_COMPLEX_BYTES, MEM_COPY_HOST_TO_DEVICE));
    }
}

template<bool include_inverse>
void destroy_buffers(FftBuffers& buffers) {
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(buffers.host_buf0));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(buffers.host_buf1));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(buffers.host_forward_twiddles0));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(buffers.host_forward_twiddles1));
    if constexpr (include_inverse) {
        DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(buffers.host_inverse_twiddles0));
        DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(buffers.host_inverse_twiddles1));
    }

    DEVICE_RT_SAFE_CALL(DEVICE_FREE(buffers.device_buf0));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(buffers.device_buf1));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(buffers.device_forward_twiddles0));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(buffers.device_forward_twiddles1));
    if constexpr (include_inverse) {
        DEVICE_RT_SAFE_CALL(DEVICE_FREE(buffers.device_inverse_twiddles0));
        DEVICE_RT_SAFE_CALL(DEVICE_FREE(buffers.device_inverse_twiddles1));
    }
}



#endif // __ZMODEL__FFT__