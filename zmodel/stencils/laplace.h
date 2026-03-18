#ifndef __ZMODEL__LAPLACE__
#define __ZMODEL__LAPLACE__

#include <mpi.h>
#include "../utils.h"

struct LaplaceBuffers {
    Complex *host_send_above;
    Complex *host_send_below;
    Complex *host_send_left;
    Complex *host_send_right;
    Complex *host_recv_above;
    Complex *host_recv_below;
    Complex *host_recv_left;
    Complex *host_recv_right;
    Complex *device_send_above;
    Complex *device_send_below;
    Complex *device_send_left;
    Complex *device_send_right;    
    Complex *device_recv_above;
    Complex *device_recv_below;
    Complex *device_recv_left;
    Complex *device_recv_right;
};

void test_laplace();

void init_laplace_buffers(LaplaceBuffers& laplace_buffers);

void destroy_laplace_buffers(LaplaceBuffers& laplace_buffers);

__global__ void laplace(Complex *device_in, Complex *device_out, LaplaceBuffers laplace_buffers, int rid, int cid);

__global__ void pack_laplace(Complex *device_in, LaplaceBuffers laplace_buffers);          

void communicate_laplace(LaplaceBuffers& laplace_buffers, MPI_Comm& row_comm, MPI_Comm& col_comm, int rid, int cid);

#endif // __ZMODEL__LAPLACE__ 