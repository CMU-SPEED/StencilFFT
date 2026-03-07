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

#endif // __ZMODEL__LAPLACE__ 