#include "riesz.h"
#include "fft.h"

// FIXME: Does this work due to the transposes?
// This is a pretty unholy way of computing the indicies but it also works
void precompute_riesz_index(Scalar *host_x_index, Scalar *host_y_index,
                            MPI_Comm *row_comm, MPI_Comm *col_comm, int rid, int cid, int id) {
    Complex *host_buf0, *host_buf1;
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&host_buf0, LOCAL_COMPLEX_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&host_buf1, LOCAL_COMPLEX_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
    init_host(rid, cid, host_buf0);

    Complex *device_buf0, *device_buf1;
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&device_buf0, LOCAL_COMPLEX_BYTES));
    DEVICE_RT_SAFE_CALL(DEVICE_MALLOC((void**)&device_buf1, LOCAL_COMPLEX_BYTES));
    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(device_buf0, host_buf0, LOCAL_COMPLEX_BYTES, MEM_COPY_HOST_TO_DEVICE));
    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(device_buf1, host_buf1, LOCAL_COMPLEX_BYTES, MEM_COPY_HOST_TO_DEVICE));

    // FIXME: need to use updated FFT function 
    // forward_fft<false>(host_buf0, host_buf1, device_buf0, device_buf1,
    //                     nullptr, nullptr, row_comm, col_comm, nullptr, nullptr, nullptr, nullptr, id);

    DEVICE_RT_SAFE_CALL(DEVICE_MEM_COPY(host_buf1, device_buf1, LOCAL_COMPLEX_BYTES, MEM_COPY_DEVICE_TO_HOST));

    for (int i = 0; i < LOCAL_DIM; i++) {
        for (int j = 0; j < LOCAL_DIM; j++) {
            host_x_index[i * LOCAL_DIM + j] = (Scalar)((long)host_buf1[i * LOCAL_DIM + j].x % (long)N_DIM);
            host_y_index[i * LOCAL_DIM + j] = (Scalar)host_buf1[i * LOCAL_DIM + j].x / (Scalar)N_DIM;
        }
    }

    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(host_buf0));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(host_buf1));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(device_buf0));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE(device_buf1));
}

// FIXME: revisit this
void precompute_riesz_weight(Scalar *host_x_index, Scalar *host_y_index, 
                             Scalar *host_M1, Scalar *host_M2, Scalar Lx, Scalar Ly) {
    Scalar half = N_DIM / 2;
    for (int i = 0; i < LOCAL_DIM; i++) {
        for (int j = 0; j < LOCAL_DIM; j++) {
            Scalar x_index = host_x_index[i * LOCAL_DIM + j];
            Scalar y_index = host_y_index[i * LOCAL_DIM + j];
            Scalar nx = (x_index <= half) ? x_index : x_index - N_DIM;
            Scalar ny = (y_index <= half) ? y_index : y_index - N_DIM;

            if (nx == 0.0 && ny == 0.0) {
                host_M1[i * LOCAL_DIM + j] = 0.0;
                host_M2[i * LOCAL_DIM + j] = 0.0;
            } else {
                Scalar kx = (2.0*M_PI/Lx) * nx;
                Scalar ky = (2.0*M_PI/Ly) * ny;
                Scalar len = std::hypot(kx, ky);
                host_M1[i * LOCAL_DIM + j] = kx / len;
                host_M2[i * LOCAL_DIM + j] = ky / len;
            }
        }
    }
}

// Assume it is called with LOCAL_DIM, LOCAL_DIM
__global__ void apply_riesz_weights(Complex *device_riesz, Complex *device_C1, Complex *device_C2,
                                    Scalar *device_M1, Scalar *device_M2) {
    int index = blockIdx.x * LOCAL_DIM + threadIdx.x;
    device_riesz[index].x = device_M1[index] * device_C1[index].y + device_M2[index] * device_C2[index].y;
    device_riesz[index].y = -device_M1[index] * device_C1[index].x - device_M2[index] * device_C2[index].x;
}

// Assume it is called with LOCAL_DIM, LOCAL_DIM
__global__ void pack_c1_c2(Scalar *device_w0, Scalar *device_w1, Complex *device_C1, Complex *device_C2) {
    int index = blockIdx.x * LOCAL_DIM + threadIdx.x; 
    device_C1[index].x = device_w0[index];
    device_C1[index].y = 0.0;
    device_C2[index].x = device_w1[index];
    device_C2[index].y = 0.0;
}

// void riesz(Scalar *device_w0, Scalar *device_w1, Scalar *device_riesz, Scalar *device_M1, Scalar, *device_M2) {

// }

void riesz() {

}

void test_riesz() {
    MPI_Init(NULL, NULL);

    int P, id;
    P = P_DIM * P_DIM;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    #ifdef __PRINT__SANITY__
        if (id == 0) std::cout << "N_DIM: " << N_DIM << ", B_DIM: " << B_DIM << " P_DIM: " << P_DIM << std::endl;
    #endif

    MPI_Comm row_comm, col_comm;

    int rid = id / P_DIM;
    MPI_Comm_split(MPI_COMM_WORLD, rid, id, &row_comm);

    int cid = id % P_DIM;
    MPI_Comm_split(MPI_COMM_WORLD, cid, id, &col_comm);

    // FIXME: Find reasonable values for this
    Scalar Lx = 1;
    Scalar Ly = 1;

    Scalar *host_x_index, *host_y_index, *host_M1, *host_M2;
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&host_x_index, LOCAL_COMPLEX_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&host_y_index, LOCAL_COMPLEX_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&host_M1, LOCAL_COMPLEX_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
    DEVICE_RT_SAFE_CALL(DEVICE_HOST_ALLOC((void**)&host_M2, LOCAL_COMPLEX_BYTES, DEVICE_HOST_ALLOC_DEFAULT));
    precompute_riesz_index(host_x_index, host_y_index, &row_comm, &col_comm, rid, cid, id);
    precompute_riesz_weight(host_x_index, host_y_index, host_M1, host_M2, Lx, Ly);

    // TODO

    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(host_x_index));
    DEVICE_RT_SAFE_CALL(DEVICE_FREE_HOST(host_y_index));

    MPI_Finalize();
}