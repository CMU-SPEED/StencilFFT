#ifndef __ZMODEL__FFT__HOST__PLANS__
#define __ZMODEL__FFT__HOST__PLANS__

#include "../utils.h"

struct FftHostPlans {
    cufftHandle plan0;
    cufftHandle plan2;
    Vector<cufftHandle> plan1(NUM_STREAMS);
    Vector<cufftHandle> plan3(NUM_STREAMS); 
    Vector<cudaStream_t> streams(NUM_STREAMS);
};

void init_plans(cufftHandle *plan0, Vector<cufftHandle> &plan1,
                cufftHandle *plan2, Vector<cufftHandle> &plan3, 
                Vector<cudaStream_t> &streams) {
    DEVICE_FFT_SAFE_CALL(DEVICE_FFT_CREATE(plan0));    
    DEVICE_FFT_SAFE_CALL(DEVICE_FFT_CREATE(plan2));
    for (int i = 0; i < NUM_STREAMS; i++) {
        DEVICE_FFT_SAFE_CALL(DEVICE_FFT_CREATE(&plan1[i]));
        DEVICE_FFT_SAFE_CALL(DEVICE_FFT_CREATE(&plan3[i]));
        DEVICE_RT_SAFE_CALL(DEVICE_STREAM_CREATE(&streams[i]));
    }

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
    for (int i = 0; i < NUM_STREAMS; i++) {
        DEVICE_FFT_SAFE_CALL(DEVICE_FFT_MAKE_PLAN_MANY(
          plan1[i], rank1, n1,
          inembed1,  istride1, idist1,
          onembed1,  ostride1, odist1,
          DEVICE_FFT_Z2Z, batch1, &workSize1));
        DEVICE_FFT_SAFE_CALL(DEVICE_FFT_STREAM_SET(plan1[i], streams[i]));
    }

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
    for (int i = 0; i < NUM_STREAMS; i++) {
        DEVICE_FFT_SAFE_CALL(DEVICE_FFT_MAKE_PLAN_MANY(
            plan3[i], rank3, n3,
            inembed3,  istride3, idist3,
            onembed3,  ostride3, odist3,
            DEVICE_FFT_Z2Z, batch3, &workSize3));
        DEVICE_FFT_SAFE_CALL(DEVICE_FFT_STREAM_SET(plan3[i], streams[i]));
    }
}

#endif // __ZMODEL__FFT__HOST__PLANS__