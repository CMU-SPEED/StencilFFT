#include "fft_host_plans.h"

void init_plans(FftHostPlans& host_plans) {
    DEVICE_FFT_SAFE_CALL(DEVICE_FFT_CREATE(&host_plans.plan0));    
    DEVICE_FFT_SAFE_CALL(DEVICE_FFT_CREATE(&host_plans.plan2));
    #ifndef __USE__FFTDX__
        host_plans.plan1.resize(NUM_STREAMS);
        host_plans.plan3.resize(NUM_STREAMS);
        host_plans.streams.resize(NUM_STREAMS);
        for (int i = 0; i < NUM_STREAMS; i++) {
            DEVICE_FFT_SAFE_CALL(DEVICE_FFT_CREATE(&host_plans.plan1[i]));
            DEVICE_FFT_SAFE_CALL(DEVICE_FFT_CREATE(&host_plans.plan3[i]));
            DEVICE_RT_SAFE_CALL(DEVICE_STREAM_CREATE(&host_plans.streams[i]));
        }
    #endif

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
      host_plans.plan0, rank0, n0,
      inembed0,  istride0, idist0,
      onembed0,  ostride0, odist0,
      DEVICE_FFT_Z2Z, batch0, &workSize0));

    #ifndef __USE__FFTDX__
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
            host_plans.plan1[i], rank1, n1,
            inembed1,  istride1, idist1,
            onembed1,  ostride1, odist1,
            DEVICE_FFT_Z2Z, batch1, &workSize1));
            DEVICE_FFT_SAFE_CALL(DEVICE_FFT_STREAM_SET(host_plans.plan1[i], host_plans.streams[i]));
        }
    #endif

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
      host_plans.plan2, rank2, n2,
      inembed2,  istride2, idist2,
      onembed2,  ostride2, odist2,
      DEVICE_FFT_Z2Z, batch2, &workSize2));

    #ifndef __USE__FFTDX__
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
                host_plans.plan3[i], rank3, n3,
                inembed3,  istride3, idist3,
                onembed3,  ostride3, odist3,
                DEVICE_FFT_Z2Z, batch3, &workSize3));
            DEVICE_FFT_SAFE_CALL(DEVICE_FFT_STREAM_SET(host_plans.plan3[i], host_plans.streams[i]));
        }
    #endif
}

void destroy_plans(FftHostPlans& host_plans) {
    DEVICE_FFT_SAFE_CALL(DEVICE_FFT_DESTROY(host_plans.plan0));
    DEVICE_FFT_SAFE_CALL(DEVICE_FFT_DESTROY(host_plans.plan2));
    #ifndef __USE__FFTDX__
        for (int i = 0; i < NUM_STREAMS; i++) {
            DEVICE_FFT_SAFE_CALL(DEVICE_FFT_DESTROY(host_plans.plan1[i]));
            DEVICE_FFT_SAFE_CALL(DEVICE_FFT_DESTROY(host_plans.plan3[i]));
            DEVICE_RT_SAFE_CALL(DEVICE_STREAM_DESTROY(host_plans.streams[i]));
        }
    #endif
}