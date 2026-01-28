#ifndef __ZMODEL__FFT__HOST__PLANS__
#define __ZMODEL__FFT__HOST__PLANS__

#include "../utils.h"

struct FftHostPlans {
    cufftHandle plan0;
    cufftHandle plan2;
    Vector<cufftHandle> plan1;
    Vector<cufftHandle> plan3;
    Vector<cudaStream_t> streams;
};

void init_plans(FftHostPlans& host_plans);
void destroy_plans(FftHostPlans& host_plans);

#endif // __ZMODEL__FFT__HOST__PLANS__