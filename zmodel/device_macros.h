#ifndef __ZMODEL__DEVICE__MACROS__HEADER__
#define __ZMODEL__DEVICE__MACROS__HEADER__

// Adapted from FFTX (https://github.com/spiralgen/fftx)
// Copyright (c) Carnegie Mellon University. Licensed under BSD 2-Clause License.

#include <iostream>

#ifdef __ZMODEL__HIP__
#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>
#include <rocfft/rocfft.h>

#define DEVICE_SUCCESS hipSuccess
#define DEVICE_EVENT_T hipEvent_t
#define DEVICE_EVENT_CREATE hipEventCreate
#define DEVICE_SET hipSetDevice
#define DEVICE_MALLOC hipMalloc
#define DEVICE_EVENT_RECORD hipEventRecord
#define DEVICE_EVENT_ELAPSED_TIME hipEventElapsedTime
#define DEVICE_SYNCHRONIZE hipDeviceSynchronize
#define DEVICE_EVENT_SYNCHRONIZE hipEventSynchronize
#define DEVICE_FREE hipFree
#define DEVICE_MEM_COPY hipMemcpy
#define DEVICE_MEM_SET hipMemset
#define MEM_COPY_DEVICE_TO_DEVICE hipMemcpyDeviceToDevice
#define MEM_COPY_DEVICE_TO_HOST hipMemcpyDeviceToHost
#define MEM_COPY_HOST_TO_DEVICE hipMemcpyHostToDevice
#define DEVICE_ERROR_T hipError_t
#define DEVICE_GET_LAST_ERROR hipGetLastError
#define DEVICE_GET_ERROR_STRING hipGetErrorString
#define DEVICE_PTR hipDeviceptr_t
#define DEVICE_FFT_TYPE hipfftType
#define DEVICE_FFT_RESULT hipfftResult
#define DEVICE_FFT_HANDLE hipfftHandle
#define DEVICE_FFT_CREATE hipfftCreate
#define DEVICE_FFT_MAKE_PLAN_3D hipfftMakePlan3d
#define DEVICE_FFT_PLAN3D hipfftPlan3d
#define DEVICE_FFT_PLAN2D hipfftPlan2d
#define DEVICE_FFT_PLAN_MANY hipfftPlanMany
#define DEVICE_FFT_EXECZ2Z hipfftExecZ2Z
#define DEVICE_FFT_EXECD2Z hipfftExecD2Z
#define DEVICE_FFT_EXECZ2D hipfftExecZ2D
#define DEVICE_FFT_DESTROY hipfftDestroy
#define DEVICE_FFT_DOUBLEREAL hipfftDoubleReal
#define DEVICE_FFT_DOUBLECOMPLEX hipfftDoubleComplex
#define DEVICE_FFT_Z2Z HIPFFT_Z2Z
#define DEVICE_FFT_D2Z HIPFFT_D2Z
#define DEVICE_FFT_Z2D HIPFFT_Z2D
#define DEVICE_FFT_SUCCESS HIPFFT_SUCCESS
#define DEVICE_FFT_FORWARD HIPFFT_FORWARD
#define DEVICE_FFT_INVERSE HIPFFT_BACKWARD

#elif defined(__ZMODEL__CUDA__)

#include <cufft.h>
#include "cuda_runtime.h"

#define DEVICE_SUCCESS cudaSuccess
#define DEVICE_EVENT_T cudaEvent_t
#define DEVICE_EVENT_CREATE cudaEventCreate
#define DEVICE_SET cudaSetDevice
#define DEVICE_MALLOC cudaMalloc
#define DEVICE_HOST_ALLOC cudaHostAlloc
#define DEVICE_HOST_ALLOC_DEFAULT cudaHostAllocDefault
#define DEVICE_EVENT_RECORD cudaEventRecord
#define DEVICE_EVENT_ELAPSED_TIME cudaEventElapsedTime
#define DEVICE_SYNCHRONIZE cudaDeviceSynchronize
#define DEVICE_EVENT_SYNCHRONIZE cudaEventSynchronize
#define DEVICE_FREE cudaFree
#define DEVICE_FREE_HOST cudaFreeHost
#define DEVICE_MEM_COPY cudaMemcpy
#define DEVICE_MEM_SET cudaMemset
#define MEM_COPY_DEVICE_TO_DEVICE cudaMemcpyDeviceToDevice
#define MEM_COPY_DEVICE_TO_HOST cudaMemcpyDeviceToHost
#define MEM_COPY_HOST_TO_DEVICE cudaMemcpyHostToDevice
#define DEVICE_ERROR_T cudaError_t
#define DEVICE_GET_LAST_ERROR cudaGetLastError
#define DEVICE_GET_ERROR_STRING cudaGetErrorString
// #define DEVICE_PTR CUdeviceptr
#define DEVICE_PTR void **
#define DEVICE_FFT_TYPE cufftType
#define DEVICE_FFT_RESULT cufftResult
#define DEVICE_FFT_HANDLE cufftHandle
#define DEVICE_FFT_CREATE cufftCreate
#define DEVICE_FFT_MAKE_PLAN_3D cufftMakePlan3d
#define DEVICE_FFT_PLAN3D cufftPlan3d
#define DEVICE_FFT_PLAN2D cufftPlan2d
#define DEVICE_FFT_PLAN_MANY cufftPlanMany
#define DEVICE_FFT_MAKE_PLAN_MANY cufftMakePlanMany
#define DEVICE_FFT_EXECZ2Z cufftExecZ2Z
#define DEVICE_FFT_EXECD2Z cufftExecD2Z
#define DEVICE_FFT_EXECZ2D cufftExecZ2D
#define DEVICE_FFT_DESTROY cufftDestroy
#define DEVICE_FFT_DOUBLEREAL cufftDoubleReal
#define DEVICE_FFT_DOUBLECOMPLEX cufftDoubleComplex
#define DEVICE_FFT_DOUBLECOMPLEX_CONSTRUCTOR make_cuDoubleComplex
#define DEVICE_FFT_DOUBLECOMPLEX_MUL cuCmul
#define DEVICE_FFT_Z2Z CUFFT_Z2Z
#define DEVICE_FFT_D2Z CUFFT_D2Z
#define DEVICE_FFT_Z2D CUFFT_Z2D
#define DEVICE_FFT_SUCCESS CUFFT_SUCCESS
#define DEVICE_FFT_FORWARD CUFFT_FORWARD
#define DEVICE_FFT_INVERSE CUFFT_INVERSE
#define DEVICE_STREAM_CREATE cudaStreamCreate
#define DEVICE_FFT_STREAM_SET cufftSetStream
#define DEVICE_STREAM_DESTROY cudaStreamDestroy
#define DEVICE_COUNT cudaGetDeviceCount
#define DEVICE_ENABLE_PA cudaDeviceEnablePeerAccess

#define DEVICE_RT_SAFE_CALL(x) do {                                    \
  cudaError_t err = (x);                                               \
  if (err != cudaSuccess) {                                            \
    std::cerr << "\nerror: " #x " failed with "                        \
              << cudaGetErrorString(err) << '\n';                      \
    std::exit(1);                                                      \
  }                                                                    \
} while(0)

inline const char* cufftGetErrorString(cufftResult status) {
    switch (status) {
        case CUFFT_SUCCESS:                      return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN:                 return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED:                 return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE:                 return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE:                return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR:               return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED:                  return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED:                 return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE:                 return "CUFFT_INVALID_SIZE";
        case CUFFT_UNALIGNED_DATA:               return "CUFFT_UNALIGNED_DATA";
        // case CUFFT_INCOMPLETE_PARAMETER_LIST:    return "CUFFT_INCOMPLETE_PARAMETER_LIST";
        case CUFFT_INVALID_DEVICE:               return "CUFFT_INVALID_DEVICE";
        // case CUFFT_PARSE_ERROR:                  return "CUFFT_PARSE_ERROR";
        case CUFFT_NO_WORKSPACE:                 return "CUFFT_NO_WORKSPACE";
        case CUFFT_NOT_IMPLEMENTED:              return "CUFFT_NOT_IMPLEMENTED";
        // case CUFFT_LICENSE_ERROR:                return "CUFFT_LICENSE_ERROR";
        case CUFFT_NOT_SUPPORTED:                return "CUFFT_NOT_SUPPORTED";
        default:                                 return "CUFFT_UNKNOWN_ERROR";
    }
}

#define DEVICE_FFT_SAFE_CALL(x) do {                                  \
    cufftResult _rc = (x);                                            \
    if (_rc != CUFFT_SUCCESS) {                                       \
        std::cerr << "\nerror: " #x " failed with "                   \
                  << cufftGetErrorString(_rc)                         \
                  << " (code " << static_cast<int>(_rc) << ")\n";     \
        std::exit(1);                                                 \
    }                                                                 \
} while(0)

#else
// neither CUDA nor HIP
#define DEVICE_SUCCESS 0
#endif

#endif