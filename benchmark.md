This file contains instructions on how to setup and benchmark the code in this repository

# Heffte

In order to run the heffte baselines on NERSC you must first clone the [repo](https://github.com/icl-utk-edu/heffte) and do the following steps:

```
cd heffte
rm -rf build; mkdir build; cd build
module load PrgEnv-nvidia cmake cray-fftw cudatoolkit
export MPICH_GPU_SUPPORT_ENABLED=1
cmake \
-D CMAKE_CXX_COMPILER=CC \
-D MPI_CXX_COMPILER=CC \
-D CMAKE_BUILD_TYPE=Release \
-D BUILD_SHARED_LIBS=ON \
-D CMAKE_INSTALL_PREFIX=$CMAKE_DIR \
-D Heffte_ENABLE_AVX=ON \
-D Heffte_ENABLE_FFTW=ON \
-D FFTW_ROOT=$FFTW_ROOT \
-D Heffte_ENABLE_CUDA=ON \
-D CMAKE_CXX_FLAGS="-target-accel=nvidia80" \
-D CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES=$CRAY_MPICH_DIR/include \
..
make heffte_example_gpu
./examples/heffte_example_gpu
```
The code in this repositories `heffte/heffte_example_cuda.cpp` or `heffte/heffte_example_cuda3d.cpp` can be copied into the examples file in the heffte directory in order to run the 2D and 3D baselines respectively. See the `heffte/run.sh` and `heffte/run3d.sh` scripts for examples on how to run a parameter sweep.

# Our FFT

In `zmodel/main.cu` you can select different test programs to run.  For example if you want to benchmark the 2D FFT select `test_fft()`.  All of the important parameters are controlled by a set of Macros in `zmodel/utils.h`.  The following is a suggestive list on how you should set things
* __PRINT__TIMING__ This should always be uncommented during benchmarking as it prints the runtime of the total FFT
* __PRINT__DETAILED__TIMING__ This should only be uncommented if you want to see timings for each of the individual components of the FFT.  There is some overhead to running this so the total FFT time will be slower
* __PRINT__RESULTS__ This should be commented out during benchmarking.  It prints a subset of the FFT output and can be used for debug
* __PRINT__RESULTS__ This should be commented out during benchmakring.  It prints a subset of the twiddles and can be used for debug
* __PRINT__SANITY__ This just prints the N, B, and P for each run. Sanity check that setting N, B, P macros on commandline works.
* __ZMODEL__COMPUTE__ This should always be uncommented during benchmarking.  This enables the computational kernels.  Without this the data is permuted but no math is done, useful for debug.
* __GPU__AWARE__MPI__ This should only be uncommented if your machine has gpu aware MPI
* __GPU__SET__ This should only be uncommented if your machine has gpu aware MPI and you want it to be fast
* __USE__FFTDX__ This should only be uncommented if you have installed FFTDX (note this makes compile times slow)
* __ZMODEL__INCLUDE__INVERSE__ This should be commented out for benchmarking.  Inverse is the same as forward.
* __USE__HDF5__ This should be commented out for benchmarking, default data is initialized.
* N_DIM, B_DIM, P_DIM Problem size parameters that should be set from the commandline. See runscripts for examples.
* RUNS This should be set to the number of runs of the FFT you want to perform
* NUM_STREAMS This should be set to the number of streams to use with the FFT1 kernel, only used if FFTDX not installed.
* CUFFTDX_TARGET_SM This is needed for use the FFTDX.  For A100s on NERSC this should be set to 800.  For ECE number cluster T4s this should be set to 750.

The code can be run without (FFTDX)[https://developer.nvidia.com/cufftdx-downloads] but this should be installed if you want good performance.  Replace the default path in the Makefile with wherever you have installed it.

The scripts `zmodel/run.sh` and `zmodel/run3d.sh` are written to perform parameter sweeps of the 2D FFT and the 3D FFT respectively on the CMU ECE number cluster.

To run the code more generally you can just set the program you want to run in `zmodel/main.cu` and do `make run`

# Stencils

TODO
