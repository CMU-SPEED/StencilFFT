This repository contains an implementation of the interleaved block cyclic format.

# How the data layout works

We start with a block cyclic distribution and then pack the ith element of all the blocks in a row together into vectors.  These vectors can be seen in Step 1 and is where we apply our stencils.  

The 2D FFT is implemented as 2 independent 1D FFTs which we will call FFT1 and FFT2 respectively.  FFT1 is done in the rows of the original matrix while FFT2 is done in the columns of the original matrix.  FFT1 and FFT2 are further decomposed into two 1D FFTs each with twiddle factors applied in between in accordance with Cooley-Tukey.  The decomposition of FFT1 in the first row and FFT2 in the first column can be seen in the top right corner.  The row and column vectors are reshaped as matricies in accordance with the decomposition.  FFT1 is decomposed into FFT1a which is done in the columns of the new matrix and FFT1b which is done in the rows of the new matrix.  Note that the vectors we have packed in Step 1 are exactly the data needed to execute FFT1a.

From here we must split the vectors in P_DIM components and pack them for the All-to-All in the rows.  After the All-to-All in the rows the data is unpacked and we apply FFT1B.  The data for FFT1B is interleaved and is executed as batch FFT.  In reality we are doing a batch of batch FFTs which is why we use CuFFTdx.  

FFT2a can be executed on this same data layout before the All-to-All in the columns is needed to execute FFT2b.  Twiddles are applied between FFTXa and FFTXb.  Note that Cooley-Tukey requries a transpose between FFTXa and FFTXb which we do not implement.  This means that there is an additional layer of permutation applied to the data that the diagrams do not capture.  This makes the application of position based operators in the fourier space complicated and is why I am currently working on a symmetric version of the current implementaiton.

![Alt text for the image](images/step1.png "Optional Title")
![Alt text for the image](images/step2.png "Optional Title")
![Alt text for the image](images/step3.png "Optional Title")

# How to use

This codebase requires MPI, CUDA, CuFFT, and CuFFTdx.

Code for the 2D fft impelmentation can be found in `./zmodel/transforms/fft.cu`.
Code for the 9-point laplacian can be found in `./zmodel/stencils/laplace.cu`.
These can be run by modifying `./zmodel/main.cu` to run the appropriate test and using `make zmodel`.  It is probably easier to look at `./zmodel/stencils/dx.cu` before looking at the laplacian code.

The default make command creates 4 processes but runs the code on a single host processor.  It is set up to be run on the ECE number clusters.  `./zmodel/utils.h` contains default parameters and allow you to change the problem size or the number of processors used.  If you want to change the number of processors you will also have to update the Makefile.  N_DIM is the dimension of the total problem size which means there are N_DIM * N_DIM elements what are split across P_DIM * P_DIM processors and blocks are of size B_DIM * B_DIM.

The ECE number clusters do not support GPU aware MPI so it is disabled by default.

You will likely need to install cuFFTdx (https://developer.nvidia.com/cufftdx-downloads) even though it is disabled by default.  You will need to replace the `FFTDX_INCLUDE` path.  Note that using this makes the compile times slow.

This code and the data layout are still in development.  The 3D implementation is incomplete.  The datalayout is designed for nice problem shapes and sizes.  We typically test with powers of two problem size up to two steps below a blocked distribution.  The code has been tested with 4 and 16 GPUs.

Please direct all questions to cjstange@andrew.cmu.edu