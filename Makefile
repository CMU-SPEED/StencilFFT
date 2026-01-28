# Standard paths on Linux
LIB_PATH = /usr
INCLUDE_PATH = LIB_PATH
FLAGS= -O3 -fopenmp

LIB = -L$(LIB_PATH)/lib -lfftw3_omp -lfftw3
INCLUDE = -I$(LIB_PATH)/include

CUDA_HOME=/usr/local/cuda

FFTDX_INCLUDE=-I/afs/andrew.cmu.edu/usr12/cjstange/private/fftdx/nvidia-mathdx-25.12.1-cuda13/nvidia/mathdx/25.12/include

ZMODEL_FILES = zmodel/main.cu zmodel/transforms/fft.cu zmodel/transforms/riesz.cu \
zmodel/stencils/dx.cu zmodel/stencils/dy.cu

.PHONY: zmodel

# This option is outdated
3D:
	mpicxx $(FLAGS) $(INCLUDE) fft_testing/stencil.cpp -o stencil.x $(LIB)
	mpiexec -n 8 ./stencil.x 8 2

# This option is outdated
2D:
	mpicxx $(FLAGS) $(INCLUDE) fft_testing/stencil2D.cpp -o stencil2D.x $(LIB)
	mpiexec -n 4 ./stencil2D.x 16 4 1 

zmodel:
	nvcc -ccbin=mpicxx -std=c++17 -DCUFFTDX_DISABLE_CUTLASS_DEPENDENCY -I$(CUDA_HOME)/include \
	$(FFTDX_INCLUDE) $(ZMODEL_FILES) -o zmodel.x -L$(CUDA_HOME)/lib64 -lcufft -lcudart
	mpiexec -n 4 ./zmodel.x
#mpiexec --mca btl_ofi_provider_exclude psm3 --hostfile hostname -n 4 ./zmodel.x

measure_cuda_ab:
	nvcc -ccbin=mpicxx measure/measure.cu -o measure_cuda.x
	mpirun -n 2 measure_cuda.x

measure_cpp_ab:
	mpicxx $(FLAGS) $(INCLUDE) measure/measure.cpp -o measure_ab.x $(LIB)
	mpirun -n 2 measure_ab.x

test_fftdx:
	nvcc -std=c++17 -DCUFFTDX_DISABLE_CUTLASS_DEPENDENCY -I$(CUDA_HOME)/include \
	$(FFTDX_INCLUDE) measure/test_fftdx.cu -o test_fftdx.x -L$(CUDA_HOME)/lib64 -lcufft -lcudart
	./test_fftdx.x

test_cufft:
	nvcc -std=c++17 -I$(CUDA_HOME)/include measure/test_cufft.cu -o test_cufft.x -L$(CUDA_HOME)/lib64 -lcufft -lcudart
	./test_cufft.x


clean:
	rm *.x