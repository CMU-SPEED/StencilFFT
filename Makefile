# Standard paths on Linux
LIB_PATH = /usr
INCLUDE_PATH = LIB_PATH
FLAGS= -O3 -fopenmp

LIB = -L$(LIB_PATH)/lib -lfftw3_omp -lfftw3
INCLUDE = -I$(LIB_PATH)/include

CUDA_HOME=/usr/local/cuda

ZMODEL_FILES = zmodel/main.cu zmodel/transforms/fft.cu zmodel/transforms/riesz.cu \
zmodel/stencils/dy.cu

.PHONY: zmodel

3D:
	mpicxx $(FLAGS) $(INCLUDE) fft_testing/stencil.cpp -o stencil.x $(LIB)
	mpiexec -n 8 ./stencil.x 8 2
#mpiexec --mca btl_ofi_provider_exclude psm3 --hostfile hostname -n 8 ./stencil.x 512 128

2D:
	mpicxx $(FLAGS) $(INCLUDE) fft_testing/stencil2D.cpp -o stencil2D.x $(LIB)
	mpiexec -n 4 ./stencil2D.x 16 4 1 
#mpiexec --mca btl_ofi_provider_exclude psm3 --hostfile hostname -n 4 ./stencil2D.x 1024 256

zmodel:
	nvcc -ccbin=mpicxx -std=c++17 -I$(CUDA_HOME)/include $(ZMODEL_FILES) -o zmodel.x -L$(CUDA_HOME)/lib64 -lcufft -lcudart
	mpiexec -n 1 ./zmodel.x
#mpiexec -n 4 ./zmodel.x

# zmodel_fft_gpu:
# 	nvcc -ccbin=mpicxx -I$(CUDA_HOME)/include zmodel/zmodel_fft.cu -o zmodel_fft.x -L$(CUDA_HOME)/lib64 -lcufft -lcudart
# 	mpiexec -n 4 ./zmodel_fft.x
#mpiexec --mca btl_ofi_provider_exclude psm3 --hostfile hostname -n 4 ./zmodel_fft.x

# laplace:
# 	nvcc zmodel/laplace.cu -o laplace.x
# 	./laplace.x

# dx:
# 	nvcc zmodel/dx.cu -o dx.x
# 	./dx.x

# dy:
# 	nvcc zmodel/dy.cu -o dy.x
# 	./dy.x

measure_cuda:
	nvcc -ccbin=mpicxx measure/measure.cu -o measure_cuda.x
	mpirun -n 2 measure_cuda.x

measure_ab:
	mpicxx $(FLAGS) $(INCLUDE) measure/measure.cpp -o measure_ab.x $(LIB)
	mpirun -n 2 measure_ab.x

clean:
	rm *.x