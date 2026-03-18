# Standard paths on Linux
LIB_PATH = /usr
LIB = -L$(LIB_PATH)/lib 
INCLUDE = -I$(LIB_PATH)/include

CUDA_HOME=/usr/local/cuda

FFTDX_INCLUDE=-I/afs/andrew.cmu.edu/usr12/cjstange/private/fftdx/nvidia-mathdx-25.12.1-cuda13/nvidia/mathdx/25.12/include

ZMODEL_FILES = zmodel/main.cu zmodel/utils.cu zmodel/transforms/fft.cu \
zmodel/stencils/laplace.cu zmodel/stencils/dx.cu zmodel/stencils/dy.cu \
zmodel/transforms/fft3d.cu zmodel/stencils/laplace3d.cu

# Uncomment and set HDF5_HOME to enable HDF5 I/O (also uncomment __USE__HDF5__ in utils.h)
# Need to have access to parallel hdf5 for this work
#HDF5_HOME=/usr/local/hdf5
#HDF5_FLAGS=-I$(HDF5_HOME)/include -L$(HDF5_HOME)/lib -lhdf5

.PHONY: zmodel

zmodel:
	nvcc -ccbin=mpicxx -std=c++17 -O3 -DCUFFTDX_DISABLE_CUTLASS_DEPENDENCY -I$(CUDA_HOME)/include \
	$(FFTDX_INCLUDE) $(ZMODEL_FILES) -o zmodel.x -L$(CUDA_HOME)/lib64 -lcufft -lcudart
# 	mpiexec -n 4 ./zmodel.x
	mpiexec -n 8 ./zmodel.x
# Uncomment and swap in this line if you want to run the code on multple machines
#mpiexec --mca btl_ofi_provider_exclude psm3 --hostfile hostname -n 4 ./zmodel.x

measure_cuda_ab:
	nvcc -ccbin=mpicxx measure/measure.cu -o measure_cuda.x
	mpirun -n 2 measure_cuda.x

measure_cpp_ab:
	mpicxx $(INCLUDE) measure/measure.cpp -o measure_ab.x $(LIB)
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