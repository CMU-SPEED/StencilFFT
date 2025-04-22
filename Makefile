# Standard paths on Linux
LIB_PATH = /usr
INCLUDE_PATH = LIB_PATH
FLAGS= -O3 -fopenmp

LIB = -L$(LIB_PATH)/lib -lfftw3_omp -lfftw3
INCLUDE = -I$(LIB_PATH)/include

3D:
	mpicxx $(FLAGS) $(INCLUDE) stencil.cpp -o stencil.x $(LIB)
	mpiexec -n 8 ./stencil.x 8 2
#mpiexec --mca btl_ofi_provider_exclude psm3 --hostfile hostname -n 8 ./stencil.x 512 128

2D:
	mpicxx $(FLAGS) $(INCLUDE) stencil2D.cpp -o stencil2D.x $(LIB)
	mpiexec -n 4 ./stencil2D.x 16 4 1 
#mpiexec --mca btl_ofi_provider_exclude psm3 --hostfile hostname -n 4 ./stencil2D.x 1024 256

clean:
	rm *.x