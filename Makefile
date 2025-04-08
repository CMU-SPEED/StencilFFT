# Standard paths on Linux
LIB_PATH = /usr/local
INCLUDE_PATH = LIB_PATH

LIB = -L$(LIB_PATH)/lib -lfftw3
INCLUDE = -I$(LIB_PATH)/include

3D:
	mpicxx $(INCLUDE) stencil.cpp -o stencil.x $(LIB)
	mpiexec -n 8 ./stencil.x 8 2
#mpiexec --mca btl_ofi_provider_exclude psm3 --hostfile hostname -n 8 ./stencil.x 512 128

2D:
	mpicxx $(INCLUDE) stencil2D.cpp -o stencil2D.x $(LIB)
	mpiexec -n 4 ./stencil2D.x 16 2 
#mpiexec --mca btl_ofi_provider_exclude psm3 --hostfile hostname -n 4 ./stencil2D.x 1024 256

clean:
	rm *.x