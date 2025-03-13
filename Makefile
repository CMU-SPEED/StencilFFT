all:
	mpicxx stencil.cpp -o stencil.x 
	mpiexec --mca btl_ofi_provider_exclude psm3 --hostfile hostname -n 8 ./stencil.x 4 2

clean:
	rm *.x