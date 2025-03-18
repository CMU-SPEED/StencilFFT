3D:
	mpicxx stencil.cpp -o stencil.x 
	mpiexec --mca btl_ofi_provider_exclude psm3 --hostfile hostname -n 8 ./stencil.x 512 128

2D:
	mpicxx stencil2D.cpp -o stencil2D.x 
#mpiexec --mca btl_ofi_provider_exclude psm3 --hostfile hostname -n 9 ./stencil2D.x 12 1
	mpiexec -n 9 ./stencil2D.x 12 1

clean:
	rm *.x