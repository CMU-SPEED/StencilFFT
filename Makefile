all:
	mpicxx stencil.cpp -o stencil.x 
	mpiexec -n 8 ./stencil.x

clean:
	rm *.x