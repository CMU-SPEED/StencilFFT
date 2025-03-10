all:
	mpicxx stencil.cpp -o stencil.x 
	mpiexec -n 8 --oversubscribe ./stencil.x

clean:
	rm *.x