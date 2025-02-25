#include <iostream>
#include <mpi.h>

using namespace std;

int main()
{

  int N = 4;  //size of global block

  int b = 2; //size of local block
  
  int r, c, d, p, id;


  

  MPI_Init(NULL, NULL);
  
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &p);  

  MPI_Comm row_comm, col_comm, dep_comm;

  /*
  //check later
  // [0-3 => 0, 4-7 => 1]
  int dep_grp = id % (r*c);
  MPI_Comm_split(MPI_COMM_WORLD, dep_grp, id, &dep_comm);

  // [0,1 => 0, 2,3 => 1 4,5 => 2, 6,7 => 3]
  int row_grp = id / r;
  MPI_Comm_split(MPI_COMM_WORLD, row_grp, id, &row_comm);

  // [0,1,4,5 => 0, 2,3,6,7 => 1]
  int col_grp = (row_grp / r) * c + id % d;
  MPI_Comm_split(MPI_COMM_WORLD, col_grp, id, &col_comm);  
  */
  
  //assume that r = c = d & p = rcd; 
  r = c = d = 2;

  int rid, cid, did;

  did = id / (r * c);
  rid = (id - (did * r*c)) / c;
  cid = (id - (did * r * c)) % r;
  double *in, *out;


  // 4 x 4 x 4 local data cube that is block cyclic dist in 2x2x2 blocks
  // total of 8 blocks per local processor.
  in = (double*) malloc(sizeof(double) * N/r  * N/c * N/d);
  out = (double*) malloc(sizeof(double) * N/r * N/c * N/d);

  //init
  //  for (int i = 0; i < N/r/b; ++i)
  //    for (int j = 0; j < N/c/b; ++j)
  //      for (int k = 0; k < N/d/b; ++k)
	{
	  for (int ii = 0; ii < b; ++ii)
	    for (int jj = 0; jj < b; ++jj)
	      for (int kk = 0; kk < b; ++kk)

		in[(kk*b*b + jj*b + ii)] = (did*N*N*b + rid * N * b + cid * b) +  //compute start offsets
		                           kk*N*N + jj*N + ii;
	}
	

  for (int j = 0; j < p; ++j)
    {
    if (id == j)
      {
	cout<<id<<": ("<<rid<<", "<<cid<<", "<<did<<") ";
	for (int i = 0; i < 8; ++i)
	  cout<<in[i]<<" ";
	cout<<endl;      
      }
    MPI_Barrier(MPI_COMM_WORLD);
    }
  
  free(in);
  free(out);

  MPI_Finalize();
  
  return 0;
}
