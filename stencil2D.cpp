#include <iostream>
#include <cassert>
#include <mpi.h>
#include <chrono>
#include <stdlib.h>
#include <cmath>
#include <complex>
//#include <fftw3.h>

// assume that r = c & p = rc;
// This will be defined at runtime
int r = 0; 
int c = 0;
int N = 0;
int b = 0;

using namespace std;
using Complex = std::complex<double>;

// FIXME: This is not performant
// void applyFFT(Complex *data, int block_size) {
//   int N = data.size();
//   int subvector_len = N / block_size;
//   int howmany = block_size;

//   fftw_complex* input = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N);
//   fftw_complex* output = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N);

//   for (int i = 0; i < N; i++) {
//       input[i][0] = data[i].real();
//       input[i][1] = data[i].imag();
//   }

//   int rank = 1;
//   int n[] = {subvector_len};
//   int istride = block_size;
//   int idist = 1;           
//   int ostride = block_size;   
//   int odist = 1;
//   int *inembed = nullptr;  
//   int *onembed = nullptr;

//   fftw_plan plan = fftw_plan_many_dft(rank, n, howmany,
//                                         input, inembed, istride, idist,
//                                         output, onembed, ostride, odist,
//                                         FFTW_FORWARD, FFTW_ESTIMATE);

//   fftw_execute(plan);

//   for (int i = 0; i < N; i++) {
//     data[i] = Complex(output[i][0], output[i][1]);
//   }

//   fftw_free(input);
//   fftw_free(output);
// }

int calc_id(int rid, int cid, int edit_r, int edit_c) {
  int new_rid, new_cid;

  switch (edit_r) {
    case -1:
      new_rid = (rid == 0) ? (r - 1) : (rid - 1);
      break;
    case 1:
      new_rid = (rid + 1) % r;
      break;
    case 0:
      new_rid = rid;
      break;
    default:
      assert(false);
  }

  switch (edit_c) {
    case -1:
      new_cid = (cid == 0) ? (c - 1) : (cid - 1);
      break;
    case 1:
      new_cid = (cid + 1) % c;
      break;
    case 0:
      new_cid = cid;
      break;
    default:
      assert(false);
  }

  return (new_rid * c) + new_cid;
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    cout<<"Please provide global size (N) and block size (b)"<<endl;
    return 1;
  }

  //size of global block
  N = atoi(argv[1]);

  //size of local block
  b = atoi(argv[2]);

  MPI_Init(NULL, NULL);

  int p, id;
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &p); 

  r = static_cast<int>(std::round(std::sqrt(p)));
  c = r;

  MPI_Comm row_comm, col_comm, dep_comm;

  int row_grp = id / c;
  MPI_Comm_split(MPI_COMM_WORLD, row_grp, id, &row_comm);

  int col_grp = id % r;
  MPI_Comm_split(MPI_COMM_WORLD, col_grp, id, &col_comm); 

  // Keep same naming convention as 3D.  In 3D grp != id
  int rid, cid;
  rid = row_grp;
  cid = col_grp;

  Complex *in = (Complex*) malloc(sizeof(Complex) * N/r * N/c);
  Complex *out = (Complex*) malloc(sizeof(Complex) * N/r * N/c);
  
  //init
  for (int i = 0; i < N/r/b; i++)
    for (int j = 0; j < N/c/b; j++)
      for (int ii = 0; ii < b; ii++) 
        for (int jj = 0; jj < b; jj++) {
          double real = (rid*N*b + cid*b) + //processor offset
                          (j * N * b * c) + (i * b * r) + //block offset
                          (jj*N + ii);
          in[(j * b * b * N/c/b) +
             (i * b * b) + 
             (jj*b + ii)] = Complex(real, 0.0);
  }

  if (id == 0) cout<<"Initial data distribution"<<endl;
  for (int j = 0; j < p; ++j) {
    if (id == j) {
	    cout<<id<<": ("<<rid<<", "<<cid<<") grp: ("<<row_grp<<", "<<col_grp<<") ";
	    for (int i = 0; i < N/r * N/c; ++i)
	        cout<<in[i].real()<<" ";
	        cout<<endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Stencil
  // int g = 1;
  // int b_per_p = (N/r * N/c) / (b * b);
  // int edge = b * g;
  // int total_edge = b_per_p * edge;
  // int corner = g * g;
  // int total_corner = b_per_p * corner;

  // TODO: Copy data to stencil buffers 

  // Top and bottom edge
  // double *edge00_01_send = (double*) malloc(sizeof(double) * total_edge);
  // double *edge00_01_recv = (double*) malloc(sizeof(double) * total_edge);
  // double *edge10_11_send = (double*) malloc(sizeof(double) * total_edge);
  // double *edge10_11_recv = (double*) malloc(sizeof(double) * total_edge);
  // int edge00_01_id = calc_id(rid, cid, 1, 0); // gets row below
  // int edge10_11_id = calc_id(rid, cid, -1, 0); // gets row above

  // // Right and left edge
  // double *edge00_10_send = (double*) malloc(sizeof(double) * total_edge);
  // double *edge00_10_recv = (double*) malloc(sizeof(double) * total_edge);
  // double *edge01_11_send = (double*) malloc(sizeof(double) * total_edge);
  // double *edge01_11_recv = (double*) malloc(sizeof(double) * total_edge);
  // int edge00_10_id = calc_id(rid, cid, 0, 1); // gets col right
  // int edge01_11_id = calc_id(rid, cid, 0, -1); // gets col left

  // // Top right and bottom left corner
  // double *corner00_send = (double*) malloc(sizeof(double) * total_corner);
  // double *corner00_recv = (double*) malloc(sizeof(double) * total_corner);
  // double *corner11_send = (double*) malloc(sizeof(double) * total_corner);
  // double *corner11_recv = (double*) malloc(sizeof(double) * total_corner);
  // int corner00_id = calc_id(rid, cid, 1, -1); // below, left
  // int corner11_id = calc_id(rid, cid, -1, 1); // above, right

  // // Top left and bottom right corner
  // double *corner10_send = (double*) malloc(sizeof(double) * total_corner);
  // double *corner10_recv = (double*) malloc(sizeof(double) * total_corner);
  // double *corner01_send = (double*) malloc(sizeof(double) * total_corner);
  // double *corner01_recv = (double*) malloc(sizeof(double) * total_corner);
  // int corner10_id = calc_id(rid, cid, -1, -1); // above, left
  // int corner01_id = calc_id(rid, cid, 1, 1); //below, right

  // auto start = std::chrono::high_resolution_clock::now();

  // MPI_Sendrecv(edge00_01_send, total_edge, MPI_DOUBLE, edge00_01_id, 0,
  //              edge10_11_recv, total_edge, MPI_DOUBLE, edge10_11_id, MPI_ANY_TAG,
  //              MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  // MPI_Sendrecv(edge10_11_send, total_edge, MPI_DOUBLE, edge10_11_id, 0,
  //              edge00_01_recv, total_edge, MPI_DOUBLE, edge00_01_id, MPI_ANY_TAG,
  //              MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  // MPI_Sendrecv(edge00_10_send, total_edge, MPI_DOUBLE, edge00_10_id, 0,
  //              edge01_11_recv, total_edge, MPI_DOUBLE, edge01_11_id, MPI_ANY_TAG,
  //              MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  // MPI_Sendrecv(edge01_11_send, total_edge, MPI_DOUBLE, edge01_11_id, 0,
  //              edge00_10_recv, total_edge, MPI_DOUBLE, edge00_10_id, MPI_ANY_TAG,
  //              MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  // MPI_Sendrecv(corner00_send, total_corner, MPI_DOUBLE, corner00_id, 0,
  //              corner11_recv, total_corner, MPI_DOUBLE, corner11_id, MPI_ANY_TAG,
  //              MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  // MPI_Sendrecv(corner11_send, total_corner, MPI_DOUBLE, corner11_id, 0,
  //              corner00_recv, total_corner, MPI_DOUBLE, corner00_id, MPI_ANY_TAG,
  //              MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  // MPI_Sendrecv(corner10_send, total_corner, MPI_DOUBLE, corner10_id, 0,
  //              corner01_recv, total_corner, MPI_DOUBLE, corner01_id, MPI_ANY_TAG,
  //              MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  // MPI_Sendrecv(corner01_send, total_corner, MPI_DOUBLE, corner01_id, 0,
  //              corner10_recv, total_corner, MPI_DOUBLE, corner10_id, MPI_ANY_TAG,
  //              MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  
  // auto end = std::chrono::high_resolution_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  // if (id == 0) std::cout << "Stencil time: " << duration.count() << " ns" << std::endl;

  // All to all 

  //local = local.transpose((1, 0, 2, 3))
  // swap the first two dimensions --> swap i and j
  int A = N/r/b;
  int B = N/c/b;
  for (int i = 0; i < A; i++) {
    for (int j = 0; j < B; j++) {
      for (int ii = 0; ii < b; ii++) {
        for (int jj = 0; jj < b; jj++) {
          int src_index = i * (B * b * b) + j * (b * b) + ii * b + jj;
          int dst_index = j * (A * b * b) + i * (b * b) + ii * b + jj;
          out[dst_index] = Complex(in[src_index].real(), in[src_index].imag());
        }
      }
    }
  }

  if (id == 0) cout<<"After Local Transpose 1"<<endl;
  for (int j = 0; j < p; ++j) {
    if (id == j) {
	    cout<<id<<": ("<<rid<<", "<<cid<<") grp: ("<<row_grp<<", "<<col_grp<<") ";
	    for (int i = 0; i < N/r * N/c; ++i)
	        cout<<out[i].real()<<" ";
	        cout<<endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // start = std::chrono::high_resolution_clock::now();

  MPI_Alltoall(out,
               (N/r * N/c) / r,
               MPI_C_DOUBLE_COMPLEX,
               in, 
               (N/r * N/c) / r,
               MPI_C_DOUBLE_COMPLEX,
               row_comm);

  if (id == 0) cout<<"After All to all in Rows"<<endl;
  for (int j = 0; j < p; ++j) {
    if (id == j) {
	    cout<<id<<": ("<<rid<<", "<<cid<<") grp: ("<<row_grp<<", "<<col_grp<<") ";
	    for (int i = 0; i < N/r * N/c; ++i)
	        cout<<in[i].real()<<" ";
	        cout<<endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  //end = std::chrono::high_resolution_clock::now();
  //duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  //if (id == 0) std::cout << "All to all rows time: " << duration.count() << " ns" << std::endl;

  //local = local.transpose((1, 0, 2, 3))
  for (int i = 0; i < A; i++) {
    for (int j = 0; j < B; j++) {
      for (int ii = 0; ii < b; ii++) {
        for (int jj = 0; jj < b; jj++) {
          // Calculate the index in the source array:
          int src_index = i * (B * b * b) + j * (b * b) + ii * b + jj;
          // Calculate the index in the destination array, where i and j are swapped:
          int dst_index = j * (A * b * b) + i * (b * b) + ii * b + jj;
          out[dst_index] = Complex(in[src_index].real(), in[src_index].imag());
        }
      }
    }
  }

  if (id == 0) cout<<"After Local Transpose 2"<<endl;
  for (int j = 0; j < p; ++j) {
    if (id == j) {
	    cout<<id<<": ("<<rid<<", "<<cid<<") grp: ("<<row_grp<<", "<<col_grp<<") ";
	    for (int i = 0; i < N/r * N/c; ++i)
	        cout<<out[i].real()<<" ";
	        cout<<endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  //start = std::chrono::high_resolution_clock::now();

  MPI_Alltoall(out,
               (N/r * N/c) / c,
               MPI_C_DOUBLE_COMPLEX,
               in, 
               (N/r * N/c) / c,
               MPI_C_DOUBLE_COMPLEX,
               col_comm);

  if (id == 0) cout<<"After All to all in Cols"<<endl;
  for (int j = 0; j < p; ++j) {
    if (id == j) {
	    cout<<id<<": ("<<rid<<", "<<cid<<") grp: ("<<row_grp<<", "<<col_grp<<") ";
	    for (int i = 0; i < N/r * N/c; ++i)
	        cout<<in[i].real()<<" ";
	        cout<<endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  //end = std::chrono::high_resolution_clock::now();
  //duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  //if (id == 0) std::cout << "All to all cols time: " << duration.count() << " ns" << std::endl;

  // Clean up
  free(in);
  free(out);
  // free(edge00_01_send);
  // free(edge00_01_recv);
  // free(edge10_11_send);
  // free(edge10_11_recv);
  // free(edge00_10_send);
  // free(edge00_10_recv);
  // free(edge01_11_send);
  // free(edge01_11_recv);
  // free(corner00_send);
  // free(corner00_recv);
  // free(corner10_send);
  // free(corner10_recv);
  // free(corner01_send);
  // free(corner01_recv);
  // free(corner11_send);
  // free(corner11_recv);

  MPI_Finalize();
  
  return 0;
}