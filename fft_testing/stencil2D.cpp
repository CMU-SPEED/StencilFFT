#include <iostream>
#include <cassert>
#include <mpi.h>
#include <chrono>
#include <stdlib.h>
#include <cmath>
#include <complex>
#include <fftw3.h>

// assume that r = c & p = rc;
// This will be defined at runtime
int r = 0; 
int c = 0;
int N = 0;
int b = 0;
int t = 0;

using namespace std;
using Complex = std::complex<double>;

/**
 * @param subvector_len Size of the dimensions of the fft
 * @param stride The stride (in terms of the number of elements)
 *               between consecutive elements in a single transform
 * @param dist Distance between the first elem in one transform
 *             and the first elem of the next in the input
 * NOTE: input data distribution is preserved in the output
 * FIXME: This is not performant
*/
double applyFFT(Complex *data, int subvector_len, int stride, int dist, int id) {
  //TODO: SOME ASSERT HERE about inputs

  // The number of separate transforms to be performed
  int howmany = (N/r * N/c) / subvector_len;

  fftw_complex* input = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N/r * N/c);
  fftw_complex* output = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N/r * N/c);

  for (int i = 0; i < N/r * N/c; i++) {
      input[i][0] = data[i].real();
      input[i][1] = data[i].imag();
  }

  int rank = 1;               // 1D FFT
  int n[] = {subvector_len};
  int istride = stride;
  int idist = dist;
  int ostride = stride;   
  int odist = dist;
  int *inembed = nullptr;     // Input array stored contiguously in memory
  int *onembed = nullptr;

  fftw_plan plan = fftw_plan_many_dft(rank, n, howmany,
                                      input, inembed, istride, idist,
                                      output, onembed, ostride, odist,
                                      FFTW_FORWARD, FFTW_ESTIMATE);

  MPI_Barrier(MPI_COMM_WORLD);

  // auto start = std::chrono::high_resolution_clock::now();

  fftw_execute(plan);

  MPI_Barrier(MPI_COMM_WORLD);

  // auto end = std::chrono::high_resolution_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  // double ns = duration.count();
  // if (id == 0) std::cout << "FFT: " << ns << " ns" << std::endl;

  for (int i = 0; i < N/r * N/c; i++) {
    data[i] = Complex(output[i][0], output[i][1]);
  }

  fftw_destroy_plan(plan);
  fftw_free(input);
  fftw_free(output);

  // return ns;
  return 0.0;
}

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
  if (argc < 4) {
    cout<<"Please provide global size (N), block size (b), and number of thread (t) per processor"<<endl;
    return 1;
  }

  //size of global block
  N = atoi(argv[1]);

  //size of local block
  b = atoi(argv[2]);

  //number of threads
  t = atoi(argv[3]);

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
          double real = (rid*N*b + cid*b) +               //processor offset
                          (j * N * b * c) + (i * b * r) + //block offset
                          (jj*N + ii);
          //double real = 1.0;
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

  // FFT

  // MPI_Barrier(MPI_COMM_WORLD);

  // auto start = std::chrono::high_resolution_clock::now();

  // Local transpose: swap i and j
  for (int i = 0; i < N/r/b; i++) {
    for (int j = 0; j < N/c/b; j++) {
      for (int ii = 0; ii < b; ii++) {
        for (int jj = 0; jj < b; jj++) {
          int src_index = i * (N/c/b * b * b) + j * (b * b) + ii * b + jj;
          int dst_index = j * (N/r/b * b * b) + i * (b * b) + ii * b + jj;
          out[dst_index] = in[src_index];
        }
      }
    }
  }

  // MPI_Barrier(MPI_COMM_WORLD);

  // auto end = std::chrono::high_resolution_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  // double compute_count = duration.count();
  // if (id == 0) std::cout << "Local Transpose 1: " << duration.count() << " ns" << std::endl;

  // applyFFT(out, N/r/b, (N/r * N/c) / (N/r/b), 1, id);

  // MPI_Barrier(MPI_COMM_WORLD);

  // start = std::chrono::high_resolution_clock::now();

  MPI_Alltoall(out,
               (N/r * N/c) / r,
               MPI_C_DOUBLE_COMPLEX,
               in, 
               (N/r * N/c) / r,
               MPI_C_DOUBLE_COMPLEX,
               row_comm);

  // MPI_Barrier(MPI_COMM_WORLD);

  // end = std::chrono::high_resolution_clock::now();
  // duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  // double comms_count = duration.count();
  // if (id == 0) std::cout << "All to all rows time: " << duration.count() << " ns" << std::endl;

  // MPI_Barrier(MPI_COMM_WORLD);

  // start = std::chrono::high_resolution_clock::now();

  // Local transpose
  for (int i = 0; i < N/r/b; i++) {
    for (int j = 0; j < N/c/b; j++) {
      for (int ii = 0; ii < b; ii++) {
        for (int jj = 0; jj < b; jj++) {
          int src_index = i * (N/c/b * b * b) + j * (b * b) + ii * b + jj;
          int dst_index = j * (N/r/b * b * b) + i * (b * b) + ii * b + jj;
          out[dst_index] = in[src_index];
        }
      }
    }
  }

  // Pack data in order to apply fft and local twiddles
  // for (size_t i = 0; i < N/r/b; i++) {    // Row of the block
    // for (size_t j = 0; j < N/c/b; j++) {  // Col of the block
      // size_t col = (j % c) * ((N/c/b)/c) + (j / c);
      // for (size_t ii = 0; ii < b; ii++) {   // Row inside the block
        // for (size_t jj = 0; jj < b; jj++) { // Col inside the block
          // size_t dst_index = (i * b * b * N/c/b) + (j * b) + (ii * N/c) + jj;
          // size_t src_index = (i * b * b * N/c/b) + (col * b * b) + (ii * b) + jj;
          // in[dst_index] = out[src_index];
        // }
      // }
    // }
  // }

  // Local twiddles 
  // for (int i = 0; i < N/r; i++) {
    // for (int j = 0; j < N/c; j += N/c/b) {
      // for (int jj = 0; jj < N/c/b; jj++) {
        // double k = (double)(cid * ((N/(b*c))/c)) + ((j + jj) / (b * c)); // Row
        // double l = (j + jj) % (b * c);                                    // Col
        // in[(i * N/c) + j + jj] *= std::exp(Complex(0.0, -2*M_PI*k*l/(N))); 
      // }
    // }
  // }
  
  // MPI_Barrier(MPI_COMM_WORLD);

  // end = std::chrono::high_resolution_clock::now();
  // duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  // compute_count += duration.count();
  // if (id == 0) std::cout << "Local Transpose 2, packing, twiddles: " << duration.count() << " ns" << std::endl;

  // applyFFT(in, b * c, 1, b * c, id);

  // MPI_Barrier(MPI_COMM_WORLD);

  // start = std::chrono::high_resolution_clock::now();

  // Unpack
  // for (size_t i = 0; i < N/r/b; i++) {    // Row of the block
    // for (size_t j = 0; j < N/c/b; j++) {  // Col of the block
      // size_t col = (j % c) * ((N/c/b)/c) + (j / c);
      // for (size_t ii = 0; ii < b; ii++) {   // Row inside the block
        // for (size_t jj = 0; jj < b; jj++) { // Col inside the block
          // size_t dst_index = (i * b * b * N/c/b) + (j * b) + (ii * N/c) + jj;
          // size_t src_index = (i * b * b * N/c/b) + (col * b * b) + (ii * b) + jj;
          // out[src_index] = in[dst_index];
        // }
      // }
    // }
  // }

  // MPI_Barrier(MPI_COMM_WORLD);

  // end = std::chrono::high_resolution_clock::now();
  // duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  // compute_count += duration.count();
  // if (id == 0) std::cout << "Unpacking: " << duration.count() << " ns" << std::endl;

  // applyFFT(out, N/r/b, (N/r * N/c) / (N/r/b), 1, id);

  // FIXME: PRINT HERE
  for (int j = 0; j < p; ++j) {
    if (id == j) {
	    cout<<id<<": ("<<rid<<", "<<cid<<") grp: ("<<row_grp<<", "<<col_grp<<") | ";
	    for (int i = 0; i < N/r * N/c; ++i)
        cout<<"("<<in[i].real()<<", "<<in[i].imag()<<") ";
	      cout<<endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // MPI_Barrier(MPI_COMM_WORLD);

  // start = std::chrono::high_resolution_clock::now();

  // MPI_Alltoall(out,
  //              (N/r * N/c) / c,
  //              MPI_C_DOUBLE_COMPLEX,
  //              in, 
  //              (N/r * N/c) / c,
  //              MPI_C_DOUBLE_COMPLEX,
  //              col_comm);

  // MPI_Barrier(MPI_COMM_WORLD);

  // end = std::chrono::high_resolution_clock::now();
  // duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  // comms_count += duration.count();
  // if (id == 0) std::cout << "All to all cols time: " << duration.count() << " ns" << std::endl;

  // MPI_Barrier(MPI_COMM_WORLD);

  // start = std::chrono::high_resolution_clock::now();

  // Pack -- transpose first so we can apply the same packing routine, twiddles, and fft
  // for (int i = 0; i < N/r/b; i++) {
    // for (int j = 0; j < N/c/b; j++) {
      // for (int ii = 0; ii < b; ii++) {
        // for (int jj = 0; jj < b; jj++) {
          // int src_index = i * (N/c/b * b * b) + j * (b * b) + ii * b + jj;
          // int dst_index = j * (N/r/b * b * b) + i * (b * b) + jj * b + ii;
          // out[dst_index] = in[src_index];
        // }
      // }
    // }
  // }

  // for (size_t i = 0; i < N/r/b; i++) {    // Row of the block
    // for (size_t j = 0; j < N/c/b; j++) {  // Col of the block
      // size_t col = (j % c) * ((N/c/b)/c) + (j / c);
      // for (size_t ii = 0; ii < b; ii++) {   // Row inside the block
        // for (size_t jj = 0; jj < b; jj++) { // Col inside the block
          // size_t dst_index = (i * b * b * N/c/b) + (j * b) + (ii * N/c) + jj;
          // size_t src_index = (i * b * b * N/c/b) + (col * b * b) + (ii * b) + jj;
          // in[dst_index] = out[src_index];
        // }
      // }
    // }
  // }

  // Local twiddles
  // for (int i = 0; i < N/r; i++) {
    // for (int j = 0; j < N/c; j += N/c/b) {
      // for (int jj = 0; jj < N/c/b; jj++) {
        // double k = (double)(rid * ((N/(b*c))/c)) + ((j + jj) / (b * c)); // Row
        // double l = (j + jj) % (b * c);                                   // Col
        // in[(i * N/c) + j + jj] *= std::exp(Complex(0.0, -2*M_PI*k*l/(N))); 
      // }
    // }
  // }
    
  // MPI_Barrier(MPI_COMM_WORLD);

  // end = std::chrono::high_resolution_clock::now();
  // duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  // compute_count += duration.count();
  // if (id == 0) std::cout << "packing, twiddles: " << duration.count() << " ns" << std::endl;

  // applyFFT(in, b * c, 1, b * c, id);

  // MPI_Barrier(MPI_COMM_WORLD);

  // start = std::chrono::high_resolution_clock::now();

  // Unpack
  // for (size_t i = 0; i < N/r/b; i++) {    // Row of the block
    // for (size_t j = 0; j < N/c/b; j++) {  // Col of the block
      // size_t col = (j % c) * ((N/c/b)/c) + (j / c);
      // for (size_t ii = 0; ii < b; ii++) {   // Row inside the block
        // for (size_t jj = 0; jj < b; jj++) { // Col inside the block
          // size_t dst_index = (i * b * b * N/c/b) + (j * b) + (ii * N/c) + jj;
          // size_t src_index = (i * b * b * N/c/b) + (col * b * b) + (ii * b) + jj;
          // out[src_index] = in[dst_index];
        // }
      // }
    // }
  // }

  // for (int i = 0; i < N/r/b; i++) {
    // for (int j = 0; j < N/c/b; j++) {
      // for (int ii = 0; ii < b; ii++) {
        // for (int jj = 0; jj < b; jj++) {
          // int src_index = i * (N/c/b * b * b) + j * (b * b) + ii * b + jj;
          // int dst_index = j * (N/r/b * b * b) + i * (b * b) + jj * b + ii;
          // in[src_index] = out[dst_index];
        // }
      // }
    // }
  // }

  // MPI_Barrier(MPI_COMM_WORLD);

  // end = std::chrono::high_resolution_clock::now();
  // duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  // compute_count += duration.count();
  // if (id == 0) std::cout << "Unpacking: " << duration.count() << " ns" << std::endl;

  // if (id == 0) cout<<"End Result"<<endl;
  // if (id == 0) cout<<in[0]<<endl;
  // if (id == 0) cout<<"Compute Count "<<compute_count<<endl;
  // if (id == 0) cout<<"Comms Count "<<comms_count<<endl;
  // for (int j = 0; j < p; ++j) {
  //   if (id == j) {
	//     cout<<id<<": ("<<rid<<", "<<cid<<") grp: ("<<row_grp<<", "<<col_grp<<") | ";
	//     for (int i = 0; i < N/r * N/c; ++i)
  //       cout<<"("<<in[i].real()<<", "<<in[i].imag()<<") ";
	//       cout<<endl;
  //   }
  //   MPI_Barrier(MPI_COMM_WORLD);
  // }

  // Clean up
  free(in);
  free(out);

  MPI_Finalize();
  
  return 0;
}