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
void applyFFT(Complex *data, int subvector_len, int stride, int dist) {
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

  fftw_execute(plan);

  for (int i = 0; i < N/r * N/c; i++) {
    data[i] = Complex(output[i][0], output[i][1]);
  }

  fftw_destroy_plan(plan);
  fftw_free(input);
  fftw_free(output);
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
          // double real = (rid*N*b + cid*b) +               //processor offset
          //                 (j * N * b * c) + (i * b * r) + //block offset
          //                 (jj*N + ii);
          double real = 1.0;
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

  int g = 1;
  int b_per_p = (N/r * N/c) / (b * b);
  int edge = b * g;
  int total_edge = b_per_p * edge;
  int corner = g * g;
  int total_corner = b_per_p * corner;

  // Top and bottom edge
  Complex *edge00_10_send = (Complex*) malloc(sizeof(Complex) * total_edge);
  Complex *edge00_10_recv = (Complex*) malloc(sizeof(Complex) * total_edge);
  Complex *edge01_11_send = (Complex*) malloc(sizeof(Complex) * total_edge);
  Complex *edge01_11_recv = (Complex*) malloc(sizeof(Complex) * total_edge);
  int edge01_11_id = calc_id(rid, cid, 1, 0); // gets row below
  int edge00_10_id = calc_id(rid, cid, -1, 0); // gets row above

  // Right and left edge
  Complex *edge00_01_send = (Complex*) malloc(sizeof(Complex) * total_edge);
  Complex *edge00_01_recv = (Complex*) malloc(sizeof(Complex) * total_edge);
  Complex *edge10_11_send = (Complex*) malloc(sizeof(Complex) * total_edge);
  Complex *edge10_11_recv = (Complex*) malloc(sizeof(Complex) * total_edge);
  int edge10_11_id = calc_id(rid, cid, 0, 1); // gets col right
  int edge00_01_id = calc_id(rid, cid, 0, -1); // gets col left

  // Pack edge data for stencil
  for (int i = 0; i < b_per_p; i++) {
    int top_block_offset = i * (b * b);
    int bottom_block_offset = top_block_offset + (b * (b - g));

    // Top / Bottom
    for (int j = 0; j < g; j++) {
      for (int k = 0; k < b; k++) {
        // Row above
        edge00_10_send[(i * g * b) +
                       (j * b) + k] = Complex(in[top_block_offset + (j * b) + k].real(),
                                              in[top_block_offset + (j * b) + k].imag());

        // Row below
        edge01_11_send[(i * g * b) +
                       (j * b) + k] = Complex(in[bottom_block_offset + (j * b) + k].real(),
                                              in[bottom_block_offset + (j * b) + k].imag());
      }
    }

    // Sides
    for (int j = 0; j < b; j++) {
      for (int k = 0; k < g; k++) {
        // Col left
        edge00_01_send[(i * g * b) +
                       (j * g) + k] = Complex(in[top_block_offset + (j * b) + k].real(),
                                              in[top_block_offset + (j * b) + k].imag());

        // Col right
        edge10_11_send[(i * g * b) +
                       (j * g) + k] = Complex(in[top_block_offset + (j * b) + k + (b - g)].real(),
                                              in[top_block_offset + (j * b) + k + (b - g)].imag());
      }
    }
  }

  // if (id == 0) cout<<"Edge"<<endl;
  // for (int j = 0; j < p; ++j) {
  //   if (id == j) {
	//     cout<<id<<": ("<<rid<<", "<<cid<<") grp: ("<<row_grp<<", "<<col_grp<<") ";
	//     for (int i = 0; i < total_edge; ++i)
	//         cout<<edge10_11_send[i].real()<<" ";
	//         cout<<endl;
  //   }
  //   MPI_Barrier(MPI_COMM_WORLD);
  // }

  // Top right and bottom left corner
  Complex *corner00_send = (Complex*) malloc(sizeof(Complex) * total_corner);
  Complex *corner00_recv = (Complex*) malloc(sizeof(Complex) * total_corner);
  Complex *corner11_send = (Complex*) malloc(sizeof(Complex) * total_corner);
  Complex *corner11_recv = (Complex*) malloc(sizeof(Complex) * total_corner);
  int corner00_id = calc_id(rid, cid, 1, -1); // below, left
  int corner11_id = calc_id(rid, cid, -1, 1); // above, right

  // Top left and bottom right corner : (x, y)
  Complex *corner10_send = (Complex*) malloc(sizeof(Complex) * total_corner);
  Complex *corner10_recv = (Complex*) malloc(sizeof(Complex) * total_corner);
  Complex *corner01_send = (Complex*) malloc(sizeof(Complex) * total_corner);
  Complex *corner01_recv = (Complex*) malloc(sizeof(Complex) * total_corner);
  int corner01_id = calc_id(rid, cid, -1, -1); // above, left
  int corner10_id = calc_id(rid, cid, 1, 1); // below, right

  // Pack corner data for stencil
  for (int i = 0; i < b_per_p; i++) {
    int top_block_offset = i * (b * b);
    int bottom_block_offset = top_block_offset + (b * (b - g));
    for (int j = 0; j < g; j++) {
      for (int k = 0; k < g; k++) {
        // Bottom left
        corner00_send[(i * g * g) +
                      (j * g) + k] = Complex(in[bottom_block_offset + (j * b) + k].real(),
                                             in[bottom_block_offset + (j * b) + k].imag());
        // Bottom right
        corner10_send[(i * g * g) +
                      (j * g) + k] = Complex(in[bottom_block_offset + (j * b) + k + (b - g)].real(),
                                             in[bottom_block_offset + (j * b) + k + (b - g)].imag());
        // Top left
        corner01_send[(i * g * g) +
                      (j * g) + k] = Complex(in[top_block_offset + (j * b) + k].real(),
                                             in[top_block_offset + (j * b) + k].imag());
        // Top right
        corner11_send[(i * g * g) +
                      (j * g) + k] = Complex(in[top_block_offset + (j * b) + k + (b - g)].real(),
                                             in[top_block_offset + (j * b) + k + (b - g)].imag());
      }
    }
  }

  // if (id == 0) cout<<"Corner"<<endl;
  // for (int j = 0; j < p; ++j) {
  //   if (id == j) {
	//     cout<<id<<": ("<<rid<<", "<<cid<<") grp: ("<<row_grp<<", "<<col_grp<<") ";
	//     for (int i = 0; i < total_corner; ++i)
	//         cout<<corner11_send[i].real()<<" ";
	//         cout<<endl;
  //   }
  //   MPI_Barrier(MPI_COMM_WORLD);
  // }

  auto start = std::chrono::high_resolution_clock::now();

  MPI_Sendrecv(edge00_10_send, total_edge, MPI_DOUBLE, edge00_10_id, 0,
               edge01_11_recv, total_edge, MPI_DOUBLE, edge01_11_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(edge01_11_send, total_edge, MPI_DOUBLE, edge01_11_id, 0,
               edge00_10_recv, total_edge, MPI_DOUBLE, edge00_10_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Sendrecv(edge00_01_send, total_edge, MPI_DOUBLE, edge00_01_id, 0,
               edge10_11_recv, total_edge, MPI_DOUBLE, edge10_11_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(edge10_11_send, total_edge, MPI_DOUBLE, edge10_11_id, 0,
               edge00_01_recv, total_edge, MPI_DOUBLE, edge00_01_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Sendrecv(corner00_send, total_corner, MPI_DOUBLE, corner00_id, 0,
               corner11_recv, total_corner, MPI_DOUBLE, corner11_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(corner11_send, total_corner, MPI_DOUBLE, corner11_id, 0,
               corner00_recv, total_corner, MPI_DOUBLE, corner00_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Sendrecv(corner10_send, total_corner, MPI_DOUBLE, corner10_id, 0,
               corner01_recv, total_corner, MPI_DOUBLE, corner01_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(corner01_send, total_corner, MPI_DOUBLE, corner01_id, 0,
               corner10_recv, total_corner, MPI_DOUBLE, corner10_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  if (id == 0) std::cout << "Stencil time: " << duration.count() << " ns" << std::endl;

  // FFT

  // Local transpose: swap i and j
  for (int i = 0; i < N/r/b; i++) {
    for (int j = 0; j < N/c/b; j++) {
      for (int ii = 0; ii < b; ii++) {
        for (int jj = 0; jj < b; jj++) {
          int src_index = i * (N/c/b * b * b) + j * (b * b) + ii * b + jj;
          int dst_index = j * (N/r/b * b * b) + i * (b * b) + ii * b + jj;
          out[dst_index] = Complex(in[src_index].real(), in[src_index].imag());
        }
      }
    }
  }

  applyFFT(out, N/r/b, (N/r * N/c) / (N/r/b), 1);

  // start = std::chrono::high_resolution_clock::now();

  MPI_Alltoall(out,
               (N/r * N/c) / r,
               MPI_C_DOUBLE_COMPLEX,
               in, 
               (N/r * N/c) / r,
               MPI_C_DOUBLE_COMPLEX,
               row_comm);

  // end = std::chrono::high_resolution_clock::now();
  // duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  // if (id == 0) std::cout << "All to all rows time: " << duration.count() << " ns" << std::endl;

  // Local transpose
  for (int i = 0; i < N/r/b; i++) {
    for (int j = 0; j < N/c/b; j++) {
      for (int ii = 0; ii < b; ii++) {
        for (int jj = 0; jj < b; jj++) {
          int src_index = i * (N/c/b * b * b) + j * (b * b) + ii * b + jj;
          int dst_index = j * (N/r/b * b * b) + i * (b * b) + ii * b + jj;
          out[dst_index] = Complex(in[src_index].real(), in[src_index].imag());
        }
      }
    }
  }

  // Pack data in order to apply fft and local twiddles
  for (int i = 0; i < N/r/b; i++) {
    for (int j = 0; j < N/c/b; j++) {
      int block_offset = (i * b * b * N/c/b) + (j * b * b);
      for (int ii = 0; ii < b; ii++) {
        for (int jj = 0; jj < b; jj++) {
          int row_offset = (i * N/r * b) + (ii * N/r);
          int col_offset = ((j % c) * N/c/b) + ((j / c) * b) + jj;
          in[row_offset + col_offset] = out[block_offset + (ii * b + jj)];
        }
      }
    }
  }

  // Local twiddles -- Double check correctness, specifically in row w/ the scalar on cid
  for (int i = 0; i < N/r; i++) {
    for (int j = 0; j < N/c; j += N/c/b) {
      for (int jj = 0; jj < N/c/b; jj++) {
        double k = (double)jj;  // col
        double l = (double)(j / (N/c/b)) + (double)(cid* b); // row 
        in[(i * N/c) + j + jj] *= std::exp(Complex(0.0, -2*M_PI*k*l/(N/c/b)));
      }
    }
  }

  // if (id == 0) cout<<"After Local Twiddles"<<endl;
  // for (int j = 0; j < p; ++j) {
  //   if (id == j) {
	//     cout<<id<<": ("<<rid<<", "<<cid<<") grp: ("<<row_grp<<", "<<col_grp<<") | ";
	//     for (int i = 0; i < N/r * N/c; ++i)
	//         cout<<"("<<in[i].real()<<", "<<in[i].imag()<<") ";
	//         cout<<endl;
  //   }
  //   MPI_Barrier(MPI_COMM_WORLD);
  // }

  applyFFT(in, N/c/b, 1, N/c/b);

  // Global Twiddles
  for (int i = 0; i < N/r; i++) {
    for (int j = 0; j < N/c; j++) {
      double k = (double)(i % b) + (double)((i / b) * (N/r/b)) + (double)(rid * b);
      double l = (double)j + (double)(cid * (N/c));
      //in[(i * N/c) + j] = Complex(k, l);
      in[(i * N/c) + j] *= std::exp(Complex(0.0, -2*M_PI*k*l/(N/c)));
    }
  }

  // if (id == 0) cout<<"After Global Twiddles"<<endl;
  // for (int j = 0; j < p; ++j) {
  //   if (id == j) {
	//     cout<<id<<": ("<<rid<<", "<<cid<<") grp: ("<<row_grp<<", "<<col_grp<<") | ";
	//     for (int i = 0; i < N/r * N/c; ++i)
	//         cout<<"("<<in[i].real()<<", "<<in[i].imag()<<") ";
	//         cout<<endl;
  //   }
  //   MPI_Barrier(MPI_COMM_WORLD);
  // }

  // Unpack
  for (int i = 0; i < N/r/b; i++) {
    for (int j = 0; j < N/c/b; j++) {
      int block_offset = (i * b * b * N/c/b) + (j * b * b);
      for (int ii = 0; ii < b; ii++) {
        for (int jj = 0; jj < b; jj++) {
          int row_offset = (i * N/r * b) + (ii * N/r);
          int col_offset = ((j % c) * N/c/b) + ((j / c) * b) + jj;
          out[block_offset + (ii * b + jj)] = in[row_offset + col_offset];
        }
      }
    }
  }

  // if (id == 0) cout<<"After Unpack"<<endl;
  // for (int j = 0; j < p; ++j) {
  //   if (id == j) {
	//     cout<<id<<": ("<<rid<<", "<<cid<<") grp: ("<<row_grp<<", "<<col_grp<<") | ";
	//     for (int i = 0; i < N/r * N/c; ++i)
  //       cout<<out[i].real()<<" ";
	//     cout<<endl;
  //   }
  //   MPI_Barrier(MPI_COMM_WORLD);
  // }

  applyFFT(out, N/r/b, (N/r * N/c) / (N/r/b), 1);

  //start = std::chrono::high_resolution_clock::now();

  MPI_Alltoall(out,
               (N/r * N/c) / c,
               MPI_C_DOUBLE_COMPLEX,
               in, 
               (N/r * N/c) / c,
               MPI_C_DOUBLE_COMPLEX,
               col_comm);

  //end = std::chrono::high_resolution_clock::now();
  //duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  //if (id == 0) std::cout << "All to all cols time: " << duration.count() << " ns" << std::endl;

  //if (id == 0) cout<<"After All to all in Cols"<<endl;
  //for (int j = 0; j < p; ++j) {
    //if (id == j) {
	    //cout<<id<<": ("<<rid<<", "<<cid<<") grp: ("<<row_grp<<", "<<col_grp<<") ";
	    //for (int i = 0; i < N/r * N/c; ++i)
	        //cout<<in[i].real()<<" ";
	        //cout<<endl;
    //}
    //MPI_Barrier(MPI_COMM_WORLD);
  //}

  // Pack -- transpose first so we can apply the same packing routine, twiddles, and fft
  for (int i = 0; i < N/r/b; i++) {
    for (int j = 0; j < N/c/b; j++) {
      for (int ii = 0; ii < b; ii++) {
        for (int jj = 0; jj < b; jj++) {
          int src_index = i * (N/c/b * b * b) + j * (b * b) + ii * b + jj;
          int dst_index = j * (N/r/b * b * b) + i * (b * b) + jj * b + ii;
          out[dst_index] = in[src_index];
        }
      }
    }
  }

  for (int i = 0; i < N/r/b; i++) {
    for (int j = 0; j < N/c/b; j++) {
      int block_offset = (i * b * b * N/c/b) + (j * b * b);
      for (int ii = 0; ii < b; ii++) {
        for (int jj = 0; jj < b; jj++) {
          int row_offset = (i * N/r * b) + (ii * N/r);
          int col_offset = ((j % c) * N/c/b) + ((j / c) * b) + jj;
          in[row_offset + col_offset] = out[block_offset + (ii * b + jj)];
        }
      }
    }
  }

  // if (id == 0) cout<<"After Packing 2"<<endl;
  // for (int j = 0; j < p; ++j) {
  //   if (id == j) {
	//     cout<<id<<": ("<<rid<<", "<<cid<<") grp: ("<<row_grp<<", "<<col_grp<<") ";
	//     for (int i = 0; i < N/r * N/c; ++i)
	//         cout<<in[i].real()<<" ";
	//         cout<<endl;
  //   }
  //   MPI_Barrier(MPI_COMM_WORLD);
  // }

  // Local twiddles
  for (int i = 0; i < N/r; i++) {
    for (int j = 0; j < N/c; j += N/c/b) {
      for (int jj = 0; jj < N/c/b; jj++) {
        double k = (double)jj;  // col
        double l = (double)(j / (N/c/b)) + (double)(cid* b); // row 
        in[(i * N/c) + j + jj] *= std::exp(Complex(0.0, -2*M_PI*k*l/(N/c/b)));
      }
    }
  }

  applyFFT(in, N/c/b, 1, N/c/b);

  // Unpack
  for (int i = 0; i < N/r/b; i++) {
    for (int j = 0; j < N/c/b; j++) {
      int block_offset = (i * b * b * N/c/b) + (j * b * b);
      for (int ii = 0; ii < b; ii++) {
        for (int jj = 0; jj < b; jj++) {
          int row_offset = (i * N/r * b) + (ii * N/r);
          int col_offset = ((j % c) * N/c/b) + ((j / c) * b) + jj;
          out[block_offset + (ii * b + jj)] = in[row_offset + col_offset];
        }
      }
    }
  }

  for (int i = 0; i < N/r/b; i++) {
    for (int j = 0; j < N/c/b; j++) {
      for (int ii = 0; ii < b; ii++) {
        for (int jj = 0; jj < b; jj++) {
          int src_index = i * (N/c/b * b * b) + j * (b * b) + ii * b + jj;
          int dst_index = j * (N/r/b * b * b) + i * (b * b) + jj * b + ii;
          in[src_index] = out[dst_index];
        }
      }
    }
  }

  if (id == 0) cout<<"End Result"<<endl;
  for (int j = 0; j < p; ++j) {
    if (id == j) {
	    cout<<id<<": ("<<rid<<", "<<cid<<") grp: ("<<row_grp<<", "<<col_grp<<") ";
	    for (int i = 0; i < N/r * N/c; ++i)
	        cout<<in[i].real()<<" ";
	        cout<<endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Clean up
  free(in);
  free(out);
  free(edge00_01_send);
  free(edge00_01_recv);
  free(edge10_11_send);
  free(edge10_11_recv);
  free(edge00_10_send);
  free(edge00_10_recv);
  free(edge01_11_send);
  free(edge01_11_recv);
  free(corner00_send);
  free(corner00_recv);
  free(corner10_send);
  free(corner10_recv);
  free(corner01_send);
  free(corner01_recv);
  free(corner11_send);
  free(corner11_recv);

  MPI_Finalize();
  
  return 0;
}