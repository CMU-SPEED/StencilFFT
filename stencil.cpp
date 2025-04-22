#include <iostream>
#include <cassert>
#include <mpi.h>
#include <chrono>
#include <stdlib.h>
#include <cmath>
#include <complex>
#include <fftw3.h>

// assume that r = c = d & p = rcd;
// This will be defined at runtime
int r = 0; 
int c = 0;
int d = 0;
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
void applyFFT(Complex *data, int subvector_len, int stride, int dist, int id, size_t size) {
  //TODO: SOME ASSERT HERE about inputs
  //size_t size = (N/r/b) * (N/r/b) * (b*b*b);

  // The number of separate transforms to be performed
  //int howmany = ((N/r/b) * (N/c/b) * (b*b*b)) / subvector_len;
  int howmany = size / subvector_len;

  fftw_complex* input = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * size);
  fftw_complex* output = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * size);

  // ((N/r/b) * (N/c/b) * (b*b*b))
  for (int i = 0; i < size; i++) {
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

  //MPI_Barrier(MPI_COMM_WORLD);

  //auto start = std::chrono::high_resolution_clock::now();

  fftw_execute(plan);

  //MPI_Barrier(MPI_COMM_WORLD);

  //auto end = std::chrono::high_resolution_clock::now();
  //auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  //if (id == 0) std::cout << "FFT: " << duration.count() << " ns" << std::endl;

  for (int i = 0; i < size; i++) {
    data[i] = Complex(output[i][0], output[i][1]);
  }

  fftw_destroy_plan(plan);
  fftw_free(input);
  fftw_free(output);
}

int calc_id(int rid, int cid, int did, int edit_r, int edit_c, int edit_d) {
  int new_rid, new_cid, new_did;

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

  switch (edit_d) {
    case -1:
      new_did = (did == 0) ? (d - 1) : (did - 1);
      break;
    case 1:
      new_did = (did + 1) % d;
      break;
    case 0:
      new_did = did;
      break;
    default:
      assert(false);
  }

  return (new_did * r * c) + (new_rid * c) + new_cid;
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

  // int len;
  // char name[MPI_MAX_PROCESSOR_NAME];
  // MPI_Get_processor_name(name, &len);
  // printf("%s\n", name);
  
  int p, id;
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &p); 

  r = static_cast<int>(std::round(std::cbrt(p)));
  c = r;
  d = r;

  MPI_Comm row_comm, col_comm, dep_comm;

  //check later
  // [0-3 => 0, 4-7 => 1]
  int dep_grp = id % (r * c);
  MPI_Comm_split(MPI_COMM_WORLD, dep_grp, id, &dep_comm);

  // [0,1 => 0, 2,3 => 1 4,5 => 2, 6,7 => 3]
  int row_grp = id / r;
  MPI_Comm_split(MPI_COMM_WORLD, row_grp, id, &row_comm);

  // [0,1,4,5 => 0, 2,3,6,7 => 1]
  int col_grp = (row_grp / r) * c + id % d;
  MPI_Comm_split(MPI_COMM_WORLD, col_grp, id, &col_comm); 

  int rid, cid, did;
  did = id / (r * c);
  rid = (id - (did * r * c)) / c;
  cid = (id - (did * r * c)) % r;

  // 4 x 4 x 4 local data cube that is block cyclic dist in 2x2x2 blocks
  // total of 8 blocks per local processor. 
  Complex *in = (Complex*) malloc(sizeof(Complex) * N/r * N/c * N/d);
  Complex *out = (Complex*) malloc(sizeof(Complex) * N/r * N/c * N/d);

  // precompute constants
  size_t bbb = b * b * b;
  size_t Nrb = N/r/b;
  size_t Ncb = N/c/b;
  size_t Ndb = N/d/b;
  
  //init
  for (size_t i = 0; i < N/r/b; ++i)
    for (size_t j = 0; j < N/c/b; ++j)
      for (size_t k = 0; k < N/d/b; ++k)
	     for (size_t ii = 0; ii < b; ++ii)
	       for (size_t jj = 0; jj < b; ++jj)
	         for (size_t kk = 0; kk < b; ++kk) {
            size_t index = ((k * (bbb * Nrb * Ncb)) +
                           (j * (bbb * Ncb)) + 
                           (i * (bbb))) + (kk*b*b + jj*b + ii);
            in[index] = (did*N*N*b + rid * N * b + cid * b) + //processor offset
                        ((k * N * N * b * d) + (j * N * b * c) + (i * b * r)) + //block offset
                        kk*N*N + jj*N + ii;
            //in[index] = Complex(1.0, 0.0);
  }

  if (id == 0) cout<<"Initial data distribution"<<endl;
  for (int j = 0; j < p; ++j) {
    if (id == j) {
	    cout<<id<<": ("<<rid<<", "<<cid<<", "<<did<<") grp: ("<<row_grp<<", "<<col_grp<<", "<<dep_grp<<") | ";
	    for (int i = 0; i < N/r * N/c * N/d; ++i)
	      cout<<in[i].real()<<" ";
	    cout<<endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Stencil
  int g = 1;
  int b_per_p = (N/r * N/c * N/d) / (b * b * b);
  int side = b * b * g;
  int total_side = b_per_p * side;
  int edge = b * g * g;
  int total_edge = b_per_p * edge;
  int corner = g * g * g;
  int total_corner = b_per_p * corner;

  // above_recv contains the data you recieve from the processor above you
  // below_recv contains the data you recieve from the processor below you
  Complex *row_above_send = (Complex*) malloc(sizeof(Complex) * total_side);
  Complex *row_above_recv = (Complex*) malloc(sizeof(Complex) * total_side);
  Complex *row_below_send = (Complex*) malloc(sizeof(Complex) * total_side);
  Complex *row_below_recv = (Complex*) malloc(sizeof(Complex) * total_side);
  int row_above_id = calc_id(rid, cid, did, -1, 0, 0);
  int row_below_id = calc_id(rid, cid, did, 1, 0, 0);

  Complex *col_left_send = (Complex*) malloc(sizeof(Complex) * total_side);
  Complex *col_left_recv = (Complex*) malloc(sizeof(Complex) * total_side);
  Complex *col_right_send = (Complex*) malloc(sizeof(Complex) * total_side);
  Complex *col_right_recv = (Complex*) malloc(sizeof(Complex) * total_side);
  int col_left_id = calc_id(rid, cid, did, 0, -1, 0);
  int col_right_id = calc_id(rid, cid, did, 0, 1, 0);

  Complex *dep_back_send = (Complex*) malloc(sizeof(Complex) * total_side);
  Complex *dep_back_recv = (Complex*) malloc(sizeof(Complex) * total_side);
  Complex *dep_front_send = (Complex*) malloc(sizeof(Complex) * total_side);
  Complex *dep_front_recv = (Complex*) malloc(sizeof(Complex) * total_side);
  int dep_back_id = calc_id(rid, cid, did, 0, 0, -1);
  int dep_front_id = calc_id(rid, cid, did, 0, 0, 1);

  // Pack side data for stencil
  // for (size_t i = 0; i < b_per_p; i++) {
  //   size_t top_block_offset = i * bbb;
  //   size_t bottom_block_offset = top_block_offset + (b * (b - g) * (b - g));

  //   for (size_t j = 0; j < b; j++) {
  //     for (size_t ii = 0; ii < g; ii++) {
  //       for (size_t jj = 0; jj < b; jj++) {
  //         row_above_send[(i * side) + 
  //                        (j * g * b) +
  //                        (ii * g) + jj] = in[top_block_offset +
  //                                           (j * b * b) + 
  //                                           (ii * b) + jj];

  //         row_below_send[(i * side) +
  //                        (j * g * b) +
  //                        (ii * g) + jj] = in[bottom_block_offset + 
  //                                           (j * b * b) + 
  //                                           (ii * b) + jj];

  //         col_left_send[(i * side) +
  //                       (j * g * b) +
  //                       (ii * g) + jj] = in[top_block_offset +
  //                                         (j * b * b) +
  //                                         (jj * b) + ii];

  //         col_right_send[(i * side) +
  //                        (j * g * b) +
  //                        (ii * g) + jj] = in[top_block_offset +
  //                                           (b - g) + // offset to get right col
  //                                           (j * b * b) +
  //                                           (jj * b) + ii];
  //       }
  //     }
  //   }

  //   for (size_t j = 0; j < g; j++) {
  //     for (size_t ii = 0; ii < b; ii++) {
  //       for (size_t jj = 0; jj < b; jj++) {
  //         dep_back_send[(i * side) +
  //                       (j * b * b) +
  //                       (ii * b) + jj] = in[top_block_offset +
  //                                           (j * b * b) +
  //                                           (ii * b) + jj];

  //         dep_front_send[(i * side) +
  //                        (j * b * b) +
  //                        (ii * b) + jj] = in[top_block_offset +
  //                                           (b * b * (b - g)) + // offset to get front side
  //                                           (j * b * b) +
  //                                           (ii * b) + jj];
  //       }
  //     }
  //   }
  // }

  // MPI_Barrier(MPI_COMM_WORLD);

  // if (id == 0) cout<<"Side"<<endl;
  // for (int j = 0; j < p; ++j) {
  //   if (id == j) {
	//     cout<<id<<": ("<<rid<<", "<<cid<<") grp: ("<<row_grp<<", "<<col_grp<<") ";
	//     for (int i = 0; i < total_side; ++i)
	//         cout<<dep_front_send[i].real()<<" ";
	//         cout<<endl;
  //   }
  //   MPI_Barrier(MPI_COMM_WORLD);
  // }

  // Top right edge and top left edge
  Complex *edge110_111_send = (Complex*) malloc(sizeof(Complex) * total_edge);
  Complex *edge110_111_recv = (Complex*) malloc(sizeof(Complex) * total_edge);
  Complex *edge010_011_send = (Complex*) malloc(sizeof(Complex) * total_edge);
  Complex *edge010_011_recv = (Complex*) malloc(sizeof(Complex) * total_edge);
  int edge110_111_id = calc_id(rid, cid, did, 1, 1, 0);
  int edge010_011_id = calc_id(rid, cid, did, -1, 1, 0);

  // Bottom right edge and bottom left edge
  Complex *edge100_101_send = (Complex*) malloc(sizeof(Complex) * total_edge);
  Complex *edge100_101_recv = (Complex*) malloc(sizeof(Complex) * total_edge);
  Complex *edge000_001_send = (Complex*) malloc(sizeof(Complex) * total_edge);
  Complex *edge000_001_recv = (Complex*) malloc(sizeof(Complex) * total_edge);
  int edge100_101_id = calc_id(rid, cid, did, 1, -1, 0);
  int edge000_001_id = calc_id(rid, cid, did, -1, -1, 0);

  // Top front edge and top back edge
  Complex *edge011_111_send = (Complex*) malloc(sizeof(Complex) * total_edge);
  Complex *edge011_111_recv = (Complex*) malloc(sizeof(Complex) * total_edge);
  Complex *edge010_110_send = (Complex*) malloc(sizeof(Complex) * total_edge);
  Complex *edge010_110_recv = (Complex*) malloc(sizeof(Complex) * total_edge);
  int edge011_111_id = calc_id(rid, cid, did, 0, 1, 1);
  int edge010_110_id = calc_id(rid, cid, did, 0, 1, -1);

  // Bottom front edge and bottom back edge
  Complex *edge001_101_send = (Complex*) malloc(sizeof(Complex) * total_edge);
  Complex *edge001_101_recv = (Complex*) malloc(sizeof(Complex) * total_edge);
  Complex *edge000_100_send = (Complex*) malloc(sizeof(Complex) * total_edge);
  Complex *edge000_100_recv = (Complex*) malloc(sizeof(Complex) * total_edge);
  int edge001_101_id = calc_id(rid, cid, did, 0, -1, 1);
  int edge000_100_id = calc_id(rid, cid, did, 0, -1, -1);

  // Right back edge and right front edge
  Complex *edge100_110_send = (Complex*) malloc(sizeof(Complex) * total_edge);
  Complex *edge100_110_recv = (Complex*) malloc(sizeof(Complex) * total_edge);
  Complex *edge101_111_send = (Complex*) malloc(sizeof(Complex) * total_edge);
  Complex *edge101_111_recv = (Complex*) malloc(sizeof(Complex) * total_edge);
  int edge100_110_id = calc_id(rid, cid, did, 1, 0, -1);
  int edge101_111_id = calc_id(rid, cid, did, 1, 0, 1);

  // Left back edge and left front edge
  Complex *edge000_010_send = (Complex*) malloc(sizeof(Complex) * total_edge);
  Complex *edge000_010_recv = (Complex*) malloc(sizeof(Complex) * total_edge);
  Complex *edge001_011_send = (Complex*) malloc(sizeof(Complex) * total_edge);
  Complex *edge001_011_recv = (Complex*) malloc(sizeof(Complex) * total_edge);
  int edge000_010_id = calc_id(rid, cid, did, -1, 0, -1);
  int edge001_011_id = calc_id(rid, cid, did, -1, 0, 1);

  // Pack data for edges
  // for (size_t i = 0; i < b_per_p; i++) {
  //   size_t top_block_offset = i * bbb;
  //   size_t bottom_block_offset = top_block_offset + (b * (b - g) * (b - g));

  //   for (size_t j = 0; j < b; j++) {
  //     for (size_t ii = 0; ii < g; ii++) {
  //       for (size_t jj = 0; jj < g; jj++) {
  //         // Top left edge
  //         edge010_011_send[(i * edge) + (j * g * g) + (ii * g) + jj] = in[top_block_offset +
  //                                                                         (j * b * b) +
  //                                                                         (ii * b) + jj];

  //         // Top right edge
  //         edge110_111_send[(i * edge) + (j * g * g) + (ii * g) + jj] = in[top_block_offset + (b - g) +
  //                                                                         (j * b * b) +
  //                                                                         (ii * b) + jj];

  //         // Bottom left edge
  //         edge000_001_send[(i * edge) + (j * g * g) + (ii * g) + jj] = in[bottom_block_offset +
  //                                                                         (j * b * b) +
  //                                                                         (ii * b) + jj];

  //         // Bottom right edge
  //         edge100_101_send[(i * edge) + (j * g * g) + (ii * g) + jj] = in[bottom_block_offset + (b - g) +
  //                                                                         (j * b * b) +
  //                                                                         (ii * b) + jj];

  //         // -- TODO --> review the above and bottom_offset

  //         // Top back edge
  //         edge010_110_send[(i * edge) + (j * g * g) + (ii * g) + jj] = in[top_block_offset + 
  //                                                                         j + (ii * b * b) + (jj * b)];

  //         // Top front edge
  //         edge011_111_send[(i * edge) + (j * g * g) + (ii * g) + jj] = in[top_block_offset + 
  //                                                                         (b * b * (b - g)) + 
  //                                                                         j + (ii * b * b) + (jj * b)];

  //         // Bottom back edge
  //         edge000_100_send[(i * edge) + (j * g * g) + (ii * g) + jj] = in[top_block_offset +
  //                                                                         (b * (b - g)) +
  //                                                                         j + (ii * b * b) + (jj * b)];

  //         // Bottom front edge
  //         //edge001_101_send[(i * edge) + (j * g * g) + (ii * g) + jj] = in[top_block_offset +
  //                                                                         //(b * b * (b - g)) +
  //                                                                         //j + (ii * b * b) + (jj * b)];
  //       }
  //     }
  //   }
  // }

  // MPI_Barrier(MPI_COMM_WORLD);

  // if (id == 0) cout<<"Side"<<endl;
  // for (int j = 0; j < p; ++j) {
  //   if (id == j) {
	//     cout<<id<<": ("<<rid<<", "<<cid<<") grp: ("<<row_grp<<", "<<col_grp<<") ";
	//     for (int i = 0; i < total_edge; ++i)
	//         cout<<edge001_101_send[i].real()<<" ";
	//         cout<<endl;
  //   }
  //   MPI_Barrier(MPI_COMM_WORLD);
  // }

  // Top front right corner and bottom back left corner
  Complex *corner000_send = (Complex*) malloc(sizeof(Complex) * total_corner);
  Complex *corner000_recv = (Complex*) malloc(sizeof(Complex) * total_corner);
  Complex *corner111_send = (Complex*) malloc(sizeof(Complex) * total_corner);
  Complex *corner111_recv = (Complex*) malloc(sizeof(Complex) * total_corner);
  int corner000_id = calc_id(rid, cid, did, -1, -1, -1);
  int corner111_id = calc_id(rid, cid, did, 1, 1, 1);

  // Bottom front right corner and top back left corner
  Complex *corner010_send = (Complex*) malloc(sizeof(Complex) * total_corner);
  Complex *corner010_recv = (Complex*) malloc(sizeof(Complex) * total_corner);
  Complex *corner101_send = (Complex*) malloc(sizeof(Complex) * total_corner);
  Complex *corner101_recv = (Complex*) malloc(sizeof(Complex) * total_corner);
  int corner010_id = calc_id(rid, cid, did, -1, 1, -1);
  int corner101_id = calc_id(rid, cid, did, 1, -1, 1);

  // Top front left corner and bottom back right corner
  Complex *corner011_send = (Complex*) malloc(sizeof(Complex) * total_corner);
  Complex *corner011_recv = (Complex*) malloc(sizeof(Complex) * total_corner);
  Complex *corner100_send = (Complex*) malloc(sizeof(Complex) * total_corner);
  Complex *corner100_recv = (Complex*) malloc(sizeof(Complex) * total_corner);
  int corner011_id = calc_id(rid, cid, did, -1, 1, 1);
  int corner100_id = calc_id(rid, cid, did, 1, -1, -1);

  // Bottom front left corner and top back right corner
  Complex *corner001_send = (Complex*) malloc(sizeof(Complex) * total_corner);
  Complex *corner001_recv = (Complex*) malloc(sizeof(Complex) * total_corner);
  Complex *corner110_send = (Complex*) malloc(sizeof(Complex) * total_corner);
  Complex *corner110_recv = (Complex*) malloc(sizeof(Complex) * total_corner);
  int corner001_id = calc_id(rid, cid, did, -1, -1, 1);
  int corner110_id = calc_id(rid, cid, did, 1, 1, -1);

  // auto start = std::chrono::high_resolution_clock::now();

  MPI_Sendrecv(row_above_send, total_side, MPI_C_DOUBLE_COMPLEX, row_above_id, 0,
               row_below_recv, total_side, MPI_C_DOUBLE_COMPLEX, row_below_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(row_below_send, total_side, MPI_C_DOUBLE_COMPLEX, row_below_id, 0,
               row_above_recv, total_side, MPI_C_DOUBLE_COMPLEX, row_above_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  
  MPI_Sendrecv(col_left_send, total_side, MPI_C_DOUBLE_COMPLEX, col_left_id, 0,
               col_right_recv, total_side, MPI_C_DOUBLE_COMPLEX, col_right_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(col_right_send, total_side, MPI_C_DOUBLE_COMPLEX, col_right_id, 0,
               col_left_recv, total_side, MPI_C_DOUBLE_COMPLEX, col_left_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  
  MPI_Sendrecv(dep_back_send, total_side, MPI_C_DOUBLE_COMPLEX, dep_back_id, 0,
               dep_front_recv, total_side, MPI_C_DOUBLE_COMPLEX, dep_front_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(dep_front_send, total_side, MPI_C_DOUBLE_COMPLEX, dep_front_id, 0,
               dep_back_recv, total_side, MPI_C_DOUBLE_COMPLEX, dep_back_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Sendrecv(edge110_111_send, total_corner, MPI_C_DOUBLE_COMPLEX, edge110_111_id, 0, 
               edge010_011_recv, total_corner, MPI_C_DOUBLE_COMPLEX, edge010_011_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(edge010_011_send, total_corner, MPI_C_DOUBLE_COMPLEX, edge010_011_id, 0,
               edge110_111_recv, total_corner, MPI_C_DOUBLE_COMPLEX, edge110_111_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Sendrecv(edge100_101_send, total_corner, MPI_C_DOUBLE_COMPLEX, edge100_101_id, 0, 
               edge000_001_recv, total_corner, MPI_C_DOUBLE_COMPLEX, edge000_001_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(edge000_001_send, total_corner, MPI_C_DOUBLE_COMPLEX, edge000_001_id, 0,
               edge100_101_recv, total_corner, MPI_C_DOUBLE_COMPLEX, edge100_101_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Sendrecv(edge011_111_send, total_corner, MPI_C_DOUBLE_COMPLEX, edge011_111_id, 0, 
               edge010_110_recv, total_corner, MPI_C_DOUBLE_COMPLEX, edge010_110_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(edge010_110_send, total_corner, MPI_C_DOUBLE_COMPLEX, edge010_110_id, 0,
               edge011_111_recv, total_corner, MPI_C_DOUBLE_COMPLEX, edge011_111_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Sendrecv(edge001_101_send, total_corner, MPI_C_DOUBLE_COMPLEX, edge001_101_id, 0, 
               edge000_100_recv, total_corner, MPI_C_DOUBLE_COMPLEX, edge000_100_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(edge000_100_send, total_corner, MPI_C_DOUBLE_COMPLEX, edge000_100_id, 0,
               edge001_101_recv, total_corner, MPI_C_DOUBLE_COMPLEX, edge001_101_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Sendrecv(edge100_110_send, total_corner, MPI_C_DOUBLE_COMPLEX, edge100_110_id, 0, 
               edge101_111_recv, total_corner, MPI_C_DOUBLE_COMPLEX, edge101_111_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(edge101_111_send, total_corner, MPI_C_DOUBLE_COMPLEX, edge101_111_id, 0,
               edge100_110_recv, total_corner, MPI_C_DOUBLE_COMPLEX, edge100_110_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Sendrecv(edge000_010_send, total_corner, MPI_C_DOUBLE_COMPLEX, edge000_010_id, 0, 
               edge001_011_recv, total_corner, MPI_C_DOUBLE_COMPLEX, edge001_011_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(edge001_011_send, total_corner, MPI_C_DOUBLE_COMPLEX, edge001_011_id, 0,
               edge000_010_recv, total_corner, MPI_C_DOUBLE_COMPLEX, edge000_010_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Sendrecv(corner000_send, total_corner, MPI_C_DOUBLE_COMPLEX, corner000_id, 0, 
               corner111_recv, total_corner, MPI_C_DOUBLE_COMPLEX, corner111_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(corner111_send, total_corner, MPI_C_DOUBLE_COMPLEX, corner111_id, 0,
               corner000_recv, total_corner, MPI_C_DOUBLE_COMPLEX, corner000_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  
  MPI_Sendrecv(corner010_send, total_corner, MPI_C_DOUBLE_COMPLEX, corner010_id, 0, 
               corner101_recv, total_corner, MPI_C_DOUBLE_COMPLEX, corner101_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(corner101_send, total_corner, MPI_C_DOUBLE_COMPLEX, corner101_id, 0,
               corner010_recv, total_corner, MPI_C_DOUBLE_COMPLEX, corner010_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Sendrecv(corner011_send, total_corner, MPI_C_DOUBLE_COMPLEX, corner011_id, 0, 
               corner100_recv, total_corner, MPI_C_DOUBLE_COMPLEX, corner100_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(corner100_send, total_corner, MPI_C_DOUBLE_COMPLEX, corner100_id, 0,
               corner011_recv, total_corner, MPI_C_DOUBLE_COMPLEX, corner011_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Sendrecv(corner001_send, total_corner, MPI_C_DOUBLE_COMPLEX, corner001_id, 0, 
               corner110_recv, total_corner, MPI_C_DOUBLE_COMPLEX, corner110_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(corner110_send, total_corner, MPI_C_DOUBLE_COMPLEX, corner110_id, 0,
               corner001_recv, total_corner, MPI_C_DOUBLE_COMPLEX, corner001_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  // auto end = std::chrono::high_resolution_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  // if (id == 0) std::cout << "Stencil time: " << duration.count() << " ns" << std::endl;

  // FFT

  // Local transpose: 
  for (size_t i = 0; i < N/r/b; ++i)
    for (size_t j = 0; j < N/c/b; ++j)
      for (size_t k = 0; k < N/d/b; ++k)
	     for (size_t ii = 0; ii < b; ++ii)
	       for (size_t jj = 0; jj < b; ++jj)
	         for (size_t kk = 0; kk < b; ++kk) {
            size_t src_index = (i * Ncb * Ndb * bbb) +
                               (j * Ndb * bbb) +
                               (k * bbb) + (ii * b * b) + (jj * b) + kk;
            size_t dst_index = (i * Ncb * Ndb * bbb) +
                               (k * Ndb * bbb) +
                               (j * bbb) + (ii * b * b) + (jj * b) + kk;
            out[dst_index] = in[src_index];
  }

  for (size_t i = 0; i < Ndb; i++) {
    size_t offset = i * Nrb * Ncb * bbb;
    applyFFT(out + offset, N/r/b, (N/r)*b*b, 1, id, (Nrb * Ncb * bbb));
  }

  // Pack before All to All
  for (size_t i = 0; i < r; i++) {
    for (size_t j = 0; j < Ndb; j++) {
      size_t tmp = j * (Nrb * Ncb * bbb);
      size_t src = i * ((Nrb * Ncb * bbb)/r);
      size_t dest = i * (((Nrb * Ncb * bbb)/r) * Ndb) + j * ((Nrb * Ncb * bbb)/r);
      for (size_t k = 0; k < (Nrb * Ncb * bbb)/r; k++) {
        in[dest + k] = out[(tmp + src) + k];
      }
    }
  }
  
  // start = std::chrono::high_resolution_clock::now();

  MPI_Alltoall(in,
               (N/r * N/c * N/d) / r,
               MPI_C_DOUBLE_COMPLEX,
               out, 
               (N/r * N/c * N/d) / r,
               MPI_C_DOUBLE_COMPLEX,
               row_comm);

  // Unpack all to all
  for (size_t i = 0; i < r; i++) {
    for (size_t j = 0; j < Ndb; j++) {
      size_t tmp = j * (Nrb * Ncb * bbb);
      size_t src = i * ((Nrb * Ncb * bbb)/r);
      size_t dest = i * (((Nrb * Ncb * bbb)/r) * Ndb) + j * ((Nrb * Ncb * bbb)/r);
      for (size_t k = 0; k < (Nrb * Ncb * bbb)/r; k++) {
        in[(tmp + src) + k] = out[dest + k];
      }
    }
  }

  // end = std::chrono::high_resolution_clock::now();
  // duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  // if (id == 0) std::cout << "All to all rows time: " << duration.count() << " ns" << std::endl;

  //Local transpose
  for (size_t i = 0; i < N/r/b; ++i)
    for (size_t j = 0; j < N/c/b; ++j)
      for (size_t k = 0; k < N/d/b; ++k)
	     for (size_t ii = 0; ii < b; ++ii)
	       for (size_t jj = 0; jj < b; ++jj)
	         for (size_t kk = 0; kk < b; ++kk) {
            size_t src_index = (i * Ncb * Ndb * bbb) +
                               (j * Ndb * bbb) +
                               (k * bbb) + (ii * b * b) + (jj * b) + kk;
            size_t dst_index = (i * Ncb * Ndb * bbb) +
                               (k * Ndb * bbb) +
                               (j * bbb) + (ii * b * b) + (jj * b) + kk;
            out[dst_index] = in[src_index];
  }

  // Pack so we can apply fft
  for (size_t i = 0; i < N/r/b; ++i)
    for (size_t j = 0; j < N/c/b; ++j)
      for (size_t k = 0; k < N/d/b; ++k) {
        for (size_t ii = 0; ii < b; ++ii)
	        for (size_t jj = 0; jj < b; ++jj)
	          for (size_t kk = 0; kk < b; ++kk) {
              size_t j_shuffle = (j % c) * ((N/c/b)/c) + (j/c);
              size_t i_shuffle = (i % r) * ((N/r/b)/r) + (i/r);
              size_t k_shuffle = (k % d) * ((N/d/b)/d) + (k/d);
              size_t gx = j_shuffle * b + jj;
              size_t gy = i_shuffle * b + ii;
              size_t gz = k_shuffle * b + kk;
              size_t dst_index = gz * (N/r) * (N/c) + gy * (N/c) + gx;
              size_t src_index = (k * Ncb * Ndb * bbb) +
                                 (i * Ndb * bbb) +
                                 (j * bbb) + (kk * b * b) + (ii * b) + jj;
              in[dst_index] = out[src_index];
           }
      }

  //Twiddles
  for (size_t i = 0; i < N/d; i++) {
    for (size_t j = 0; j < N/r; j++) {
      for (size_t k = 0; k < N/c; k += N/c/b) {
        for (size_t kk = 0; kk < N/c/b; kk++) {
          double foo = (double)(cid * ((N/(b*c))/c)) + ((k + kk) / (b * c));
          double l = (k + kk) % (b * c);;
          size_t index = i * ((N/r) * (N/c)) + j * (N/c) + k + kk;
          in[index] *= std::exp(Complex(0.0, -2*M_PI*foo*l/(N))); 
        }
      }
    }
  }

  for (size_t i = 0; i < Ndb; i++) {
    size_t offset = i * Nrb * Ncb * bbb;
    applyFFT(in + offset, b*c, 1, b*c, id, (Nrb * Ncb * bbb));
  }

  for (size_t i = 0; i < N/r/b; ++i)
    for (size_t j = 0; j < N/c/b; ++j)
      for (size_t k = 0; k < N/d/b; ++k) {
        for (size_t ii = 0; ii < b; ++ii)
	        for (size_t jj = 0; jj < b; ++jj)
	          for (size_t kk = 0; kk < b; ++kk) {
              size_t j_shuffle = (j % c) * ((N/c/b)/c) + (j/c);
              size_t i_shuffle = (i % r) * ((N/r/b)/r) + (i/r);
              size_t k_shuffle = (k % d) * ((N/d/b)/d) + (k/d);
              size_t gx = j_shuffle * b + jj;
              size_t gy = i_shuffle * b + ii;
              size_t gz = k_shuffle * b + kk;
              size_t dst_index = gz * (N/r) * (N/c) + gy * (N/c) + gx;
              size_t src_index = (k * Ncb * Ndb * bbb) +
                                 (i * Ndb * bbb) +
                                 (j * bbb) + (kk * b * b) + (ii * b) + jj;
              out[src_index] = in[dst_index];
           }
      }

  for (size_t i = 0; i < Ndb; i++) {
    size_t offset = i * Nrb * Ncb * bbb;
    applyFFT(out + offset, N/r/b, (N/r)*b*b, 1, id, (Nrb * Ncb * bbb));
  }

  // Pack before All to All
  for (size_t i = 0; i < c; i++) {
    for (size_t j = 0; j < Ndb; j++) {
      size_t tmp = j * (Nrb * Ncb * bbb);
      size_t src = i * ((Nrb * Ncb * bbb)/c);
      size_t dest = i * (((Nrb * Ncb * bbb)/c) * Ndb) + j * ((Nrb * Ncb * bbb)/c);
      for (size_t k = 0; k < (Nrb * Ncb * bbb)/c; k++) {
        in[dest + k] = out[(tmp + src) + k];
      }
    }
  }

  // start = std::chrono::high_resolution_clock::now();

  MPI_Alltoall(in,
               (N/r * N/c * N/d) / c,
               MPI_C_DOUBLE_COMPLEX,
               out, 
               (N/r * N/c * N/d) / c,
               MPI_C_DOUBLE_COMPLEX,
               col_comm);

  // MPI_Barrier(MPI_COMM_WORLD);

  // Unpack after All to All
  for (size_t i = 0; i < c; i++) {
    for (size_t j = 0; j < Ndb; j++) {
      size_t tmp = j * (Nrb * Ncb * bbb);
      size_t src = i * ((Nrb * Ncb * bbb)/c);
      size_t dest = i * (((Nrb * Ncb * bbb)/c) * Ndb) + j * ((Nrb * Ncb * bbb)/c);
      for (size_t k = 0; k < (Nrb * Ncb * bbb)/c; k++) {
        in[(tmp + src) + k] = out[dest + k];
      }
    }
  }

  // end = std::chrono::high_resolution_clock::now();
  // duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  // if (id == 0) std::cout << "All to all cols time: " << duration.count() << " ns" << std::endl;


  // Transpose before packing!
  for (size_t i = 0; i < N/r/b; ++i)
    for (size_t j = 0; j < N/c/b; ++j)
      for (size_t k = 0; k < N/d/b; ++k)
        for (size_t ii = 0; ii < b; ++ii)
	        for (size_t jj = 0; jj < b; ++jj)
	          for (size_t kk = 0; kk < b; ++kk) {
              size_t src_index = (i * Ncb * Ndb * bbb) +
                                 (j * Ndb * bbb) +
                                 (k * bbb) + (ii * b * b) + (jj * b) + kk;
              size_t dst_index = (i * Ncb * Ndb * bbb) +
                                 (k * Ndb * bbb) +
                                 (j * bbb) + (ii * b * b) + (kk * b) + jj;
              out[dst_index] = in[src_index];
           }
  
  // Packing
  for (size_t i = 0; i < N/r/b; ++i)
    for (size_t j = 0; j < N/c/b; ++j)
      for (size_t k = 0; k < N/d/b; ++k) {
        for (size_t ii = 0; ii < b; ++ii)
	        for (size_t jj = 0; jj < b; ++jj)
	          for (size_t kk = 0; kk < b; ++kk) {
              size_t j_shuffle = (j % c) * ((N/c/b)/c) + (j/c);
              size_t i_shuffle = (i % r) * ((N/r/b)/r) + (i/r);
              size_t k_shuffle = (k % d) * ((N/d/b)/d) + (k/d);
              size_t gx = j_shuffle * b + jj;
              size_t gy = i_shuffle * b + ii;
              size_t gz = k_shuffle * b + kk;
              size_t dst_index = gz * (N/r) * (N/c) + gy * (N/c) + gx;
              size_t src_index = (k * Ncb * Ndb * bbb) +
                                 (i * Ndb * bbb) +
                                 (j * bbb) + (kk * b * b) + (ii * b) + jj;
              in[dst_index] = out[src_index];
           }
      }

  //Twiddles
  for (size_t i = 0; i < N/d; i++) {
    for (size_t j = 0; j < N/r; j++) {
      for (size_t k = 0; k < N/c; k += N/c/b) {
        for (size_t kk = 0; kk < N/c/b; kk++) {
          double foo = (double)(rid * ((N/(b*c))/c)) + ((k + kk) / (b * c));
          double l = (k + kk) % (b * c);;
          size_t index = i * ((N/r) * (N/c)) + j * (N/c) + k + kk;
          in[index] *= std::exp(Complex(0.0, -2*M_PI*foo*l/(N))); 
        }
      }
    }
  }

  for (size_t i = 0; i < Ndb; i++) {
    size_t offset = i * Nrb * Ncb * bbb;
    applyFFT(in + offset, b*c, 1, b*c, id, (Nrb * Ncb * bbb));
  }

  for (size_t i = 0; i < N/r/b; ++i)
    for (size_t j = 0; j < N/c/b; ++j)
      for (size_t k = 0; k < N/d/b; ++k) {
        for (size_t ii = 0; ii < b; ++ii)
	        for (size_t jj = 0; jj < b; ++jj)
	          for (size_t kk = 0; kk < b; ++kk) {
              size_t j_shuffle = (j % c) * ((N/c/b)/c) + (j/c);
              size_t i_shuffle = (i % r) * ((N/r/b)/r) + (i/r);
              size_t k_shuffle = (k % d) * ((N/d/b)/d) + (k/d);
              size_t gx = j_shuffle * b + jj;
              size_t gy = i_shuffle * b + ii;
              size_t gz = k_shuffle * b + kk;
              size_t dst_index = gz * (N/r) * (N/c) + gy * (N/c) + gx;
              size_t src_index = (k * Ncb * Ndb * bbb) +
                                 (i * Ndb * bbb) +
                                 (j * bbb) + (kk * b * b) + (ii * b) + jj;
              out[src_index] = in[dst_index];
           }
      }

  // Unpack
  for (size_t i = 0; i < N/r/b; ++i)
    for (size_t j = 0; j < N/c/b; ++j)
      for (size_t k = 0; k < N/d/b; ++k)
        for (size_t ii = 0; ii < b; ++ii)
	        for (size_t jj = 0; jj < b; ++jj)
	          for (size_t kk = 0; kk < b; ++kk) {
              size_t src_index = (i * Ncb * Ndb * bbb) +
                                 (j * Ndb * bbb) +
                                 (k * bbb) + (ii * b * b) + (jj * b) + kk;
              size_t dst_index = (i * Ncb * Ndb * bbb) +
                                 (k * Ndb * bbb) +
                                 (j * bbb) + (ii * b * b) + (kk * b) + jj;
              in[src_index] = out[dst_index];
           }

  //applyFFT(in, 2, ((N/r) * (N/c) * (N/d)) / 2, 1, id, ((N/r) * (N/c) * (N/d)));
  applyFFT(in, Ndb, ((N/r) * (N/c) * (N/d)) / Ndb, 1, id, ((N/r) * (N/c) * (N/d)));

  // start = std::chrono::high_resolution_clock::now();

  MPI_Alltoall(in,
               (N/r * N/c * N/d) / d,
               MPI_C_DOUBLE_COMPLEX,
               out, 
               (N/r * N/c * N/d) / d,
               MPI_C_DOUBLE_COMPLEX,
               dep_comm);

  // end = std::chrono::high_resolution_clock::now();
  // duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  // if (id == 0) std::cout << "All to all depth time: " << duration.count() << " ns" << std::endl;

  // Transpose before packing
  for (size_t i = 0; i < N/r/b; ++i)
    for (size_t j = 0; j < N/c/b; ++j)
      for (size_t k = 0; k < N/d/b; ++k)
        for (size_t ii = 0; ii < b; ++ii)
	        for (size_t jj = 0; jj < b; ++jj)
	          for (size_t kk = 0; kk < b; ++kk) {
              size_t src_index = (i * Ncb * Ndb * bbb) +
                                 (j * Ndb * bbb) +
                                 (k * bbb) + (ii * b * b) + (jj * b) + kk;
              size_t dst_index = (k * Ncb * Ndb * bbb) +
                                 (j * Ndb * bbb) +
                                 (i * bbb) + (kk * b * b) + (jj * b) + ii;
              in[dst_index] = out[src_index];
           }

  // Packing
  for (size_t i = 0; i < N/r/b; ++i)
    for (size_t j = 0; j < N/c/b; ++j)
      for (size_t k = 0; k < N/d/b; ++k) {
        for (size_t ii = 0; ii < b; ++ii)
	        for (size_t jj = 0; jj < b; ++jj)
	          for (size_t kk = 0; kk < b; ++kk) {
              size_t j_shuffle = (j % c) * ((N/c/b)/c) + (j/c);
              size_t i_shuffle = (i % r) * ((N/r/b)/r) + (i/r);
              size_t k_shuffle = (k % d) * ((N/d/b)/d) + (k/d);
              size_t gx = j_shuffle * b + jj;
              size_t gy = i_shuffle * b + ii;
              size_t gz = k_shuffle * b + kk;
              size_t dst_index = gz * (N/r) * (N/c) + gy * (N/c) + gx;
              size_t src_index = (k * Ncb * Ndb * bbb) +
                                 (i * Ndb * bbb) +
                                 (j * bbb) + (kk * b * b) + (ii * b) + jj;
              out[dst_index] = in[src_index];
           }
      }

  //Twiddles
  for (size_t i = 0; i < N/d; i++) {
    for (size_t j = 0; j < N/r; j++) {
      for (size_t k = 0; k < N/c; k += N/c/b) {
        for (size_t kk = 0; kk < N/c/b; kk++) {
          double foo = (double)(did * ((N/(b*c))/c)) + ((k + kk) / (b * c));
          double l = (k + kk) % (b * c);;
          size_t index = i * ((N/r) * (N/c)) + j * (N/c) + k + kk;
          out[index] *= std::exp(Complex(0.0, -2*M_PI*foo*l/(N))); 
        }
      }
    }
  }

  for (size_t i = 0; i < Ndb; i++) {
    size_t offset = i * Nrb * Ncb * bbb;
    applyFFT(out + offset, b*c, 1, b*c, id, (Nrb * Ncb * bbb));
  }

  for (size_t i = 0; i < N/r/b; ++i)
    for (size_t j = 0; j < N/c/b; ++j)
      for (size_t k = 0; k < N/d/b; ++k) {
        for (size_t ii = 0; ii < b; ++ii)
	        for (size_t jj = 0; jj < b; ++jj)
	          for (size_t kk = 0; kk < b; ++kk) {
              size_t j_shuffle = (j % c) * ((N/c/b)/c) + (j/c);
              size_t i_shuffle = (i % r) * ((N/r/b)/r) + (i/r);
              size_t k_shuffle = (k % d) * ((N/d/b)/d) + (k/d);
              size_t gx = j_shuffle * b + jj;
              size_t gy = i_shuffle * b + ii;
              size_t gz = k_shuffle * b + kk;
              size_t dst_index = gz * (N/r) * (N/c) + gy * (N/c) + gx;
              size_t src_index = (k * Ncb * Ndb * bbb) +
                                 (i * Ndb * bbb) +
                                 (j * bbb) + (kk * b * b) + (ii * b) + jj;
              in[src_index] = out[dst_index];
           }
      }

  for (size_t i = 0; i < N/r/b; ++i)
    for (size_t j = 0; j < N/c/b; ++j)
      for (size_t k = 0; k < N/d/b; ++k)
        for (size_t ii = 0; ii < b; ++ii)
	        for (size_t jj = 0; jj < b; ++jj)
	          for (size_t kk = 0; kk < b; ++kk) {
              size_t src_index = (i * Ncb * Ndb * bbb) +
                                 (j * Ndb * bbb) +
                                 (k * bbb) + (ii * b * b) + (jj * b) + kk;
              size_t dst_index = (i * Ncb * Ndb * bbb) +
                                 (k * Ndb * bbb) +
                                 (j * bbb) + (ii * b * b) + (kk * b) + jj;
              out[src_index] = in[dst_index];
           }

  if (id == 0) cout<<"The End"<<endl;
  for (int j = 0; j < p; ++j) {
    if (id == j) {
	    cout<<id<<": ("<<rid<<", "<<cid<<", "<<did<<") grp: ("<<row_grp<<", "<<col_grp<<", "<<dep_grp<<") ";
	    for (int i = 0; i < N/r * N/c * N/d; ++i)
	      cout<<out[i].real()<<" ";
	    cout<<endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  
  // Clean up
  free(in);
  free(out);
  free(row_above_send);
  free(row_above_recv);
  free(row_below_send);
  free(row_below_recv);
  free(col_left_send);
  free(col_left_recv);
  free(col_right_send);
  free(col_right_recv);
  free(dep_back_send);
  free(dep_back_recv);
  free(dep_front_send);
  free(dep_front_recv);
  free(edge110_111_send);
  free(edge110_111_recv);
  free(edge010_011_send);
  free(edge010_011_recv);
  free(edge100_101_send); 
  free(edge100_101_recv);
  free(edge000_001_send);
  free(edge000_001_recv);
  free(edge011_111_send);
  free(edge011_111_recv);
  free(edge010_110_send);
  free(edge010_110_recv);
  free(edge001_101_send);
  free(edge001_101_recv);
  free(edge000_100_send);
  free(edge000_100_recv);
  free(edge100_110_send);
  free(edge100_110_recv);
  free(edge101_111_send);
  free(edge101_111_recv);
  free(edge000_010_send);
  free(edge000_010_recv);
  free(edge001_011_send);
  free(edge001_011_recv);
  free(corner000_send);
  free(corner000_recv);
  free(corner111_send);
  free(corner111_recv);
  free(corner010_send);
  free(corner010_recv);
  free(corner101_send);
  free(corner101_recv);
  free(corner011_send);
  free(corner011_recv);
  free(corner100_send);
  free(corner100_recv);
  free(corner001_send);
  free(corner001_recv);
  free(corner110_send);
  free(corner110_recv);

  MPI_Finalize();
  
  return 0;
}
