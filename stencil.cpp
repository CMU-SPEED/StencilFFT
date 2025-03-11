#include <iostream>
#include <cassert>
#include <mpi.h>
#include <chrono>

//size of global block
#define N (4)

//assume that r = c = d & p = rcd;
#define R (2)

#define C (2)

#define D (2)

using namespace std;

int calc_id(int rid, int cid, int did, int edit_r, int edit_c, int edit_d) {
  int new_rid, new_cid, new_did;

  switch (edit_r) {
    case -1:
      new_rid = (rid == 0) ? (R - 1) : (rid - 1);
      break;
    case 1:
      new_rid = (rid + 1) % R;
      break;
    case 0:
      new_rid = rid;
      break;
    default:
      assert(false);
  }

  switch (edit_c) {
    case -1:
      new_cid = (cid == 0) ? (C - 1) : (cid - 1);
      break;
    case 1:
      new_cid = (cid + 1) % C;
      break;
    case 0:
      new_cid = cid;
      break;
    default:
      assert(false);
  }

  switch (edit_d) {
    case -1:
      new_did = (did == 0) ? (D - 1) : (did - 1);
      break;
    case 1:
      new_did = (did + 1) % D;
      break;
    case 0:
      new_did = did;
      break;
    default:
      assert(false);
  }

  return (new_did * R * C) + (new_rid * C) + new_cid;
}

int main() {
  int b = 2; //size of local block

  MPI_Init(NULL, NULL);
  
  int p, id;
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &p);  

  MPI_Comm row_comm, col_comm, dep_comm;

  //check later
  // [0-3 => 0, 4-7 => 1]
  int dep_grp = id % (R * C);
  MPI_Comm_split(MPI_COMM_WORLD, dep_grp, id, &dep_comm);

  // [0,1 => 0, 2,3 => 1 4,5 => 2, 6,7 => 3]
  int row_grp = id / R;
  MPI_Comm_split(MPI_COMM_WORLD, row_grp, id, &row_comm);

  // [0,1,4,5 => 0, 2,3,6,7 => 1]
  int col_grp = (row_grp / R) * C + id % D;
  MPI_Comm_split(MPI_COMM_WORLD, col_grp, id, &col_comm); 

  int rid, cid, did;
  did = id / (R * C);
  rid = (id - (did * R * C)) / C;
  cid = (id - (did * R * C)) % R;

  // 4 x 4 x 4 local data cube that is block cyclic dist in 2x2x2 blocks
  // total of 8 blocks per local processor. 
  double *in = (double*) malloc(sizeof(double) * N/R * N/C * N/D);
  double *out = (double*) malloc(sizeof(double) * N/R * N/C * N/D);
  
  //init
  for (int i = 0; i < N/R/b; ++i)
    for (int j = 0; j < N/C/b; ++j)
      for (int k = 0; k < N/D/b; ++k)
	     for (int ii = 0; ii < b; ++ii)
	       for (int jj = 0; jj < b; ++jj)
	         for (int kk = 0; kk < b; ++kk) {
            in[((k * (b*b*b * N/R/b * N/C/b)) +
                (j * (b*b*b * N/C/b)) + 
                (i * (b*b*b))) + (kk*b*b + jj*b + ii)] = (did*N*N*b + rid * N * b + cid * b) + //processor offset
                                                         ((k * N * N * b * D) + (j * N * b * C) + (i * b * R)) + //block offset
                                                         kk*N*N + jj*N + ii;
  }

  if (id == 0) cout<<"Initial data distribution"<<endl;
  for (int j = 0; j < p; ++j) {
    if (id == j) {
	    cout<<id<<": ("<<rid<<", "<<cid<<", "<<did<<") grp: ("<<row_grp<<", "<<col_grp<<", "<<dep_grp<<") ";
	    for (int i = 0; i < N/R * N/C * N/D; ++i)
	      cout<<in[i]<<" ";
	    cout<<endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Stencil
  int g = 1;
  int b_per_p = (N/R * N/C * N/D) / (b * b * b);
  int side = b * b * g;
  int total_side = b_per_p * side;
  int edge = b * g * g;
  int total_edge = b_per_p * edge;
  int corner = g * g * g;
  int total_corner = b_per_p * corner;

  // TODO: Copy data to stencil buffers 

  // above_recv contains the data you recieve from the processor above you
  // below_recv contains the data you recieve from the processor below you
  double *row_above_send = (double*) malloc(sizeof(double) * total_side);
  double *row_above_recv = (double*) malloc(sizeof(double) * total_side);
  double *row_below_send = (double*) malloc(sizeof(double) * total_side);
  double *row_below_recv = (double*) malloc(sizeof(double) * total_side);
  int row_above_id = calc_id(rid, cid, did, -1, 0, 0);
  int row_below_id = calc_id(rid, cid, did, 1, 0, 0);

  double *col_left_send = (double*) malloc(sizeof(double) * total_side);
  double *col_left_recv = (double*) malloc(sizeof(double) * total_side);
  double *col_right_send = (double*) malloc(sizeof(double) * total_side);
  double *col_right_recv = (double*) malloc(sizeof(double) * total_side);
  int col_left_id = calc_id(rid, cid, did, 0, -1, 0);
  int col_right_id = calc_id(rid, cid, did, 0, 1, 0);

  double *dep_back_send = (double*) malloc(sizeof(double) * total_side);
  double *dep_back_recv = (double*) malloc(sizeof(double) * total_side);
  double *dep_front_send = (double*) malloc(sizeof(double) * total_side);
  double *dep_front_recv = (double*) malloc(sizeof(double) * total_side);
  int dep_back_id = calc_id(rid, cid, did, 0, 0, -1);
  int dep_front_id = calc_id(rid, cid, did, 0, 0, 1);

  // Top right edge and top left edge
  double *edge110_111_send = (double*) malloc(sizeof(double) * total_edge);
  double *edge110_111_recv = (double*) malloc(sizeof(double) * total_edge);
  double *edge010_011_send = (double*) malloc(sizeof(double) * total_edge);
  double *edge010_011_recv = (double*) malloc(sizeof(double) * total_edge);
  int edge110_111_id = calc_id(rid, cid, did, 1, 1, 0);
  int edge010_011_id = calc_id(rid, cid, did, -1, 1, 0);

  // Bottom right edge and bottom left edge
  double *edge100_101_send = (double*) malloc(sizeof(double) * total_edge);
  double *edge100_101_recv = (double*) malloc(sizeof(double) * total_edge);
  double *edge000_001_send = (double*) malloc(sizeof(double) * total_edge);
  double *edge000_001_recv = (double*) malloc(sizeof(double) * total_edge);
  int edge100_101_id = calc_id(rid, cid, did, 1, -1, 0);
  int edge000_001_id = calc_id(rid, cid, did, -1, -1, 0);

  // Top front edge and top back edge
  double *edge011_111_send = (double*) malloc(sizeof(double) * total_edge);
  double *edge011_111_recv = (double*) malloc(sizeof(double) * total_edge);
  double *edge010_110_send = (double*) malloc(sizeof(double) * total_edge);
  double *edge010_110_recv = (double*) malloc(sizeof(double) * total_edge);
  int edge011_111_id = calc_id(rid, cid, did, 0, 1, 1);
  int edge010_110_id = calc_id(rid, cid, did, 0, 1, -1);

  // Bottom front edge and bottom back edge
  double *edge001_101_send = (double*) malloc(sizeof(double) * total_edge);
  double *edge001_101_recv = (double*) malloc(sizeof(double) * total_edge);
  double *edge000_100_send = (double*) malloc(sizeof(double) * total_edge);
  double *edge000_100_recv = (double*) malloc(sizeof(double) * total_edge);
  int edge001_101_id = calc_id(rid, cid, did, 0, -1, 1);
  int edge000_100_id = calc_id(rid, cid, did, 0, -1, -1);

  // Right back edge and right front edge
  double *edge100_110_send = (double*) malloc(sizeof(double) * total_edge);
  double *edge100_110_recv = (double*) malloc(sizeof(double) * total_edge);
  double *edge101_111_send = (double*) malloc(sizeof(double) * total_edge);
  double *edge101_111_recv = (double*) malloc(sizeof(double) * total_edge);
  int edge100_110_id = calc_id(rid, cid, did, 1, 0, -1);
  int edge101_111_id = calc_id(rid, cid, did, 1, 0, 1);

  // Left back edge and left front edge
  double *edge000_010_send = (double*) malloc(sizeof(double) * total_edge);
  double *edge000_010_recv = (double*) malloc(sizeof(double) * total_edge);
  double *edge001_011_send = (double*) malloc(sizeof(double) * total_edge);
  double *edge001_011_recv = (double*) malloc(sizeof(double) * total_edge);
  int edge000_010_id = calc_id(rid, cid, did, -1, 0, -1);
  int edge001_011_id = calc_id(rid, cid, did, -1, 0, 1);

  // Top front right corner and bottom back left corner
  double *corner000_send = (double*) malloc(sizeof(double) * total_corner);
  double *corner000_recv = (double*) malloc(sizeof(double) * total_corner);
  double *corner111_send = (double*) malloc(sizeof(double) * total_corner);
  double *corner111_recv = (double*) malloc(sizeof(double) * total_corner);
  int corner000_id = calc_id(rid, cid, did, -1, -1, -1);
  int corner111_id = calc_id(rid, cid, did, 1, 1, 1);

  // Bottom front right corner and top back left corner
  double *corner010_send = (double*) malloc(sizeof(double) * total_corner);
  double *corner010_recv = (double*) malloc(sizeof(double) * total_corner);
  double *corner101_send = (double*) malloc(sizeof(double) * total_corner);
  double *corner101_recv = (double*) malloc(sizeof(double) * total_corner);
  int corner010_id = calc_id(rid, cid, did, -1, 1, -1);
  int corner101_id = calc_id(rid, cid, did, 1, -1, 1);

  // Top front left corner and bottom back right corner
  double *corner011_send = (double*) malloc(sizeof(double) * total_corner);
  double *corner011_recv = (double*) malloc(sizeof(double) * total_corner);
  double *corner100_send = (double*) malloc(sizeof(double) * total_corner);
  double *corner100_recv = (double*) malloc(sizeof(double) * total_corner);
  int corner011_id = calc_id(rid, cid, did, -1, 1, 1);
  int corner100_id = calc_id(rid, cid, did, 1, -1, -1);

  // Bottom front left corner and top back right corner
  double *corner001_send = (double*) malloc(sizeof(double) * total_corner);
  double *corner001_recv = (double*) malloc(sizeof(double) * total_corner);
  double *corner110_send = (double*) malloc(sizeof(double) * total_corner);
  double *corner110_recv = (double*) malloc(sizeof(double) * total_corner);
  int corner001_id = calc_id(rid, cid, did, -1, -1, 1);
  int corner110_id = calc_id(rid, cid, did, 1, 1, -1);

  auto start = std::chrono::high_resolution_clock::now();

  MPI_Sendrecv(row_above_send, total_side, MPI_DOUBLE, row_above_id, 0,
               row_below_recv, total_side, MPI_DOUBLE, row_below_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(row_below_send, total_side, MPI_DOUBLE, row_below_id, 0,
               row_above_recv, total_side, MPI_DOUBLE, row_above_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  
  MPI_Sendrecv(col_left_send, total_side, MPI_DOUBLE, col_left_id, 0,
               col_right_recv, total_side, MPI_DOUBLE, col_right_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(col_right_send, total_side, MPI_DOUBLE, col_right_id, 0,
               col_left_recv, total_side, MPI_DOUBLE, col_left_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  
  MPI_Sendrecv(dep_back_send, total_side, MPI_DOUBLE, dep_back_id, 0,
               dep_front_recv, total_side, MPI_DOUBLE, dep_front_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(dep_front_send, total_side, MPI_DOUBLE, dep_front_id, 0,
               dep_back_recv, total_side, MPI_DOUBLE, dep_back_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Sendrecv(edge110_111_send, total_corner, MPI_DOUBLE, edge110_111_id, 0, 
               edge010_011_recv, total_corner, MPI_DOUBLE, edge010_011_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(edge010_011_send, total_corner, MPI_DOUBLE, edge010_011_id, 0,
               edge110_111_recv, total_corner, MPI_DOUBLE, edge110_111_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Sendrecv(edge100_101_send, total_corner, MPI_DOUBLE, edge100_101_id, 0, 
               edge000_001_recv, total_corner, MPI_DOUBLE, edge000_001_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(edge000_001_send, total_corner, MPI_DOUBLE, edge000_001_id, 0,
               edge100_101_recv, total_corner, MPI_DOUBLE, edge100_101_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Sendrecv(edge011_111_send, total_corner, MPI_DOUBLE, edge011_111_id, 0, 
               edge010_110_recv, total_corner, MPI_DOUBLE, edge010_110_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(edge010_110_send, total_corner, MPI_DOUBLE, edge010_110_id, 0,
               edge011_111_recv, total_corner, MPI_DOUBLE, edge011_111_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Sendrecv(edge001_101_send, total_corner, MPI_DOUBLE, edge001_101_id, 0, 
               edge000_100_recv, total_corner, MPI_DOUBLE, edge000_100_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(edge000_100_send, total_corner, MPI_DOUBLE, edge000_100_id, 0,
               edge001_101_recv, total_corner, MPI_DOUBLE, edge001_101_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Sendrecv(edge100_110_send, total_corner, MPI_DOUBLE, edge100_110_id, 0, 
               edge101_111_recv, total_corner, MPI_DOUBLE, edge101_111_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(edge101_111_send, total_corner, MPI_DOUBLE, edge101_111_id, 0,
               edge100_110_recv, total_corner, MPI_DOUBLE, edge100_110_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Sendrecv(edge000_010_send, total_corner, MPI_DOUBLE, edge000_010_id, 0, 
               edge001_011_recv, total_corner, MPI_DOUBLE, edge001_011_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(edge001_011_send, total_corner, MPI_DOUBLE, edge001_011_id, 0,
               edge000_010_recv, total_corner, MPI_DOUBLE, edge000_010_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Sendrecv(corner000_send, total_corner, MPI_DOUBLE, corner000_id, 0, 
               corner111_recv, total_corner, MPI_DOUBLE, corner111_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(corner111_send, total_corner, MPI_DOUBLE, corner111_id, 0,
               corner000_recv, total_corner, MPI_DOUBLE, corner000_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  
  MPI_Sendrecv(corner010_send, total_corner, MPI_DOUBLE, corner010_id, 0, 
               corner101_recv, total_corner, MPI_DOUBLE, corner101_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(corner101_send, total_corner, MPI_DOUBLE, corner101_id, 0,
               corner010_recv, total_corner, MPI_DOUBLE, corner010_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Sendrecv(corner011_send, total_corner, MPI_DOUBLE, corner011_id, 0, 
               corner100_recv, total_corner, MPI_DOUBLE, corner100_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(corner100_send, total_corner, MPI_DOUBLE, corner100_id, 0,
               corner011_recv, total_corner, MPI_DOUBLE, corner011_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Sendrecv(corner001_send, total_corner, MPI_DOUBLE, corner001_id, 0, 
               corner110_recv, total_corner, MPI_DOUBLE, corner110_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(corner110_send, total_corner, MPI_DOUBLE, corner110_id, 0,
               corner001_recv, total_corner, MPI_DOUBLE, corner001_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end- start);
  if (id == 0) std::cout << "Stencil time: " << duration.count() << " ns" << std::endl;

  // All to all in the rows

  start = std::chrono::high_resolution_clock::now();

  MPI_Alltoall(in,
               (N/R * N/C * N/D) / R,
               MPI_DOUBLE,
               in, 
               (N/R * N/C * N/D) / R,
               MPI_DOUBLE,
               row_comm);

  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end- start);
  if (id == 0) std::cout << "All to all time: " << duration.count() << " ns" << std::endl;

  if (id == 0) cout<<"After All to all in rows"<<endl;
  for (int j = 0; j < p; ++j) {
    if (id == j) {
	    cout<<id<<": ("<<rid<<", "<<cid<<", "<<did<<") grp: ("<<row_grp<<", "<<col_grp<<", "<<dep_grp<<") ";
	    for (int i = 0; i < N/R * N/C * N/D; ++i)
	      cout<<in[i]<<" ";
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
