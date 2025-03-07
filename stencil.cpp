#include <iostream>
#include <cassert>
#include <mpi.h>

#define N (4)

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
  // int N = 4;  //size of global block

  int b = 2; //size of local block
  
  int r, c, d, p, id;

  //assume that r = c = d & p = rcd; 
  r = c = d = 2;

  MPI_Init(NULL, NULL);
  
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &p);  

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
  double *in = (double*) malloc(sizeof(double) * N/r  * N/c * N/d);
  double *out = (double*) malloc(sizeof(double) * N/r * N/c * N/d);

  //init
  //  for (int i = 0; i < N/r/b; ++i)
  //    for (int j = 0; j < N/c/b; ++j)
  //      for (int k = 0; k < N/d/b; ++k)
	//{
	  for (int ii = 0; ii < b; ++ii)
	    for (int jj = 0; jj < b; ++jj)
	      for (int kk = 0; kk < b; ++kk) {
		      in[(kk*b*b + jj*b + ii)] = (did*N*N*b + rid * N * b + cid * b) +  //compute start offsets
		                                                   kk*N*N + jj*N + ii;
        }
	//}

  // for (int j = 0; j < p; ++j) {
  //   if (id == j) {
	//     cout<<id<<": ("<<rid<<", "<<cid<<", "<<did<<") grp: ("<<row_grp<<", "<<col_grp<<", "<<dep_grp<<") ";
	//     for (int i = 0; i < 8; ++i)
	//       cout<<in[i]<<" ";
	//     cout<<endl;      
  //   }
  //   MPI_Barrier(MPI_COMM_WORLD);
  // }

  // Stencil
  int g = 1;
  int side = b * b * g;
  int edge = b * g * g;
  int corner = g * g * g;

  // above_recv contains the data you recieve from the processor above you
  // below_recv contains the data you recieve from the processor below you
  double *row_above_send = (double*) malloc(sizeof(double) * side);
  double *row_above_recv = (double*) malloc(sizeof(double) * side);
  double *row_below_send = (double*) malloc(sizeof(double) * side);
  double *row_below_recv = (double*) malloc(sizeof(double) * side);
  int row_above_id = calc_id(rid, cid, did, -1, 0, 0);
  int row_below_id = calc_id(rid, cid, did, 1, 0, 0);
  MPI_Sendrecv(row_above_send, side, MPI_DOUBLE, row_above_id, 0,
               row_below_recv, side, MPI_DOUBLE, row_below_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(row_below_send, side, MPI_DOUBLE, row_below_id, 0,
               row_above_recv, side, MPI_DOUBLE, row_above_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  double *col_left_send = (double*) malloc(sizeof(double) * side);
  double *col_left_recv = (double*) malloc(sizeof(double) * side);
  double *col_right_send = (double*) malloc(sizeof(double) * side);
  double *col_right_recv = (double*) malloc(sizeof(double) * side);
  int col_left_id = calc_id(rid, cid, did, 0, -1, 0);
  int col_right_id = calc_id(rid, cid, did, 0, 1, 0);
  MPI_Sendrecv(col_left_send, side, MPI_DOUBLE, col_left_id, 0,
               col_right_recv, side, MPI_DOUBLE, col_right_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(col_right_send, side, MPI_DOUBLE, col_right_id, 0,
               col_left_recv, side, MPI_DOUBLE, col_left_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  
  double *dep_back_send = (double*) malloc(sizeof(double) * side);
  double *dep_back_recv = (double*) malloc(sizeof(double) * side);
  double *dep_front_send = (double*) malloc(sizeof(double) * side);
  double *dep_front_recv = (double*) malloc(sizeof(double) * side);
  int dep_back_id = calc_id(rid, cid, did, 0, 0, -1);
  int dep_front_id = calc_id(rid, cid, did, 0, 0, 1);
  MPI_Sendrecv(dep_back_send, side, MPI_DOUBLE, dep_back_id, 0,
               dep_front_recv, side, MPI_DOUBLE, dep_front_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(dep_front_send, side, MPI_DOUBLE, dep_front_id, 0,
               dep_back_recv, side, MPI_DOUBLE, dep_back_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  // Top right edge and top left edge
  double *edge110_111_send = (double*) malloc(sizeof(double) * edge);
  double *edge110_111_recv = (double*) malloc(sizeof(double) * edge);
  double *edge010_011_send = (double*) malloc(sizeof(double) * edge);
  double *edge010_011_recv = (double*) malloc(sizeof(double) * edge);
  int edge110_111_id = calc_id(rid, cid, did, 1, 1, 0);
  int edge010_011_id = calc_id(rid, cid, did, -1, 1, 0);
  MPI_Sendrecv(edge110_111_send, corner, MPI_DOUBLE, edge110_111_id, 0, 
               edge010_011_recv, corner, MPI_DOUBLE, edge010_011_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(edge010_011_send, corner, MPI_DOUBLE, edge010_011_id, 0,
               edge110_111_recv, corner, MPI_DOUBLE, edge110_111_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  // Bottom right edge and bottom left edge
  double *edge100_101_send = (double*) malloc(sizeof(double) * edge);
  double *edge100_101_recv = (double*) malloc(sizeof(double) * edge);
  double *edge000_001_send = (double*) malloc(sizeof(double) * edge);
  double *edge000_001_recv = (double*) malloc(sizeof(double) * edge);
  int edge100_101_id = calc_id(rid, cid, did, 1, -1, 0);
  int edge000_001_id = calc_id(rid, cid, did, -1, -1, 0);
  MPI_Sendrecv(edge100_101_send, corner, MPI_DOUBLE, edge100_101_id, 0, 
               edge000_001_recv, corner, MPI_DOUBLE, edge000_001_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(edge000_001_send, corner, MPI_DOUBLE, edge000_001_id, 0,
               edge100_101_recv, corner, MPI_DOUBLE, edge100_101_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  // Top front edge and top back edge
  double *edge011_111_send = (double*) malloc(sizeof(double) * edge);
  double *edge011_111_recv = (double*) malloc(sizeof(double) * edge);
  double *edge010_110_send = (double*) malloc(sizeof(double) * edge);
  double *edge010_110_recv = (double*) malloc(sizeof(double) * edge);
  int edge011_111_id = calc_id(rid, cid, did, 0, 1, 1);
  int edge010_110_id = calc_id(rid, cid, did, 0, 1, -1);
  MPI_Sendrecv(edge011_111_send, corner, MPI_DOUBLE, edge011_111_id, 0, 
               edge010_110_recv, corner, MPI_DOUBLE, edge010_110_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(edge010_110_send, corner, MPI_DOUBLE, edge010_110_id, 0,
               edge011_111_recv, corner, MPI_DOUBLE, edge011_111_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  // Bottom front edge and bottom back edge
  double *edge001_101_send = (double*) malloc(sizeof(double) * edge);
  double *edge001_101_recv = (double*) malloc(sizeof(double) * edge);
  double *edge000_100_send = (double*) malloc(sizeof(double) * edge);
  double *edge000_100_recv = (double*) malloc(sizeof(double) * edge);
  int edge001_101_id = calc_id(rid, cid, did, 0, -1, 1);
  int edge000_100_id = calc_id(rid, cid, did, 0, -1, -1);
  MPI_Sendrecv(edge001_101_send, corner, MPI_DOUBLE, edge001_101_id, 0, 
               edge000_100_recv, corner, MPI_DOUBLE, edge000_100_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(edge000_100_send, corner, MPI_DOUBLE, edge000_100_id, 0,
               edge001_101_recv, corner, MPI_DOUBLE, edge001_101_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  // Right back edge and right front edge
  double *edge100_110_send = (double*) malloc(sizeof(double) * edge);
  double *edge100_110_recv = (double*) malloc(sizeof(double) * edge);
  double *edge101_111_send = (double*) malloc(sizeof(double) * edge);
  double *edge101_111_recv = (double*) malloc(sizeof(double) * edge);
  int edge100_110_id = calc_id(rid, cid, did, 1, 0, -1);
  int edge101_111_id = calc_id(rid, cid, did, 1, 0, 1);
  MPI_Sendrecv(edge100_110_send, corner, MPI_DOUBLE, edge100_110_id, 0, 
               edge101_111_recv, corner, MPI_DOUBLE, edge101_111_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(edge101_111_send, corner, MPI_DOUBLE, edge101_111_id, 0,
               edge100_110_recv, corner, MPI_DOUBLE, edge100_110_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  // Left back edge and left front edge
  double *edge000_010_send = (double*) malloc(sizeof(double) * edge);
  double *edge000_010_recv = (double*) malloc(sizeof(double) * edge);
  double *edge001_011_send = (double*) malloc(sizeof(double) * edge);
  double *edge001_011_recv = (double*) malloc(sizeof(double) * edge);
  int edge000_010_id = calc_id(rid, cid, did, -1, 0, -1);
  int edge001_011_id = calc_id(rid, cid, did, -1, 0, 1);
  MPI_Sendrecv(edge000_010_send, corner, MPI_DOUBLE, edge000_010_id, 0, 
               edge001_011_recv, corner, MPI_DOUBLE, edge001_011_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(edge001_011_send, corner, MPI_DOUBLE, edge001_011_id, 0,
               edge000_010_recv, corner, MPI_DOUBLE, edge000_010_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  // Top front right corner and bottom back left corner
  double *corner000_send = (double*) malloc(sizeof(double) * corner);
  double *corner000_recv = (double*) malloc(sizeof(double) * corner);
  double *corner111_send = (double*) malloc(sizeof(double) * corner);
  double *corner111_recv = (double*) malloc(sizeof(double) * corner);
  int corner000_id = calc_id(rid, cid, did, -1, -1, -1);
  int corner111_id = calc_id(rid, cid, did, 1, 1, 1);
  MPI_Sendrecv(corner000_send, corner, MPI_DOUBLE, corner000_id, 0, 
               corner111_recv, corner, MPI_DOUBLE, corner111_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(corner111_send, corner, MPI_DOUBLE, corner111_id, 0,
               corner000_recv, corner, MPI_DOUBLE, corner000_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  // Bottom front right corner and top back left corner
  double *corner010_send = (double*) malloc(sizeof(double) * corner);
  double *corner010_recv = (double*) malloc(sizeof(double) * corner);
  double *corner101_send = (double*) malloc(sizeof(double) * corner);
  double *corner101_recv = (double*) malloc(sizeof(double) * corner);
  int corner010_id = calc_id(rid, cid, did, -1, 1, -1);
  int corner101_id = calc_id(rid, cid, did, 1, -1, 1);
  MPI_Sendrecv(corner010_send, corner, MPI_DOUBLE, corner010_id, 0, 
               corner101_recv, corner, MPI_DOUBLE, corner101_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(corner101_send, corner, MPI_DOUBLE, corner101_id, 0,
               corner010_recv, corner, MPI_DOUBLE, corner010_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  // Top front left corner and bottom back right corner
  double *corner011_send = (double*) malloc(sizeof(double) * corner);
  double *corner011_recv = (double*) malloc(sizeof(double) * corner);
  double *corner100_send = (double*) malloc(sizeof(double) * corner);
  double *corner100_recv = (double*) malloc(sizeof(double) * corner);
  int corner011_id = calc_id(rid, cid, did, -1, 1, 1);
  int corner100_id = calc_id(rid, cid, did, 1, -1, -1);
  MPI_Sendrecv(corner011_send, corner, MPI_DOUBLE, corner011_id, 0, 
               corner100_recv, corner, MPI_DOUBLE, corner100_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(corner100_send, corner, MPI_DOUBLE, corner100_id, 0,
               corner011_recv, corner, MPI_DOUBLE, corner011_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  // Bottom front left corner and top back right corner
  double *corner001_send = (double*) malloc(sizeof(double) * corner);
  double *corner001_recv = (double*) malloc(sizeof(double) * corner);
  double *corner110_send = (double*) malloc(sizeof(double) * corner);
  double *corner110_recv = (double*) malloc(sizeof(double) * corner);
  int corner001_id = calc_id(rid, cid, did, -1, -1, 1);
  int corner110_id = calc_id(rid, cid, did, 1, 1, -1);
  MPI_Sendrecv(corner001_send, corner, MPI_DOUBLE, corner001_id, 0, 
               corner110_recv, corner, MPI_DOUBLE, corner110_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(corner110_send, corner, MPI_DOUBLE, corner110_id, 0,
               corner001_recv, corner, MPI_DOUBLE, corner001_id, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  // for (int j = 0; j < p; ++j) {
  //   if (id == j) {
	//     cout<<id<<" "<<row_above_id<<" "<<row_below_id<<" ";
  //     cout<<col_left_id<<" "<<col_right_id<<" ";
  //     cout<<dep_back_id<<" "<<dep_front_id<<endl;
  //   }
  //   MPI_Barrier(MPI_COMM_WORLD);
  // }
  
  free(in);
  free(out);

  MPI_Finalize();
  
  return 0;
}
