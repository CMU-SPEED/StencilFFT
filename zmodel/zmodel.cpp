#include <complex>
#include <mpi.h>

// assume that r = c & p = rc;
// This will be defined at runtime
// int r = 0; 
// int c = 0;
int p = 0;
int N = 0;
int b = 0;
// int t = 0;

using namespace std;
using Complex = std::complex<double>;

int main(int argc, char* argv[]) {

    //size of global block
    N = atoi(argv[1]);

    //size of local block
    b = atoi(argv[2]);

    MPI_Init(NULL, NULL);

    int P, id;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    p = static_cast<int>(std::round(std::sqrt(P)));

    MPI_Comm row_comm, col_comm;

    int row_grp = id / p;
    MPI_Comm_split(MPI_COMM_WORLD, row_grp, id, &row_comm);

    int col_grp = id % p;
    MPI_Comm_split(MPI_COMM_WORLD, col_grp, id, &col_comm); 

    // Keep same naming convention as 3D.  In 3D grp != id
    int rid, cid;
    rid = row_grp;
    cid = col_grp;

    int local = N / p;

    // FIXME: change this to fftw malloc
    Complex *in = (Complex*) calloc(local * local, sizeof(Complex));
    Complex *out = (Complex*) calloc(local * local, sizeof(Complex));

    // Initialize block cyclic data layout
    int row_offset = rid * b;       // offset based on which processor in the row
    int col_offset = cid * (N*b);   // offset based on which processor in the col
    int offset = row_offset + col_offset;
    
    for (int i = 0; i < local/b; i++) {            // This iterates over the rows of blocks (steps from block to block in a column)
        for (int ii = 0; ii < b; ii++) {         // This iterates over the rows in a block
            for (int j = 0; j < local/b; j++) {    // This iterates over the columns of blocks (steps from block to block in row)
                for (int jj = 0; jj < b; jj++) { // This iterates over the columns in a block
                    int array_index = (i * b * (local/b) * b) + (ii * (local/b) * b) + (j * b) + jj;
                    
                    // j * b * c     --> Step over c blocks in a row
                    // ii * N        --> 
                    // i * N * b * r --> 

                    double real = (i * N * b * p) + (ii * N) + (j * b * p) + jj + offset;

                    in[array_index] = Complex(real, 0.0);
                }
            }
        }
    }

    if (id == 0) cout<<"Block cylic data distribution"<<endl;
    for (int j = 0; j < P; ++j) {
        if (id == j) {
	        cout<<id<<": ("<<rid<<", "<<cid<<") grp: ("<<row_grp<<", "<<col_grp<<") ";
	        for (int i = 0; i < local * local; ++i)
	            cout<<in[i].real()<<" ";
	        cout<<endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /**
     * @brief Pack in the layout that will be used for the rest of the problem
     * 
     * We take the first element from the blocks in row pack it togther
     * Take the second element from the blocks in a row pack it togeter
     * Repeat
     * 
     * Note that we handle each row of the blocks independently
     * We are only rearranging elements within a row
     */
    // int local = N / r;                         // Dimension of the local block on each processor
    // for (int i = 0; i < local; i++) {           // Iterate over the rows
    //     for (int j = 0; j < local/b; j++) {     // Iterate over the blocks within a row
    //         for (int k = 0; k < b; k++) {       // Iterate over the elements in the row of a block
    //             // Just swap the j and k indicies
    //             // out[(i * local) + (k * b) + j] = in[(i * local) + (j * b) + k];
    //             // FIXME THIS DOESNT QUITE WORK
    //             // double real = (i * local); // FIXME
    //             out[(i * local) + (j * b) + k] = in[(i * local) + (k * b)];
    //         }
    //     }
    // }
    for (int i = 0; i < local; i++) {
        for (int j = 0; j < b; j++) {
            for (int jj = 0; jj < local/b; jj++) {
                int row = i * local;
                out[row + (j * (local/b))+ jj] = in[row + j + (jj * b)];
            }
        }
    }

    if (id == 0) cout<<"Initial problem data distribution"<<endl;
    for (int j = 0; j < P; ++j) {
        if (id == j) {
	        cout<<id<<": ("<<rid<<", "<<cid<<") grp: ("<<row_grp<<", "<<col_grp<<") ";
	        for (int i = 0; i < local * local; ++i)
	            cout<<out[i].real()<<" ";
	        cout<<endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // TODO: FFT

    // Pack for All to All in Rows
    int split = local / 2;
    for (int i = 0; i < local/8; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 4; k++) {

                int offset = split * j;

                
                // We access out in a liner fashion

                // TODO: explain in indexing
                in[(i * 4) + offset + k] = out[(i * 2 * 4) + (j * 4) + k];
            }
        }
    }

    if (id == 0) cout<<"Pack for all to all in rows"<<endl;
    for (int j = 0; j < P; ++j) {
        if (id == j) {
	        cout<<id<<": ("<<rid<<", "<<cid<<") grp: ("<<row_grp<<", "<<col_grp<<") ";
	        for (int i = 0; i < local * local; ++i)
	            cout<<in[i].real()<<" ";
	        cout<<endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }


    // All to all in rows
    MPI_Alltoall(in,
                (local * local) / p,
                MPI_C_DOUBLE_COMPLEX,
                out, 
                (local * local) / p,
                MPI_C_DOUBLE_COMPLEX,
                row_comm);

    
    if (id == 0) cout<<"After all to all in rows"<<endl;
    for (int j = 0; j < P; ++j) {
        if (id == j) {
	        cout<<id<<": ("<<rid<<", "<<cid<<") grp: ("<<row_grp<<", "<<col_grp<<") ";
	        for (int i = 0; i < local * local; ++i)
	            cout<<out[i].real()<<" ";
	        cout<<endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Pack for FFT

    for (int i = 0; i < local; i++) {

        // FIXME: this naming is confusing --> change to buf1 and buf2 & then swap these two
        int in_offset = i * local;
        int out_offset = i * (local/p);

        for (int ii = 0; ii < p; ii++) {
            for (int jj = 0; jj < local / p; jj++) {
                in[in_offset + (ii * (local/p)) + jj] = out[out_offset + (ii * local * (local/p)) + jj];
            }
        }
    }

    if (id == 0) cout<<"Pack for FFT"<<endl;
    for (int j = 0; j < P; ++j) {
        if (id == j) {
	        cout<<id<<": ("<<rid<<", "<<cid<<") grp: ("<<row_grp<<", "<<col_grp<<") ";
	        for (int i = 0; i < local * local; ++i)
	            cout<<in[i].real()<<" ";
	        cout<<endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }


    // All to all in cols
    MPI_Alltoall(in,
                (local * local) / p,
                MPI_C_DOUBLE_COMPLEX,
                out, 
                (local * local) / p,
                MPI_C_DOUBLE_COMPLEX,
                col_comm);

    
    if (id == 0) cout<<"After all to all in cols"<<endl;
    for (int j = 0; j < P; ++j) {
        if (id == j) {
	        cout<<id<<": ("<<rid<<", "<<cid<<") grp: ("<<row_grp<<", "<<col_grp<<") ";
	        for (int i = 0; i < local * local; ++i)
	            cout<<out[i].real()<<" ";
	        cout<<endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Pack for final FFT
    for (int i = 0; i < local/b/p; i++) {
        int in_offset = i * local * b * p;
        int out_offset = i * local * b;
        for (int ii = 0; ii < p; ii++) {
            for (int jj = 0; jj < b * local; jj++) {
                in[in_offset + ii * b * local + jj] = out[out_offset + (ii * local * (local/p)) + jj];
            }
        }
    }

    

    if (id == 0) cout<<"Pack for final FFT"<<endl;
    for (int j = 0; j < P; ++j) {
        if (id == j) {
	        cout<<id<<": ("<<rid<<", "<<cid<<") grp: ("<<row_grp<<", "<<col_grp<<") ";
	        for (int i = 0; i < local * local; ++i)
	            cout<<in[i].real()<<" ";
	        cout<<endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Clean up
    free(in);
    free(out);

    MPI_Finalize();

    return 0;
}