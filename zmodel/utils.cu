#include "utils.h"

#ifdef __USE__HDF5__
#include <hdf5.h>
#endif

/******************** 2D Init ********************/

template<typename T>
void init_host(int rid, int cid, T *host_in) {
    size_t row_offset = cid * B_DIM;
    size_t col_offset = rid * (N_DIM*B_DIM);
    size_t offset = row_offset + col_offset;

    T *block_cyclic = (T*)malloc(LOCAL_COMPLEX_BYTES);

    for (size_t i = 0; i < LOCAL_DIM/B_DIM; i++) {
        for (size_t ii = 0; ii < B_DIM; ii++) {
            for (size_t j = 0; j < LOCAL_DIM/B_DIM; j++) {
                for (size_t jj = 0; jj < B_DIM; jj++) {
                    size_t array_index = (i * B_DIM * (LOCAL_DIM/B_DIM) * B_DIM) + (ii * (LOCAL_DIM/B_DIM) * B_DIM) + (j * B_DIM) + jj;
                    double real = (i * N_DIM * B_DIM * P_DIM) + (ii * N_DIM) + (j * B_DIM * P_DIM) + jj + offset;

                    if constexpr (std::is_same_v<T, Complex>) {
                        block_cyclic[array_index] = {real, 0.0};
                    } else {
                        block_cyclic[array_index] = real;
                    }
                }
            }
        }
    }

    vector_pack(block_cyclic, host_in, LOCAL_DIM);
    free(block_cyclic);
}

template void init_host<Complex>(int, int, Complex*);
template void init_host<Scalar>(int, int, Scalar*);

/******************** 3D Init ********************/

template<typename T>
void init_host_3d(int rid, int cid, int did, T *host_in) {
    size_t row_offset = cid * B_DIM;
    size_t col_offset = rid * (N_DIM*B_DIM);
    size_t dep_offset = did * (N_DIM * N_DIM * B_DIM);
    size_t offset = row_offset + col_offset + dep_offset;

    T *block_cyclic = (T*)malloc(LOCAL_COMPLEX_BYTES_3D);

    for (size_t i = 0; i < LOCAL_DIM/B_DIM; i++) {
        for (size_t ii = 0; ii < B_DIM; ii++) {
            for (size_t j = 0; j < LOCAL_DIM/B_DIM; j++) {
                for (size_t jj = 0; jj < B_DIM; jj++) {
                    for (size_t k = 0; k < LOCAL_DIM/B_DIM; k++) {
                        for (size_t kk = 0; kk < B_DIM; kk++) {
                            size_t array_index = (i * B_DIM * LOCAL_DIM * LOCAL_DIM) + (ii * LOCAL_DIM * LOCAL_DIM) +
                                                 (j * B_DIM * LOCAL_DIM) + (jj * LOCAL_DIM) + (k * B_DIM) + kk;
                            double real = (i * N_DIM * N_DIM * B_DIM * P_DIM) + (ii * N_DIM * N_DIM) +
                                          (j * N_DIM * B_DIM * P_DIM) + (jj * N_DIM) +
                                          (k * B_DIM * P_DIM) + kk + offset;

                            if constexpr (std::is_same_v<T, Complex>) {
                               block_cyclic[array_index] = {real, 0.0};
                            } else {
                                block_cyclic[array_index] = real;
                            }
                        }
                    }
                }
            }
        }
    }

    vector_pack(block_cyclic, host_in, LOCAL_DIM * LOCAL_DIM);
    free(block_cyclic);
}

template void init_host_3d<Complex>(int, int, int, Complex*);
template void init_host_3d<Scalar>(int, int, int, Scalar*);

/******************** HDF5 I/O ********************/

#ifdef __USE__HDF5__

// Creates the HDF5 compound type matching cuDoubleComplex {double x, y}
static hid_t create_complex_type() {
    hid_t type = H5Tcreate(H5T_COMPOUND, sizeof(Complex));
    H5Tinsert(type, "real", offsetof(Complex, x), H5T_NATIVE_DOUBLE);
    H5Tinsert(type, "imag", offsetof(Complex, y), H5T_NATIVE_DOUBLE);
    return type;
}

/**
 * Load a 2D N x N complex dataset from HDF5 into the local vector-packed buffer.
 *
 * Each processor reads only its block-cyclic chunk using a hyperslab selection:
 *   start = (rid*B, cid*B), stride = (B*P, B*P), count = (L/B, L/B), block = (B, B)
 *
 * The data is read in block-cyclic order then vector-packed into host_in.
 */
void init_host_hdf5(int rid, int cid, Complex *host_in, const char *filename, const char *dataset_name) {
    // Open file with parallel access
    hid_t plist = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist, MPI_COMM_WORLD, MPI_INFO_NULL);
    hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, plist);
    H5Pclose(plist);

    hid_t dset = H5Dopen2(file, dataset_name, H5P_DEFAULT);
    hid_t file_space = H5Dget_space(dset);

    // Select this processor's block-cyclic hyperslab
    hsize_t start[2]  = { (hsize_t)(rid * B_DIM), (hsize_t)(cid * B_DIM) };
    hsize_t stride[2] = { (hsize_t)(B_DIM * P_DIM), (hsize_t)(B_DIM * P_DIM) };
    hsize_t count[2]  = { (hsize_t)(LOCAL_DIM / B_DIM), (hsize_t)(LOCAL_DIM / B_DIM) };
    hsize_t block[2]  = { (hsize_t)B_DIM, (hsize_t)B_DIM };
    H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, stride, count, block);

    // Memory space: contiguous L x L
    hsize_t mem_dims[2] = { LOCAL_DIM, LOCAL_DIM };
    hid_t mem_space = H5Screate_simple(2, mem_dims, NULL);

    // Read into block-cyclic buffer, then vector-pack
    Complex *block_cyclic = (Complex*)malloc(LOCAL_COMPLEX_BYTES);
    hid_t ctype = create_complex_type();

    hid_t xfer_plist = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
    H5Dread(dset, ctype, mem_space, file_space, xfer_plist, block_cyclic);
    H5Pclose(xfer_plist);

    vector_pack(block_cyclic, host_in, LOCAL_DIM);
    free(block_cyclic);

    H5Tclose(ctype);
    H5Sclose(mem_space);
    H5Sclose(file_space);
    H5Dclose(dset);
    H5Fclose(file);
}

/**
 * Store the local 2D vector-packed buffer to an N x N complex HDF5 dataset.
 *
 * Each processor vector-unpacks its data to block-cyclic order, then writes its
 * block-cyclic chunk via a hyperslab selection. Creates the file and dataset
 * if they don't exist.
 */
void store_host_hdf5(int rid, int cid, const Complex *host_in, const char *filename, const char *dataset_name) {
    // Vector-unpack to block-cyclic order
    Complex *block_cyclic = (Complex*)malloc(LOCAL_COMPLEX_BYTES);
    vector_unpack(host_in, block_cyclic, LOCAL_DIM);

    // Create or open file with parallel access
    hid_t plist = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist, MPI_COMM_WORLD, MPI_INFO_NULL);
    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, plist);
    H5Pclose(plist);

    // Create dataset
    hsize_t dims[2] = { N_DIM, N_DIM };
    hid_t file_space = H5Screate_simple(2, dims, NULL);
    hid_t ctype = create_complex_type();
    hid_t dset = H5Dcreate2(file, dataset_name, ctype, file_space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // Select this processor's block-cyclic hyperslab
    hsize_t start[2]  = { (hsize_t)(rid * B_DIM), (hsize_t)(cid * B_DIM) };
    hsize_t stride[2] = { (hsize_t)(B_DIM * P_DIM), (hsize_t)(B_DIM * P_DIM) };
    hsize_t count[2]  = { (hsize_t)(LOCAL_DIM / B_DIM), (hsize_t)(LOCAL_DIM / B_DIM) };
    hsize_t block[2]  = { (hsize_t)B_DIM, (hsize_t)B_DIM };
    H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, stride, count, block);

    hsize_t mem_dims[2] = { LOCAL_DIM, LOCAL_DIM };
    hid_t mem_space = H5Screate_simple(2, mem_dims, NULL);

    hid_t xfer_plist = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
    H5Dwrite(dset, ctype, mem_space, file_space, xfer_plist, block_cyclic);
    H5Pclose(xfer_plist);

    free(block_cyclic);

    H5Tclose(ctype);
    H5Sclose(mem_space);
    H5Sclose(file_space);
    H5Dclose(dset);
    H5Fclose(file);
}

/**
 * Load a 3D N x N x N complex dataset from HDF5 into the local vector-packed buffer.
 *
 * Each processor reads only its block-cyclic chunk using a hyperslab selection:
 *   start = (did*B, rid*B, cid*B), stride = (B*P, B*P, B*P),
 *   count = (L/B, L/B, L/B), block = (B, B, B)
 *
 * The data is read in block-cyclic order then vector-packed into host_in.
 * Only the column (innermost) dimension is vector-packed.
 */
void init_host_hdf5_3d(int rid, int cid, int did, Complex *host_in, const char *filename, const char *dataset_name) {
    hid_t plist = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist, MPI_COMM_WORLD, MPI_INFO_NULL);
    hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, plist);
    H5Pclose(plist);

    hid_t dset = H5Dopen2(file, dataset_name, H5P_DEFAULT);
    hid_t file_space = H5Dget_space(dset);

    hsize_t start[3]  = { (hsize_t)(did * B_DIM), (hsize_t)(rid * B_DIM), (hsize_t)(cid * B_DIM) };
    hsize_t stride[3] = { (hsize_t)(B_DIM * P_DIM), (hsize_t)(B_DIM * P_DIM), (hsize_t)(B_DIM * P_DIM) };
    hsize_t count[3]  = { (hsize_t)(LOCAL_DIM / B_DIM), (hsize_t)(LOCAL_DIM / B_DIM), (hsize_t)(LOCAL_DIM / B_DIM) };
    hsize_t block[3]  = { (hsize_t)B_DIM, (hsize_t)B_DIM, (hsize_t)B_DIM };
    H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, stride, count, block);

    hsize_t mem_dims[3] = { LOCAL_DIM, LOCAL_DIM, LOCAL_DIM };
    hid_t mem_space = H5Screate_simple(3, mem_dims, NULL);

    Complex *block_cyclic = (Complex*)malloc(LOCAL_COMPLEX_BYTES_3D);
    hid_t ctype = create_complex_type();

    hid_t xfer_plist = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
    H5Dread(dset, ctype, mem_space, file_space, xfer_plist, block_cyclic);
    H5Pclose(xfer_plist);

    vector_pack(block_cyclic, host_in, LOCAL_DIM * LOCAL_DIM);
    free(block_cyclic);

    H5Tclose(ctype);
    H5Sclose(mem_space);
    H5Sclose(file_space);
    H5Dclose(dset);
    H5Fclose(file);
}

/**
 * Store the local 3D vector-packed buffer to an N x N x N complex HDF5 dataset.
 *
 * Each processor vector-unpacks its data to block-cyclic order, then writes its
 * block-cyclic chunk via a hyperslab selection. Creates the file and dataset
 * if they don't exist.
 */
void store_host_hdf5_3d(int rid, int cid, int did, const Complex *host_in, const char *filename, const char *dataset_name) {
    Complex *block_cyclic = (Complex*)malloc(LOCAL_COMPLEX_BYTES_3D);
    vector_unpack(host_in, block_cyclic, LOCAL_DIM * LOCAL_DIM);

    hid_t plist = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist, MPI_COMM_WORLD, MPI_INFO_NULL);
    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, plist);
    H5Pclose(plist);

    hsize_t dims[3] = { N_DIM, N_DIM, N_DIM };
    hid_t file_space = H5Screate_simple(3, dims, NULL);
    hid_t ctype = create_complex_type();
    hid_t dset = H5Dcreate2(file, dataset_name, ctype, file_space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    hsize_t start[3]  = { (hsize_t)(did * B_DIM), (hsize_t)(rid * B_DIM), (hsize_t)(cid * B_DIM) };
    hsize_t stride[3] = { (hsize_t)(B_DIM * P_DIM), (hsize_t)(B_DIM * P_DIM), (hsize_t)(B_DIM * P_DIM) };
    hsize_t count[3]  = { (hsize_t)(LOCAL_DIM / B_DIM), (hsize_t)(LOCAL_DIM / B_DIM), (hsize_t)(LOCAL_DIM / B_DIM) };
    hsize_t block[3]  = { (hsize_t)B_DIM, (hsize_t)B_DIM, (hsize_t)B_DIM };
    H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, stride, count, block);

    hsize_t mem_dims[3] = { LOCAL_DIM, LOCAL_DIM, LOCAL_DIM };
    hid_t mem_space = H5Screate_simple(3, mem_dims, NULL);

    hid_t xfer_plist = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
    H5Dwrite(dset, ctype, mem_space, file_space, xfer_plist, block_cyclic);
    H5Pclose(xfer_plist);

    free(block_cyclic);

    H5Tclose(ctype);
    H5Sclose(mem_space);
    H5Sclose(file_space);
    H5Dclose(dset);
    H5Fclose(file);
}

#endif // __USE__HDF5__
