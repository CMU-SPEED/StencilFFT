#ifndef __ZMODEL__UTILS__
#define __ZMODEL__UTILS__

#define __ZMODEL__CUDA__
// #define __ZMODEL__HIP__
#include "device_macros.h"
#include <vector>

using Complex = DEVICE_FFT_DOUBLECOMPLEX;
using Scalar = double;

template <typename T>
using Vector = std::vector<T>;

#define __PRINT__TIMING__
#define __PRINT__DETAILED__TIMING__
// #define __PRINT__RESULTS__
// #define __PRINT__TWIDDLES__
#define __PRINT__SANITY__
#define __ZMODEL__COMPUTE__
// #define __GPU__AWARE__MPI__
// #define __GPU__SET__
// #define __USE__FFTDX__
// #define  __ZMODEL__INCLUDE__INVERSE__
// #define __USE__HDF5__

// NOTE: these macros can be set on the commandline using: -UB_DIM -DB_DIM=8
#ifndef N_DIM
    #define N_DIM (16)
#endif
#ifndef B_DIM
    #define B_DIM (4)
#endif
#ifndef P_DIM
    // Number of processors in the rows or columns. ex: p = 2 --> 4 total processors
    // This is really sqrt(P) for 2D and cbrt(P) for 3D
    #define P_DIM (2)
#endif

#define RUNS (1)
#define NUM_STREAMS (4)

#define CUFFTDX_TARGET_SM 750

/******************** 2D MACROS ********************/

#define LOCAL_DIM (N_DIM/P_DIM)

#define LOCAL_COMPLEX_BYTES (LOCAL_DIM*LOCAL_DIM*sizeof(Complex))
#define PACKED_ROW_COMPLEX (LOCAL_DIM*(LOCAL_DIM/B_DIM))
#define PACKED_ROW_COMPLEX_BYTES (PACKED_ROW_COMPLEX*sizeof(Complex))
#define PACKED_COL_COMPLEX (LOCAL_DIM*VEC)
#define PACKED_COL_COMPLEX_BYTES (PACKED_COL_COMPLEX*sizeof(Complex))

#define LOCAL_SCALAR_BYTES (LOCAL_DIM*LOCAL_DIM*sizeof(Scalar))
#define PACKED_ROW_SCALAR_BYTES (LOCAL_DIM*(LOCAL_DIM/B_DIM)*sizeof(Scalar))
#define PACKED_COL_SCALAR_BYTES (LOCAL_DIM*VEC*sizeof(Scalar))

// This is the size of the vector the stencil will be batched with
#define VEC (LOCAL_DIM/B_DIM)
// The total number of vectors on the local processor
#define VEC_TOTAL ((LOCAL_DIM*LOCAL_DIM)/VEC)

#define VEC_ROW (LOCAL_DIM) // Number of vectors in a col
#define VEC_COL (B_DIM)     // Number of vectors in a row

/******************** 3D MACROS ********************/

#define LOCAL_COMPLEX_BYTES_3D (LOCAL_DIM*LOCAL_DIM*LOCAL_DIM*sizeof(Complex))
#define PACKED_FACE_COMPLEX_3D (LOCAL_DIM * PACKED_ROW_COMPLEX)
#define PACKED_FACE_COMPLEX_BYTES_3D (PACKED_FACE_COMPLEX_3D * sizeof(Complex))

/******************** Operators ********************/

__host__ __device__ static inline Complex operator*(double a, const Complex& b) {
    return {a * b.x, a * b.y};
}

__host__ __device__ static inline Complex& operator+=(Complex& a, const Complex& b) {
    a.x += b.x;
    a.y += b.y;
    return a;
}

__host__ __device__ static inline Complex& operator-=(Complex& a, const Complex& b) {
    a.x -= b.x;
    a.y -= b.y;
    return a;
}

__host__ __device__ static inline Complex operator*(const Complex& a, const Complex& b) {
    return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

/******************** Vector Packing ********************/


// Converts from a block cyclic layout to the vector packed layout used internally by the repo
template<typename T>
inline void vector_pack(const T* block_cyclic, T* packed, size_t num_rows) {
    for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j < (size_t)B_DIM; j++) {
            for (size_t jj = 0; jj < (size_t)VEC; jj++) {
                packed[i * LOCAL_DIM + j * VEC + jj] = block_cyclic[i * LOCAL_DIM + j + jj * B_DIM];
            }
        }
    }
}

template<typename T>
inline void vector_unpack(const T* packed, T* block_cyclic, size_t num_rows) {
    for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j < (size_t)B_DIM; j++) {
            for (size_t jj = 0; jj < (size_t)VEC; jj++) {
                block_cyclic[i * LOCAL_DIM + j + jj * B_DIM] = packed[i * LOCAL_DIM + j * VEC + jj];
            }
        }
    }
}

// Column index conversions between packed and block cyclic layouts
inline int packed_to_block_cyclic(int packed_col) {
    return (packed_col / VEC) + (packed_col % VEC) * B_DIM;
}

inline int block_cyclic_to_packed(int block_cyclic_col) {
    return (block_cyclic_col % B_DIM) * VEC + (block_cyclic_col / B_DIM);
}

/******************** Init ********************/

// Fills host_in with test data
template<typename T>
void init_host(int rid, int cid, T *host_in);

template<typename T>
void init_host_3d(int rid, int cid, int did, T *host_in);

/******************** HDF5 I/O ********************/

#ifdef __USE__HDF5__

// Fills host_in with data from hdf5 file
void init_host_hdf5(int rid, int cid, Complex *host_in, const char *filename, const char *dataset_name);

void store_host_hdf5(int rid, int cid, const Complex *host_in, const char *filename, const char *dataset_name);

void init_host_hdf5_3d(int rid, int cid, int did, Complex *host_in, const char *filename, const char *dataset_name);

void store_host_hdf5_3d(int rid, int cid, int did, const Complex *host_in, const char *filename, const char *dataset_name);

#endif // __USE__HDF5__

/******************** Real-Space Indexing ********************/

/**
 * Maps a local position in the vector-packed buffer to global grid coordinates.
 *
 * The block-cyclic distribution assigns each processor (rid, cid) a set of
 * B x B blocks from the N x N grid. The local array stores these blocks
 * contiguously, with the column dimension vector-packed.
 *
 * Given local position (local_row, local_col) on processor (rid, cid),
 * returns (global_row, global_col) in the N x N grid.
 */
inline void real_to_global(int rid, int cid, int lr, int lc, int& gr, int& gc) {
    int block_cyclic_col = packed_to_block_cyclic(lc);
    gr = (lr / B_DIM) * B_DIM * P_DIM + rid * B_DIM + (lr % B_DIM);
    gc = (block_cyclic_col / B_DIM) * B_DIM * P_DIM + cid * B_DIM + (block_cyclic_col % B_DIM);
}

/**
 * Inverse of real_to_global: given global grid coordinates (global_row, global_col),
 * find which processor (rid, cid) holds that element and at what local position
 * (local_row, local_col) in the vector-packed buffer.
 */
inline void global_to_real(int gr, int gc, int& rid, int& cid, int& lr, int& lc) {
    rid = (gr / B_DIM) % P_DIM;
    int local_row_block = gr / (B_DIM * P_DIM);
    lr = local_row_block * B_DIM + (gr % B_DIM);

    cid = (gc / B_DIM) % P_DIM;
    int local_col_block = gc / (B_DIM * P_DIM);
    int block_cyclic_col = local_col_block * B_DIM + (gc % B_DIM);
    lc = block_cyclic_to_packed(block_cyclic_col);
}

/**
 * Maps local position (ld, lr, lc) on processor (rid, cid, did)
 * to global grid coordinates (gd, gr, gc). Only the column dimension (lc) is
 * vector-packed; depth and row use the block-cyclic layout directly.
 */
inline void real_to_global_3d(int rid, int cid, int did, int ld, int lr, int lc, int& gd, int& gr, int& gc) {
    int block_cyclic_col = packed_to_block_cyclic(lc);
    gd = (ld / B_DIM) * B_DIM * P_DIM + did * B_DIM + (ld % B_DIM);
    gr = (lr / B_DIM) * B_DIM * P_DIM + rid * B_DIM + (lr % B_DIM);
    gc = (block_cyclic_col / B_DIM) * B_DIM * P_DIM + cid * B_DIM + (block_cyclic_col % B_DIM);
}

/**
 * Inverse of real_to_global_3d: given global coordinates (gd, gr, gc), find
 * which processor (rid, cid, did) holds it and at what local position (ld, lr, lc).
 */
inline void global_to_real_3d(int gd, int gr, int gc, int& rid, int& cid, int& did, int& ld, int& lr, int& lc) {
    did = (gd / B_DIM) % P_DIM;
    ld = (gd / (B_DIM * P_DIM)) * B_DIM + (gd % B_DIM);

    rid = (gr / B_DIM) % P_DIM;
    lr = (gr / (B_DIM * P_DIM)) * B_DIM + (gr % B_DIM);

    cid = (gc / B_DIM) % P_DIM;
    int block_cyclic_col = (gc / (B_DIM * P_DIM)) * B_DIM + (gc % B_DIM);
    lc = block_cyclic_to_packed(block_cyclic_col);
}

/******************** Reciprocal-Space Indexing ********************/

/**
 * lp (local permutation) and ct (cooley-tukey) are the two permutations
 * that describe how our distributed FFT scrambles its output relative to a standard DFT.
 *
 * lp undoes the stride-VP (VEC/P_DIM) interleaving from plan 1:
 *   plan 1 computes VP interleaved M-point FFTs at stride VP, so element i
 *   in the local row maps to position (i/VP) + (i%VP)*M in frequency space.
 *
 * ct is the Cooley-Tukey digit reversal from the two-stage decomposition N = M * VEC:
 *   the VEC-point FFT (plan 0) produces the "inner" index and the M-point FFT (plan 1)
 *   produces the "outer" index, giving ct(p) = (p%M)*VEC + (p/M).
 *
 * lp_inv and ct_inv are the inverses of lp and ct respectively.
 */
inline int lp(int i) {
    const int VP = VEC / P_DIM;
    const int M = B_DIM * P_DIM;
    return (i / VP) + (i % VP) * M;
}

inline int ct(int p) {
    const int M = B_DIM * P_DIM;
    return (p % M) * VEC + (p / M);
}

inline int lp_inv(int j) {
    const int VP = VEC / P_DIM;
    const int M = B_DIM * P_DIM;
    return (j % M) * VP + (j / M);
}

inline int ct_inv(int k) {
    const int M = B_DIM * P_DIM;
    return (k % VEC) * M + (k / VEC);
}

/**
 * Maps a local output position after the 2D distributed FFT to global frequency indices.
 *
 * After the forward FFT, element (local_row, local_col) on processor (rid, cid)
 * corresponds to frequency (ky, kx) in the global N x N DFT. The transpose from
 * the repack step is already applied, so the caller can directly index: A[ky][kx].
 *
 * Why the cross-mapping: local_row was processed by the x-direction FFT (using cid),
 * and local_col was processed by the y-direction FFT (using rid) after the
 * repack-transpose swapped rows and cols.
 */
inline void reciprocal_to_global(int rid, int cid, int local_row, int local_col, int& ky, int& kx) {
    kx = ct(cid * LOCAL_DIM + lp(local_row));
    ky = ct(rid * LOCAL_DIM + lp(local_col));
}

/**
 * Inverse of reciprocal_to_global: given a global frequency (ky, kx), find which
 * processor (rid, cid) holds it and at what local position (local_row, local_col).
 */
inline void global_to_reciprocal(int ky, int kx, int& rid, int& cid, int& local_row, int& local_col) {
    int px = ct_inv(kx);
    cid = px / LOCAL_DIM;
    local_row = lp_inv(px % LOCAL_DIM);

    int py = ct_inv(ky);
    rid = py / LOCAL_DIM;
    local_col = lp_inv(py % LOCAL_DIM);
}

/**
 * Maps a local output position after the 3D distributed FFT to global frequency indices.
 *
 * After the forward FFT, element (ld, lr, lc) on processor (rid, cid, did)
 * corresponds to frequency (kd, kr, kc) in the global N x N x N DFT. The two
 * repack-transposes (xy then yz) are already applied, so the caller can directly
 * index: A[kd][kr][kc].
 *
 * Why the cross-mapping: the two transposes permute which local axis maps to which
 * frequency axis:
 *   lr went through x-direction (cid) -> kc
 *   ld went through y-direction (rid) -> kr
 *   lc went through z-direction (did) -> kd
 */
inline void reciprocal_to_global_3d(int rid, int cid, int did, int ld, int lr, int lc, int& kd, int& kr, int& kc) {
    kc = ct(cid * LOCAL_DIM + lp(lr));
    kr = ct(rid * LOCAL_DIM + lp(ld));
    kd = ct(did * LOCAL_DIM + lp(lc));
}

/**
 * Inverse of reciprocal_to_global_3d: given a global frequency (kd, kr, kc), find
 * which processor (rid, cid, did) holds it and at what local position (ld, lr, lc).
 */
inline void global_to_reciprocal_3d(int kd, int kr, int kc, int& rid, int& cid, int& did, int& ld, int& lr, int& lc) {
    int pc = ct_inv(kc);
    cid = pc / LOCAL_DIM;
    lr = lp_inv(pc % LOCAL_DIM);

    int pr = ct_inv(kr);
    rid = pr / LOCAL_DIM;
    ld = lp_inv(pr % LOCAL_DIM);

    int pd = ct_inv(kd);
    did = pd / LOCAL_DIM;
    lc = lp_inv(pd % LOCAL_DIM);
}

#endif // __ZMODEL__UTILS__
