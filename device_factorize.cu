
#include <device_factorize.cuh>

#include <thrust/device_ptr.h>
#include <thrust/complex.h>
#include <thrust/gather.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

struct swapXY {
  long long M, B;
  swapXY(long long M, long long N) : M(M), B(M * N) {}
  __host__ __device__ long long operator()(long long i) const {
    long long x = i / B; long long y = i - x * B;
    long long z = y / M; long long w = y - z * M;
    return x * B + z + w * M;
  }
};

struct conjugateFunc {
  __host__ __device__ thrust::complex<float> operator()(const thrust::complex<float>& z) const {
    return thrust::conj(z);
  }
  __host__ __device__ thrust::complex<double> operator()(const thrust::complex<double>& z) const {
    return thrust::conj(z);
  }
};

template<class T> struct StridedBlock {
  long long M, B, LD;
  T **pA;
  StridedBlock(long long M, long long N, long long LD, T** pA) : M(M), B(M * N), LD(LD), pA(pA) {}
  __host__ __device__ T& operator()(long long i) const {
    long long x = i / B; long long y = i - x * B;
    long long z = y / M; long long w = y - z * M;
    return pA[x][z * LD + w];
  }
};

template<class T> struct copyFunc {
  const T** srcs;
  T** dsts;
  long long M, B, ls, ld;
  copyFunc(long long M, long long N, const T* srcs[], long long ls, T* dsts[], long long ld) :
    srcs(srcs), dsts(dsts), M(M), B(M * N), ls(ls), ld(ld) {}
  __host__ __device__ void operator()(long long i) const {
    long long x = i / B; long long y = i - x * B;
    long long z = y / M; long long w = y - z * M;
    T e = srcs[x][z * ls + w];
    dsts[x][z * ld + w] = e;
  }
};

void compute_factorize(deviceHandle_t handle, deviceMatrixDesc_t A, deviceMatrixDesc_t Al) {
  long long bdim = A.bdim;
  long long rank = A.rank;
  long long block = bdim * bdim;
  long long rblock = rank * rank;

  long long D = A.diag_offset;
  long long M = A.lenM;
  long long N = A.lenN;
  long long lenA = A.lenA;
  long long lenL = Al.lenA;

  cudaStream_t stream = handle->compute_stream;
  cublasHandle_t cublasH = handle->cublasH;

  long long rdim = bdim - rank;
  long long reduc_len = A.reducLen;
  int info_host = 0;
  std::complex<double> constants[3] = { 1., 0., -1. };
  cuDoubleComplex& one = reinterpret_cast<cuDoubleComplex&>(constants[0]);
  cuDoubleComplex& zero = reinterpret_cast<cuDoubleComplex&>(constants[1]); 
  cuDoubleComplex& minus_one = reinterpret_cast<cuDoubleComplex&>(constants[2]); 

  auto inc_iter = thrust::make_counting_iterator(0ll);
  auto mapV = thrust::make_transform_iterator(inc_iter, swapXY(bdim, bdim));
  thrust::device_ptr<thrust::complex<double>> Uptr(reinterpret_cast<thrust::complex<double>*>(&(A.Udata)[D * block]));
  thrust::device_ptr<thrust::complex<double>> Vptr(reinterpret_cast<thrust::complex<double>*>(A.Vdata));
  thrust::gather(thrust::cuda::par.on(stream), mapV, mapV + (block * M), thrust::make_transform_iterator(Uptr, conjugateFunc()), Vptr);
  
  if (0 < lenL) {
    long long len = Al.rank * Al.rank * lenL;
    thrust::for_each(thrust::cuda::par.on(stream), inc_iter, inc_iter + len, copyFunc(Al.rank, Al.rank, Al.A_unsort, Al.bdim, A.A_dst, bdim));
  }

  if (M == 1) {
    if (A.MergeComm)
      ncclAllReduce(const_cast<const cuDoubleComplex*>(A.Adata), A.Adata, block * lenA * 2, ncclDouble, ncclSum, A.MergeComm, stream);
    if (A.DupComm)
      ncclBroadcast(const_cast<const cuDoubleComplex*>(A.Adata), A.Adata, block * lenA * 2, ncclDouble, 0, A.DupComm, stream);
  }

  cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, bdim, bdim, bdim, &one, A.V_rows, bdim, A.A_ss, bdim, &zero, A.B_ind, bdim, M);
  cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, bdim, bdim, bdim, &one, A.V_rows, bdim, A.B_ind, bdim, &zero, A.A_ss, bdim, M);

  cublasZgetrfBatched(cublasH, rdim, A.A_rr, bdim, A.Ipiv, A.Info, M);
  cublasZgetrsBatched(cublasH, CUBLAS_OP_N, rdim, bdim, A.A_rr, bdim, A.Ipiv, A.V_R, bdim, &info_host, M);

  if (0 < rank) {
    cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, rdim, rank, bdim, &one, A.V_R, bdim, A.B_ind, bdim, &zero, A.A_rs, bdim, M);
    cudaMemsetAsync(A.ACdata, 0, reduc_len * M * rblock * sizeof(cuDoubleComplex), stream);

    for (long long i = M; i < lenA; i += N) {
      long long len = std::min(lenA - i, N);
      cublasZgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, &one, &(A.U_cols)[i], bdim, &(A.A_ss)[i], bdim, &zero, A.B_ind, bdim, len);
      cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, bdim, bdim, bdim, &one, &(A.V_rows)[i], bdim, A.B_ind, bdim, &zero, &(A.A_ss)[i], bdim, len);
    }

    cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, &minus_one, A.A_sr_rows, bdim, A.A_rs, bdim, &one, A.A_ss, bdim, lenA);
    cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rdim, rdim, bdim, &one, A.V_R, bdim, A.U_R, bdim, &zero, A.B_R, bdim, M);
    thrust::for_each(thrust::cuda::par.on(stream), inc_iter, inc_iter + (rdim * rank * M), copyFunc(rdim, rank, const_cast<const cuDoubleComplex**>(A.A_rs), bdim, &(A.B_ind)[D], bdim));
    
    ncclGroupStart();
    for (long long p = 0; p < A.LenComms; p++) {
      long long start = A.Neighbor[p] * block;
      long long len = A.Neighbor[p + 1] * block - start;
      ncclBroadcast(const_cast<const cuDoubleComplex*>(&(A.Bdata)[start]), &(A.Bdata)[start], len * 2, ncclDouble, A.NeighborRoots[p], A.NeighborComms[p], stream);
    }

    if (A.DupComm)
      ncclBroadcast(const_cast<const cuDoubleComplex*>(A.Bdata), A.Bdata, block * N * 2, ncclDouble, 0, A.DupComm, stream);
    ncclGroupEnd();

    if (M < lenA)
      cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, &minus_one, &(A.A_sr)[M], bdim, &(A.B_cols)[M], bdim, &one, &(A.A_ss)[M], bdim, lenA - M);

    for (long long i = M; i < lenA; i += N) {
      long long len = std::min(lenA - i, N);
      cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, rdim, rank, rdim, &one, &(A.B_R)[i], bdim, &(A.A_sr)[i], bdim, &zero, A.B_ind, bdim, len);
      cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, &minus_one, &(A.A_sr)[i], bdim, A.B_ind, bdim, &zero, &A.AC_ind[i], rank, len);
    }
    cublasZgemvStridedBatched(cublasH, CUBLAS_OP_N, M * rank, reduc_len, &one, A.ACdata, M * rblock, M * rank, A.ONEdata, 1, 0, &zero, A.Bdata, 1, M * rank, rank);

    thrust::device_ptr<thrust::complex<double>> ACptr(reinterpret_cast<thrust::complex<double>*>(A.Bdata));
    auto Aiter = thrust::make_transform_iterator(inc_iter, StridedBlock(rank, rank, bdim, reinterpret_cast<thrust::complex<double>**>(A.A_ss)));
    thrust::transform(thrust::cuda::par.on(stream), Aiter, Aiter + (rblock * M), ACptr, Aiter, thrust::plus<thrust::complex<double>>());
  }
}

void compute_forward_substitution(deviceHandle_t handle, deviceMatrixDesc_t A, const std::complex<double>* X) {
  long long bdim = A.bdim;
  long long rank = A.rank;
  long long rdim = bdim - rank;
  long long block = bdim * bdim;

  long long D = A.diag_offset;
  long long M = A.lenM;
  long long N = A.lenN;
  long long lenA = A.lenA;
  long long reduc_len = A.reducLen;

  cudaStream_t stream = handle->compute_stream;
  cublasHandle_t cublasH = handle->cublasH;

  std::complex<double> constants[3] = { 1., 0., -1. };
  cuDoubleComplex& one = reinterpret_cast<cuDoubleComplex&>(constants[0]);
  cuDoubleComplex& zero = reinterpret_cast<cuDoubleComplex&>(constants[1]);
  cuDoubleComplex& minus_one = reinterpret_cast<cuDoubleComplex&>(constants[2]);
  const cuDoubleComplex* X_in = reinterpret_cast<const cuDoubleComplex*>(&X[A.lower_offset]);

  cublasZgemmStridedBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, bdim, 1, bdim, &one, A.Vdata, bdim, block, X_in, bdim, bdim, &zero, &(A.Ydata)[D * bdim], bdim, bdim, M);

  if (1 < N) {
    ncclGroupStart();
    for (long long p = 0; p < A.LenComms; p++) {
      long long start = A.Neighbor[p] * bdim;
      long long len = A.Neighbor[p + 1] * bdim - start;
      ncclBroadcast(const_cast<const cuDoubleComplex*>(&(A.Ydata)[start]), &(A.Ydata)[start], len * 2, ncclDouble, A.NeighborRoots[p], A.NeighborComms[p], stream);
    }
    if (A.DupComm)
      ncclBroadcast(const_cast<const cuDoubleComplex*>(A.Ydata), A.Ydata, bdim * N * 2, ncclDouble, 0, A.DupComm, stream);
    ncclGroupEnd();
  }

  size_t sizeX = rank * sizeof(cuDoubleComplex);
  cudaMemcpy2DAsync(&(A.Xdata)[D * rank], sizeX, &(A.Ydata)[D * bdim], bdim * sizeof(cuDoubleComplex), sizeX, M, cudaMemcpyDeviceToDevice, stream);
  if (0 < rank && 0 < rdim) {
    cudaMemsetAsync(A.ACdata, 0, reduc_len * M * rank * sizeof(cuDoubleComplex), stream);
    cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, 1, rdim, &minus_one, A.A_sr, bdim, A.Y_R_cols, bdim, &zero, A.AC_X, rank, lenA);
    cublasZgemv(cublasH, CUBLAS_OP_N, M * rank, reduc_len, &one, A.ACdata, M * rank, A.ONEdata, 1, &one, &(A.Xdata)[D * rank], 1);
  }

  if (1 < N) {
    ncclGroupStart();
    for (long long p = 0; p < A.LenComms; p++) {
      long long start = A.Neighbor[p] * rank;
      long long len = A.Neighbor[p + 1] * rank - start;
      ncclBroadcast(const_cast<const cuDoubleComplex*>(&(A.Xdata)[start]), &(A.Xdata)[start], len * 2, ncclDouble, A.NeighborRoots[p], A.NeighborComms[p], stream);
    }
    if (A.DupComm)
      ncclBroadcast(const_cast<const cuDoubleComplex*>(A.Xdata), A.Xdata, rank * N * 2, ncclDouble, 0, A.DupComm, stream);
    ncclGroupEnd();
  }
}

void compute_backward_substitution(deviceHandle_t handle, deviceMatrixDesc_t A, std::complex<double>* X) {
  long long bdim = A.bdim;
  long long rank = A.rank;
  long long rdim = bdim - rank;
  long long block = bdim * bdim;

  long long D = A.diag_offset;
  long long M = A.lenM;
  long long N = A.lenN;
  long long lenA = A.lenA;
  long long reduc_len = A.reducLen;

  cudaStream_t stream = handle->compute_stream;
  cublasHandle_t cublasH = handle->cublasH;

  std::complex<double> constants[3] = { 1., 0., -1. };
  cuDoubleComplex& one = reinterpret_cast<cuDoubleComplex&>(constants[0]);
  cuDoubleComplex& zero = reinterpret_cast<cuDoubleComplex&>(constants[1]);
  cuDoubleComplex& minus_one = reinterpret_cast<cuDoubleComplex&>(constants[2]);
  cuDoubleComplex* X_out = reinterpret_cast<cuDoubleComplex*>(&X[A.lower_offset]);

  if (1 < N) {
    ncclGroupStart();
    for (long long p = 0; p < A.LenComms; p++) {
      long long start = A.Neighbor[p] * rank;
      long long len = A.Neighbor[p + 1] * rank - start;
      ncclBroadcast(const_cast<const cuDoubleComplex*>(&(A.Xdata)[start]), &(A.Xdata)[start], len * 2, ncclDouble, A.NeighborRoots[p], A.NeighborComms[p], stream);
    }
    if (A.DupComm)
      ncclBroadcast(const_cast<const cuDoubleComplex*>(A.Xdata), A.Xdata, rank * N * 2, ncclDouble, 0, A.DupComm, stream);
    ncclGroupEnd();
  }

  size_t sizeX = rank * sizeof(cuDoubleComplex);
  cudaMemcpy2DAsync(&(A.Ydata)[D * bdim], bdim * sizeof(cuDoubleComplex), &(A.Xdata)[D * rank], sizeX, sizeX, M, cudaMemcpyDeviceToDevice, stream);
  if (0 < rank && 0 < rdim) {
    cudaMemsetAsync(A.ACdata, 0, reduc_len * M * bdim * sizeof(cuDoubleComplex), stream);
    cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rdim, 1, rank, &minus_one, A.A_rs, bdim, A.X_cols, bdim, &zero, A.AC_X_R, bdim, lenA);
    cublasZgemv(cublasH, CUBLAS_OP_N, M * bdim, reduc_len, &one, A.ACdata, M * bdim, A.ONEdata, 1, &one, &(A.Ydata)[D * bdim], 1);
  }

  cublasZgemmStridedBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_C, 1, bdim, bdim, &one, &(A.Ydata)[D * bdim], 1, bdim, &(A.Udata)[D * block], bdim, block, &zero, X_out, 1, bdim, M);
}

void matSolvePreconditionDeviceH2(deviceHandle_t handle, long long levels, deviceMatrixDesc_t desc[], std::complex<double>* devX) {
  if (0 <= levels) {
    compute_forward_substitution(handle, desc[levels], devX);
    for (long long l = levels - 1; l >= 0; l--)
      compute_forward_substitution(handle, desc[l], reinterpret_cast<std::complex<double>*>(desc[l + 1].Xdata));

    for (long long l = 0; l < levels; l++)
      compute_backward_substitution(handle, desc[l], reinterpret_cast<std::complex<double>*>(desc[l + 1].Xdata));
    compute_backward_substitution(handle, desc[levels], devX);
  }
}
