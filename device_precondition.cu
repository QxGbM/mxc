
#include <device_factorize.cuh>
#include <comm-mpi.hpp>

#include <numeric>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/tuple.h>
#include <thrust/transform.h>
#include <thrust/gather.h>
#include <thrust/partition.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/inner_product.h>

struct keysD {
  long long D;
  keysD(long long D) : D(D) {}
  __host__ __device__ bool operator()(thrust::tuple<long long, long long, long long, long long> x) const {
    return D + thrust::get<0>(x) == thrust::get<1>(x);
  }
};

template<class T> struct setDevicePtr {
  T* data;
  long long ldx, ldy, ldz;
  setDevicePtr(T* data, long long ldx, long long ldy = 0, long long ldz = 0) : 
    data(data), ldx(ldx), ldy(ldy), ldz(ldz) {}
  __host__ __device__ T* operator()(long long x) const {
    return data + x * ldx;
  }
  __host__ __device__ T* operator()(long long y, long long x) const {
    return data + (x * ldx + y * ldy);
  }
  __host__ __device__ T* operator()(thrust::tuple<long long, long long, long long> x) const {
    return data + (thrust::get<0>(x) * ldx + thrust::get<1>(x) * ldy + thrust::get<2>(x) * ldz);
  }
};

void createMatrixDesc(deviceMatrixDesc_t* desc, long long bdim, long long rank, deviceMatrixDesc_t lower, const ColCommMPI& comm, const ncclComms nccl_comms) {
  desc->bdim = bdim;
  desc->rank = rank;
  desc->diag_offset = comm.oLocal();
  desc->lower_offset = (comm.LowerX + lower.diag_offset) * lower.rank;
  long long lenA = desc->lenA = comm.ARowOffsets.back();
  long long M = desc->lenM = comm.lenLocal();
  long long N = desc->lenN = comm.lenNeighbors();

  thrust::device_vector<long long> ARowOffset(comm.ARowOffsets.begin(), comm.ARowOffsets.end());
  thrust::device_vector<long long> ARows(lenA, 0ll);
  thrust::device_vector<long long> ACols(comm.AColumns.begin(), comm.AColumns.end());
  thrust::device_vector<long long> ADistCols(lenA);
  thrust::device_vector<long long> AInd(lenA);
  
  auto one_iter = thrust::make_constant_iterator(1ll);
  auto A_iter = thrust::make_zip_iterator(ARows.begin(), ACols.begin(), ADistCols.begin(), AInd.begin());
  thrust::scatter(one_iter, one_iter + (M - 1), ARowOffset.begin() + 1, ARows.begin()); 
  thrust::inclusive_scan(ARows.begin(), ARows.end(), ARows.begin());
  thrust::exclusive_scan_by_key(ARows.begin(), ARows.end(), one_iter, ADistCols.begin(), 0ll);

  thrust::sequence(AInd.begin(), AInd.end(), 0);
  thrust::stable_partition(A_iter, A_iter + lenA, keysD(desc->diag_offset));

  desc->reducLen = 1ll + thrust::reduce(ADistCols.begin(), ADistCols.end(), 0ll, thrust::maximum<long long>());
  long long lenLA = comm.LowerIndA.size();
  const thrust::tuple<long long, long long, long long>* commLA = reinterpret_cast<const thrust::tuple<long long, long long, long long>*>(comm.LowerIndA.data());
  thrust::device_vector<thrust::tuple<long long, long long, long long>> LInd(commLA, commLA + lenLA);

  cudaMalloc(reinterpret_cast<void**>(&desc->A_ss), lenA * sizeof(cuDoubleComplex*));
  cudaMalloc(reinterpret_cast<void**>(&desc->A_sr), lenA * sizeof(cuDoubleComplex*));
  cudaMalloc(reinterpret_cast<void**>(&desc->A_rs), lenA * sizeof(cuDoubleComplex*));
  cudaMalloc(reinterpret_cast<void**>(&desc->A_rr), lenA * sizeof(cuDoubleComplex*));
  cudaMalloc(reinterpret_cast<void**>(&desc->A_sr_rows), lenA * sizeof(cuDoubleComplex*));
  cudaMalloc(reinterpret_cast<void**>(&desc->A_dst), lenLA * sizeof(cuDoubleComplex*));
  cudaMalloc(reinterpret_cast<void**>(&desc->A_unsort), lenA * sizeof(cuDoubleComplex*));

  cudaMalloc(reinterpret_cast<void**>(&desc->U_cols), lenA * sizeof(cuDoubleComplex*));
  cudaMalloc(reinterpret_cast<void**>(&desc->U_R), M * sizeof(cuDoubleComplex*));
  cudaMalloc(reinterpret_cast<void**>(&desc->V_rows), lenA * sizeof(cuDoubleComplex*));
  cudaMalloc(reinterpret_cast<void**>(&desc->V_R), M * sizeof(cuDoubleComplex*));

  cudaMalloc(reinterpret_cast<void**>(&desc->B_ind), N * sizeof(cuDoubleComplex*));
  cudaMalloc(reinterpret_cast<void**>(&desc->B_cols), lenA * sizeof(cuDoubleComplex*));
  cudaMalloc(reinterpret_cast<void**>(&desc->B_R), lenA * sizeof(cuDoubleComplex*));

  cudaMalloc(reinterpret_cast<void**>(&desc->X_cols), lenA * sizeof(cuDoubleComplex*));
  cudaMalloc(reinterpret_cast<void**>(&desc->Y_R_cols), lenA * sizeof(cuDoubleComplex*));

  cudaMalloc(reinterpret_cast<void**>(&desc->AC_X), lenA * sizeof(cuDoubleComplex*));
  cudaMalloc(reinterpret_cast<void**>(&desc->AC_X_R), lenA * sizeof(cuDoubleComplex*));
  cudaMalloc(reinterpret_cast<void**>(&desc->AC_ind), lenA * sizeof(cuDoubleComplex*));

  long long block = bdim * bdim;
  long long rblock = rank * rank;
  long long acc_len = desc->reducLen * M * std::max(rblock, bdim);

  cudaMalloc(reinterpret_cast<void**>(&desc->Adata), lenA * block * sizeof(cuDoubleComplex));
  cudaMalloc(reinterpret_cast<void**>(&desc->Udata), N * block * sizeof(cuDoubleComplex));
  cudaMalloc(reinterpret_cast<void**>(&desc->Vdata), M * block * sizeof(cuDoubleComplex));
  cudaMalloc(reinterpret_cast<void**>(&desc->Bdata), N * block * sizeof(cuDoubleComplex));
  cudaMalloc(reinterpret_cast<void**>(&desc->ACdata), acc_len * sizeof(cuDoubleComplex));

  cudaMalloc(reinterpret_cast<void**>(&desc->Xdata), N * bdim * sizeof(cuDoubleComplex));
  cudaMalloc(reinterpret_cast<void**>(&desc->Ydata), N * bdim * sizeof(cuDoubleComplex));
  cudaMalloc(reinterpret_cast<void**>(&desc->ONEdata), desc->reducLen * sizeof(cuDoubleComplex));
  cudaMalloc(reinterpret_cast<void**>(&desc->Ipiv), M * bdim * sizeof(int));
  cudaMalloc(reinterpret_cast<void**>(&desc->Info), M * sizeof(int));

  auto inc_iter = thrust::make_counting_iterator(0ll);
  auto rwise_diag_iter = thrust::make_permutation_iterator(AInd.begin(), ARows.begin());
  long long offset_SR = rank * bdim, offset_RS = rank, offset_RR = rank * (bdim + 1);

  thrust::transform(AInd.begin(), AInd.end(), thrust::device_ptr<cuDoubleComplex*>(desc->A_ss), setDevicePtr(desc->Adata, block));
  thrust::transform(AInd.begin(), AInd.end(), thrust::device_ptr<cuDoubleComplex*>(desc->A_sr), setDevicePtr(&(desc->Adata)[offset_SR], block));
  thrust::transform(AInd.begin(), AInd.end(), thrust::device_ptr<cuDoubleComplex*>(desc->A_rs), setDevicePtr(&(desc->Adata)[offset_RS], block));
  thrust::transform(AInd.begin(), AInd.end(), thrust::device_ptr<cuDoubleComplex*>(desc->A_rr), setDevicePtr(&(desc->Adata)[offset_RR], block));
  thrust::transform(rwise_diag_iter, rwise_diag_iter + lenA, thrust::device_ptr<cuDoubleComplex*>(desc->A_sr_rows), setDevicePtr(&(desc->Adata)[offset_SR], block));
  thrust::transform(LInd.begin(), LInd.end(), thrust::device_ptr<cuDoubleComplex*>(desc->A_dst), setDevicePtr(desc->Adata, block, bdim * lower.rank, lower.rank));
  thrust::transform(inc_iter, inc_iter + lenA, thrust::device_ptr<const cuDoubleComplex*>(desc->A_unsort), setDevicePtr(desc->Adata, block));

  thrust::transform(ACols.begin(), ACols.end(), thrust::device_ptr<cuDoubleComplex*>(desc->U_cols), setDevicePtr(desc->Udata, block));
  thrust::transform(ACols.begin(), ACols.begin() + M, thrust::device_ptr<cuDoubleComplex*>(desc->U_R), setDevicePtr(&(desc->Udata)[offset_SR], block));
  thrust::transform(ARows.begin(), ARows.end(), thrust::device_ptr<cuDoubleComplex*>(desc->V_rows), setDevicePtr(desc->Vdata, block));
  thrust::transform(inc_iter, inc_iter + M, thrust::device_ptr<cuDoubleComplex*>(desc->V_R), setDevicePtr(&(desc->Vdata)[offset_RS], block));

  thrust::transform(inc_iter, inc_iter + N, thrust::device_ptr<cuDoubleComplex*>(desc->B_ind), setDevicePtr(desc->Bdata, block));
  thrust::transform(ACols.begin(), ACols.end(), thrust::device_ptr<cuDoubleComplex*>(desc->B_cols), setDevicePtr(desc->Bdata, block));
  thrust::transform(ACols.begin(), ACols.end(), thrust::device_ptr<cuDoubleComplex*>(desc->B_R), setDevicePtr(&(desc->Bdata)[offset_SR], block));

  thrust::transform(ACols.begin(), ACols.end(), thrust::device_ptr<cuDoubleComplex*>(desc->X_cols), setDevicePtr(desc->Xdata, rank));
  thrust::transform(ACols.begin(), ACols.end(), thrust::device_ptr<cuDoubleComplex*>(desc->Y_R_cols), setDevicePtr(&(desc->Ydata)[offset_RS], bdim));

  thrust::transform(ARows.begin(), ARows.end(), ADistCols.begin(), thrust::device_ptr<cuDoubleComplex*>(desc->AC_X), setDevicePtr(desc->ACdata, M * rank, rank));
  thrust::transform(ARows.begin(), ARows.end(), ADistCols.begin(), thrust::device_ptr<cuDoubleComplex*>(desc->AC_X_R), setDevicePtr(&(desc->ACdata)[offset_RS], M * bdim, bdim));
  thrust::transform(ARows.begin(), ARows.end(), ADistCols.begin(), thrust::device_ptr<cuDoubleComplex*>(desc->AC_ind), setDevicePtr(desc->ACdata, M * rblock, rblock));
  
  thrust::fill(thrust::device_ptr<cuDoubleComplex>(desc->ONEdata), thrust::device_ptr<cuDoubleComplex>(&(desc->ONEdata)[desc->reducLen]), make_cuDoubleComplex(1., 0.));

  desc->Neighbor = reinterpret_cast<long long*>(std::malloc(comm.BoxOffsets.size() * sizeof(long long)));
  std::copy(comm.BoxOffsets.begin(), comm.BoxOffsets.end(), desc->Neighbor);

  desc->LenComms = comm.NeighborComm.size();
  if (desc->LenComms) {
    desc->NeighborRoots = reinterpret_cast<long long*>(std::malloc(desc->LenComms * sizeof(long long)));
    desc->NeighborComms = reinterpret_cast<ncclComm_t*>(std::malloc(desc->LenComms * sizeof(ncclComm_t)));

    std::transform(comm.NeighborComm.begin(), comm.NeighborComm.end(), desc->NeighborRoots, 
      [](const std::pair<int, MPI_Comm>& comm) { return static_cast<long long>(comm.first); });
    std::transform(comm.NeighborComm.begin(), comm.NeighborComm.end(), desc->NeighborComms, 
      [=](const std::pair<int, MPI_Comm>& comm) { return findNcclComm(comm.second, nccl_comms); });
  }

  desc->DupComm = findNcclComm(comm.DupComm, nccl_comms);
  desc->MergeComm = findNcclComm(comm.MergeComm, nccl_comms);
}

void destroyMatrixDesc(deviceMatrixDesc_t desc) {
  cudaFree(desc.A_ss);
  cudaFree(desc.A_sr);
  cudaFree(desc.A_rs);
  cudaFree(desc.A_rr);
  cudaFree(desc.A_sr_rows);
  cudaFree(desc.A_dst);
  cudaFree(desc.A_unsort);

  cudaFree(desc.U_cols);
  cudaFree(desc.U_R);
  cudaFree(desc.V_rows);
  cudaFree(desc.V_R);

  cudaFree(desc.B_ind);
  cudaFree(desc.B_cols);
  cudaFree(desc.B_R);
  cudaFree(desc.AC_ind);

  cudaFree(desc.X_cols);
  cudaFree(desc.Y_R_cols);
  cudaFree(desc.AC_X);
  cudaFree(desc.AC_X_R);

  cudaFree(desc.Adata);
  cudaFree(desc.Udata);
  cudaFree(desc.Vdata);
  cudaFree(desc.Bdata);
  cudaFree(desc.ACdata);

  cudaFree(desc.Xdata);
  cudaFree(desc.Ydata);
  cudaFree(desc.ONEdata);
  cudaFree(desc.Ipiv);
  cudaFree(desc.Info);

  if (desc.LenComms) {
    std::free(desc.NeighborRoots);
    std::free(desc.NeighborComms);
  }
}

void copyDataInMatrixDesc(deviceMatrixDesc_t desc, const std::complex<double>* A, const std::complex<double>* U, cudaStream_t stream) {
  long long block = desc.bdim * desc.bdim * sizeof(cuDoubleComplex);
  cudaMemcpyAsync(desc.Adata, A, block * desc.lenA, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(desc.Udata, U, block * desc.lenN, cudaMemcpyHostToDevice, stream);
}

void copyDataOutMatrixDesc(deviceMatrixDesc_t desc, std::complex<double>* A, std::complex<double>* V, cudaStream_t stream) {
  long long block = desc.bdim * desc.bdim * sizeof(cuDoubleComplex);
  cudaMemcpyAsync(A, desc.Adata, block * desc.lenA, cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(V, desc.Vdata, block * desc.lenM, cudaMemcpyDeviceToHost, stream);
}

int check_info(deviceMatrixDesc_t A, const ColCommMPI& comm) {
  long long M = comm.lenLocal();
  thrust::device_ptr<int> info_ptr(A.Info);
  int sum = thrust::inner_product(info_ptr, info_ptr + M, info_ptr, 0);
  return 0 < sum;
}
