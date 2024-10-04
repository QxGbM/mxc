#pragma once

#include <complex>
#include <map>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <mpi.h>
#include <nccl.h>

/*template <typename T> class deviceMatrixDesc_t {
public:
  cublasHandle_t cublasH;
  cudaStream_t stream;

  thrust::device_vector<long long> ARowOffset_vec;
  thrust::device_vector<long long> ARows_vec;
  thrust::device_vector<long long> ACols_vec;
  thrust::device_vector<long long> ADistCols_vec;
  thrust::device_vector<long long> AInd_vec;

  thrust::device_vector<T*> A_ss_vec;
  thrust::device_vector<T*> A_sr_vec;
  thrust::device_vector<T*> A_rs_vec;
  thrust::device_vector<T*> A_rr_vec;
  thrust::device_vector<T*> A_sr_rows_vec;
  thrust::device_vector<T*> U_cols_vec;
  thrust::device_vector<T*> U_R_vec;
  thrust::device_vector<T*> V_rows_vec;
  thrust::device_vector<T*> V_R_vec;
  thrust::device_vector<T*> B_ind_vec;
  thrust::device_vector<T*> B_cols_vec;
  thrust::device_vector<T*> B_R_vec;
  thrust::device_vector<T*> AC_ind_vec;

  thrust::device_vector<T> Avec;
  thrust::device_vector<T> Uvec;
  thrust::device_vector<T> Vvec;
  thrust::device_vector<T> Bvec;
  thrust::device_vector<T> ACvec;
  
  thrust::device_vector<int> Ipiv;
  thrust::device_vector<int> Info;
};*/

class ColCommMPI;

void init_gpu_envs(cudaStream_t* stream, cublasHandle_t* cublasH, std::map<const MPI_Comm, ncclComm_t>& nccl_comms, const std::vector<MPI_Comm>& comms, MPI_Comm world = MPI_COMM_WORLD);
void finalize_gpu_envs(cudaStream_t stream, cublasHandle_t cublasH, std::map<const MPI_Comm, ncclComm_t>& nccl_comms);

void compute_factorize(cublasHandle_t cublasH, long long bdim, long long rank, std::complex<double>* A, std::complex<double>* R, const std::complex<double>* Q, long long ldim, long long lrank, const std::complex<double>* L, const ColCommMPI& comm, const std::map<const MPI_Comm, ncclComm_t>& nccl_comms);
