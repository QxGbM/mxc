#pragma once

#include <complex>
#include <map>
#include <vector>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <mpi.h>
#include <nccl.h>

#define STD_CTYPE std::complex<double>
#define THRUST_CTYPE thrust::complex<double>
#define CUDA_CTYPE cuDoubleComplex

class ColCommMPI;
class devicePreconditioner_t {
public:
  long long bdim = 0;
  long long rank = 0;
  long long diag_offset = 0;
  long long reducLen = 0;
  long long lowerTriLen = 0;

  CUDA_CTYPE** A_ss = nullptr;
  CUDA_CTYPE** A_sr = nullptr;
  CUDA_CTYPE** A_rs = nullptr;
  CUDA_CTYPE** A_rr = nullptr;
  CUDA_CTYPE** A_sr_rows = nullptr;
  const CUDA_CTYPE** A_unsort = nullptr;

  CUDA_CTYPE** U_cols = nullptr;
  CUDA_CTYPE** U_R = nullptr;
  CUDA_CTYPE** V_rows = nullptr;
  CUDA_CTYPE** V_R = nullptr;

  CUDA_CTYPE** X_rows = nullptr;
  CUDA_CTYPE** X_R_rows = nullptr;
  CUDA_CTYPE** Y_cols = nullptr;
  CUDA_CTYPE** Y_R_cols = nullptr;

  CUDA_CTYPE** B_ind = nullptr;
  CUDA_CTYPE** B_cols = nullptr;
  CUDA_CTYPE** B_R = nullptr;
  CUDA_CTYPE** AC_ind = nullptr;
  CUDA_CTYPE** AC_X = nullptr;
  CUDA_CTYPE** L_dst = nullptr;
  long long* Xlocs = nullptr;

  CUDA_CTYPE* Adata = nullptr;
  CUDA_CTYPE* Udata = nullptr;
  CUDA_CTYPE* Vdata = nullptr;
  CUDA_CTYPE* Bdata = nullptr;
  CUDA_CTYPE* ACdata = nullptr;

  CUDA_CTYPE* Xdata = nullptr;
  CUDA_CTYPE* Ydata = nullptr;
  
  int* Ipiv = nullptr;
  int* Info = nullptr;
};

class hostMatrix_t {
public:
  long long lenA;
  CUDA_CTYPE* Adata;
};

void initGpuEnvs(cudaStream_t* memory_stream, cudaStream_t* compute_stream, cublasHandle_t* cublasH, std::map<const MPI_Comm, ncclComm_t>& nccl_comms, const std::vector<MPI_Comm>& comms, MPI_Comm world = MPI_COMM_WORLD);
void finalizeGpuEnvs(cudaStream_t memory_stream, cudaStream_t compute_stream, cublasHandle_t cublasH, std::map<const MPI_Comm, ncclComm_t>& nccl_comms);

void createMatrixDesc(devicePreconditioner_t* desc, long long bdim, long long rank, devicePreconditioner_t lower, const ColCommMPI& comm);
void destroyMatrixDesc(devicePreconditioner_t desc);

void createHostMatrix(hostMatrix_t* h, long long bdim, long long lenA);
void destroyHostMatrix(hostMatrix_t h);

void copyDataInMatrixDesc(devicePreconditioner_t desc, long long lenA, const STD_CTYPE* A, long long lenU, const STD_CTYPE* U, cudaStream_t stream);
void copyDataOutMatrixDesc(devicePreconditioner_t desc, long long lenA, STD_CTYPE* A, long long lenV, STD_CTYPE* V, cudaStream_t stream);

void compute_factorize(devicePreconditioner_t A, devicePreconditioner_t Al, cudaStream_t stream, cublasHandle_t cublasH, const ColCommMPI& comm, const std::map<const MPI_Comm, ncclComm_t>& nccl_comms);
void compute_forward_substitution(devicePreconditioner_t A, devicePreconditioner_t Al, cudaStream_t stream, cublasHandle_t cublasH, const ColCommMPI& comm, const std::map<const MPI_Comm, ncclComm_t>& nccl_comms);
