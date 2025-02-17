#pragma once

#include <gpu_handles.cuh>
#include <complex>
#include <cuComplex.h>

class ColCommMPI;
struct deviceMatrixDesc_t {
  long long bdim = 0;
  long long rank = 0;
  long long lenM = 0;
  long long lenN = 0;
  long long diag_offset = 0;
  long long lenA = 0;
  long long lower_offset = 0;
  long long reducLen = 0;

  cuDoubleComplex** A_ss = nullptr;
  cuDoubleComplex** A_sr = nullptr;
  cuDoubleComplex** A_rs = nullptr;
  cuDoubleComplex** A_rr = nullptr;
  cuDoubleComplex** A_sr_rows = nullptr;
  cuDoubleComplex** A_dst = nullptr;
  const cuDoubleComplex** A_unsort = nullptr;

  cuDoubleComplex** U_cols = nullptr;
  cuDoubleComplex** U_R = nullptr;
  cuDoubleComplex** V_rows = nullptr;
  cuDoubleComplex** V_R = nullptr;

  cuDoubleComplex** B_ind = nullptr;
  cuDoubleComplex** B_cols = nullptr;
  cuDoubleComplex** B_R = nullptr;

  cuDoubleComplex** X_cols = nullptr;
  cuDoubleComplex** Y_R_cols = nullptr;

  cuDoubleComplex** AC_X = nullptr;
  cuDoubleComplex** AC_X_R = nullptr;
  cuDoubleComplex** AC_ind = nullptr;

  cuDoubleComplex* Adata = nullptr;
  cuDoubleComplex* Udata = nullptr;
  cuDoubleComplex* Vdata = nullptr;
  cuDoubleComplex* Bdata = nullptr;

  cuDoubleComplex* ACdata = nullptr;
  cuDoubleComplex* Xdata = nullptr;
  cuDoubleComplex* Ydata = nullptr;
  cuDoubleComplex* ONEdata = nullptr;
  
  int* Ipiv = nullptr;
  int* Info = nullptr;

  long long LenComms = 0;
  long long* Neighbor = nullptr;
  long long* NeighborRoots = nullptr;
  ncclComm_t* NeighborComms = nullptr;
  ncclComm_t MergeComm = nullptr;
  ncclComm_t DupComm = nullptr;
};

void createMatrixDesc(deviceMatrixDesc_t* desc, long long bdim, long long rank, deviceMatrixDesc_t lower, const ColCommMPI& comm, const ncclComms nccl_comms);
void destroyMatrixDesc(deviceMatrixDesc_t desc);

void copyDataInMatrixDesc(deviceMatrixDesc_t desc, const std::complex<double>* A, const std::complex<double>* U, cudaStream_t stream);
void copyDataOutMatrixDesc(deviceMatrixDesc_t desc, std::complex<double>* A, std::complex<double>* V, cudaStream_t stream);

void compute_factorize(deviceHandle_t handle, deviceMatrixDesc_t A, deviceMatrixDesc_t Al);
void compute_forward_substitution(deviceHandle_t handle, deviceMatrixDesc_t A, const std::complex<double>* X);
void compute_backward_substitution(deviceHandle_t handle, deviceMatrixDesc_t A, std::complex<double>* X);
void matSolvePreconditionDeviceH2(deviceHandle_t handle, long long levels, deviceMatrixDesc_t A[], std::complex<double>* devX);

int check_info(deviceMatrixDesc_t A, const ColCommMPI& comm);

