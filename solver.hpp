#pragma once

#include <build_tree.hpp>
#include <comm-mpi.hpp>
#include <h2matrix.hpp>
#include <h-matrix.hpp>
#include <kernel.hpp>
#include <device_factorize.cuh>
#include <device_csr_matrix.cuh>

class H2MatrixSolver {
public:
  long long levels;
  std::vector<H2Matrix> A;
  std::vector<ColCommMPI> comm;
  std::vector<MPI_Comm> allocedComm;

  std::vector<CsrMatVecDesc_t> A_mv;

  std::vector<deviceMatrixDesc_t> desc;
  cuDoubleComplex* X_dev;

  std::pair<long long, long long> local_bodies;
  std::vector<double> resid;
  long long iters;
  
  H2MatrixSolver();
  H2MatrixSolver(const Accessor& eval_d, const MatrixAccessor& eval, double epi, long long rank, long long leveled_rank, const std::vector<Cell>& cells, double theta, const double bodies[], long long levels, MPI_Comm world = MPI_COMM_WORLD);
  void init_gpu_handles(const ncclComms nccl_comms);

  void allocSparseMV(deviceHandle_t handle, const ncclComms nccl_comms);
  void matVecMulSp(deviceHandle_t handle, std::complex<double> X[]);

  void matVecMul(std::complex<double> X[]);
  void factorizeM();
  void factorizeDeviceM(deviceHandle_t handle);
  void solvePrecondition(std::complex<double> X[]);
  void solvePreconditionDevice(deviceHandle_t handle, std::complex<double> X[]);
  void solveGMRES(double tol, H2MatrixSolver& M, std::complex<double> X[], const std::complex<double> B[], long long inner_iters, long long outer_iters);
  void solveGMRESDevice(deviceHandle_t handle, double tol, H2MatrixSolver& M, std::complex<double> X[], const std::complex<double> B[], long long inner_iters, long long outer_iters, const ncclComms nccl_comms);

  void free_all_comms();
  void freeSparseMV();
  void free_gpu_handles();
  static double solveRelErr(long long lenX, const std::complex<double> X[], const std::complex<double> ref[], MPI_Comm world = MPI_COMM_WORLD);
};
