#pragma once

#include <build_tree.hpp>
#include <comm-mpi.hpp>
#include <h2matrix.hpp>
#include <kernel.hpp>
#include <factorize.cuh>
#include <device_csr_matrix.cuh>

class H2MatrixSolver {
private:
  long long levels;
  std::vector<H2Matrix> A;
  std::vector<ColCommMPI> comm;
  std::vector<MPI_Comm> allocedComm;

  std::vector<CsrMatVecDesc_t> A_mv;

  deviceHandle_t handle;
  ncclComms nccl_comms;
  std::vector<deviceMatrixDesc_t> desc;
  CUDA_CTYPE* X_dev;

public:
  std::pair<long long, long long> local_bodies;
  std::vector<double> resid;
  long long iters;
  
  H2MatrixSolver();
  H2MatrixSolver(const MatrixAccessor& eval, double epi, long long rank, long long leveled_rank, const std::vector<Cell>& cells, double theta, const double bodies[], long long levels, MPI_Comm world = MPI_COMM_WORLD);
  void init_gpu_handles();
  void move_data_gpu();

  void allocSparseMV();
  void matVecMulSp(std::complex<double> X[]);

  void matVecMul(std::complex<double> X[]);
  void factorizeM();
  void factorizeDeviceM();
  void solvePrecondition(std::complex<double> X[]);
  void solvePreconditionDevice(std::complex<double> X[]);
  void solveGMRES(double tol, H2MatrixSolver& M, std::complex<double> X[], const std::complex<double> B[], long long inner_iters, long long outer_iters);

  void free_all_comms();
  void freeSparseMV();
  void free_gpu_handles();
  static double solveRelErr(long long lenX, const std::complex<double> X[], const std::complex<double> ref[], MPI_Comm world = MPI_COMM_WORLD);
};
