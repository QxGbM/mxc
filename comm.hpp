
#pragma once

#include "mpi.h"
#include <vector>
#include <cstdint>
#include <complex>

class CSR;
class Cell;

class CellComm {
public:
  int64_t Proc;
  std::vector<std::pair<int64_t, int64_t>> ProcBoxes;
  
  std::vector<std::pair<int, MPI_Comm>> Comm_box;
  MPI_Comm Comm_share, Comm_merge;

  std::pair<double, double>* timer;

  int64_t iLocal(int64_t iglobal) const;
  int64_t iGlobal(int64_t ilocal) const;
  int64_t oLocal() const;
  int64_t oGlobal() const;
  int64_t lenLocal() const;
  int64_t lenNeighbors() const;

  void level_merge(std::complex<double>* data, int64_t len) const;
  void dup_bcast(double* data, int64_t len) const;
  void dup_bcast(std::complex<double>* data, int64_t len) const;
  void neighbor_bcast(double* data, const int64_t box_dims[]) const;
  void neighbor_bcast(std::complex<double>* data, const int64_t box_dims[]) const;

  void neighbor_bcast_sizes(int64_t* data) const;

  void record_mpi() const;
};

void buildComm(CellComm* comms, int64_t ncells, const Cell* cells, const CSR* cellFar, const CSR* cellNear, int64_t levels);

void cellComm_free(CellComm* comms, int64_t levels);

