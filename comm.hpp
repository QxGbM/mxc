
#pragma once

#include "mpi.h"

#include <vector>
#include <utility>
#include <cstdint>
#include <cstddef>

class CellComm {
public:
  int64_t Proc;
  std::vector<std::pair<int64_t, int64_t>> ProcBoxes;
  std::vector<std::pair<int64_t, int64_t>> LocalChild, LocalParent;
  
  std::vector<std::pair<int, MPI_Comm>> Comm_box;
  MPI_Comm Comm_share, Comm_merge;

  std::pair<double, double>* timer;

  int64_t iLocal(int64_t iglobal) const;
  int64_t iGlobal(int64_t ilocal) const;

  void level_merge(double* data, int64_t len) const;
  void dup_bcast(double* data, int64_t len) const;

  void record_mpi() const;
};

struct CSR;

void buildComm(struct CellComm* comms, int64_t ncells, const struct Cell* cells, const CSR* cellFar, const CSR* cellNear, int64_t levels);

void cellComm_free(struct CellComm* comms, int64_t levels);

void relations(CSR rels[], const CSR* cellRel, int64_t levels, const struct CellComm* comm);

void content_length(int64_t* local, int64_t* neighbors, int64_t* local_off, const struct CellComm* comm);

void neighbor_bcast_sizes_cpu(int64_t* data, const struct CellComm* comm);

void neighbor_bcast_cpu(double* data, int64_t seg, const struct CellComm* comm);

void neighbor_reduce_cpu(double* data, int64_t seg, const struct CellComm* comm);

