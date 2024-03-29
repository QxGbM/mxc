
#pragma once

#include <vector>
#include <array>

class Cell {
public:
  long long Parent;
  std::array<long long, 2> Child;
  std::array<long long, 2> Body;
  std::array<double, 3> R;
  std::array<double, 3> C;

  Cell();
};

class CSR {
public:
  std::vector<long long> RowIndex;
  std::vector<long long> ColIndex;

  CSR(char NoF, long long ncells, const Cell* cells, double theta);
};

typedef std::vector<Cell> Cells;

void buildTree(Cell* cells, double* bodies, long long nbodies, long long levels);
