
#include <build_tree.hpp>
#include <basis.hpp>
#include <comm.hpp>

#include <cmath>
#include <algorithm>

void get_bounds(const double* bodies, long long nbodies, double R[], double C[]) {
  double Xmin[3];
  double Xmax[3];
  Xmin[0] = Xmax[0] = bodies[0];
  Xmin[1] = Xmax[1] = bodies[1];
  Xmin[2] = Xmax[2] = bodies[2];

  for (long long i = 1; i < nbodies; i++) {
    const double* x_bi = &bodies[i * 3];
    Xmin[0] = std::min(x_bi[0], Xmin[0]);
    Xmin[1] = std::min(x_bi[1], Xmin[1]);
    Xmin[2] = std::min(x_bi[2], Xmin[2]);

    Xmax[0] = std::max(x_bi[0], Xmax[0]);
    Xmax[1] = std::max(x_bi[1], Xmax[1]);
    Xmax[2] = std::max(x_bi[2], Xmax[2]);
  }

  C[0] = (Xmin[0] + Xmax[0]) / 2.;
  C[1] = (Xmin[1] + Xmax[1]) / 2.;
  C[2] = (Xmin[2] + Xmax[2]) / 2.;

  double d0 = Xmax[0] - Xmin[0];
  double d1 = Xmax[1] - Xmin[1];
  double d2 = Xmax[2] - Xmin[2];

  R[0] = (d0 == 0. && Xmin[0] == 0.) ? 0. : (1.e-8 + d0 / 2.);
  R[1] = (d1 == 0. && Xmin[1] == 0.) ? 0. : (1.e-8 + d1 / 2.);
  R[2] = (d2 == 0. && Xmin[2] == 0.) ? 0. : (1.e-8 + d2 / 2.);
}

void buildTree(Cell* cells, double* bodies, long long nbodies, long long levels) {
  cells[0].Body[0] = 0;
  cells[0].Body[1] = nbodies;
  get_bounds(bodies, nbodies, cells[0].R.data(), cells[0].C.data());

  long long nleaf = (long long)1 << levels;
  long long len = 1;
  for (long long i = 0; i < nleaf - 1; i++) {
    Cell& ci = cells[i];
    long long sdim = 0;
    double maxR = ci.R[0];
    if (ci.R[1] > maxR)
    { sdim = 1; maxR = ci.R[1]; }
    if (ci.R[2] > maxR)
    { sdim = 2; maxR = ci.R[2]; }

    long long i_begin = ci.Body[0];
    long long i_end = ci.Body[1];
    long long nbody_i = i_end - i_begin;

    std::array<double, 3>* bodies3 = reinterpret_cast<std::array<double, 3>*>(&bodies[i_begin * 3]);
    std::array<double, 3>* bodies3_end = reinterpret_cast<std::array<double, 3>*>(&bodies[i_end * 3]);
    std::sort(bodies3, bodies3_end, 
      [=](std::array<double, 3>& i, std::array<double, 3>& j) { return i[sdim] < j[sdim]; });
    long long loc = i_begin + nbody_i / 2;

    Cell& c0 = cells[len];
    Cell& c1 = cells[len + 1];
    ci.Child[0] = len;
    ci.Child[1] = len + 2;
    len = len + 2;

    c0.Body[0] = i_begin;
    c0.Body[1] = loc;
    c0.Parent = i;
    c1.Body[0] = loc;
    c1.Body[1] = i_end;
    c1.Parent = i;

    get_bounds(&bodies[i_begin * 3], loc - i_begin, c0.R.data(), c0.C.data());
    get_bounds(&bodies[loc * 3], i_end - loc, c1.R.data(), c1.C.data());
  }
}

int admis_check(double theta, const double C1[], const double C2[], const double R1[], const double R2[]) {
  double dCi[3];
  dCi[0] = C1[0] - C2[0];
  dCi[1] = C1[1] - C2[1];
  dCi[2] = C1[2] - C2[2];

  dCi[0] = dCi[0] * dCi[0];
  dCi[1] = dCi[1] * dCi[1];
  dCi[2] = dCi[2] * dCi[2];

  double dRi[3];
  dRi[0] = R1[0] * R1[0];
  dRi[1] = R1[1] * R1[1];
  dRi[2] = R1[2] * R1[2];

  double dRj[3];
  dRj[0] = R2[0] * R2[0];
  dRj[1] = R2[1] * R2[1];
  dRj[2] = R2[2] * R2[2];

  double dC = dCi[0] + dCi[1] + dCi[2];
  double dR = (dRi[0] + dRi[1] + dRi[2] + dRj[0] + dRj[1] + dRj[2]) * theta;
  return (int)(dC > dR);
}

void getList(char NoF, std::vector<std::pair<long long, long long>>& rels, const Cell cells[], long long i, long long j, double theta) {
  int admis = admis_check(theta, cells[i].C.data(), cells[j].C.data(), cells[i].R.data(), cells[j].R.data());
  int write_far = NoF == 'F' || NoF == 'f';
  int write_near = NoF == 'N' || NoF == 'n';
  if (admis ? write_far : write_near)
    rels.emplace_back(i, j);
  
  if (!admis && cells[i].Child[0] >= 0 && cells[j].Child[0] >= 0)
    for (long long k = cells[i].Child[0]; k < cells[i].Child[1]; k++)
      for (long long l = cells[j].Child[0]; l < cells[j].Child[1]; l++)
        getList(NoF, rels, cells, k, l, theta);
}

Cell::Cell() {
  Child[0] = Child[1] = Parent = -1;
  Body[0] = Body[1] = -1;
  C[0] = C[1] = C[2] = 0.;
  R[0] = R[1] = R[2] = 0.;
}

CSR::CSR(char NoF, long long ncells, const Cell* cells, double theta) {
  std::vector<std::pair<long long, long long>> LIL;
  getList(NoF, LIL, cells, 0, 0, theta);
  std::sort(LIL.begin(), LIL.end());

  long long len = LIL.size();
  RowIndex.resize(ncells + 1);
  ColIndex.resize(len);
  std::transform(LIL.begin(), LIL.end(), ColIndex.begin(), 
    [](const std::pair<long long, long long>& i) { return i.second; });

  RowIndex[0] = 0;
  for (long long n = 1; n <= ncells; n++)
    RowIndex[n] = std::distance(LIL.begin(), 
      std::find_if(LIL.begin() + RowIndex[n - 1], LIL.end(), 
        [=](const std::pair<long long, long long>& i) { return n <= i.first; }));
}
