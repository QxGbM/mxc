
#include <build_tree.hpp>
#include <cmath>
#include <numeric>
#include <algorithm>

void get_bounds(const double* bodies, long long nbodies, double R[], double C[]) {
  const std::array<double, 3>* bodies3 = reinterpret_cast<const std::array<double, 3>*>(&bodies[0]);
  const std::array<double, 3>* bodies3_end = reinterpret_cast<const std::array<double, 3>*>(&bodies[nbodies * 3]);

  double Xmin[3], Xmax[3];
  for (int i = 0; i < 3; i++) {
    auto minmax = std::minmax_element(bodies3, bodies3_end, 
      [=](const std::array<double, 3>& x, const std::array<double, 3>& y) { return x[i] < y[i]; });
    Xmin[i] = (*minmax.first)[i];
    Xmax[i] = (*minmax.second)[i];
  }

  std::transform(Xmin, &Xmin[3], Xmax, C, [](double min, double max) { return (min + max) * 0.5; });
  std::transform(Xmin, &Xmin[3], Xmax, R, [](double min, double max) { return (min == max && min == 0.) ? 0. : ((max - min) * 0.5 + 1.e-8); });
}

void buildBinaryTree(Cell* cells, double* bodies, long long nbodies, long long levels) {
  cells[0].Body[0] = 0;
  cells[0].Body[1] = nbodies;
  get_bounds(bodies, nbodies, cells[0].R.data(), cells[0].C.data());

  long long nleaf = (long long)1 << levels;
  for (long long i = 0; i < nleaf - 1; i++) {
    Cell& ci = cells[i];
    long long sdim = std::distance(&ci.R[0], std::max_element(&ci.R[0], &ci.R[3]));
    long long i_begin = ci.Body[0];
    long long i_end = ci.Body[1];

    std::array<double, 3>* bodies3 = reinterpret_cast<std::array<double, 3>*>(&bodies[i_begin * 3]);
    std::array<double, 3>* bodies3_end = reinterpret_cast<std::array<double, 3>*>(&bodies[i_end * 3]);
    std::sort(bodies3, bodies3_end, 
      [=](std::array<double, 3>& i, std::array<double, 3>& j) { return i[sdim] < j[sdim]; });

    long long len = (i << 1) + 1;
    Cell& c0 = cells[len];
    Cell& c1 = cells[len + 1];
    ci.Child[0] = len;
    ci.Child[1] = len + 2;

    long long loc = i_begin + (i_end - i_begin) / 2;
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

void getList(char NoF, std::vector<std::pair<long long, long long>>& rels, const Cell ci[], long long i, const Cell cj[], long long j, double theta) {
  double dC = std::transform_reduce(&ci[i].C[0], &ci[i].C[3], &cj[j].C[0], (double)0., std::plus<double>(), [](double x, double y) { return (x - y) * (x - y); });
  double dR1 = std::transform_reduce(&ci[i].R[0], &ci[i].R[3], &ci[i].R[0], (double)0., std::plus<double>(), std::multiplies<double>());
  double dR2 = std::transform_reduce(&cj[j].R[0], &cj[j].R[3], &cj[j].R[0], (double)0., std::plus<double>(), std::multiplies<double>());

  bool admis = dC > (theta * (dR1 + dR2));
  bool write_far = NoF == 'F' || NoF == 'f';
  bool write_near = NoF == 'N' || NoF == 'n';
  if (admis ? write_far : write_near)
    rels.emplace_back(i, j);
  
  if (!admis && ci[i].Child[0] >= 0 && cj[j].Child[0] >= 0)
    for (long long k = ci[i].Child[0]; k < ci[i].Child[1]; k++)
      for (long long l = cj[j].Child[0]; l < cj[j].Child[1]; l++)
        getList(NoF, rels, ci, k, cj, l, theta);
}

CSR::CSR(char NoF, const std::vector<Cell>& ci, const std::vector<Cell>& cj, double theta) {
  long long ncells = ci.size();
  std::vector<std::pair<long long, long long>> LIL;
  getList(NoF, LIL, &ci[0], 0, &cj[0], 0, theta);
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
