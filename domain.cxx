
#include "domain.hxx"

#include <cmath>
#include <algorithm>
#include <cstdio>

using namespace nbd;

void nbd::Z_index(int64_t i, int64_t dim, int64_t X[]) {
  std::fill(X, X + dim, 0);
  int64_t iter = 0;
  while(i > 0) {
    for (int64_t d = 0; d < dim; d++) {
      int64_t bit = i & 1;
      i = i >> 1;
      X[d] = X[d] | (bit << iter);
    }
    iter = iter + 1;
  }
}


void nbd::Z_index_i(const int64_t X[], int64_t dim, int64_t& i) {
  std::vector<int64_t> Xi(dim);
  std::copy(X, X + dim, Xi.data());
  std::vector<int64_t>::iterator it = std::find_if(Xi.begin(), Xi.end(), [](int64_t& x){ return x < 0; });
  i = -1;
  if (it == Xi.end()) { 
    i = 0;
    bool run = true;
    int64_t iter = 0;
    while (run) {
      run = false;
      for (int64_t d = 0; d < dim; d++) {
        int64_t bit = Xi[d] & 1;
        Xi[d] = Xi[d] >> 1;
        i = i | (bit << (iter * dim + d));
        run = run || (Xi[d] > 0);
      }
      iter = iter + 1;
    }
  }
}


int64_t nbd::Z_neighbors(int64_t N[], int64_t i, int64_t dim, int64_t max_i, int64_t theta) {
  std::vector<int64_t> Xi(dim);
  std::vector<int64_t> Xj(dim);
  Z_index(i, dim, Xi.data());

  int64_t len = (int64_t)std::floor(std::sqrt(theta));
  int64_t width = len * 2 + 1;
  int64_t ncom = (int64_t)std::pow(width, dim);
  std::vector<int64_t> ks(std::min(ncom, max_i));
  int64_t nk = 0;

  for (int64_t j = 0; j < ncom; j++) {
    Xj[0] = j;
    for (int64_t d = 1; d < dim; d++) {
      Xj[d] = Xj[d - 1] / width;
      Xj[d - 1] = Xj[d - 1] - Xj[d] * width;
    }
    int64_t r2 = 0;
    for (int64_t d = 0; d < dim; d++) {
      Xj[d] = Xj[d] - len;
      r2 = r2 + Xj[d] * Xj[d];
    }
    
    if (r2 <= theta) {
      for (int64_t d = 0; d < dim; d++)
        Xj[d] = Xj[d] + Xi[d];
      int64_t k;
      Z_index_i(Xj.data(), dim, k);
      if (k >= 0 && k < max_i)
        ks[nk++] = k;
    }
  }

  std::sort(ks.begin(), ks.begin() + nk);
  std::copy(ks.begin(), ks.begin() + nk, N);
  return nk;
}

void nbd::slices_level(int64_t slices[], int64_t lbegin, int64_t lend, int64_t dim) {
  std::fill(slices, slices + dim, 1);
  int64_t sdim = lbegin % dim;
  for (int64_t i = lbegin; i < lend; i++) {
    slices[sdim] <<= 1;
    sdim = (sdim == dim - 1) ? 0 : sdim + 1;
  }
}

void nbd::Global_Partition(GlobalDomain& goDomain, int64_t Nbodies, int64_t Ncrit, int64_t dim, double min, double max) {
  goDomain.DIM = dim;
  goDomain.NBODY = Nbodies;

  goDomain.LEVELS = (int64_t)std::floor(std::log2(Nbodies / Ncrit));
  goDomain.Xmin.resize(dim, min);
  goDomain.Xmax.resize(dim, max);
}


void nbd::Interactions(CSC& rels, int64_t y, int64_t xbegin, int64_t xend, int64_t dim, int64_t theta) {
  rels.M = y;
  rels.N = xend - xbegin;
  rels.CSC_COLS.resize(rels.N + 1);
  rels.CSC_ROWS.clear();
  rels.CSC_COLS[0] = 0;

  std::vector<int64_t> work(y);
  for (int64_t j = 0; j < rels.N; j++) {
    int64_t len = Z_neighbors(work.data(), j + xbegin, dim, y, theta);
    rels.CSC_ROWS.resize(rels.CSC_ROWS.size() + len);
    std::copy(work.begin(), work.begin() + len, rels.CSC_ROWS.begin() + rels.CSC_COLS[j]);
    rels.CSC_COLS[j + 1] = rels.CSC_ROWS.size();
  }
  rels.NNZ = rels.CSC_COLS[rels.N];
}


void nbd::Local_Partition(LocalDomain& loDomain, const GlobalDomain& goDomain, int64_t rank, int64_t size, int64_t theta) {
  loDomain.RANK = rank;
  loDomain.MY_LEVEL = (int64_t)std::floor(std::log2(size));
  loDomain.DIM = goDomain.DIM;

  loDomain.LOCAL_LEVELS = goDomain.LEVELS - loDomain.MY_LEVEL;
  int64_t boxes_local = (int64_t)1 << loDomain.LOCAL_LEVELS;

  loDomain.MY_IDS.resize(goDomain.LEVELS + 1);
  for (int64_t i = 0; i <= goDomain.LEVELS; i++) {
    GlobalIndex& gi = loDomain.MY_IDS[i];

    int64_t lvl_diff = 0;
    size = (int64_t)1 << loDomain.MY_LEVEL;
    if (i < loDomain.MY_LEVEL) {
      lvl_diff = loDomain.MY_LEVEL - i;
      size = (int64_t)1 << i;
    }

    int64_t my_rank = rank >> lvl_diff;
    std::vector<int64_t> work(size);
    int64_t len = Z_neighbors(work.data(), my_rank, loDomain.DIM, size, theta);
    gi.NGB_RNKS.resize(len);
    std::copy(work.begin(), work.begin() + len, gi.NGB_RNKS.begin());

    gi.SELF_I = std::distance(work.begin(), std::find(work.begin(), work.begin() + len, my_rank));
    int64_t ilocal = std::max((int64_t)0, i - loDomain.MY_LEVEL);
    gi.BOXES = (int64_t)1 << ilocal;
    
    int64_t mask = rank - my_rank << lvl_diff;
    gi.COMM_RNKS.resize(len);
    for (int64_t r = 0; r < len; r++)
      gi.COMM_RNKS[r] = (gi.NGB_RNKS[r] << lvl_diff) | mask;

    int64_t y = (int64_t)1 << i;
    int64_t xbegin = my_rank * gi.BOXES;
    int64_t xend = xbegin + gi.BOXES;
    Interactions(gi.RELS, y, xbegin, xend, goDomain.DIM, theta);
  }
}


void nbd::Local_bounds(double* Xmin, double* Xmax, const GlobalDomain& goDomain, const LocalDomain& loDomain) {
  std::vector<int64_t> Xi(goDomain.DIM);
  std::vector<int64_t> Slice(goDomain.DIM);
  Z_index(loDomain.RANK << loDomain.LOCAL_LEVELS, goDomain.DIM, Xi.data());
  slices_level(Slice.data(), loDomain.LOCAL_LEVELS, goDomain.LEVELS, goDomain.DIM);

  int64_t d = 0;
  for (int64_t i = 0; i < loDomain.LOCAL_LEVELS; i++) {
    Xi[d] >>= 1;
    d = (d == goDomain.DIM - 1) ? 0 : d + 1;
  }

  for (d = 0; d < goDomain.DIM; d++) {
    double bo_len = (goDomain.Xmax[d] - goDomain.Xmin[d]) / Slice[d];
    Xmin[d] = goDomain.Xmin[d] + Xi[d] * bo_len;
    Xmax[d] = Xmin[d] + bo_len;
  }
}

void nbd::lookupIJ(int64_t& ij, const CSC& rels, int64_t i, int64_t j) {
  if (j < 0 || j >= rels.N)
  { ij = -1; return; }
  int64_t k = std::distance(rels.CSC_ROWS.data(), 
    std::find(rels.CSC_ROWS.data() + rels.CSC_COLS[j], rels.CSC_ROWS.data() + rels.CSC_COLS[j + 1], i));
  if (k < rels.CSC_COLS[j + 1])
    ij = k;
}

void nbd::Lookup_GlobalI(int64_t& ilocal, const GlobalIndex& gi, int64_t iglobal) {
  int64_t nboxes = gi.BOXES;
  int64_t box_rank = iglobal / nboxes;
  int64_t i_rank = iglobal - box_rank * nboxes;

  int64_t box_ind = std::distance(gi.NGB_RNKS.begin(), std::find(gi.NGB_RNKS.begin(), gi.NGB_RNKS.end(), box_rank));
  if (box_ind < gi.NGB_RNKS.size())
    ilocal = box_ind * nboxes + i_rank;
  else
    ilocal = -1;
}
