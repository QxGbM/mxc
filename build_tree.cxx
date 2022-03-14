
#include "build_tree.hxx"
#include "dist.hxx"

#include <cmath>
#include <random>
#include <numeric>
#include <cstdio>

using namespace nbd;

void nbd::randomBodies(Bodies& bodies, int64_t nbody, const double dmin[], const double dmax[], int64_t dim, int seed) {
  if (seed > 0)
    srand(seed);

  std::vector<double> range(dim + 1);
  for (int64_t d = 0; d <= dim; d++)
    range[d] = dmax[d] - dmin[d];

  for (int64_t i = 0; i < nbody; i++) {
    for (int64_t d = 0; d < dim; d++) {
      double r = ((double)rand() / RAND_MAX) * range[d] + dmin[d];
      bodies[i].X[d] = r;
    }
    double r = ((double)rand() / RAND_MAX) * range[dim] + dmin[dim];
    bodies[i].B = r;
  }
}

int64_t nbd::getIndex(const int64_t iX[], int64_t dim) {
  std::vector<int64_t> jX(dim);
  for (int64_t d = 0; d < dim; d++) {
    jX[d] = iX[d];
    if (iX[d] < 0)
      return -1;
  }

  int run = 1;
  int64_t l = 0;
  int64_t index = 0;
  while (run) {
    run = 0;
    for (int64_t d = 0; d < dim; d++) {
      index += (jX[d] & 1) << (dim * l + d);
      if (jX[d] >>= 1)
        run = 1;
    }
    l++;
  }
  return index;
}

void nbd::getIX(int64_t iX[], int64_t index, int64_t dim) {
  if (index < 0)
    return;
  int64_t l = 0;
  for (int64_t d = 0; d < dim; d++)
    iX[d] = 0;
  while (index > 0) {
    for (int64_t d = 0; d < dim; d++) {
      iX[d] += (index & 1) << l;
      index >>= 1;
    }
    l++;
  }
}

void nbd::bucketSort(Bodies& bodies, int64_t buckets[], int64_t slices[], const double dmin[], const double dmax[], int64_t dim) {
  int64_t nbody = bodies.size();
  int64_t nboxes = 1;
  std::vector<double> box_dim(dim);
  std::vector<double> adj_dmin(dim);
  for (int64_t d = 0; d < dim; d++) {
    nboxes = nboxes * slices[d];
    double adj_dmax = dmax[d] + 1.e-6;
    adj_dmin[d] = dmin[d] - 1.e-6;
    box_dim[d] = (adj_dmax - adj_dmin[d]) / slices[d];
  }

  std::fill(buckets, buckets + nboxes, 0);
  std::vector<int64_t> bodies_i(nbody);
  std::vector<int64_t> offsets(nboxes);
  Bodies bodies_cpy(nbody);
  std::vector<int64_t> Xi(dim);

  for (int64_t i = 0; i < nbody; i++) {
    const Body& bi = bodies[i];
    for (int64_t d = 0; d < dim; d++)
      Xi[d] = (int64_t)((bi.X[d] - adj_dmin[d]) / box_dim[d]);
    int64_t ind = getIndex(Xi.data(), dim);
    bodies_i[i] = ind;
    buckets[ind] = buckets[ind] + 1;
  }

  offsets[0] = 0;
  for (int64_t i = 1; i < nboxes; i++)
    offsets[i] = offsets[i - 1] + buckets[i - 1];

  for (int64_t i = 0; i < nbody; i++) {
    int64_t bi = bodies_i[i];
    const Body& src = bodies[i];
    int64_t offset_bi = offsets[bi];
    Body& tar = bodies_cpy[offset_bi];
    for (int64_t d = 0; d < dim; d++)
      tar.X[d] = src.X[d];
    tar.B = src.B;
    offsets[bi] = offset_bi + 1;
  }

  for (int64_t i = 0; i < nbody; i++) {
    for (int64_t d = 0; d < dim; d++)
      bodies[i].X[d] = bodies_cpy[i].X[d];
    bodies[i].B = bodies_cpy[i].B;
  }
}

void iterNonLeaf(int64_t levels, Cells& cells, Cell* head, int64_t ncells, int64_t dim) {
  if (levels > 0) {
    levels = levels - 1;
    int64_t ci = -1;
    int64_t count = 0;

    for (int64_t i = 0; i < ncells; i++) {
      int64_t pi = (head[i].ZID) >> 1;
      if (pi > ci) {
        ci = pi;
        cells.emplace_back();
        Cell* tail = &cells.back();
        tail->BODY = head[i].BODY;
        tail->NBODY = head[i].NBODY;
        tail->CHILD = head + i;
        tail->NCHILD = 1;
        tail->ZID = pi;
        tail->LEVEL = levels;
        getIX(tail->ZX, pi, dim);
        count += 1;
      }
      else {
        Cell* tail = &cells.back();
        tail->NBODY = tail->NBODY + head[i].NBODY;
        tail->NCHILD = tail->NCHILD + 1;
      }
    }

    iterNonLeaf(levels, cells, head + ncells, count, dim);
  }
}

int64_t nbd::buildTree(Cells& cells, Bodies& bodies, int64_t ncrit, const double dmin[], const double dmax[], int64_t dim) {
  int64_t nbody = bodies.size();
  int64_t levels = (int64_t)(std::log2(nbody / ncrit));
  int64_t len = (int64_t)1 << levels;

  cells.reserve(len * 2);
  cells.resize(1);

  std::vector<int64_t> slices(dim);
  for (int64_t d = 0; d < dim; d++)
    slices[d] = 1;
  for (int64_t l = 0; l < levels; l++)
    slices[l % dim] <<= 1;

  std::vector<int64_t> buckets(len);
  bucketSort(bodies, buckets.data(), slices.data(), dmin, dmax, dim);

  int64_t bcount = 0;
  for (int64_t i = 0; i < len; i++)
    if (buckets[i] > 0) {
      cells.emplace_back();
      Cell* ci = &cells.back();
      ci->BODY = &bodies[bcount];
      ci->NBODY = buckets[i];
      ci->CHILD = NULL;
      ci->NCHILD = 0;
      ci->ZID = i;
      ci->LEVEL = levels;
      getIX(ci->ZX, i, dim);
      bcount += buckets[i];
    }

  iterNonLeaf(levels, cells, &cells[1], cells.size() - 1, dim);
  Cell* root = &cells.back();
  cells[0].BODY = root->BODY;
  cells[0].CHILD = root->CHILD;
  cells[0].NBODY = root->NBODY;
  cells[0].NCHILD = root->NCHILD;
  cells[0].ZID = root->ZID;
  cells[0].LEVEL = root->LEVEL;
  getIX(cells[0].ZX, root->ZID, dim);
  cells.pop_back();
  return levels;
}


void nbd::getList(Cell* Ci, Cell* Cj, int64_t dim, int64_t theta) {
  if (Ci->LEVEL < Cj->LEVEL)
    for (Cell* ci = Ci->CHILD; ci != Ci->CHILD + Ci->NCHILD; ci++)
      getList(ci, Cj, dim, theta);
  else if (Cj->LEVEL < Ci->LEVEL)
    for (Cell* cj = Cj->CHILD; cj != Cj->CHILD + Cj->NCHILD; cj++)
      getList(Ci, cj, dim, theta);
  else {
    int64_t dX = 0;
    for (int64_t d = 0; d < dim; d++) {
      int64_t diff = Ci->ZX[d] - Cj->ZX[d];
      dX = dX + diff * diff;
    }

    if (dX > theta * theta)
      Ci->listFar.push_back(Cj);
    else {
      Ci->listNear.push_back(Cj);

      if (Ci->NCHILD > 0)
        for (Cell* ci = Ci->CHILD; ci != Ci->CHILD + Ci->NCHILD; ci++)
          getList(ci, Cj, dim, theta);
    }
  }
}

void nbd::findCellsAtLevel(const Cell* cells[], int64_t* len, const Cell* cell, int64_t level) {
  if (level == cell->LEVEL) {
    int64_t i = *len;
    cells[i] = cell;
    *len = i + 1;
  }
  else if (level > cell->LEVEL && cell->NCHILD > 0)
    for (int64_t i = 0; i < cell->NCHILD; i++)
      findCellsAtLevel(cells, len, cell->CHILD + i, level);
}


void nbd::remoteBodies(Bodies& remote, int64_t size, const Cell& cell, const Bodies& bodies, int64_t dim) {
  int64_t avail = bodies.size();
  int64_t len = cell.listNear.size();
  std::vector<int64_t> offsets(len);
  std::vector<int64_t> lens(len);

  const Body* begin = &bodies[0];
  for (int64_t i = 0; i < len; i++) {
    const Cell* c = cell.listNear[i];
    avail = avail - c->NBODY;
    offsets[i] = c->BODY - begin;
    lens[i] = c->NBODY;
  }

  size = size > avail ? avail : size;
  remote.resize(size);

  for (int64_t i = 0; i < size; i++) {
    int64_t loc = (int64_t)((double)(avail * i) / size);
    for (int64_t j = 0; j < len; j++)
      if (loc >= offsets[j])
        loc = loc + lens[j];

    for (int64_t d = 0; d < dim; d++)
      remote[i].X[d] = bodies[loc].X[d];
    remote[i].B = bodies[loc].B;
  }
}

void nbd::traverse(Cells& cells, Cell* locals[], int64_t levels, int64_t dim, int64_t theta, int64_t mpi_rank, int64_t mpi_size) {
  getList(&cells[0], &cells[0], dim, theta);
  int64_t mpi_levels = mpi_size > 1 ? (int64_t)(std::log2(mpi_size)) : 0;

  Cell* iter = &cells[0];
  for (int64_t i = 0; i <= mpi_levels; i++) {
    int64_t lvl_diff = mpi_levels - i;
    int64_t my_rank = mpi_rank >> lvl_diff;
    if (iter->LEVEL < i)
      for (Cell* ci = iter->CHILD; ci != iter->CHILD + iter->NCHILD; ci++)
        if (ci->ZID == my_rank)
          iter = ci;
    if (iter->ZID == my_rank) {
      locals[i] = iter;
      int64_t nlen = iter->listNear.size();
      std::vector<int64_t> ngbs(nlen);
      for (int64_t n = 0; n < nlen; n++) {
        Cell* c = iter->listNear[n];
        ngbs[n] = c->ZID;
      }

      configureComm(mpi_rank, i, &ngbs[0], nlen);
    }
  }

  if (iter->ZID == mpi_rank)
    for (int64_t i = mpi_levels + 1; i <= levels; i++)
      locals[i] = iter;
}

void nbd::evaluateBasis(EvalFunc ef, Cells& cells, Cell* c, const Bodies& bodies, int64_t sp_pts, int64_t rank, int64_t dim) {
  if (c->NCHILD > 0)
    for (int64_t i = 0; i < c->NCHILD; i++)
      evaluateBasis(ef, cells, c->CHILD + i, bodies, sp_pts, rank, dim);

  Bodies remote;
  remoteBodies(remote, sp_pts, *c, bodies, dim);
  P2Mmat(ef, c, remote.data(), remote.size(), dim, c->Base, 1.e-12, rank);
  invBasis(c->Base, c->Biv);
}


void nbd::relationsNear(CSC rels[], const Cells& cells, int64_t mpi_rank, int64_t mpi_size) {
  int64_t levels = 0;
  int64_t len = cells.size();
  for (int64_t i = 0; i < len; i++)
    levels = levels > cells[i].LEVEL ? levels : cells[i].LEVEL;

  int64_t mpi_levels = mpi_size > 1 ? (int64_t)(std::log2(mpi_size)) : 0;
  for (int64_t i = 0; i <= levels; i++) {
    int64_t mpi_boxes = i > mpi_levels ? (int64_t)1 << (i - mpi_levels) : 1;
    int64_t mpi_dups = i < mpi_levels ? (mpi_levels - i) : 0;
    CSC& csc = rels[i];

    csc.M = (int64_t)1 << i;
    csc.N = mpi_boxes;
    csc.CSC_COLS.resize(mpi_boxes + 1);
    std::fill(csc.CSC_COLS.begin(), csc.CSC_COLS.end(), 0);
    csc.CSC_ROWS.clear();
    csc.NNZ = 0;
    csc.CBGN = (mpi_rank >> mpi_dups) * mpi_boxes;
  }

  for (int64_t i = 0; i < len; i++) {
    const Cell& c = cells[i];
    int64_t l = c.LEVEL;
    CSC& csc = rels[l];
    int64_t n = c.ZID - csc.CBGN;
    if (n >= 0 && n < csc.N) {
      int64_t ent = c.listNear.size();
      csc.CSC_COLS[n] = ent;
      for (int64_t j = 0; j < ent; j++)
        csc.CSC_ROWS.emplace_back((c.listNear[j])->ZID);
      csc.NNZ = csc.NNZ + ent;
    }
  }

  for (int64_t i = 0; i <= levels; i++) {
    CSC& csc = rels[i];
    int64_t count = 0;
    for (int64_t j = 0; j <= csc.N; j++) {
      int64_t ent = csc.CSC_COLS[j];
      csc.CSC_COLS[j] = count;
      count = count + ent;
    }
  }
}

void nbd::evaluateLeafNear(Matrices& d, EvalFunc ef, const Cell* cell, int64_t dim, const CSC& csc) {
  if (cell->NCHILD > 0)
    for (int64_t i = 0; i < cell->NCHILD; i++)
      evaluateLeafNear(d, ef, cell->CHILD + i, dim, csc);
  else {
    int64_t n = cell->ZID - csc.CBGN;
    if (n >= 0 && n < csc.N) {
      int64_t len = cell->listNear.size();
      int64_t off = csc.CSC_COLS[n];
      for (int64_t i = 0; i < len; i++)
        P2Pmat(ef, cell->listNear[i], cell, dim, d[off + i]);
    }
  }
}

void nbd::lookupIJ(int64_t& ij, const CSC& rels, int64_t i, int64_t j) {
  int64_t lj = j - rels.CBGN;
  if (lj < 0 || lj >= rels.N)
  { ij = -1; return; }
  int64_t k = std::distance(rels.CSC_ROWS.data(), 
    std::find(rels.CSC_ROWS.data() + rels.CSC_COLS[lj], rels.CSC_ROWS.data() + rels.CSC_COLS[lj + 1], i));
  ij = (k < rels.CSC_COLS[lj + 1]) ? k : -1;
}


void writeIntermediate(Matrix& d, EvalFunc ef, const Cell* ci, const Cell* cj, const Cells& cells, int64_t dim, const Matrices& d_child, const CSC& csc_child) {
  int64_t m = 0;
  int64_t n = 0;
  for (int64_t i = 0; i < ci->NCHILD; i++) {
    const Cell* cii = ci->CHILD + i;
    m = m + cii->Multipole.size();
  }
  for (int64_t j = 0; j < cj->NCHILD; j++) {
    const Cell* cjj = cj->CHILD + j;
    n = n + cjj->Multipole.size();
  }

  d.A.resize(m * n);
  d.M = m;
  d.N = n;

  int64_t y_off = 0;
  for (int64_t i = 0; i < ci->NCHILD; i++) {
    const Cell* cii = ci->CHILD + i;
    const std::vector<Cell*>& cii_m2l = cii->listFar;
    int64_t x_off = 0;
    for (int64_t j = 0; j < cj->NCHILD; j++) {
      const Cell* cjj = cj->CHILD + j;
      if (std::find(cii_m2l.begin(), cii_m2l.end(), cjj) != cii_m2l.end())
        L2C(ef, cii, cjj, dim, d, y_off, x_off);
      else {
        int64_t zii = cii->ZID;
        int64_t zjj = cjj->ZID;
        int64_t ij;
        lookupIJ(ij, csc_child, zii, zjj);
        if (ij >= 0)
          D2C(d_child[ij], cii->Biv, cjj->Biv, d, y_off, x_off);
      }
      x_off = x_off + cjj->Multipole.size();
    }
    y_off = y_off + cii->Multipole.size();
  }
}

void evaluateIntermediate(EvalFunc ef, const Cell* c, const Cells& cells, int64_t dim, const CSC csc[], Matrices d[]) {
  if (c->NCHILD > 0) {
    for (int64_t i = 0; i < c->NCHILD; i++)
      evaluateIntermediate(ef, c->CHILD + i, cells, dim, csc, d);

    int64_t zj = c->ZID;
    int64_t level = c->LEVEL;
    int64_t lnear = c->listNear.size();
    for (int64_t i = 0; i < lnear; i++) {
      const Cell* ci = c->listNear[i];
      int64_t zi = ci->ZID;
      int64_t ij;
      lookupIJ(ij, csc[level], zi, zj);
      if (ij >= 0)
        writeIntermediate(d[level][ij], ef, ci, c, cells, dim, d[level + 1], csc[level + 1]);
    }
  }
}

void nbd::evaluateNear(Matrices d[], EvalFunc ef, const Cells& cells, int64_t dim, const CSC rels[], int64_t levels) {
  Matrices& dleaf = d[levels];
  const CSC& cleaf = rels[levels];
  for (int64_t i = 0; i <= levels; i++)
    d[i].resize(rels[i].NNZ);
  evaluateLeafNear(d[levels], ef, &cells[0], dim, cleaf);
  evaluateIntermediate(ef, &cells[0], cells, dim, &rels[0], &d[0]);
}

void nbd::loadX(Vectors& X, const Cell* cell, int64_t level) {
  int64_t xlen = (int64_t)1 << level;
  neighborContentLength(xlen, level);
  X.resize(xlen);

  int64_t ibegin = 0;
  int64_t iend = (int64_t)1 << level;
  selfLocalRange(ibegin, iend, level);
  int64_t nodes = iend - ibegin;

  int64_t len = 0;
  std::vector<const Cell*> cells(nodes);
  findCellsAtLevel(&cells[0], &len, cell, level);

  for (int64_t i = 0; i < len; i++) {
    const Cell* ci = cells[i];
    int64_t li = ci->ZID;
    neighborsILocal(li, ci->ZID, level);
    Vector& Xi = X[li];
    cVector(Xi, ci->NBODY);

    for (int64_t n = 0; n < ci->NBODY; n++)
      Xi.X[n] = ci->BODY[n].B;
  }
  DistributeVectorsList(X, level);
}