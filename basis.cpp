#include <basis.hpp>
#include <build_tree.hpp>
#include <comm.hpp>
#include <kernel.hpp>

#include <mkl.h>
#include <algorithm>
#include <numeric>
#include <cmath>

WellSeparatedApproximation::WellSeparatedApproximation(const MatrixAccessor& eval, double epi, int64_t rank, int64_t lbegin, int64_t len, const Cell cells[], const CSR& Far, const double bodies[], const WellSeparatedApproximation& upper) :
  lbegin(lbegin), lend(lbegin + len), M(len) {
  std::vector<std::vector<double>> Fbodies(len);
  for (int64_t i = upper.lbegin; i < upper.lend; i++)
    for (int64_t c = cells[i].Child[0]; c < cells[i].Child[1]; c++)
      if (lbegin <= c && c < lend)
        M[c - lbegin] = std::vector<double>(upper.M[i - upper.lbegin].begin(), upper.M[i - upper.lbegin].end());

  for (int64_t y = lbegin; y < lend; y++) {
    for (int64_t yx = Far.RowIndex[y]; yx < Far.RowIndex[y + 1]; yx++) {
      int64_t x = Far.ColIndex[yx];
      int64_t m = cells[y].Body[1] - cells[y].Body[0];
      int64_t n = cells[x].Body[1] - cells[x].Body[0];
      const double* Xbodies = &bodies[3 * cells[x].Body[0]];
      const double* Ybodies = &bodies[3 * cells[y].Body[0]];

      int64_t k = std::min(rank, std::min(m, n));
      std::vector<int64_t> ipiv(k);
      std::vector<std::complex<double>> U(n * k);
      int64_t iters = interpolative_decomp_aca(epi, eval, n, m, k, Xbodies, Ybodies, &ipiv[0], &U[0], n);
      std::vector<double> Fbodies(3 * iters);
      for (int64_t i = 0; i < iters; i++)
        std::copy(&Xbodies[3 * ipiv[i]], &Xbodies[3 * (ipiv[i] + 1)], &Fbodies[3 * i]);
      M[y - lbegin].insert(M[y - lbegin].end(), Fbodies.begin(), Fbodies.end());
    }
  }
}

int64_t WellSeparatedApproximation::fbodies_size_at_i(int64_t i) const {
  return 0 <= i && i < (int64_t)M.size() ? M[i].size() / 3 : 0;
}

const double* WellSeparatedApproximation::fbodies_at_i(int64_t i) const {
  return 0 <= i && i < (int64_t)M.size() ? M[i].data() : nullptr;
}

int64_t compute_basis(const MatrixAccessor& eval, double epi, int64_t M, int64_t N, double Xbodies[], const double Fbodies[], std::complex<double> A[], int64_t LDA, std::complex<double> R[], int64_t LDR) {
  int64_t K = std::max(M, N);
  std::complex<double> one(1., 0.), zero(0., 0.);
  std::vector<std::complex<double>> B(M * K), TAU(M);
  std::vector<double> S(M * 3);
  std::vector<MKL_INT> jpiv(M, 0);

  gen_matrix(eval, N, M, Fbodies, Xbodies, &B[0], K);
  LAPACKE_zgeqrf(LAPACK_COL_MAJOR, N, M, &B[0], K, &TAU[0]);
  LAPACKE_zlaset(LAPACK_COL_MAJOR, 'L', M - 1, M - 1, zero, zero, &B[1], K);
  LAPACKE_zgeqp3(LAPACK_COL_MAJOR, M, M, &B[0], K, &jpiv[0], &TAU[0]);
  int64_t rank = 0;
  double s0 = epi * std::abs(B[0]);
  if (std::numeric_limits<double>::min() < s0)
    while (rank < M && s0 <= std::abs(B[rank * (K + 1)]))
      ++rank;
  
  if (rank > 0) {
    if (rank < M)
      cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, rank, M - rank, &one, &B[0], K, &B[rank * K], K);
    LAPACKE_zlaset(LAPACK_COL_MAJOR, 'F', rank, rank, zero, one, &B[0], K);

    for (int64_t i = 0; i < M; i++) {
      int64_t piv = (int64_t)jpiv[i] - 1;
      std::copy(&B[i * K], &B[i * K + rank], &R[piv * LDR]);
      std::copy(&Xbodies[piv * 3], &Xbodies[piv * 3 + 3], &S[i * 3]);
    }
    std::copy(&S[0], &S[M * 3], Xbodies);

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, M, rank, M, &one, A, LDA, R, LDR, &zero, &B[0], M);
    LAPACKE_zgeqrf(LAPACK_COL_MAJOR, M, rank, &B[0], M, &TAU[0]);
    LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'L', M, rank, &B[0], M, A, LDA);
    LAPACKE_zungqr(LAPACK_COL_MAJOR, M, M, rank, A, LDA, &TAU[0]);
    LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'U', rank, rank, &B[0], M, R, LDR);
    LAPACKE_zlaset(LAPACK_COL_MAJOR, 'L', rank - 1, rank - 1, zero, zero, &R[1], LDR);
  }
  return rank;
}

ClusterBasis::ClusterBasis(const MatrixAccessor& eval, double epi, const Cell cells[], const CSR& Far, const double bodies[], const WellSeparatedApproximation& wsa, const CellComm& comm, const ClusterBasis& prev_basis, const CellComm& prev_comm) {
  int64_t xlen = comm.lenNeighbors();
  int64_t ibegin = comm.oLocal();
  int64_t nodes = comm.lenLocal();
  int64_t ybegin = comm.oGlobal();

  localChildOffsets = std::vector<int64_t>(nodes + 1);
  localChildLrDims = std::vector<int64_t>(cells[ybegin + nodes - 1].Child[1] - cells[ybegin].Child[0]);
  localChildIndex = prev_comm.iLocal(cells[ybegin].Child[0]);
  std::transform(&cells[ybegin], &cells[ybegin + nodes], localChildOffsets.begin() + 1, [&](const Cell& c) { return c.Child[1] - cells[ybegin].Child[0]; });
  std::copy(&prev_basis.DimsLr[localChildIndex], &prev_basis.DimsLr[localChildIndex + localChildLrDims.size()], localChildLrDims.begin());
  localChildOffsets[0] = 0;

  Dims = std::vector<int64_t>(xlen, 0);
  DimsLr = std::vector<int64_t>(xlen, 0);
  elementsOnRow = std::vector<int64_t>(xlen);
  S = std::vector<const double*>(xlen);
  Q = std::vector<const std::complex<double>*>(xlen);
  R = std::vector<std::complex<double>*>(xlen);

  for (int64_t i = 0; i < nodes; i++)
    Dims[i + ibegin] = localChildOffsets[i] == localChildOffsets[i + 1] ? (cells[i + ybegin].Body[1] - cells[i + ybegin].Body[0]) :
      std::reduce(&localChildLrDims[localChildOffsets[i]], &localChildLrDims[localChildOffsets[i + 1]]);

  const std::vector<int64_t> ones(xlen, 1);
  comm.neighbor_bcast(Dims.data(), ones.data());
  comm.dup_bcast(Dims.data(), xlen);

  std::vector<int64_t> Qoffsets(xlen + 1);
  std::transform(Dims.begin(), Dims.end(), elementsOnRow.begin(), [](const int64_t d) { return d * d; });
  std::inclusive_scan(elementsOnRow.begin(), elementsOnRow.end(), Qoffsets.begin() + 1);
  Qoffsets[0] = 0;
  Qdata = std::vector<std::complex<double>>(Qoffsets[xlen], std::complex<double>(0., 0.));
  Rdata = std::vector<std::complex<double>>(Qoffsets[xlen], std::complex<double>(0., 0.));
  std::transform(Qoffsets.begin(), Qoffsets.end(), Q.begin(), [&](const int64_t d) { return &Qdata[d]; });
  std::transform(Qoffsets.begin(), Qoffsets.end(), R.begin(), [&](const int64_t d) { return &Rdata[d]; });

  std::vector<int64_t> Ssizes(xlen), Soffsets(xlen + 1);
  std::transform(Dims.begin(), Dims.end(), Ssizes.begin(), [](const int64_t d) { return 3 * d; });
  std::inclusive_scan(Ssizes.begin(), Ssizes.end(), Soffsets.begin() + 1);
  Soffsets[0] = 0;
  Sdata = std::vector<double>(Soffsets[xlen], 0.);
  std::transform(Soffsets.begin(), Soffsets.end(), S.begin(), [&](const int64_t d) { return &Sdata[d]; });

  for (int64_t i = 0; i < nodes; i++) {
    int64_t dim = Dims[i + ibegin];
    std::complex<double>* matrix = &Qdata[Qoffsets[i + ibegin]];
    double* ske = &Sdata[Soffsets[i + ibegin]];

    int64_t ci = i + ybegin;
    int64_t childi = localChildIndex + localChildOffsets[i];
    int64_t cend = localChildIndex + localChildOffsets[i + 1];

    if (cend <= childi) {
      std::copy(&bodies[3 * cells[ci].Body[0]], &bodies[3 * cells[ci].Body[1]], ske);
      for (int64_t j = 0; j < dim; j++)
        matrix[j * (dim + 1)] = std::complex<double>(1., 0.);
    }
    for (int64_t j = childi; j < cend; j++) {
      int64_t offset = std::reduce(&prev_basis.DimsLr[childi], &prev_basis.DimsLr[j]);
      int64_t len = prev_basis.DimsLr[j];
      if (0 < len) {
        std::copy(prev_basis.S[j], prev_basis.S[j] + (len * 3), &ske[offset * 3]);
        MKL_Zomatcopy('C', 'N', len, len, std::complex<double>(1., 0.), prev_basis.R[j], prev_basis.Dims[j], &matrix[offset * (dim + 1)], dim);
      }
    }

    int64_t fsize = wsa.fbodies_size_at_i(i);
    const double* fbodies = wsa.fbodies_at_i(i);
    int64_t rank = (dim > 0 && fsize > 0) ? compute_basis(eval, epi, dim, fsize, ske, fbodies, matrix, dim, &Rdata[Qoffsets[i + ibegin]], dim) : 0;
    DimsLr[i + ibegin] = rank;
  }

  comm.neighbor_bcast(DimsLr.data(), ones.data());
  comm.neighbor_bcast(Sdata.data(), Ssizes.data());
  comm.neighbor_bcast(Qdata.data(), elementsOnRow.data());
  comm.neighbor_bcast(Rdata.data(), elementsOnRow.data());
  comm.dup_bcast(DimsLr.data(), xlen);
  comm.dup_bcast(Sdata.data(), Soffsets[xlen]);
  comm.dup_bcast(Qdata.data(), Qoffsets[xlen]);
  comm.dup_bcast(Rdata.data(), Qoffsets[xlen]);

  CRows = std::vector<int64_t>(&Far.RowIndex[ybegin], &Far.RowIndex[ybegin + nodes + 1]);
  CCols = std::vector<int64_t>(&Far.ColIndex[CRows[0]], &Far.ColIndex[CRows[nodes]]);
  int64_t offset = CRows[0];
  std::for_each(CRows.begin(), CRows.end(), [=](int64_t& i) { i = i - offset; });
  std::for_each(CCols.begin(), CCols.end(), [=](int64_t& i) { i = comm.iLocal(i); });

  CM = std::vector<int64_t>(CRows[nodes]);
  CN = std::vector<int64_t>(CRows[nodes]);
  for (int64_t i = 0; i < nodes; i++)
    std::fill(&CM[CRows[i]], &CM[CRows[i + 1]], DimsLr[i + ibegin]);
  std::transform(CCols.begin(), CCols.end(), CN.begin(), [&](int64_t col) { return DimsLr[col]; });

  std::vector<int64_t> Csizes(CRows[nodes]), Coffsets(CRows[nodes] + 1);
  std::transform(CM.begin(), CM.end(), CN.begin(), Csizes.begin(), [](int64_t m, int64_t n) { return m * n; });
  std::inclusive_scan(Csizes.begin(), Csizes.end(), Coffsets.begin() + 1);
  Coffsets[0] = 0;
  C = std::vector<const std::complex<double>*>(CRows[nodes]);
  Cdata = std::vector<std::complex<double>>(Coffsets.back());
  std::transform(Coffsets.begin(), Coffsets.begin() + CRows[nodes], C.begin(), [&](const int64_t d) { return &Cdata[d]; });

  for (int64_t i = 0; i < nodes; i++)
    for (int64_t ij = CRows[i]; ij < CRows[i + 1]; ij++) {
      int64_t j = CCols[ij], m = CM[ij], n = CN[ij];
      std::complex<double> one(1., 0.);
      gen_matrix(eval, m, n, S[i + ibegin], S[j], &Cdata[Coffsets[ij]], m);
      cblas_ztrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, m, n, &one, R[i + ibegin], Dims[i + ibegin], &Cdata[Coffsets[ij]], m);
      cblas_ztrmm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, CblasNonUnit, m, n, &one, R[j], Dims[j], &Cdata[Coffsets[ij]], m);
    }
}

int64_t compute_recompression(double epi, int64_t M, int64_t N, std::complex<double> Q[], int64_t LDQ, const std::complex<double> R[], int64_t LDR) {
  if (0 < N && N < M) {
    int64_t C = M - N;
    std::vector<std::complex<double>> A(M * M), TAU(C);
    std::complex<double> one(1., 0.), zero(0., 0.);
    std::vector<MKL_INT> jpiv(M, 0);

    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasTrans, C, M, M, &one, &Q[M * N], LDQ, R, LDR, &zero, &A[0], C);
    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasTrans, N, M, M, &one, Q, LDQ, R, LDR, &zero, &A[M * C], N);
    double nrmC = cblas_dznrm2(M * C, &A[0], 1);
    double nrmQ = cblas_dznrm2(M * N, &A[M * C], 1);

    if (epi * nrmQ <= nrmC) {
      int64_t rank = 0;
      LAPACKE_zgeqp3(LAPACK_COL_MAJOR, C, M, &A[0], C, &jpiv[0], &TAU[0]);
      double s0 = epi * std::abs(A[0]) * std::max(nrmQ / nrmC, (double)1.);
      if (std::numeric_limits<double>::min() < s0)
        while (rank < C && s0 <= std::abs(A[rank * (C + 1)]))
          ++rank;
      if (rank > 0)
        LAPACKE_zunmqr(LAPACK_COL_MAJOR, 'R', 'N', M, C, rank, &A[0], C, &TAU[0], &Q[M * N], LDQ);
      return N + rank;
    }
  }
  return N;
}

void ClusterBasis::recompressR(double epi, const CellComm& comm) {
  int64_t xlen = comm.lenNeighbors();
  int64_t ibegin = comm.oLocal();
  int64_t nodes = comm.lenLocal();

  for (int64_t i = 0; i < nodes; i++) {
    int64_t M = Dims[i + ibegin];
    std::complex<double>* Qptr = const_cast<std::complex<double>*>(Q[i + ibegin]);
    int64_t rank = compute_recompression(epi, M, DimsLr[i + ibegin], Qptr, M, R[i + ibegin], M);
    DimsLr[i + ibegin] = rank;
  }

  const std::vector<int64_t> ones(xlen, 1);
  comm.neighbor_bcast(DimsLr.data(), ones.data());
  comm.neighbor_bcast(Qdata.data(), elementsOnRow.data());
  comm.dup_bcast(DimsLr.data(), xlen);
  comm.dup_bcast(Qdata.data(), std::reduce(elementsOnRow.begin(), elementsOnRow.end()));
}

void compute_rowbasis_null_space(int64_t M, int64_t N, std::complex<double> A[], int64_t LDA) {
  if (0 < N && N < M) {
    std::complex<double> one(1., 0.);
    std::vector<std::complex<double>> B(N * N), TAU(N);

    LAPACKE_zgeqrf(LAPACK_COL_MAJOR, M, N, A, LDA, &TAU[0]);
    LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'U', N, N, A, LDA, &B[0], N);
    LAPACKE_zungqr(LAPACK_COL_MAJOR, M, M, N, A, LDA, &TAU[0]);
    cblas_ztrmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, M, N, &one, &B[0], N, A, LDA);
  }
}

void ClusterBasis::adjustLowerRankGrowth(const ClusterBasis& prev_basis, const CellComm& comm) {
  int64_t xlen = comm.lenNeighbors();
  int64_t ibegin = comm.oLocal();
  int64_t nodes = comm.lenLocal();

  std::vector<int64_t> oldDims(nodes);
  std::vector<std::vector<std::complex<double>>> oldQ(nodes);
  std::vector<int64_t> newLocalChildLrDims(localChildLrDims.size());
  std::copy(&prev_basis.DimsLr[localChildIndex], &prev_basis.DimsLr[localChildIndex + localChildLrDims.size()], newLocalChildLrDims.begin());

  for (int64_t i = 0; i < nodes; i++) {
    oldDims[i] = Dims[i + ibegin];
    oldQ[i] = std::vector<std::complex<double>>(oldDims[i] * DimsLr[i + ibegin]);
    std::copy(Q[i + ibegin], Q[i + ibegin] + oldQ[i].size(), oldQ[i].begin());
    Dims[i + ibegin] = std::reduce(&newLocalChildLrDims[localChildOffsets[i]], &newLocalChildLrDims[localChildOffsets[i + 1]]);
  }

  const std::vector<int64_t> ones(xlen, 1);
  comm.neighbor_bcast(Dims.data(), ones.data());
  comm.dup_bcast(Dims.data(), xlen);

  std::vector<int64_t> Qoffsets(xlen + 1);
  std::transform(Dims.begin(), Dims.end(), elementsOnRow.begin(), [](const int64_t d) { return d * d; });
  std::inclusive_scan(elementsOnRow.begin(), elementsOnRow.end(), Qoffsets.begin() + 1);
  Qoffsets[0] = 0;
  Qdata = std::vector<std::complex<double>>(Qoffsets[xlen], std::complex<double>(0., 0.));
  Rdata = std::vector<std::complex<double>>(Qoffsets[xlen], std::complex<double>(0., 0.));
  std::transform(Qoffsets.begin(), Qoffsets.end(), Q.begin(), [&](const int64_t d) { return &Qdata[d]; });
  std::transform(Qoffsets.begin(), Qoffsets.end(), R.begin(), [&](const int64_t d) { return &Rdata[d]; });

  for (int64_t i = 0; i < nodes; i++) {
    int64_t M = Dims[i + ibegin];
    int64_t N = DimsLr[i + ibegin];
    int64_t childi = localChildOffsets[i];
    int64_t cend = localChildOffsets[i + 1];
    for (int64_t j = childi; j < cend && 0 < N; j++) {
      int64_t offsetOld = std::reduce(&localChildLrDims[childi], &localChildLrDims[j]);
      int64_t offsetNew = std::reduce(&newLocalChildLrDims[childi], &newLocalChildLrDims[j]);
      int64_t len = localChildLrDims[j];
      MKL_Zomatcopy('C', 'N', len, N, std::complex<double>(1., 0.), &oldQ[i][offsetOld], oldDims[i], &Qdata[Qoffsets[i + ibegin] + offsetNew], M);
    }
    compute_rowbasis_null_space(M, N, &Qdata[Qoffsets[i + ibegin]], M);
  }

  std::copy(newLocalChildLrDims.begin(), newLocalChildLrDims.end(), localChildLrDims.begin());
  comm.neighbor_bcast(Qdata.data(), elementsOnRow.data());
  comm.dup_bcast(Qdata.data(), Qoffsets[xlen]);
}

MatVec::MatVec(const MatrixAccessor& eval, const ClusterBasis basis[], const double bodies[], const Cell cells[], const CSR& near, const CellComm comm[], int64_t levels) :
  EvalFunc(&eval), Basis(basis), Bodies(bodies), Cells(cells), Near(&near), Comm(comm), Levels(levels) {
}

void MatVec::operator() (int64_t nrhs, std::complex<double> X[], int64_t ldX) const {
  int64_t lbegin = Comm[Levels].oLocal();
  int64_t llen = Comm[Levels].lenLocal();

  std::vector<std::vector<std::complex<double>>> rhsX(Levels + 1), rhsY(Levels + 1);
  std::vector<std::vector<std::complex<double>*>> rhsXptr(Levels + 1), rhsYptr(Levels + 1);
  std::vector<std::vector<std::pair<std::complex<double>*, int64_t>>> rhsXoptr(Levels + 1), rhsYoptr(Levels + 1);

  for (int64_t l = Levels; l >= 0; l--) {
    int64_t xlen = Comm[l].lenNeighbors();
    std::vector<int64_t> offsets(xlen + 1, 0);
    std::inclusive_scan(Basis[l].Dims.begin(), Basis[l].Dims.end(), offsets.begin() + 1);

    rhsX[l] = std::vector<std::complex<double>>(offsets[xlen] * nrhs, std::complex<double>(0., 0.));
    rhsY[l] = std::vector<std::complex<double>>(offsets[xlen] * nrhs, std::complex<double>(0., 0.));
    rhsXptr[l] = std::vector<std::complex<double>*>(xlen, nullptr);
    rhsYptr[l] = std::vector<std::complex<double>*>(xlen, nullptr);
    rhsXoptr[l] = std::vector<std::pair<std::complex<double>*, int64_t>>(xlen, std::make_pair(nullptr, 0));
    rhsYoptr[l] = std::vector<std::pair<std::complex<double>*, int64_t>>(xlen, std::make_pair(nullptr, 0));

    std::transform(offsets.begin(), offsets.begin() + xlen, rhsXptr[l].begin(), [&](const int64_t d) { return &rhsX[l][0] + d * nrhs; });
    std::transform(offsets.begin(), offsets.begin() + xlen, rhsYptr[l].begin(), [&](const int64_t d) { return &rhsY[l][0] + d * nrhs; });

    if (l < Levels)
      for (int64_t i = 0; i < xlen; i++) {
        int64_t ci = Comm[l].iGlobal(i);
        int64_t child = Comm[l + 1].iLocal(Cells[ci].Child[0]);
        int64_t clen = Cells[ci].Child[1] - Cells[ci].Child[0];

        if (child >= 0 && clen > 0) {
          std::vector<int64_t> offsets_child(clen + 1, 0);
          std::inclusive_scan(&Basis[l + 1].DimsLr[child], &Basis[l + 1].DimsLr[child + clen], offsets_child.begin() + 1);
          int64_t ldi = Basis[l].Dims[i];
          std::transform(offsets_child.begin(), offsets_child.begin() + clen, &rhsXoptr[l + 1][child], 
            [&](const int64_t d) { return std::make_pair(rhsXptr[l][i] + d, ldi); });
          std::transform(offsets_child.begin(), offsets_child.begin() + clen, &rhsYoptr[l + 1][child], 
            [&](const int64_t d) { return std::make_pair(rhsYptr[l][i] + d, ldi); });
        }
      }
  }

  int64_t Y = 0;
  for (int64_t i = 0; i < llen; i++) {
    int64_t M = Basis[Levels].Dims[lbegin + i];
    MKL_Zomatcopy('C', 'N', M, nrhs, std::complex<double>(1., 0.), &X[Y], ldX, rhsXptr[Levels][lbegin + i], M);
    Y = Y + M;
  }

  const std::complex<double> one(1., 0.), zero(0., 0.);
  for (int64_t i = Levels; i > 0; i--) {
    int64_t ibegin = Comm[i].oLocal();
    int64_t iboxes = Comm[i].lenLocal();
    int64_t xlen = Comm[i].lenNeighbors();

    std::vector<int64_t> lens(xlen);
    std::transform(Basis[i].Dims.begin(), Basis[i].Dims.end(), lens.begin(), [=](const int64_t& i) { return i * nrhs; });
    int64_t lenI = nrhs * std::reduce(&Basis[i].Dims[0], &Basis[i].Dims[xlen]);
    Comm[i].level_merge(rhsX[i].data(), lenI);
    Comm[i].neighbor_bcast(rhsX[i].data(), lens.data());
    Comm[i].dup_bcast(rhsX[i].data(), lenI);

    for (int64_t y = 0; y < iboxes; y++) {
      int64_t M = Basis[i].Dims[y + ibegin];
      int64_t N = Basis[i].DimsLr[y + ibegin];
      if (M > 0 && N > 0)
        cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, N, nrhs, M, &one, Basis[i].Q[y + ibegin], M, 
          rhsXptr[i][y + ibegin], M, &zero, rhsXoptr[i][y + ibegin].first, rhsXoptr[i][y + ibegin].second);
    }
  }

  if (Basis[0].Dims[0] > 0) {
    Comm[0].level_merge(rhsX[0].data(), Basis[0].Dims[0] * nrhs);
    Comm[0].dup_bcast(rhsX[0].data(), Basis[0].Dims[0] * nrhs);
  }

  for (int64_t i = 1; i <= Levels; i++) {
    int64_t ibegin = Comm[i].oLocal();
    int64_t iboxes = Comm[i].lenLocal();

    for (int64_t y = 0; y < iboxes; y++) {
      for (int64_t yx = Basis[i].CRows[y]; yx < Basis[i].CRows[y + 1]; yx++) {
        int64_t x = Basis[i].CCols[yx];
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Basis[i].CM[yx], nrhs, Basis[i].CN[yx], &one, Basis[i].C[yx], Basis[i].CM[yx], 
          rhsXoptr[i][x].first, rhsXoptr[i][x].second, &one, rhsYoptr[i][y + ibegin].first, rhsYoptr[i][y + ibegin].second);
      }
      int64_t K = Basis[i].DimsLr[y + ibegin];
      int64_t M = Basis[i].Dims[y + ibegin];
      if (M > 0 && K > 0)
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, nrhs, K, &one, Basis[i].Q[y + ibegin], M, 
          rhsYoptr[i][y + ibegin].first, rhsYoptr[i][y + ibegin].second, &zero, rhsYptr[i][y + ibegin], M);
    }
  }

  int64_t gbegin = Comm[Levels].oGlobal();
  for (int64_t y = 0; y < llen; y++)
    for (int64_t yx = Near->RowIndex[y + gbegin]; yx < Near->RowIndex[y + gbegin + 1]; yx++) {
      int64_t x = Near->ColIndex[yx];
      int64_t x_loc = Comm[Levels].iLocal(x);
      int64_t M = Cells[y + gbegin].Body[1] - Cells[y + gbegin].Body[0];
      int64_t N = Cells[x].Body[1] - Cells[x].Body[0];
      mat_vec_reference(*EvalFunc, M, N, nrhs, rhsYptr[Levels][y + lbegin], M, rhsXptr[Levels][x_loc], N, &Bodies[3 * Cells[y + gbegin].Body[0]], &Bodies[3 * Cells[x].Body[0]]);
    }
  Y = 0;
  for (int64_t i = 0; i < llen; i++) {
    int64_t M = Basis[Levels].Dims[lbegin + i];
    MKL_Zomatcopy('C', 'N', M, nrhs, std::complex<double>(1., 0.), rhsYptr[Levels][lbegin + i], M, &X[Y], ldX);
    Y = Y + M;
  }
}
