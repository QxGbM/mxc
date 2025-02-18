
#include <h-matrix.hpp>
#include <build_tree.hpp>
#include <kernel.hpp>

#include <Eigen/Dense>
#include <Eigen/SVD>

LowRankMatrix::LowRankMatrix(double epi, long long m, long long n, long long k, long long p, long long niters, const Accessor& eval, long long iA, long long jA) : M(m), N(n), rank(k), U(m * rank), V(N * rank), S(rank) {
  Zrsvd(epi, m, n, &k, p, niters, eval, iA, jA, S.data(), U.data(), m, V.data(), n);

  if (k < rank) {
    rank = k;
    U.resize(m * rank);
    V.resize(n * rank);
    S.resize(rank);
  }
  else if (0. < epi && k == rank && S[0] * epi < S[rank - 1]) {
    rank = n;
    U.resize(m * n);
    V.clear();
    S.clear();

    Eigen::MatrixXcd id = Eigen::MatrixXcd::Identity(n, n);
    eval.op_Aij_mulB('N', m, n, n, iA, jA, id.data(), n, U.data(), m);
  }
}

void LowRankMatrix::projectBasis(char opU, char opV, long long m, long long n, const std::complex<double>* Up, long long ldu, const std::complex<double>* Vp, long long ldv, std::complex<double>* Sp, long long lds) const {
  Eigen::Stride<Eigen::Dynamic, 1> ldU(ldu, 1), ldV(ldv, 1), ldS(lds, 1);
  Eigen::Map<Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> Sout(Sp, m, n, ldS);

  Eigen::MatrixXcd X = Eigen::Map<const Eigen::MatrixXcd>(U.data(), M, rank) * Eigen::Map<const Eigen::VectorXd>(S.data(), rank).asDiagonal();
  Eigen::MatrixXcd Y(m, rank), Z(n, rank);
  Eigen::Map<const Eigen::MatrixXcd> matV(V.data(), N, rank);

  if (opU == 'T' || opU == 't')
    Y.noalias() = X.adjoint() * Eigen::Map<const Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>>(Up, m, M, ldU).transpose();
  else if (opU == 'C' || opU == 'c')
    Y.noalias() = X.adjoint() * Eigen::Map<const Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>>(Up, m, M, ldU).adjoint();
  else
    Y.noalias() = X.adjoint() * Eigen::Map<const Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>>(Up, M, m, ldU);

  if (opV == 'T' || opV == 't')
    Z.noalias() = Eigen::Map<const Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>>(Vp, N, n, ldU).transpose() * matV;
  else if (opV == 'C' || opV == 'c')
    Z.noalias() = Eigen::Map<const Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>>(Vp, N, n, ldU).adjoint() * matV;
  else
    Z.noalias() = Eigen::Map<const Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>>(Vp, n, N, ldU) * matV;

  Sout.noalias() = Y.adjoint() * Z.adjoint();
}

void LowRankMatrix::lowRankSumRow(double epi, long long m, long long n, long long* k, std::complex<double>* A, long long lda, long long lenL, const LowRankMatrix L[]) {
  Eigen::Stride<Eigen::Dynamic, 1> ldA(lda, 1);
  Eigen::Map<const Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> matA_in(A, m, n, ldA);
  Eigen::MatrixXcd U = matA_in;
  for (long long i = 0; i < lenL; ++i) {
    long long x = U.cols();
    long long r = L[i].rank;
    U.resize(m, x + r);
    U.rightCols(r) = Eigen::Map<const Eigen::MatrixXcd>(L[i].U.data(), m, r) * Eigen::Map<const Eigen::VectorXd>(L[i].S.data(), r).asDiagonal();
  }

  long long x = U.cols();
  Eigen::MatrixXcd W = Eigen::MatrixXcd::Zero(m, std::min(m, x));
  if (m < x) {
    Eigen::HouseholderQR<Eigen::MatrixXcd> qr(U.adjoint());
    W = qr.matrixQR().triangularView<Eigen::Upper>().adjoint();
  }
  else
    W.leftCols(x) = U;

  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(W, Eigen::ComputeThinU);  
  if (0. < epi && epi < 1.)
  { svd.setThreshold(epi); *k = std::min(*k, (long long)svd.rank()); }

  Eigen::Map<Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> matA_out(A, m, *k, ldA);
  matA_out = svd.matrixU().leftCols(*k) * svd.singularValues().topRows(*k).asDiagonal();
}

BlockLowRankMatrix1DRow::BlockLowRankMatrix1DRow(double epi, long long b, long long k, long long p, long long niters, const Accessor& eval, long long iA, long long mA) : dimM(mA), dimN(eval.N) {
  M = (long long)std::ceil(dimM / b);
  N = (long long)std::ceil(dimN / b);
  long long rM = dimM - (M - 1) * b, rN = dimN - (N - 1) * b;
  A.reserve(M * N);

  for (long long j = 0; j < N; ++j)
    for (long long i = 0; i < M; ++i)
      A.emplace_back(epi, i == (M - 1) ? rM : b, j == (N - 1) ? rN : b, k, p, niters, eval, iA + i * b, j * b);
}

void BlockLowRankMatrix1DRow::matVecMul(const std::complex<double> X[], std::complex<double> B[]) const {
  long long y = 0;
  for (long long i = 0; i < M; ++i) {
    long long mi = A[i].M;
    Eigen::Map<Eigen::VectorXcd> Bi(&B[y], mi);
    long long x = 0;
    for (long long j = 0; j < N; ++j) {
      long long nj = A[i + j * M].N;
      long long r = A[i + j * M].rank;
      Eigen::Map<const Eigen::VectorXcd> Xj(&X[x], nj);
      Eigen::Map<const Eigen::MatrixXcd> matU(A[i + j * M].U.data(), mi, r);
      if (r < nj) {
        Eigen::Map<const Eigen::MatrixXcd> matV(A[i + j * M].V.data(), nj, r);
        Eigen::Map<const Eigen::VectorXd> vecS(A[i + j * M].S.data(), r);
        Bi.noalias() += matU * (vecS.asDiagonal() * (matV.adjoint() * Xj));
      }
      else
        Bi.noalias() += matU * Xj;
      x += nj;
    }
    y += mi;
  }
}

void Hmatrix::construct(double epi, const Accessor& eval, long long rank, long long p, long long niters, long long lbegin, long long len, const Cell cells[], const CSR& Far, const double bodies[], const Hmatrix& upper) {
  Hmatrix::lbegin = lbegin;
  lend = lbegin + len;

  long long llen = Far.RowIndex[lend] - Far.RowIndex[lbegin];
  M.resize(len);
  N.resize(len);
  A.resize(len);
  L.reserve(llen);
  m.resize(len);

  for (long long i = upper.lbegin; i < upper.lend; i++)
    for (long long c = cells[i].Child[0]; c < cells[i].Child[1]; c++)
      if (lbegin <= c && c < lend) {
        long long y = cells[c].Body[1] - cells[c].Body[0];
        long long x = upper.N[i - upper.lbegin];
        long long off = cells[c].Body[0] - cells[i].Body[0];
        Eigen::Stride<Eigen::Dynamic, 1> lda(upper.M[i - upper.lbegin], 1);
        Eigen::Map<const Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> matA(&(upper.A[i - upper.lbegin])[off], y, x, lda);

        N[c - lbegin] = x;
        A[c - lbegin].reserve(y * rank);
        A[c - lbegin].resize(y * x);
        Eigen::Map<Eigen::MatrixXcd>(A[c - lbegin].data(), y, x) = matA;

        m[c - lbegin].insert(m[c - lbegin].end(), upper.m[i - upper.lbegin].begin(), upper.m[i - upper.lbegin].end());
      }
  
  for (long long y = lbegin; y < lend; y++) {
    M[y - lbegin] = cells[y].Body[1] - cells[y].Body[0];

    for (long long yx = Far.RowIndex[y]; yx < Far.RowIndex[y + 1]; yx++) {
      long long x = Far.ColIndex[yx];
      L.emplace_back(epi, M[y - lbegin], cells[x].Body[1] - cells[x].Body[0], rank, p, niters, eval, cells[y].Body[0], cells[x].Body[0]);
    }

    long long r = 0;
    long long llis = Far.RowIndex[y + 1] - Far.RowIndex[y];
    if (0 < llis) {
      r = std::min(M[y - lbegin], rank);
      if (N[y - lbegin] < r)
        A[y - lbegin].resize(M[y - lbegin] * r);
      const LowRankMatrix* lis = &L[Far.RowIndex[y] - Far.RowIndex[lbegin]];
      LowRankMatrix::lowRankSumRow(epi, M[y - lbegin], N[y - lbegin], &r, A[y - lbegin].data(), M[y - lbegin], llis, lis);
    }

    if (r < rank)
      A[y - lbegin].resize(M[y - lbegin] * r);
    N[y - lbegin] = r;

    for (long long yx = Far.RowIndex[y]; yx < Far.RowIndex[y + 1]; yx++) {
      long long x = Far.ColIndex[yx];
      long long n = cells[x].Body[1] - cells[x].Body[0];
      const double* xbodies = &bodies[3 * cells[x].Body[0]];
      m[y - lbegin].insert(m[y - lbegin].end(), xbodies, &xbodies[3 * n]);
    }
  }
}

long long Hmatrix::fbodies_size_at_i(long long i) const {
  return 0 <= i && i < (long long)m.size() ? m[i].size() / 3 : 0;
}
  
const double* Hmatrix::fbodies_at_i(long long i) const {
  return 0 <= i && i < (long long)m.size() ? m[i].data() : nullptr;
}

