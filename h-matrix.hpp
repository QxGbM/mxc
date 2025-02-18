
#pragma once

#include <vector>
#include <complex>

class Accessor;
class CSR;
class Cell;

class LowRankMatrix {
public:
  long long M, N, rank;
  std::vector<std::complex<double>> U, V;
  std::vector<double> S;

  LowRankMatrix(double epi, long long m, long long n, long long k, long long p, long long niters, const Accessor& eval, long long iA, long long jA);
  void projectBasis(char opU, char opV, long long m, long long n, const std::complex<double>* Up, long long ldu, const std::complex<double>* Vp, long long ldv, std::complex<double>* Sp, long long lds) const;
  static void lowRankSumRow(double epi, long long m, long long n, long long* k, std::complex<double>* A, long long lda, long long lenL, const LowRankMatrix L[]);
};

class BlockLowRankMatrix1DRow {
private:
  long long dimM, dimN;
  long long M, N;
  std::vector<LowRankMatrix> A;

public:
  BlockLowRankMatrix1DRow(double epi, long long b, long long k, long long p, long long niters, const Accessor& eval, long long iA, long long mA);
  void matVecMul(const std::complex<double> X[], std::complex<double> B[]) const;
};

class Hmatrix {
private:
  long long lbegin = 0;
  long long lend = 0;
  std::vector<long long> M, N;
  std::vector<std::vector<std::complex<double>>> A;
  std::vector<LowRankMatrix> L;

  std::vector<std::vector<double>> m;

public:
  void construct(double epi, const Accessor& eval, long long rank, long long p, long long niters, long long lbegin, long long len, const Cell cells[], const CSR& Far, const double bodies[], const Hmatrix& upper);
  long long fbodies_size_at_i(long long i) const;
  const double* fbodies_at_i(long long i) const;
};

