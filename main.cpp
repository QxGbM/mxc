
#include <kernel.hpp>
#include <build_tree.hpp>
#include <h2matrix.hpp>
#include <comm-mpi.hpp>

#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void uniform_unit_cube(double* bodies, long long nbodies, double diameter, long long dim);
void uniform_unit_cube_rnd(double* bodies, long long nbodies, double diameter, long long dim, unsigned int seed);
void mesh_sphere(double* bodies, long long nbodies, double r);
void solveRelErr(double* err_out, const std::complex<double>* X, const std::complex<double>* ref, long long lenX);
void read_sorted_bodies(long long* nbodies, long long lbuckets, double* bodies, long long buckets[], const char* fname);

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  long long Nbody = argc > 1 ? std::atoll(argv[1]) : 2048;
  double theta = argc > 2 ? std::atof(argv[2]) : 1e0;
  long long leaf_size = argc > 3 ? std::atoll(argv[3]) : 256;
  double epi = argc > 4 ? std::atof(argv[4]) : 1e-10;
  long long rank = argc > 5 ? std::atoll(argv[5]) : 100;

  leaf_size = Nbody < leaf_size ? Nbody : leaf_size;
  long long levels = (long long)std::log2((double)Nbody / leaf_size);
  long long Nleaf = (long long)1 << levels;
  long long ncells = Nleaf + Nleaf - 1;

  int mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  
  //Laplace3D eval(1.);
  //Yukawa3D eval(1, 1.);
  //Gaussian eval(8);
  Helmholtz3D eval(1.e-1, 1.e-1);
  
  std::vector<double> body(Nbody * 3);
  std::vector<std::complex<double>> Xbody(Nbody);
  std::vector<Cell> cell(ncells);

  std::vector<ColCommMPI> communicator(levels + 1);
  std::vector<H2Matrix> A(levels + 1);

  //mesh_sphere(&body[0], Nbody, std::pow(Nbody, 1./2.));
  uniform_unit_cube_rnd(&body[0], Nbody, std::pow(Nbody, 1./3.), 3, 999);
  //uniform_unit_cube(&body[0], Nbody, std::pow(Nbody, 1./3.), 3);
  buildBinaryTree(&cell[0], &body[0], Nbody, levels);

  std::mt19937 gen(999);
  std::uniform_real_distribution uniform_dist(0., 1.);
  std::generate(Xbody.begin(), Xbody.end(), 
    [&]() { return std::complex<double>(uniform_dist(gen), 0.); });

  /*cell.erase(cell.begin() + 1, cell.begin() + Nleaf - 1);
  cell[0].Child[0] = 1; cell[0].Child[1] = Nleaf + 1;
  ncells = Nleaf + 1;
  levels = 1;*/

  CSR cellNear('N', cell, cell, theta);
  CSR cellFar('F', cell, cell, theta);
  CSR cellNeighbor(cellNear, cellFar);

  std::pair<double, double> timer(0, 0);
  std::vector<std::pair<long long, long long>> mapping(mpi_size, std::make_pair(0, 1));
  std::vector<std::pair<long long, long long>> tree(ncells);
  std::transform(cell.begin(), cell.end(), tree.begin(), [](const Cell& c) { return std::make_pair(c.Child[0], c.Child[1]); });
  
  for (long long i = 0; i <= levels; i++) {
    communicator[i] = ColCommMPI(&tree[0], &mapping[0], cellNeighbor.RowIndex.data(), cellNeighbor.ColIndex.data());
    communicator[i].timer = &timer;
  }

  std::vector<WellSeparatedApproximation> wsa(levels + 1);
  double h_construct_time = MPI_Wtime();

  for (long long l = 1; l <= levels; l++)
    wsa[l] = WellSeparatedApproximation(eval, epi, rank, communicator[l].oGlobal(), communicator[l].lenLocal(), &cell[0], cellFar, &body[0], wsa[l - 1]);
  
  h_construct_time = MPI_Wtime() - h_construct_time;
  MPI_Barrier(MPI_COMM_WORLD);
  double h2_construct_time = MPI_Wtime(), h2_construct_comm_time;

  A[levels] = H2Matrix(eval, epi, &cell[0], cellNear, cellFar, &body[0], wsa[levels], communicator[levels], A[levels], communicator[levels]);
  for (long long l = levels - 1; l >= 0; l--)
    A[l] = H2Matrix(eval, epi, &cell[0], cellNear, cellFar, &body[0], wsa[l], communicator[l], A[l + 1], communicator[l + 1]);

  MPI_Barrier(MPI_COMM_WORLD);
  h2_construct_time = MPI_Wtime() - h2_construct_time;
  h2_construct_comm_time = timer.first;
  timer.first = 0;

  long long llen = communicator[levels].lenLocal();
  long long gbegin = communicator[levels].oGlobal();
  long long body_local[2] = { cell[gbegin].Body[0], cell[gbegin + llen - 1].Body[1] };
  long long lenX = body_local[1] - body_local[0];
  std::vector<std::complex<double>> X1(lenX, std::complex<double>(0., 0.));
  std::vector<std::complex<double>> X2(lenX, std::complex<double>(0., 0.));

  H2MatrixSolver solver(&A[0], &cell[0], &communicator[0], levels);
  std::copy(&Xbody[0] + body_local[0], &Xbody[0] + body_local[1], &X1[0]);

  MPI_Barrier(MPI_COMM_WORLD);
  double matvec_time = MPI_Wtime(), matvec_comm_time;
  solver.matVecMul(&X1[0]);

  MPI_Barrier(MPI_COMM_WORLD);
  matvec_time = MPI_Wtime() - matvec_time;
  matvec_comm_time = timer.first;
  timer.first = 0;

  double cerr = 0.;
  double refmatvec_time = MPI_Wtime();

  mat_vec_reference(eval, lenX, Nbody, &X2[0], &Xbody[0], &body[body_local[0] * 3], &body[0]);
  refmatvec_time = MPI_Wtime() - refmatvec_time;

  solveRelErr(&cerr, &X1[0], &X2[0], lenX);

  std::fill(X1.begin(), X1.end(), std::complex<double>(0., 0.));

  MPI_Barrier(MPI_COMM_WORLD);
  double gmres_time = MPI_Wtime(), gmres_comm_time;
  std::pair<double, long long> gmres_ret = solver.solveGMRES(epi, &X1[0], &X2[0], 50, 40);

  MPI_Barrier(MPI_COMM_WORLD);
  gmres_time = MPI_Wtime() - gmres_time;
  gmres_comm_time = timer.first;
  timer.first = 0;

  if (mpi_rank == 0) {
    std::cout << "Construct Err: " << cerr << std::endl;
    std::cout << "H-Matrix Time: " << h_construct_time << std::endl;
    std::cout << "H^2-Matrix Time: " << h2_construct_time << ", " << h2_construct_comm_time << std::endl;
    std::cout << "Matvec Time: " << matvec_time << ", " << matvec_comm_time << std::endl;
    std::cout << "Dense Matvec Time: " << refmatvec_time << std::endl;
    std::cout << "GMRES Residual: " << gmres_ret.first << ", Iters: " << gmres_ret.second << std::endl;
    std::cout << "GMRES Time: " << gmres_time << ", " << gmres_comm_time << std::endl;
  }

  for (auto& c : communicator)
    c.free_all_comms();
  MPI_Finalize();
  return 0;
}

void uniform_unit_cube(double* bodies, long long nbodies, double diameter, long long dim) {
  long long side = ceil(pow(nbodies, 1. / dim));
  long long lens[3] = { dim > 0 ? side : 1, dim > 1 ? side : 1, dim > 2 ? side : 1 };
  double step = diameter / side;

  for (long long i = 0; i < lens[0]; ++i)
    for (long long j = 0; j < lens[1]; ++j)
       for (long long k = 0; k < lens[2]; ++k) {
    long long x = k + lens[2] * (j + lens[1] * i);
    if (x < nbodies) {
      bodies[x * 3] = i * step;
      bodies[x * 3 + 1] = j * step;
      bodies[x * 3 + 2] = k * step;
    }
  }
}

void uniform_unit_cube_rnd(double* bodies, long long nbodies, double diameter, long long dim, unsigned int seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution uniform_dist(0., diameter);

  std::array<double, 3>* b3 = reinterpret_cast<std::array<double, 3>*>(bodies);
  std::array<double, 3>* b3_end = reinterpret_cast<std::array<double, 3>*>(&bodies[3 * nbodies]);
  std::for_each(b3, b3_end, [&](std::array<double, 3>& body) {
    for (int i = 0; i < 3; i++)
      body[i] = i < dim ? uniform_dist(gen) : 0.;
  });
}

void mesh_sphere(double* bodies, long long nbodies, double r) {
  const double phi = M_PI * (3. - std::sqrt(5.));  // golden angle in radians
  const double d = r + r;
  const double r2 = r * r;

  for (long long i = 0; i < nbodies; ++i) {
    const double y = r - ((double)i / (double)(nbodies - 1)) * d;  // y goes from r to -r

    // Note: setting constant radius = 1 will produce a cylindrical shape
    const double radius = std::sqrt(r2 - y * y);  // radius at y
    const double theta = (double)i * phi;

    const double x = radius * std::cos(theta);
    const double z = radius * std::sin(theta);
    bodies[i * 3] = x;
    bodies[i * 3 + 1] = y;
    bodies[i * 3 + 2] = z;
  }
}

void solveRelErr(double* err_out, const std::complex<double>* X, const std::complex<double>* ref, long long lenX) {
  double err[2] = { 0., 0. };
  for (long long i = 0; i < lenX; i++) {
    std::complex<double> diff = X[i] - ref[i];
    err[0] = err[0] + (diff.real() * diff.real());
    err[1] = err[1] + (ref[i].real() * ref[i].real());
  }
  MPI_Allreduce(MPI_IN_PLACE, err, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  *err_out = std::sqrt(err[0] / err[1]);
}

void read_sorted_bodies(long long* nbodies, long long lbuckets, double* bodies, long long buckets[], const char* fname) {
  std::ifstream file(fname);

  long long curr = 1, cbegin = 0, iter = 0, len = *nbodies;
  while (iter < len && !file.eof()) {
    long long b = 0;
    double x = 0., y = 0., z = 0.;
    file >> x >> y >> z >> b;

    if (lbuckets < b)
      len = iter;
    else if (!file.eof()) {
      bodies[iter * 3] = x;
      bodies[iter * 3 + 1] = y;
      bodies[iter * 3 + 2] = z;
      while (curr < b && curr <= lbuckets) {
        buckets[curr - 1] = iter - cbegin;
        cbegin = iter;
        curr++;
      }
      iter++;
    }
  }
  while (curr <= lbuckets) {
    buckets[curr - 1] = iter - cbegin;
    cbegin = iter;
    curr++;
  }
  *nbodies = iter;
}
