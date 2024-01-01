
#include <geometry.hpp>
#include <kernel.hpp>
#include <build_tree.hpp>
#include <basis.hpp>
#include <comm.hpp>
#include <sparse_row.hpp>
#include <linalg.hpp>
#include <umv.hpp>

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int64_t Nbody = argc > 1 ? atol(argv[1]) : 8192;
  double theta = argc > 2 ? atof(argv[2]) : 1e0;
  int64_t leaf_size = argc > 3 ? atol(argv[3]) : 256;
  double epi = argc > 4 ? atof(argv[4]) : 1e-10;
  const char* fname = argc > 5 ? argv[5] : NULL;

  leaf_size = Nbody < leaf_size ? Nbody : leaf_size;
  int64_t levels = (int64_t)log2((double)Nbody / leaf_size);
  int64_t Nleaf = (int64_t)1 << levels;
  int64_t ncells = Nleaf + Nleaf - 1;
  
  Laplace3D eval(1);
  //Yukawa3D eval(1, 1.);
  //Gaussian eval(0.2);
  
  std::vector<double> body(Nbody * 3);
  std::vector<double> Xbody(Nbody);
  std::vector<Cell> cell(ncells);
  CSR cellNear, cellFar;
  std::vector<CSR> rels_far(levels + 1), rels_near(levels + 1);

  std::vector<CellComm> cell_comm(levels + 1);
  std::vector<Base> basis(levels + 1);
  std::vector<Node> nodes(levels + 1);

  if (fname == NULL) {
    mesh_unit_sphere(&body[0], Nbody, std::pow(Nbody, 1./2.));
    //mesh_unit_cube(&body[0], Nbody);
    //uniform_unit_cube(&body[0], Nbody, 3);
    buildTree(&cell[0], &body[0], Nbody, levels);
  }
  else {
    std::vector<int64_t> buckets(Nleaf);
    read_sorted_bodies(&Nbody, Nleaf, &body[0], &buckets[0], fname);
    //buildTreeBuckets(cell, body, buckets, levels);
    buildTree(&cell[0], &body[0], Nbody, levels);
  }
  body_neutral_charge(&Xbody[0], Nbody, 999);

  /*cell.erase(cell.begin() + 1, cell.begin() + Nleaf - 1);
  cell[0].Child[0] = 1; cell[0].Child[1] = Nleaf + 1;
  ncells = Nleaf + 1;
  levels = 1;*/

  traverse('N', &cellNear, ncells, &cell[0], theta);
  traverse('F', &cellFar, ncells, &cell[0], theta);

  std::pair<double, double> timer(0, 0);
  buildComm(&cell_comm[0], ncells, &cell[0], &cellFar, &cellNear, levels);
  for (int64_t i = 0; i <= levels; i++) {
    cell_comm[i].timer = &timer;
  }
  relations(&rels_near[0], &cellNear, levels, &cell_comm[0]);
  relations(&rels_far[0], &cellFar, levels, &cell_comm[0]);

  int64_t lbegin = 0, llen = 0;
  content_length(&llen, NULL, &lbegin, &cell_comm[levels]);
  int64_t gbegin = cell_comm[levels].iGlobal(lbegin);

  MPI_Barrier(MPI_COMM_WORLD);
  double construct_time = MPI_Wtime(), construct_comm_time;
  buildBasis(eval, &basis[0], &cell[0], &cellNear, levels, &cell_comm[0], &body[0], Nbody, epi, 4);

  MPI_Barrier(MPI_COMM_WORLD);
  construct_time = MPI_Wtime() - construct_time;
  construct_comm_time = timer.first;
  timer.first = 0;

  allocNodes(&nodes[0], &basis[0], &rels_near[0], &rels_far[0], &cell_comm[0], levels);

  evalD(eval, nodes[levels].A, &cellNear, &cell[0], &body[0], &cell_comm[levels]);
  for (int64_t i = 0; i <= levels; i++)
    evalS(eval, nodes[i].S, &basis[i], &rels_far[i], &cell_comm[i]);

  int64_t lenX = llen * basis[levels].dimN;
  std::vector<double> X1(lenX, 0);
  std::vector<double> X2(lenX, 0);

  std::copy(Xbody.begin() + cell[gbegin].Body[0], Xbody.begin() + cell[gbegin + llen - 1].Body[1], &X1[0]);
  double matvec_time = MPI_Wtime(), matvec_comm_time;
  //matVecA(&nodes[0], &basis[0], &cellNear, &X1[0], &cell_comm[0], levels);
  matVecA(eval, &basis[0], &body[0], &cell[0], &cellNear, &cellFar, &X1[0], &cell_comm[0], levels);

  matvec_time = MPI_Wtime() - matvec_time;
  matvec_comm_time = timer.first;
  timer.first = 0;

  double cerr = 0.;
  int64_t body_local[2] = { cell[gbegin].Body[0], cell[gbegin + llen - 1].Body[1] };
  mat_vec_reference(eval, body_local[1] - body_local[0], Nbody, &X2[0], &Xbody[0], &body[body_local[0] * 3], &body[0]);

  solveRelErr(&cerr, &X1[0], &X2[0], body_local[1] - body_local[0]);

  std::cout << cerr << std::endl;
  std::cout << construct_time << ", " << construct_comm_time << std::endl;
  std::cout << matvec_time << ", " << matvec_comm_time << std::endl;
  
  for (int64_t i = 0; i <= levels; i++) {
    basis_free(&basis[i]);
    node_free(&nodes[i]);
  }
  cellComm_free(&cell_comm[0], levels);

  MPI_Finalize();
  return 0;
}
