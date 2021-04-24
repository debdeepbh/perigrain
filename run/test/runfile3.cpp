#include <iostream>
#include <vector>

#include  "/home/debdeep/Downloads/eigen-3.3.8/Eigen/Dense"
using namespace Eigen;

#include "../includes/particle.h"

using namespace std;

/// generalization of typedef to templates
//template <typename ttype, size_t dimension>
//using V = Array<ttype, 1, dimension>;

// Row vector that holds double values
template <size_t dimension>
using V = Array<double, 1, dimension>;

int main(int argc, char *argv[])
{

    vector<vector<int>> NbdArr(5);
    NbdArr[0] = {3, 2};
    //NbdArr[1].resize(0);
    NbdArr[1] = {};
    NbdArr[2] = {4, 0, 1};
    NbdArr[3] = { 2 };
    NbdArr[4] = {1, 2, 3, 0};

    // edit vectors
    NbdArr[3] = { 1, 3 };


    V<2> v;
    v = {3.5, 4.2};

    Particle<V<2>> P(5);
    //std::cout << P.NbdArr << std::endl;

    //P.pos = {{{2.4, 5.6}}, {{1.1, 2.1}}, {{4.0, 3.9}}, {{7.7, 3.2}}, {{0.1, 6.2}}};
    P.pos = {{2.4, 5.6}, {1.1, 2.1}, {4.0, 3.9}, {7.7, 3.2}, {0.1, 6.2}};

    //index row matrix
    vector<vector<int>> row_mat_ind(5);
    row_mat_ind = { {{0}}, {{1}}, {{2}}, {{3}}, {{4}} };
    std::cout << "Row linspace: " <<  row_mat_ind << std::endl;

    std::cout << "P.pos: " << P.pos << std::endl;

    P.NbdArr = NbdArr;
    std::cout << "NbdArr: " << P.NbdArr << std::endl;

    std::cout << "Nbd Pos: " << P.nbd_pos() << std::endl;
    //std::cout << eval_at_ind(P.pos, NbdArr) << std::endl;

    auto pos_row = eval_at_ind(P.pos, row_mat_ind);
    std::cout << "Pos row: " << pos_row << std::endl;

    std::cout << "Sum of graph and vector: " << P.nbd_pos() + P.pos  << std::endl;

    return 0;
}


