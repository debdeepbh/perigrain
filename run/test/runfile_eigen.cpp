// Only uses vector and Eigen, to be compatible with modern community conventions
#include <iostream>
#include <vector>

#include  "/home/debdeep/Downloads/eigen-3.3.8/Eigen/Dense"
using namespace Eigen;

//#include "../includes/particle_valarr.h"

using namespace std;

/// generalization of typedef to templates
//template <typename ttype, size_t dimension>
//using V = Array<ttype, 1, dimension>;

//// Row vector that holds double values
template <size_t dimension>
using V = Array<double, 1, dimension>;

int main(int argc, char *argv[])
{
    int N= 5;

    //Array<V<2>, 5, 1> NbdArr;

    valarray<double> va = {1, 3, 0, 4};

    //valarray<double> vc = va + 5.0;
    //std::cout << vc << std::endl;

    valarray<valarray<int>> NbdArr(5);

    NbdArr[0] = {3, 2};
    //NbdArr[1].resize(0);
    NbdArr[1] = {};
    NbdArr[2] = {4, 0, 1};
    NbdArr[3] = { 2 };
    NbdArr[4] = {1, 2, 3, 0};

    //std::cout << NbdArr << std::endl;

    valarray<valarray<int>> N2 = 2.0 *  NbdArr;
    //valarray<valarray<int>> N2 = NbdArr +  NbdArr;
    std::cout << "N2: " << N2 << std::endl;

    std::valarray<double> data = {0,1,2,3,4,5,6,7,8,9};
    valarray<double> data2 = data.apply([](double n)->double {
                    return std::pow(n, 2);
                });

    std::cout << "New data: "  << data2 << std::endl;

    valarray<int> v = {1,2,3,4,5,6,7,8,9,10};
    v = v.apply([](int n)->int {
                    return std::round(std::tgamma(n+1));
                });
    //data[data > 5.0] = -1;
    //std::cout << "Search: " <<  data << std::endl;

    Particle<V<2>> P(5);

    //P.pos = {{{2.4, 5.6}}, {{1.1, 2.1}}, {{4.0, 3.9}}, {{7.7, 3.2}}, {{0.1, 6.2}}};
    P.pos = {{2.4, 5.6}, {1.1, 2.1}, {4.0, 3.9}, {7.7, 3.2}, {0.1, 6.2}};

    P.disp = {{2.8, 5.5}, {1.9, 2.2}, {4.9, 3.5}, {7.0, 3.7}, {2.2, 4.3}};

    ////index row matrix
    //vector<vector<int>> row_mat_ind(5);
    //row_mat_ind = { {{0}}, {{1}}, {{2}}, {{3}}, {{4}} };
    //std::cout << "Row linspace: " <<  row_mat_ind << std::endl;

    std::cout << "P.pos: " << P.pos << std::endl;

    P.NbdArr = NbdArr;
    std::cout << "NbdArr: " << P.NbdArr << std::endl;

    auto ev = eval_at_ind(P.pos, NbdArr);

    std::valarray<double> d2 = data + data;

    std::cout << 2 * data << std::endl;

    std::cout << "Nbd Pos: " << P.nbd_pos() << std::endl;


    //std::cout << eval_at_ind(P.pos, NbdArr) << std::endl;

    //auto pos_row = eval_at_ind(P.pos, row_mat_ind);
    //std::cout << "Pos row: " << pos_row << std::endl;

    //std::cout << "Sum of graph and vector: " << P.nbd_pos() + P.pos  << std::endl;
    
/**********************************************************************/
    // Peridynamics

    valarray<valarray<V<2>>> xi = P.nbd_pos();
    for (int i = 0; i < N; ++i) {
	xi[i] -= P.pos[i];
    }
    std::cout << "xi: " <<  xi << std::endl;

    valarray<valarray<V<2>>> eta = P.nbd_pos();
    for (int i = 0; i < N; ++i) {
	eta[i] -= P.disp[i];
    }

    valarray<valarray<V<2>>> eta_p_xi = eta + xi;

    std::cout << "eta+xi: " << eta_p_xi << std::endl;

    //valarray<valarray<V<2>>> eta_p_xi_norm = 



   // // update nbdarr
   // for (int i = 0; i < N; ++i) {
   //     for (int j = 0; j < count; ++j) {
   //         if (s > s_0) {
   //             P.NbdArr[i].erase(j);
   //             xi[i].erase(j);
   //         }
   //         
   //     }
   // }

    return 0;
}


