#include <iostream>
#include <vector>

#include  "/home/debdeep/Downloads/eigen-3.3.8/Eigen/Dense"
using namespace Eigen;

#include "../headers/nbdarr_clean.h"
//#include "../headers/particle.h"

using namespace std;

//// 2d vector
//typedef Array<double, 1, 2> V<2>;

/// generalization of typedef to templates
//template <typename ttype, size_t dimension>
//using V = Array<ttype, 1, dimension>;

// Row vector that holds double values
template <size_t dimension>
using V = Array<double, 1, dimension>;

int main(int argc, char *argv[])
{

    AdjList<int> NbdArr(5);

    NbdArr.graph[0] = {3, 2};
    NbdArr.graph[1] = {0, 3, 2};
    NbdArr.graph[2] = {4, 0, 1};
    NbdArr.graph[3] = { 2 };
    NbdArr.graph[4] = {1, 2, 3, 5};

    // edit vectors
    NbdArr.graph[3] = { 5, 3 };

    //NbdArr.print();

    V<2> v;
    v = {3.5, 4.2};
    V<2> u;
    u = v;
    vector<V<2>> vv;
    vv.resize(2);
    vv[0] = v;
    vv[1] = v + v;

    // Force of the neighbors
    AdjList<V<2>> NbdForce(5);

    //NbdForce.graph[0][0] = v;
    NbdForce.graph[0].push_back(v);
    NbdForce.graph[0].push_back(v+v);
    NbdForce.graph[1].push_back(4 * v);

    //NbdForce.print();
    std::cout << NbdForce << std::endl;
    

    vector<int> ww = NbdForce.number_of_nbrs();
    std::cout << ww[0] << std::endl;

    std::cout << NbdForce.max_number_of_nbrs() << std::endl;

    vector<double> b = {1, 4, 9, 1.3};
    //c = b;
    auto rs = b;
    rs.pop_back();

    //auto c = b + rs;
    
    //for (int i = 0; i < c.size(); ++i) {
       //std::cout << c[i] << std::endl; 
    //}

    auto FF = NbdForce + NbdForce;
    //FF.print();
    //std::cout << FF << std::endl;

    //Particle P(5);


    // pseudo code
    //Particle P;
    //NA = P.NbdArr;
    //NF = P.get_nbd_Force();
    //PP = P.get_nbd_Pos();
    //xi = (PP - P.Pos());
    //eta = ;
    //sum_nbd() /
    //
    // P.force()gt


    //V<2> cc = v * v;
    //V<2> p(2,5);
    //std::cout << p *p << std::endl;

    Particle<V<2>> P(5);
    std::cout << P.NbdArr << std::endl;

    P.pos = {{2.4, 5.6}, {1.1, 2.1}, {4.0, 3.9}, {7.7, 3.2}, {0.1, 6.2}};

    std::cout << P.pos << std::endl;

    P.NbdArr = NbdArr;
    std::cout << P.NbdArr << std::endl;

    std::cout << P.nbd_pos() << std::endl;

    return 0;
}


