#include <iostream>
#include <vector>

//#include  "/home/debdeep/Downloads/eigen-3.3.8/Eigen/Dense"
//using namespace Eigen;
//#include "../headers/nbdarr.h"

#include "../headers/nbdarr_clean.h"

using namespace std;


int main(int argc, char *argv[])
{
    Particle P(5);

    //P.NbdArr[0] = {3, 2};
    //P.NbdArr[1] = {0, 3, 2};
    //P.NbdArr[2] = {4, 0, 1};
    //P.NbdArr[3] = { 2 };

    vector<double> Pos;
    Pos = {8.1, 2.34, 1.7, 4.5, 0.5};

    P.print_NbdArr();

    VectorXd a(2) =  {3, 5} ;
    //a << 75, 5;
    //VectorXd a {{1.5, 2.5, 3.5}};  

    std::cout << a << std::endl;

    return 0;
}


