#ifndef NBDARR_H
#define NBDARR_H 

#include "Eigen/Dense"
using namespace Eigen;

#include "particle/particle2.h"
#include <vector>

using namespace std;

template <typename ttype>
vector<vector<unsigned>> gen_NbdArr(vector<ttype> Pos_center, vector<ttype> Pos_nbr, double delta, bool remove_center){
    vector<vector<unsigned>> NbdArr;
    NbdArr.resize(Pos_center.size());

    for (unsigned i = 0; i < Pos_center.size(); ++i) {
	for (unsigned j = 0; j < Pos_nbr.size(); ++j) {
	    if ((Pos_center[i] - Pos_nbr[j]).norm() < delta){
		if (!remove_center){
		    NbdArr[i].push_back(j);
		}else{
		    if (j != i) {
			NbdArr[i].push_back(j);
		    }
		}
	    }
	}
    }
    return NbdArr;
};

// probably irrelevant
template <unsigned dim>
vector<vector<unsigned>> gen_NbdArr(ParticleN<dim> P_C, ParticleN<dim> P_N, double delta, bool remove_center){
    vector<vector<unsigned>> NbdArr;
    NbdArr.resize(P_C.nnodes);

    for (unsigned i = 0; i < P_C.size(); ++i) {
	for (unsigned j = 0; j < P_N.size(); ++j) {
	    if ((P_C[i] - P_N[j]).norm() < delta){
		if (!remove_center){
		    NbdArr[i].push_back(j);
		}else{
		    if (j != i) {
			NbdArr[i].push_back(j);
		    }
		}
	    }
	}
    }
    return NbdArr;
};

#endif /* ifndef NBDARR_H */
