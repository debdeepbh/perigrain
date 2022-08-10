#ifndef PARTICLE_2_H
#define PARTICLE_2_H

#include <iostream>
#include <float.h>
#include <vector>
#include "Eigen/Dense"
using namespace Eigen;

#include "compat/overloads.h"

using namespace std;

#include <cstdlib>

//template <unsigned dim>
//double my_norm(Matrix<double, 1, dim> v){
//double out = 0;
//for (unsigned i = 0; i < dim; ++i) {
//out += v[i] * v[i];
//}
//out = sqrt(out);
//return out;
//};


// particle class
template <unsigned dim>
class ParticleN
{
    public:
	unsigned nnodes;
	vector<Matrix<double, 1, dim>> pos, disp, CurrPos, vel, acc, force, extforce;
	vector<double> vol;
	vector<unsigned> boundary_nodes;
	vector<unsigned> clamped_nodes;
	vector<vector<unsigned>> NbdArr;

	vector<unsigned> total_neighbors;

	//vector<vector<double>> xi_1;
	//vector<vector<double>> xi_2;

	vector<vector<Matrix<double, 1, dim>>> xi;
	vector<vector<double>> xi_norm;

	vector<vector<double>> stretch;

	double delta, rho, cnot, snot;

	int breakable, movable, stoppable;

	bool break_bonds;

	//torque
	unsigned torque_axis;
	double torque_val;


	ParticleN ();

	ParticleN (unsigned N){
	    pos.resize(N);
	    vol.resize(N);

	    disp.resize(N);
	    vel.resize(N);
	    acc.resize(N);
	    force.resize(N);
	    extforce.resize(N);
	    NbdArr.resize(N);
	    stretch.resize(N);
	    //boundary_nodes.resize(N);
	    nnodes = N;

	    xi.resize(N);
	    xi_norm.resize(N);

	    total_neighbors.resize(N);

	    break_bonds = 0;

	    breakable = 1;
	    movable = 1;
	    stoppable = 1;
	};

	// return the mean current position
	Matrix<double,1, dim> mean_CurrPos(){
	    Matrix<double,1, dim> mean;
	    mean.setZero();
	    for (unsigned i = 0; i < nnodes; i++) {
		mean += (pos[i] + disp[i]);
	    }
	    mean /= nnodes;
	    return mean;
	    
	};

	// computes peridynamic force
	// and updates public var: stretch
	vector<Matrix<double, 1, dim>> get_peridynamic_force(){
	    vector<Matrix<double, 1, dim>> tot_peri_force;
	    tot_peri_force.resize(nnodes);


	    //cout.precision(30);
	    //std::cout << "CurrPos:" <<  CurrPos << std::endl;

//#pragma omp parallel for num_threads(2)
//#pragma omp parallel for
	    for (unsigned i = 0; i < nnodes; ++i) {

		//std::cout << "Started " << omp_get_num_threads() << std::endl;

		//std::cout << omp_get_num_threads() << std::endl;
		//initialize as zero
		Matrix<double, 1, dim> v = Matrix<double, 1, dim>::Zero();
		//v *= 0;
		//unsigned max_nbds = total_neighbors[i];
		//stretch[i].resize(max_nbds);

//#pragma omp parallel for firstprivate(v) shared(i, stretch)
//#pragma omp parallel for num_threads(2)
		for (unsigned j = 0; j < total_neighbors[i]; ++j) {
		//std::cout << "Started " << omp_get_num_threads() << std::endl;
		    unsigned idx_nbr = NbdArr[i][j];
		    //// This causes HUGE round-off errors: when eta = 0, eta_p_xi_norm ~= xi_norm, even though eta_p_xi = xi !!!
		    //auto eta_p_xi = CurrPos[idx_nbr] - CurrPos[i];
		    //auto eta_p_xi_alt = CurrPos[idx_nbr] - CurrPos[i];
		    //auto eta_p_xi_alt = (pos[idx_nbr] + disp[idx_nbr]) - (pos[i] + disp[i]);
		    //auto eta_p_xi = (disp[idx_nbr] - disp[i]) + (pos[idx_nbr] - pos[i]);

		    auto eta_p_xi = (disp[idx_nbr] - disp[i]) + xi[i][j];
		    //double eta_p_xi_norm = my_norm<dim>(eta_p_xi);
		    double eta_p_xi_norm = eta_p_xi.norm();
		    auto unit_dir = eta_p_xi / eta_p_xi_norm;

		    double xi_norm_here = xi_norm[i][j];
		    //double xi_norm_here = xi[i][j].norm();
		    //double xi_norm_here = my_norm<dim>(xi[i][j]);

		    double diff_norm = (eta_p_xi_norm - xi_norm_here);
		    double str = diff_norm / xi_norm_here;
		    auto v_more = (cnot * str * vol[idx_nbr]) * unit_dir;
		    v += v_more;

		    stretch[i][j] = str;

		}

		tot_peri_force[i] = v;

		//// bond breaking
		//if (break_bonds) {
		//    // super ugly implementation
		//    // See if can be improved logically
		//    //
		//    // pointers that go in parallel with j, but point to memory locations within other vectors
		//    auto j_NbdArr = NbdArr[i].begin();
		//    auto j_xi = xi[i].begin();
		//    auto j_xi_norm = xi_norm[i].begin();
		//    for (auto j = stretch[i].begin(); j != stretch[i].end(); ++j) {
		//	//
		//	if (*j > snot) {

		//	    stretch[i].erase(j);

		//	    NbdArr[i].erase(j_NbdArr);
		//	    xi[i].erase(j_xi);
		//	    xi_norm[i].erase(j_xi_norm);

		//	    --j;
		//	    --j_NbdArr;
		//	    --j_xi;
		//	    --j_xi_norm;
		//	}

		//	++j_NbdArr;
		//	++j_xi;
		//	++j_xi_norm;
		//    }
		//}
		// end of bond breaking
	    }

	    return tot_peri_force;
	};

	//void remove_bonds(int t){
	void remove_bonds(){
	    if (break_bonds) {
//#pragma omp parallel for num_threads(2)
//#pragma omp parallel for
		for (unsigned i = 0; i < nnodes; ++i) {

		    // super ugly implementation
		    // See if can be improved logically
		    //
		    // pointers that go in parallel with j, but point to memory locations within other vectors
		    auto j_NbdArr = NbdArr[i].begin();
		    auto j_xi = xi[i].begin();
		    auto j_xi_norm = xi_norm[i].begin();
		    for (auto j = stretch[i].begin(); j != stretch[i].end(); ++j) {
			//
			// only break bonds with positive stretch
			if (*j > snot) {
			// Break with both positive and negative stretch, produces unrealistic output, e.g. for crack_prenotch() experiment
			//if (abs(*j) > snot) {

			    stretch[i].erase(j);

			    NbdArr[i].erase(j_NbdArr);
			    xi[i].erase(j_xi);
			    xi_norm[i].erase(j_xi_norm);

			    --total_neighbors[i];

			    --j;
			    --j_NbdArr;
			    --j_xi;
			    --j_xi_norm;
			}

			++j_NbdArr;
			++j_xi;
			++j_xi_norm;
		    }
		    // end ugly loop
		}
	    }
	};


	// Populate xi related information for the neighbors
	void gen_xi(){	
	    for (unsigned i = 0; i < nnodes; ++i) {
		unsigned ss = NbdArr[i].size();

		total_neighbors[i] = ss;

		xi_norm[i].resize(ss);
		xi[i].resize(ss);
		stretch[i].resize(ss);

		for (unsigned j = 0; j < ss; ++j) {
		    unsigned idx_nbr = NbdArr[i][j];
		    xi[i][j] = pos[idx_nbr] - pos[i];
		    xi_norm[i][j] = (xi[i][j]).norm();
		}
	    }
	};

	void print(){
	    std::cout << "delta: " << delta << std::endl;
	    std::cout << "rho: " << rho << std::endl;
	    std::cout << "cnot: " << cnot << std::endl;
	    std::cout << "snot: " << snot << std::endl;
	};


    private:
	/* data */
};

// print a particle
template <unsigned dim>
ostream& operator << (ostream& o, ParticleN<dim> P){
    o << "[Particle: "<< P.nnodes << " nodes]";
    return o;
};



#endif /* ifndef PARTICLE_2_H */
