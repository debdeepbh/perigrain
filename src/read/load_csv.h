#include "read/mycsv.h" 

#include <stdio.h>
#include "particle/particle2.h"
#include "particle/contact.h"
#include "compat/overloads.h"

using namespace std;

// Compatibility for matlab-generated csv files
// Edit for vector containing indices
// Zeros are removed, index starts from 0, not 1
template <typename ttype>
void start_from_zero(vector<ttype> &V){
    unsigned ss = V.size();
    for (unsigned i = 0; i < ss; ++i) {
	V[i] -= 1;
    }
};



vector<ParticleN<2>> load_particles_univ_2(){
    string data_loc = "data/";

    vector<unsigned> tot_part = read_col<unsigned>(data_loc+"total_particles_univ.csv");
    unsigned total_particles_univ = tot_part[0];

    vector<ParticleN<2>> PArr;


    for (unsigned i = 0; i < total_particles_univ; ++i) {
	
	char buf[20];
	// matlab to c++: index increased by 1
	sprintf(buf, "_%05u.csv", i+1);
	// conver char to string
	string file_suffix = string(buf);

	string file_Pos = data_loc + "Pos_univ" + file_suffix;
	//std::cout << file_Pos << std::endl;

	auto V = read_rowvecs<RowVector2d>(file_Pos);
	unsigned total_nodes = V.size();


	ParticleN<2> P(total_nodes);
	P.pos = V;

	P.vol = read_col<double>(data_loc+"Vol_univ"+file_suffix);
	P.boundary_nodes = read_col<unsigned>(data_loc+"bdry_nodes_univ"+file_suffix);
	start_from_zero<unsigned> (P.boundary_nodes);

	P.disp = read_rowvecs<RowVector2d>(data_loc+"uold_univ"+file_suffix);
	//std::cout << P.disp << std::endl;
	P.vel = read_rowvecs<RowVector2d>(data_loc+"uolddot_univ"+file_suffix);
	P.acc = read_rowvecs<RowVector2d>(data_loc+"uolddotdot_univ"+file_suffix);
	P.extforce = read_rowvecs<RowVector2d>(data_loc+"extforce_univ"+file_suffix);
	//std::cout << P.acc << std::endl;

	// load a vector of variable length vectors, trim all the zeros
	P.NbdArr = read_vecvec<unsigned> (data_loc+"NbdArr_univ"+file_suffix, 1);
	// start from zero
	start_from_zero<vector<unsigned>>(P.NbdArr);

	//std::cout << P.NbdArr << std::endl;

	//P.xi_1 = read_vecvec<double> (data_loc+"xi_1_univ"+file_suffix);
	//P.xi_2 = read_vecvec<double> (data_loc+"xi_2_univ"+file_suffix);
	//P.xi_norm = read_vecvec<double> (data_loc+"xi_norm_univ"+file_suffix);
	//////  Issue with exporting xi: the arithmetic precision in matlab is different from that of C++, leading to VERY HIGH errors in C++.
	////// Better to do the arithmetic in C++ directly
	////// Topological information is not affected (node connectivity)
	////// Position of the nodes is ok too
	P.gen_xi();

	P.delta = read_col<double>(data_loc+"delta_univ"+file_suffix)[0];
	P.rho = read_col<double>(data_loc+"rho_univ"+file_suffix)[0];
	P.cnot = read_col<double>(data_loc+"cnot_univ"+file_suffix)[0];
	P.snot = read_col<double>(data_loc+"snot_univ"+file_suffix)[0];

	// read only if available
	if (0) {
	    P.torque_axis = read_col<unsigned>(data_loc+"torque_axis"+file_suffix)[0];
	    P.torque_val = read_col<double>(data_loc+"torque_val"+file_suffix)[0];
	}

	PArr.push_back(P);
    }

    //std::cout << PArr[0].disp << std::endl;
    //Particle<RowVector2d> P[total_shapes];
    //return total_shapes;
    return PArr;
};

Contact load_contact(){
    Contact contact;
    string data_loc = "data/";
    auto ci = read_col<double>(data_loc+"pairwise_properties.csv");
    //PairWise = [contact_radius; normal_stiffness; friction_coefficient; damping_ratio];
    contact.contact_rad = ci[0];
    contact.normal_stiffness  = ci[1];
    contact.friction_coefficient  = ci[2];
    contact.damping_ratio  = ci[3];

    return contact;
};

RectWall load_wall(){

    string data_loc = "data/";
    auto ci = read_col<double>(data_loc+"geom_wall_info.csv");
    RectWall Wall(ci[0]);
    if (ci.size() != 1) {
        Wall.left = ci[1];
        Wall.right = ci[2];
        Wall.top = ci[3];
        Wall.bottom = ci[4];
    }
    return Wall;
};


// write vector of RowVectors
template <typename ttype>
void write_rowvecs(string filename, vector<ttype> V) {
  ofstream file(filename);
  for (unsigned i = 0; i < V.size(); ++i) {
      unsigned dim = V[i].cols();
      for (unsigned j = 0; j < dim; ++j) {
	  file << V[i](j);
	  if (j < (dim-1)) {
	      file << " ";
	  }
      }
      file << "\n";
  }
  file.close();
};
