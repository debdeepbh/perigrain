#ifndef lOAD_HDF5_H
#define lOAD_HDF5_H 

#include "read/rw_hdf5.h" 

#include "particle/particle2.h"
#include "particle/contact.h"
#include "compat/overloads.h"

#include "read/read_config.h"
//#include "read/load_hdf5.h"

#include <vector>
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

// convert connectivity matrix to NbdArr
auto conn2NArr(vector<Matrix<unsigned, 1, 2>> Conn, unsigned nnodes){
    vector<vector<unsigned>> NArr;
    NArr.resize(nnodes);
    for (unsigned l = 0; l < Conn.size(); l++) {
        auto i = Conn[l](0);
        auto j = Conn[l](1);
	NArr[i].push_back(j);
	NArr[j].push_back(i);
    }
    return NArr;
};


template <unsigned dim>
vector<ParticleN<dim>> load_particles(ConfigVal CFGV){
    string data_loc = "data/hdf5/";
    string h5file = "all.h5";
    string filename = data_loc + h5file;

    //std::cout << "I am here" << std::endl;
    auto tot_part = load_col<unsigned>(filename, "total_particles_univ");
    unsigned total_particles_univ = tot_part[0];

    std::cout << "Reading total particles: " << total_particles_univ << std::endl;

    vector<ParticleN<dim>> PArr;


    for (unsigned i = 0; i < total_particles_univ; ++i) {
	
	char buf[20];
	// matlab to c++: index increased by 1
	sprintf(buf, "P_%05u", i+1);
	// conver char to string
	string file_suffix = string(buf);

	auto V = load_rowvecs<double, dim>(filename, file_suffix+"/Pos");
	unsigned total_nodes = V.size();

	ParticleN<dim> P(total_nodes);
	P.pos = V;

	P.vol = load_col<double>(filename, file_suffix+"/Vol");
	P.boundary_nodes = load_col<unsigned>(filename, file_suffix+"/bdry_nodes");

	// use load_rank1 to load rank-1 arrays (i.e. vectors). Works for empty arrays too!
	P.clamped_nodes = load_rank1<unsigned>(filename, file_suffix+"/clamped_nodes");


	//start_from_zero<unsigned> (P.boundary_nodes);

	P.disp = load_rowvecs<double, dim>(filename, file_suffix+"/disp");
	P.vel = load_rowvecs<double, dim>(filename, file_suffix+"/vel");
	P.acc = load_rowvecs<double, dim>(filename, file_suffix+"/acc");

	P.extforce = load_rowvecs<double, dim>(filename, file_suffix+"/extforce");

	//// load a vector of variable length vectors, trim all the zeros
	//// for csv
	//P.NbdArr = read_vecvec<unsigned> (data_loc+"NbdArr_univ"+file_suffix, 1);
	//// start from zero
	//start_from_zero<vector<unsigned>>(P.NbdArr);
	
	auto Conn = load_rowvecs<unsigned, 2>(filename, file_suffix+"/Connectivity");
	P.NbdArr = conn2NArr(Conn, total_nodes);
	//std::cout << P.NbdArr << std::endl;

	P.gen_xi();

	P.delta = load_col<double>(filename, file_suffix+"/delta")[0];
    //std::cout << "Now here" << std::endl;
	P.rho = load_col<double>(filename, file_suffix+"/rho")[0];
	P.cnot = load_col<double>(filename, file_suffix+"/cnot")[0];
	P.snot = load_col<double>(filename, file_suffix+"/snot")[0];

	P.movable = load_col<int>(filename, file_suffix+"/movable")[0];
	P.breakable = load_col<int>(filename, file_suffix+"/breakable")[0];
	P.stoppable = load_col<int>(filename, file_suffix+"/stoppable")[0];


	if (CFGV.enable_torque) {
	    P.torque_axis = load_col<unsigned>(filename, file_suffix+"/torque_axis")[0];
	    P.torque_val = load_col<double>(filename, file_suffix+"/torque_val")[0];
	}
	
	//// python output saves scalars
	//P.delta = load_col<double>(filename, file_suffix+"/delta");
	//P.rho = load_col<double>(filename, file_suffix+"/rho");
	//P.cnot = load_col<double>(filename, file_suffix+"/cnot");
	//P.snot = load_col<double>(filename, file_suffix+"/snot");


	PArr.push_back(P);

    }
    std::cout << "Done reading particles." << std::endl;

    ////return total_shapes;
    return PArr;
};

Contact load_contact(ConfigVal CFGV){
    Contact contact;
    string data_loc = "data/hdf5/";
    string h5file = "all.h5";
    string filename = data_loc + h5file;

    contact.contact_rad = load_col<double>(filename, "pairwise/contact_radius")[0];
    contact.normal_stiffness = load_col<double>(filename, "pairwise/normal_stiffness")[0];
    contact.friction_coefficient = load_col<double>(filename, "pairwise/friction_coefficient")[0];
    contact.damping_ratio = load_col<double>(filename, "pairwise/damping_ratio")[0];

    ////python saves scalars
    //contact.contact_rad = load_col<double>(filename, "pairwise/contact_radius");
    //contact.normal_stiffness = load_col<double>(filename, "pairwise/normal_stiffness");
    //contact.friction_coefficient = load_col<double>(filename, "pairwise/friction_coefficient");
    //contact.damping_ratio = load_col<double>(filename, "pairwise/damping_ratio");

    std::cout << "Done reading contact." << std::endl;
    return contact;
};

//// Make this templated
//RectWall load_wall(){
    //string data_loc = "data/hdf5/";
    //string h5file = "all.h5";
    //string filename = data_loc + h5file;

    //auto aw = (bool) load_col<unsigned>(filename, "wall/allow_wall")[0];
    //RectWall Wall(aw);
    //auto ci = load_col<double>(filename, "wall/geom_wall_info");
    //if (ci.size() != 1) {
	//Wall.left = ci[0];
	//Wall.right = ci[1];
	//Wall.top = ci[2];
	//Wall.bottom = ci[3];
    //}
    //std::cout << "Done reading wall size: " << Wall.lrtb() << std::endl;
    //return Wall;
//};


template <unsigned dim>
RectWall<dim> load_wall(ConfigVal CFGV){
    string data_loc = "data/hdf5/";
    string h5file = "all.h5";
    string filename = data_loc + h5file;

    auto aw = (bool) load_col<unsigned>(filename, "wall/allow_wall")[0];
    RectWall<dim> Wall(aw);
    auto ci = load_col<double>(filename, "wall/geom_wall_info");
    if (ci.size() != 1) {
	if (dim == 2) {
	    Wall.left = ci[0];
	    Wall.right = ci[1];
	    Wall.top = ci[2];
	    Wall.bottom = ci[3];
	}
	else if (dim == 3) {
	    Wall.x_min = ci[0];
	    Wall.y_min = ci[1];
	    Wall.z_min = ci[2];
	    Wall.x_max = ci[3];
	    Wall.y_max = ci[4];
	    Wall.z_max = ci[5];
	}
	else{
	    std::cout << "hdf data geom_wall_info length is not 4 or 6. " << std::endl;
	}
    }
    //std::cout << "Done reading wall size: " << Wall.lrtb() << std::endl;
    std::cout << "Done reading wall size." << std::endl;
    return Wall;
};

//RectWall3d load_wall_3d(){
    //string data_loc = "data/hdf5/";
    //string h5file = "all.h5";
    //string filename = data_loc + h5file;

    //auto aw = (bool) load_col<unsigned>(filename, "wall/allow_wall")[0];
    //RectWall3d Wall(aw);
    //auto ci = load_col<double>(filename, "wall/geom_wall_info");
    //if (ci.size() != 1) {
	//Wall.x_min = ci[0];
	//Wall.y_min = ci[1];
	//Wall.z_min = ci[2];
	//Wall.x_max = ci[3];
	//Wall.y_max = ci[4];
	//Wall.z_max = ci[5];
    //}
    //std::cout << "Done reading wall size: " << Wall.lrtb() << std::endl;
    //return Wall;
//};


//// write vector of RowVectors
//template <typename ttype>
//void write_rowvecs(string filename, vector<ttype> V) {
  //ofstream file(filename);
  //for (unsigned i = 0; i < V.size(); ++i) {
      //unsigned dim = V[i].cols();
      //for (unsigned j = 0; j < dim; ++j) {
	  //file << V[i](j);
	  //if (j < (dim-1)) {
	      //file << " ";
	  //}
      //}
      //file << "\n";
  //}
  //file.close();
//};
//
#endif /* ifndef lOAD_HDF5_H */
