#define EIGEN_DONT_PARALLELIZE
#include <chrono> 
using namespace std::chrono; 
# include <string>
# include <iostream>
#include "Eigen/Dense"
using namespace Eigen;

// redirect output to file with freopen()
#include <cstdio>

#include<omp.h>

//#include "read/varload.h"

//#include "particle/particle.h"
//#include "particle/particle2.h"
//#include "read/load_mesh.h"

//#include "read/load_csv.h"
#include "read/load_hdf5.h"
#include "read/read_config.h"
#include "particle/timeloop.h"


//#include "particle/nbdarr.h"

using namespace std;

int main(){
    const unsigned dim = 2;
    //Eigen::initParallel();

    // Redirect stdout to file
    //freopen( "output.log", "w", stdout );
    //freopen( "error.log", "w", stderr );

    //std::cout << "Available threads: " <<  omp_get_num_threads() << std::endl;
    // Allow nested parallel computation
    omp_set_nested(1);

    // load from hdf5 files
    auto PArr = load_particles<dim>();
    Contact CN = load_contact();
    RectWall<dim> Wall = load_wall<dim>();

    // Load config file
    auto CFGV = ConfigVal();
    CFGV.read_file("config/main.conf");
    CFGV.print();

    // update from config values
    Timeloop TL(CFGV.timesteps, CFGV.modulo);
    TL.dt = CFGV.dt;
    TL.do_resume = CFGV.do_resume;
    TL.wall_resume = CFGV.wall_resume;
    TL.resume_ind = CFGV.resume_ind;
    TL.save_file = CFGV.save_file;
    TL.enable_fracture = CFGV.enable_fracture;
    TL.run_parallel = CFGV.is_parallel;

    CN.allow_damping = CFGV.allow_damping;
    CN.allow_friction = CFGV.allow_friction;


    CN.nl_bdry_only = CFGV.nl_bdry_only;

    // Override if specified in config file
    if (CFGV.damping_ratio != (-1)) {
	CN.damping_ratio = CFGV.damping_ratio;
    }
    if (CFGV.friction_coefficient != (-1)) {
	CN.friction_coefficient = CFGV.friction_coefficient;
    }
    if (CFGV.normal_stiffness != (-1)) {
	CN.normal_stiffness = CFGV.normal_stiffness;
    }

    // self-contact
    CN.self_contact = CFGV.self_contact;

    if (CFGV.self_contact_rad != (-1)) {
	CN.self_contact_rad = CFGV.self_contact_rad;
    } else {
	CN.self_contact_rad = CN.contact_rad;
    }

    // wall
    if (CFGV.wall_top != (-999)) {
	if (dim ==2) {
	    Wall.top = CFGV.wall_top;
	}
	else{
	    Wall.z_max = CFGV.wall_top;
	}
    }
    if (CFGV.wall_right != (-999)) {
	if (dim ==2) {
	    Wall.right = CFGV.wall_right;
	}
	else{
	    Wall.y_max = CFGV.wall_right;
	}
    }

    // Load wall dimension (if defined) and speed 
    if (dim==2) {
	if (CFGV.wall_left != (-999)) { Wall.left = CFGV.wall_left; }
	if (CFGV.wall_right != (-999)) { Wall.right = CFGV.wall_right; }
	if (CFGV.wall_top != (-999)) { Wall.top = CFGV.wall_top; }
	if (CFGV.wall_bottom != (-999)) { Wall.bottom = CFGV.wall_bottom; }

	Wall.speed_left = CFGV.speed_wall_left;
	Wall.speed_right = CFGV.speed_wall_right;
	Wall.speed_top = CFGV.speed_wall_top;
	Wall.speed_bottom = CFGV.speed_wall_bottom;
    }
    else{
	if (CFGV.wall_x_min != (-999)) { Wall.x_min = CFGV.wall_x_min; }
	if (CFGV.wall_y_min != (-999)) { Wall.y_min = CFGV.wall_y_min; }
	if (CFGV.wall_z_min != (-999)) { Wall.z_min = CFGV.wall_z_min; }
	if (CFGV.wall_x_max != (-999)) { Wall.x_max = CFGV.wall_x_max; }
	if (CFGV.wall_y_max != (-999)) { Wall.y_max = CFGV.wall_y_max; }
	if (CFGV.wall_z_max != (-999)) { Wall.z_max = CFGV.wall_z_max; }

	Wall.speed_x_min = CFGV.speed_wall_x_min;
	Wall.speed_y_min = CFGV.speed_wall_y_min;
	Wall.speed_z_min = CFGV.speed_wall_z_min;
	Wall.speed_x_max = CFGV.speed_wall_x_max;
	Wall.speed_y_max = CFGV.speed_wall_y_max;
	Wall.speed_z_max = CFGV.speed_wall_z_max;
    }





    // print info
    CN.print();
    Wall.print();
    PArr[0].print();
	
    // Debug
    std::cout << "Debug: making particles breakable so that the contact force utilizes all the nodes, not just the boundary nodes" << std::endl;
    std::cout << "Debug: Note that breaking bonds depends only on: TL.enable_fracture." << std::endl;
    for (unsigned i = 0; i < PArr.size(); i++) {
	PArr[i].break_bonds = 1;
    }
  
auto start = system_clock::now(); 
    run_timeloop_worm<dim> (PArr, TL, CN, Wall);
auto stop = system_clock::now(); 


    // are nanoseconds, microseconds, milliseconds, seconds, minutes, hours
    auto duration = duration_cast<seconds>(stop - start); 
    // To get the value of duration use the count(), member function on the duration object 
    cout << "Runtime: " << duration.count() << "s" << endl; 

    return 0;
}
