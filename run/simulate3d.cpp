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

#include "read/load_hdf5.h"
#include "read/read_config.h"

#include "particle/timeloop.h"


//#include "particle/nbdarr.h"

using namespace std;

int main(){
    //const unsigned dim = 2;
    const unsigned dim = 3;
    //Eigen::initParallel();

    // redirect stdout to file
    //freopen( "output.log", "w", stdout );
    //freopen( "error.log", "w", stderr );

    //std::cout << omp_get_num_threads() << std::endl;
    
    // Allow nested parallel computation
    omp_set_nested(1);

    auto PArr = load_particles<dim>();
    Contact CN = load_contact();
    auto Wall = load_wall<dim>();

    // Load config file
    auto CFGV = ConfigVal();
    CFGV.read_file("config/main.conf");
    CFGV.print();

    // update from config values
    Timeloop TL(CFGV.timesteps, CFGV.modulo);
    TL.dt = CFGV.dt;
    TL.do_resume = CFGV.do_resume;
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

    // print info
    CN.print();
    Wall.print();
    PArr[0].print();

    // Debug
    std::cout << "Debug: making particles breakable so that the contact force utilizes all the nodes, not just the boundary nodes" << std::endl;
    std::cout << "Debug: Note that breaking bonds depends only on: TL.enable_fracture." << std::endl;
    for (unsigned i = 0; i < PArr.size(); i++) {
	//PArr[i].break_bonds = 1;
    }
  
auto start = system_clock::now(); 
	// issue: Wall is not RectWall3d yet
    run_timeloop<dim> (PArr, TL, CN, Wall);
auto stop = system_clock::now(); 

    // are nanoseconds, microseconds, milliseconds, seconds, minutes, hours
    auto duration = duration_cast<seconds>(stop - start); 
    // To get the value of duration use the count(), member function on the duration object 
    cout << "Runtime: " << duration.count() << "s" << endl; 

    return 0;
}
