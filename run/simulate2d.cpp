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


    // Load config file
    auto CFGV = ConfigVal();
    CFGV.read_file("config/main.conf");

    // load from hdf5 files
    //auto PArr = load_particles<dim>();
    //Contact CN = load_contact();
    //RectWall<dim> Wall = load_wall<dim>();
    auto PArr = load_particles<dim>(CFGV);
    Contact CN = load_contact(CFGV);
    RectWall<dim> Wall = load_wall<dim>(CFGV);

    // default value if not set
    Timeloop TL(100);

    //CFGV.print();
    //CFGV.apply<dim>(TL, CN, Wall);
    TL.apply_config(CFGV);
    CN.apply_config(CFGV);
    Wall.apply_config(CFGV);

    std::cout << "extforce_gradient" << TL.gradient_extforce << std::endl;

    // print info
    //CN.print();
    //Wall.print();
    //PArr[0].print();
	
    // Debug
    //std::cout << "Debug: making particles breakable so that the contact force utilizes all the nodes, not just the boundary nodes" << std::endl;
    //std::cout << "Debug: Note that breaking bonds depends only on: TL.enable_fracture." << std::endl;

    for (unsigned i = 0; i < PArr.size(); i++) {
	PArr[i].break_bonds = 1;
    }
  
auto start = system_clock::now(); 
    run_timeloop<dim> (PArr, TL, CN, Wall);

    // more compact code but much slower
    //run_timeloop_compact<dim> (PArr, TL, CN, Wall);
auto stop = system_clock::now(); 


    // are nanoseconds, microseconds, milliseconds, seconds, minutes, hours
    auto duration = duration_cast<seconds>(stop - start); 
    // To get the value of duration use the count(), member function on the duration object 
    cout << "Runtime: " << duration.count() << "s" << endl; 

    return 0;
}
