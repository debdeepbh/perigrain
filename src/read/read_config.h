#ifndef READ_CONF
#define READ_CONF 

#include <iostream>
#include <fstream>
#include <algorithm>

//#include "particle/timeloop.h"
//#include "particle/contact.h"

//using namespace std;

//template <unsigned dim>
class ConfigVal
{
public:
    unsigned timesteps;
    unsigned modulo;
    double dt;
    bool is_parallel;

    bool do_resume, wall_resume;
    unsigned resume_ind;
    bool save_file;
    bool enable_fracture;
    bool nl_bdry_only;

    bool allow_damping, allow_friction;
    double  normal_stiffness; 
    double damping_ratio, friction_coefficient;

    bool self_contact;
    double self_contact_rad;

    // torque
    bool enable_torque;

    // messing with the wall
    double wall_left, wall_right, wall_top, wall_bottom;
    double wall_x_min, wall_y_min, wall_z_min, wall_x_max, wall_y_max, wall_z_max;

    double speed_wall_left, speed_wall_right, speed_wall_top, speed_wall_bottom;
    double speed_wall_x_min, speed_wall_y_min, speed_wall_z_min, speed_wall_x_max, speed_wall_y_max, speed_wall_z_max;

    bool gradient_extforce = 0;
    unsigned extforce_maxstep = 0;

    ConfigVal (){
	//Timeloop TL(8000, 100);
	//TL.dt = 0.02/1e5;
	//
	timesteps = 8000;
	modulo = 100;
	dt = 0.02/1e5;
	is_parallel = 1;

	// TL
	do_resume = 0;
	wall_resume = 0;
	resume_ind = 50;
	save_file = 1;
	enable_fracture = 0;
	nl_bdry_only = 0;

	// CN
	allow_damping = 1;
	allow_friction = 1;

	// self_contact
	self_contact = 1;
	self_contact_rad = -1;

	enable_torque = 0;

	// -1 means not specified
	normal_stiffness = -1;
	damping_ratio = -1; 
	friction_coefficient = -1;
	
	// default value implies not loaded from the file
       wall_left   = -999;
       wall_right  = -999;
       wall_top    = -999;
       wall_bottom = -999;
       wall_x_min = -999;
       wall_y_min = -999;
       wall_z_min = -999;
       wall_x_max = -999;
       wall_y_max = -999;
       wall_z_max = -999;

       // default value is zero, which is correct, if not defined
       speed_wall_left   = 0;
       speed_wall_right  = 0;
       speed_wall_top    = 0;
       speed_wall_bottom = 0;

       speed_wall_x_min = 0;
       speed_wall_y_min = 0;
       speed_wall_z_min = 0;
       speed_wall_x_max = 0;
       speed_wall_y_max = 0;
       speed_wall_z_max = 0;

       // gradient of applied force 
    };

    // Read values from a file: name = value format, (whitespace is removed), # for comment
    void read_file(std::string filename){

	std::ifstream cFile (filename);
	if (cFile.is_open())
	{
	    std::cout << "Reading config file: " << filename << std::endl;
	    std::string line;
	    while(getline(cFile, line)){
		line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
		if(line[0] == '#' || line.empty())
		    continue;
		auto delimiterPos = line.find("=");
		auto name = line.substr(0, delimiterPos);
		auto value = line.substr(delimiterPos + 1);

		if (name == "timesteps") {
		    timesteps = std::stoi(value);
		}
		else if (name == "modulo") {
		    modulo = std::stoi(value);
		}
		else if (name == "dt") {
		    dt = std::stod(value);
		}
		else if (name == "is_parallel") {
		    is_parallel = std::stoi(value);
		}
		else if (name == "do_resume") {
		    do_resume = std::stoi(value);
		}
		else if (name == "wall_resume") {
		    wall_resume = std::stoi(value);
		}
		else if (name == "resume_ind") {
		    resume_ind = std::stoi(value);
		}
		else if (name == "save_file") {
		    save_file = std::stoi(value);
		}
		else if (name == "enable_fracture") {
		    enable_fracture = std::stoi(value);
		}
		else if (name == "nl_bdry_only") {
		    nl_bdry_only = std::stoi(value);
		}

		else if (name == "allow_damping") {
		    allow_damping = std::stoi(value);
		}
		else if (name == "allow_friction") {
		    allow_friction = std::stoi(value);
		}
		else if (name == "damping_ratio") {
		    damping_ratio = std::stod(value);
		}
		else if (name == "friction_coefficient") {
		    friction_coefficient = std::stod(value);
		}
		else if (name == "normal_stiffness") {
		    normal_stiffness = std::stod(value);
		}

		else if (name == "self_contact") {
		    self_contact = std::stoi(value);
		}
		else if (name == "self_contact_rad") {
		    self_contact_rad = std::stod(value);
		}

		else if (name == "enable_torque") {
		    enable_torque = std::stoi(value);
		}

		else if (name == "wall_left") {
		    wall_left = std::stod(value);
		}
		else if (name == "wall_right") {
		    wall_right = std::stod(value);
		}
		else if (name == "wall_top") {
		    wall_top = std::stod(value);
		}
		else if (name == "wall_bottom") {
		    wall_bottom = std::stod(value);
		}
		else if (name == "speed_wall_left") {
		    speed_wall_left = std::stod(value);
		}
		else if (name == "speed_wall_right") {
		    speed_wall_right = std::stod(value);
		}
		else if (name == "speed_wall_top") {
		    speed_wall_top = std::stod(value);
		}
		else if (name == "speed_wall_bottom") {
		    speed_wall_bottom = std::stod(value);
		}

		else if (name == "wall_x_min") {
		    wall_x_min = std::stod(value);
		}
		else if (name == "wall_y_min") {
		    wall_y_min = std::stod(value);
		}
		else if (name == "wall_z_min") {
		    wall_z_min = std::stod(value);
		}
		else if (name == "wall_x_max") {
		    wall_x_max = std::stod(value);
		}
		else if (name == "wall_y_max") {
		    wall_y_max = std::stod(value);
		}
		else if (name == "wall_z_max") {
		    wall_z_max = std::stod(value);
		}
		else if (name == "speed_wall_x_min") {
		    speed_wall_x_min = std::stod(value);
		}
		else if (name == "speed_wall_y_min") {
		    speed_wall_y_min = std::stod(value);
		}
		else if (name == "speed_wall_z_min") {
		    speed_wall_z_min = std::stod(value);
		}
		else if (name == "speed_wall_x_max") {
		    speed_wall_x_max = std::stod(value);
		}
		else if (name == "speed_wall_y_max") {
		    speed_wall_y_max = std::stod(value);
		}
		else if (name == "speed_wall_z_max") {
		    speed_wall_z_max = std::stod(value);
		}

		else if (name == "gradient_extforce") {
		    gradient_extforce = std::stoi(value);
		}
		else if (name == "extforce_maxstep") {
		    extforce_maxstep = std::stoi(value);
		}


		else{
		    std::cerr << "Error: Wrong config name: " << name << " !!\n";
		}
	    }
	    
	}
	else {
	    std::cerr << "Couldn't open config file for reading.\n";
	}
    };

    void print(){
	std::cout << "timesteps: " << timesteps << std::endl;
	std::cout << "modulo: "  << modulo << std::endl;
	std::cout << "dt: " << dt << std::endl;
	std::cout << "is_parallel: " << is_parallel << std::endl;
	std::cout << "do_resume: " << do_resume << std::endl;
	std::cout << "wall_resume: " << wall_resume << std::endl;
	std::cout << "resume_ind: " << resume_ind << std::endl;
	std::cout << "save_file: " << save_file << std::endl;
	std::cout << "enable_fracture: " << enable_fracture << std::endl;
	std::cout << "nl_bdry_only: " << nl_bdry_only << std::endl;
	std::cout << "allow_damping: " << allow_damping << std::endl;
	std::cout << "allow_friction: " << allow_friction << std::endl;
	std::cout << "damping_ratio: " << damping_ratio << std::endl;
	std::cout << "friction_coefficient: " << friction_coefficient << std::endl;
	std::cout << "normal_stiffness: " << normal_stiffness << std::endl;

	//std::cout << "speed_wall_top: " << speed_wall_top << std::endl;
	std::cout << "----------------------------------------------------------------------" << std::endl;
    };

private:
    /* data */
};


#endif /* ifndef READ_CONF */
