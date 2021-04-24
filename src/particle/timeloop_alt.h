#ifndef TIMELOOP_H
#define TIMELOOP_H

#include "particle/particle2.h"
#include "particle/timeloop.h"
#include "compat/overloads.h"

#include "particle/contact.h"

#include<omp.h>

class Timeloop
{
public:
    bool do_resume = 0;
    bool save_file;

    unsigned resume_ind;
    double dt;
    unsigned timesteps, modulo;
    double amplification;

    bool enable_fracture;


//output_file_string = 'sandbucket'

//output_counter = 1;

    Timeloop (unsigned ts){
	timesteps = ts;

	do_resume = 0;
	resume_ind = 0;

	dt = 0.02/1e5;
	modulo = 100;

	//use_influence_function = 0;
	
	enable_fracture = 0;

	amplification= 1;

	save_file = 1;

    };

    Timeloop (unsigned ts, unsigned m): Timeloop(ts) {
	modulo = m;
    };	
    //virtual ~timeloop ();

private:
    /* data */
};

template <unsigned dim>
void run_timeloop(vector<ParticleN<dim>> & PArr, Timeloop TL, Contact CN, RectWall Wall){

    unsigned counter = 1;

    unsigned total_particles_univ = PArr.size();



    vector<Matrix<double, 1, 1>> plotinfo;
    Matrix<double, 1, 1> piv;
    piv << (TL.timesteps/TL.modulo); plotinfo.push_back(piv);
    piv << total_particles_univ; plotinfo.push_back(piv);

    piv << Wall.allow_wall; plotinfo.push_back(piv);
    piv << Wall.left; plotinfo.push_back(piv);
    piv << Wall.right; plotinfo.push_back(piv);
    piv << Wall.bottom; plotinfo.push_back(piv);
    piv << Wall.top; plotinfo.push_back(piv);
    write_rowvecs("output/csv/plotinfo.csv", plotinfo);


	// external storage
    	vector<vector<Matrix<double, 1, dim>>> f_tot;
	f_tot.resize(total_particles_univ);
	Matrix<double, 1, dim> zeromat = Matrix<double, 1, dim>::Zero();

    std::cout << "Starting time loop." << std::endl;
    for (unsigned t = 1; t <= TL.timesteps; ++t) {
	//std::cout << "t = " << t << std::endl;

	//std::cout << PArr[0].disp << std::endl;
	//std::cout << PArr[0].vel << std::endl;
	//std::cout << PArr[0].acc << std::endl;
	
	//cout << "begin" << endl;
	
    //std::cout << "Number of threads: " << omp_get_max_threads() << std::endl;



	// new pos
//#pragma omp parallel for 
	for (unsigned i = 0; i < total_particles_univ; ++i) {
		//std::cout << "Started " << omp_get_num_threads() << std::endl;
	    PArr[i].disp += TL.dt * PArr[i].vel + (TL.dt * TL.dt * 0.5) * PArr[i].acc;
	//std::cout << "Pos: " <<  PArr[0].pos[0] << std::endl;
	//std::cout << "disp: " <<  PArr[0].disp[0] << std::endl;
	    PArr[i].CurrPos = PArr[i].pos + PArr[i].disp;

	    // initialize by zero
	    unsigned i_nnodes = PArr[i].nnodes;
	    f_tot[i].resize(i_nnodes);
	    for (unsigned j = 0; j < i_nnodes; j++) {
		f_tot[i][j] = zeromat;
	    }

	    //std::cout << omp_get_max_threads() << std::endl;
	}
	//std::cout << "CurrPos: " << PArr[0].CurrPos << std::endl;


//#pragma omp parallel for num_threads(2)
//#pragma omp parallel for schedule(dynamic)
#pragma omp parallel for
	for (unsigned i = 0; i < total_particles_univ; ++i) {
		//std::cout << "Started " << omp_get_num_threads() << std::endl;
	    //std::cout << "computing" << std::endl;
	    //PArr[i].force = PArr[i].get_peridynamic_force();
	    f_tot[i] += PArr[i].get_peridynamic_force();
	    //std::cout << "assigned" << std::endl;
	    if (TL.enable_fracture) {
		PArr[i].remove_bonds(t);
	    }

	
	//}
//#pragma omp parallel for
	//for (unsigned i = 0; i < total_particles_univ; ++i) {
	// contact force
	    vector<double> C_lrtb = get_lrtb(PArr[i].CurrPos);
	    for (unsigned j = 0; j < total_particles_univ; j++) {
		if (j != i) {
		    vector<double> N_lrtb = get_lrtb(PArr[j].CurrPos);
		    // if extended bounding boxes intersect
		    if (intersects_box_fext(C_lrtb, N_lrtb, CN.contact_rad)) {
			//std::cout << "contact" << std::endl;
			//PArr[i].force += get_contact_force<dim>(PArr[i], PArr[j], CN);
			f_tot[i] += get_contact_force<dim>(PArr[i], PArr[j], CN);
		    }
		}
	    }
	//}
//#pragma omp parallel for
	//for (unsigned i = 0; i < total_particles_univ; ++i) {
	    // wall forces
	    if (Wall.allow_wall) {
	    vector<double> C_lrtb = get_lrtb(PArr[i].CurrPos);
		if (! within_interior_fext(C_lrtb, Wall.lrtb(), CN.contact_rad)) {
		    //std::cout << i << std::endl;
			//PArr[i].force += get_wall_contact_force<dim>(PArr[i], Wall, CN);
			f_tot[i] += get_wall_contact_force<dim>(PArr[i], Wall, CN);
		}
	    }
	}
//#pragma omp parallel for
	for (unsigned i = 0; i < total_particles_univ; ++i) {
	
	    // store the old acceleration before replacing by the new one
	    auto temp_acc = PArr[i].acc;
	    //PArr[i].acc = (1/PArr[i].rho) * (PArr[i].force + PArr[i].extforce);
	    PArr[i].acc = (1/PArr[i].rho) * (f_tot[i] + PArr[i].extforce);
	    //	// velocity
	    //	u0dot_univ{i} = uolddot_univ{i} + dt * 0.5 * uolddotdot_univ{i} + dt * 0.5 * u0dotdot_univ{i};
	    //PArr[i].vel = PArr[i].vel  + (dt * 0.5) * PArr[i].acc + (dt * 0.5) * PArr[i].acc_old;
	    temp_acc += PArr[i].acc;
	    temp_acc *= (0.5 * TL.dt);
	    PArr[i].vel += temp_acc;

	}
	//std::cout << PArr[0].disp << std::endl;
	//std::cout << PArr[0].pos << std::endl;
	//std::cout << PArr[0].extforce << std::endl;
	//std::cout << PArr[0].extforce << std::endl;
	//std::cout << "force: "  << PArr[0].force[0] << std::endl;
	//std::cout << PArr[0].acc << std::endl;
	//std::cout << PArr[0].CurrPos[0] << std::endl;
	
	// save
	if (TL.save_file) {
	    if (t == 1) {
		vector<int> v;
		v.push_back(total_particles_univ);
		// compute the correct number
		v.push_back(TL.timesteps % TL.modulo);
		v.push_back(TL.timesteps);
	        
		// save this to file
	    }

	    if ((t % TL.modulo) == 0) {
		std::cout << "t: " << t << " counter: " << counter << endl;

		char buf[25];
		sprintf(buf, "_%05u", counter);
		string tcounter = string(buf);

//#pragma omp parallel for
		// edit this for speed
		for (unsigned i = 0; i < total_particles_univ; i++) {
		    char buf2[25];
		    sprintf(buf2, "_particle_%05u", i);
		    string particlenum = string(buf2);

		    string filename = "output/csv/tcounter"+tcounter+particlenum+".csv";
		    //std::cout << filename << std::endl;


		    //write_rowvecs(filename, PArr[0].pos + TL.amplification * PArr[0].disp);
		    vector<Matrix<double, 1, dim+1>> V;
		    V.resize(PArr[i].nnodes);	

		    for (unsigned j = 0; j < PArr[i].nnodes; ++j) {
			//V[j] << PArr[i].CurrPos[j], PArr[i].force[j].norm();
			V[j] << PArr[i].CurrPos[j], f_tot[i][j].norm();
			//V[i] << PArr[0].CurrPos[i], PArr[0].disp[i].norm();
			//V[i] << PArr[0].CurrPos[i], PArr[0].disp[i](1);
		    }
		    write_rowvecs(filename, V);
		    //write_rowvecs(filename, PArr[0].CurrPos);
		}


		std::cout << counter << std::endl;

		++counter;
	    }
	}


    }

    //cout << "Normal Stiffness: " << CN.normal_stiffness << endl;

};


#endif /* ifndef TIMELOOP_H */
