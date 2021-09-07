#ifndef TIMELOOP_H
#define TIMELOOP_H

#include <iomanip>

#include <chrono>
using namespace std::chrono;

#include "read/read_config.h"

#include "compat/overloads.h"
#include "particle/particle2.h"
#include "particle/timeloop.h"

#include "particle/contact.h"

#include "read/rw_hdf5.h"

#include <omp.h>

#include <ctime>
#include <time.h>

// Convert NbdArr to connectivity
vector<Matrix<unsigned, 1, 2>> NbdArr2conn(vector<vector<unsigned>> NbdArr)
{
    vector<Matrix<unsigned, 1, 2>> Conn;

    //// get total number of bonds to allocate memory in advance
    //unsigned nbonds  = 0;
    //for (unsigned i = 0; i < NbdArr.size(); i++) {
	//nbonds += NbdArr[i].size();
    //}
    //Conn.resize(nbonds);

    for (unsigned i = 0; i < NbdArr.size(); i++) {
	for (unsigned j = 0; j < NbdArr[i].size(); j++) {
	    auto q = NbdArr[i][j];

	    // To avoid counting all the bonds twice, we only allow sub-diagonal pairs, this essentially gets rid of the copy with reverse order
	    if (i < q) {
              Matrix<unsigned, 1, 2> v = {i, q};
              //// append
              Conn.push_back(v);
            }
	}
    }
    return Conn;
};

template <unsigned dim> auto mean(vector<Matrix<double, 1, dim>> v) {
  auto nnodes = v.size();
  Matrix<double, 1, dim> mean;
  mean.setZero();
  for (unsigned inx = 0; inx < nnodes; inx++) {
    mean += v[inx];
  }
  mean /= nnodes;
  return mean;
};

class Timeloop {
public:
  bool do_resume = 0;
  bool wall_resume = 0;
  bool save_file;

  unsigned resume_ind;
  double dt;
  unsigned timesteps, modulo;

  bool enable_fracture;

  bool run_parallel;

  unsigned counter;
  unsigned first_counter, last_counter;

  bool gradient_extforce = 0;
  bool enable_torque = 0;
  unsigned extforce_maxstep = 0;

  int set_movable_index = -1;
  int set_movable_timestep = -1;
  int set_stoppable_index = -1;
  int set_stoppable_timestep = -1;


  bool reset_partzero_y = 0;
  unsigned reset_partzero_y_timestep;
  double wheel_rad;

  // saving the runtime
  // double start_time;
  std::chrono::_V2::system_clock::time_point start_time;

  vector<double> run_time, t_ind;

  Timeloop(unsigned ts) {
    timesteps = ts;

    do_resume = 0;
    resume_ind = 0;

    dt = 0.02 / 1e5;
    modulo = 100;

    // use_influence_function = 0;

    enable_fracture = 0;

    save_file = 1;

    run_parallel = 1;

    start_time = system_clock::now();
  };

  Timeloop(unsigned ts, unsigned m) : Timeloop(ts) { modulo = m; };

  template <unsigned dim>
  void update_on_resume(vector<ParticleN<dim>> &PArr, RectWall<dim> &Wall) {
    if (do_resume) {
      counter = resume_ind + 1;
      first_counter = counter;
      last_counter = resume_ind + timesteps / modulo;

      string data_loc = "output/hdf5/";
      char buf[20];
      sprintf(buf, "tc_%05u.h5", resume_ind);
      string h5file = string(buf);
      string filename = data_loc + h5file;
      std::cout << "Loading from file: " << filename << std::endl;

      for (unsigned i = 0; i < PArr.size(); ++i) {
        char buf2[100];
        sprintf(buf2, "P_%05u", i);
        string particlenum = string(buf2);
        PArr[i].disp =
            load_rowvecs<double, dim>(filename, particlenum + "/CurrPos") -
            PArr[i].pos;
        PArr[i].vel = load_rowvecs<double, dim>(filename, particlenum + "/vel");
        PArr[i].acc = load_rowvecs<double, dim>(filename, particlenum + "/acc");
      }

      if (wall_resume) {
        // resume wall info
        char buf_w[20];
        sprintf(buf_w, "wall_%05u.h5", resume_ind);
        string wall_file = string(buf_w);
        string filename_wall = data_loc + wall_file;
        std::cout << "Loading wall from file: " << filename_wall << std::endl;
        auto vv = load_col<double>(filename_wall, "wall_info");
        Wall.set_lrtb(vv);

        // std::cout << "wall info now: " << Wall.lrtb() << std::endl;
      }
      std::cout << "Resuming from counter " << counter << std::endl;
    } else {
      counter = 1;
      first_counter = counter;
      last_counter = timesteps / modulo;
    }
  };

  void save_runtime() {
    string runtime_filename = "output/hdf5/run_time.h5";
    H5::H5File rt_fp(runtime_filename, H5F_ACC_TRUNC);
    store_col<double>(rt_fp, "run_time", run_time);
    store_col<double>(rt_fp, "t_ind", t_ind);
    rt_fp.close();
  };

  void apply_config(ConfigVal CFGV) {
    // update from config values
    timesteps = CFGV.timesteps;
    modulo = CFGV.modulo;
    dt = CFGV.dt;
    do_resume = CFGV.do_resume;
    wall_resume = CFGV.wall_resume;
    resume_ind = CFGV.resume_ind;
    save_file = CFGV.save_file;
    enable_fracture = CFGV.enable_fracture;
    run_parallel = CFGV.is_parallel;

    gradient_extforce = CFGV.gradient_extforce;
    extforce_maxstep = CFGV.extforce_maxstep;
    // self-contact
    enable_torque = CFGV.enable_torque;
    // turn on movable
    set_movable_index = CFGV.set_movable_index;
    set_movable_timestep = CFGV.set_movable_timestep;
    // turn on stoppable
    set_stoppable_index = CFGV.set_stoppable_index;
    set_stoppable_timestep = CFGV.set_stoppable_timestep;

    // move wheel to top of bulk
    reset_partzero_y = CFGV.reset_partzero_y;
    reset_partzero_y_timestep = CFGV.reset_partzero_y_timestep;
    wheel_rad = CFGV.wheel_rad;

  };

private:
  /* data */
};

template <unsigned dim> void save_plotinfo(Timeloop TL, RectWall<dim> Wall) {

  string plt_filename = "output/hdf5/plotinfo.h5";
  H5::H5File pl_fp(plt_filename, H5F_ACC_TRUNC);

  vector<int> al_wall = {Wall.allow_wall};
  vector<double> geom_wall_info = Wall.lrtb();
  vector<unsigned> f_l_counter = {TL.first_counter, TL.last_counter};
  // std::cout << "geom_wall_info" << geom_wall_info << std::endl;
  store_col<unsigned>(pl_fp, "f_l_counter", f_l_counter);

  vector<unsigned> vec_dim = {dim};
  store_col<unsigned>(pl_fp, "dimension", vec_dim);

  vector<double> vec_dt = {TL.dt};
  store_col<double>(pl_fp, "dt", vec_dt);

  vector<unsigned> vec_modulo = {TL.modulo};
  store_col<unsigned>(pl_fp, "modulo", vec_modulo);

  pl_fp.createGroup("/wall");
  store_col<int>(pl_fp, "/wall/allow_wall", al_wall);
  store_col<double>(pl_fp, "/wall/geom_wall_info", geom_wall_info);
  pl_fp.close();
};

template <unsigned dim>
void save_timestep_to_file(unsigned t, Timeloop &TL,
                           vector<ParticleN<dim>> PArr, RectWall<dim> Wall) {
  if (TL.save_file) {
    if ((t % TL.modulo) == 0) {
      char buf[100];
      sprintf(buf, "%05u", TL.counter);
      string tcounter = string(buf);

      string output_loc = "output/hdf5/";
      string filename = output_loc + "tc_" + tcounter + ".h5";
      H5::H5File fp(filename, H5F_ACC_TRUNC);

// pragma works here [4]
#pragma omp parallel for if (TL.run_parallel)
      for (unsigned i = 0; i < PArr.size(); i++) {
        char buf2[100];
        sprintf(buf2, "P_%05u", i);
        string particlenum = string(buf2);
        fp.createGroup(particlenum);
        store_rowvec<double, dim>(fp, particlenum + "/CurrPos",
                                  PArr[i].CurrPos);
        store_rowvec<double, dim>(fp, particlenum + "/vel", PArr[i].vel);
        store_rowvec<double, dim>(fp, particlenum + "/acc", PArr[i].acc);
        // This is actually the force density (Force/Volume), NOT the Force
        store_rowvec<double, dim>(fp, particlenum + "/force", PArr[i].force);
        // We will output the Force (Newton), not the force density
        // (Newton/vol)
        // store_rowvec<double, dim>(fp, particlenum + "/force",
        // PArr[i].force*PArr[i].vol);

        // debug, works
        // store_col<double>(fp, particlenum+"/vol", PArr[i].vol);
      }
      fp.close();

      // save wall boundary info
      // if (Wall.move_bdry) {
      // if the wall moves, otherwise a waste of space
      // Save this file always
      char buf_wall[100];
      sprintf(buf_wall, "%05u", TL.counter);
      string tcounter_wall = string(buf);

      string filename_wall = output_loc + "wall_" + tcounter + ".h5";
      H5::H5File fp_w(filename_wall, H5F_ACC_TRUNC);

      store_col<double>(fp_w, "wall_info", Wall.lrtb());
      store_rowvec<double, dim>(fp_w, "reaction", Wall.get_reaction());

      fp.close();
      //}

      /**********************************************************************
       * Done saving files
       */

      time_t my_time = time(NULL);
      // std::cout << "t: " << t << " TL.counter: " << TL.counter << " "  <<
      // ctime(&my_time); std::cout << "--------------------------" <<
      // std::endl;

      auto stop_time = system_clock::now();
      // are nanoseconds, microseconds, milliseconds, seconds, minutes, hours
      // auto duration = duration_cast<seconds>(stop_time - start_time);
      auto duration = duration_cast<milliseconds>(stop_time - TL.start_time);
      // update for next loop
      TL.start_time = stop_time;
      // To get the value of duration use the count(), member function on the
      // duration object
      // cout << "Runtime: " << duration.count() << "s" << endl;
      // std::cout << "Duration: " << duration.count() << std::endl;

      // prediction: from last n avg
      int avgsteps = 10;
      double avg_duration = 0;
      if (TL.run_time.size() >= 10) {
        for (unsigned cc = (TL.run_time.size() - avgsteps);
             cc < TL.run_time.size(); cc++) {
          avg_duration += TL.run_time[cc];
        }
      }
      avg_duration /= avgsteps;
      int rem_counts = TL.last_counter - TL.counter;
      // in mins
      double rem_duration = (rem_counts * avg_duration) / 1e3 / 60;

      std::cout << "t: " << t << " TL.counter: " << TL.counter << " duration: "
                << duration.count()
                //<< " rem counts: " << rem_counts
                //<< " avg duration: " << avg_duration
                << " rem: " << rem_duration << "m " << ctime(&my_time);
      TL.run_time.push_back(duration.count());
      TL.t_ind.push_back(t);

      // update
      ++TL.counter;
    }
  }
};

template <unsigned dim>
void run_timeloop_compact(vector<ParticleN<dim>> &PArr, Timeloop TL, Contact CN,
                          RectWall<dim> Wall) {

  unsigned total_particles_univ = PArr.size();

  // if resuming, update initial data and wall
  TL.update_on_resume<dim>(PArr, Wall);
  // Save plotinfo
  save_plotinfo<dim>(TL, Wall);

  time_t my_time = time(NULL);
  std::cout << "---------------------------------------------------------------"
            << std::endl;
  std::cout << "Starting time loop "
            << "(Parallel=" << TL.run_parallel << "): " << ctime(&my_time);
  TL.start_time = system_clock::now();

  std::cout << "t: "
                  << "\t"
                  << " counter "
                  << "\t"
                  << " duration (s) "
                  << "\t"
                  << " rem (m) ";

  for (unsigned t = 1; t <= TL.timesteps; ++t) {
    // set wall reaction to zero
    Wall.setzero_reaction();

    // new position update
// pragma here works [1]
#pragma omp parallel for if (TL.run_parallel)
    for (unsigned i = 0; i < total_particles_univ; ++i) {
      PArr[i].disp += TL.dt * PArr[i].vel + (TL.dt * TL.dt * 0.5) * PArr[i].acc;
      PArr[i].CurrPos = PArr[i].pos + PArr[i].disp;
    }

    /**********************************************************************
    force computation
    **********************************************************************/
//#pragma omp parallel for num_threads(2)
// pragma here works [2]
//#pragma omp parallel for
#pragma omp parallel for schedule(dynamic) if (TL.run_parallel)
    for (unsigned i = 0; i < total_particles_univ; ++i) {
      // PArr[i].force = PArr[i].get_peridynamic_force();

      // define ft_i and initialize by zero
      vector<Matrix<double, 1, dim>> ft_i;
      ft_i.resize(PArr[i].nnodes);
      for (unsigned ll = 0; ll < ft_i.size(); ll++) {
        ft_i[ll] = Matrix<double, 1, dim>::Zero();
      }

      // compute total internal force only if it is movable
      if (PArr[i].movable) {

        // Peridynamic forces
        ft_i = PArr[i].get_peridynamic_force();
        // auto ft_i = PArr[i].get_peridynamic_force();

        // Fracture update
        if (TL.enable_fracture) {
          if (PArr[i].breakable) {
            // PArr[i].remove_bonds(t);
            PArr[i].remove_bonds();
            // self-contact: update ft_i, 1==is_self_contact
            if (CN.self_contact) {
              update_contact_force_by_boundary<dim>(PArr[i], PArr[i], CN, TL.dt,
                                                    ft_i, 1);
            }
          }
        }

        // contact forces
        vector<double> C_lrtb = get_minmax<dim>(PArr[i].CurrPos);
        for (unsigned j = 0; j < total_particles_univ; j++) {
          if (j != i) {
            vector<double> N_lrtb = get_minmax<dim>(PArr[j].CurrPos);
            // if extended bounding boxes intersect
            if (intersects_box_fext<dim>(C_lrtb, N_lrtb, CN.contact_rad)) {
              update_contact_force_by_boundary<dim>(PArr[i], PArr[j], CN, TL.dt,
                                                    ft_i, 0);
            }
          }
        }

        // wall forces
        if (Wall.allow_wall) {
          if (!within_interior_fext(C_lrtb, Wall.lrtb(), CN.contact_rad)) {
            update_wall_contact_force_by_boundary<dim>(PArr[i], Wall, CN, TL.dt,
                                                       ft_i);
          }
        }

        // external forces
        // std::cout << TL.gradient_extforce << std::endl;
        if (TL.gradient_extforce) {
          double extf_sc;
          extf_sc = (double)t / (double)TL.extforce_maxstep;

          if (extf_sc > 1) {
            extf_sc = 1;
          }
          std::cout << extf_sc << std::endl;
          ft_i += extf_sc * PArr[i].extforce;

        } else {
          ft_i += PArr[i].extforce;
        }
      }
      // Debug: moved it into the loop
      // store the computed value into the particle force
      PArr[i].force = ft_i;
    }

// Cannot merge this for loop with the previous one since in the previous loop
// we are using the value of velocity whereas in this loop, we are modifying the
// velocity in an arbitrary order. So, there could be a scenario where we read
// and write the velocity of a certain particle simultaneously, causing
// std::bad_alloc error. pragma here works [3]
// Particle state update for the next timestep
#pragma omp parallel for if (TL.run_parallel)
    for (unsigned i = 0; i < total_particles_univ; ++i) {
      if (PArr[i].movable) {
        // store the old acceleration before replacing by the new one
        auto temp_acc = PArr[i].acc;
        // New acceleration
        PArr[i].acc = (1 / PArr[i].rho) * (PArr[i].force);
        //	// velocity
        //	u0dot_univ{i} = uolddot_univ{i} + dt * 0.5 * uolddotdot_univ{i}
        //+ dt * 0.5 * u0dotdot_univ{i}; PArr[i].vel = PArr[i].vel  + (dt * 0.5)
        //*
        // PArr[i].acc + (dt * 0.5) * PArr[i].acc_old;
        temp_acc += PArr[i].acc;   // Now temp_acc = (acc + acc_old)
        temp_acc *= (0.5 * TL.dt); // now temp_acc = (dt*0.5) *(acc + acc_old)
        PArr[i].vel += temp_acc;

        // clamped nodes are set to zero
        if (PArr[i].clamped_nodes.size()) {
          for (unsigned cn = 0; cn < PArr[i].clamped_nodes.size(); cn++) {
            auto cid = PArr[i].clamped_nodes[cn];
            // setting disp=0 is unnecessary if vel, acc = 0 are already set
            // But while saving, we want disp=0 even when vel>0 applied to a
            // clamped node, so we keep it
            PArr[i].disp[cid] = Matrix<double, 1, dim>::Zero();
            PArr[i].vel[cid] = Matrix<double, 1, dim>::Zero();
            PArr[i].acc[cid] = Matrix<double, 1, dim>::Zero();
          }
        }
      }
    }

    // wall boundary update
    Wall.update_boundary(TL.dt);

    // save particle state to disk
    save_timestep_to_file(t, TL, PArr, Wall);
  }

  // save runtime info to file
  TL.save_runtime();
};

template <unsigned dim>
void run_timeloop(vector<ParticleN<dim>> &PArr, Timeloop TL, Contact CN,
                  RectWall<dim> Wall) {

  unsigned total_particles_univ = PArr.size();
  unsigned counter;
  unsigned first_counter, last_counter;

  if (TL.do_resume) {
    counter = TL.resume_ind + 1;
    first_counter = counter;
    last_counter = TL.resume_ind + TL.timesteps / TL.modulo;

    string data_loc = "output/hdf5/";
    char buf[20];
    sprintf(buf, "tc_%05u.h5", TL.resume_ind);
    string h5file = string(buf);
    string filename = data_loc + h5file;
    std::cout << "Loading from file: " << filename << std::endl;

    for (unsigned i = 0; i < total_particles_univ; ++i) {
      char buf2[100];
      sprintf(buf2, "P_%05u", i);
      string particlenum = string(buf2);
      PArr[i].disp =
          load_rowvecs<double, dim>(filename, particlenum + "/CurrPos") -
          PArr[i].pos;
      PArr[i].vel = load_rowvecs<double, dim>(filename, particlenum + "/vel");
      PArr[i].acc = load_rowvecs<double, dim>(filename, particlenum + "/acc");

      if (TL.enable_fracture) {
        // load connectivity
        auto Conn =
            load_rowvecs<unsigned, 2>(filename, particlenum + "/Connectivity");
        PArr[i].NbdArr = conn2NArr(Conn, PArr[i].nnodes);
        PArr[i].gen_xi();
      }
    }

    if (TL.wall_resume) {
      // resume wall info
      char buf_w[20];
      sprintf(buf_w, "wall_%05u.h5", TL.resume_ind);
      string wall_file = string(buf_w);
      string filename_wall = data_loc + wall_file;
      std::cout << "Loading wall from file: " << filename_wall << std::endl;
      auto vv = load_col<double>(filename_wall, "wall_info");
      Wall.set_lrtb(vv);

      // std::cout << "wall info now: " << Wall.lrtb() << std::endl;
    }

    std::cout << "Resuming from counter " << counter << std::endl;
  } else {
    counter = 1;
    first_counter = counter;
    last_counter = TL.timesteps / TL.modulo;
  }


  // Save plotinfo
  string plt_filename = "output/hdf5/plotinfo.h5";
  H5::H5File pl_fp(plt_filename, H5F_ACC_TRUNC);

  vector<int> al_wall = {Wall.allow_wall};
  vector<double> geom_wall_info = Wall.lrtb();
  vector<unsigned> f_l_counter = {first_counter, last_counter};
  // std::cout << "geom_wall_info" << geom_wall_info << std::endl;
  store_col<unsigned>(pl_fp, "f_l_counter", f_l_counter);

  vector<unsigned> vec_dim = {dim};
  store_col<unsigned>(pl_fp, "dimension", vec_dim);

  vector<double> vec_dt = {TL.dt};
  store_col<double>(pl_fp, "dt", vec_dt);

  vector<unsigned> vec_modulo = {TL.modulo};
  store_col<unsigned>(pl_fp, "modulo", vec_modulo);

  pl_fp.createGroup("/wall");
  store_col<int>(pl_fp, "/wall/allow_wall", al_wall);
  store_col<double>(pl_fp, "/wall/geom_wall_info", geom_wall_info);
  pl_fp.close();

  time_t my_time = time(NULL);

  std::cout << "---------------------------------------------------------------"
               "-------"
            << std::endl;
  std::cout << "Starting time loop "
            << "(Parallel=" << TL.run_parallel << "): " << ctime(&my_time);

  unsigned column_w = 12;

  char str_count[50];
  sprintf(str_count,"count[%d:%d]", first_counter, last_counter);

  std::cout 
      << left << std::setw(column_w) << "t "
      << left << std::setw(column_w) << string(str_count)
      << left << std::setw(column_w) << "duration(s)"
      << left << std::setw(column_w+5) << "rem(h:m:s.ms)"
      << left << std::setw(column_w) << "date" 
      << std::endl;

  auto start_time = system_clock::now();
  vector<double> run_time, t_ind;

  for (unsigned t = 1; t <= TL.timesteps; ++t) {
    // set wall reaction to zero
    Wall.setzero_reaction();

    if (TL.set_movable_index != (-1)) {
	auto part_ind = (unsigned) TL.set_movable_index;
	auto part_ts = (unsigned) TL.set_movable_timestep;
	if (part_ts == t) {
	    std::cout << "Setting particle " << TL.set_movable_index << " to movable on timestep " << t << std::endl;
	    PArr[part_ind].movable = 1;
	}
    }
    if (TL.set_stoppable_index != (-1)) {
	auto part_ind = (unsigned) TL.set_stoppable_index;
	auto part_ts = (unsigned) TL.set_stoppable_timestep;
	if (part_ts == t) {
	    std::cout << "Setting particle " << TL.set_stoppable_index << " to stoppable on timestep " << t << std::endl;
	    PArr[part_ind].stoppable = 1;
	}
    }

    if (TL.reset_partzero_y) {
	auto part_ts = (unsigned) TL.reset_partzero_y_timestep;
	if (part_ts == t) {
	    // find max_y of the bulk: i=1,...
	    double max_bulk_y = PArr[1].pos[0](1) + PArr[1].disp[1](1);
	    for (unsigned i = 1; i < total_particles_univ; ++i) {
		for (unsigned j = 0; j < PArr[i].nnodes; j++) {
		    double now_y = PArr[i].pos[j](1) + PArr[i].disp[j](1); 
		      if (now_y > max_bulk_y){
			max_bulk_y = now_y;
		      }
		}
	    }
	    // move the mean by this amount 
	    double dest = max_bulk_y + TL.wheel_rad + CN.contact_rad;
	    double to_move_by = dest - PArr[0].mean_CurrPos()(1);
	    std::cout << "Setting particle zero mean y val to  " << dest << std::endl;
	    for (unsigned j = 0; j < PArr[0].nnodes; j++) {
		PArr[0].pos[j](1) += to_move_by;
	    }
	}
    }





    // std::cout << "t = " << t << std::endl;
    // std::cout << t << " ";

    // new pos
// pragma here works [1]
#pragma omp parallel for if (TL.run_parallel)
    for (unsigned i = 0; i < total_particles_univ; ++i) {
      if (PArr[i].movable) {
	  PArr[i].disp += TL.dt * PArr[i].vel + (TL.dt * TL.dt * 0.5) * PArr[i].acc;
      }
      PArr[i].CurrPos = PArr[i].pos + PArr[i].disp;

      // std::cout << PArr[i].mean_CurrPos() << std::endl;
    }

    //unsigned iind = 20;
    //std::cout << t  << " "
	//<< " pos " << PArr[0].pos[iind] 
	//<< " CurrPos " << PArr[0].CurrPos[iind] 
	//<< " u " << PArr[0].disp[iind] 
	//<< " v " << PArr[0].vel[iind] 
	//<< " a " << PArr[0].acc[iind] 
	//<< " f " << PArr[0].force[iind] << PArr[0].force[iind] << std::endl;

/**********************************************************************
contact force computation
**********************************************************************/
//#pragma omp parallel for num_threads(2)
// pragma here works [2]
#pragma omp parallel for schedule(dynamic) if (TL.run_parallel)
    //#pragma omp parallel for
    for (unsigned i = 0; i < total_particles_univ; ++i) {
      // PArr[i].force = PArr[i].get_peridynamic_force();

      // define ft_i and initialize by zero
      vector<Matrix<double, 1, dim>> ft_i;
      ft_i.resize(PArr[i].nnodes);
      for (unsigned ll = 0; ll < ft_i.size(); ll++) {
        ft_i[ll] = Matrix<double, 1, dim>::Zero();
      }

      // compute total internal force only if it is movable
      if (PArr[i].movable) {

        // auto ft_i = PArr[i].get_peridynamic_force();
        ft_i = PArr[i].get_peridynamic_force();

        if (TL.enable_fracture) {
          if (PArr[i].breakable) {
            // PArr[i].remove_bonds(t);
            PArr[i].remove_bonds();

            // self-contact: update ft_i, 1==is_self_contact
            if (CN.self_contact) {
              update_contact_force_by_boundary<dim>(PArr[i], PArr[i], CN, TL.dt,
                                                    ft_i, 1);
            }
          }
        }

        //}
        ////#pragma omp parallel for
        // for (unsigned i = 0; i < total_particles_univ; ++i) {
        // contact forces from other particles
	// Only compute contact force if stoppable
	if (PArr[i].stoppable) {

          vector<double> C_lrtb = get_minmax<dim>(PArr[i].CurrPos);
          //#pragma omp parallel for
          for (unsigned j = 0; j < total_particles_univ; j++) {
            if (j != i) {
		//std::cout << "yes" << std::endl;
              vector<double> N_lrtb = get_minmax<dim>(PArr[j].CurrPos);
              // if extended bounding boxes intersect
              if (intersects_box_fext<dim>(C_lrtb, N_lrtb, CN.contact_rad)) {
                // std::cout << "contact" << std::endl;
                // Debug
                // PArr[i].force += get_contact_force<dim>(PArr[i], PArr[j],
                // CN); ft_i += get_contact_force<dim>(PArr[i], PArr[j], CN);
                // ft_i += get_contact_force_by_boundary<dim>(PArr[i], PArr[j],
                // CN); std::cout << "central particle i = " << i << std::endl;
                // std::cout << "Here nbd particle j = " << j  << std::endl;

                //	Debug
                //	// latest
                update_contact_force_by_boundary<dim>(PArr[i], PArr[j], CN,
                                                      TL.dt, ft_i, 0);
                // bare bone implementation
                // simple_update_contact_force<dim>(PArr[i], PArr[j], CN, ft_i);
              }
            }
          }

          //}
          //#pragma omp parallel for
          // for (unsigned i = 0; i < total_particles_univ; ++i) {
          // wall forces
          if (Wall.allow_wall) {
            // vector<double> C_lrtb = get_minmax<dim>(PArr[i].CurrPos);
            //
            if (!within_interior_fext(C_lrtb, Wall.lrtb(), CN.contact_rad)) {
              // std::cout << "wall_lrtb: " << Wall.lrtb() << std::endl;
              // std::cout << i << std::endl;
              // Debug
              // PArr[i].force += get_wall_contact_force<dim>(PArr[i], Wall,
              // CN); ft_i += get_wall_contact_force<dim>(PArr[i], Wall, CN);
              // update_wall_contact_force<dim>(PArr[i], Wall, CN, ft_i);
              // ft_i += get_wall_contact_force_by_boundary<dim>(PArr[i], Wall,
              // CN); std::cout << "Adding 2d Wall force" << std::endl;
              update_wall_contact_force_by_boundary<dim>(PArr[i], Wall, CN,
                                                         TL.dt, ft_i);
              // std::cout << Wall.get_reaction() << std::endl;
            }
          }
        }

        // external forces
        // std::cout << TL.gradient_extforce << std::endl;
        if (TL.gradient_extforce) {
          double extf_sc;
          extf_sc = (double)t / (double)TL.extforce_maxstep;

          if (extf_sc > 1) {
            extf_sc = 1;
          }
          // std::cout << extf_sc << " " ;
          ft_i += extf_sc * PArr[i].extforce;

        } else {
          ft_i += PArr[i].extforce;
        }


	// external torque about the centroid
	if (TL.enable_torque) {
	    // centroid about which the torque is applied
	    Matrix<double, 1, dim> c = PArr[i].mean_CurrPos();
	    unsigned taxis = PArr[i].torque_axis;

	    for (unsigned nn = 0; nn < PArr[i].CurrPos.size(); nn++) {
		Matrix<double, 1, dim> r_vec_proj = PArr[i].CurrPos[nn] - c;
		// project on the plane perpendicular to torque_axis
		// In 2d, always set torque_axis=2 (i.e, z-axis)
		if (dim==3) {
		    r_vec_proj(taxis) = 0;
		}
		// perpendicular to r vector -> r_perp
		Matrix<double, 1, dim> r_perp;
		if (dim==3) {
		    r_perp(taxis) = 0;
		}
		r_perp( (taxis+1)%3 ) = - r_vec_proj( (taxis+2)%3 );
		r_perp( (taxis+2)%3 ) = r_vec_proj( (taxis+1)%3 );

		// unit perpendicular direction to r vector -> r_perp
		auto perp_norm = r_perp.norm();
		if (perp_norm > 0) {
		    r_perp /= perp_norm;
		}
		else
		{
		    r_perp( (taxis+1)%3 ) = 0;
		    r_perp( (taxis+2)%3 ) = 0;
		}

		// add force density due to torque to the total force
		ft_i[nn] += (PArr[i].torque_val * r_perp);
	    }
	}

      }


      // store the computed value into the particle force
      PArr[i].force = ft_i;
    }

// pragma here works [3]
#pragma omp parallel for if (TL.run_parallel)
    // update quantities
    for (unsigned i = 0; i < total_particles_univ; ++i) {
      // update if movable
      if (PArr[i].movable) {
        // Element in vector.
        // store the old acceleration before replacing by the new one
        auto temp_acc = PArr[i].acc;
        // New acceleration
        PArr[i].acc = (1 / PArr[i].rho) * (PArr[i].force);
        //	// velocity
        //	u0dot_univ{i} = uolddot_univ{i} + dt * 0.5 * uolddotdot_univ{i}
        //+ dt * 0.5 * u0dotdot_univ{i}; PArr[i].vel = PArr[i].vel  + (dt * 0.5)
        //*
        // PArr[i].acc + (dt * 0.5) * PArr[i].acc_old;
        temp_acc += PArr[i].acc;   // Now temp_acc = (acc + acc_old)
        temp_acc *= (0.5 * TL.dt); // now temp_acc = (dt*0.5) *(acc + acc_old)
        PArr[i].vel += temp_acc;

        // clamped nodes are set to be zero
        if (PArr[i].clamped_nodes.size()) {
          for (unsigned cn = 0; cn < PArr[i].clamped_nodes.size(); cn++) {
            auto cid = PArr[i].clamped_nodes[cn];
            PArr[i].disp[cid] = Matrix<double, 1, dim>::Zero();
            PArr[i].vel[cid] = Matrix<double, 1, dim>::Zero();
            PArr[i].acc[cid] = Matrix<double, 1, dim>::Zero();
          }
        }
      }
    }

    // wall boundary update
    if (dim == 2) {
      Wall.left += Wall.speed_left * TL.dt;
      Wall.right += Wall.speed_right * TL.dt;
      Wall.top += Wall.speed_top * TL.dt;
      Wall.bottom += Wall.speed_bottom * TL.dt;
    } else {
      Wall.x_min += Wall.speed_x_min * TL.dt;
      Wall.y_min += Wall.speed_y_min * TL.dt;
      Wall.z_min += Wall.speed_z_min * TL.dt;
      Wall.x_max += Wall.speed_x_max * TL.dt;
      Wall.y_max += Wall.speed_y_max * TL.dt;
      Wall.z_max += Wall.speed_z_max * TL.dt;
    }

    // save
    if (TL.save_file) {

      if ((t % TL.modulo) == 0) {

        /***********************************************************************/
        // print
        // std::cout << PArr[1].mean_CurrPos() << std::endl;

        // auto P = PArr[1];
        // auto pred_val = P.disp + TL.dt * P.vel + (TL.dt * TL.dt * 0.5) *
        // P.acc;

        ////std::cout << "pos " << mean0 << std::endl;
        // std::cout << "disp: " << mean<dim>(P.disp) << std::endl;
        // std::cout << "vel: " << mean<dim>(P.vel) << std::endl;
        // std::cout << "acc: " << mean<dim>(P.acc) << std::endl;
        // std::cout << "extforce: " << mean<dim>(P.extforce) << std::endl;
        // std::cout << "force: " << mean<dim>(P.force) << std::endl;
        // std::cout << "predicted next disp: " << mean<dim>(pred_val) <<
        // std::endl;
        /**********************************************************************/

        char buf[100];
        sprintf(buf, "%05u", counter);
        string tcounter = string(buf);

        string output_loc = "output/hdf5/";
        string filename = output_loc + "tc_" + tcounter + ".h5";
        H5::H5File fp(filename, H5F_ACC_TRUNC);

// pragma works here [4]
#pragma omp parallel for if (TL.run_parallel)
        for (unsigned i = 0; i < total_particles_univ; i++) {
          char buf2[100];
          sprintf(buf2, "P_%05u", i);
          string particlenum = string(buf2);
          fp.createGroup(particlenum);
          store_rowvec<double, dim>(fp, particlenum + "/CurrPos",
                                    PArr[i].CurrPos);
          store_rowvec<double, dim>(fp, particlenum + "/vel", PArr[i].vel);
          store_rowvec<double, dim>(fp, particlenum + "/acc", PArr[i].acc);
          // This is actually the force density (Force/Volume), NOT the Force
          store_rowvec<double, dim>(fp, particlenum + "/force", PArr[i].force);
          // We will output the Force (Newton), not the force density
          // (Newton/vol)
          // store_rowvec<double, dim>(fp, particlenum + "/force",
          // PArr[i].force*PArr[i].vol);

	  // save connectivity, converting NbdArr
          store_rowvec<unsigned, 2>(fp, particlenum + "/Connectivity", NbdArr2conn(PArr[i].NbdArr));

          // debug, works
          // store_col<double>(fp, particlenum+"/vol", PArr[i].vol);
        }
        fp.close();

        // save wall boundary info
        // if (Wall.move_bdry) {
        // if the wall moves, otherwise a waste of space
        // Save this file always
        char buf_wall[100];
        sprintf(buf_wall, "%05u", counter);
        string tcounter_wall = string(buf);

        string filename_wall = output_loc + "wall_" + tcounter + ".h5";
        H5::H5File fp_w(filename_wall, H5F_ACC_TRUNC);

        store_col<double>(fp_w, "wall_info", Wall.lrtb());
        store_rowvec<double, dim>(fp_w, "reaction", Wall.get_reaction());

        fp.close();
        //}

        auto stop_time = system_clock::now();
        // are nanoseconds, microseconds, milliseconds, seconds, minutes, hours
        // auto duration = duration_cast<seconds>(stop_time - start_time);
        auto duration = duration_cast<milliseconds>(stop_time - start_time);
        // update for next loop
        start_time = stop_time;
        // To get the value of duration use the count(), member function on the
        // duration object
        // cout << "Runtime: " << duration.count() << "s" << endl;
        // std::cout << "Duration: " << duration.count() << std::endl;

        // prediction: from last n avg
        int avgsteps = 10;
        double avg_duration = 0;
        if (run_time.size() >= 10) {
          for (unsigned cc = (run_time.size() - avgsteps); cc < run_time.size();
               cc++) {
            avg_duration += run_time[cc];
          }
	  avg_duration /= avgsteps;
        }
        int rem_counts = last_counter - counter;
        // in mins
        //double rem_duration = (rem_counts * avg_duration) / 1e3 / 60;
	int rem_duration_s = (rem_counts * avg_duration) / 1e3;

	int rem_sec = (rem_duration_s) % 60;
	int rem_min = (rem_duration_s / 60) % 60;
	int rem_hr = (rem_duration_s / 60) / 60;

        // time_t my_time;
        time_t my_time = time(NULL);
        char timestring[80];
	strftime(timestring, 80, "%F-%T", localtime(&my_time));

	//std::cout << left  << std::printf("Here: %02d:%02d:%02d :end", rem_hr, rem_min, rem_sec) << std::endl;

        std::cout 
	    << left << std::setw(column_w) << t 
	    << left << std::setw(column_w) << counter 
	    << left << std::setw(column_w) << (double)duration.count() / 1000
	    //<< left << std::setw(column_w) << std::printf("%d:%d:%02.1f", (int)(rem_duration/60), (int)rem_duration, (rem_duration - (int)rem_duration) * 60 )
	    << left << std::setw(column_w) << std::printf("%02d:%02d:%02d ", rem_hr, rem_min, rem_sec)
	    << left << std::setw(column_w+10) << timestring
	    << std::endl;

        run_time.push_back(duration.count());
        t_ind.push_back(t);

        // update
        ++counter;
      }
    }
  }

  // save runtime to file
  string runtime_filename = "output/hdf5/run_time.h5";
  H5::H5File rt_fp(runtime_filename, H5F_ACC_TRUNC);
  store_col<double>(rt_fp, "run_time", run_time);
  store_col<double>(rt_fp, "t_ind", t_ind);
  rt_fp.close();
};

template <unsigned dim>
void run_timeloop_worm(vector<ParticleN<dim>> &PArr, Timeloop TL, Contact CN,
                       RectWall<dim> Wall) {
  unsigned total_particles_univ = PArr.size();
  unsigned counter;
  unsigned first_counter, last_counter;

  if (TL.do_resume) {
    counter = TL.resume_ind + 1;
    first_counter = counter;
    last_counter = TL.resume_ind + TL.timesteps / TL.modulo;

    string data_loc = "output/hdf5/";
    char buf[20];
    sprintf(buf, "tc_%05u.h5", TL.resume_ind);
    string h5file = string(buf);
    string filename = data_loc + h5file;
    std::cout << "Loading from file: " << filename << std::endl;

    for (unsigned i = 0; i < total_particles_univ; ++i) {
      char buf2[100];
      sprintf(buf2, "P_%05u", i);
      string particlenum = string(buf2);
      PArr[i].disp =
          load_rowvecs<double, dim>(filename, particlenum + "/CurrPos") -
          PArr[i].pos;
      PArr[i].vel = load_rowvecs<double, dim>(filename, particlenum + "/vel");
      PArr[i].acc = load_rowvecs<double, dim>(filename, particlenum + "/acc");
    }

    if (TL.wall_resume) {
      // resume wall info
      char buf_w[20];
      sprintf(buf_w, "wall_%05u.h5", TL.resume_ind);
      string wall_file = string(buf_w);
      string filename_wall = data_loc + wall_file;
      std::cout << "Loading wall from file: " << filename_wall << std::endl;
      auto vv = load_col<double>(filename_wall, "wall_info");
      Wall.set_lrtb(vv);
    }

    std::cout << "Resuming from counter " << counter << std::endl;
  } else {
    counter = 1;
    first_counter = counter;
    last_counter = TL.timesteps / TL.modulo;
  }

  // Save plotinfo
  string plt_filename = "output/hdf5/plotinfo.h5";
  H5::H5File pl_fp(plt_filename, H5F_ACC_TRUNC);

  vector<int> al_wall = {Wall.allow_wall};
  vector<double> geom_wall_info = Wall.lrtb();
  vector<unsigned> f_l_counter = {first_counter, last_counter};
  // std::cout << "geom_wall_info" << geom_wall_info << std::endl;
  store_col<unsigned>(pl_fp, "f_l_counter", f_l_counter);

  vector<unsigned> vec_dim = {dim};
  store_col<unsigned>(pl_fp, "dimension", vec_dim);

  vector<double> vec_dt = {TL.dt};
  store_col<double>(pl_fp, "dt", vec_dt);

  vector<unsigned> vec_modulo = {TL.modulo};
  store_col<unsigned>(pl_fp, "modulo", vec_modulo);

  pl_fp.createGroup("/wall");
  store_col<int>(pl_fp, "/wall/allow_wall", al_wall);
  store_col<double>(pl_fp, "/wall/geom_wall_info", geom_wall_info);
  pl_fp.close();

  time_t my_time = time(NULL);

  std::cout << "---------------------------------------------------------------"
               "-------"
            << std::endl;
  std::cout << "Starting time loop "
            << "(Parallel=" << TL.run_parallel << "): " << ctime(&my_time);

  auto start_time = system_clock::now();
  vector<double> run_time, t_ind;

  for (unsigned t = 1; t <= TL.timesteps; ++t) {

    // set wall reaction to zero
    Wall.setzero_reaction();

    // worm - clamp either the front (0,3) or back nodes (1,2)
    // int t_num = ((int) (t/2000) ) % 2;
    //
    int t_front_fixed = 1000;
    // fix back nodes for this multiple of t_front_fixed
    int t_relaxation_factor = 2;

    int t_num;
    if ((t / t_front_fixed) % (t_relaxation_factor + 1) == 0) {
      t_num = 0;
    } else {
      t_num = 1;
    }

    // new pos
// pragma here works [1]
#pragma omp parallel for if (TL.run_parallel)
    for (unsigned i = 0; i < total_particles_univ; ++i) {
      PArr[i].disp += TL.dt * PArr[i].vel + (TL.dt * TL.dt * 0.5) * PArr[i].acc;
      PArr[i].CurrPos = PArr[i].pos + PArr[i].disp;
      // std::cout << PArr[i].mean_CurrPos() << std::endl;
    }

/**********************************************************************
contact force computation
**********************************************************************/
//#pragma omp parallel for num_threads(2)
// pragma here works [2]
#pragma omp parallel for schedule(dynamic) if (TL.run_parallel)
    //#pragma omp parallel for
    for (unsigned i = 0; i < total_particles_univ; ++i) {
      // PArr[i].force = PArr[i].get_peridynamic_force();

      // define ft_i and initialize by zero
      vector<Matrix<double, 1, dim>> ft_i;
      ft_i.resize(PArr[i].nnodes);
      for (unsigned ll = 0; ll < ft_i.size(); ll++) {
        ft_i[ll] = Matrix<double, 1, dim>::Zero();
      }

      // compute total internal force only if it is movable
      if (PArr[i].movable) {

        // auto ft_i = PArr[i].get_peridynamic_force();
        ft_i = PArr[i].get_peridynamic_force();

        if (TL.enable_fracture) {
          if (PArr[i].breakable) {
            // PArr[i].remove_bonds(t);
            PArr[i].remove_bonds();
            // self-contact: update ft_i, 1==is_self_contact
            if (CN.self_contact) {
              update_contact_force_by_boundary<dim>(PArr[i], PArr[i], CN, TL.dt,
                                                    ft_i, 1);
            }
          }
        }

        vector<double> C_lrtb = get_minmax<dim>(PArr[i].CurrPos);
        //#pragma omp parallel for
        for (unsigned j = 0; j < total_particles_univ; j++) {
          if (j != i) {
            vector<double> N_lrtb = get_minmax<dim>(PArr[j].CurrPos);
            // if extended bounding boxes intersect
            if (intersects_box_fext<dim>(C_lrtb, N_lrtb, CN.contact_rad)) {
              update_contact_force_by_boundary<dim>(PArr[i], PArr[j], CN, TL.dt,
                                                    ft_i, 0);
            }
          }
        }
        //#pragma omp parallel for
        if (Wall.allow_wall) {
          if (!within_interior_fext(C_lrtb, Wall.lrtb(), CN.contact_rad)) {
            update_wall_contact_force_by_boundary<dim>(PArr[i], Wall, CN, TL.dt,
                                                       ft_i);
          }
        }
        // add the external force
        // ft_i += PArr[i].extforce;
        ft_i += ((t_num + 1) % 2) * PArr[i].extforce;
      }
      // store the computed value into the particle force
      PArr[i].force = ft_i;
    }

// pragma here works [3]
#pragma omp parallel for if (TL.run_parallel)
    // update quantities
    for (unsigned i = 0; i < total_particles_univ; ++i) {
      // update if movable
      if (PArr[i].movable) {
        auto temp_acc = PArr[i].acc;
        // New acceleration
        PArr[i].acc = (1 / PArr[i].rho) * (PArr[i].force);
        //	// velocity
        temp_acc += PArr[i].acc;   // Now temp_acc = (acc + acc_old)
        temp_acc *= (0.5 * TL.dt); // now temp_acc = (dt*0.5) *(acc + acc_old)
        PArr[i].vel += temp_acc;

        // clamped nodes are set to be zero
        if (PArr[i].clamped_nodes.size()) {
          for (unsigned cn = 0; cn < PArr[i].clamped_nodes.size(); cn++) {
            auto cid = PArr[i].clamped_nodes[cn];
            PArr[i].disp[cid] = Matrix<double, 1, dim>::Zero();
            PArr[i].vel[cid] = Matrix<double, 1, dim>::Zero();
            PArr[i].acc[cid] = Matrix<double, 1, dim>::Zero();
          }
        }

        // worm - clamp either the front (0,3) or back nodes (1,2)
        if (t_num == 0) {
          unsigned cid = 0;
          // PArr[i].disp[cid] = Matrix<double, 1, dim>::Zero();
          PArr[i].vel[cid] = Matrix<double, 1, dim>::Zero();
          PArr[i].acc[cid] = Matrix<double, 1, dim>::Zero();

          cid = 3;
          // PArr[i].disp[cid] = Matrix<double, 1, dim>::Zero();
          PArr[i].vel[cid] = Matrix<double, 1, dim>::Zero();
          PArr[i].acc[cid] = Matrix<double, 1, dim>::Zero();
        }

        if (t_num == 1) {
          unsigned cid = 1;
          // PArr[i].disp[cid] = Matrix<double, 1, dim>::Zero();
          PArr[i].vel[cid] = Matrix<double, 1, dim>::Zero();
          PArr[i].acc[cid] = Matrix<double, 1, dim>::Zero();

          cid = 2;
          // PArr[i].disp[cid] = Matrix<double, 1, dim>::Zero();
          PArr[i].vel[cid] = Matrix<double, 1, dim>::Zero();
          PArr[i].acc[cid] = Matrix<double, 1, dim>::Zero();
        }

        // restrict y-motion
        vector<unsigned> cid_list = {0, 1, 2, 3};
        for (unsigned cid_l = 0; cid_l < cid_list.size(); cid_l++) {
          auto cid = cid_list[cid_l];
          PArr[i].vel[cid][1] = 0;
          PArr[i].acc[cid][1] = 0;
        }
      }
    }

    // wall boundary update
    if (dim == 2) {
      Wall.left += Wall.speed_left * TL.dt;
      Wall.right += Wall.speed_right * TL.dt;
      Wall.top += Wall.speed_top * TL.dt;
      Wall.bottom += Wall.speed_bottom * TL.dt;
    } else {
      Wall.x_min += Wall.speed_x_min * TL.dt;
      Wall.y_min += Wall.speed_y_min * TL.dt;
      Wall.z_min += Wall.speed_z_min * TL.dt;
      Wall.x_max += Wall.speed_x_max * TL.dt;
      Wall.y_max += Wall.speed_y_max * TL.dt;
      Wall.z_max += Wall.speed_z_max * TL.dt;
    }

    // save
    if (TL.save_file) {

      if ((t % TL.modulo) == 0) {
        char buf[100];
        sprintf(buf, "%05u", counter);
        string tcounter = string(buf);

        string output_loc = "output/hdf5/";
        string filename = output_loc + "tc_" + tcounter + ".h5";
        H5::H5File fp(filename, H5F_ACC_TRUNC);

// pragma works here [4]
#pragma omp parallel for if (TL.run_parallel)
        for (unsigned i = 0; i < total_particles_univ; i++) {
          char buf2[100];
          sprintf(buf2, "P_%05u", i);
          string particlenum = string(buf2);
          fp.createGroup(particlenum);
          store_rowvec<double, dim>(fp, particlenum + "/CurrPos",
                                    PArr[i].CurrPos);
          store_rowvec<double, dim>(fp, particlenum + "/vel", PArr[i].vel);
          store_rowvec<double, dim>(fp, particlenum + "/acc", PArr[i].acc);
          // This is actually the force density (Force/Volume), NOT the Force
          store_rowvec<double, dim>(fp, particlenum + "/force", PArr[i].force);
        }
        fp.close();

        // save wall boundary info
        // if (Wall.move_bdry) {
        // if the wall moves, otherwise a waste of space
        // Save this file always
        char buf_wall[100];
        sprintf(buf_wall, "%05u", counter);
        string tcounter_wall = string(buf);

        string filename_wall = output_loc + "wall_" + tcounter + ".h5";
        H5::H5File fp_w(filename_wall, H5F_ACC_TRUNC);

        store_col<double>(fp_w, "wall_info", Wall.lrtb());
        store_rowvec<double, dim>(fp_w, "reaction", Wall.get_reaction());

        fp.close();
        //}

        time_t my_time = time(NULL);
        // std::cout << "t: " << t << " counter: " << counter << " "  <<
        // ctime(&my_time); std::cout << "--------------------------" <<
        // std::endl;

        auto stop_time = system_clock::now();
        // are nanoseconds, microseconds, milliseconds, seconds, minutes, hours
        // auto duration = duration_cast<seconds>(stop_time - start_time);
        auto duration = duration_cast<milliseconds>(stop_time - start_time);
        // update for next loop
        start_time = stop_time;
        // To get the value of duration use the count(), member function on the
        // duration object
        // cout << "Runtime: " << duration.count() << "s" << endl;
        // std::cout << "Duration: " << duration.count() << std::endl;

        // prediction: from last n avg
        int avgsteps = 10;
        double avg_duration = 0;
        if (run_time.size() >= 10) {
          for (unsigned cc = (run_time.size() - avgsteps); cc < run_time.size();
               cc++) {
            avg_duration += run_time[cc];
          }
        }
        avg_duration /= avgsteps;
        int rem_counts = last_counter - counter;
        // in mins
        double rem_duration = (rem_counts * avg_duration) / 1e3 / 60;

        // std::cout << "t: " << t << " counter: " << counter << " duration: "
        //<< duration.count()
        ////<< " rem counts: " << rem_counts
        ////<< " avg duration: " << avg_duration
        //<< " rem: " << rem_duration << "m " << ctime(&my_time);
        std::cout << t 
		    << "\t"
		    << counter
		    << "\t"
		    << duration.count()/100
		    << "\t"
		    << (int) rem_duration << "m" << (rem_duration - (int) rem_duration)*60 << "s"
		    << "\t"
		    << ctime(&my_time);

        run_time.push_back(duration.count());
        t_ind.push_back(t);

        // update
        ++counter;
      }
    }
  }

  // save runtime to file
  string runtime_filename = "output/hdf5/run_time.h5";
  H5::H5File rt_fp(runtime_filename, H5F_ACC_TRUNC);
  store_col<double>(rt_fp, "run_time", run_time);
  store_col<double>(rt_fp, "t_ind", t_ind);
  rt_fp.close();
};

#endif /* ifndef TIMELOOP_H */
