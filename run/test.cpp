# include <string>
# include <iostream>
#include "Eigen/Dense"
using namespace Eigen;

//#include "read/varload.h"

//#include "particle/particle.h"
#include "particle/particle2.h"
//#include "read/load_mesh.h"
#include "read/load_hdf5.h"


int main() {
  //std::vector<double> data = {1, 2, 3, 4, 5, 6}; //row
  //store("mydata.h5", "my dataset", data);
  //auto data_read = load("mydata.h5", "my dataset");
  //auto data_read = load("data/all.h5", "P_00001/Vol");
  //auto data_read = load_rowvecs<double, 2> ("data/all.h5", "P_00001/Pos");
  //std::cout << data_read << std::endl;

  //auto data_read2 = load_col<double> ("data/all.h5", "P_00001/Vol");
  //std::cout << data_read2 << std::endl;

  //vector<Matrix<double, 1, 7>> v;
  //vector<double> vd(5);
  //v.resize(5);
  //int ndata[5][7];
  //for (unsigned i = 0; i < 5; i++) {
      //for (unsigned j = 0; j < 7; j++) {
          //ndata[i][j] = i * j;
	  //v[i](j) = i * j;
      //}
      //vd[i] = ((double)i*i)/10;
  //}

  //std::cout << "vd is: " << vd << std::endl;

  ////string filename = "mydata_new.h5";
  ////H5::H5File fp(filename, H5F_ACC_TRUNC);
  ////hsize_t dims[2] = {5, 7};
  ////string dataset = "my dataset";
  ////store_2d<int>(fp, dataset, &ndata[0][0], &dims[0]);
  //////store_2d<int>(fp, dataset, ndata, dims);
  ////fp.close();  

  //std::cout << "v is: " << v << std::endl;

  //std::cout << "Storing vector "  << std::endl;
  //string filename2 = "mydata_vec.h5";
  //H5::H5File fp2(filename2, H5F_ACC_TRUNC);
  //string dataset2 = "ds";
  //store_rowvec<double, 7>(fp2, dataset2, v);
  //store_col<double>(fp2,"coldata", vd);
  //fp2.close();  


  //std::cout << "Loading vector "  << std::endl;
  //auto out = load_rowvecs<double, 7> (filename2, dataset2);
  //std::cout << out << std::endl;

  //std::cout << "Loading coldata "  << std::endl;
  //std::cout << load_col<double> (filename2, "coldata") << std::endl;


  ////auto data_read3 = load_rowvecs<int, 7> ("mydata_new.h5", dataset);
  ////std::cout << data_read3 << std::endl;

  ////auto data_read = load2d_vec("data/all.h5", "P_00001/Vol");
  ////std::cout << data_read << std::endl;
  ////for(auto item: data_read) {
    ////std::cout<<item<<" ";
  ////}
  ////store("mydata.h5", "my dataset", data_read);

  ////std::cout<<std::endl;

    const unsigned dim = 2;
    auto PArr = load_particles<dim>();

    Contact CN = load_contact();

    RectWall Wall = load_wall();

    unsigned first_counter = 3;
    unsigned last_counter = 10;


    string plt_filename = "output/test.h5";
    H5::H5File pl_fp(plt_filename, H5F_ACC_TRUNC);

    vector<int> al_wall = { Wall.allow_wall };
    vector<double> geom_wall_info = Wall.lrtb();
    vector<unsigned> f_l_counter = { first_counter, last_counter};
    std::cout << "geom_wall_info" << geom_wall_info << std::endl;

    std::cout << "Saving" << std::endl;
    // Create a group in the file
    //std::cout << f_l_counter << std::endl;
    store_col<unsigned>(pl_fp, "f_l_counter", f_l_counter);
    H5::Group group(pl_fp.createGroup( "/wall" ));
    store_col<int>(pl_fp, "wall/allow_wall", al_wall);
    store_col<double>(pl_fp, "/wall/geom_wall_info", geom_wall_info);
    pl_fp.close();

  return 0;
}
