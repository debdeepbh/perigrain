#ifndef RW_HDF5_H
#define RW_HDF5_H 
# include <string>
# include <iostream>
#include "Eigen/Dense"
using namespace Eigen;

//#include "read/varload.h"

//#include "particle/particle.h"
#include "debug/err.h"
//#include "read/load_mesh.h"

//#include "read/load_csv.h"
//#include "particle/timeloop.h"

//#include "particle/nbdarr.h"

#include<vector>
using namespace std;

#include<H5Cpp.h>

// src: https://stackoverflow.com/questions/20358644/open-hdf5-string-dataset

//template <typename ttype>
//void store_2d(H5::H5File fp, string dataset, ttype * data, hsize_t * dims) {
  ////H5::H5File fp(filename.c_str(), H5F_ACC_TRUNC);
  ////hsize_t dim[2] = {data.size(), 1};
  //std::cout << "Here, DIM0: " << dims[0] << " DIM1: " << dims[1] << std::endl;
  //hsize_t rank=2;
  //H5::DataSpace dspace(rank, dims); // 1 is the rank of the matrix
  //if (typeid(ttype) == typeid(double)) {
      //H5::DataSet dset = fp.createDataSet(dataset.c_str(), H5::PredType::NATIVE_DOUBLE, dspace);
      //dset.write(data, H5::PredType::NATIVE_INT);
  //} else if (typeid(ttype) == typeid(int)) {
      //H5::DataSet dset = fp.createDataSet(dataset.c_str(), H5::PredType::NATIVE_INT, dspace);
  //dset.write(data, H5::PredType::NATIVE_INT);
  //} else if (typeid(ttype) == typeid(unsigned)) {
      //H5::DataSet dset = fp.createDataSet(dataset.c_str(), H5::PredType::NATIVE_UINT, dspace);
  //dset.write(data, H5::PredType::NATIVE_UINT);
  //} else{
	    //errout("Wrong template type");
  //}
  ////fp.close();  
//};

template <typename ttype, unsigned dim>
void store_rowvec(H5::H5File fp, string dataset, vector<Matrix<ttype, 1, dim>> v) {
  //H5::H5File fp(filename.c_str(), H5F_ACC_TRUNC);
  hsize_t dims[2] = {v.size(), dim};
  //std::cout << "Here, DIM0: " << dims[0] << " DIM1: " << dims[1] << std::endl;
  ttype data[dims[0]][dims[1]];
  for (unsigned i = 0; i < dims[0]; i++) {
      for (unsigned j = 0; j < dims[1]; j++) {
	  data[i][j] = v[i](j);
      }
      
  }
  hsize_t rank=2;
  H5::DataSpace dspace(rank, dims); // 1 is the rank of the matrix
  if (typeid(ttype) == typeid(double)) {
      H5::DataSet dset = fp.createDataSet(dataset, H5::PredType::NATIVE_DOUBLE, dspace);
      dset.write(data, H5::PredType::NATIVE_DOUBLE);
  } else if (typeid(ttype) == typeid(int)) {
      H5::DataSet dset = fp.createDataSet(dataset, H5::PredType::NATIVE_INT, dspace);
  dset.write(data, H5::PredType::NATIVE_INT);
  } else if (typeid(ttype) == typeid(unsigned)) {
      H5::DataSet dset = fp.createDataSet(dataset, H5::PredType::NATIVE_UINT, dspace);
  dset.write(data, H5::PredType::NATIVE_UINT);
  } else{
	    errout("Wrong template type");
  }
  //fp.close();  
};

template <typename ttype>
void store_col(H5::H5File fp, string dataset, vector<ttype> v) {
  //H5::H5File fp(filename.c_str(), H5F_ACC_TRUNC);
  hsize_t dims[2] = {v.size(), 1};
  ttype data[dims[0]][1];
  for (unsigned i = 0; i < dims[0]; i++) {
	  data[i][0] = v[i];
	  //std::cout << data[i][0] << std::endl;
  }
  hsize_t rank=2;
  H5::DataSpace dspace(rank, dims); // 1 is the rank of the matrix
  if (typeid(ttype) == typeid(double)) {
      H5::DataSet dset = fp.createDataSet(dataset, H5::PredType::NATIVE_DOUBLE, dspace);
      dset.write(data, H5::PredType::NATIVE_DOUBLE);
  } else if (typeid(ttype) == typeid(int)) {
      H5::DataSet dset = fp.createDataSet(dataset, H5::PredType::NATIVE_INT, dspace);
  dset.write(data, H5::PredType::NATIVE_INT);
  } else if (typeid(ttype) == typeid(unsigned)) {
      H5::DataSet dset = fp.createDataSet(dataset, H5::PredType::NATIVE_UINT, dspace);
  dset.write(data, H5::PredType::NATIVE_UINT);
  } else{
	    errout("Wrong template type");
  }
  //fp.close();  
};


//void store(const std::string& filename, const std::string& dataset, const std::vector<double>& data) {
  //H5::H5File fp(filename.c_str(), H5F_ACC_TRUNC);
  //hsize_t dim[2] = {data.size(), 1};
  //H5::DataSpace dspace(1, dim); // 1 is the rank of the matrix
  //H5::DataSet dset = fp.createDataSet(dataset.c_str(), H5::PredType::NATIVE_DOUBLE, dspace);
  //dset.write(data.data(), H5::PredType::NATIVE_DOUBLE);
  //fp.close();  
//}

//std::vector<double> load(const std::string& filename, const std::string& dataset) {
  //H5::H5File fp(filename.c_str(), H5F_ACC_RDONLY);
  //H5::DataSet dset = fp.openDataSet(dataset.c_str());
  //H5::DataSpace dspace = dset.getSpace();
  //hsize_t rank=2;
  //hsize_t dims[rank];  
  //rank = dspace.getSimpleExtentDims(dims, nullptr);
  //std::cout << "DIM0:" << dims[0] << " DIM1:" << dims[1] << std::endl;
  //std::vector<double> data;
  //data.resize(dims[0]);
  //dset.read(data.data(), H5::PredType::NATIVE_DOUBLE, dspace);
  //fp.close();
  //return data;
//};

//auto load2d(const std::string& filename, const std::string& dataset) {
  //H5::H5File fp(filename.c_str(), H5F_ACC_RDONLY);
  //H5::DataSet dset = fp.openDataSet(dataset.c_str());
  //H5::DataSpace dspace = dset.getSpace();
  //hsize_t rank = 2;
  //hsize_t dims[rank];  
  //rank = dspace.getSimpleExtentDims(dims, nullptr);
  //std::cout << "DIM0:" << dims[0] << " DIM1:" << dims[1] << std::endl;
  //double data[dims[0]][dims[1]];
  ////data.resize(dims[0]);
  //dset.read(data, H5::PredType::NATIVE_DOUBLE, dspace);
  //fp.close();
  ////std::cout << data[0][0] << data[1][0] << std::endl;
  //for (unsigned i = 0; i < dims[0]; i++) {
      //for (unsigned j = 0; j < dims[1]; j++) {
          //std::cout << data[i][j] << " ";
      //}
      //std::cout << " " << std::endl;
  //}
  //return data;
//};
//
//template <typename ttype>
//std::vector<ttype> load_vector(const std::string& filename, const std::string& dataset) {
  //H5::H5File fp(filename.c_str(), H5F_ACC_RDONLY);
  //H5::DataSet dset = fp.openDataSet(dataset.c_str());
  //H5::DataSpace dspace = dset.getSpace();
  //hsize_t rank;
  //hsize_t dims[2];  
  //rank = dspace.getSimpleExtentDims(dims, nullptr);
  //std::cout << "DIM0:" << dims[0] << " DIM1:" << dims[1] << std::endl;
  //std::vector<ttype> data;
  //data.resize(dims[0]);
  //dset.read(data.data(), H5::PredType::NATIVE_DOUBLE, dspace);
  //fp.close();
  //return data;
//};

template <typename ttype>
auto load_rank1(const std::string& filename, const std::string& dataset) {
  H5::H5File fp(filename.c_str(), H5F_ACC_RDONLY);
  H5::DataSet dset = fp.openDataSet(dataset.c_str());
  H5::DataSpace dspace = dset.getSpace();
  hsize_t rank = 1;
  hsize_t dims[rank];  
  rank = dspace.getSimpleExtentDims(dims, nullptr);
  //std::cout << "DIM0:" << dims[0] << " DIM1:" << dims[1] << std::endl;
  //if (dims[1] !=1) {
      //cout << "Number of columns is not 1. Reading 1 column." << endl;
  //}
  //std::cout << "dims " << dims[0] << std::endl;
  ttype data[dims[0]];
  if (typeid(ttype) == typeid(double)) {
      dset.read(data, H5::PredType::NATIVE_DOUBLE, dspace);
  } else if (typeid(ttype) == typeid(int)) {
      dset.read(data, H5::PredType::NATIVE_INT, dspace);
  } else if (typeid(ttype) == typeid(unsigned)) {
      dset.read(data, H5::PredType::NATIVE_UINT, dspace);
  } else {
	    errout("Wrong template type");
  }

  vector<ttype> v;
  v.resize(dims[0]);
  for (unsigned i = 0; i < dims[0]; i++) {
	      v[i] = data[i];
  }
  return v;
};

template <typename ttype>
auto load_col(const std::string& filename, const std::string& dataset) {
  H5::H5File fp(filename.c_str(), H5F_ACC_RDONLY);
  H5::DataSet dset = fp.openDataSet(dataset.c_str());
  H5::DataSpace dspace = dset.getSpace();
  hsize_t rank = 2;
  hsize_t dims[rank];  
  rank = dspace.getSimpleExtentDims(dims, nullptr);
  //std::cout << "DIM0:" << dims[0] << " DIM1:" << dims[1] << std::endl;
  if (dims[1] !=1) {
      cout << "Number of columns is not 1. Reading 1 column." << endl;
  }
  ttype data[dims[0]][dims[1]];
  if (typeid(ttype) == typeid(double)) {
      dset.read(data, H5::PredType::NATIVE_DOUBLE, dspace);
  } else if (typeid(ttype) == typeid(int)) {
      dset.read(data, H5::PredType::NATIVE_INT, dspace);
  } else if (typeid(ttype) == typeid(unsigned)) {
      dset.read(data, H5::PredType::NATIVE_UINT, dspace);
  } else {
	    errout("Wrong template type");
  }

  vector<ttype> v;
  v.resize(dims[0]);
  for (unsigned i = 0; i < dims[0]; i++) {
	      v[i] = data[i][0];
  }
  return v;
};

template <typename ttype, unsigned dim>
auto load_rowvecs(const std::string& filename, const std::string& dataset) {
  H5::H5File fp(filename.c_str(), H5F_ACC_RDONLY);
  H5::DataSet dset = fp.openDataSet(dataset.c_str());
  H5::DataSpace dspace = dset.getSpace();
  hsize_t rank = 2;
  hsize_t dims[rank];  
  rank = dspace.getSimpleExtentDims(dims, nullptr);
  //std::cout << "DIM0:" << dims[0] << " DIM1:" << dims[1] << std::endl;
  if (dim != dims[1]) {
      cout << "Number of columns do not match. Reading specified columns as Eigen::Vector." << endl;
  }
  ttype data[dims[0]][dims[1]];
  if (typeid(ttype) == typeid(double)) {
      dset.read(data, H5::PredType::NATIVE_DOUBLE, dspace);
  } else if (typeid(ttype) == typeid(int)) {
      dset.read(data, H5::PredType::NATIVE_INT, dspace);
  } else if (typeid(ttype) == typeid(unsigned)) {
      dset.read(data, H5::PredType::NATIVE_UINT, dspace);
  } else {
	    errout("Wrong template type");
  }

  vector<Matrix<ttype, 1, dim>> v;
  v.resize(dims[0]);
  for (unsigned i = 0; i < dims[0]; i++) {
	  for (unsigned j = 0; j < dim; j++) {
	      v[i](j) = data[i][j];
	  }
  }
  return v;
};

//int main() {
  //std::vector<double> data = {1, 2, 3, 4, 5, 6};
  //store("mydata.h5", "my dataset", data);
  //auto data_read = load("mydata.h5", "my dataset");
  //for(auto item: data_read) {
    //std::cout<<item<<" ";
  //}
  //
#endif /* ifndef RW_HDF5_H */
