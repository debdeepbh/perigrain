#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "debug/err.h"

using namespace std;

// From
// https://stackoverflow.com/questions/1120140/how-can-i-read-and-parse-csv-files-in-c

// read vector of vectors
template <typename ttype>
vector<vector<ttype>> read_vecvec(string filename) {
  vector<vector<ttype>> V;
  std::ifstream file(filename);
  string line;
  while (getline(file, line)) {
    stringstream lineStream(line);
    // read each line
    string cell;
    vector<ttype> row;
    while (getline(lineStream, cell, ' ')) {
	if (typeid(ttype) == typeid(double)) {
      row.push_back(stod(cell));
	} 
	else if (typeid(ttype) == typeid(int)) {
      row.push_back(stoi(cell));
	}
	else if (typeid(ttype) == typeid(unsigned)) {
	    auto ii = (unsigned) stoi(cell);
      row.push_back(ii);
	}
	else	
	{
	    errout("Wrong template type");
	}

    }
    V.push_back(row);
  }
  return V;
};

// read vector of vectors, with possible trimming of zeros
template <typename ttype>
vector<vector<ttype>> read_vecvec(string filename, bool trim_zero) {
    vector<vector<ttype>> V;
    std::ifstream file(filename);
    string line;
    while (getline(file, line)) {
	stringstream lineStream(line);
	// read each line
	string cell;
	vector<ttype> row;
	while (getline(lineStream, cell, ' ')) {
	    if (typeid(ttype) == typeid(double)) {
		auto ii = stod(cell);
		if (!(trim_zero && (ii == 0))) {
		    row.push_back(ii);
		}
	    } 
	    else if (typeid(ttype) == typeid(int)) {
		auto ii = stoi(cell);
		if (!(trim_zero && (ii == 0))) {
		    row.push_back(ii);
		}
	    }
	    else if (typeid(ttype) == typeid(unsigned)) {
		auto ii = (unsigned) stoi(cell);
		if (!(trim_zero && (ii == 0))) {
		    row.push_back(ii);
		}
	    }
	    else	
	    {
		errout("Wrong template type");
	    }


    }
    V.push_back(row);
  }
  return V;
};

// read vector of RowVectors
template <typename ttype>
vector<ttype> read_rowvecs(string filename) {
  vector<ttype> V;
  std::ifstream file(filename);
  string line;
  while (getline(file, line)) {
    int i = 0;
    stringstream lineStream(line);
    // read each line
    string cell;
    ttype row;
    while (getline(lineStream, cell, ' ')) {
      row(i) = stod(cell);
      i++;
    }
    V.push_back(row);
  }
  return V;
};

// read a column
template <typename ttype>
vector<ttype> read_col(string filename) {
    vector<ttype> V;
    std::ifstream file(filename);
    string line;
    while (getline(file, line)) {
	if (typeid(ttype) == typeid(double)) {
	    V.push_back(stod(line));
	} else if (typeid(ttype) == typeid(int)) {
	    V.push_back(stoi(line));
	} else if (typeid(ttype) == typeid(unsigned)) {
	    auto ii = (unsigned) stoi(line);
	    V.push_back(ii);
	}
	else	
	{
	    errout("Wrong template type");
	}
    }
    return V;
};

