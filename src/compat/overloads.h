#ifndef OVERLOADS_H
#define OVERLOADS_H

#include <vector>
#include "particle/particle2.h"
#include "debug/err.h"

using namespace std;

// print a vector of anything
template <typename ttype>
ostream& operator << (ostream& o, vector<ttype> v){
    size_t N = v.size();
    for (size_t i = 0; i < N; ++i) {
	o << "[" << v[i] << "]";
    }
    return o;
};

// Vector and scaler operation
template <typename ttype, typename t2>
vector<ttype> & operator -= (vector<ttype> &v1, t2 v2){
	for (unsigned i = 0; i < v1.size(); ++i) {
	    v1[i] -= v2;
	}
	return v1;
};
template <typename ttype, typename t2>
vector<ttype> & operator += (vector<ttype> &v1, t2 v2){
	for (unsigned i = 0; i < v1.size(); ++i) {
	    v1[i] += v2;
	}
	return v1;
};
template <typename ttype, typename t2>
vector<ttype> & operator *= (vector<ttype> &v1, t2 v2){
	for (unsigned i = 0; i < v1.size(); ++i) {
	    v1[i] *= v2;
	}
	return v1;
};
template <typename ttype, typename t2>
vector<ttype> & operator /= (vector<ttype> &v1, t2 v2){
	for (unsigned i = 0; i < v1.size(); ++i) {
	    v1[i] /= v2;
	}
	return v1;
};


// Vector and vector operation
// Addition of vector of objects, isn't it obvious? Now we need to define _every_ type of point wise operator :(
template <typename ttype>
vector<ttype> operator + (vector<ttype> v1, vector<ttype> v2){
    int N1 = v1.size();
    int N2 = v2.size();

    if (N1 == N2) {
	vector<ttype> v3(N1);
	for (int i = 0; i < N1; ++i) {
	    v3[i] = v1[i] + v2[i];
	}
	return v3;
    }
    else 
    {
	//errout("Incompatible dimension.");
	terminate();
    }
};

template <typename ttype>
vector<ttype> operator - (vector<ttype> v1, vector<ttype> v2){
    int N1 = v1.size();
    int N2 = v2.size();

    if (N1 == N2) {
	vector<ttype> v3(N1);
	for (int i = 0; i < N1; ++i) {
	    v3[i] = v1[i] - v2[i];
	}
	return v3;
    }
    else {
	errout("Incompatible dimension.");
	terminate();
    }
};

template <typename ttype>
vector<ttype> operator * (vector<ttype> v1, vector<ttype> v2){
    int N1 = v1.size();
    int N2 = v2.size();

    if (N1 == N2) {
	vector<ttype> v3(N1);
	for (int i = 0; i < N1; ++i) {
	    v3[i] = v1[i] * v2[i];
	}
	return v3;
    }
    else 
    {
	errout("Incompatible dimension.");
	terminate();
    }
};

template <typename ttype>
vector<ttype> operator / (vector<ttype> v1, vector<ttype> v2){
    int N1 = v1.size();
    int N2 = v2.size();

    if (N1 == N2) {
	vector<ttype> v3(N1);
	for (int i = 0; i < N1; ++i) {
	    v3[i] = v1[i] / v2[i];
	}
	return v3;
    }
    else 
    {
	errout("Incompatible dimension.");
	terminate();
    }
};

// Unary vector operators
template <typename ttype>
vector<ttype> & operator += (vector<ttype> &v1, vector<ttype> v2){
    int N1 = v1.size();
    int N2 = v2.size();

    if (N1 == N2) {
	vector<ttype> v3(N1);
	for (int i = 0; i < N1; ++i) {
	    v1[i] += v2[i];
	}
	return v1;
    }
    else 
    {
	errout("Incompatible dimension.");
	terminate();
    }
};


// scalar and vector operation

template <typename ttype>
vector<ttype> operator * (double s, vector<ttype> v1){
    unsigned N1 = v1.size();
    vector<ttype> v3(N1);
    for (unsigned i = 0; i < N1; ++i) {
	v3[i] = s * v1[i];
    }
    return v3;
};

#endif /* ifndef OVERLOADS_H */
