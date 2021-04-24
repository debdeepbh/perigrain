#include<iostream>
#include<omp.h>
#include <vector>
#include <cmath>

using namespace std;

int main()
{

    std::cout << "Number of processors: " << omp_get_num_procs() << std::endl;
    std::cout << "Number of threads: " << omp_get_max_threads() << std::endl;

    //int N = 2000000000;
    //int n = 100;
    //double sum = 0.0;
//#pragma omp parallel for reduction(+: sum)
//#pragma omp parallel for 
//#pragma omp parallel for shared(n, sum)
    //for(int k=1 ; k<=N ; k++) {
	//sum += 1.0/(k * (double)n);
    //}

    //std::cout << sum << std::end;
    
    const int size = 1024;
    double sinTable[size];

    omp_set_nested(1);
//#pragma omp parallel for ordered schedule(dynamic)
#pragma omp parallel for num_threads(2)
    for(int n=0; n<size; ++n){

	//std::cout << "Current thread level: " << omp_get_level() << std::endl;
	//std::cout << "Started " << omp_get_num_threads() << std::endl;
	sinTable[n] = std::sin(2 * M_PI * n / size);

#pragma omp parallel for num_threads(2)
	for (unsigned i = 0; i < 200; i++) {
	std::cout << "Current thread level: " << omp_get_level() << std::endl;
	std::cout << "Started " << omp_get_num_threads() << std::endl;
	    sinTable[i + n] = 5 * i * n;
	}

//#pragma omp ordered
	//{
	//std::cout << n << std::endl;
	//}
    }

}
