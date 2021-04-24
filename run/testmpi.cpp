#include "stdio.h"
#include <stdlib.h>

#include <mpi.h>
int main(int argc, char *argv[])
{

    int tid,nthreads;

    /* add in MPI startup routines */
    /* 1st: launch the MPI processes on each node */
    MPI_Init(&argc,&argv);

    /* 2nd: request a thread id, sometimes called a "rank" from
       the MPI master process, which has rank or tid == 0
       */
    MPI_Comm_rank(MPI_COMM_WORLD, &tid);

    /* 3rd: this is often useful, get the number of threads
       or processes launched by MPI, this should be NCPUs-1
       */
    MPI_Comm_size(MPI_COMM_WORLD, &nthreads);

    ///* on EVERY process, allocate space for the machine name */
    //cpu_name    = (char *)calloc(80,sizeof(char));
    ///* get the machine name of this particular host ... well
    //   at least the first 80 characters of it ... */
    //gethostname(cpu_name,80);

    char processor_name[MPI_MAX_PROCESSOR_NAME]; // gets the name of the processor
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    printf("hello MPI user: from process = %i on machine=%s, of NCPU=%i processes\n",
	    tid, processor_name, nthreads);

    MPI_Finalize();
    return(0);
}
