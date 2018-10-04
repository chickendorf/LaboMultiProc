/*
============================================================================
Filename    : pi.c
Author      : Arthur Vernet, Simon Maulini
SCIPER		: 245828, ??????
============================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include "utility.h"

double calculate_pi (int num_threads, int samples);

int main (int argc, const char *argv[]) {

    int num_threads, num_samples;
    double pi;

    if (argc != 3) {
		printf("Invalid input! Usage: ./pi <num_threads> <num_samples> \n");
		return 1;
	} else {
        num_threads = atoi(argv[1]);
        num_samples = atoi(argv[2]);
	}

    set_clock();
    pi = calculate_pi (num_threads, num_samples);

    printf("- Using %d threads: pi = %.15g computed in %.4gs.\n", num_threads, pi, elapsed_time());

    return 0;
}


double calculate_pi (int num_threads, int samples) {
	omp_set_num_threads(num_threads);

	int pointsIn = 0;
  double x, y;
  int tid;

  #pragma omp parallel
  {
    rand_gen rand = init_rand();
    #pragma omp for private(x, y) reduction(+:pointsIn)
    for(int i = 0; i < samples; i++){
      x = next_rand(rand);
      y = next_rand(rand);
      tid = omp_get_thread_num();
      //printf("i = %d, thread %d\n", i, tid);
  		if(x*x + y*y <= 1){
  		    pointsIn++;
  		 }
  	 }
   }

	return 4*((double)pointsIn)/samples;
}
