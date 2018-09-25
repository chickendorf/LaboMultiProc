/*
============================================================================
Filename    : pi.c
Author      : Your names goes here
SCIPER		: Your SCIPER numbers
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
  rand_gen rand = init_rand();
  float x, y;

  #pragma omp parallel for private(x, y) reduction(+:pointsIn)
  for(int i=0;i<samples;i++){
    x = next_rand(rand);
    y = next_rand(rand);

		if(x*x + y*y <= 1){
		    pointsIn++;
		 }
	 }

	return 4*((float)pointsIn)/samples;
}
