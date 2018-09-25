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
    double pi;

	int pointsIn = 0;

    rand_gen rand = init_rand();
    
    #pragma omp parallel for
    for(int i=0;i<samples;i++){
		float x=next_rand(rand);
		float y=next_rand(rand);
		//printf("x : %f - y : %f\n",x,y);
		
		if(x*x+y*y <= 1){
			pointsIn++;
			//printf("x : %f - y : %f ---- %i - %d\n",x,y,i,pointsIn);
		}
	}
	
	return ((float)pointsIn)/((float)samples)*4.0;
    
    //printf("%f\n", next_rand(rand));

    return pi;
}
