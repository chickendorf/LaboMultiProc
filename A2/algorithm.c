/*
============================================================================
Filename    : algorithm.c
Author      : Your names go here
SCIPER      : Your SCIPER numbers

============================================================================
*/
#include <math.h>

#define INPUT(I,J) input[(I)*length+(J)]
#define OUTPUT(I,J) output[(I)*length+(J)]

void simulate(double *input, double *output, int threads, int length, int iterations)
{
    double *temp;
	
	int stepSize = length / threads;
	
	omp_set_num_threads(threads);
    
    // Parallelize this!!
    for(int n=0; n < iterations; n++)
    {
		#pragma omp parallel for collapse(2)
		for(int stepY = 0; stepY < length; stepY += stepSize){
			for(int stepX = 0; stepX < length; stepX += stepSize){
				int endX = stepX + stepSize;
				endX = (endX > length - 1) ? length-1 : endX;
				
				int endY = stepY + stepSize;
				endY = (endY > length - 1) ? length-1 : endY;
				
				int tmpY = (stepY == 0) ? 1 : stepY;
				int tmpX = (stepX == 0) ? 1 : stepX;
				
				for(int y = tmpY; y < endY; y++){
					for(int x = tmpX; x < endX; x++){
						if (((x == length/2-1) || (x== length/2)) && ((y == length/2-1) || (y == length/2))){
							continue;
						}

						//OUTPUT(x,y) = (INPUT(x-1,y-1) + INPUT(x-1,y) + INPUT(x-1,y+1) +
						OUTPUT(x,y) = (INPUT(x-1,y-1) + INPUT(x-1,y) + INPUT(x-1,y+1) +
                                   INPUT(x,y-1)   + INPUT(x,y)   + INPUT(x,y+1)   +
                                   INPUT(x+1,y-1) + INPUT(x+1,y) + INPUT(x+1,y+1))/9.0;
					}
				}
			}
		}
		
		temp = input;
        input = output;
        output = temp;
    }
}
