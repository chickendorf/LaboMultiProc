/*
============================================================================
Filename    : algorithm.c
Author      : Simon MAULINI & Arthur VERNET
SCIPER      : 248115 & XXXXXX

============================================================================
*/
#include <math.h>

#define INPUT(I,J) input[(I)*length+(J)]
#define OUTPUT(I,J) output[(I)*length+(J)]

void simulate(double *input, double *output, int threads, int length, int iterations)
{
    double *temp;
	
	int stepSize = length / threads;
	int endX, endY, tmpX, tmpY, stepY, stepX, x, y;
	double tmpVal;
	
	omp_set_num_threads(threads);
    
    for(int n=0; n < iterations; n++)
    {
		#pragma omp parallel for collapse(2)
		for(stepY = 0; stepY < length; stepY += stepSize){
			for(stepX = 0; stepX < length; stepX += stepSize){
				
				//Check if the end of the step is out of bound
				endX = stepX + stepSize;
				endX = (endX > length - 1) ? length-1 : endX;
				endY = stepY + stepSize;
				endY = (endY > length - 1) ? length-1 : endY;
				
				//Ignore the 0 coordonates
				tmpY = (stepY == 0) ? 1 : stepY;
				tmpX = (stepX == 0) ? 1 : stepX;
				
				for(y = tmpY; y < endY; y++){
					for(x = tmpX; x < endX; x++){
						if (((x == length/2-1) || (x== length/2)) && ((y == length/2-1) || (y == length/2))){
							continue;
						}
						
						//We slit the initial one line instruction to
						// multiple line to be more readable
						tmpVal = 0.0;
						tmpVal += INPUT(x-1,y-1) + INPUT(x-1,y) + INPUT(x-1,y+1);
						tmpVal += INPUT(x,y-1)   + INPUT(x,y)   + INPUT(x,y+1);
						tmpVal += INPUT(x+1,y-1) + INPUT(x+1,y) + INPUT(x+1,y+1);
						OUTPUT(x,y) = tmpVal/9.0;
					}
				}
			}
		}
		
		temp = input;
        input = output;
        output = temp;
    }
}
