/*
============================================================================
Filename    : algorithm.c
Author      : Arthur Vernet, Simon Maulini
SCIPER      : 245828, 248115
============================================================================
*/

#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <cuda_runtime.h>
using namespace std;

// CPU Baseline
void array_process(double *input, double *output, int length, int iterations)
{
    double *temp;

    for(int n=0; n<(int) iterations; n++)
    {
        for(int i=1; i<length-1; i++)
        {
            for(int j=1; j<length-1; j++)
            {
                output[(i)*(length)+(j)] = (input[(i-1)*(length)+(j-1)] +
                                            input[(i-1)*(length)+(j)]   +
                                            input[(i-1)*(length)+(j+1)] +
                                            input[(i)*(length)+(j-1)]   +
                                            input[(i)*(length)+(j)]     +
                                            input[(i)*(length)+(j+1)]   +
                                            input[(i+1)*(length)+(j-1)] +
                                            input[(i+1)*(length)+(j)]   +
                                            input[(i+1)*(length)+(j+1)] ) / 9;

            }
        }
        output[(length/2-1)*length+(length/2-1)] = 1000;
        output[(length/2)*length+(length/2-1)]   = 1000;
        output[(length/2-1)*length+(length/2)]   = 1000;
        output[(length/2)*length+(length/2)]     = 1000;

        temp = input;
        input = output;
        output = temp;
    }
}

__global__ void kernel_row(double *input, double *output, int length) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int i = y * length + x;

  //check if the coordinates are out of bounds or corresponding to the heat core
  if(x >= length || y >= length || x == 0 || x == length - 1 || (y == length / 2 || y == length / 2 - 1)
    && (x == length / 2 - 1 || x == length / 2))
    return;

  output[i] = input[i];
  output[i] += input[i - 1];
  output[i] += input[i + 1];
}

__global__ void kernel_column(double *input, double *output, int length) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int i = y * length + x;

  //check if the coordinates are out of bounds or corresponding to the heat core
  if(x >= length || y >= length || y == 0 || y == length - 1 || (y == length / 2 || y == length / 2 - 1)
    && (x == length / 2 - 1 || x == length / 2))
    return;

  output[i] = input[i];
  output[i] += input[i - length];
  output[i] += input[i + length];
  output[i] /= 9; //divide by 9 as this kernel is called the last
}

// GPU Optimized function
void GPU_array_process(double *input, double *output, int length, int iterations)
{
    //Cuda events for calculating elapsed time
    cudaEvent_t cpy_H2D_start, cpy_H2D_end, comp_start, comp_end, cpy_D2H_start, cpy_D2H_end;
    cudaEventCreate(&cpy_H2D_start);
    cudaEventCreate(&cpy_H2D_end);
    cudaEventCreate(&cpy_D2H_start);
    cudaEventCreate(&cpy_D2H_end);
    cudaEventCreate(&comp_start);
    cudaEventCreate(&comp_end);

    /* Preprocessing goes here */
    cudaSetDevice(0);
    size_t size = length*length*sizeof(double);
    double* input_data;
    double* output_data;

    dim3 threadPerBlocks(32, 32);
    dim3 blocks(4, 4);

    // allocate array on device
  	if (cudaMalloc((void **) &input_data, size) != cudaSuccess)
  		cout << "error in cudaMalloc" << endl;
    if (cudaMalloc((void **) &output_data, size) != cudaSuccess)
      cout << "error in cudaMalloc" << endl;

    cudaEventRecord(cpy_H2D_start);

    /* Copying array from host to device goes here */
    if (cudaMemcpy(input_data, input, size, cudaMemcpyHostToDevice) != cudaSuccess)
      cout << "error in cudaMemcpy" << endl;

    cudaEventRecord(cpy_H2D_end);
    cudaEventSynchronize(cpy_H2D_end);

    cudaEventRecord(comp_start);

    /* GPU calculation goes here */
    for(int i = 0; i < iterations; ++i) {
      kernel_row <<< blocks, threadPerBlocks >>> (input_data, output_data, length);
      kernel_column <<< blocks, threadPerBlocks >>> (output_data, input_data, length);
      cudaThreadSynchronize(); //synchronize at every iterations, works as a barrier
    }

    cudaEventRecord(comp_end);
    cudaEventSynchronize(comp_end);

    cudaEventRecord(cpy_D2H_start);

    /* Copying array from device to host goes here */
    if(cudaMemcpy(output, output_data, size, cudaMemcpyDeviceToHost) != cudaSuccess)
      cout << "Cuda Memcpy DeviceToHost Error: cannot copy output\n";

    cudaEventRecord(cpy_D2H_end);
    cudaEventSynchronize(cpy_D2H_end);

    /* Postprocessing goes here */
    cudaFree(input_data);
    cudaFree(output_data);

    float time;
    cudaEventElapsedTime(&time, cpy_H2D_start, cpy_H2D_end);
    cout<<"Host to Device MemCpy takes "<<setprecision(4)<<time/1000<<"s"<<endl;

    cudaEventElapsedTime(&time, comp_start, comp_end);
    cout<<"Computation takes "<<setprecision(4)<<time/1000<<"s"<<endl;

    cudaEventElapsedTime(&time, cpy_D2H_start, cpy_D2H_end);
    cout<<"Device to Host MemCpy takes "<<setprecision(4)<<time/1000<<"s"<<endl;
}
