/*
============================================================================
Filename    : implementation.cu
Author      : Romain Jufer
SCIPER      : 229801
============================================================================
*/

#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <cuda_runtime.h>
using namespace std;

const int NB_THREADS = 32;

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

__global__ void GPU_array_rowKernel(double *input, double *output, int length) {
  int xCuda = blockDim.x * blockIdx.x + threadIdx.x;
  int yCuda = blockDim.y * blockIdx.y + threadIdx.y;
  int idx = yCuda * length + xCuda;

  if(xCuda >= length || yCuda >= length)
    return;
  if(xCuda == 0 || xCuda == length - 1) {
    output[idx] = 0;
    return;
  }

  output[idx] = input[idx];
  output[idx] += xCuda == 0 ? 0 : input[idx - 1];
  output[idx] += xCuda == length - 1 ? 0 : input[idx + 1];
}

__global__ void GPU_array_colKernel(double *input, double *output, int length) {
  int xCuda = blockDim.x * blockIdx.x + threadIdx.x;
  int yCuda = blockDim.y * blockIdx.y + threadIdx.y;
  int idx = yCuda * length + xCuda;

  if(xCuda >= length || yCuda >= length)
    return;
  if(yCuda == 0 || yCuda == length - 1) {
    output[idx] = 0;
    return;
  }

  output[idx] = input[idx];
  output[idx] += yCuda == 0 ? 0 : input[idx - length];
  output[idx] += yCuda == length - 1 ? 0 : input[idx + length];
  output[idx] /= 9;

  if((yCuda == length / 2 || yCuda == length / 2 - 1)
    && (xCuda == length / 2 - 1 || xCuda == length / 2)) {
    output[idx] = 1000;
  }
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
    double* input_d = NULL;
    double* output_d = NULL;
    size_t dataSize = length * length * sizeof(double);

    int nbThreadsPerDim = NB_THREADS;
    dim3 threadPerBlocks(nbThreadsPerDim, nbThreadsPerDim);
    int nbBlockPerDim = length / nbThreadsPerDim + 1;
    dim3 blocks(nbBlockPerDim,nbBlockPerDim);

    double* tmpSwap = input_d;

    if(cudaMalloc((void**) &input_d, dataSize) != cudaSuccess) {
      cout << "Cuda Malloc Error : cannot allocate memory for input\n";
    }

    if(cudaMalloc((void**) &output_d, dataSize) != cudaSuccess) {
      cout << "Cuda Malloc Error : cannot allocate memory for output\n";
    }

    cudaEventRecord(cpy_H2D_start);
    /* Copying array from host to device goes here */
    if(cudaMemcpy(input_d, input, dataSize, cudaMemcpyHostToDevice) != cudaSuccess)
      cout << "Cuda Memcpy HostToDevice Error: cannot copy input\n";

    cudaEventRecord(cpy_H2D_end);
    cudaEventSynchronize(cpy_H2D_end);

    cudaEventRecord(comp_start);
    /* GPU calculation goes here */
    for(int i = 0; i < iterations; ++i) {
      GPU_array_rowKernel <<< blocks, threadPerBlocks >>> (input_d, output_d, length);
      GPU_array_colKernel <<< blocks, threadPerBlocks >>> (output_d, input_d, length);
    }
    cudaThreadSynchronize();

    tmpSwap = input_d;
    input_d = output_d;
    output_d = tmpSwap;

    cudaEventRecord(comp_end);
    cudaEventSynchronize(comp_end);

    cudaEventRecord(cpy_D2H_start);
    /* Copying array from device to host goes here */
    if(cudaMemcpy(output, output_d, dataSize, cudaMemcpyDeviceToHost) != cudaSuccess)
      cout << "Cuda Memcpy DeviceToHost Error: cannot copy output\n";

    cudaEventRecord(cpy_D2H_end);
    cudaEventSynchronize(cpy_D2H_end);

    /* Postprocessing goes here */
    cudaFree(input_d);
    cudaFree(output_d);

    float time;
    cudaEventElapsedTime(&time, cpy_H2D_start, cpy_H2D_end);
    cout<<"Host to Device MemCpy takes "<<setprecision(4)<<time/1000<<"s"<<endl;

    cudaEventElapsedTime(&time, comp_start, comp_end);
    cout<<"Computation takes "<<setprecision(4)<<time/1000<<"s"<<endl;

    cudaEventElapsedTime(&time, cpy_D2H_start, cpy_D2H_end);
    cout<<"Device to Host MemCpy takes "<<setprecision(4)<<time/1000<<"s"<<endl;
}
