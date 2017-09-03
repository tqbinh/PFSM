#include "reduction.h"


#define funcCheck(stmt) {                                            \
    cudaError_t err = stmt;                                          \
    if (err != cudaSuccess)                                          \
    {                                                                \
        printf( "Failed to run stmt %d ", __LINE__);                 \
        printf( "Got CUDA error ...  %s ", cudaGetErrorString(err)); \
        return cudaStatus;                                                   \
    }                                                                \
}

__global__  void total(float * input, float * output, int len) 
{
	// Load a segment of the input vector into shared memory
	__shared__ float partialSum[2*BLOCK_SIZE];
	int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int t = threadIdx.x;
	unsigned int start = 2*blockIdx.x*blockDim.x;

	if ((start + t) < len)
	{
		partialSum[t] = input[start + t];      
	}
	else
	{       
		partialSum[t] = 0.0;
	}
	if ((start + blockDim.x + t) < len)
	{   
		partialSum[blockDim.x + t] = input[start + blockDim.x + t];
	}
	else
	{
		partialSum[blockDim.x + t] = 0.0;
	}

	// Traverse reduction tree
	for (unsigned int stride = blockDim.x; stride > 0; stride /= 2)
	{
		__syncthreads();
		if (t < stride)
			partialSum[t] += partialSum[t + stride];
	}
	__syncthreads();

	// Write the computed sum of the block to the output vector at correct index
	if (t == 0 && (globalThreadId*2) < len)
	{
		output[blockIdx.x] = partialSum[t];
	}
}


cudaError_t reduction(float *deviceInput,int len,float &support){
	cudaError_t cudaStatus;	
	
    float * deviceOutput;

	int numInputElements = len; // number of elements in the input list
	int numOutputElements; // number of elements in the output list

	numOutputElements = numInputElements / (BLOCK_SIZE<<1);
	if (numInputElements % (BLOCK_SIZE<<1)) 
	{
		numOutputElements++;
	}
		
    funcCheck(cudaMalloc((void **)&deviceOutput, numOutputElements * sizeof(float)));

	// Initialize the grid and block dimensions here
    dim3 DimGrid( numOutputElements, 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    // Launch the GPU Kernel here
    total<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numInputElements);
	
	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaDeviceSynchronize() reduction failed");
		goto Error;
	}
	//printf("\n");
	//printFloat(deviceOutput,numOutputElements);
	cudaMemcpy(&support,deviceOutput,numOutputElements*sizeof(float),cudaMemcpyDeviceToHost);

//	cudaFree(deviceInput);
//	cudaFree(deviceOutput);
Error:
	return cudaStatus;
}


inline cudaError_t reduction(int *input,int len,int &support){
		cudaError_t cudaStatus;	
	
    float * deviceOutput;
	float *deviceInput;


	int numInputElements = len; // number of elements in the input list
	int numOutputElements; // number of elements in the output list

	numOutputElements = numInputElements / (BLOCK_SIZE<<1);
	if (numInputElements % (BLOCK_SIZE<<1)) 
	{
		numOutputElements++;
	}
		
    funcCheck(cudaMalloc((void **)&deviceOutput, numOutputElements * sizeof(float)));		
    funcCheck(cudaMalloc((void **)&deviceInput, numInputElements * sizeof(float)));

	dim3 block(512);
	dim3 grid((numInputElements+block.x-1)/block.x);
	kernelCastingInt2Float<<<grid,block>>>(deviceInput,input,numInputElements);
	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n kernelCastingInt2Float() reduction() failed");
		goto Error;
	}

	// Initialize the grid and block dimensions here
    dim3 DimGrid( numOutputElements, 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    // Launch the GPU Kernel here
    total<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numInputElements);	
	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n total kernel in reduction() failed");
		goto Error;
	}

	int *deviceOutputInt=nullptr;
	cudaStatus = cudaMalloc((void**)&deviceOutputInt,numOutputElements*sizeof(int));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMalloc((void**)&deviceOutputInt in reduction() failed");
		goto Error;
	}

	kernelCastingFloat2Int<<<grid,block>>>(deviceOutputInt,deviceOutput,numOutputElements);
	cudaDeviceSynchronize();

	funcCheck(cudaMemcpy(&support,deviceOutputInt,numOutputElements*sizeof(int),cudaMemcpyDeviceToHost));

	cudaFree(deviceInput);
	cudaFree(deviceOutput);
	cudaFree(deviceOutputInt);
Error:
	return cudaStatus;
}
