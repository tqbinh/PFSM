#include "kernelPrintf.h"


//__device__ void __syncthreads(void);
__global__ void kernelPrintf(int *O,int sizeO){
	int i=blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i<sizeO){			
		printf("[%d]:%d ; ",i,O[i]);
	}

}


cudaError_t printInt(int* d_array,int noElem_d_Array){
	cudaError cudaStatus;

	dim3 block(1024);
	dim3 grid((noElem_d_Array+block.x-1)/block.x);

	kernelPrintf<<<grid,block>>>(d_array,noElem_d_Array);
	cudaDeviceSynchronize();

	cudaStatus=cudaGetLastError();
	if(cudaStatus != cudaSuccess){
		fprintf(stderr,"\nkernelPrintInt failed");
		goto Error;
	}
Error:
	
	return cudaStatus;
}

__global__ void kernelprintUnsignedInt(unsigned int *O,int sizeO){
	int i=blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i<sizeO){			
		printf("[%d]:%d ; ",i,O[i]);
	}

}

inline cudaError_t printUnsignedInt(unsigned int* d_array,int noElem_d_Array){
	cudaError cudaStatus;

	dim3 block(1024);
	dim3 grid((noElem_d_Array+block.x-1)/block.x);

	kernelprintUnsignedInt<<<grid,block>>>(d_array,noElem_d_Array);
	cudaDeviceSynchronize();

	cudaStatus=cudaGetLastError();
	if(cudaStatus != cudaSuccess){
		fprintf(stderr,"\nkernelPrintInt failed");
		goto Error;
	}
Error:
	
	return cudaStatus;
}



__global__ void kernelPrintFloat(float* A,int n){
	int i=blockIdx.x*blockDim.x + threadIdx.x;
	if (i<n){
		printf("[%d]:%.0f ;",i,A[i]);
	}

}

cudaError_t printFloat(float* d_array,int numberElementOfArray){
	cudaError cudaStatus;

	dim3 block(1024);
	dim3 grid((numberElementOfArray+block.x-1)/block.x);

	kernelPrintFloat<<<grid,block>>>(d_array,numberElementOfArray);
	cudaDeviceSynchronize();

	cudaStatus=cudaGetLastError();
	if(cudaStatus != cudaSuccess){
		fprintf(stderr,"\nkernelPrintExtention failed");
		goto Error;
	}
Error:
	
	return cudaStatus;
}





