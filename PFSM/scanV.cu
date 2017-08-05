#include "scanV.h"


inline cudaError_t scanV(int *V,unsigned int numberElementV,int *index){
	cudaError_t cudaStatus;

	
	//Khởi tạo một mảng indexFloat
	float* indexFloat; //mảng này dùng để chứa kết quả trả về của phép scan
   cudaStatus=cudaMalloc((float**)&indexFloat,numberElementV*sizeof(float));
	if (cudaStatus != cudaSuccess){
		fprintf(stderr,"cudaMalloc failed",cudaStatus);
		goto Error;
	}

	float* VFloat; //mảng này dùng để chứa kết quả của phép chuyển từ kiểu int sang float của mảng V
	cudaStatus = cudaMalloc((float**)&VFloat,numberElementV*sizeof(float));
	if (cudaStatus != cudaSuccess){
		fprintf(stderr,"cudaMalloc failed",cudaStatus);
		goto Error;
	}
	
	dim3 block(512);
	dim3 grid((numberElementV+block.x-1)/block.x);


	kernelCastingInt2Float<<<grid,block>>>(VFloat,V,numberElementV);

	

	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaDeviceSynchronize failed",cudaStatus);
		goto Error;
	}

	//printf("\n Array VLoat: ");
	//kernelPrintFloat<<<grid,block>>>(VFloat,numberElementV);
	
	preallocBlockSums(numberElementV);
	prescanArray(indexFloat,VFloat,numberElementV);

	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaDeviceSynchronize prescanArray failed",cudaStatus);
		goto Error;
	}

	

	kernelCastingFloat2Int<<<grid,block>>>(index,indexFloat,numberElementV);


	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaDeviceSynchronize kernelCastingFloat2Int failed",cudaStatus);
		goto Error;
	}


Error:
	//cudaFree(VFloat);
	//cudaFree(indexFloat);
	//cudaFree(index);
	//cudaFree(V);
	return cudaStatus;
}
inline cudaError_t scanV(unsigned int *V,unsigned int numberElementV,unsigned int *index){
	cudaError_t cudaStatus;

	
	//Khởi tạo một mảng indexFloat
	float* indexFloat; //mảng này dùng để chứa kết quả trả về của phép scan
   cudaStatus=cudaMalloc((float**)&indexFloat,numberElementV*sizeof(float));
	if (cudaStatus != cudaSuccess){
		fprintf(stderr,"cudaMalloc failed",cudaStatus);
		goto Error;
	}

	float* VFloat; //mảng này dùng để chứa kết quả của phép chuyển từ kiểu int sang float của mảng V
	cudaStatus = cudaMalloc((float**)&VFloat,numberElementV*sizeof(float));
	if (cudaStatus != cudaSuccess){
		fprintf(stderr,"cudaMalloc failed",cudaStatus);
		goto Error;
	}
	
	dim3 block(512);
	dim3 grid((numberElementV+block.x-1)/block.x);


	kernelCastingUnsignedInt2Float <<<grid,block>>>(VFloat,V,numberElementV);

	

	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaDeviceSynchronize kernelCastingUnsignedInt2Float() failed",cudaStatus);
		goto Error;
	}

	//printf("\n *********Array VLoat***********\n ");
	//kernelPrintFloat<<<grid,block>>>(VFloat,numberElementV);
	//
	preallocBlockSums(numberElementV);
	prescanArray(indexFloat,VFloat,numberElementV);

	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaDeviceSynchronize prescanArray failed",cudaStatus);
		goto Error;
	}

	

	kernelCastingFloat2UnsignedInt<<<grid,block>>>(index,indexFloat,numberElementV);


	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaDeviceSynchronize kernelCastingFloat2UnsignedInt() failed",cudaStatus);
		goto Error;
	}

	/*printf("\n Scan Result int: ");
	kernelPrintf<<<grid,block>>>(index,numberElementV);

	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaDeviceSynchronize kernelPrintf failed",cudaStatus);
		goto Error;
	}*/



Error:
	//cudaFree(VFloat);
	//cudaFree(indexFloat);
	//cudaFree(index);
	//cudaFree(V);
	return cudaStatus;
}