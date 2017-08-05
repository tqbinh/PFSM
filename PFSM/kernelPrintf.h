#include "castingIntToFloat.h"
#include <math_functions.h>
#include "device_functions.h"
#include <math.h>
#include <device_types.h>
#include <device_functions_decls.h>
#include <device_launch_parameters.h>
#include <deviceaccess.h>
#include <math_functions.h>


__global__ void kernelCastingUnsignedInt2Float(float* d_out,unsigned int* d_in,unsigned int n){
	int i=blockIdx.x*blockDim.x + threadIdx.x;
	if(i<n){
		d_out[i]= __uint2float_rn(d_in[i]);	
	}
}

__global__ void kernelCastingInt2Float(float* d_out,int* d_in,unsigned int n){
	int i=blockIdx.x*blockDim.x + threadIdx.x;
	if(i<n){
		d_out[i]= __int2float_rd(d_in[i]);
	}

}

__global__ void kernelCastingFloat2Int(int* d_out,float* d_in,unsigned int n){
	int i=blockIdx.x*blockDim.x + threadIdx.x;
	if(i<n){
		d_out[i]=__float2int_rd(d_in[i]);
	}
}


__global__ void kernelCastingFloat2UnsignedInt(unsigned int* d_out,float* d_in,unsigned int n){
	int i=blockIdx.x*blockDim.x + threadIdx.x;
	if(i<n){
		d_out[i]=__float2uint_rd(d_in[i]);
	}
}
