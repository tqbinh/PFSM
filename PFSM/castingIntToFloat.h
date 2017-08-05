#pragma once
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_functions_decls.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
using namespace std;
extern inline __global__ void kernelCastingUnsignedInt2Float(float* d_out,unsigned int* d_in,unsigned int n);
extern  inline __global__ void kernelCastingInt2Float(float*,int*,unsigned int);
extern  inline __global__ void kernelCastingFloat2Int(int*,float*,unsigned int);
extern inline __global__ void kernelCastingFloat2UnsignedInt(unsigned int* d_out,float* d_in,unsigned int n);