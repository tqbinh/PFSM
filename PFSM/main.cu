#pragma once

#include "moderngpu.cuh"		// Include all MGPU kernels.

using namespace mgpu;


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <vector>
#include <iostream>
#include <string>
#include <map>
#include "conio.h"
#include <fstream>
#include "pms.cuh"
//#include "kernelPrintf.h"
//#include "kernelCountLabelInGraphDB.h"
//#include "kernelMarkInvalidVertex.h"
//#include "markInvalidVertex.h"
//#include "checkArray.h"
//#include "displayArray.h"
//#include "checkDataBetweenHostAndGPU.h"
//#include "access_d_LO_from_idx_of_d_O.h"
//#include "countNumberOfLabelVetex.h"
//#include "countNumberOfEdgeLabel.h"
//#include "extractUniqueEdge.h"
//#include "ExtensionStructure.h"
//#include "getAndStoreExtension.h"
//#include "validEdge.h"
//#include "scanV.h"
//#include "getLastElement.h"
//#include "getValidExtension.h"
//#include "getUniqueExtension.h"
//#include "calcLabelAndStoreUniqueExtension.h"
//#include "calcBoundary.h"
//#include "calcSupport.h"
//#include "getSatisfyEdge.h"
//#include "header.h"
//
//
//#include "helper_timer.h"
using namespace std;


//
//#define CHECK(call) \
//{ \
//const cudaError_t error = call; \
//if (error != cudaSuccess) \
//{ \
//printf("Error: %s:%d, ", __FILE__, __LINE__); \
//printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
//exit(1); \
//} \
//}



int main(int argc, char** argv){	
	int status=0;
	ContextPtr ctx = CreateCudaDevice(argc, argv, true);
	StopWatchWin timer;
	
	system("pause");
#pragma region "load database"

	std::ofstream fout("result.txt", std::ios_base::app | std::ios_base::out);
	
	timer.start();
	PMS pms;
	pms.os=&fout;
	FUNCHECK(status=pms.prepareDataBase()); //chuẩn bị dữ liệu
	if(status!=0){
		cout<<endl<<"prepareDataBase function failed"<<endl;
		exit(1);
	}

	timer.stop();
	pms.printdb(); //hiển thị dữ liệu
	
	std::printf("\n\n**===-------------------------------------------------===**\n");
    std::printf("Loading data...\n");
	std::printf("Processing time: %f (ms)\n", timer.getTime());
	hTime=timer.getTime();
	timer.reset();

#pragma endregion "end load database"

	FUNCHECK(pms.extractAllEdgeInDB()); //Từ CSDL đã nạp vào device, trích tất cả các cạnh trong CSDL song song
	pms.displayArrExtension(pms.hExtension.at(0).dExtension,pms.hExtension.at(0).noElem); //Những cạnh này được xem như là một mở rộng của pattern P

	FUNCHECK(pms.getValidExtension_pure()); //Trích các mở rộng hợp lệ (li<lj: chỉ xét cho đơn đồ thị vô hướng)
	
	FUNCHECK(pms.extractUniEdge());

	FUNCHECK(pms.computeSupport()); //Tính độ hộ trợ của cả cạnh trong UniEdge và loại bỏ những mở rộng không thoả minsup
	//Đến đây, chúng ta đã thu thập được các mở rộng một cạnh thoả minsup (hUniEdgeSatisfyMinSup)
	//
	//FUNCHECK(pms.Mining()); //kiểm tra DFS_CODE có phải là min hay không, nếu là min thì ghi kết quả vào file result.txt, và xây dựng Embedding Columns
	FUNCHECK(pms.initialize()); //Duyệt qua các cạnh thoả minsup để xây dựng DFSCODE, hEmbedding, hLevelPtrEmbedding, hLevelListVerRMP và hLevelRMP để chuẩn bị khai thác.

	system("pause");

	return 0;
}




//int main(int argc, char** argv) 
//{
//    ContextPtr context = CreateCudaDevice(argc, argv, true);
//
//   int noElem = 5;
//   int* ptr = (int*)malloc(sizeof(int)*noElem);
//   for (int i = 0; i < noElem; i++)
//   {
//	   ptr[i]=i;
//	   cout<<ptr[i]<<" ";
//   }
//   cout<<endl;
//   int *p=nullptr;
//   cudaMalloc((void**)&p,sizeof(int)*noElem);
//   cudaMemcpy(p,ptr,noElem*sizeof(int),cudaMemcpyHostToDevice);
//   cout<<"Input data"<<endl;
//   kernelPrintdArr<<<1,100>>>(p,noElem);
//   cudaDeviceSynchronize();
//   cout<<endl;
//  //// int result = Reduce(p, noElem, *context);
//  //// printf("Reduction total: %d\n\n", result);
//   int result=0;
//   //ScanExc(p, noElem, &result, *context);
//   ScanExc(p, noElem, *context);
////   PrintArray(*data, "%4d", 10);
//    kernelPrintdArr<<<1,100>>>(p,noElem);
//    cudaDeviceSynchronize();
//    //printf("Exclusive scan:\n");
//    //printf("Scan total: %d\n", result);
//
//	cudaFree(p);
//
//    //// Run an exclusive scan.
//    //ScanExc(data->get(), N, &total, context);
//    //printf("Exclusive scan:\n");
//    //PrintArray(*data, "%4d", 10);
//    //printf("Scan total: %d\n", total);
//
//	_getch();
//    return 0;
//}