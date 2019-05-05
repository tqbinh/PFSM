#pragma once

#include "moderngpu.cuh"		// Include all MGPU kernels.
#include <typeinfo>
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
using namespace mgpu;

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


ContextPtr ctx;




int main(int argc, char** argv){
	int status=0;
	cudaDeviceReset();
	ctx = CreateCudaDevice(argc, argv, true);
	cout << typeid(ctx).name() << endl;

	//device_info();
	//cdactx=*ctx;
	StopWatchWin timer;
	//exit(0);
	//system("pause");
#pragma region "load database"
	//Open file result.txt to write append
	std::ofstream fout("result.txt", std::ios_base::app | std::ios_base::out);
	
	timer.start();
	PMS pms; //Tạo đối tượng PMS.
	pms.os=&fout;
	FUNCHECK(status=pms.prepareDataBase()); //chuẩn bị dữ liệu
	if(status!=0){
		cout<<endl<<"prepareDataBase function failed"<<endl;
		exit(1);
	}

	timer.stop();
	//pms.printdb(); //hiển thị dữ liệu
	
	std::printf("\n\n**===-------------------------------------------------===**\n");
	std::printf("Loading data...\n");
	std::printf("Processing time: %f (ms)\n", timer.getTime());//Processing time: 6595
	hTime=timer.getTime();
	timer.reset();

#pragma endregion "end load database"

	FUNCHECK(pms.extractAllEdgeInDB()); //Từ CSDL đã nạp vào device, trích tất cả các cạnh trong CSDL song song
	//pms.displayArrExtension(pms.hExtension.at(0).dExtension,pms.hExtension.at(0).noElem); //Những cạnh này được xem như là một mở rộng của pattern P. Bước này chỉ đơn thuần là xây dựng DFS Code cho các cạnh trong đồ thị.
	timer.start();
	FUNCHECK(pms.getValidExtension_pure()); //Trích các mở rộng hợp lệ (li<lj: chỉ xét cho đơn đồ thị vô hướng) ==> Notes: Cần phải xét cho trường hợp đa đồ thị vô hướng và có hướng
	timer.stop();
	std::printf("\n\n**===-------------------------------------------------===**\n");
	std::printf("getValidExtension_pure\n");
	std::printf("Processing time: %f (ms)\n", timer.getTime());//Processing time: 8.730469 (ms)
	hTime=timer.getTime();
	timer.reset();
	timer.start();
	FUNCHECK(pms.extractUniEdge());
	timer.stop();
	std::printf("\n\n**===-------------------------------------------------===**\n");
	std::printf("extractUniEdge\n");
	std::printf("Processing time: %f (ms)\n", timer.getTime());//Processing time: 1.730469 (ms)
	hTime=timer.getTime();
	timer.reset();
	timer.start();
	FUNCHECK(pms.computeSupport()); //Tính độ hộ trợ của cả cạnh trong UniEdge và loại bỏ những mở rộng không thoả minsup
	//Đến đây, chúng ta đã thu thập được các mở rộng một cạnh thoả minsup (hUniEdgeSatisfyMinSup)
	//
	//FUNCHECK(pms.Mining()); //kiểm tra DFS_CODE có phải là min hay không, nếu là min thì ghi kết quả vào file result.txt, và xây dựng Embedding Columns
	timer.stop();
	std::printf("\n\n**===-------------------------------------------------===**\n");
	std::printf("computeSupport\n");
	std::printf("Processing time: %f (ms)\n", timer.getTime()/1000);//Processing time: 15.730469 (s)
	hTime=timer.getTime();
	timer.reset();
	
	timer.start();
	//Duyệt qua các cạnh thoả minsup để xây dựng:
	//DFSCODE, hEmbedding, hLevelPtrEmbedding, hLevelListVerRMP và hLevelRMP để chuẩn bị khai thác.
	//FUNCHECK(pms.initialize());
	//Trích các mở rộng thoả minDFS_CODE ban đầu
	FUNCHECK(pms.MiningDeeper(pms.hLevelEXT.at(0).vE.at(0), pms.hLevelUniEdgeSatisfyMinsup.at(0).vecUES.at(0)));
	timer.stop();
	std::printf("\n\n**===-------------------------------------------------===**\n");
	std::printf("Mining()\n");
	std::printf("Processing time: %f (ms)\n", timer.getTime());//Processing time:  (ms)
	hTime=timer.getTime();
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