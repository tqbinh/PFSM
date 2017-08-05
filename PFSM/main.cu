#pragma once
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



int main(int argc, char * const  argv[]){	
	StopWatchWin timer;

#pragma region "load database"

	std::ofstream fout("result.txt", std::ios_base::app | std::ios_base::out);
	
	timer.start();
	PMS pms;
	pms.os=&fout;
	pms.prepareDataBase(); //chuẩn bị dữ liệu
	timer.stop();
	pms.printdb(); //hiển thị dữ liệu
	
	printf("\n\n**===-------------------------------------------------===**\n");
    printf("Loading data...\n");
	printf("Processing time: %f (ms)\n", timer.getTime());
	hTime=timer.getTime();
	timer.reset();

#pragma endregion "end load database"

	FUNCHECK(pms.extractAllEdgeInDB()); //Từ CSDL đã nạp vào device, trích tất cả các cạnh trong CSDL song song
	pms.displayArrExtension(pms.hExtension.at(0).dExtension,pms.hExtension.at(0).noElem); //Những cạnh này được xem như là một mở rộng của pattern P

	FUNCHECK(pms.getValidExtension()); //Trích các mở rộng hợp lệ (li<lj: chỉ xét cho đơn đồ thị vô hướng)
	
	FUNCHECK(pms.extractUniEdge());

	FUNCHECK(pms.computeSupport()); //Tính độ hộ trợ của cả cạnh trong UniEdge và loại bỏ những mở rộng không thoả minsup

	FUNCHECK(pms.Mining()); //kiểm tra DFS_CODE có phải là min hay không, nếu là min thì ghi kết quả vào file result.txt


	system("pause");

	return 0;
}

