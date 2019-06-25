#pragma once
// Include all MGPU kernels.
#include "moderngpu.cuh"
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

using namespace std;
using namespace mgpu;

ContextPtr ctx;

void device_info()
{
	int devCount = 0;
	cudaGetDeviceCount(&devCount);
	cout<<endl<<"So luong device:"<<devCount<<endl;
	cudaDeviceProp devProp;
	for (int i = 0; i < devCount; i++)
	{
		cudaGetDeviceProperties(&devProp,0);
		cout<<endl<<"name: "<<devProp.name<<endl;
		cout<<endl<<"major: "<<devProp.major<<endl;
		cout<<endl<<"minor: "<<devProp.minor<<endl;
		cout<<endl<<"totalGlobalMem: "<<devProp.totalGlobalMem<<endl;
		cout<<endl<<"totalConstMem: "<<devProp.totalConstMem<<endl;
		cout<<endl<<"maxGridSize x,y,z,all: "<<devProp.maxGridSize[0]<<","<< \
			devProp.maxGridSize[1]<<","<<devProp.maxGridSize[2]<<","<<devProp.maxGridSize[3]<<endl;
		cout<<endl<<"maxThreadsDim x,y,z,all: "<<devProp.maxThreadsDim[0]<<","<< \
			devProp.maxThreadsDim[1]<<","<<devProp.maxThreadsDim[2]<<","<<devProp.maxThreadsDim[3]<<endl;
		cout<<endl<<"maxThreadsPerBlock(so luong tieu trinh toi da 1 block): "<<devProp.maxThreadsPerBlock<<endl;
		cout<<endl<<"devProp.maxThreadsPerMultiProcessor(so luong tieu trinh toi da 1 SM):"<<devProp.maxThreadsPerMultiProcessor<<endl;
		cout<<endl<<"sharedMemPerBlock (Dung luong shareMem cua 1 Block) (KB): "<<devProp.sharedMemPerBlock<<endl;
		cout<<endl<<"multiProcessorCount(so luong SM): "<<devProp.multiProcessorCount<<endl;
		cout<<endl<<"regsPerBlock: "<<devProp.regsPerBlock<<endl;
		cout<<endl<<"warpSize: "<<devProp.warpSize<<endl;
		cout<<endl<<"concurrentKernels: "<<devProp.concurrentKernels<<endl;
	}
	system("pause");
	exit(0);
}
int main(int argc, char** argv){
	//int status=0;
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
	pms.prepareDataBase(); //chuẩn bị dữ liệu
	timer.stop();
	//pms.printdb(); //hiển thị dữ liệu
	
	std::printf("\n\n**===-------------------------------------------------===**\n");
	std::printf("Loading data...\n");
	std::printf("Processing time: %f (ms)\n", timer.getTime());//Processing time: 6595
	hTime=timer.getTime();
	timer.reset();

#pragma endregion "end load database"

	pms.extractAllEdgeInDB(); //Từ CSDL đã nạp vào device, trích tất cả các cạnh trong CSDL song song
	timer.start();
	//Trích các mở rộng hợp lệ (li<lj: chỉ xét cho đơn đồ thị vô hướng) \
	//==> Notes: Cần phải xét cho trường hợp đa đồ thị vô hướng và có hướng
	pms.getValidExtension_pure(); 
	timer.stop();
	std::printf("\n\n**===-------------------------------------------------===**\n");
	std::printf("getValidExtension_pure\n");
	std::printf("Processing time: %f (ms)\n", timer.getTime());//Processing time: 8.730469 (ms)
	hTime=timer.getTime();
	timer.reset();
	timer.start();
	pms.extractUniEdge();
	timer.stop();
	std::printf("\n\n**===-------------------------------------------------===**\n");
	std::printf("extractUniEdge\n");
	std::printf("Processing time: %f (ms)\n", timer.getTime());//Processing time: 1.730469 (ms)
	hTime=timer.getTime();
	timer.reset();
	timer.start();
	pms.computeSupport(); //Tính độ hộ trợ của cả cạnh trong UniEdge và loại bỏ những mở rộng không thoả minsup
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
	pms.MiningDeeper(pms.hLevelEXT.at(0).vE.at(0), pms.hLevelUniEdgeSatisfyMinsup.at(0).vecUES.at(0));
	timer.stop();
	std::printf("\n\n**===-------------------------------------------------===**\n");
	std::printf("MiningDeeper()\n");
	std::printf("Processing time: %f (ms)\n", timer.getTime());//Processing time:  (ms)
	hTime=timer.getTime();
	//system("pause");

	return 0;
}