#pragma once
#include "gspan.cuh"
#include <stdio.h>
#include <vector>
#include "helper_timer.h"
#include "scan_largearray_kernel.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include "scanV.h"
#include "reduction.h"

#define blocksize 512

using namespace std;

#define FUNCHECK(call) \
{ \
const int error = call; \
if (error != 0) \
{ \
printf("Error: %s:%d, ", __FILE__, __LINE__); \
printf("code:%d, reason: %s\n", error, "Function failed"); \
exit(1); \
} \
}

#define CHECK(call) \
{ \
const cudaError_t error = call; \
if (error != cudaSuccess) \
{ \
printf("Error: %s:%d, ", __FILE__, __LINE__); \
printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
printf("go to next line"); \
} \
}


extern float hTime;
extern float dTime;


struct DB
{
	int noElemdO;
	int noElemdN;
	int* dO;
	int* dLO; 
	int* dN;
	int* dLN;
	DB():noElemdO(0),noElemdN(0),dO(0),dLO(0),dN(0),dLN(0){};
};

struct Extension
{
	int vi,vj,li,lij,lj; //DFS_code của cạnh mở rộng
	int vgi,vgj; //global id của đỉnh
	//struct_Embedding *d_rowpointer;//lưu trữ pointer trỏ đến embedding mà nó mở rộng.
	Extension():vi(0),vj(0),li(0),lij(0),lj(0),vgi(0),vgj(0){};//khởi tạo cấu trúc
};

struct arrExtension
{
	int noElem;
	Extension *dExtension;
	arrExtension():noElem(0),dExtension(0){};
};

struct UniEdge
{	
	int li;
	int lij;
	int lj;
	UniEdge():li(-1),lij(-1),lj(-1){};
};

struct arrUniEdge
{
	int noElem;
	UniEdge *dUniEdge;
	arrUniEdge():noElem(0),dUniEdge(0){};
};

struct arrUniEdgeSatisfyMinSup
{
	int noElem;
	UniEdge* dUniEdge;
	int *hArrSup;
};

struct Embedding
{
	int idx;
	int vid;
	Embedding():idx(0),vid(0){};
};

struct EmbeddingColumn
{
	int noElem; //Số lượng phần tử mảng dArrEmbedding
	int prevCol; //prevCol, dùng để xây dựng RMPath
	Embedding *dArrEmbedding;
	EmbeddingColumn():noElem(0),prevCol(0),dArrEmbedding(0){};
};

struct RMP
{
	int noElem;
	int *hArrRMP;
	int minLabel;
	int maxId;
	RMP():noElem(0),hArrRMP(0),minLabel(0),maxId(0){};
};


extern __global__ void kernelPrintdArr(int *deviceArray,unsigned int noElem);
extern __global__ void kernelPrintdArr(int *dArr,int noElem);
extern __global__ void kernelPrintdArr(float *dArr,int noElem);
extern __global__ void kernelCountNumberOfLabelVertex(int *d_LO,int *d_Lv,unsigned int sizeOfArrayLO);
extern __global__ void kernelGetAndStoreExtension(int *d_O,int *d_LO,unsigned int numberOfElementd_O,int *d_N,int *d_LN,unsigned int numberOfElementd_N,Extension *d_Extension);
extern __global__ void kernelPrintExtention(Extension *d_Extension, int n);
extern __global__ void	kernelValidEdge(Extension *d_Extension,int *dV,int numberElementd_Extension);
extern __global__ void kernelGetSize(int *dV,int *dVScanResult,int noElem,int *size);
extern __global__ void kernelExtractValidExtension(Extension *d_Extension,int *dV,int *dVScanResult,int numberElementd_Extension,Extension *d_ValidExtension);
extern __global__ void kernelMarkLabelEdge(Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,unsigned int Lv,unsigned int Le,int *d_allPossibleExtension);
extern __global__ void kernelCalcLabelAndStoreUniqueExtension(int *d_allPossibleExtension,int *d_allPossibleExtensionScanResult,unsigned int noElem_allPossibleExtension,UniEdge *d_UniqueExtension,unsigned int Le,unsigned int Lv);
extern __global__ void kernelCalcBoundary(Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,int *dB,unsigned int maxOfVer);
extern __global__ void kernelGetLastElement(int *dScanResult,unsigned int noElem,int *output);
extern __global__ void kernelSetValuedF(UniEdge *dUniEdge,int noElemdUniEdge,Extension *dValidExtension,int noElemdValidExtension,int *dBScanResult,float *dF,int noElemF);
extern __global__ void kernelCopyFromdFtoTempF(int *d_F,int *tempF,int from,int noElemNeedToCopy);
extern __global__ void	kernelMarkUniEdgeSatisfyMinsup(int *dResultSup,int noElemUniEdge,int *dV,unsigned int minsup);
extern __global__ void	kernelExtractUniEdgeSatifyMinsup(UniEdge *dUniEdge,int *dV,int *dVScanResult,int noElemUniEdge,UniEdge *dUniEdgeSatisfyMinsup,int *dSup,int *dResultSup);
extern __global__ void kernelGetGraphIdContainEmbedding(int li,int lij,int lj,Extension *d_ValidExtension,int noElem_d_ValidExtension,int *d_arr_graphIdContainEmbedding,unsigned int maxOfVer);
extern __global__ void kernelGetGraph(int *dV,int noElemdV,int *d_kq,int *dVScanResult);
extern __global__ void kernelMarkExtension(const Extension *d_ValidExtension, int noElem_d_ValidExtension,int *dV,int li,int lij,int lj);
extern __global__ void kernelSetValueForFirstTwoEmbeddingColumn(const Extension *d_ValidExtension,int noElem_d_ValidExtension,Embedding *dQ1,Embedding *dQ2,int *d_scanResult,int li,int lij,int lj);
extern __global__ void	kernelPrintEmbedding(Embedding *dArrEmbedding,int noElem);



extern void sumUntilReachZero(int *h_Lv,unsigned int n,int &result);
extern cudaError_t validEdge(Extension *d_Extension,int *&V,unsigned int numberElementd_Extension);
extern cudaError_t getSizeBaseOnScanResult(int *dV,int *dVScanResult,int noElem,int &output);
extern cudaError_t extractValidExtension(Extension *d_Extension,int *dV,int *dVScanResult, int numberElementd_Extension,Extension *&d_ValidExtension);
extern cudaError_t markLabelEdge(Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,unsigned int Lv,unsigned int Le,int *&d_allPossibleExtension);
extern cudaError_t calcLabelAndStoreUniqueExtension(int *d_allPossibleExtension,int *d_allPossibleExtensionScanResult,unsigned int noElem_allPossibleExtension,UniEdge *&d_UniqueExtension,unsigned int noElem_d_UniqueExtension,unsigned int Le,unsigned int Lv);
extern cudaError_t calcBoundary(Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,int *&dB,unsigned int maxOfVer);
extern cudaError_t getLastElement(int *dScanResult,unsigned int noElem,int &output);
extern cudaError_t calcSupport(UniEdge *dUniEdge,int noElemdUniEdge,Extension *dValidExtension,int noElemdValidExtension,int *dBScanResult,int *dF,int noElemF,int *&hResultSup);
extern cudaError_t getLastElementExtension(Extension* inputArray,unsigned int numberElementOfInputArray,int &outputValue,unsigned int maxOfVer);
extern void  myScanV(int *dArrInput,int noElem,int *&dResult);

class PMS:public gSpan
{
public:	
	vector<DB> hdb;
	vector<arrExtension> hExtension;
	vector<arrExtension> hValidExtension;
	vector<arrUniEdge> hUniEdge;
	vector<arrUniEdgeSatisfyMinSup> hUniEdgeSatisfyMinsup;

	vector<EmbeddingColumn> hEmbedding; //Mỗi phần tử của vector là một Embedding column
	

	
	PMS();
	~PMS();
	unsigned int Lv;
	unsigned int Le;
	unsigned int maxOfVer;
	unsigned int numberOfGraph;
	//std::ostream* os;
	//std::ofstream fos; //fos là pointer trỏ tới tập tin /result.graph


public:
	void prepareDataBase();
	void displayArray(int*, const unsigned int);
	int displayDeviceArr(int *,int);
	int displayDeviceArr(float*,int);
	//void displayEmbeddingColumn(EmbeddingColumn);
	void displayArrExtension(Extension*, int);
	void displayArrUniEdge(UniEdge*,int);
	bool checkArray(int*, int*, const int);
	void printdb();
	int countNumberOfDifferentValue(int*,unsigned int,unsigned int&);
	int extractAllEdgeInDB();
	int getAndStoreExtension(Extension*&);
	int getValidExtension();
	int extractUniEdge();
	int computeSupport();
	int extractUniEdgeSatisfyMinsup(int*,int,unsigned int);
	int Mining();
	int getGraphIdContainEmbedding(UniEdge,int*&,int&);	
	int buildFirstEmbedding(UniEdge);
};

