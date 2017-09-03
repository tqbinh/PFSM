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
} \
}

#define CHECK(call) \
{ \
const cudaError_t error = call; \
if (error != cudaSuccess) \
{ \
printf("Error: %s:%d, ", __FILE__, __LINE__); \
printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
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

struct UniEdgek
{
	int noElem;
	UniEdge *dArrUniEdge;
	UniEdgek():noElem(0),dArrUniEdge(0){};
};

struct vecArrUniEdge
{
	int noElem;
	vector<UniEdgek> vUE; //mỗi phần tử của vector vE sẽ quản lý 1 phần tử dArrExt
	vecArrUniEdge():noElem(0),vUE(0){};
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
	vector<int> hArrRMP;
	RMP():noElem(0),hArrRMP(0){};
};

struct EXT
{
	int vi,vj,li,lij,lj; /*DFS code của cạnh mở rộng, Ở đây giá trị của (vi,vj) tuỳ thuộc vào 2 thông tin: Mở rộng là backward hay forward và mở rộng tử đỉnh nào trên RMPath. Nếu là mở rộng forward thì cần phải biết maxid của DFS_CODE hiện tại */
	int vgi,vgj;	/* globalId vertex của cạnh mở rộng*/
	//Hai thông tin bên dưới cho biết cạnh mở rộng từ embedding nào.
	//int posColumn; /* vị trí của embedding column trong mảng các embedding Q*/
	int posRow; //vị trí của embedding trong cột Q
	EXT():vi(0),vj(0),li(0),lij(0),lj(0),vgi(0),vgj(0),posRow(0){};
};

struct EXTk
{
	int noElem; //Mỗi EXTk có bao nhiêu phần tử EXT
	EXT *dArrExt;
	EXTk():noElem(0),dArrExt(0){};
};

struct vecArrEXT
{
	int noElem;
	vector<EXTk> vE; //mỗi phần tử của vector vE sẽ quản lý 1 phần tử dArrExt
	vecArrEXT():noElem(0),vE(0){};
};

struct UniEdgeStatisfyMinSup
{
	int noElem;
	UniEdge *dArrUniEdge;
	int *hArrSupport;
	UniEdgeStatisfyMinSup():noElem(0),dArrUniEdge(0),hArrSupport(0){};
};

struct vecArrUniEdgeStatisfyMinSup
{
	int noElem;
	vector<UniEdgeStatisfyMinSup> vecUES;
	vecArrUniEdgeStatisfyMinSup():noElem(0),vecUES(0){};
};


struct V
{
	int noElem;
	int *valid;
	int *backward;
	V():noElem(0),valid(0),backward(0){};
};

struct ptrArrEmbedding
{
	int noElem; //số lượng embedding column
	int noElemEmbedding; //số lượng phần tử ở cột cuối cùng
	Embedding **dArrPointerEmbedding;
	ptrArrEmbedding():noElem(0),noElemEmbedding(0),dArrPointerEmbedding(0){};
};

struct listVer
{
	int noElem;
	int *dListVer;
	listVer():noElem(0),dListVer(0){};
};


//struct listVer
//{
//	int noElem;
//	int *dListVer;
//	listVer():noElem(0),dListVer(0){}
//};
extern __global__ void kernelFindVidOnRMP(Embedding **dArrPointerEmbedding,int noElemEmbedding,int *rmp,int noElemVerOnRMP,int *dArrVidOnRMP,int step);
extern __global__ void kernelDisplaydArrPointerEmbedding(Embedding **dArrPointerEmbedding,int noElemEmbeddingCol,int noElemEmbedding);
extern __global__ void kernelSetValueForEmbeddingColumn(EXT *dArrExt,int noElemInArrExt,Embedding *dArrQ,int *dM,int *dMScanResult);
extern __global__ void kernelMarkEXT(const EXT *d_ValidExtension,int noElem_d_ValidExtension,int *dV,int li,int lij,int lj);
extern __global__ void kernelFilldF(UniEdge *dArrUniEdge,int pos,EXT *dArrExt,int noElemdArrExt,int *dArrBoundaryScanResult,float *dF);
extern __global__ void kernelfindBoundary(EXT *dArrExt, int noElemdArrExt, int *dArrBoundary,unsigned int maxOfVer);
//extern __global__ void kernelFindValidFBExtension(Embedding **dArrPointerEmbedding,int noElem_dArrPointerEmbedding,int noElem_Embedding,int *d_O,int *d_LO,int *d_N,int *d_LN,float *dArrDegreeOfVid,int maxDegreeOfVer,V *dArrV,EXT *dArrExtension,int *listOfVer,int minLabel,int maxId,int fromRMP, int *dArrVidOnRMP,int segdArrVidOnRMP,int *fromPosCol);
extern __global__ void	kernelExtractFromListVer(int *listVer,int from,int noElemEmbedding,int *temp);
extern __global__ void kernelFindListVer(Embedding **dArrPointerEmbedding,int noElemEmbedding,int *rmp,int noElemVerOnRMP,int *listVer);
extern __global__ void kernelPrintdArrPointerEmbedding(Embedding **dArrPointerEmbedding,int noElem,int sizeArr);
extern __global__ void	kernelGetPointerdArrEmbedding(Embedding *dArrEmbedding,Embedding **dArrPointerEmbedding,int idx);
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
extern __global__ void kernelCalDegreeOfVid(int *listOfVer,int *d_O, int numberOfElementd_O,int noElem_Embedding,int numberOfElementd_N,unsigned int maxOfVer,float *dArrDegreeOfVid);
extern __global__ void find_maximum_kernel(float *array, float *max, int *mutex, unsigned int n);
extern __global__ void kernelFindValidForwardExtension(Embedding **dArrPointerEmbedding,int noElem_dArrPointerEmbedding,int noElem_Embedding,int *d_O,int *d_LO,int *d_N,int *d_LN,float *dArrDegreeOfVid,int maxDegreeOfVer,EXT *dArrExtension,int *listOfVer,int minLabel,int maxId,int fromRMP,int *dArrV_valid,int *dArrV_backward);
extern 	__global__ void printdArrUniEdge(UniEdge *dArrUniEdge,int i);
extern __global__ void	kernelGetvivj(EXT *dArrEXT,int noElemdArrEXT,int li,int lij,int lj,int *dvi,int *dvj);
extern __global__ void kernelGetLastElementEXT(EXT *inputArray,int noEleInputArray,int *value,unsigned int maxOfVer);
extern __global__ void kernelGetGraphIdContainEmbeddingv2(int li,int lij,int lj,EXT *d_ValidExtension,int noElem_d_ValidExtension,int *dV,unsigned int maxOfVer);


extern cudaError_t getLastElementEXT(EXT *inputArray,int numberElementOfInputArray,int &outputValue,unsigned int maxOfVer);
extern cudaError_t ADM(int *&devicePointer,size_t nBytes);
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
extern int displayDeviceEXT(EXT *dArrEXT,int noElemdArrEXT);


class PMS:public gSpan
{
public:	
	int currentColEmbedding;
	int Level;
	int idxLevel;
	vector<DB> hdb;
	vector<arrExtension> hExtension;
	vector<arrExtension> hValidExtension;
	vector<arrUniEdge> hUniEdge;
	vector<arrUniEdgeSatisfyMinSup> hUniEdgeSatisfyMinsup;
	vector<vecArrUniEdgeStatisfyMinSup> hLevelUniEdgeSatisfyMinsup;

	vector<EmbeddingColumn> hEmbedding; //Mỗi phần tử của vector là một Embedding column
	//Embedding **dArrPointerEmbedding;
	vector<ptrArrEmbedding> hLevelPtrEmbedding;
	vector<ptrArrEmbedding> hLevelPtrEmbeddingv2;

	 
	vector<vecArrUniEdge> hLevelUniEdge;
	vector<listVer> hListVer;

	vector<RMP> hRMP;
	vector<RMP> hRMPv2;
	
	//vector<EXTk> hEXTk; //Có bao nhiêu EXTk
	vector<vecArrEXT> hLevelEXT; //quản lý EXTk theo Level, mỗi một Level là 1 lần gọi đệ quy FSMining function
	
	PMS();
	~PMS();
	unsigned int Lv;
	unsigned int Le;
	unsigned int maxOfVer;
	unsigned int numberOfGraph;
	int minLabel;
	int maxId;

	//std::ostream* os;
	//std::ofstream fos; //fos là pointer trỏ tới tập tin /result.graph


public:
	void prepareDataBase();
	void displayArray(int*, const unsigned int);
	int displayDeviceArr(int *,int);
	int displayDeviceArr(float*,int);
	//void displayEmbeddingColumn(EmbeddingColumn);
	int displayArrExtension(Extension*, int);
	int displayArrUniEdge(UniEdge*,int);
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
	int Miningv2(int,UniEdge*,int*,EXT*,int,int);
	int Miningv3(int,UniEdge*,int*,EXT*,int,int);

	int getGraphIdContainEmbedding(UniEdge,int*&,int&);	
	int buildFirstEmbedding(UniEdge);
	int buildRMP();
	int FSMining();
	int FSMiningv2();
	int forwardExtension(int,int*,int,int);
	int findMaxDegreeOfVer(int*,int&,float*&,int);
	int findDegreeOfVer(int*,float*&,int);
	int extractValidExtensionTodExt(EXT*,V*,int,int);
	int computeSupportv2(EXT*,int,UniEdge*,int,int&,UniEdge*&,int*&);
	int findBoundary(EXT*,int,int *&);
	int extractUniEdgeSatisfyMinsupV2(int *,UniEdge*,int ,unsigned int ,int &,UniEdge *&,int *&);
	int getvivj(EXT*,int,int,int,int,int&,int&);
	int getGraphIdContainEmbeddingv2(UniEdge ,int *&,int &,EXT *,int );
	int extendEmbedding(UniEdge ue,int idxExt);
	int updateRMP();
	int displaydArrPointerEmbedding(Embedding** ,int noElemEmbeddingCol,int noElemEmbedding);
};

