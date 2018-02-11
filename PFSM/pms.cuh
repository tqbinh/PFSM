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

struct Extension //Thông tin của một cạnh thuộc CSDL
{
	int vi,vj,li,lij,lj; //DFS_code của cạnh mở rộng
	int vgi,vgj; //global id của đỉnh
	//struct_Embedding *d_rowpointer;//lưu trữ pointer trỏ đến embedding mà nó mở rộng.
	Extension():vi(0),vj(0),li(0),lij(0),lj(0),vgi(0),vgj(0){};//khởi tạo cấu trúc
};

struct arrExtension
{
	int noElem; //Tổng số cạnh của tất cả các đồ thị trong CSDL
	Extension *dExtension; //Mảng liên tục, lưu trữ thông tin của cạnh thuộc CSDL.
	arrExtension():noElem(0),dExtension(0){};
};

struct UniEdge
{	
	int vi;
	int vj;
	int li;
	int lij;
	int lj;
	UniEdge():vi(-1),vj(-1),li(-1),lij(-1),lj(-1){};
	//UniEdge():li(-1),lij(-1),lj(-1){};
};

struct arrUniEdge
{
	int noElem;
	UniEdge *dUniEdge;
	arrUniEdge():noElem(0),dUniEdge(0){};
};

struct UniEdgek
{
	int noElem; //số lượng phần tử của mảng dArrUniEdge
	int firstIndexForwardExtension; //Đây là index của phần tử uniEdge forward đầu tiên trong mảng dArrUniEdge
	int Li; //Nhãn của đỉnh mà từ đó thực hiện mở rộng
	UniEdge *dArrUniEdge;
	UniEdgek():noElem(-1),firstIndexForwardExtension(-1),Li(-1),dArrUniEdge(0){};
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

struct EmbeddingBWCol
{
	int noElem; //Số lượng phần tử mảng dArrEmbedding
	int prevCol; //prevCol, dùng để xây dựng RMPath
	Embedding *dArrEmbedding;
	EmbeddingBWCol():noElem(0),prevCol(0),dArrEmbedding(0){};

};


struct EmbeddingColumn
{
	vector<EmbeddingBWCol> hBackwardEmbedding;
	int noElem; //Số lượng phần tử mảng dArrEmbedding
	int prevCol; //prevCol, dùng để xây dựng RMPath
	Embedding *dArrEmbedding;
	EmbeddingColumn():noElem(0),prevCol(0),dArrEmbedding(0){};
};



struct RMP
{
	int noElem;
	vector<int> hArrRMP;
	int *dRMP;
	RMP():noElem(0),hArrRMP(0),dRMP(0){};
};

struct EXT
{
	int vi,vj,li,lij,lj; /*DFS code của cạnh mở rộng, Ở đây giá trị của (vi,vj) tuỳ thuộc vào 2 thông tin: Mở rộng là backward hay forward và mở rộng tử đỉnh nào trên RMPath. Nếu là mở rộng forward thì cần phải biết maxid của DFS_CODE hiện tại */
	int vgi,vgj;	/* globalId vertex của cạnh mở rộng*/
	//Hai thông tin bên dưới cho biết cạnh mở rộng từ embedding nào. Chúng ta cần phải biết thông tin này để 
	//int posColumn; /* vị trí của embedding column trong mảng các embedding Q*/
	int posRow; //vị trí của embedding trong cột Q
	EXT():vi(0),vj(0),li(0),lij(0),lj(0),vgi(0),vgj(0),posRow(0){};
};

struct EXTk
{
	int noElem; //Số lượng phần tử dArrExt
	EXT *dArrExt;
	EXTk():noElem(0),dArrExt(0){};
};

struct vecArrEXT
{
	int noElem; //Số lượng phần tử vE dựa vào số lượng đỉnh thuộc right most path của Level đang xét. Nếu Level đang xét có 3 đỉnh thuộc right most path thì chúng ta tạo ra 3 phần tử vE, mỗi phần tử vE sẽ lưu trữ các mở rộng hợp lệ của các đỉnh thuộc embedding column đang xét.
	vector<EXTk> vE; //mỗi phần tử của vector vE sẽ quản lý 1 phần tử dArrExt
	vecArrEXT():noElem(0),vE(0){};
};

struct UniEdgeStatisfyMinSup
{
	//Số lượng phần tử của mảng dArrUniEdge
	int noElem;
	//Chứa các mở rộng duy nhất thoả mãn minsup
	UniEdge *dArrUniEdge;
	//Chứa độ hỗ trợ tương ứng với mở rộng duy nhất ở dArrUniEdge
	int *hArrSupport;

	UniEdgeStatisfyMinSup():noElem(0),dArrUniEdge(0),hArrSupport(0){};
};

struct vecArrUniEdgeStatisfyMinSup
{
	//Số lượng phần tử của vecUES
	//Được dùng để resize vector vecUES tại level đang khai thác.
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
	int noElem; //số lượng embedding column. Nên đoi tên là noElemColumn
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
//extern __global__ void	kernelGetRow(int *dV,int *dVScanResult,int noElemdV,int *dArrRow);
extern __global__ void kernelFindValidFBExtensionv3(Embedding **dArrPointerEmbedding,int noElem_dArrPointerEmbedding,int noElem_Embedding,int *d_O,int *d_LO,int *d_N,int *d_LN,float *dArrDegreeOfVid,int maxDegreeOfVer,int *dArrV_valid,int *dArrV_backward,EXT *dArrExtension,int *listOfVer,int minLabel,int maxId,int fromRMP, int *dArrVidOnRMP,int segdArrVidOnRMP,int *rmp,int *dArrVj,int noElemdArrVj);
extern __global__ void kernelFindValidForwardExtensionv3(Embedding **dArrPointerEmbedding,int noElem_dArrPointerEmbedding,int noElem_Embedding,int *d_O,int *d_LO,int *d_N,int *d_LN,float *dArrDegreeOfVid,int maxDegreeOfVer,int *dArrV_valid,int *dArrV_backward,EXT *dArrExtension,int *listOfVer,int minLabel,int maxId,int fromRMP, int *dArrVidOnRMP,int segdArrVidOnRMP,int *rmp);

extern __global__ void kernelSetValuedF_pure(UniEdge *dUniEdge,int noElemdUniEdge,EXT *dValidExtension,int noElemdValidExtension,int *dBScanResult,int *dF,int noElemF);
extern __device__ bool IsVertexOnEmbedding(int vertex,Embedding **dArrPointerEmbedding,int noElem_dArrPointerEmbedding,int row);
extern __global__ void kernelGetFromLabelv3(EXT *dArrExt,int *dFromVi,int *dFromLi);
extern __global__ void kernelCopyResultToUE(UniEdge *fwdArrUniEdge,UniEdge *bwdArrUniEdge,int bwnoElem,UniEdge *uedArrUniEdge,int uenoElem);
extern __global__ void kernelCopyResultToUE(UniEdge *fwdArrUniEdge,UniEdge *uedArrUniEdge,int uenoElem);
extern __global__ void kernelExtractBWEmbeddingRow(Embedding* dArrBWEmbedding,int *dV,int *dVScanResult,int noElemdV,Embedding *dArrEmbedding);
extern __global__ void	kernelExtractRowFromEXT(EXT *dArrExt,int noElemdArrExt,int *dV,int vj);
extern __global__ void kernelGetGraphIdContainEmbeddingBW(int vj,EXT *d_ValidExtension,int noElem_d_ValidExtension,int *dV,unsigned int maxOfVer);
extern __global__ void	kernelExtractUniEdgeSatifyMinsupV3(UniEdge *dUniEdge,int *dV,int *dVScanResult,int noElemUniEdge,UniEdge *dUniEdgeSatisfyMinsup,int *dSup,int *dResultSup);
extern __global__ void kernelFilldArrUniEdgev2(int *dArrAllPossibleExtension,int *dArrAllPossibleExtensionScanResult,int noElem_dArrAllPossibleExtension,UniEdge *dArrUniEdge,int Lv,int *dFromLi,int *dFromVi,int maxId);
extern __global__ void kernelGetFromLabelv2(EXT *dArrExt,int noElem,int *dFromVi,int *dFromLi);
extern __global__ void kernelextractValidBWExtension(UniEdge *dsrcUniEdge,UniEdge *ddstUniEdge,int noElem,int *dAllPossibleExtension,int *dAllPossibleExtensionScanResult);
extern __global__ void kernelextractAllBWExtension(EXT *dArrExt,int noElemdArrExt,UniEdge* dArrUniEdge,int *dAllPossibelExtension);
//extern __global__ void kernelmarkValidBackwardEdge_LastExt(EXT* dArrExt, int noElemdArrExt,unsigned int Lv,int *dAllPossibleExtension);
extern __global__ void kernelmarkValidForwardEdge_LastExt(EXT* dArrExt, int noElemdArrExt,unsigned int Lv,int *dAllPossibleExtension);
extern __global__ void kernelFindVidOnRMP(Embedding **dArrPointerEmbedding,int noElemEmbedding,int *rmp,int noElemVerOnRMP,int *dArrVidOnRMP,int step);
extern __global__ void kernelFindVidOnRMPv2(Embedding **dArrPointerEmbedding,int noElemEmbedding,int *rmp,int noElemVerOnRMP,int *dArrVidOnRMP,int step);

extern __global__ void kernelDisplaydArrPointerEmbedding(Embedding **dArrPointerEmbedding,int noElemEmbeddingCol,int noElemEmbedding);
extern __global__ void kernelSetValueForEmbeddingColumn(EXT *dArrExt,int noElemInArrExt,Embedding *dArrQ,int *dM,int *dMScanResult);
extern __global__ void kernelMarkEXT(const EXT *d_ValidExtension,int noElem_d_ValidExtension,int *dV,int li,int lij,int lj);
extern __global__ void kernelFilldF(UniEdge *dArrUniEdge,int pos,EXT *dArrExt,int noElemdArrExt,int *dArrBoundaryScanResult,float *dF);
extern __global__ void kernelfindBoundary(EXT *dArrExt, int noElemdArrExt, int *dArrBoundary,unsigned int maxOfVer);
extern __global__ void kernelFilldFbw(UniEdge *dArrUniEdge,int pos,EXT *dArrExt,int noElemdArrExt,int *dArrBoundaryScanResult,float *dF);
extern __global__ void kernelFindValidFBExtension(Embedding **dArrPointerEmbedding,int noElem_dArrPointerEmbedding,int noElem_Embedding,int *d_O,int *d_LO,int *d_N,int *d_LN,float *dArrDegreeOfVid,int maxDegreeOfVer,int *dArrV_valid,int *dArrV_backward,EXT *dArrExtension,int *listOfVer,int minLabel,int maxId,int fromRMP, int *dArrVidOnRMP,int segdArrVidOnRMP,int *rmp);
extern __global__ void kernelFindValidFBExtensionv2(Embedding **dArrPointerEmbedding,int noElem_dArrPointerEmbedding,int noElem_Embedding,int *d_O,int *d_LO,int *d_N,int *d_LN,float *dArrDegreeOfVid,int maxDegreeOfVer,int *dArrV_valid,int *dArrV_backward,EXT *dArrExtension,int *listOfVer,int minLabel,int maxId,int fromRMP, int *dArrVidOnRMP,int segdArrVidOnRMP,int *rmp,int *dArrVj,int noElemdArrVj);

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
extern __global__ void kernelGetGraphIdContainEmbedding_pure(int li,int lij,int lj,EXT *d_ValidExtension,int noElem_d_ValidExtension,int *dV,unsigned int maxOfVer);

extern __global__ void kernelGetGraph(int *dV,int noElemdV,int *d_kq,int *dVScanResult);
extern __global__ void kernelMarkExtension(const Extension *d_ValidExtension, int noElem_d_ValidExtension,int *dV,int li,int lij,int lj);
extern __global__ void kernelMarkExtension_pure(const EXT *d_ValidExtension,int noElem_d_ValidExtension,int *dV,int li,int lij,int lj);
extern __global__ void kernelSetValueForFirstTwoEmbeddingColumn(const EXT *d_ValidExtension,int noElem_d_ValidExtension,Embedding *dQ1,Embedding *dQ2,int *d_scanResult,int li,int lij,int lj);
extern __global__ void	kernelPrintEmbedding(Embedding *dArrEmbedding,int noElem);
extern __global__ void kernelCalDegreeOfVid(int *listOfVer,int *d_O, int numberOfElementd_O,int noElem_Embedding,int numberOfElementd_N,unsigned int maxOfVer,float *dArrDegreeOfVid);
extern __global__ void find_maximum_kernel(float *array, float *max, int *mutex, unsigned int n);
extern __global__ void kernelFindValidForwardExtension(Embedding **dArrPointerEmbedding,int noElem_dArrPointerEmbedding,int noElem_Embedding,int *d_O,int *d_LO,int *d_N,int *d_LN,float *dArrDegreeOfVid,int maxDegreeOfVer,EXT *dArrExtension,int *listOfVer,int minLabel,int maxId,int fromRMP,int *dArrV_valid,int *dArrV_backward);
extern 	__global__ void printdArrUniEdge(UniEdge *dArrUniEdge,int i);
extern __global__ void	kernelGetvivj(EXT *dArrEXT,int noElemdArrEXT,int li,int lij,int lj,int *dvi,int *dvj);
extern __global__ void kernelGetLastElementEXT(EXT *inputArray,int noEleInputArray,int *value,unsigned int maxOfVer);

extern __global__ void kernelGetGraphIdContainEmbeddingv2(int li,int lij,int lj,EXT *d_ValidExtension,int noElem_d_ValidExtension,int *dV,unsigned int maxOfVer);
extern __global__ void kernelExtractValidExtension_pure(Extension *d_Extension,int *dV,int *dVScanResult,int numberElementd_Extension,EXT *d_ValidExtension);
extern __global__ void kernelMarkLabelEdge_pure(EXT *d_ValidExtension,unsigned int noElem_d_ValidExtension,unsigned int Lv,unsigned int Le,int *d_allPossibleExtension);
extern __global__ void kernelCalcBoundary_pure(EXT *d_ValidExtension,unsigned int noElem_d_ValidExtension,int *dB,unsigned int maxOfVer);
extern __global__ void	kernelExtractUniEdgeSatifyMinsup_pure(UniEdge *dUniEdge,int *dV,int *dVScanResult,int noElemUniEdge,UniEdge *dUniEdgeSatisfyMinsup,int *dSup,int *dResultSup);
extern __global__ void kernelMarkExtension(const EXT *d_ValidExtension,int noElem_d_ValidExtension,int *dV,int li,int lij,int lj);

extern cudaError_t getLastElementEXT(EXT *inputArray,int numberElementOfInputArray,int &outputValue,unsigned int maxOfVer);
extern cudaError_t ADM(int *&devicePointer,size_t nBytes);
extern void sumUntilReachZero(int *h_Lv,unsigned int n,int &result);
extern cudaError_t validEdge(Extension *d_Extension,int *&dV,unsigned int numberElementd_Extension);
extern cudaError_t getSizeBaseOnScanResult(int *dV,int *dVScanResult,int noElem,int &output);
extern cudaError_t extractValidExtension(Extension *d_Extension,int *dV,int *dVScanResult, int numberElementd_Extension,Extension *&d_ValidExtension);
extern cudaError_t extractValidExtension_pure(Extension *d_Extension,int *dV,int *dVScanResult, int numberElementd_Extension,EXT *&d_ValidExtension);
extern cudaError_t markLabelEdge(Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,unsigned int Lv,unsigned int Le,int *&d_allPossibleExtension);
extern cudaError_t markLabelEdge_pure(EXT *&d_ValidExtension,unsigned int Lv,unsigned int Le,int *&d_allPossibleExtension);

extern cudaError_t calcLabelAndStoreUniqueExtension(int *d_allPossibleExtension,int *d_allPossibleExtensionScanResult,unsigned int noElem_allPossibleExtension,UniEdge *&d_UniqueExtension,unsigned int noElem_d_UniqueExtension,unsigned int Le,unsigned int Lv);
extern cudaError_t calcBoundary(Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,int *&dB,unsigned int maxOfVer);
extern cudaError_t calcBoundary_pure(EXT *&d_ValidExtension,unsigned int noElem_d_ValidExtension,int *&dB,unsigned int maxOfVer);
extern cudaError_t getLastElement(int *dScanResult,unsigned int noElem,int &output);
extern cudaError_t calcSupport(UniEdge *dUniEdge,int noElemdUniEdge,Extension *dValidExtension,int noElemdValidExtension,int *dBScanResult,int *dF,int noElemF,int *&hResultSup);
extern cudaError_t calcSupport_pure(UniEdge *dUniEdge,int noElemdUniEdge,EXT *dValidExtension,int noElemdValidExtension,int *dBScanResult,int *dF,int noElemF,int *&hResultSup);
extern cudaError_t getLastElementExtension(Extension* inputArray,unsigned int numberElementOfInputArray,int &outputValue,unsigned int maxOfVer);
extern cudaError_t getLastElementExtension_pure(EXT* inputArray,unsigned int numberElementOfInputArray,int &outputValue,unsigned int maxOfVer);

extern cudaError_t  myScanV(int *dArrInput,int noElem,int *&dResult);
extern int displayDeviceEXT(EXT *dArrEXT,int noElemdArrEXT);
///<sumary>
///This is use to manage Level
///</sumary>
class clsLevel
{
public:
	int Level;
	int size;
	int prevLevel;
	clsLevel()
	{
		Level=0;
		size=1;
		prevLevel=-1;
	}
};



class PMS:public gSpan
{
public:	
	clsLevel objLevel;
	int currentColEmbedding; //index của Embedding column hiện đang được xử lý
	int Level;
	int idxLevel;
	vector<DB> hdb;
	vector<arrExtension> hExtension; 
	vector<arrExtension> hValidExtension; //Lưu các cạnh hợp lệ ban đầu
	vector<arrUniEdge> hUniEdge;
	vector<arrUniEdgeSatisfyMinSup> hUniEdgeSatisfyMinsup;
	vector<vecArrUniEdgeStatisfyMinSup> hLevelUniEdgeSatisfyMinsup;
	vector<vecArrUniEdgeStatisfyMinSup> hLevelUniEdgeSatisfyMinsupv2;

	vector<EmbeddingColumn> hEmbedding; //Mỗi phần tử của vector là một Embedding column
	//Embedding **dArrPointerEmbedding;
	vector<ptrArrEmbedding> hLevelPtrEmbedding;
	vector<ptrArrEmbedding> hLevelPtrEmbeddingv2;

	 
	vector<vecArrUniEdge> hLevelUniEdge;
	vector<vecArrUniEdge> hLevelUniEdgev2;
	vector<listVer> hListVer;
	vector<listVer> hListVerv2;
	vector<listVer> hLevelListVerRMP;

	vector<RMP> hRMP;
	vector<RMP> hRMPv2;
	vector<RMP> hLevelRMP;

	
	//vector<EXTk> hEXTk; //Có bao nhiêu EXTk
	vector<vecArrEXT> hLevelEXT; //quản lý EXTk theo Level, mỗi một Level là 1 lần gọi đệ quy FSMining function
	vector<vecArrEXT> hLevelEXTv2;
	
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
	void increaseLevel();	
	int prepareDataBase();
	void displayArray(int*, const unsigned int);
	int displayDeviceArr(int *,int);
	int displayDeviceArr(float*,int);
	//void displayEmbeddingColumn(EmbeddingColumn);
	int displayArrExtension(Extension*, int);
	int displaydArrEXT(EXT*,int);
	int displayArrUniEdge(UniEdge*,int);
	bool checkArray(int*, int*, const int);
	void printdb();
	int countNumberOfDifferentValue(int*,unsigned int,unsigned int&);
	int extractAllEdgeInDB();
	int getAndStoreExtension(Extension*&);
	int getValidExtension_pure();
	int extractUniEdge();
	int computeSupport();
	int extractUniEdgeSatisfyMinsup(int*,int,unsigned int);
	//int Mining();
	int initialize();
	int Miningv2(int,UniEdge*,int*,EXT*,int,int);
	int Miningv3(int,UniEdge*,int*,EXT*,int,int);

	int getGraphIdContainEmbedding(UniEdge,int*&,int&);	
	int getGraphIdContainEmbedding_pure(UniEdge edge,int *&hArrGraphId,int &noElemhArrGraphId);

	//int buildFirstEmbedding(UniEdge);
	int buildEmbedding_pure(UniEdge);

	int buildRMP();
	int FSMining(int*,int);
	int FSMiningv2();
	int FSMiningv3(int);
	int FSMiningv4(int);

	int forwardExtension(int,int*,int,int);
	int findMaxDegreeOfVer(int*,int&,float*&,int);
	int findDegreeOfVer(int*,float*&,int);
	int extractValidExtensionTodExt(EXT*,V*,int,int);
	int extractValidExtensionTodExtv2(EXT*,V*,int,int);
	int extractValidExtensionTodExtv3(EXT*,V*,int,int);
	int extractValidExtensionTodExtv4(EXT*,V*,int,int);

	int computeSupportv2(EXT*,int,UniEdge*,int,int&,UniEdge*&,int*&);
	int computeSupportv3(EXT*,int,UniEdge*,int,int,int,int&,UniEdge*&,int*&);
	int computeSupportv4(EXT*,int,UniEdge*,int,int,int,int&,UniEdge*&,int*&);

	int findBoundary(EXT*,int,int *&);
	int extractUniEdgeSatisfyMinsupV2(int *,UniEdge*,int ,unsigned int ,int &,UniEdge *&,int *&);
	int extractUniEdgeSatisfyMinsupV3(int *,UniEdge*,int ,unsigned int ,int &,UniEdge *&,int *&);
	int getvivj(EXT*,int,int,int,int,int&,int&);
	int getGraphIdContainEmbeddingv2(UniEdge ,int *&,int &,EXT *,int );
	int extendEmbedding(UniEdge ue,int idxExt);
	int extendEmbeddingv2(UniEdge,EXT*,int);

	int updateRMP();
	int updateRMPBW();
	int displaydArrPointerEmbedding(Embedding** ,int noElemEmbeddingCol,int noElemEmbedding);
	int buildArrPointerEmbedding(vector<EmbeddingColumn>,vector<ptrArrEmbedding>&);
	int buildArrPointerEmbeddingv2(vector<EmbeddingColumn>,vector<ptrArrEmbedding>&);

	int buildArrPointerEmbeddingbw(vector<EmbeddingColumn>,vector<ptrArrEmbedding>&);

	int buildrmpOnDevice(RMP,int*&);
	int findListVer(Embedding**,int,int*,int,vector<listVer>&);
	int findVerOnRMPForBWCheck(ptrArrEmbedding,int*,int,int*&);
	int findVerOnRMPForBWCheckv2(ptrArrEmbedding,int*,int,int*&);

	int findValidFBExtension(int*,ptrArrEmbedding,int,int,int*,int*);
	int findValidFBExtensionv2(int*,ptrArrEmbedding,int,int,int*,int*);
	int findValidForwardExtensionForNonLastSegment(int*,ptrArrEmbedding,int,int,int*,int*);

	int findValidForwardExtensionv2(int*,ptrArrEmbedding,int,int,int*,int*);


	int extractUniqueForwardBackwardEdge_LastExt(EXTk,UniEdgek&);
	int extractUniqueForwardBackwardEdge_LastExtv2(EXTk,UniEdgek&);
	int extractUniqueForwardEdge_NonLastExtv2(EXTk,UniEdgek&);


	int markValidForwardEdge(EXT*,int,unsigned int,int*);
	//int markValidBackwardEdge(EXT*,int,unsigned int,int*);
	int cpResultToUE(UniEdgek,UniEdgek,int*,UniEdgek&);
	int cpResultToUEfw(UniEdgek,int*,UniEdgek&);
	int cpResultToUEbw(UniEdgek,int*,UniEdgek&);

	int extractAllBWExtension(UniEdgek& ,EXTk);
	int extractValidBWExtension(UniEdge* ,int,UniEdge*&,int*,int*);
	int extractAllBWExtensionv2(UniEdgek& ,EXTk);

	int extractAllFWExtension(UniEdgek& ,EXTk);
	int computeSupportBW(EXT*,int*,int,UniEdge*,int,float*,int,int&);
	int computeSupportFW(EXT*,int*,int,UniEdge*,int,float*,int,int&);
	int getGraphIdContainEmbeddingFW(UniEdge,int*&,int&,EXT*,int);
	int getGraphIdContainEmbeddingBW(UniEdge,int*&,int&,EXT*,int);
	int extendEmbeddingBW2(UniEdge,EmbeddingColumn&,Embedding*,EXT*,int);
	int extendEmbeddingBW(UniEdge,EmbeddingColumn&,EXT*,int);

	int displayBWEmbeddingCol(Embedding*,int);
	int getVjFromDFSCODE(int*&,int);
};

