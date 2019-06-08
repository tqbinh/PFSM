#pragma once
#include "gspan.cuh"
#include <stdio.h>
#include <vector>
#include <list>
#include <stdlib.h> /* exit, EXIT_FAILURE */
#include <stack>
#include <algorithm>
#include "helper_timer.h"
#include "scan_largearray_kernel.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include "scanV.h"
//#include "reduction.h"
#include "moderngpu.cuh"		// Include all MGPU kernels.

using namespace mgpu;
using namespace std;

#define blocksize 512

#define FUNCHECK(call) \
{ \
const int error = call; \
if (error != 0) \
{ \
std::printf("Error: %s:%d:%s, ", __FILE__, __LINE__,__FUNCTION__); \
std::printf("code:%d, reason: %s\n", error, "Function failed"); \
} \
}

#define CHECK(call) \
{ \
const cudaError_t error = call; \
if (error != cudaSuccess) \
{ \
std::printf("Error: %s:%d, ", __FILE__, __LINE__); \
std::printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
} \
}

#define FCHECK(call) \
{ \
const int error = call; \
if (error != 0) \
{ \
std::printf("Error: %s:%d:%s, ", __FILE__, __LINE__,__FUNCTION__); \
std::printf("code:%d, reason: %s\n", error, "Function failed"); \
system("pause"); \
exit(0); \
} \
}

#define CUCHECK(call) \
{ \
const cudaError_t error = call; \
if (error != cudaSuccess) \
{ \
std::printf("Error: %s:%d, ", __FILE__, __LINE__); \
std::printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
system("pause"); \
exit(0); \
} \
}

extern float hTime;
extern float dTime;
//void GridBlockCompute(int noElem,dim3 &block,dim3 &grid)
//{
//	dim3 b(blocksize);
//	dim3 g((noElem+b.x-1)/b.x);
//	block = b;
//	grid = g;
//}

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
//use
struct Extension //Thông tin của một cạnh thuộc CSDL
{
	int vi,vj,li,lij,lj; //DFS_code của cạnh mở rộng
	int vgi,vgj; //global id của đỉnh
	//struct_Embedding *d_rowpointer;//lưu trữ pointer trỏ đến embedding mà nó mở rộng.
	Extension():vi(0),vj(1),li(0),lij(0),lj(0),vgi(0),vgj(0){};//khởi tạo cấu trúc
};

//use
struct arrExtension
{
	int noElem; //Tổng số cạnh của tất cả các đồ thị trong CSDL
	Extension *dExtension; //Mảng liên tục, lưu trữ thông tin của cạnh thuộc CSDL.
	arrExtension():noElem(0),dExtension(0){};
};
//use
struct UniEdge
{
	int vi;
	int vj;
	int li;
	int lij;
	int lj;
	UniEdge():vi(-1),vj(-1),li(-1),lij(-1),lj(-1){};
public:
	void print()
	{
		std::printf("\n(%d,%d,%d,%d,%d)",vi,vj,li,lij,lj);
	}
};
//use
struct arrUniEdge
{
	int noElem;
	UniEdge *hUniEdge; //array unique edge
	int* hSupport;//array contains support of unique edge.
	UniEdge *dUniEdge;
	arrUniEdge():noElem(0),hSupport(nullptr),dUniEdge(nullptr){};

	void show()
	{
		UniEdge* hUniEdge = (UniEdge*)malloc(sizeof(UniEdge)*noElem);
		if(hUniEdge == NULL) FCHECK(-1);
		CUCHECK(cudaMemcpy(hUniEdge,dUniEdge,sizeof(UniEdge)*noElem,cudaMemcpyDeviceToHost));
		for (int i = 0; i < noElem; i++)
		{
			hUniEdge[i].print();
		}
		free(hUniEdge);
	}
	//use
	void showSupport()
	{
		cout<<"\nSupport:\n";
		for (int i = 0; i < noElem; i++)
		{
			std::printf("S[%d]:%d ",i,hSupport[i]);
		}
	}
	//use
	void copyDTH();
	void ReleaseMemory()
	{
		if (noElem>0)
		{
			if(dUniEdge!=nullptr) CUCHECK(cudaFree(dUniEdge));
			if(hUniEdge!=nullptr) free(hUniEdge);
			if(hSupport!=nullptr) free(hSupport);
		}
	}

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
//use
struct Embedding
{
	int idx; //index dòng của previous embedding column
	int vid; //id của đỉnh
	Embedding():idx(0),vid(0){};
};

struct EmbeddingBWCol
{
	int noElem; //Số lượng phần tử mảng dArrEmbedding
	int prevCol; //prevCol, dùng để xây dựng RMPath
	Embedding *dArrEmbedding;
	EmbeddingBWCol():noElem(0),prevCol(0),dArrEmbedding(0){};
public:
	void ReleaseMemory()
	{
		if (noElem > 0)
		{
			CUCHECK(cudaFree(dArrEmbedding));
		}
	}

	int show()
	{
		int status = 0;
		cudaError_t cudaStatus =cudaSuccess;
		//Allocate host memory for Embedding
		Embedding *hArrEmbedding = (Embedding*)malloc(sizeof(Embedding)*noElem);
		if (hArrEmbedding==NULL){status = -1;goto Error;}
		CHECK(cudaStatus = cudaMemcpy(hArrEmbedding,dArrEmbedding,sizeof(Embedding)*noElem,cudaMemcpyDeviceToHost));
		if(cudaStatus!=cudaSuccess){status = -1; goto Error;}
		//Print element in hArrExt
		std::printf("\n");
		std::printf("prevCol:%d",prevCol);
		for(int i=0;i<noElem;++i)
		{
			std::printf("\n hArrEmbedding[%d]:(idx,vid) (%d,%d)",i, \
				hArrEmbedding[i].idx, \
				hArrEmbedding[i].vid);
		}
		std::printf("\n");
		//Free memory
		free(hArrEmbedding);
	Error:
		return status;
	}
};


struct EmbeddingColumn
{
	vector<EmbeddingBWCol> hBackwardEmbedding;
	int noElem; //Số lượng embeddings của embedding column.
	int prevCol; //prevCol, dùng để xây dựng RMPath
	Embedding *dArrEmbedding;
	EmbeddingColumn():noElem(0),prevCol(0),dArrEmbedding(0){};

public:
	void ReleaseMemory()
	{
		if(hBackwardEmbedding.size()>0)
		{
			hBackwardEmbedding.back().ReleaseMemory();
			return;
		}

		if (noElem > 0)
		{
			CUCHECK(cudaFree(dArrEmbedding));
		}
	}

	int show()
	{
		int status = 0;
		cudaError_t cudaStatus =cudaSuccess;
		//Allocate host memory for Embedding
		Embedding *hArrEmbedding = (Embedding*)malloc(sizeof(Embedding)*noElem);
		if (hArrEmbedding==NULL){status = -1;goto Error;}
		CHECK(cudaStatus = cudaMemcpy(hArrEmbedding,dArrEmbedding,sizeof(Embedding)*noElem,cudaMemcpyDeviceToHost));
		if(cudaStatus!=cudaSuccess){status = -1; goto Error;}
		//Print element in hArrExt
		std::printf("\n");
		std::printf("prevCol:%d",prevCol);
		for(int i=0;i<noElem;++i)
		{
			std::printf("\n hArrEmbedding[%d]:(idx,vid) (%d,%d)",i, \
				hArrEmbedding[i].idx, \
				hArrEmbedding[i].vid);
		}
		std::printf("\n");
		//Free memory
		free(hArrEmbedding);
	Error:
		return status;
	}

};
//use
struct EmCol
{
	vector<EmCol> hBackwardEmbedding;
	int noElem; //Số lượng embeddings của embedding column.
	int prevCol; //prevCol, dùng để xây dựng RMPath
	Embedding *dArrEmbedding;
	EmCol():noElem(0),prevCol(0),dArrEmbedding(nullptr){};

public:
	//không nên dùng hàm ReleaseMemory vì nó không chủ động release vector
	void ReleaseMemory()
	{
		if(hBackwardEmbedding.size()>0)
		{
			CUCHECK(cudaFree(hBackwardEmbedding.back().dArrEmbedding));
			hBackwardEmbedding.pop_back();
			return;
		}

		if (noElem > 0)
		{
			CUCHECK(cudaFree(dArrEmbedding));
		}
	}
	void show()
	{
		//Allocate host memory for Embedding
		Embedding *hArrEmbedding = (Embedding*)malloc(sizeof(Embedding)*noElem);
		if (hArrEmbedding==NULL){FCHECK(-1);}
		CUCHECK(cudaMemcpy(hArrEmbedding,dArrEmbedding,sizeof(Embedding)*noElem,cudaMemcpyDeviceToHost));
		//Print element in hArrExt
		std::printf("\n");
		std::printf("prevCol:%d",prevCol);
		for(int i=0;i<noElem;++i)
		{
			std::printf("\n hArrEmbedding[%d]:(idx,vid) (%d,%d)",i, \
				hArrEmbedding[i].idx, \
				hArrEmbedding[i].vid);
		}
		std::printf("\n");
		//Free memory
		free(hArrEmbedding);
		return;
	}
};

//Struct lưu trữ RMP của DFS_CODE trên host lẫn device
struct RMP
{
	//Số đỉnh trên DFS_CODE thuộc RMP
	int noElem;
	//vector lưu trữ các đỉnh thuộc RMP trên host
	vector<int> hArrRMP;
	//vector lưu trữ các đỉnh thuộc RMP trên device
	int *dRMP;
	RMP():noElem(0),hArrRMP(0),dRMP(0){};
public:
	int ReleaseMemory()
	{
		int status = 0;
		cudaError_t cudaStatus;
		if (hArrRMP.size()>0)
		{
			hArrRMP.clear();
			CHECK(cudaStatus = cudaFree(dRMP));
			if(cudaStatus!=cudaSuccess)
			{
				status = -1;
				goto Error;
			}
		}
Error:
		return status;
	}
};
//use
struct EXT
{
	int vi,vj,li,lij,lj; /*DFS code của cạnh mở rộng, Ở đây giá trị của (vi,vj) tuỳ thuộc vào 2 thông tin: Mở rộng là backward hay forward và mở rộng tử đỉnh nào trên RMPath. Nếu là mở rộng forward thì cần phải biết maxid của DFS_CODE hiện tại */
	int vgi,vgj;/* globalId vertex của cạnh mở rộng*/
	//Hai thông tin bên dưới cho biết cạnh mở rộng từ embedding nào. Chúng ta cần phải biết thông tin này để 
	int posRow; //vị trí của embedding trong cột Q
	EXT():vi(0),vj(1),li(0),lij(0),lj(0),vgi(0),vgj(0),posRow(-1){};
};
//use
extern void getSizeBaseOnScanResult(int *dV,int *dVScanResult,int noElem,int &output);
//use
extern void myScanV(int *dArrInput,int noElem,int *&dResult);
//use
extern __global__ void kernelExtractValidExtensionTodExt(EXT *dArrExtension,int *dArrValid,int *dArrValidScanResult,int noElem_dArrV,EXT *dExt,int noElem_dExt);
extern int displayDeviceArr(int*,int);
//use
struct UniEdgeStatisfyMinSup
{
	//Số lượng phần tử của mảng dArrUniEdge
	int noElem;
	//Chứa các mở rộng duy nhất thoả mãn minsup
	UniEdge *dArrUniEdge;
	UniEdge *hArrUniEdge;
	//Chứa độ hỗ trợ tương ứng với mở rộng duy nhất ở dArrUniEdge
	int *hArrSupport;

	UniEdgeStatisfyMinSup():noElem(0),dArrUniEdge(nullptr),hArrSupport(nullptr){};
public:
	//use
	void ReleaseMemory()
	{
		if (noElem > 0)
		{
			free(hArrUniEdge);
			free(hArrSupport);
			CUCHECK(cudaFree(dArrUniEdge));
		}
		return;
	}
	void show()
	{
		if(noElem<0)
		{
			std::printf("\n Empty UniEdge Satisfy minsup");
			return;
		}
		UniEdge* temp = nullptr;
		temp = (UniEdge*)malloc(sizeof(UniEdge)*noElem);
		if(temp==nullptr) {FCHECK(-1);}
		CUCHECK(cudaMemcpy(temp,dArrUniEdge,sizeof(UniEdge)*noElem,cudaMemcpyDeviceToHost));
		std::printf("\nUnique Edges Statisfy minsup\n");
		for (int i = 0; i < noElem; i++)
		{
			temp[i].print();
		}
		free(temp);
		return;
	}
};

struct existBackwardInfo
{
	//Có số lượng phần tử bằng noElemRMP
	int* dValidBackward;
	int* dVj;
	existBackwardInfo():dValidBackward(nullptr),dVj(nullptr){};
};


//Lưu trữ các mở rộng hợp lệ trên device.
//use
struct EXTk
{
	//Số lượng phần tử dArrExt.
	int noElem;
	//Lưu trữ các mở rộng hợp lệ trên device.
	EXT *dArrExt;
	arrUniEdge uniFE; //uniForward Extension
	arrUniEdge uniBE; //uniBackard Extension
	UniEdgeStatisfyMinSup uniFES; //unique forward edge statisfy minsup
	UniEdgeStatisfyMinSup uniBES; //unique backward edge statisfy minsup
	EXTk():noElem(0),dArrExt(nullptr){};
public:
	void mark_edge(int vi,int vj,int li,int lij,int lj,int *&dValid);
	//use
	void ReleaseMemory()
	{
		if (noElem>0)
		{
			CUCHECK(cudaFree(dArrExt));
		}
		return;
	}
	void show()
	{
		int status = 0;
		//Allocate host memory for EXT
		EXT *hArrExt = (EXT*)malloc(sizeof(EXT)*noElem);
		if (hArrExt==NULL){FCHECK(-1)}
		CUCHECK(cudaMemcpy(hArrExt,dArrExt,sizeof(EXT)*noElem,cudaMemcpyDeviceToHost));
		//Print element in hArrExt
		std::printf("\n");
		for(int i=0;i<noElem;++i)
		{
			std::printf("\n Ext[%d]:(vi,vj,li,lij,lj,vgi,vgj,posRow) (%d,%d,%d,%d,%d,%d,%d,%d)",i, \
				hArrExt[i].vi, \
				hArrExt[i].vj, \
				hArrExt[i].li, \
				hArrExt[i].lij, \
				hArrExt[i].lj, \
				hArrExt[i].vgi, \
				hArrExt[i].vgj, \
				hArrExt[i].posRow);
		}
		std::printf("\n");
		//Free memory
		free(hArrExt);
		return;
	}
	void extractUniForwardExtension(unsigned int&,unsigned int&,int&);
	//use
	void extractUniBackwardExtension(unsigned int&,unsigned int&,int& noElemRMP,int*& dRMP,int*& dRMPLabel, int& noElemMappingVj,int& vi,int& li);
	//use
	void findSupport(unsigned int&);
	//use
	void findBoundary(unsigned int&, int*&);
	//use
	void findSupportFW(int*& dArrBoundaryScanResult,UniEdge*& dArrUniEdge,int& idxUniEdge, int*& dF,int& noElemdF,int& support);
	//use
	void extractStatisfyMinsup(unsigned int& minsup,arrUniEdge& uniEdge,UniEdgeStatisfyMinSup& uniES);
};
//use
struct structValid
{
	int noElem;
	int *dArrValid;
	EXT *dArrEXT;
	structValid():noElem(0),dArrValid(0),dArrEXT(0){};

	void show()
	{
		size_t noBytes = sizeof(int)*noElem;
		int* hArrValid = (int*)malloc(noBytes);
		EXT* hArrEXT = (EXT*)malloc(sizeof(EXT)*noElem);

		if(hArrValid == nullptr) {FCHECK(-1);}
		if(hArrEXT == nullptr) {FCHECK(-1);}
		CUCHECK(cudaMemcpy(hArrValid,dArrValid,noBytes,cudaMemcpyDeviceToHost));
		CUCHECK(cudaMemcpy(hArrEXT, dArrEXT,sizeof(EXT)*noElem,cudaMemcpyDeviceToHost));

		cout<<endl;
		for (int i = 0; i < noElem; i++)
		{
			std::printf("%d ==> V:%d E:(%d,%d,%d,%d,%d,%d,%d,%d)\n",i,hArrValid[i], \
				hArrEXT[i].vi, hArrEXT[i].vj, \
				hArrEXT[i].li,hArrEXT[i].lij,hArrEXT[i].lj, \
				hArrEXT[i].vgi,hArrEXT[i].vgj, \
				hArrEXT[i].posRow);
		}
		free(hArrValid);
		free(hArrEXT);
	}

	void extractValid(EXTk &outputEXT);
	void ReleaseMemory()
	{
		if(noElem>0)
		{
			CUCHECK(cudaFree(dArrValid));
			CUCHECK(cudaFree(dArrEXT));
		}
	}
};
/*Số lượng phần tử vE dựa vào số lượng đỉnh thuộc right most path của Level đang xét.
	Nếu Level đang xét có 3 đỉnh thuộc right most path thì chúng ta tạo ra 3 phần tử vE, 
	mỗi phần tử vE sẽ lưu trữ các mở rộng hợp lệ của các đỉnh thuộc embedding column đang 
	xét.
*/
//use
struct vecArrEXT
{	
	//Số lượng phần tử vE
	int noElem; 
	//Mỗi phần tử của vector vE sẽ quản lý 1 phần tử dArrExt
	vector<EXTk> vE; 
	vecArrEXT():noElem(0),vE(0){};
public:
	void ReleaseMemory()
	{
		int status = 0;
		for(int i=0; i<vE.size(); ++i)
		{
			vE.at(i).ReleaseMemory();
		}
		vE.clear();
		return;
	}
};


struct vecArrUniEdgeStatisfyMinSup
{
	//Số lượng phần tử của vecUES
	//Được dùng để resize vector vecUES tại level đang khai thác.
	int noElem;
	vector<UniEdgeStatisfyMinSup> vecUES;
	vecArrUniEdgeStatisfyMinSup():noElem(0),vecUES(0){};
public:
	void ReleaseMemory()
	{
		int status = 0;
		for(int i = 0; i<vecUES.size(); ++i)
		{
			vecUES.at(i).ReleaseMemory();
		}
		vecUES.clear();
		return;
	}
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
	//số lượng embedding column. Nên đổi tên là noElemColumn
	int noElem;
	//số lượng phần tử ở cột cuối cùng
	int noElemEmbedding;
	//trỏ đến embedding column trên device.
	Embedding **dArrPointerEmbedding;
	ptrArrEmbedding():noElem(0),noElemEmbedding(0),dArrPointerEmbedding(0){};
public:
	int ReleaseMemory()
	{
		int status =0;
		cudaError_t cudaStatus ;
		CHECK(cudaStatus = cudaFree(dArrPointerEmbedding));
		if(cudaStatus != cudaSuccess)
		{
			status = -1;
		}
		return status;
	}
};

//Lưu giữ danh sách đỉnh thuộc RMP trên device
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

extern ContextPtr ctx;

class PMS:public gSpan
{
public:	
	//clsLevel objLevel;
	//int currentColEmbedding; //index của Embedding column hiện đang được xử lý
	//int Level;
	//int idxLevel;

	//use
	vector<DB> hdb;
	//use
	vector<arrExtension> hExtension; 
	//vector<arrExtension> hValidExtension; //Lưu các cạnh hợp lệ ban đầu
	//use
	vector<arrUniEdge> hUniEdge;
	//vector<arrUniEdgeSatisfyMinSup> hUniEdgeSatisfyMinsup;
	//use
	vector<vecArrUniEdgeStatisfyMinSup> hLevelUniEdgeSatisfyMinsup;
	
	//vector<vecArrUniEdgeStatisfyMinSup> hLevelUniEdgeSatisfyMinDFSCODE;
	//vector<vecArrUniEdgeStatisfyMinSup> hLevelUniEdgeSatisfyMinsupv2;

	//vector Embedding column ban đầu. Mỗi phần tử là một embedding column.
	//vector<EmbeddingColumn> hEmbedding; //Mỗi phần tử của vector là một Embedding column
	//use
	vector<EmCol> hEm;
	//use
	void getEmCol(Embedding** &dEmCol,int &noElemdEmCol);
	//use
	void getEmColRMP(Embedding** &dEmCol,const int &noElemRMP);
	//use
	void getnoElemEmbedding(int &noElemEmbedding);
	//use
	void createMarkEmColRMP(int* &dRMP,int &noElemdRMP,int* &dEmColRMP);
	//use
	void createRMP(int* &dRMP,int &noElem);

	//Embedding **dArrPointerEmbedding;
	//Lưu trữ embedding ở từng level.
	//vector<ptrArrEmbedding> hLevelPtrEmbedding;
	//vector<ptrArrEmbedding> hLevelPtrEmbeddingv2;

	 
	//vector<vecArrUniEdge> hLevelUniEdge;
	//vector<vecArrUniEdge> hLevelUniEdgev2;
	//vector<listVer> hListVer;
	//vector<listVer> hListVerv2;

	//Các đỉnh thuộc RMP ở từng Level
	//vector<listVer> hLevelListVerRMP;

	//vector<RMP> hRMP;
	//vector<RMP> hRMPv2;
	//Dùng để lưu trữ Right Most Path của DFS_CODE ban đầu (level 0)
	//vector<RMP> hLevelRMP;

	
	//vector<EXTk> hEXTk; //Có bao nhiêu EXTk

	//Quản lý EXTk theo Level, mỗi một Level là 1 lần gọi đệ quy FSMining function
	//use
	vector<vecArrEXT> hLevelEXT; 
	//vector<vecArrEXT> hLevelEXTv2;
	
	PMS();
	~PMS();
	//use
	unsigned int Lv;
	//use
	unsigned int Le;
	//use
	unsigned int maxOfVer;
	//use
	unsigned int numberOfGraph;
	//use
	int minLabel;
	//use
	int maxId;

	//std::ostream* os;
	//std::ofstream fos; //fos là pointer trỏ tới tập tin /result.graph


public:	
	//use
	void prepareDataBase();
	//void displayArray(int*, const unsigned int);
	//use
	void displayHostArray(int*&,const unsigned int);
	//use
	void displayDeviceArr(int*,int);
	//use
	void displayDeviceArr(float* &dArr,int &noElem);
	//int displayArrExtension(Extension*, int);
	//int displaydArrEXT(EXT*,int);
	//int displayArrUniEdge(UniEdge*,int);
	//bool checkArray(int*, int*, const int);
	//use
	void printdb();
	//use
	void countNumberOfDifferentValue(int*,unsigned int,unsigned int&);
	//use
	void extractAllEdgeInDB();
	//use
	void getAndStoreExtension(Extension*&);
	//use
	void getValidExtension_pure();
	//use
	void extractUniEdge();
	//use
	void computeSupport();
	//use
	void extractUniEdgeSatisfyMinsup(int*,int,unsigned int);
	//int Mining();
	//int initialize();
	//use
	void MiningDeeper(EXTk&,UniEdgeStatisfyMinSup&);
	
	//int Miningv2(int,UniEdge*,int*,EXT*,int,int);
	//int Miningv3(int,UniEdge*,int*,EXT*,int,int);

	//int getGraphIdContainEmbedding(UniEdge,int*&,int&);
	//int getGraphIdContainEmbedding_pure(UniEdge edge,int *&hArrGraphId,int &noElemhArrGraphId);
	//use
	void buildEmbedding(UniEdge&,EXTk&,int*&,int*&);
	//use
	void buildBackwardEmbedding(UniEdge& ue,EXTk& ext,int*& dValid,int*& dIdx);
	//use
	void buildNewEmbeddingCol(UniEdge&,EXTk&,int*&,int*&);
	//use
	void buildFirstEmbedding(UniEdge&,EXTk&,int*&,int*&);
	//use
	void removeEmbedding();
	//use
	void removeFirstEmbedding();
	//int buildEmbedding_pure(UniEdge&);

	//int buildRMP();
	//int FSMining(int*,int);
	//int FSMiningv2();
	//int FSMiningv3(int);
	//int FSMiningv4(int);

	//int forwardExtension(int,int*,int,int);
	//int findMaxDegreeOfVer(int*&,int&,float*&,int&);

	//use
	void findMaxDegreeVid(Embedding** &dEmCol,int* &dEmRMP,int &noElemdEmCol, int &noElemVid, \
			int &noElemRMP, int &noElemEmbedding, \
			float* &dArrDegreeOfVid,int &maxDegreeOfVer);

	//int findMaxDegreeOfVerEmbeddingColumn(int&,int&,float*&);
	//int findDegreeOfVer(int*&,float*&,int&);
	//int findDegreeOfVerEmbeddingColumn(int&,float*&,int&);
	//int extractValidExtensionTodExt(EXT*,V*,int,int);
	//int extractValidExtensionTodExtv2(EXT*,V*,int,int);
	//int extractValidExtensionTodExtv3(EXT*,V*,int,int);
	//int extractValidExtensionTodExtv4(EXT*,V*,int,int);

	//int computeSupportv2(EXT*,int,UniEdge*,int,int&,UniEdge*&,int*&);
	//int computeSupportv3(EXT*,int,UniEdge*,int,int,int,int&,UniEdge*&,int*&);
	//int computeSupportv4(EXT*,int,UniEdge*,int,int,int,int&,UniEdge*&,int*&);

	//int findBoundary(EXT*,int,int *&);
	//int extractUniEdgeSatisfyMinsupV2(int *,UniEdge*,int ,unsigned int ,int &,UniEdge *&,int *&);
	//int extractUniEdgeSatisfyMinsupV3(int *,UniEdge*,int ,unsigned int ,int &,UniEdge *&,int *&);
	//int getvivj(EXT*,int,int,int,int,int&,int&);
	//int getGraphIdContainEmbeddingv2(UniEdge ,int *&,int &,EXT *,int );

	//use
	void get_graphid(UniEdge& ,int *&,int &,EXT *,int );

	//int WriteResult(UniEdge &,EXTk &,int &);
	//int extendEmbedding(UniEdge ue,int idxExt);
	//int extendEmbeddingv2(UniEdge,EXT*,int);
	//int ExtendEmbedding(UniEdge &ue,EXT *&,int &);

	//int updateRMP();
	//int updateRMP_DFSCODE();
	//int updateRMPBW();
	//int displaydArrPointerEmbedding(Embedding** ,int noElemEmbeddingCol,int noElemEmbedding);
	//int displayEmbeddingColumn(const vector<ptrArrEmbedding>&);
	//int saveEmbeddingColumn(vector<ptrArrEmbedding>&);
	//int buildArrPointerEmbedding(vector<EmbeddingColumn>,vector<ptrArrEmbedding>&);
	//int buildArrPointerEmbeddingv2(vector<EmbeddingColumn>,vector<ptrArrEmbedding>&);
	//int buildArrPointerEmbeddingv3();
	//void write_embedding_column();

	//int buildArrPointerEmbeddingbw(vector<EmbeddingColumn>,vector<ptrArrEmbedding>&);

	//int buildrmpOnDevice(RMP,int*&);
	//int findListVer(Embedding**,int,int*,int);
	//int findListVerOnRMP();
	//int findVerOnRMPForBWCheck(ptrArrEmbedding,int*,int,int*&);
	//int findVerOnRMPForBWCheckv2(ptrArrEmbedding,int*,int,int*&);

	//int findValidFBExtension(int*,ptrArrEmbedding,int,int,int*,int*);
	//int findValidFBExtensionv2(int*,ptrArrEmbedding,int,int,int*,int*);

	//use
	void findValidExtension(vector<EXTk>&);

	//int findValidForwardExtensionForNonLastSegment(int*,ptrArrEmbedding,int,int,int*,int*);
	//int findValidForwardExtensionv2(int*,ptrArrEmbedding,int,int,int*,int*);
	//int findForwardExtension(int*,ptrArrEmbedding,int,int,int*,int*);
	//int extractUniqueForwardBackwardEdge_LastExt(EXTk,UniEdgek&);
	//int extractUniqueForwardBackwardEdge_LastExtv2(EXTk,UniEdgek&);
	//int extractUniqueForwardEdge_NonLastExtv2(EXTk,UniEdgek&);
	//int markValidForwardEdge(EXT*,int,unsigned int,int*);
	//int markValidBackwardEdge(EXT*,int,unsigned int,int*);
	//int cpResultToUE(UniEdgek,UniEdgek,int*,UniEdgek&);
	//int cpResultToUEfw(UniEdgek,int*,UniEdgek&);
	//int cpResultToUEbw(UniEdgek,int*,UniEdgek&);
	//int extractAllBWExtension(UniEdgek& ,EXTk);
	//int extractValidBWExtension(UniEdge* ,int,UniEdge*&,int*,int*);
	//int extractAllBWExtensionv2(UniEdgek& ,EXTk);
	//int extractAllFWExtension(UniEdgek& ,EXTk);
	//int computeSupportBW(EXT*,int*,int,UniEdge*,int,int*,int,int&);
	//int computeSupportFW(EXT*,int*,int,UniEdge*,int,int*,int,int&);
	//int getGraphIdContainEmbeddingFW(UniEdge,int*&,int&,EXT*,int);
	//int getGraphId(UniEdge&,int*&,int&,EXT*&,int&);
	//int getGraphIdContainEmbeddingBW(UniEdge,int*&,int&,EXT*,int);
	//int extendEmbeddingBW2(UniEdge,EmbeddingColumn&,Embedding*,EXT*,int);
	//int extendEmbeddingBW(UniEdge,EmbeddingColumn&,EXT*,int);
	//int displayBWEmbeddingCol(Embedding*,int);
	//int getVjFromDFSCODE(int*&,int);

	//use
	void buildRMPLabel(int*& dRMP, int*& dRMPLabel,int& noElemMappingVj,int& vi,int& li);
	//use
	void buildExistBackwardInfo(int* &dRMP,int &noElemOnRMP, \
								 int* &dValidBackward);
	//use
	void getVjBackwardDFSCODE(int* &dRMP,int &noElemOnRMP, \
								int* &dVj,int &noElemdVj);
};



//extern __global__ void	kernelGetRow(int *dV,int *dVScanResult,int noElemdV,int *dArrRow);
//use
extern __global__ void kernelCopyDeviceEXT(EXT** dPointerArr,EXT* dArr,int at);
//use
extern void write_minDFS_CODE(DFSCode dfscode);
//use
extern __global__ void kernelCopyDevice(int** dPointerArr,int* dArr,int at);
//use
extern void markLabelEdge_pure(EXT *&d_ValidExtension,unsigned int noElem_d_ValidExtension,unsigned int Lv,unsigned int Le,int *&d_allPossibleExtension);
//use
extern __global__ void kernelCopyDeviceArray(int *dArrInput,int *dResult,int noElem);

//extern __global__ void kernelCopyDevice(int* dPointerArr,int* dArr,int at);
//extern __global__ void kernelCopyDeviceEXT(EXT* dPointerArr,EXT* dArr,int at);

//use
extern __global__ void kernelFillValidBackward(int* dValidBackward,int* dVj,int noElem, int* dLookupArrVj,int noElemLookup);
//use
extern __device__ void deviceGetVid(Embedding** &dEmCol, int* &dEmRMP,int &noElemdEmCol, \
						int &noElemEmbedding,int &idxCol, int &idxRow, int &noElemOnRMP, \
						int &vid,int &idxOnRMP);
//use
extern __device__ void deviceIsVidOnEm(int &toVid,Embedding** &dEmCol,int* &dEmRMP,int &noElemdEmCol, int &idxRow,int &noElemRMP, \
				int &onEm, int &onRMP,int &idxOnRMPtovid);
//use
extern __device__ void deviceFindVid(int &thread , Embedding** &dEmCol, int* &dEmRMP,int &noElemdEmCol, \
						int &noElemEmbedding,int &noElemOnRMP, \
						 int &idxCol, int &idxRow,int& vid,int &idxOnRMP);
//use
extern __global__ void kernelFindValidExtension1(Embedding **dEmCol,int* dEmRMP,int noElemdEmCol,int* dArrRMP, int noElemRMP, \
										  int noElemEmbedding, \
										 int *dO,int *dLO,int *dN,int *dLN, float *dArrDegreeOfVid, \
										 int maxDegreeOfVer,int** dPointerArrValid, \
										 EXT** dPointerArrEXT, int minLabel,int maxId, int* dValidBackward,int* dVj);

//extern __global__ void kernelFindValidExtension(Embedding **dPointerdArrEmbedding,int* dArrRMP, int noElemRMP,int noElemEmbedding, \
//										 int *dO,int *dLO,int *dN,int *dLN, float *dArrDegreeOfVid, \
//										 int maxDegreeOfVer,int** dPointerArrValid, \
//										 EXT** dPointerTempArrEXT, int minLabel,int maxId, int* dValidBackward,int* dVj, \
//										 Embedding** dEmCol,int* dEmRMP,int noElemdEmCol);
//
//extern __global__ void kernelFindValidFBExtensionv3(Embedding **dArrPointerEmbedding,int noElem_dArrPointerEmbedding,int noElem_Embedding,int *d_O,int *d_LO,int *d_N,int *d_LN,float *dArrDegreeOfVid,int maxDegreeOfVer,int *dArrV_valid,int *dArrV_backward,EXT *dArrExtension,int *listOfVer,int minLabel,int maxId,int fromRMP, int *dArrVidOnRMP,int segdArrVidOnRMP,int *rmp,int *dArrVj,int noElemdArrVj);
//extern __global__ void kernelFindValidForwardExtensionv3(Embedding **dArrPointerEmbedding,int noElem_dArrPointerEmbedding,int noElem_Embedding,int *d_O,int *d_LO,int *d_N,int *d_LN,float *dArrDegreeOfVid,int maxDegreeOfVer,int *dArrV_valid,int *dArrV_backward,EXT *dArrExtension,int *listOfVer,int minLabel,int maxId,int fromRMP, int *dArrVidOnRMP,int segdArrVidOnRMP,int *rmp);

//use
extern __global__ void kernelSetValuedF_pure(UniEdge *dUniEdge,int noElemdUniEdge,EXT *dValidExtension,int noElemdValidExtension,int *dBScanResult,int *dF,int noElemF);

//extern __device__ bool IsVertexOnEmbedding(int vertex,Embedding **dArrPointerEmbedding,int noElem_dArrPointerEmbedding,int row);
//extern __global__ void kernelGetFromLabelv3(EXT *dArrExt,int *dFromVi,int *dFromLi);
//extern __global__ void kernelCopyResultToUE(UniEdge *fwdArrUniEdge,UniEdge *bwdArrUniEdge,int bwnoElem,UniEdge *uedArrUniEdge,int uenoElem);
//extern __global__ void kernelCopyResultToUE(UniEdge *fwdArrUniEdge,UniEdge *uedArrUniEdge,int uenoElem);

//use
extern __global__ void kernelExtractBWEmbeddingRow(Embedding* dArrBWEmbedding,int *dV,int *dVScanResult,int noElemdV,Embedding *dArrEmbedding);
//use
extern __global__ void	kernelExtractRowFromEXT(EXT *dArrExt,int noElemdArrExt,int *dV,int vj);

//extern __global__ void kernelGetGraphIdContainEmbeddingBW(int vj,EXT *d_ValidExtension,int noElem_d_ValidExtension,int *dV,unsigned int maxOfVer);

//use
extern __global__ void	kernelExtractUniEdgeSatifyMinsupV3(UniEdge *dUniEdge,int *dV,int *dVScanResult,int noElemUniEdge,UniEdge *dUniEdgeSatisfyMinsup,int *dSup,int *dResultSup);

//extern __global__ void kernelFilldArrUniEdgev2(int *dArrAllPossibleExtension,int *dArrAllPossibleExtensionScanResult,int noElem_dArrAllPossibleExtension,UniEdge *dArrUniEdge,int Lv,int *dFromLi,int *dFromVi,int maxId);

//use
extern __global__ void kernelGet_vivjlj(EXT* dArrExt,int* dvi,int* dvj,int* dli,int maxId);
//use
extern __global__ void kernelExtractUniBE(int* dAllExtension,int noElemdAllExtension, \
									int* dRMP,int* dRMPLabel,int Lv,UniEdge* dUniEdge, \
									int* dAllExtensionIdx,int vi,int li);
//use
extern __global__ void kernelMarkUniBE(int* dMappingVj,int* dAllExtension,int Lv,int noElem,EXT* dArrEXT);
//use
extern __global__ void kernelFilldMappingVj(int noElemBW,int* dMappingVj,int* dRMP);
//use
extern __global__ void kernelFillUniFE( int *dArrAllPossibleExtension, \
								int *dArrAllPossibleExtensionScanResult, \
								int noElem_dArrAllPossibleExtension, \
								UniEdge *dArrUniEdge, \
								int Lv,int *dvi, \
								int *dvj,int *dlj);


//extern __global__ void kernelGetFromLabelv2(EXT *dArrExt,int noElem,int *dFromVi,int *dFromLi);
//extern __global__ void kernelextractValidBWExtension(UniEdge *dsrcUniEdge,UniEdge *ddstUniEdge,int noElem,int *dAllPossibleExtension,int *dAllPossibleExtensionScanResult);
//extern __global__ void kernelextractAllBWExtension(EXT *dArrExt,int noElemdArrExt,UniEdge* dArrUniEdge,int *dAllPossibelExtension);
//extern __global__ void kernelmarkValidBackwardEdge_LastExt(EXT* dArrExt, int noElemdArrExt,unsigned int Lv,int *dAllPossibleExtension);

//use
extern __global__ void kernelmarkValidForwardEdge_LastExt(EXT* dArrExt, int noElemdArrExt,unsigned int Lv,int *dAllPossibleExtension);

//extern __global__ void kernelFindVidOnRMP(Embedding **dArrPointerEmbedding,int noElemEmbedding,int *rmp,int noElemVerOnRMP,int *dArrVidOnRMP,int step);
//extern __global__ void kernelFindVidOnRMPv2(Embedding **dArrPointerEmbedding,int noElemEmbedding,int *rmp,int noElemVerOnRMP,int *dArrVidOnRMP,int step);
//extern __global__ void kernel_GetEmbeddings(Embedding **dArrPointerEmbedding,int noElemEmbeddingCol,int noElemEmbedding);
//extern __global__ void kernelDisplaydArrPointerEmbedding(Embedding **dArrPointerEmbedding,int noElemEmbeddingCol,int noElemEmbedding);

//use
extern __global__ void kernelSetValueForEmbeddingColumn(EXT *dArrExt,int noElemInArrExt,Embedding *dArrQ,int *dM,int *dMScanResult);

//extern __global__ void kernelMarkEXT(const EXT *d_ValidExtension,int noElem_d_ValidExtension,int *dV,int vi,int vj,int li,int lij,int lj);

//use
extern void  myReduce(int *dArrInput,int noElem,int &hResult);
//use
extern __global__ void kernelFilldF(UniEdge *dArrUniEdge,int pos,EXT *dArrExt,int noElemdArrExt,int *dArrBoundaryScanResult,int *dF);
//use
extern __global__ void kernelfindBoundary(EXT *dArrExt, int noElemdArrExt, int *dArrBoundary,unsigned int maxOfVer);

//extern __global__ void kernelFilldFbw(UniEdge *dArrUniEdge,int pos,EXT *dArrExt,int noElemdArrExt,int *dArrBoundaryScanResult,int *dF);
//extern __global__ void kernelFindValidFBExtension(Embedding **dArrPointerEmbedding,int noElem_dArrPointerEmbedding,int noElem_Embedding,int *d_O,int *d_LO,int *d_N,int *d_LN,float *dArrDegreeOfVid,int maxDegreeOfVer,int *dArrV_valid,int *dArrV_backward,EXT *dArrExtension,int *listOfVer,int minLabel,int maxId,int fromRMP, int *dArrVidOnRMP,int segdArrVidOnRMP,int *rmp);
//extern __global__ void kernelFindValidFBExtensionv2(Embedding **dArrPointerEmbedding,int noElem_dArrPointerEmbedding,int noElem_Embedding,int *d_O,int *d_LO,int *d_N,int *d_LN,float *dArrDegreeOfVid,int maxDegreeOfVer,int *dArrV_valid,int *dArrV_backward,EXT *dArrExtension,int *listOfVer,int minLabel,int maxId,int fromRMP, int *dArrVidOnRMP,int segdArrVidOnRMP,int *rmp,int *dArrVj,int noElemdArrVj);
//extern __global__ void	kernelExtractFromListVer(int *listVer,int from,int noElemEmbedding,int *temp);
//extern __global__ void kernelFindListVer(Embedding **dArrPointerEmbedding,int noElemEmbedding,int *rmp,int noElemVerOnRMP,int *listVer);
//extern __global__ void kernelPrintdArrPointerEmbedding(Embedding **dArrPointerEmbedding,int noElem,int sizeArr);

//use
extern __global__ void kernelCreatedEmRMP(int* dArrRMP,int* dEmRMP,int noElemRMP);
//use
extern __global__ void	kernelGetPointerdArrEmbedding(Embedding *dArrEmbedding,Embedding **dArrPointerEmbedding,int idx);

//extern __global__ void kernelPrintdArr(int *deviceArray,unsigned int noElem);
//extern __global__ void kernelPrintdArr(int *dArr,int noElem);
//extern __global__ void kernelPrintdArr(float *dArr,int noElem);

//use
extern __global__ void kernelCountNumberOfLabelVertex(int *d_LO,int *d_Lv,unsigned int sizeOfArrayLO);
//use
extern __global__ void kernelGetAndStoreExtension(int *d_O,int *d_LO,unsigned int numberOfElementd_O,int *d_N,int *d_LN,unsigned int numberOfElementd_N,Extension *d_Extension);

//extern __global__ void kernelPrintExtention(Extension *d_Extension, int n);
//extern __global__ void	kernelValidEdge(Extension *d_Extension,int *dV,int numberElementd_Extension);

//use
extern __global__ void kernelGetSize(int *dV,int *dVScanResult,int noElem,int *size);

//extern __global__ void kernelExtractValidExtension(Extension *d_Extension,int *dV,int *dVScanResult,int numberElementd_Extension,Extension *d_ValidExtension);
//extern __global__ void kernelMarkLabelEdge(Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,unsigned int Lv,unsigned int Le,int *d_allPossibleExtension);

//use
extern __global__ void kernelCalcLabelAndStoreUniqueExtension(int *d_allPossibleExtension,int *d_allPossibleExtensionScanResult,unsigned int noElem_allPossibleExtension,UniEdge *d_UniqueExtension,unsigned int Le,unsigned int Lv);

//extern __global__ void kernelCalcBoundary(Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,int *dB,unsigned int maxOfVer);

//use
extern __global__ void kernelGetLastElement(int *dScanResult,unsigned int noElem,int *output);

//extern __global__ void kernelSetValuedF(UniEdge *dUniEdge,int noElemdUniEdge,Extension *dValidExtension,int noElemdValidExtension,int *dBScanResult,float *dF,int noElemF);
//extern __global__ void kernelCopyFromdFtoTempF(int *d_F,int *tempF,int from,int noElemNeedToCopy);

//use
extern __global__ void	kernelMarkUniEdgeSatisfyMinsup(int *dResultSup,int noElemUniEdge,int *dV,unsigned int minsup);

//extern __global__ void	kernelExtractUniEdgeSatifyMinsup(UniEdge *dUniEdge,int *dV,int *dVScanResult,int noElemUniEdge,UniEdge *dUniEdgeSatisfyMinsup,int *dSup,int *dResultSup);
//extern __global__ void kernelGetGraphIdContainEmbedding(int li,int lij,int lj,Extension *d_ValidExtension,int noElem_d_ValidExtension,int *d_arr_graphIdContainEmbedding,unsigned int maxOfVer);
//extern __global__ void kernelGetGraphIdContainEmbedding_pure(int li,int lij,int lj,EXT *d_ValidExtension,int noElem_d_ValidExtension,int *dV,unsigned int maxOfVer);

//use
extern __global__ void kernelGetGraph(int *dV,int noElemdV,int *d_kq,int *dVScanResult);

//extern __global__ void kernelMarkExtension(const Extension *d_ValidExtension, int noElem_d_ValidExtension,int *dV,int li,int lij,int lj);
//extern __global__ void kernelMarkExtension_pure(const EXT *d_ValidExtension,int noElem_d_ValidExtension,int *dV,int li,int lij,int lj);

//use
extern __global__ void kernelSetValueForFirstTwoEmbeddingColumn(const EXT *d_ValidExtension,int noElem_d_ValidExtension,Embedding *dQ1,Embedding *dQ2,int *d_scanResult,int li,int lij,int lj);

//extern __global__ void	kernelPrintEmbedding(Embedding *dArrEmbedding,int noElem);
//extern __global__ void kernelCalDegreeOfVid(int *listOfVer,int *d_O, int numberOfElementd_O,int noElem_Embedding,int numberOfElementd_N,unsigned int maxOfVer,float *dArrDegreeOfVid);
//extern __global__ void kernelCalDegreeOfVidOnEmbeddingColumn(Embedding *dArrEmbedding,int *d_O, int numberOfElementd_O,int noElem_Embedding,int numberOfElementd_N,unsigned int maxOfVer,float *dArrDegreeOfVid);

//use
extern __global__ void kernelCalDegreeOfVidOnEmbeddingColumnv2(Embedding** dPointerEmbedding,int* dEmRMP,int noElemdEmCol, \
									 int *d_O, int numberOfElementd_O,int noElemVid, int noElemEmbedding, \
									 int numberOfElementd_N,unsigned int maxOfVer,float *dArrDegreeOfVid);
//use
extern __global__ void find_maximum_kernel(float *array, float *max, int *mutex, unsigned int n);

//extern __global__ void kernelFindValidForwardExtension(Embedding **dArrPointerEmbedding,int noElem_dArrPointerEmbedding,int noElem_Embedding,int *d_O,int *d_LO,int *d_N,int *d_LN,float *dArrDegreeOfVid,int maxDegreeOfVer,EXT *dArrExtension,int *listOfVer,int minLabel,int maxId,int fromRMP,int *dArrV_valid,int *dArrV_backward);
//extern 	__global__ void printdArrUniEdge(UniEdge *dArrUniEdge,int i);
//extern __global__ void	kernelGetvivj(EXT *dArrEXT,int noElemdArrEXT,int li,int lij,int lj,int *dvi,int *dvj);

//use
extern __global__ void kernelGetLastElementEXT(EXT *inputArray,int noEleInputArray,int *value,unsigned int maxOfVer);
//use
extern __global__ void kernelGetGraphIdContainEmbeddingv2(int vi,int vj,int li,int lij,int lj,EXT *d_ValidExtension,int noElem_d_ValidExtension,int *dV,unsigned int maxOfVer);
//use
extern __global__ void kernelExtractValidExtension_pure(Extension *d_Extension,int *dV,int *dVScanResult,int numberElementd_Extension,EXT *d_ValidExtension);
//use
extern __global__ void kernelMarkLabelEdge_pure(EXT *d_ValidExtension,unsigned int noElem_d_ValidExtension,unsigned int Lv,unsigned int Le,int *d_allPossibleExtension);
//use
extern __global__ void kernelCalcBoundary_pure(EXT *d_ValidExtension,unsigned int noElem_d_ValidExtension,int *dB,unsigned int maxOfVer);
//use
extern __global__ void	kernelExtractUniEdgeSatifyMinsup_pure(UniEdge *dUniEdge,int *dV,int *dVScanResult,int noElemUniEdge,UniEdge *dUniEdgeSatisfyMinsup,int *dSup,int *dResultSup);
extern __global__ void kernelMarkExtension(const EXT *d_ValidExtension,int noElem_d_ValidExtension,int *dV,int li,int lij,int lj);
//use
extern void allocate_gpu_memory(EXT* &d_array,int noElem);

//extern cudaError_t  myScanV_beta();

//use
extern void getLastElementEXT(EXT *inputArray,int numberElementOfInputArray,int &outputValue,unsigned int maxOfVer);

//extern cudaError_t ADM(int *&devicePointer,size_t nBytes);

//use
extern void sumUntilReachZero(int *h_Lv,unsigned int n,int &result); 
//use
extern void validEdge(Extension *d_Extension,int *&dV,unsigned int numberElementd_Extension);
//use
extern void getSizeBaseOnScanResultv2(int *&dV,int *&dVScanResult,int& noElem,int &output);
//use
extern void get_noElem_valid(int*& dV,int*& dVScanResult,int& noElem,int &output);

//extern cudaError_t extractValidExtension(Extension *d_Extension,int *dV,int *dVScanResult, int numberElementd_Extension,Extension *&d_ValidExtension);

//use
extern void extractValidExtension_pure(Extension *d_Extension,int *dV,int *dVScanResult, int numberElementd_Extension,EXT *&d_ValidExtension);

//extern cudaError_t markLabelEdge(Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,unsigned int Lv,unsigned int Le,int *&d_allPossibleExtension);
//extern cudaError_t markLabelEdge_pure(EXT *&d_ValidExtension,unsigned int Lv,unsigned int Le,int *&d_allPossibleExtension);

//use
extern void calcLabelAndStoreUniqueExtension(int *d_allPossibleExtension,int *d_allPossibleExtensionScanResult,unsigned int noElem_allPossibleExtension,UniEdge *&d_UniqueExtension,unsigned int noElem_d_UniqueExtension,unsigned int Le,unsigned int Lv);

//extern cudaError_t calcBoundary(Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,int *&dB,unsigned int maxOfVer);

//use
extern void calcBoundary_pure(EXT *&d_ValidExtension,unsigned int noElem_d_ValidExtension,int *&dB,unsigned int maxOfVer);
//use
extern void getLastElement(int *dScanResult,unsigned int noElem,int &output);

//extern cudaError_t calcSupport(UniEdge *dUniEdge,int noElemdUniEdge,Extension *dValidExtension,int noElemdValidExtension,int *dBScanResult,int *dF,int noElemF,int *&hResultSup);

//use
extern void calcSupport_pure(UniEdge *dUniEdge,int noElemdUniEdge,EXT *dValidExtension,int noElemdValidExtension,int *dBScanResult,int *dF,int noElemF,int *&hResultSup);

//extern cudaError_t getLastElementExtension(Extension* inputArray,unsigned int numberElementOfInputArray,int &outputValue,unsigned int maxOfVer);
//extern cudaError_t getLastElementExtension_pure(EXT* inputArray,unsigned int numberElementOfInputArray,int &outputValue,unsigned int maxOfVer);

//use
extern void get_idx(int*& dArrInput,int& noElem,int*& dResult);

//extern void  myReduction(int *dArrInput,int noElem,int &dResult);
//extern int displayDeviceEXT(EXT *dArrEXT,int noElemdArrEXT);

//use
extern void SegReduce(int* dF,int number_unique_extension,int noElem_of_graph_per_unique_ext,int *&resultDevice);
//use
extern void generate_segment_index(int noElem_of_graph_per_unique_ext,int noElem_unique_ext,int *&SegmentStarts);
//use
extern __global__ void kernel_generate_segment_index(int* SegmentStarts,int noElem_segment,int noElem_of_graph_per_unique_ext);
//use
extern bool fexists(const char *filename);
//use
extern __global__ void kernel_mark_edge(int vi,int vj,int li,int lij,int lj,EXT *ext,int *dValid,int noElem);
