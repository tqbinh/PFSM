#pragma once
#include "pms.cuh"
#include "moderngpu.cuh"		// Include all MGPU kernels.

using namespace mgpu;
StopWatchWin timer;

#define PMS_PRINT std::printf


float hTime=0.0;
float dTime=0.0;

int nthDFSCODE = 0;

//Ghi DFS_CODE xuống file
void write_minDFS_CODE(DFSCode dfscode)
{
	//Mở file minDFSCODE.txt để ghi thêm vào
	char* dfsfile = "minDFSCODE.txt";
	fstream of;
	of.open(dfsfile,ios::out|ios::app);
	if(!of.is_open()){
		cout<<"Open file minDFSCODE.txt fail"<<endl;
		return;
	}

	//Kiểm tra số cạnh của dfscode
	int no_edges_in_dfscode = dfscode.size();
	//cout<<"so canh cua mindfscode: "<<no_edges_in_dfscode<<endl;


	of<<"d "<<nthDFSCODE<<" "<<dfscode.size()<<endl;
	++nthDFSCODE;
	for(int i=0;i<dfscode.size();++i){
		of<<dfscode.at(i).from<<" "<<dfscode.at(i).to<<" "<<dfscode.at(i).fromlabel<<" "<<dfscode.at(i).elabel<<" "<<dfscode.at(i).tolabel<<endl;
	}
}

int nthNotMinDFSCODE =0;
//Ghi DFS_CODE xuống file
void write_notMinDFS_CODE(DFSCode dfscode)
{
	//Mở file minDFSCODE.txt để ghi thêm vào
	char* dfsfile = "notMinDFSCODE.txt";
	fstream of;
	of.open(dfsfile,ios::out|ios::app);
	if(!of.is_open()){
		cout<<"Open file notMinDFSCODE.txt fail"<<endl;
		return;
	}

	//Kiểm tra số cạnh của dfscode
	int no_edges_in_dfscode = dfscode.size();
	//cout<<"so canh cua mindfscode: "<<no_edges_in_dfscode<<endl;

	nthNotMinDFSCODE=nthDFSCODE+1;
	of<<"d "<<nthNotMinDFSCODE<<" "<<dfscode.size()<<endl;
	++nthNotMinDFSCODE;
	for(int i=0;i<dfscode.size();++i){
		of<<dfscode.at(i).from<<" "<<dfscode.at(i).to<<" "<<dfscode.at(i).fromlabel<<" "<<dfscode.at(i).elabel<<" "<<dfscode.at(i).tolabel<<endl;
	}
}


//use
void write_array(int *a, int n,char *filename="temp.csv"){
	//Mở file minDFSCODE.txt để ghi thêm vào
	fstream of;
	//Mở file để ghi thêm vào
	of.open(filename,ios::out|ios::app);

	if(!of.is_open()){
		cout<<"Open file: "<<filename<< " fail"<<endl;
		return;
	}

	for(int i=0;i<n;++i){
		of<<a[i]<<endl;
	}
	of.flush();
	of.close();
	std::printf("\nWrite %s successfully",filename);
}

//use
PMS::PMS()
{
	Lv=0;
	Le=0;
	maxOfVer=0;
	numberOfGraph=0;
	minLabel = -1;
	maxId = -1;	
}

//use
PMS::~PMS()
{
	if(hLevelEXT.size()>0){
		hLevelEXT.clear();
	}

	if(hdb.size()!=0){
		for (int i = 0; i < hdb.size(); i++)
		{
			cudaFree(hdb.at(i).dO);
			cudaFree(hdb.at(i).dLO);
			cudaFree(hdb.at(i).dN);
			cudaFree(hdb.at(i).dLN);
		}
		hdb.clear();
	}
	if(hExtension.size()!=0){
		for (int i = 0; i < hExtension.size(); i++)
		{
			CUCHECK(cudaFree(hExtension.at(i).dExtension));
		}
		hExtension.clear();
	}

	if(hUniEdge.size()!=0){
		for (int i = 0; i < hUniEdge.size(); i++)
		{
			CUCHECK(cudaFree(hUniEdge.at(i).dUniEdge));
		}
		hUniEdge.clear();
	}

	//if(hUniEdgeSatisfyMinsup.size()!=0)
	//{
	//	for (int i = 0; i < hUniEdgeSatisfyMinsup.size(); i++)
	//	{			
	//		cudaFree(hUniEdgeSatisfyMinsup.at(i).dUniEdge);
	//		free(hUniEdgeSatisfyMinsup.at(i).hArrSup);					
	//	}
	//	hUniEdgeSatisfyMinsup.clear();
	//}
}

//use
bool fexists(const char *filename)
{
  ifstream ifile(filename);
  return ifile;
}

//use
void PMS::prepareDataBase()
{
	//unsigned int minsup = 5000;
	unsigned int minsup = 30;
	unsigned int maxpat = 2;
	//unsigned int maxpat = 0x00000000;
	unsigned int minnodes = 0;
	bool where = true;
	bool enc = false;
	bool directed = false;

	//int opt;
	char* fname;
	//fname = "Klesscus";
	//fname = "Klessorigin";
	//fname = "KlessoriginCust1";
	//fname= "G0G1G2_custom"; //Kết quả giống với gSpan
	//fname= "G0G1G2_custom1"; //Kết quả giống với gSpan
	fname="Chemical_340Origin";
	//fname="dbgraph";


	ofstream fout("result.txt");
	char* minDFSCODE = "minDFSCODE.txt";
	char* notMinDFSCODE= "notMinDFSCODE.txt";
	if(fexists(minDFSCODE)==true)
	{
		remove(minDFSCODE);
		cout<<"Xoa file minDFSCODE dang ton tai"<<endl;
	}
	if(fexists(notMinDFSCODE)==true)
	{
		remove(notMinDFSCODE);
		cout<<"Xoa file notMinDFSCODE dang ton tai"<<endl;
	}
	//Chuyển dữ liệu từ fname sang TRANS
	run(fname,fout,minsup,maxpat,minnodes,enc,where,directed);
	maxOfVer=findMaxVertices();
	numberOfGraph=noGraphs();
	int sizeOfarrayO=maxOfVer*numberOfGraph;
	//Tạo mảng arrayO có kích thước D*m
	int* arrayO = new int[sizeOfarrayO];
	if(arrayO==NULL)
	{
		PMS_PRINT("\n!!!Memory Problem ArrayO");
		exit(0);
	}else
	{
		// gán giá trị cho các phần tử mảng bằng -1
		memset(arrayO, -1, sizeOfarrayO*sizeof(int));
	}
	//Tổng bậc của tất cả các đỉnh trong csdl đồ thị TRANS
	unsigned int noDeg;
	noDeg = sumOfDeg();
	unsigned int sizeOfArrayN=noDeg;
	//Mảng arrayN lưu trữ id của các đỉnh kề với đỉnh tương ứng trong mảng arrayO.
	int* arrayN = new int[sizeOfArrayN];
	if(arrayN==NULL)
	{
		PMS_PRINT("\n!!!Memory Problem ArrayN");
		exit(0);
	}else
	{
		memset(arrayN, -1, noDeg*sizeof(int));
	}
	 //Mảng arrayLO lưu trữ label cho tất cả các đỉnh trong TRANS.
	int* arrayLO = new int[sizeOfarrayO];
	if(arrayLO==NULL)
	{
		PMS_PRINT("\n!!!Memory Problem ArrayLO");
		exit(0);
	}else
	{
		memset(arrayLO, -1, sizeOfarrayO*sizeof(int));
	}


	//Mảng arrayLN lưu trữ label của tất cả các cạnh trong TRANS
	int* arrayLN = new int[noDeg];
	if(arrayLN==NULL){
		PMS_PRINT("\n!!!Memory Problem ArrayLN");
		exit(0);
	}else
	{
		memset(arrayLN, -1, noDeg*sizeof(int));
	}

	importDataToArray(arrayO,arrayLO,arrayN,arrayLN,sizeOfarrayO,noDeg,maxOfVer);

	/*write_array(arrayO,sizeOfarrayO,"arrayO.csv");
	write_array(arrayLO,sizeOfarrayO,"arrayLO.csv");
	write_array(arrayN,noDeg,"arrayN.csv");
	write_array(arrayLN,noDeg,"arrayLN.csv");*/

	//kích thước của dữ liệu
	size_t nBytesO = sizeOfarrayO*sizeof(int);
	size_t nBytesN = noDeg*sizeof(int);

	DB graphdb;
	graphdb.noElemdO = sizeOfarrayO;
	graphdb.noElemdN = noDeg;

	CUCHECK(cudaMalloc((void**)&graphdb.dO,nBytesO));
	//Cấp phát bộ nhớ trên GPU được quản lý bởi pointer dLO
	CUCHECK(cudaMalloc((void**)&graphdb.dLO,nBytesO));
	CUCHECK(cudaMalloc((void**)&graphdb.dN,nBytesN));
	CUCHECK(cudaMalloc((void**)&graphdb.dLN,nBytesN));

	//Chép dữ liệu từ mảng arrayO trên CPU sang GPU được quản lý bởi pointer dO
	CUCHECK(cudaMemcpy(graphdb.dO,arrayO,nBytesO,cudaMemcpyHostToDevice));
	//	delete(arrayO);
	CUCHECK(cudaMemcpy(graphdb.dLO,arrayLO,nBytesO,cudaMemcpyHostToDevice));
	//delete(arrayLO);
	CUCHECK(cudaMemcpy(graphdb.dN,arrayN,nBytesN,cudaMemcpyHostToDevice));
	//delete(arrayN);
	CUCHECK(cudaMemcpy(graphdb.dLN,arrayLN,nBytesN,cudaMemcpyHostToDevice));
	//delete(arrayLN);
	//pms.db.push_back(graphdb); //Đưa cơ sở dữ liệu vào vector db
	//pms.countNumberOfDifferentValue(pms.db.at(0).dLO,pms.db.at(0).noElemdO,pms.Lv);
	//pms.countNumberOfDifferentValue(pms.db.at(0).dLN,pms.db.at(0).noElemdN,pms.Le);
	
	hdb.push_back(graphdb); //Đưa cơ sở dữ liệu vào vector db
	countNumberOfDifferentValue(hdb.at(0).dLO,hdb.at(0).noElemdO,Lv);
	countNumberOfDifferentValue(hdb.at(0).dLN,hdb.at(0).noElemdN,Le);
	//pms.printdb();
	return;
}

//bool PMS::checkArray(int *hostRef, int *gpuRef, const int N) {
//	bool result=true;
//	double epsilon = 1.0E-8;
//	int match = 1;
//	for (int i = 0; i < N; i++) {
//		if ((float)(abs(hostRef[i] - gpuRef[i])) > epsilon) {
//			match = 0;
//			result=false;
//			PMS_PRINT("Arrays do not match!\n");
//			PMS_PRINT("host %5.2f gpu %5.2f at current %d\n",
//				hostRef[i], gpuRef[i], i);
//			break;
//		}
//	}
//	if (match){
//		PMS_PRINT("Arrays match.\n\n");		
//	}
//
//	return result;
//}


//void PMS::displayArray(int *p, const unsigned int pSize=0)
//{
//	for(int i=0;i<pSize;i++){
//		PMS_PRINT("P[%d]:%d ",i,p[i]);
//	}
//	PMS_PRINT("\n");
//	return;
//}

//use
void PMS::displayHostArray(int *&p, const unsigned int pSize=0)
{
	std::printf("\n");
	for(int i=0;i<pSize;i++){
		PMS_PRINT("[%d]:%d ",i,p[i]);
	}
	PMS_PRINT("\n");
	return;
}

//use
__global__ void kernelCopyDeviceArray(int *dArrInput,int *dResult,int noElem)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i<noElem)
	{
		dResult[i]=dArrInput[i];
	}
}

//use
__global__ void kernelCopyDevice(int** dPointerArr,int* dArr,int at)
{
	dPointerArr[at] = dArr;
	//PMS_PRINT("\n dPointerArr:%d, dArr:%d",dPointerArr[at],dArr);
}

//use
__global__ void kernelCopyDeviceEXT(EXT** dPointerArr,EXT* dArr,int at)
{
	dPointerArr[at] = dArr;
}

//use
void myReduce(int *dArrInput,int noElem,int &hResult)
{
	CudaContext& cdactx = *ctx;
	hResult = Reduce(dArrInput,noElem,cdactx);
	/*cout<<"reduce output: "<<hResult<<endl;*/
}

//use
void  myScanV(int *dArrInput,int noElem,int *&dResult)
{
	dim3 block(blocksize);
	dim3 grid((noElem + block.x -1)/block.x);

	CUCHECK(cudaMalloc((void**)&dResult,noElem * sizeof(int)));
	//Copy dArrInput to dResult
	kernelCopyDeviceArray<<<grid,block>>>(dArrInput,dResult,noElem);
	CUCHECK(cudaDeviceSynchronize());
	CudaContext& cdactx = *ctx;
	mgpu::ScanExc(dResult, noElem,cdactx);
	CUCHECK(cudaDeviceSynchronize());
	CUCHECK(cudaGetLastError());

	return;
}

//use
void get_idx(int*& dArrInput,int& noElem,int*& dResult)
{
	dim3 block(blocksize);
	dim3 grid((noElem + block.x -1)/block.x);

	CUCHECK(cudaMalloc((void**)&dResult,noElem * sizeof(int)));

	//Copy dArrInput to dResult
	CUCHECK(cudaMemcpy(dResult,dArrInput,noElem*sizeof(int),cudaMemcpyDeviceToDevice))

	//displayDeviceArr(dResult,noElem);
	CudaContext& cdactx = *ctx;
	mgpu::ScanExc(dResult, noElem,cdactx);
	//displayDeviceArr(dResult,noElem);
	CUCHECK(cudaDeviceSynchronize());
	CUCHECK(cudaGetLastError());
	return;
}

//use
void  myReduction(int *dArrInput,int noElem,int &hResult){

	CudaContext& cdactx = *ctx;
	int total = Reduce(dArrInput, noElem, cdactx);
	//PMS_PRINT("Reduction total: %d\n\n", total);
	hResult = total;

	return;
}


//use
__global__ void kernelCountNumberOfLabelVertex(int *d_LO,int *d_Lv,unsigned int sizeOfArrayLO){
	int i= blockDim.x*blockIdx.x + threadIdx.x;
	if(i<sizeOfArrayLO){
		if(d_LO[i]!=-1){
			d_Lv[d_LO[i]]=1;
		}
	}
}

//chưa sửa đối tham chiếu

//use
void sumUntilReachZero(int *h_Lv,unsigned int n,int &result)
{
	for(int i=0;i<n && h_Lv[i]!=0;++i)
	{
		++result;
	}
}

//chưa sửa thành đối tham chiếu

//use
void  PMS::countNumberOfDifferentValue(int* d_LO,unsigned int sizeOfArrayLO, unsigned int &numberOfSaperateVertex){
	numberOfSaperateVertex=0;
	size_t nBytesLv = sizeOfArrayLO*sizeof(int);
	//cấp phát mảng d_Lv trên device
	int *d_Lv;
	CUCHECK(cudaMalloc((int**)&d_Lv,nBytesLv));
	CUCHECK(cudaMemset(d_Lv,0,nBytesLv));
	
	//Cấp phát threads
	dim3 block(blocksize);
	dim3 grid((sizeOfArrayLO+block.x-1)/block.x);
	kernelCountNumberOfLabelVertex<<<grid,block>>>(d_LO,d_Lv,sizeOfArrayLO);
	CUCHECK(cudaDeviceSynchronize());
	CUCHECK(cudaGetLastError());

	int* h_Lv=NULL;
	h_Lv=(int*)malloc(nBytesLv);
	if(h_Lv==NULL)
	{
		PMS_PRINT("h_Lv malloc memory fail");
		exit(0);
	}
	CUCHECK(cudaMemcpy(h_Lv,d_Lv,nBytesLv,cudaMemcpyDeviceToHost));
	
	int result=0;
	sumUntilReachZero(h_Lv,sizeOfArrayLO,result);
	numberOfSaperateVertex=result;

	CUCHECK(cudaFree(d_Lv));
	return;
}

//use
__global__ void kernelGetAndStoreExtension(int *d_O,int *d_LO,unsigned int numberOfElementd_O, \
										   int *d_N,int *d_LN,unsigned int numberOfElementd_N, \
										   Extension *d_Extension)
{
	//Kernel trích tất cả các mở rộng hợp lệ ban đầu vào mảng d_Extension
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i<numberOfElementd_O)
	{
		if (d_O[i]!=-1)
		{
			int j;
			int ek;
			for(j=i+1;j<numberOfElementd_O;++j)
			{
				if(d_O[j]!=-1) {break;}
			}
			if (j==numberOfElementd_O)
			{
				ek=numberOfElementd_N;
			}
			else
			{
				ek=d_O[j];
			}
			for(int k=d_O[i];k<ek;k++){
				int index= k;
				d_Extension[index].vi=0;
				d_Extension[index].vj=1;
				d_Extension[index].li=d_LO[i];
				d_Extension[index].lij=d_LN[k];
				d_Extension[index].lj=d_LO[d_N[k]];
				d_Extension[index].vgi=i;
				d_Extension[index].vgj=d_N[k];
			}
		}
	}
}

//use
void PMS::getAndStoreExtension(Extension *&d_Extension)
{
	dim3 block(blocksize);
	unsigned int numberOfElementd_O = hdb.at(0).noElemdO;
	dim3 grid((numberOfElementd_O+block.x-1)/block.x);

	kernelGetAndStoreExtension<<<grid,block>>>( \
		hdb.at(0).dO,hdb.at(0).dLO, \
		numberOfElementd_O, \
		hdb.at(0).dN,hdb.at(0).dLN,hdb.at(0).noElemdN, \
		d_Extension);
	CUCHECK(cudaDeviceSynchronize());
	CUCHECK(cudaGetLastError());
	return;
}

//use
void PMS::extractAllEdgeInDB()
{
	arrExtension arrE;
	//cấp phát bộ nhớ cho d_Extension
	arrE.noElem =hdb.at(0).noElemdN; //Lấy số lượng cạnh của tất cả các đồ thị
	size_t nBytesOfArrayExtension = arrE.noElem*sizeof(Extension); //Cấp phát bộ nhớ để lưu trữ tất cả các mở rộng ban đầu tương ứng với số lượng cạnh thu được;

	CUCHECK(cudaMalloc((Extension**)&arrE.dExtension,nBytesOfArrayExtension));

	//Trích tất cả các cạnh từ database rồi lưu vào d_Extension
	getAndStoreExtension(arrE.dExtension);
	hExtension.push_back(arrE);
	return;
}

//use
__global__ void	kernelValidEdge(Extension *d_Extension,int *dV,unsigned int numberElementd_Extension){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i<numberElementd_Extension){	
		if(d_Extension[i].li<=d_Extension[i].lj){
			dV[i]=1;
		}
	}
}

//use
void validEdge(Extension *d_Extension,int *&dV,unsigned int numberElementd_Extension)
{
	dim3 block(blocksize);
	dim3 grid((numberElementd_Extension+block.x-1)/block.x);
	std::printf("\n gird:%d block:%d");
	kernelValidEdge<<<grid,block>>>(d_Extension,dV,numberElementd_Extension);
	CUCHECK(cudaDeviceSynchronize());
	CUCHECK(cudaGetLastError());
	return;
}

//use
void PMS::displayDeviceArr(int *dArr,int noElem)
{
	int *temp = (int*)malloc(sizeof(int)*noElem);
	if(temp==NULL)
	{
		PMS_PRINT("\n Malloc temp in displayDeviceArr() failed");
		FCHECK(-1);
	}
	CUCHECK(cudaMemcpy(temp,dArr,noElem*sizeof(int),cudaMemcpyDeviceToHost));
	for (int i = 0; i < noElem; i++)
	{
		PMS_PRINT(" A[%d]:%d  ",i,temp[i]);
	}
	free(temp);
	return;
}


//use
void PMS::displayDeviceArr(float* &dArr,int &noElem)
{
	try
	{
		float *temp = (float*)malloc(sizeof(float)*noElem);
		if(temp == nullptr){FCHECK(-1);}
		CUCHECK(cudaMemcpy(temp,dArr,noElem*sizeof(float),cudaMemcpyDeviceToHost));
		cout<<endl;
		for (int i = 0; i < noElem; i++)
		{
			int a = (int)temp[i];
			PMS_PRINT(" A[%d]:%d  ",i,a);
		}
		free(temp);
	}
	catch(...)
	{
		FCHECK(-1);
	}
}

//use
__global__ void kernelGetSize(int *dV,int *dVScanResult,int noElem,int *size)
{
	*size = dVScanResult[noElem-1];
	if(dV[noElem-1]==1)
	{
		*size = *size + 1;
	}
}

//chưa sửa đối tham chiếu

//use
void getSizeBaseOnScanResult(int *dV,int *dVScanResult,int noElem,int &output)
{
	int temp=0;
	int *size=nullptr;
	CUCHECK(cudaMalloc((void**)&size,sizeof(int)));
	CUCHECK(cudaMemset(size,0,sizeof(int)));
	
	kernelGetSize<<<1,1>>>(dV,dVScanResult,noElem,size);
	CUCHECK(cudaDeviceSynchronize());
	CUCHECK(cudaGetLastError());
	CUCHECK(cudaMemcpy(&temp,size,sizeof(int),cudaMemcpyDeviceToHost));
	output = (int)temp;

	CUCHECK(cudaFree(size));
	return;
}

//use
void getSizeBaseOnScanResultv2(int *&dV,int *&dVScanResult,int& noElem,int &output)
{
	int temp=0;
	int *size=nullptr;
	CUCHECK(cudaMalloc((void**)&size,sizeof(int)));
	CUCHECK(cudaMemset(size,0,sizeof(int)));

	kernelGetSize<<<1,1>>>(dV,dVScanResult,noElem,size);
	CUCHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError());

	CUCHECK(cudaMemcpy(&temp,size,sizeof(int),cudaMemcpyDeviceToHost));
	output = (int)temp;

	CUCHECK(cudaFree(size));
	return;
}

//use
void get_noElem_valid(int*& dV,int*& dVScanResult,int& noElem,int &output)
{
	int temp=0;
	int *size=nullptr;
	CUCHECK(cudaMalloc((void**)&size,sizeof(int)));
	CUCHECK(cudaMemset(size,0,sizeof(int)));

	kernelGetSize<<<1,1>>>(dV,dVScanResult,noElem,size);
	CUCHECK(cudaDeviceSynchronize());
	CUCHECK(cudaGetLastError());

	CUCHECK(cudaMemcpy(&temp,size,sizeof(int),cudaMemcpyDeviceToHost));
	output = (int)temp;
	CUCHECK(cudaFree(size));
	return;
}

//use
void allocate_gpu_memory(EXT* &d_array,int noElem)
{
	size_t n_bytes = sizeof(EXT)*noElem;
	CUCHECK(cudaMalloc((void**)&d_array,n_bytes));
	return;
}


//use
__global__ void kernelExtractValidExtension_pure(Extension *d_Extension,int *dV,int *dVScanResult, \
												 int numberElementd_Extension,EXT *d_ValidExtension)
{
	//Trích các mở rộng duy nhất ban đầu
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i<numberElementd_Extension){
		if(dV[i]==1){
			int index = dVScanResult[i];
			//PMS_PRINT("\nV[%d]:%d, index[%d]:%d,d_Extension[%d], d_Extension[%d]:%d\n",i,V[i],i,index[i],i,i,d_Extension[i].vgi);
			d_ValidExtension[index].li=d_Extension[i].li;
			d_ValidExtension[index].lj=d_Extension[i].lj;
			d_ValidExtension[index].lij=d_Extension[i].lij;
			d_ValidExtension[index].vgi=d_Extension[i].vgi;
			d_ValidExtension[index].vgj=d_Extension[i].vgj;
			d_ValidExtension[index].vi=d_Extension[i].vi;
			d_ValidExtension[index].vj=d_Extension[i].vj;
			d_ValidExtension[index].posRow = -1; //posRow ban đầu chưa gắn với bất kỳ embedding column row nào.
		}
	}
}

//use
void extractValidExtension_pure(Extension *d_Extension,int *dV,int *dVScanResult, int numberElementd_Extension,EXT *&d_ValidExtension)
{
	dim3 block(blocksize);
	dim3 grid((numberElementd_Extension+block.x)/block.x);
	kernelExtractValidExtension_pure<<<grid,block>>>(d_Extension,dV,dVScanResult,numberElementd_Extension,d_ValidExtension);
	CUCHECK(cudaDeviceSynchronize());
	CUCHECK(cudaGetLastError());
	return;
}

//use
void PMS::getValidExtension_pure()
{
	//Phase 1: đánh dấu vị trí những cạnh hợp lệ (li<=lj)
	int *dV;
	size_t nBytesdV= hExtension.at(0).noElem *sizeof(int);
	CUCHECK(cudaMalloc((void**)&dV,nBytesdV));
	CUCHECK(cudaMemset(dV,0,nBytesdV));

	//Đánh dấu các mở rộng hợp lệ trong hExtension.at(0).dExtension
	validEdge(hExtension.at(0).dExtension,dV,hExtension.at(0).noElem);

	int* dVScanResult;
	CUCHECK(cudaMalloc((void**)&dVScanResult,hExtension.at(0).noElem*sizeof(int)));
	CUCHECK(cudaMemset(dVScanResult,0,hExtension.at(0).noElem*sizeof(int)));
	myScanV(dV,hExtension.at(0).noElem,dVScanResult);

	hLevelEXT.resize(1); 
	hLevelEXT.at(0).noElem=1;
	hLevelEXT.at(0).vE.resize(1);

	myReduction(dV,hExtension.at(0).noElem,hLevelEXT.at(0).vE.at(0).noElem);
	allocate_gpu_memory(hLevelEXT.at(0).vE.at(0).dArrExt,hLevelEXT.at(0).vE.at(0).noElem);
	extractValidExtension_pure(hExtension.at(0).dExtension,dV,dVScanResult,hExtension.at(0).noElem,hLevelEXT.at(0).vE.at(0).dArrExt);
	//free memory
	CUCHECK(cudaFree(dV));
	CUCHECK(cudaFree(dVScanResult));
	return;
}

//use
__global__ void kernelMarkLabelEdge_pure(EXT *d_ValidExtension, \
										 unsigned int noElem_d_ValidExtension, \
										 unsigned int Lv,unsigned int Le, \
										 int *d_allPossibleExtension)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i<noElem_d_ValidExtension)
	{
		int index = d_ValidExtension[i].li*Lv*Le + d_ValidExtension[i].lij*Lv + d_ValidExtension[i].lj;
		d_allPossibleExtension[index]=1;
	}
}

//use
void markLabelEdge_pure(EXT *&d_ValidExtension,unsigned int noElem_d_ValidExtension,unsigned int Lv,unsigned int Le, \
						int *&d_allPossibleExtension)
{
	//Các cạnh mở rộng hợp lệ có thể giống nhau==> Hàm này sẽ đi ánh xạ chúng vào không gian d_allPossibleExtension.
	dim3 block(blocksize);
	dim3 grid((noElem_d_ValidExtension+block.x-1)/block.x);

	kernelMarkLabelEdge_pure<<<grid,block>>>(d_ValidExtension,noElem_d_ValidExtension,Lv,Le,d_allPossibleExtension);
	CUCHECK(cudaDeviceSynchronize());
	CUCHECK(cudaGetLastError());

	return;
}

//use
__global__ void kernelCalcLabelAndStoreUniqueExtension(int *d_allPossibleExtension,int *d_allPossibleExtensionScanResult, \
													   unsigned int noElem_allPossibleExtension,UniEdge *d_UniqueExtension, \
													   unsigned int Le,unsigned int Lv)
{
	//Ánh xạ từ vị trí trong d_allPossibleExtension sang cạnh tương ứng trong UniEdge 
	int i=blockIdx.x*blockDim.x + threadIdx.x;
	if(i<noElem_allPossibleExtension && d_allPossibleExtension[i]==1){
		int li,lj,lij;
		li=i/(Le*Lv);
		lij=(i%(Le*Lv))/Lv;
		lj=(i%(Le*Lv))-((i%(Le*Lv))/Lv)*Lv;
		int index = d_allPossibleExtensionScanResult[i];
		//PMS_PRINT("\n[%d]:%d li:%d lij:%d lj:%d",i,d_allPossibleExtensionScanResult[i],li,lij,lj);
		d_UniqueExtension[index].li=li;
		d_UniqueExtension[index].lij=lij;
		d_UniqueExtension[index].lj=lj;
	}
}


//use
void calcLabelAndStoreUniqueExtension(int *d_allPossibleExtension,int *d_allPossibleExtensionScanResult, \
									  unsigned int noElem_allPossibleExtension,UniEdge *&d_UniqueExtension, \
									  unsigned int noElem_d_UniqueExtension,unsigned int Le,unsigned int Lv)
{
	//Ánh xạ và lưu cạnh vào dUniEdge từ vị trí có giá trị 1 trong d_allPossibleExtension
	dim3 block(blocksize);
	dim3 grid((noElem_allPossibleExtension+block.x-1)/block.x);
	kernelCalcLabelAndStoreUniqueExtension<<<grid,block>>>(d_allPossibleExtension,d_allPossibleExtensionScanResult,noElem_allPossibleExtension,d_UniqueExtension,Le,Lv);

	CUCHECK(cudaDeviceSynchronize());
	CUCHECK(cudaGetLastError());

	return;
}


//use
void PMS::extractUniEdge()
{
	//Trích các mở rộng duy nhất
	//Tính số lượng tất cả các cạnh có thể có dựa vào nhãn của chúng
	unsigned int noElem_dallPossibleExtension=Le*Lv*Lv; //(Mỗi một đỉnh sẽ có thể có Le*Lv mở rộng. Mà chúng ta có Lv đỉnh, nên ta có: Le*Lv*Lv mở rộng có thể có).
	int *d_allPossibleExtension;

	//cấp phát bộ nhớ cho mảng d_allPossibleExtension
	CUCHECK(cudaMalloc((void**)&d_allPossibleExtension, noElem_dallPossibleExtension*sizeof(int)));
	CUCHECK(cudaMemset(d_allPossibleExtension, 0, noElem_dallPossibleExtension*sizeof(int)));
	
	//Hàm markLabelEdge hoạt động theo nguyên tắc: "Mỗi mở rộng trong dExtension đều có 1 vị trí duy nhất trong d_allPossibleExtension. Và nhiệm vụ của hàm này là bậc giá trị 1 cho vị trí đó"
	//CHECK(cudaStatus=markLabelEdge(hValidExtension.at(0).dExtension,hValidExtension.at(0).noElem,Lv,Le,d_allPossibleExtension)); 
	markLabelEdge_pure(hLevelEXT.at(0).vE.at(0).dArrExt,hLevelEXT.at(0).vE.at(0).noElem,Lv,Le,d_allPossibleExtension);

	int *d_allPossibleExtensionScanResult;
	CUCHECK(cudaMalloc((void**)&d_allPossibleExtensionScanResult,noElem_dallPossibleExtension*sizeof(int)));

	myScanV(d_allPossibleExtension,noElem_dallPossibleExtension,d_allPossibleExtensionScanResult);

	arrUniEdge strUniEdge;
	int noElem_d_UniqueExtension=0;
	//Tính kích thước của mảng d_UniqueExtension dựa vào kết quả exclusive scan
	getSizeBaseOnScanResult(d_allPossibleExtension,d_allPossibleExtensionScanResult,noElem_dallPossibleExtension,noElem_d_UniqueExtension);

	strUniEdge.noElem = noElem_d_UniqueExtension;
	//Tạo mảng d_UniqueExtension với kích thước mảng vừa tính được
	CUCHECK(cudaMalloc((void**)&strUniEdge.dUniEdge,noElem_d_UniqueExtension*sizeof(UniEdge)));
	CUCHECK(cudaMemset(strUniEdge.dUniEdge,0,noElem_d_UniqueExtension*sizeof(UniEdge)));
	
	//Ánh xạ ngược lại từ vị trí trong d_allPossibleExtension thành cạnh và lưu kết quả vào d_UniqueExtension
	calcLabelAndStoreUniqueExtension(d_allPossibleExtension,d_allPossibleExtensionScanResult,noElem_dallPossibleExtension,strUniEdge.dUniEdge,noElem_d_UniqueExtension,Le,Lv);

	hUniEdge.push_back(strUniEdge);
	CUCHECK(cudaFree(d_allPossibleExtension));
	CUCHECK(cudaFree(d_allPossibleExtensionScanResult));
	return;
}

//use
__global__ void kernelCalcBoundary_pure(EXT *d_ValidExtension,unsigned int noElem_d_ValidExtension,int *dB,unsigned int maxOfVer)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i<noElem_d_ValidExtension-1){
		unsigned int graphIdAfter=d_ValidExtension[i+1].vgi/maxOfVer;
		unsigned int graphIdCurrent=d_ValidExtension[i].vgi/maxOfVer;
		unsigned int resultDiff=graphIdAfter-graphIdCurrent;
		dB[i]=resultDiff;
	}
}


//use
void calcBoundary_pure(EXT *&d_ValidExtension,unsigned int noElem_d_ValidExtension,int *&dB,unsigned int maxOfVer)
{
	//Xây dựng boundary cho các mở rộng hợp lệ trong d_ValidExtension để tính support
	dim3 block(blocksize);
	dim3 grid((noElem_d_ValidExtension+block.x)/block.x);

	kernelCalcBoundary_pure<<<grid,block>>>(d_ValidExtension,noElem_d_ValidExtension,dB,maxOfVer);

	CUCHECK(cudaDeviceSynchronize());
	CUCHECK(cudaGetLastError());
	return;
}

//use
__global__ void kernelGetLastElement(int *dScanResult,unsigned int noElem,int *output)
{
	output[0]=dScanResult[noElem-1];
}

//use
void getLastElement(int *dScanResult,unsigned int noElem,int &output)
{
	dim3 block(blocksize);
	dim3 grid((noElem+block.x-1)/block.x);

	int *value=nullptr;
	CUCHECK(cudaMalloc((int**)&value,sizeof(int)));

	kernelGetLastElement<<<1,1>>>(dScanResult,noElem,value);
	CUCHECK(cudaDeviceSynchronize());
	CUCHECK(cudaGetLastError());

	CUCHECK(cudaMemcpy(&output,value,sizeof(int),cudaMemcpyDeviceToHost));

	//PMS_PRINT("\n\nnumberElementd_UniqueExtension:%d",numberElementd_UniqueExtension);

	CUCHECK(cudaFree(value));
	return;
}

//use
__global__ void kernelSetValuedF_pure(UniEdge *dUniEdge,int noElemdUniEdge,EXT *dValidExtension,int noElemdValidExtension,int *dBScanResult,int *dF,int noElemF)
{
	int i = blockDim.x * blockIdx.x +threadIdx.x; //i là các mở rộng hợp lệ
	if(i<noElemdValidExtension){
		for (int j = 0; j < noElemdUniEdge; j++)//j là các mở rộng duy nhất
		{
			if(dUniEdge[j].li==dValidExtension[i].li && dUniEdge[j].lij==dValidExtension[i].lij && dUniEdge[j].lj==dValidExtension[i].lj){ //Nếu mở rộng hợp lệ có trong mở rộng duy nhất (? hình như sai sai)
				dF[dBScanResult[i]+j*noElemF]=1; //Bật 1 tại vị trí tương ứng.
			}
		}
	}
}


//Chưa sửa lại biến là tham chiếu

//use
void calcSupport_pure(UniEdge *dUniEdge,int noElemdUniEdge,EXT *dValidExtension,int noElemdValidExtension,int *dBScanResult,int *dF,int noElemF,int *&hResultSup)
{
	//Tính support cho các mở rộng duy nhất
	//Đánh dấu những đồ thị chứa embedding trong mảng d_F
	dim3 block(blocksize);
	dim3 grid((noElemdValidExtension+block.x - 1)/block.x);
	kernelSetValuedF_pure<<<grid,block>>>(dUniEdge,noElemdUniEdge,dValidExtension,noElemdValidExtension,dBScanResult,dF,noElemF);
	CUCHECK(cudaDeviceSynchronize());
	CUCHECK(cudaGetLastError());
	
	hResultSup = (int*)malloc(noElemdUniEdge*sizeof(int));
	if (hResultSup==NULL)
	{
		PMS_PRINT("\n Malloc hResultSup in calcSupport() failed");
		exit(0);
	}

	int *d_supports = nullptr;
	//timer.start();
	int status=0;
	SegReduce(dF,noElemdUniEdge,noElemF,d_supports);
	/*timer.stop();
	std::printf("Time myReduction for Segmented Extension: %f (ms)\n",timer.getTime());
	timer.reset();*/
	//displayDeviceArr(d_supports,noElemdUniEdge);
	CUCHECK(cudaMemcpy(hResultSup,d_supports,noElemdUniEdge*sizeof(int),cudaMemcpyDeviceToHost));
	CUCHECK(cudaFree(d_supports));
	return;
}

//use
__global__ void	kernelMarkUniEdgeSatisfyMinsup(int *dResultSup,int noElemUniEdge,int *dV,unsigned int minsup){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElemUniEdge){
		int temp = dResultSup[i];
		if(temp >= minsup){
			dV[i]=1;
		}
	}
}

//use
__global__ void	kernelExtractUniEdgeSatifyMinsup_pure(UniEdge *dUniEdge,int *dV,int *dVScanResult,int noElemUniEdge,UniEdge *dUniEdgeSatisfyMinsup,int *dSup,int *dResultSup)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElemUniEdge){
		if(dV[i]==1){
			dUniEdgeSatisfyMinsup[dVScanResult[i]].vi = 0;
			dUniEdgeSatisfyMinsup[dVScanResult[i]].vj = 1;
			dUniEdgeSatisfyMinsup[dVScanResult[i]].li = dUniEdge[i].li;
			dUniEdgeSatisfyMinsup[dVScanResult[i]].lij = dUniEdge[i].lij;
			dUniEdgeSatisfyMinsup[dVScanResult[i]].lj=dUniEdge[i].lj;
			dSup[dVScanResult[i]]=dResultSup[i];
		}
	}
}

//use
__global__ void	kernelExtractUniEdgeSatifyMinsupV3(UniEdge *dUniEdge,int *dV,int *dVScanResult,int noElemUniEdge, \
												   UniEdge *dUniEdgeSatisfyMinsup,int *dSup,int *dResultSup)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElemUniEdge)
	{
		if(dV[i]==1)
		{
			dUniEdgeSatisfyMinsup[dVScanResult[i]].vi = dUniEdge[i].vi;
			dUniEdgeSatisfyMinsup[dVScanResult[i]].vj = dUniEdge[i].vj;
			dUniEdgeSatisfyMinsup[dVScanResult[i]].li = dUniEdge[i].li;
			dUniEdgeSatisfyMinsup[dVScanResult[i]].lij = dUniEdge[i].lij;
			dUniEdgeSatisfyMinsup[dVScanResult[i]].lj=dUniEdge[i].lj;
			dSup[dVScanResult[i]]=dResultSup[i];
		}
	}
}

//chưa sửa lại đối tham chiếu

//use
void PMS::extractUniEdgeSatisfyMinsup(int *hResultSup,int noElemUniEdge,unsigned int minsup)
{
	//1. Cấp phát mảng trên device có kích thước bằng noElemUniEdge
	int *dResultSup=nullptr;
	CUCHECK(cudaMalloc((void**)&dResultSup,noElemUniEdge*sizeof(int)));

	CUCHECK(cudaMemcpy(dResultSup,hResultSup,noElemUniEdge*sizeof(int),cudaMemcpyHostToDevice));

	//2. Đánh dấu 1 trên dV cho những phần tử thoả minsup
	int *dV=nullptr;
	CUCHECK(cudaMalloc((void**)&dV,noElemUniEdge*sizeof(int)));
	
	CUCHECK(cudaMemset(dV,0,sizeof(int)*noElemUniEdge));

	dim3 block(blocksize);
	dim3 grid((noElemUniEdge + block.x - 1)/block.x);

	kernelMarkUniEdgeSatisfyMinsup<<<grid,block>>>(dResultSup,noElemUniEdge,dV,minsup);
	CUCHECK(cudaDeviceSynchronize());
	CUCHECK(cudaGetLastError());


	int *dVScanResult=nullptr;
	CUCHECK(cudaMalloc((void**)&dVScanResult,noElemUniEdge*sizeof(int)));
	myScanV(dV,noElemUniEdge,dVScanResult);

	hLevelUniEdgeSatisfyMinsup.resize(1);
	hLevelUniEdgeSatisfyMinsup.at(0).vecUES.resize(1);
	getSizeBaseOnScanResult(dV,dVScanResult,noElemUniEdge, \
		hLevelUniEdgeSatisfyMinsup.at(0).vecUES.at(0).noElem);

	CUCHECK(cudaMalloc((void**)&hLevelUniEdgeSatisfyMinsup.at(0).vecUES.at(0).dArrUniEdge, \
		hLevelUniEdgeSatisfyMinsup.at(0).vecUES.at(0).noElem*sizeof(UniEdge)));
	
	hLevelUniEdgeSatisfyMinsup.at(0).vecUES.at(0).hArrSupport = (int*)malloc(sizeof(int)*hLevelUniEdgeSatisfyMinsup.at(0).vecUES.at(0).noElem);
	if (hLevelUniEdgeSatisfyMinsup.at(0).vecUES.at(0).hArrSupport ==NULL)
	{
		PMS_PRINT("\n malloc hArrSup of hUniEdgeSatisfyMinsup failed()");
		exit(0);
	}

	int *dSup=nullptr;
	CUCHECK(cudaMalloc((void**)&dSup,hLevelUniEdgeSatisfyMinsup.at(0).vecUES.at(0).noElem*sizeof(int)));

	dim3 blocka(blocksize);
	dim3 grida((noElemUniEdge + blocka.x -1)/blocka.x);
	kernelExtractUniEdgeSatifyMinsup_pure<<<grida,blocka>>>(hUniEdge.at(0).dUniEdge,dV, \
		dVScanResult,noElemUniEdge, \
		hLevelUniEdgeSatisfyMinsup.at(0).vecUES.at(0).dArrUniEdge,dSup,dResultSup);
	CUCHECK(cudaDeviceSynchronize());
	CUCHECK(cudaGetLastError());

	CUCHECK(cudaMemcpy(hLevelUniEdgeSatisfyMinsup.at(0).vecUES.at(0).hArrSupport,dSup, \
		sizeof(int)*hLevelUniEdgeSatisfyMinsup.at(0).vecUES.at(0).noElem,cudaMemcpyDeviceToHost));
	
	CUCHECK(cudaFree(dResultSup));
	CUCHECK(cudaFree(dV));
	CUCHECK(cudaFree(dVScanResult));
	CUCHECK(cudaFree(dSup));
	if(hUniEdge.at(0).noElem>0)
	{
		CUCHECK(cudaFree(hUniEdge.at(0).dUniEdge));
		hUniEdge.clear();
	}
	return;
}

//use
void PMS::computeSupport()
{
	/* Xây dựng Boundary cho mảng d_ValidExtension */
	//1. Cấp phát một mảng d_B và gán các giá trị 0 cho mọi phần tử của d_B
	unsigned int noElement_dB=hLevelEXT.at(0).vE.at(0).noElem;
	int* dB = nullptr;
	CUCHECK(cudaMalloc((int**)&dB,noElement_dB*sizeof(int)));
	CUCHECK(cudaMemset(dB,0,noElement_dB*sizeof(int)));

	
	//Gián giá trị boundary cho d_B
	calcBoundary_pure(hLevelEXT.at(0).vE.at(0).dArrExt,noElement_dB,dB,maxOfVer);

	//2. Exclusive Scan mảng d_B
	int* dBScanResult;
	CUCHECK(cudaMalloc((int**)&dBScanResult,noElement_dB*sizeof(int)));
	CUCHECK(cudaMemset(dBScanResult,0,noElement_dB*sizeof(int)));
	
	myScanV(dB,noElement_dB,dBScanResult);
	

	//3. Tính độ hỗ trợ cho các mở rộng trong d_UniqueExtension
	//3.1 Tạo mảng d_F có số lượng phần tử bằng với giá trị cuối cùng của mảng d_scanB_Result cộng 1 và gán giá trị 0 cho các phần tử.
	int noElemF=0;
	getLastElement(dBScanResult,noElement_dB,noElemF);
	++noElemF;

	int noElem_d_UniqueExtension= hUniEdge.at(0).noElem;
	int *dF;
	CUCHECK(cudaMalloc((int**)&dF,noElem_d_UniqueExtension*noElemF*sizeof(int)));
	CUCHECK(cudaMemset(dF,0,noElem_d_UniqueExtension*noElemF*sizeof(int)));
	
	int *hResultSup=nullptr;
	calcSupport_pure(hUniEdge.at(0).dUniEdge, \
		hUniEdge.at(0).noElem,hLevelEXT.at(0).vE.at(0).dArrExt, \
		hLevelEXT.at(0).vE.at(0).noElem,dBScanResult,dF,noElemF,hResultSup);

	extractUniEdgeSatisfyMinsup(hResultSup,noElem_d_UniqueExtension,minsup);
	CUCHECK(cudaFree(dBScanResult));
	CUCHECK(cudaFree(dB));
	return;
}

//use
__global__ void kernel_generate_segment_index(int* SegmentStarts,\
											  int noElem_segment,\
											  int noElem_of_graph_per_unique_ext)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElem_segment)
	{
		SegmentStarts[i]=i*noElem_of_graph_per_unique_ext;
	}
}

//use
void generate_segment_index(int noElem_of_graph_per_unique_ext,\
						   int number_unique_extension,\
						   int *&SegmentStarts)
{
	int noElem_segment = number_unique_extension;
	dim3 block(blocksize);
	dim3 grid((noElem_segment + block.x -1)/block.x);
	CUCHECK(cudaMalloc((void**)&SegmentStarts,noElem_segment*sizeof(int)));

	kernel_generate_segment_index<<<grid,block>>>(SegmentStarts, noElem_segment, noElem_of_graph_per_unique_ext);
	CUCHECK(cudaDeviceSynchronize());
	
	CUCHECK(cudaGetLastError());
	
	return;
}

//use
void SegReduce(int* dF,int number_unique_extension,int noElem_of_graph_per_unique_ext,int *&resultsDevice) 
{
	CudaContext& context = *ctx;
	int count = number_unique_extension*noElem_of_graph_per_unique_ext;
	int *SegmentStarts = nullptr;
	generate_segment_index(noElem_of_graph_per_unique_ext,number_unique_extension,SegmentStarts);
	const int NumSegments = number_unique_extension;
	CUCHECK(cudaMalloc((void**)&resultsDevice,number_unique_extension*sizeof(int)));
	SegReduceCsr(dF, SegmentStarts, count, number_unique_extension,\
				false, resultsDevice, (int)0, mgpu::plus<int>(), context);

	CUCHECK(cudaFree(SegmentStarts));
	return;
}

//use
__global__ void kernelGetGraph(int *dV,int noElemdV,int *d_kq,int *dVScanResult)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElemdV)
	{
		if(dV[i]!=0){
			d_kq[dVScanResult[i]]=i;
		}
	}
}


//use
__global__ void kernelGetLastElementEXT(EXT *inputArray,int noEleInputArray,int *value,unsigned int maxOfVer)
{
	//Lấy global vertex id chia cho tổng số đỉnh của đồ thị (maxOfVer). 
	//Ở đây các đồ thị luôn có số lượng đỉnh bằng nhau (maxOfVer)
	*value = inputArray[noEleInputArray-1].vgi/maxOfVer; 
}

//chưa sửa đối tham chiếu

//use
void getLastElementEXT(EXT *inputArray,int numberElementOfInputArray,int &outputValue,unsigned int maxOfVer)
{
	int *temp=nullptr;
	CUCHECK(cudaMalloc((int**)&temp,sizeof(int)));

	/* Lấy graphId chứa embedding cuối cùng */
	kernelGetLastElementEXT<<<1,1>>>(inputArray,numberElementOfInputArray,temp,maxOfVer);
	CUCHECK(cudaDeviceSynchronize());
	CUCHECK(cudaGetLastError());
	CUCHECK(cudaMemcpy(&outputValue,temp,sizeof(int),cudaMemcpyDeviceToHost));

	if(temp!=nullptr) CUCHECK(cudaFree(temp));
	return;
}

//use
__global__ void kernelGetGraphIdContainEmbeddingv2(int vi,int vj,int li,int lij,int lj, \
												   EXT *d_ValidExtension,int noElem_d_ValidExtension, \
												   int *dV,unsigned int maxOfVer)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i<noElem_d_ValidExtension)
	{
		if(d_ValidExtension[i].li == li && d_ValidExtension[i].lij == lij && d_ValidExtension[i].lj == lj && \
			d_ValidExtension[i].vi == vi && d_ValidExtension[i].vj == vj)
		{
			int graphid = (d_ValidExtension[i].vgi/maxOfVer);
			dV[graphid]=1;
		}
	}
}

//use
void PMS::get_graphid(UniEdge &edge,int *&hArrGraphId,int &noElemhArrGraphId,EXT *dArrEXT,int noElemdArrEXT)
{
	dim3 block(blocksize);
	dim3 grid((noElemdArrEXT+block.x-1)/block.x);

	int *d_graphid=nullptr;
	int noElem_d_graphid=0;
	//How many graphs contains embeddings of DFS_CODE?
	getLastElementEXT(dArrEXT,noElemdArrEXT,noElem_d_graphid,maxOfVer);
	++noElem_d_graphid;

	CUCHECK(cudaMalloc((void**)&d_graphid,noElem_d_graphid*sizeof(int)));
	CUCHECK(cudaMemset(d_graphid,0,noElem_d_graphid*sizeof(int)));

	kernelGetGraphIdContainEmbeddingv2<<<grid,block>>>(edge.vi,edge.vj,edge.li,edge.lij,edge.lj,dArrEXT, \
		noElemdArrEXT,d_graphid,maxOfVer);
	CUCHECK(cudaDeviceSynchronize());
	CUCHECK(cudaGetLastError());

	int *dVScanResult=nullptr; //2. need cudaFree
	CUCHECK(cudaMalloc((void**)&dVScanResult,noElem_d_graphid*sizeof(int)));
	CUCHECK(cudaMemset(dVScanResult,0,noElem_d_graphid*sizeof(int)));

	myScanV(d_graphid,noElem_d_graphid,dVScanResult);

	int noElem_kq=0;
	getSizeBaseOnScanResultv2(d_graphid,dVScanResult,noElem_d_graphid,noElem_kq);

	int *d_kq;
	CUCHECK(cudaMalloc((void**)&d_kq,sizeof(int)*noElem_kq));

	dim3 blocka(blocksize);
	dim3 grida((noElem_d_graphid + blocka.x -1)/blocka.x);

	kernelGetGraph<<<grida,blocka>>>(d_graphid,noElem_d_graphid,d_kq,dVScanResult);
	CUCHECK(cudaDeviceSynchronize());
	CUCHECK(cudaGetLastError());

	hArrGraphId=(int*)malloc(sizeof(int)*noElem_kq);
	if(hArrGraphId==nullptr) {FCHECK(-1);}
	noElemhArrGraphId=noElem_kq;

	CUCHECK(cudaMemcpy(hArrGraphId,d_kq,sizeof(int)*noElem_kq,cudaMemcpyDeviceToHost));
	//free memory
	CUCHECK(cudaFree(d_kq));
	CUCHECK(cudaFree(d_graphid));
	CUCHECK(cudaFree(dVScanResult));
	return;
}

//use
void PMS::MiningDeeper(EXTk &ext,UniEdgeStatisfyMinSup &UES)
{
	try
	{
		if (UES.noElem <= 0) return;
		UES.hArrUniEdge = (UniEdge*)malloc(sizeof(UniEdge)*UES.noElem);

		for(int idx_ues = 0; idx_ues < UES.noElem ; ++idx_ues)
		{
			if(UES.hArrUniEdge==nullptr) {return;}
			CUCHECK(cudaMemcpy(UES.hArrUniEdge,UES.dArrUniEdge,sizeof(UniEdge)*UES.noElem,cudaMemcpyDeviceToHost));
			DFS_CODE.add(UES.hArrUniEdge[idx_ues].vi,UES.hArrUniEdge[idx_ues].vj, \
				UES.hArrUniEdge[idx_ues].li,UES.hArrUniEdge[idx_ues].lij,UES.hArrUniEdge[idx_ues].lj);
			//Check minDFSCode
			/*if(!is_min())
			{
				is_min();
				write_notMinDFS_CODE(DFS_CODE);
			}*/
			if(is_min())
			{
				write_minDFS_CODE(DFS_CODE);
				//Đánh tất cả các embedding của unique edge trong UES.hArrUniEdge
				//Kết quả đánh dấu của lưu vào dValid
				int *dValid = nullptr;
				//Tại sao mark_edge không truyền vào posRow? Điều này có ảnh hưởng gì không?
				ext.mark_edge(UES.hArrUniEdge[idx_ues].vi,UES.hArrUniEdge[idx_ues].vj, \
					UES.hArrUniEdge[idx_ues].li,UES.hArrUniEdge[idx_ues].lij,UES.hArrUniEdge[idx_ues].lj,dValid);
				//Hiển thị các phần tử trong ext.
				//std::printf("\n********ext.show()************\n");
				//ext.show();

				//Tạo mảng index của các phần tử hợp lệ trong mảng mới.
				int *dIdx = nullptr;
				get_idx(dValid,ext.noElem,dIdx);
				//displayDeviceArr(dIdx,ext.noElem);

				//Tìm số lượng phần tử hợp lệ.
				//Lưu ý, tuỳ vào phần tử cuối của dValid có =1 hay không để tính noElem_valid cho đúng.
				int noElem_valid = 0;
				get_noElem_valid(dValid,dIdx,ext.noElem,noElem_valid);

				//get_graph_id
				int *hArrGraphId=nullptr;
				int noElemhArrGraphId=0;
				get_graphid(UES.hArrUniEdge[idx_ues],hArrGraphId,noElemhArrGraphId,ext.dArrExt,ext.noElem);
				//displayHostArray(hArrGraphId,noElemhArrGraphId);
				report(hArrGraphId,noElemhArrGraphId,UES.hArrSupport[idx_ues]);
				free(hArrGraphId);

				//build embedding for min DFS_CODE
				buildEmbedding(UES.hArrUniEdge[idx_ues],ext,dValid,dIdx);
				//Display Embedding columns.
				//for (int i = 0; i < hEm.size(); i++)
				//{
				//	cout<<endl;
				//	hEm.at(i).show();
				//	if(hEm.at(i).hBackwardEmbedding.size()>0)
				//	{
				//		for (int j = 0; j < hEm.at(i).hBackwardEmbedding.size(); j++)
				//		{
				//			std::printf("\nBackward:(%d;%d)\n:",i,j);
				//			hEm.at(i).hBackwardEmbedding.at(j).show();
				//		}
				//	}
				//}

				vector<EXTk> vecValidEXTk;
				//Find valid extension and return vector EXTk<i>
				findValidExtension(vecValidEXTk);
				//Trích các Unique Forward Extension ở tất cả các EXTk
				for (int idxEXTk = 0; idxEXTk < vecValidEXTk.size(); idxEXTk++)
				{
					if(vecValidEXTk.at(idxEXTk).noElem>0)
					{
						//vecValidEXTk.at(idxEXTk).show();
						vecValidEXTk.at(idxEXTk).extractUniForwardExtension(Lv,Le,DFS_CODE.maxId);
					}
				}
				//Trích các Unique Backward Extension ở EXTk cuối.
				//Các đỉnh trên RMP phải > 2 thì mới có khả năng có mở rộng backward từ đỉnh cuối của RMP.
				if (DFS_CODE.noElemOnRMP >2)
				{
					int lastIdxEXTk = vecValidEXTk.size()-1;
					//Nếu là phần tử cuối thì xét mở rộng backward
					if(vecValidEXTk.at(lastIdxEXTk).noElem > 0)
					{
						//vecValidEXTk.at(lastIdxEXTk).show();
						//Trích các unique edge backward.
						int* dRMP = nullptr;
						int* dRMPLabel = nullptr;
						int noElemMappingVj = 0;
						int vi = 0;
						int li = 0; //Có vẻ như tạo right most path trên device bị fail.
						buildRMPLabel(dRMP,dRMPLabel,noElemMappingVj,vi,li);
						vecValidEXTk.at(lastIdxEXTk).extractUniBackwardExtension( \
							Lv,Le, \
							DFS_CODE.noElemOnRMP, \
							dRMP,dRMPLabel, \
							noElemMappingVj,vi,li);
					}
				}

				//Compute support
				//Duyệt qua các unique trong mảng arrUniEdgea trong vecValidEXTk để tính support cho từng cạnh
				int lastidxEXTk = vecValidEXTk.size()-1;
				for (int idxvecValidEXTk = lastidxEXTk; idxvecValidEXTk >=0; idxvecValidEXTk--)
				{
					if (vecValidEXTk.at(idxvecValidEXTk).noElem>0)
					{
						vecValidEXTk.at(idxvecValidEXTk).findSupport(maxOfVer);
						//Trích các mở rộng forward thoả minsup
						vecValidEXTk.at(idxvecValidEXTk).extractStatisfyMinsup(minsup, \
							vecValidEXTk.at(idxvecValidEXTk).uniFE,vecValidEXTk.at(idxvecValidEXTk).uniFES);
						//Trích các mở rộng backward thoả minsup
						vecValidEXTk.at(idxvecValidEXTk).extractStatisfyMinsup(minsup, \
							vecValidEXTk.at(idxvecValidEXTk).uniBE,vecValidEXTk.at(idxvecValidEXTk).uniBES);
					}
				}

				//Duyet qua các EXTk và gọi MiningDeeper
				for (int idxvecValidEXTk = lastidxEXTk; idxvecValidEXTk >=0; idxvecValidEXTk--)
				{
					if (vecValidEXTk.at(idxvecValidEXTk).noElem>0)
					{
						//MiningDeeper cho backward truoc cho forward sau
						if(vecValidEXTk.at(idxvecValidEXTk).uniBES.noElem>0)
						{
							MiningDeeper(vecValidEXTk.at(idxvecValidEXTk),vecValidEXTk.at(idxvecValidEXTk).uniBES);
							//Đã xử lý xong uniBES có thể giải phóng
							vecValidEXTk.at(idxvecValidEXTk).uniBES.ReleaseMemory();
						}
						if(vecValidEXTk.at(idxvecValidEXTk).uniFES.noElem>0)
						{
							MiningDeeper(vecValidEXTk.at(idxvecValidEXTk),vecValidEXTk.at(idxvecValidEXTk).uniFES);
							//Đã xử lý xong uniFES có thể giải phóng
							vecValidEXTk.at(idxvecValidEXTk).uniFES.ReleaseMemory();
						}
						//Đã xử lý xong một phần tử EXTk
						vecValidEXTk.at(idxvecValidEXTk).ReleaseMemory();
					}
				}



				//Đã xử lý xong tất cả các phần tử trong EXTk
				vecValidEXTk.clear();
				CUCHECK(cudaFree(dValid));
				CUCHECK(cudaFree(dIdx));
				removeEmbedding();
			}
			//Khi khai thác xong thì gỡ bỏ cạnh vừa thêm ra khỏi DFS_CODE
			DFS_CODE.remove(UES.hArrUniEdge[idx_ues].vi,UES.hArrUniEdge[idx_ues].vj);
		}
		return;
	}
	catch(std::exception &exc)
	{
		cout<<exc.what();
		FCHECK(-1);
	}
}

//use
__global__ void kernelExtractBWEmbeddingRow(Embedding* dArrBWEmbedding,int *dV, \
											int *dVScanResult,int noElemdV,Embedding *dArrEmbedding)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElemdV)
	{
		if(dV[i]==1)
		{
			dArrBWEmbedding[dVScanResult[i]].idx = dArrEmbedding[i].idx;
			dArrBWEmbedding[dVScanResult[i]].vid = dArrEmbedding[i].vid;
		}
	}
}

//use
__global__ void	kernelExtractRowFromEXT(EXT *dArrExt,int noElemdArrExt,int *dV,int vj)
{
	//Đánh dấu những dòng nào (embeddings nào) trong Embedding column có mở rộng backward.
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElemdArrExt)
	{
		if(dArrExt[i].vj==vj)
		{
			dV[dArrExt[i].posRow]=1;
			//PMS_PRINT("\n Thread %d: dV[%d]:%d",i,dArrExt[i].posRow,dV[dArrExt[i].posRow]);
		}
	}
}

//use
__global__ void kernelSetValueForFirstTwoEmbeddingColumn(const EXT *d_ValidExtension,int noElem_d_ValidExtension, \
														 Embedding *dQ1,Embedding *dQ2,int *d_scanResult, \
														 int li,int lij,int lj)
{
	int i = blockDim.x *blockIdx.x +threadIdx.x;
	if(i<noElem_d_ValidExtension)
	{
		if(d_ValidExtension[i].li==li && d_ValidExtension[i].lij == lij && d_ValidExtension[i].lj==lj)
		{
			dQ1[d_scanResult[i]].idx=-1;
			dQ1[d_scanResult[i]].vid=d_ValidExtension[i].vgi;

			dQ2[d_scanResult[i]].idx=d_scanResult[i];
			dQ2[d_scanResult[i]].vid=d_ValidExtension[i].vgj;
		}
	}
}

//use
__global__ void kernelSetValueForEmbeddingColumn(EXT *dArrExt,int noElemInArrExt,Embedding *dArrQ,int *dM, \
												 int *dMScanResult)
{
	int i = blockDim.x *blockIdx.x + threadIdx.x;
	if(i<noElemInArrExt)
	{
		if(dM[i]==1)
		{
			int posRow = dArrExt[i].posRow;
			int vgj =dArrExt[i].vgj;
			dArrQ[dMScanResult[i]].idx=posRow;
			dArrQ[dMScanResult[i]].vid=vgj;
		}
	}
}

//use
void PMS::removeEmbedding()
{
	if(DFS_CODE.size() == 1)
	{
		removeFirstEmbedding();
	}
	else
	{
		//Kiểm tra xem hEm cuối có backward column hay không
		//Nếu có thì giải phóng embedding col backward.
		if(hEm.back().hBackwardEmbedding.size()>0)
		{
			CUCHECK(cudaFree(hEm.back().hBackwardEmbedding.back().dArrEmbedding));
			hEm.back().hBackwardEmbedding.pop_back();
		}
		else
		{
			//Ngược lại thì giải phóng hEm
			CUCHECK(cudaFree(hEm.back().dArrEmbedding));
			hEm.pop_back();
		}
	}
}

//use
void PMS::removeFirstEmbedding()
{
	CUCHECK(cudaFree(hEm.at(1).dArrEmbedding));
	CUCHECK(cudaFree(hEm.at(0).dArrEmbedding));
	hEm.pop_back();
	hEm.pop_back();
}

//use
void PMS::buildEmbedding(UniEdge &ue,EXTk &ext,int *&dValid,int *&dIdx)
{
	if(DFS_CODE.size() == 1)
	{
		buildFirstEmbedding(ue,ext,dValid,dIdx);
	}
	else if (ue.vi<ue.vj)
	{
		//Mở rộng embedding column forward.
		buildNewEmbeddingCol(ue,ext,dValid,dIdx);
	}
	else
	{
		//Mở rộng embedding column backward.
		buildBackwardEmbedding(ue,ext,dValid,dIdx);
	}
}

//use
void PMS::buildFirstEmbedding(UniEdge &ue,EXTk &ext,int*&dValid,int*&dIdx)
{
	//Mỗi phần tử của Vector sẽ quản lý 1 dArrEmbedding trên device. 
	//Khi cần thiết có thể tập hợp chúng lại thành 1 mảng trên device.
	hEm.resize(2);
	hEm.at(0).noElem;

	int noElemOfdArEmbedding=0;

	getSizeBaseOnScanResultv2(dValid,dIdx,ext.noElem,noElemOfdArEmbedding);

	hEm.at(0).noElem=hEm.at(1).noElem=noElemOfdArEmbedding;
	//Cấp phát bộ nhớ cho các embedding Columns.
	CUCHECK(cudaMalloc((void**)&hEm.at(0).dArrEmbedding,noElemOfdArEmbedding*sizeof(Embedding)));
	CUCHECK(cudaMalloc((void**)&hEm.at(1).dArrEmbedding,noElemOfdArEmbedding*sizeof(Embedding)));

	dim3 block(blocksize);
	dim3 grid((ext.noElem+block.x-1)/block.x);
	kernelSetValueForFirstTwoEmbeddingColumn<<<grid,block>>>(ext.dArrExt,ext.noElem,hEm.at(0).dArrEmbedding, \
		hEm.at(1).dArrEmbedding, dIdx,ue.li,ue.lij,ue.lj);
	CUCHECK(cudaDeviceSynchronize());
	CUCHECK(cudaGetLastError());

	//Gán giá trị prevCol cho các embedding columns. 
	hEm.at(0).prevCol=-1;
	hEm.at(1).prevCol=0;

	//show embedding column
	//hEm.at(0).show();
	//cout<<endl;
	//hEm.at(1).show();
	return;
}

//use
void PMS::buildNewEmbeddingCol(UniEdge &ue,EXTk &ext,int*&dValid,int*&dIdx)
{
	int currentSize = hEm.size();
	int newSize = currentSize + 1;
	int lastIdx = currentSize;
	hEm.resize(newSize);

	int noElemOfdArEmbedding=0;
	get_noElem_valid(dValid,dIdx,ext.noElem,noElemOfdArEmbedding);

	hEm.at(lastIdx).noElem = noElemOfdArEmbedding;
	//Cấp phát bộ nhớ cho các embedding Columns.
	CUCHECK(cudaMalloc((void**)&hEm.at(lastIdx).dArrEmbedding,noElemOfdArEmbedding*sizeof(Embedding)));

	dim3 block(blocksize);
	dim3 grid((ext.noElem+block.x-1)/block.x);
	kernelSetValueForEmbeddingColumn<<<grid,block>>>(ext.dArrExt,ext.noElem,hEm.at(lastIdx).dArrEmbedding,dValid,dIdx);
	CUCHECK(cudaDeviceSynchronize());
	CUCHECK(cudaGetLastError());

	//Gán giá trị prevCol cho các embedding columns. 
	hEm.at(lastIdx).prevCol=ue.vi;

	//show embedding column
	//hEm.at(lastIdx).show();
	return;
}

//use
void PMS::buildBackwardEmbedding(UniEdge& ue,EXTk& ext,int*& dValid,int*& dIdx)
{
	try
	{
		//Trích các row trong Embedding column cuối 
		//cout<<endl;
		//ue.print();
		//cout<<endl;
		//ext.show();
		//cout<<endl;
		//displayDeviceArr(dValid,ext.noElem);
		//cout<<endl;
		//displayDeviceArr(dIdx,ext.noElem);
		//cout<<endl;
		//for (int i = 0; i < hEm.size(); i++)
		//{
		//	hEm.at(i).show();
		//}

		//1. Khởi tạo một mảng <int> có số lượng phần tử bằng với số lượng Embedding gọi là dV 
		//	và đánh dấu các posRow chứa backward extension
		int *dV=nullptr;
		int noElemdV = 0;
		//Lấy số lượng Embedding dựa vào Embedding Column cuối.
		//noElemEmbedding tuỳ thuộc vào Embedding Colmn đó đã có backward embedding column nào hay chưa?
		if(hEm.back().hBackwardEmbedding.size()>0)
		{
			noElemdV = hEm.back().hBackwardEmbedding.back().noElem;
		}
		else
		{
			noElemdV = hEm.back().noElem;
		}
		//Cấp phát bộ nhớ cho dV trên device
		size_t nBytedV=noElemdV*sizeof(int);
		CUCHECK(cudaMalloc((void**)&dV,nBytedV));
		CUCHECK(cudaMemset(dV,0,nBytedV));

		//Trích các dòng từ EXT
		dim3 block(blocksize);
		dim3 grid((ext.noElem+block.x-1)/block.x);
		kernelExtractRowFromEXT<<<grid,block>>>(ext.dArrExt,ext.noElem,dV,ue.vj);
		CUCHECK(cudaDeviceSynchronize());
		CUCHECK(cudaGetLastError());

		//PMS_PRINT("\n********dV**********\n");
		//FCHECK(displayDeviceArr(dV,noElemdV));

		//1.1 scan dV để biết kích thước của backward column embedding
		int *dVScanResult = nullptr;
		CUCHECK(cudaMalloc((void**)&dVScanResult,noElemdV*sizeof(int)));

		get_idx(dV,noElemdV,dVScanResult);

		int noElemBW=0;
		get_noElem_valid(dV,dVScanResult,noElemdV,noElemBW);

		//2.Dựa vào dV để trích các embedding chứa backward extension sang một embedding column mới.
		int currentSize = hEm.back().hBackwardEmbedding.size();
		int newSizeOfBackwardEmCol = currentSize + 1;
		hEm.back().hBackwardEmbedding.resize(newSizeOfBackwardEmCol);

		CUCHECK(cudaMalloc((void**)&hEm.back().hBackwardEmbedding.back().dArrEmbedding,noElemBW*sizeof(Embedding)));

		dim3 blocka(blocksize);
		dim3 grida((noElemdV + blocka.x -1)/blocka.x);
		if(hEm.back().hBackwardEmbedding.size()>=2)
		{
			kernelExtractBWEmbeddingRow<<<grida,blocka>>>(hEm.back().hBackwardEmbedding.back().dArrEmbedding, \
				dV,dVScanResult,noElemdV, \
				hEm.back().hBackwardEmbedding.at(currentSize-1).dArrEmbedding);
			CUCHECK(cudaDeviceSynchronize());
			CUCHECK(cudaGetLastError());
			//Cập nhật số lượng phần tử và preCol cho backward EmCol vừa mới thêm vào.
			hEm.back().hBackwardEmbedding.back().noElem = noElemBW;
			hEm.back().hBackwardEmbedding.back().prevCol = hEm.back().prevCol;
		}
		else
		{
			noElemdV = hEm.back().noElem;
			kernelExtractBWEmbeddingRow<<<grida,blocka>>>(hEm.back().hBackwardEmbedding.back().dArrEmbedding, \
				dV,dVScanResult,noElemdV, \
				hEm.back().dArrEmbedding);
			CUCHECK(cudaDeviceSynchronize());
			CUCHECK(cudaGetLastError());
			//Cập nhật số lượng phần tử và preCol cho backward EmCol vừa mới thêm vào.
			hEm.back().hBackwardEmbedding.back().noElem = noElemBW;
			hEm.back().hBackwardEmbedding.back().prevCol = hEm.back().prevCol;
		}
		//PMS_PRINT("\n ************* dArrBWEmbeddingCol ***********\n");
		//hEm.back().hBackwardEmbedding.back().show();
		//free memory
		CUCHECK(cudaFree(dVScanResult));
		return;
	}
	catch(...)
	{
		FCHECK(-1);
	}
}


//use
__global__ void kernel_mark_edge(int vi,int vj,int li,int lij,int lj,EXT *ext,int *dValid,int noElem)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i<noElem)
	{
		if(ext[i].li == li && ext[i].lij == lij && ext[i].lj == lj && ext[i].vi == vi && ext[i].vj == vj)
		{
			dValid[i]=1;
		}
	}
}

//use
void PMS::findValidExtension(vector<EXTk> &vecValidEXTk)
{
	try
	{
		//Create device RMP
		int* dRMP = nullptr; //đây chính là bwInfo.dVj
		int noElemRMP = 0;
		createRMP(dRMP,noElemRMP);

		//Get all Embedding Column of RMP.
		int noElemdEmCol = 0;
		Embedding** dEmCol = nullptr;
		getEmCol(dEmCol,noElemdEmCol);

		//Create mark EmCol RMP
		int* dEmRMP = nullptr;
		createMarkEmColRMP(dRMP,noElemRMP,dEmRMP);
		
		//existBackwardInfo bwInfo;
		int* dValidBackward = nullptr;
		buildExistBackwardInfo(dRMP,noElemRMP, \
			dValidBackward);

		//Tim bac lon nhat
		Embedding** dArrEmbeddingColRMP = nullptr;
		getEmColRMP(dArrEmbeddingColRMP,noElemRMP);

		// Lấy số lượng embedding
		int noElemEmbedding = 0;
		getnoElemEmbedding(noElemEmbedding);
		

		//Tìm bậc của các đỉnh vid của các embeding thuộc RMP
		int noElemVid = noElemRMP*noElemEmbedding;
		float *dArrDegreeOfVid = nullptr;
		int maxDegreeOfVer = 0;
		findMaxDegreeVid(dEmCol,dEmRMP,noElemdEmCol, noElemVid,\
			noElemRMP, noElemEmbedding, \
			dArrDegreeOfVid,maxDegreeOfVer);

		int noPossibleExt = noElemRMP * noElemEmbedding * maxDegreeOfVer;

		//5. Khai thác được các mở rộng
		//vecValid lưu kết quả tìm các mở rộng hợp lệ của kernel. Trong đó,
		//dArrValid <-- đánh dấu các mở rộng hợp lệ
		//dArrBackward <-- đánh dấu các mở rộng là backward
		//dArrEXT <-- thông tin chi tiết của mở rộng (vi,vj,li,lij,lj,vgi,vgj,posRow)
		vector<structValid> vecValid;
		vecValid.resize(noElemRMP);
		for (int i = 0; i < noElemRMP; i++)
		{
			vecValid[i].noElem = noElemEmbedding*maxDegreeOfVer;
			CUCHECK(cudaMalloc((void**)&vecValid.at(i).dArrValid,noElemEmbedding*sizeof(int)*maxDegreeOfVer));
			CUCHECK(cudaMalloc((void**)&vecValid.at(i).dArrEXT,noElemEmbedding*sizeof(EXT)*maxDegreeOfVer));
			CUCHECK(cudaMemset(vecValid.at(i).dArrValid,0,noElemEmbedding*sizeof(int)*maxDegreeOfVer));
		}

		int** dPointerArrValid = nullptr;
		EXT** dPointerArrEXT = nullptr;

		CUCHECK(cudaMalloc((void**)&dPointerArrValid,noElemRMP*sizeof(int**)));
		CUCHECK(cudaMalloc((void**)&dPointerArrEXT,noElemRMP*sizeof(EXT**)));

		for (int i = 0; i < noElemRMP; i++)
		{
			kernelCopyDevice<<<1,1>>> (dPointerArrValid,vecValid.at(i).dArrValid, i);
			kernelCopyDeviceEXT<<<1,1>>>(dPointerArrEXT, vecValid.at(i).dArrEXT, i);
		}
		CUCHECK(cudaDeviceSynchronize());
		CUCHECK(cudaGetLastError());
		dim3 block(blocksize);
		dim3 grid((noElemVid + block.x -1)/block.x);
		//displayDeviceArr(dEmRMP,noElemdEmCol);
		kernelFindValidExtension1<<<grid,block>>>( \
			dEmCol, \
			dEmRMP, \
			noElemdEmCol, \
			dRMP, \
			noElemRMP, \
			noElemEmbedding, \
			hdb.at(0).dO,hdb.at(0).dLO,hdb.at(0).dN,hdb.at(0).dLN, \
			dArrDegreeOfVid, \
			maxDegreeOfVer, \
			dPointerArrValid,dPointerArrEXT, \
			DFS_CODE.minLabel,DFS_CODE.maxId, \
			dValidBackward,dRMP);

		CUCHECK(cudaDeviceSynchronize());
		CUCHECK(cudaGetLastError());

		//Hiển thị thông tin của vecValid
		//for (int i = 0; i < noElemRMP; i++)
		//{
		//	vecValid.at(i).show();
		//}
		//6. Trích được các mở rộng FW/BW từ vecValid lưu vào vecValidEXTk
		//Khởi tạo số lượng phần tử của vecValidEXTk bằng số lượng phần tử của vecValid
		vecValidEXTk.resize(vecValid.size());
		//Duyệt qua các vecValid và trích các mở rộng hợp lệ sang vecValidEXTk tương ứng.
		for (int idxVecValid = 0; idxVecValid < vecValid.size(); idxVecValid++)
		{
			//std::printf("\nMark the valid backward/forward extentions:");
			//vecValid.at(idxVecValid).show();
			vecValid.at(idxVecValid).extractValid(vecValidEXTk.at(idxVecValid));
			vecValid.at(idxVecValid).ReleaseMemory();
			//std::printf("\nShow the valid backward/forward extentions:");
			//vecValidEXTk.at(idxVecValid).show();
		}
		CUCHECK(cudaDeviceSynchronize());
		CUCHECK(cudaGetLastError());
		//Chưa dùng dArrBackward của vecValid.
		vecValid.clear();
		//7. Release memory
		CUCHECK(cudaFree(dArrEmbeddingColRMP));
		CUCHECK(cudaFree(dArrDegreeOfVid));
		CUCHECK(cudaFree(dEmRMP));
		CUCHECK(cudaFree(dPointerArrValid));
		CUCHECK(cudaFree(dPointerArrEXT));
		CUCHECK(cudaFree(dValidBackward));
	} catch (std::exception &exc)
	{
		cout<<endl<<exc.what()<<endl;
		FCHECK(-1);
	}
}

//use
__global__ void kernelFillValidBackward(int* dValidBackward,int* dRMP,int noElem, int* dVjBackward,int noElemdVjBackward)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElem)
	{
		int vj = dRMP[i];
		for (int k = 0; k < noElemdVjBackward; k++)
		{
			if(vj == dVjBackward[k])
			{
				dValidBackward[i] = 1;
				break;
			}
		}
	}
}

//use
__device__ void deviceGetVid(Embedding** &dEmCol, int* &dEmRMP,int &noElemdEmCol, \
						int &noElemEmbedding,int &idxCol, int &idxRow, int &noElemOnRMP, \
						int &vid,int &idxOnRMP)
{
	int idxLastdEmCol = noElemdEmCol - 1;
	int idxEmColVidOfThread = idxLastdEmCol;
	int idxEmRowVidOfThread = idxRow;
	
	idxOnRMP = noElemOnRMP;
	int countCol=-1;
	{
		do
		{
			if(dEmRMP[idxEmColVidOfThread] == 1)
			{
				--idxOnRMP;
				++countCol;
			}
			if(countCol == idxCol)
			{
				vid = dEmCol[idxEmColVidOfThread][idxEmRowVidOfThread].vid;
				return;
			}
			idxEmRowVidOfThread = dEmCol[idxEmColVidOfThread][idxEmRowVidOfThread].idx;
			--idxEmColVidOfThread;
		}while(idxEmRowVidOfThread != -1);
	}
}

//use
__device__ void deviceFindVid(int &thread , Embedding** &dEmCol, int* &dEmRMP,int &noElemdEmCol, \
						int &noElemEmbedding,int &noElemOnRMP, \
						 int &idxCol, int &idxRow,int& vid,int &idxOnRMP)
{
	//Get idxCol and idxRow of thread base on noElemEmbedding
	idxCol = thread / noElemEmbedding;
	idxRow = thread % noElemEmbedding;

	//initialize vid equal to minus one.
	vid = -1;
	deviceGetVid(dEmCol,dEmRMP,noElemdEmCol,noElemEmbedding,idxCol,idxRow, noElemOnRMP, \
		vid,idxOnRMP);

	//PMS_PRINT("\nthread:%d; idxCol:%d; idxRow:%d; vid:%d idxOnRMP:%d", \
	//	thread,idxCol,idxRow,vid, idxOnRMP);

}

//use
__device__ void deviceIsVidOnEm(int &toVid,Embedding** &dEmCol,int* &dEmRMP,int &noElemdEmCol, int &idxRow,int &noElemRMP, \
				int &onEm, int &onRMP,int &idxOnRMPtovid)
{
	int idxLastdEmCol = noElemdEmCol - 1;
	int idxEmCol = idxLastdEmCol;
	int idxEmRow = idxRow;

	idxOnRMPtovid = noElemRMP; //Dùng để lấy giá trị cho Vj
	do
	{
		if(dEmRMP[idxEmCol] == 1)
		{
			--idxOnRMPtovid;
		}
		if(toVid == dEmCol[idxEmCol][idxEmRow].vid)
		{
			onEm = 1;
			if (dEmRMP[idxEmCol] == 1) { onRMP = 1;}
			return;
		}
		idxEmRow = dEmCol[idxEmCol][idxEmRow].idx;
		--idxEmCol;
	}while(idxEmRow != -1);
}

//use
__global__ void kernelFindValidExtension1(Embedding **dEmCol,int* dEmRMP,int noElemdEmCol,int* dArrRMP, int noElemRMP, \
										 int noElemEmbedding, \
										 int *dO,int *dLO,int *dN,int *dLN, float *dArrDegreeOfVid, \
										 int maxDegreeOfVer,int** dPointerArrValid, \
										 EXT** dPointerArrEXT, int minLabel,int maxId, int* dValidBackward,int* dVj)
{
	//Mỗi Thread sẽ tìm mở rộng cho một vid trên RMP của embedding tương ứng.
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int noElemVid = noElemEmbedding*noElemRMP;
	if(i<noElemVid)
	{
		//get idxCol, idxRow and vid in which current thread needed processing.
		int idxLastCol = noElemRMP-1;
		int idxRow,idxCol,vid;
		int idxOnRMPvid; //biết vid mà thread đang xử lý thuộc RMP nào. Giúp lấy được Vj từ existbackard
		deviceFindVid(i,dEmCol, dEmRMP, noElemdEmCol,\
						noElemEmbedding,noElemRMP, \
						idxCol, idxRow, vid,idxOnRMPvid);

		int degreeOfVid = __float2int_rn(dArrDegreeOfVid[i]); //Lấy bậc của vid
		for (int idxToVid = 0; idxToVid < degreeOfVid; idxToVid++)
		{
			//int idxRMP = noElemRMP;
			int indexToVidIndN=dO[vid]+idxToVid; //Lấy index trong mảng nhãn cạnh.
			int labelFromVid = dLO[vid]; //Lấy nhãn của đỉnh được mở rộng.
			int toVid=dN[indexToVidIndN]; //vid của đỉnh kề.
			int labelToVid = dLO[toVid]; //nhãn của đỉnh kề
			
			int onEm = -1;
			int onRMP = -1;
			int idxOnRMPtovid = -1;
			deviceIsVidOnEm(toVid,dEmCol, dEmRMP, noElemdEmCol,idxRow,noElemRMP, \
				onEm, onRMP, idxOnRMPtovid);


			int idxColEXT,idxRowEXT;
			idxColEXT = (noElemRMP-1) - idxCol;
			//idxColEXT = idxCol;
			idxRowEXT = idxRow*maxDegreeOfVer + idxToVid;
			
			EXT* dArrEXT = dPointerArrEXT[idxColEXT];
			int* dArrValid = dPointerArrValid[idxColEXT];
			int b = -1;
			if(idxCol == 0) //Nếu vid mở rộng thuộc embedding column cuối
			{
				if(onRMP == 1) //Nếu tovid thuộc RMP
				{
					if(idxOnRMPvid - idxOnRMPtovid>=2) //Nếu khoảng cách giữa đỉnh mở rộng và đỉnh kề cách nhau ít nhất 2 EmCols.
					{
						if(dValidBackward[idxOnRMPtovid]==-1) //Nếu mở rộng backward đến tovid đó chưa tồn tại
						{
							//Backward được xem là hợp lệ
							dArrValid[idxRowEXT] = 1;

							dArrEXT[idxRowEXT].vi = maxId;
							dArrEXT[idxRowEXT].vj = dVj[idxOnRMPtovid];
							dArrEXT[idxRowEXT].li = labelFromVid;
							dArrEXT[idxRowEXT].lij = dLN[indexToVidIndN];
							dArrEXT[idxRowEXT].lj = labelToVid;
							dArrEXT[idxRowEXT].vgi = vid; 
							dArrEXT[idxRowEXT].vgj = toVid;
							dArrEXT[idxRowEXT].posRow = idxRow;
							b = 1;
						}
					}
				}
			}
			
			if (onEm==-1 && labelToVid>=minLabel)
			{
				//save valid forward
				dArrValid[idxRowEXT] = 1;
				dArrEXT[idxRowEXT].vi = dVj[idxOnRMPvid];
				dArrEXT[idxRowEXT].vj = maxId + 1;
				dArrEXT[idxRowEXT].li = labelFromVid;
				dArrEXT[idxRowEXT].lij = dLN[indexToVidIndN];
				dArrEXT[idxRowEXT].lj = labelToVid;
				dArrEXT[idxRowEXT].vgi = vid; 
				dArrEXT[idxRowEXT].vgj = toVid;
				dArrEXT[idxRowEXT].posRow = idxRow;
				b=2;
				//return;
			}
			//if (i ==8) std::printf("\n\nThread:%d idxToVid:%d toVid:%d \n labelToVid:%d degreeOfVid:%d \n onEm:%d onRMP:%d idxOnRMPvid:%d \n idxColEXT:%d idxRowEXT:%d minLabel:%d dArrEXT[idxRowEXT].vi:%d dArrValid[idxRowEXT]:%d \n dVj[0]:%d dVj[1]:%d dVj[2]:%d b:%d dArrEXT[idxRowEXT].vj:%d idxRow:%d", \
			//	i,idxToVid,toVid,labelToVid,degreeOfVid,onEm,onRMP,idxOnRMPvid,idxColEXT,idxRowEXT,minLabel,dArrEXT[idxRowEXT].vi,dArrValid[idxRowEXT],dVj[0],dVj[idxColEXT],dVj[2],b,dArrEXT[idxRowEXT].vj,idxRow);
		}



		//Dùng để so sánh với idxCol, nếu bằng nhau thì getVid
		//int countOnedEmRMP =-1;
		////idxRMP dùng để kiểm tra mở rộng backward có hợp lệ không.
		////int idxRMP = noElemRMP;
		////PMS_PRINT("\n Thread: %d idxCol:%d idxRow:%d idxLastCol:%d ",i,idxCol,idxRow,idxLastCol);
		//for(int s = noElemdEmCol-1,int idxRowTemp=idxRow; s>=0; s-- )
		//{
		//	Embedding* dEmTempCol = dEmCol[s];
		//	if(dEmRMP[s]==1)
		//	{
		//		countOnedEmRMP++;
		//		//idxRMP--;
		//	}
		//	if(countOnedEmRMP == idxCol)
		//	{
		//		//get div at current column s
		//		vid = dEmTempCol[idxRowTemp].vid;
		//		//then break
		//		break;
		//	}
		//	//Cập nhật idx row cần truy xuất trong column trước.
		//	idxRowTemp = dEmTempCol[idxRowTemp].idx;
		//}

		////Duyệt qua các đỉnh kề với vid trong dN dựa vào bậc của vid.
		//int degreeOfVid = __float2int_rn(dArrDegreeOfVid[i]); //Lấy bậc của vid
		//for (int idxToVid = 0; idxToVid < degreeOfVid; idxToVid++)
		//{
		//	int idxRMP = noElemRMP;
		//	int indexToVidIndN=dO[vid]+idxToVid; //Lấy index trong mảng nhãn cạnh.
		//	int labelFromVid = dLO[vid]; //Lấy nhãn của đỉnh được mở rộng.
		//	int toVid=dN[indexToVidIndN]; //vid của đỉnh kề.
		//	int labelToVid = dLO[toVid]; //nhãn của đỉnh kề

		//	//Xét đỉnh kề có thoả các điều kiện của mở rộng forward hay không.
		//	//đk1: nếu nhãn đỉnh kề nhỏ hơn minLabel của DFS_CODE thì continue xét đỉnh kề tiếp theo.
		//	if(labelToVid<minLabel) continue;
		//	//đk2: nếu đỉnh kề đã thuộc RMP của embedding rồi thì xét xem nó có là backward hay không.
		//	//Nếu đỉnh kề tồn tại trong lstVidOnRMP thì xem như nó đã thuộc embedding, tiếp tục xét đỉnh kề khác
		//	bool isExist = false;
		//	bool onRMP = false;
		//	for(int s = noElemdEmCol-1,int idxRowTemp=idxRow; s>=0; s-- )
		//	{
		//		Embedding* dEmTempCol = dEmCol[s];
		//		if(dEmRMP[s]==1)
		//		{
		//			idxRMP--;
		//		}
		//		if(toVid == dEmTempCol[idxRowTemp].vid)
		//		{
		//			isExist = true;
		//			//idxRMP = s;
		//			if (dEmRMP[s] == 1) onRMP = true;
		//			PMS_PRINT("\n Thread: %d isExist:%d onRMP:%d\n",i,isExist,onRMP);
		//			break;
		//		}
		//		idxRowTemp = dEmTempCol[idxRowTemp].idx;
		//	}
		//	//PMS_PRINT("\n Thread: %d ,vid: %d toVid:%d isExist:%d \n",i,vid,toVid,isExist);
		//	//vid có tối đa là maxDegreeOfVer mở rộng hợp lệ được lưu trữ trong dPointerArrEXT tương ứng tại idxCol, idxRow.
		//	EXT* dArrEXT = dPointerArrEXT[idxCol];
		//	int* dArrValid = dPointerArrValid[idxCol];
		//	int idxRowEXT = idxRow*maxDegreeOfVer + idxToVid;
		//	if (isExist == true) 
		//	{
		//		//Nếu Thread đang xử lý cho vid thuộc Embedding Column cuối và 
		//		//nhiều hơn 2 đỉnh thuộc RMP thì mới xét mở rộng backard và
		//		//Đỉnh kề phải cách đỉnh cuối ít nhất 1 đỉnh.
		//		if (onRMP == true && (noElemRMP - idxRMP)>=3 && noElemRMP >2) 
		//		{
		//			PMS_PRINT("\nThread:%d YES backward noElemRMP:%d idxRMP:%d\n",i,noElemRMP,idxRMP);
		//			goto considerBackward;
		//		}
		//		continue;
		//	};
		//	//Lưu mở rộng hợp lệ forward
		//	//Lưu mở rộng forward hợp lệ
		//	dArrValid[idxRowEXT] = 1;
		//	//dArrEXT[idxRowEXT].vi = dArrRMP[idxCol]; //<<<<<<<< chú ý lại
		//	dArrEXT[idxRowEXT].vi = dVj[idxCol];
		//	dArrEXT[idxRowEXT].vj = maxId + 1;
		//	dArrEXT[idxRowEXT].li = labelFromVid;
		//	dArrEXT[idxRowEXT].lij = dLN[indexToVidIndN];
		//	dArrEXT[idxRowEXT].lj = labelToVid;
		//	dArrEXT[idxRowEXT].vgi = vid; 
		//	dArrEXT[idxRowEXT].vgj = toVid;
		//	dArrEXT[idxRowEXT].posRow = idxRow;
		//	/*PMS_PRINT("\n Thread: %d, vid:%d,idxCol:%d, dVj: %d, maxId: %d (%d,%d,%d,%d,%d,vgi:%d,vgj:%d,posRow:%d)  \
		//			  idxRowEXT:%d dArrValid[idxRowEXT]:%d", \
		//		i,vid,idxCol, dVj[idxCol],maxId,dArrEXT[idxRowEXT].vi,dArrEXT[idxRowEXT].vj, \
		//		dArrEXT[idxRowEXT].li,dArrEXT[idxRowEXT].lij,dArrEXT[idxRowEXT].lj, \
		//		dArrEXT[idxRowEXT].vgi,dArrEXT[idxRowEXT].vgj,dArrEXT[idxRowEXT].posRow, \
		//		idxRowEXT, dArrValid[idxRowEXT]);*/
		//	//lưu xong thì continue xét đỉnh kề khác
		//	continue;

		//	considerBackward:
		//	//Kiểm tra backward có hợp lệ hay không
		//	//Nếu backward đã tồn tại rồi thì continue xét đỉnh kề khác.
		//	if(1==dValidBackward[idxRMP]) continue;
		//	//Lưu lại mở rộng backward hợp lệ.
		//	dArrValid[idxRowEXT] = 1;
		//	dArrEXT[idxRowEXT].vi = maxId;
		//	dArrEXT[idxRowEXT].vj = dVj[idxRMP];
		//	dArrEXT[idxRowEXT].li = labelFromVid;
		//	dArrEXT[idxRowEXT].lij = dLN[indexToVidIndN];
		//	dArrEXT[idxRowEXT].lj = labelToVid;
		//	dArrEXT[idxRowEXT].vgi = vid; 
		//	dArrEXT[idxRowEXT].vgj = toVid;
		//	dArrEXT[idxRowEXT].posRow = idxRow;
		//	/*PMS_PRINT("\n Thread: %d, vid:%d,idxCol:%d, dVj: %d, maxId: %d (%d,%d,%d,%d,%d,vgi:%d,vgj:%d,posRow:%d)  \
		//			  idxRowEXT:%d dArrValid[idxRowEXT]:%d", \
		//		i,vid,idxCol, dVj[idxRMP],maxId,dArrEXT[idxRowEXT].vi,dArrEXT[idxRowEXT].vj, \
		//		dArrEXT[idxRowEXT].li,dArrEXT[idxRowEXT].lij,dArrEXT[idxRowEXT].lj, \
		//		dArrEXT[idxRowEXT].vgi,dArrEXT[idxRowEXT].vgj,dArrEXT[idxRowEXT].posRow, \
		//		idxRowEXT, dArrValid[idxRowEXT]);*/
		//}
	}
}


//use
__global__ void kernelCreatedEmRMP(int* dArrRMP,int* dEmRMP,int noElemRMP)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i<noElemRMP)
	{
		int idx = dArrRMP[i];
		dEmRMP[idx] = 1;
	}
}

//use
__global__ void	kernelGetPointerdArrEmbedding(Embedding *dArrEmbedding,Embedding **dArrPointerEmbedding,int idx)
{
	//Copy the address of dArrEmbedding into dArrPointerEmbedding
	dArrPointerEmbedding[idx]=dArrEmbedding;
	//PMS_PRINT("\n PointerdArrEmbedding:%p, PointerdArrPointerEmbedding:%p",dArrEmbedding,dArrPointerEmbedding[idx]);
}

//use
__global__ void kernelExtractValidExtensionTodExt(EXT *dArrExtension,int *dArrValid,int *dArrValidScanResult, \
												  int noElem_dArrV,EXT *dExt,int noElem_dExt)
{
	//kernel trích các mở rộng hợp lệ từ mảng dArrExtension sang mảng dExt
	int i =blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElem_dArrV)
	{
		if(dArrValid[i]==1)
		{
			dExt[dArrValidScanResult[i]].vi = dArrExtension[i].vi;
			dExt[dArrValidScanResult[i]].vj = dArrExtension[i].vj;
			dExt[dArrValidScanResult[i]].li = dArrExtension[i].li;
			dExt[dArrValidScanResult[i]].lij = dArrExtension[i].lij;
			dExt[dArrValidScanResult[i]].lj = dArrExtension[i].lj;
			dExt[dArrValidScanResult[i]].vgi = dArrExtension[i].vgi;
			dExt[dArrValidScanResult[i]].vgj = dArrExtension[i].vgj;
			dExt[dArrValidScanResult[i]].posRow = dArrExtension[i].posRow;
		}
	}
}

//use
__global__ void kernelGet_vivjlj(EXT* dArrExt,int* dvi,int* dvj,int* dli,int maxId)
{
	*dvi = dArrExt[0].vi;
	*dvj = maxId+1;
	*dli = dArrExt[0].li;
}

//use
__global__ void kernelExtractUniBE(int* dAllExtension,int noElemdAllExtension, \
									int* dRMP,int* dRMPLabel,int Lv,UniEdge* dUniEdge, \
									int* dAllExtensionIdx,int vi,int li)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i<noElemdAllExtension)
	{
		if(dAllExtension[i] == 1)
		{
			int lij = i / Lv;
			int idxVj = i % Lv;
			int vj = dRMP[idxVj];
			int idxUniEdge = dAllExtensionIdx[i];
			dUniEdge[idxUniEdge].lij = lij;
			dUniEdge[idxUniEdge].vj = vj;
			dUniEdge[idxUniEdge].lj = dRMPLabel[idxVj];
			dUniEdge[idxUniEdge].vi = vi;
			dUniEdge[idxUniEdge].li = li;
		}
	}
}

//use
__global__ void kernelMarkUniBE(int* dMappingVj,int* dAllExtension, int Lv,int noElem,EXT* dArrEXT)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i<noElem)
	{
		int vi = dArrEXT[i].vi;
		int vj = dArrEXT[i].vj;
		if (vi > vj)
		{
			int lij = dArrEXT[i].lij;
			int idxVj = dMappingVj[vj];
			int idxAllExtension = lij*Lv + idxVj;
			dAllExtension[idxAllExtension] = 1;
			//PMS_PRINT("\n Thread: %d, vi:%d vj:%d, lij:%d lv:%d  idxVj:%d idxAllExtension:%d)", \
			//	i,vi,vj,lij,Lv,idxVj,idxAllExtension);
		}
	}
}

//use
__global__ void kernelFilldMappingVj(int noElemBW,int* dMappingVj,int* dRMP)
{
	int i = blockDim.x*blockIdx.x +threadIdx.x;
	if(i<noElemBW)
	{
		int vj = dRMP[i];
		dMappingVj[vj] = i;
	}
}

//use
__global__ void kernelFillUniFE( int *dArrAllPossibleExtension, \
								int *dArrAllPossibleExtensionScanResult, \
								int noElem_dArrAllPossibleExtension, \
								UniEdge *dArrUniEdge, \
								int Lv,int *dvi, \
								int *dvj,int *dli)
{
	//Kernel fill unique forward extension
	int i = blockDim.x*blockIdx.x +threadIdx.x;
	if(i<noElem_dArrAllPossibleExtension)
	{
		if(dArrAllPossibleExtension[i]==1)
		{
			int li,lij,lj;
			li=*dli;
			lij = i/Lv;
			lj=i%Lv;
			dArrUniEdge[dArrAllPossibleExtensionScanResult[i]].vi=*dvi;
			dArrUniEdge[dArrAllPossibleExtensionScanResult[i]].vj=*dvj;
			dArrUniEdge[dArrAllPossibleExtensionScanResult[i]].li=li;
			dArrUniEdge[dArrAllPossibleExtensionScanResult[i]].lij=lij;
			dArrUniEdge[dArrAllPossibleExtensionScanResult[i]].lj=lj;
		}
	}
}



//use
__global__ void kernelmarkValidForwardEdge_LastExt(EXT* dArrExt, int noElemdArrExt,unsigned int Lv, \
												   int *dAllPossibleExtension)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i<noElemdArrExt)
	{
		//Chỉ xét các forward
		if(dArrExt[i].vi < dArrExt[i].vj)
		{
			int index=	dArrExt[i].lij*Lv + dArrExt[i].lj;
			dAllPossibleExtension[index]=1;
		}
	}
}

//use
__global__ void kernelFilldF(UniEdge *dArrUniEdge,int pos,EXT *dArrExt,int noElemdArrExt,int *dArrBoundaryScanResult,int *dF)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i<noElemdArrExt)
	{
		int vi = dArrUniEdge[pos].vi;
		int vj = dArrUniEdge[pos].vj;
		int li = dArrUniEdge[pos].li;
		int lij = dArrUniEdge[pos].lij;
		int lj = dArrUniEdge[pos].lj;
		int Li = dArrExt[i].li;
		int Lij = dArrExt[i].lij;
		int Lj = dArrExt[i].lj;
		int Vi = dArrExt[i].vi;
		int Vj = dArrExt[i].vj;
		if(li==Li && lij==Lij && lj==Lj && vi == Vi && vj == Vj)
		{
			dF[dArrBoundaryScanResult[i]]=1;
		}
		//PMS_PRINT("\nThread %d: UniEdge(li:%d lij:%d lj:%d) (Li:%d Lij:%d Lj:%d idxdF:%d dF:%d)",i,li,lij,lj,Li,Lij,Lj,dArrBoundaryScanResult[i],dF[dArrBoundaryScanResult[i]]);
	}
}

//use
__global__ void kernelfindBoundary(EXT *dArrExt,int noElemdArrExt,int *dArrBoundary,unsigned int maxOfVer)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i<noElemdArrExt-1)
	{
		unsigned int graphIdAfter=dArrExt[i+1].vgi/maxOfVer;
		unsigned int graphIdCurrent=dArrExt[i].vgi/maxOfVer;
		if(graphIdAfter!=graphIdCurrent)
		{
			dArrBoundary[i]=1;
		}
	}
}

//use
__global__ void find_maximum_kernel(float *array, float *max, int *mutex, unsigned int n)
{
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int stride = gridDim.x*blockDim.x;
	unsigned int offset = 0;

	__shared__ float cache[256];


	float temp = -1.0;
	while(index + offset < n){
		temp = fmaxf(temp, array[index + offset]);

		offset += stride;
	}

	cache[threadIdx.x] = temp;

	__syncthreads();


	// reduction
	unsigned int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i){
			cache[threadIdx.x] = fmaxf(cache[threadIdx.x], cache[threadIdx.x + i]);
		}

		__syncthreads();
		i /= 2;
	}

	if(threadIdx.x == 0){
		while(atomicCAS(mutex,0,1) != 0);  //lock
		*max = fmaxf(*max, cache[0]);
		atomicExch(mutex, 0);  //unlock
	}
}


//use
void PMS::findMaxDegreeVid(Embedding** &dEmCol,int* &dEmRMP,int &noElemdEmCol, int &noElemVid, \
			int &noElemRMP, int &noElemEmbedding, \
			float* &dArrDegreeOfVid,int &maxDegreeOfVer)
{
	try
	{
		if(noElemVid<=0) return;
		CUCHECK(cudaMalloc((void**)&dArrDegreeOfVid,noElemVid*sizeof(float)));

		dim3 block(blocksize);
		dim3 grid((noElemVid + block.x -1)/block.x);

		kernelCalDegreeOfVidOnEmbeddingColumnv2<<<grid,block>>>( \
			dEmCol,dEmRMP,noElemdEmCol, \
			hdb.at(0).dO,hdb.at(0).noElemdO,noElemRMP, noElemEmbedding, \
			hdb.at(0).noElemdN, maxOfVer,dArrDegreeOfVid);
		CUCHECK(cudaDeviceSynchronize());
		CUCHECK(cudaGetLastError());

		//displayDeviceArr(dArrDegreeOfVid,noElemVid);
		float* h_max = (float*)malloc(sizeof(float));
		if(h_max==nullptr) FCHECK(-1);

		float *d_max = nullptr;
		int *d_mutex = nullptr;
		CUCHECK(cudaMalloc((void**)&d_max,sizeof(float)));
		CUCHECK(cudaMemset(d_max,0,sizeof(float)));

		CUCHECK(cudaMalloc((void**)&d_mutex,sizeof(int)));
		CUCHECK(cudaMemset(d_mutex,0,sizeof(int)));

		dim3 gridSize = 256;
		dim3 blockSize = 256;
		find_maximum_kernel<<<gridSize, blockSize>>>(dArrDegreeOfVid, d_max, d_mutex, noElemVid);
		CUCHECK(cudaDeviceSynchronize());
		CUCHECK(cudaGetLastError());

		CUCHECK(cudaMemcpy(h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost));
		maxDegreeOfVer = (int)*h_max;

		free(h_max);
		CUCHECK(cudaFree(d_max));
		CUCHECK(cudaFree(d_mutex));
	}
	catch(const std::exception &exc)
	{
		std::cerr<<exc.what();
		FCHECK(-1);
	}
}


//use
__global__ void kernelCalDegreeOfVidOnEmbeddingColumnv2(Embedding** dEmCol,int* dEmRMP,int noElemdEmCol, \
									 int *d_O, int numberOfElementd_O,int noElemRMP, int noElemEmbedding, \
									 int numberOfElementd_N,unsigned int maxOfVer,float *dArrDegreeOfVid)
{
	//Mỗi Thread sẽ tìm mở rộng cho một vid trên RMP của embedding tương ứng.
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int noElemVid = noElemEmbedding*noElemRMP;
	if(i<noElemVid)
	{
		int idxLastCol = noElemRMP-1;
		int idxRow = i % noElemEmbedding;
		int idxCol = i / noElemEmbedding;
		//vid mà thread i cần xử lý
		int vid=-1;
		//Dùng để so sánh với idxCol, nếu bằng nhau thì getVid
		int countOnedEmRMP =-1;
		for(int s = noElemdEmCol-1,int idxRowTemp=idxRow; s>=0; s-- )
		{
			Embedding* dEmTempCol = dEmCol[s];
			if(dEmRMP[s]==1)
			{
				countOnedEmRMP++;
			}
			if(countOnedEmRMP == idxCol) 
			{
				//get div at current column s
				vid = dEmTempCol[idxRowTemp].vid;
				break;
			}
			//Cập nhật idx row cần truy xuất trong column trước.
			idxRowTemp = dEmTempCol[idxRowTemp].idx;
		}
		//PMS_PRINT("\nThread %d proccess vid: %d",i,vid);
		float degreeOfV =0;
		int nextVid=-1;
		int graphid=-1;
		int lastGraphId=(numberOfElementd_O-1)/maxOfVer;
		if (vid==numberOfElementd_O-1)
		{ 
			//nếu như đây là đỉnh cuối cùng trong d_O
			//thì bậc của đỉnh vid chính bằng tổng số cạnh trừ cho giá trị của d_O[vid].
			degreeOfV=numberOfElementd_N-d_O[vid]; 
		}
		else
		{
			nextVid = vid+1; //xét đỉnh phía sau có khác 1 hay không?
			graphid=vid/maxOfVer;
			if(d_O[nextVid]==-1 && graphid==lastGraphId)
			{
				degreeOfV=numberOfElementd_N-d_O[vid];
			}
			else if(d_O[nextVid]==-1 && graphid!=lastGraphId)
			{
				nextVid=(graphid+1)*maxOfVer;
				degreeOfV=d_O[nextVid]-d_O[vid];
			}
			else
			{
				degreeOfV=d_O[nextVid]-d_O[vid];
			}
		}
		dArrDegreeOfVid[i]=degreeOfV;
	}
}


//use
void EXTk::mark_edge(int vi,int vj,int li,int lij,int lj,int *&dValid)
{
	CUCHECK(cudaMalloc((void**)&dValid,sizeof(int)*noElem));
	CUCHECK(cudaMemset(dValid,0,sizeof(int)*noElem));

	dim3 block(blocksize);
	dim3 grid((noElem+block.x-1)/block.x);
	kernel_mark_edge<<<grid,block>>>(vi,vj,li,lij,lj,dArrExt,dValid,noElem);
	CUCHECK(cudaDeviceSynchronize());
	CUCHECK(cudaGetLastError());
	return;
}

//use
void structValid::extractValid(EXTk &outputEXT)
{
	//doing somethings here
	//1. Scan on dArrValid to get index
	int *dArrValidScanResult=nullptr;
	CUCHECK(cudaMalloc((void**)&dArrValidScanResult,sizeof(int)*noElem));
	CUCHECK(cudaMemset(dArrValidScanResult,0,sizeof(int)*noElem));

	myScanV(dArrValid,noElem,dArrValidScanResult);
	int noElem_dExt=0;
	get_noElem_valid(dArrValid,dArrValidScanResult,noElem,noElem_dExt);
	if (noElem_dExt == 0) 
	{
		CUCHECK(cudaFree(dArrValidScanResult));
		return;
	}
	outputEXT.noElem = noElem_dExt;

	CUCHECK(cudaMalloc((void**)&outputEXT.dArrExt,sizeof(EXT)*outputEXT.noElem));
	dim3 block(blocksize);
	dim3 grid((noElem+block.x -1)/block.x);
	kernelExtractValidExtensionTodExt<<<grid,block>>>(dArrEXT,dArrValid,dArrValidScanResult,noElem,outputEXT.dArrExt,noElem_dExt);
	CUCHECK(cudaFree(dArrValidScanResult));
}

//use
void EXTk::extractUniForwardExtension(unsigned int& Lv,unsigned int& Le,int& maxId)
{
	//Tính số lượng tất cả các cạnh có thể có dựa vào nhãn của chúng
	int noElem_dallPossibleExtension=Le*Lv;

	int *d_allPossibleExtension=nullptr;
	int *d_allPossibleExtensionScanResult=nullptr;

	CUCHECK(cudaMalloc((void**)&d_allPossibleExtension,noElem_dallPossibleExtension*sizeof(int)));
	CUCHECK(cudaMemset(d_allPossibleExtension,0,noElem_dallPossibleExtension*sizeof(int)));
	CUCHECK(cudaMalloc((void**)&d_allPossibleExtensionScanResult,noElem_dallPossibleExtension*sizeof(int)));

	dim3 block(blocksize);
	dim3 grid((noElem+block.x-1)/block.x);
	//Đánh dấu vị trí các mở rộng forward hợp lệ là 1 tại vị trí d_allPossibleExtension tương ứng
	kernelmarkValidForwardEdge_LastExt<<<grid,block>>>(dArrExt,noElem,Lv,d_allPossibleExtension);
	CUCHECK(cudaDeviceSynchronize());
	CUCHECK(cudaGetLastError());

	get_idx(d_allPossibleExtension,noElem_dallPossibleExtension,d_allPossibleExtensionScanResult);

	int noElem_UniEdge=0;
	get_noElem_valid(d_allPossibleExtension,d_allPossibleExtensionScanResult, \
		noElem_dallPossibleExtension,noElem_UniEdge);

	if(noElem_UniEdge!=0)
	{
		uniFE.noElem = noElem_UniEdge;
		CUCHECK(cudaMalloc((void**)&uniFE.dUniEdge,uniFE.noElem*sizeof(UniEdge)));
		int* dvi = nullptr;
		int* dvj = nullptr;
		int* dli = nullptr;
		CUCHECK(cudaMalloc((void**)&dvi,sizeof(int)*1));
		CUCHECK(cudaMalloc((void**)&dvj,sizeof(int)*1));
		CUCHECK(cudaMalloc((void**)&dli,sizeof(int)*1));
		
		kernelGet_vivjlj<<<1,1>>>(dArrExt,dvi,dvj,dli,maxId);
		CUCHECK(cudaDeviceSynchronize());
		CUCHECK(cudaGetLastError());

		//displayDeviceArr(dvi,1);
		//displayDeviceArr(dvj,1);
		//displayDeviceArr(dli,1);

		dim3 block1(blocksize);
		dim3 grid1((noElem_dallPossibleExtension + block1.x -1)/block1.x);

		kernelFillUniFE<<<grid1,block1>>>( d_allPossibleExtension, \
								d_allPossibleExtensionScanResult, \
								noElem_dallPossibleExtension, \
								uniFE.dUniEdge, \
								Lv, dvi, dvj, dli);

		CUCHECK(cudaDeviceSynchronize());
		CUCHECK(cudaGetLastError());

		//uniFE.show();
	}

	if (d_allPossibleExtension != nullptr) CUCHECK(cudaFree(d_allPossibleExtension));
	if (d_allPossibleExtensionScanResult != nullptr) CUCHECK(cudaFree(d_allPossibleExtensionScanResult));
	return;
}

//use
void EXTk::extractUniBackwardExtension(unsigned int& Lv,unsigned int& Le,int& noElemRMP, \
									   int*& dRMP,int*& dRMPLabel, int& noElemMappingVj,int& vi,int& li)
{
	try
	{
		int noElemdAllExtension = Le * (noElemRMP-2);
		int tempLv = noElemRMP-2;
		//Tính số lượng tất cả các cạnh có thể có dựa vào nhãn của chúng
		int noElemBW = noElemRMP -2;
		//cout<<endl<<"dRMP: "<<noElemBW<<endl;
		//displayDeviceArr(dRMP,noElemBW);
		//cout<<endl<<"dRMPLabel: "<<noElemBW<<endl;
		//displayDeviceArr(dRMPLabel,noElemBW);
		//Chứa kết quả đánh dấu các mở rộng backward có thể có.
		int *dAllExtension=nullptr;
		int *dAllExtensionIdx=nullptr;
		int *dMappingVj = nullptr;

		CUCHECK(cudaMalloc((void**)&dAllExtension,noElemdAllExtension*sizeof(int)));
		CUCHECK(cudaMemset(dAllExtension,0,noElemdAllExtension*sizeof(int)));
		CUCHECK(cudaMalloc((void**)&dAllExtensionIdx,noElemdAllExtension*sizeof(int)));
		CUCHECK(cudaMalloc((void**)&dMappingVj, noElemMappingVj * sizeof(int)));
		CUCHECK(cudaMemset(dMappingVj,-1, noElemMappingVj * sizeof(int)));
		//Xây dựng dMappingVj để ánh xạ vj trong EXT sang idxVj
		dim3 block(blocksize);
		dim3 grid((noElemBW + block.x - 1)/block.x);
		//Mỗi thread sẽ đọc 1 phần tử (lij,vj) từ EXT
		//Đọc mảng dRMP[vj]
		kernelFilldMappingVj<<<grid,block>>>(noElemBW,dMappingVj,dRMP);
		CUCHECK(cudaDeviceSynchronize());
		CUCHECK(cudaGetLastError());
		//cout<<endl<<"dMappingVj:"<<endl;
		//displayDeviceArr(dMappingVj,noElemMappingVj);
		//Bật 1 cho các unique backward extension trong dAllExtension
		dim3 block1(blocksize);
		dim3 grid1((noElem + block.x - 1)/block.x);

		kernelMarkUniBE<<<grid1,block1>>>(dMappingVj,dAllExtension,tempLv,noElem,dArrExt);
		CUCHECK(cudaDeviceSynchronize());
		CUCHECK(cudaGetLastError());

		//cout<<endl<<"dAllExtension:"<<endl;
		//displayDeviceArr(dAllExtension,noElemdAllExtension);

		//Scan on dAllExtension to get index, noElem and cudamalloc
		get_idx(dAllExtension,noElemdAllExtension,dAllExtensionIdx);
		get_noElem_valid(dAllExtension,dAllExtensionIdx,noElemdAllExtension,uniBE.noElem);
		CUCHECK(cudaMalloc((void**)&uniBE.dUniEdge,sizeof(UniEdge)*uniBE.noElem));
		//Check again
		//Ánh xạ ngược từ dAllExtension sang UniEdge Backward
		dim3 block2(blocksize);
		dim3 grid2((noElemdAllExtension + block2.x -1)/block2.x);
		kernelExtractUniBE<<<grid2,block2>>>(dAllExtension,noElemdAllExtension, \
			dRMP,dRMPLabel,tempLv,uniBE.dUniEdge,dAllExtensionIdx,vi,li);
		CUCHECK(cudaDeviceSynchronize());
		CUCHECK(cudaGetLastError());

		//uniBE.show();


		if (dAllExtension != nullptr) CUCHECK(cudaFree(dAllExtension));
		if (dAllExtensionIdx != nullptr) CUCHECK(cudaFree(dAllExtensionIdx));
		if (dMappingVj != nullptr) CUCHECK(cudaFree(dMappingVj));
		if (dRMPLabel != nullptr) CUCHECK(cudaFree(dRMPLabel));
		if (dRMP != nullptr) CUCHECK(cudaFree(dRMP));
		return;
	}
	catch(std::exception &exc)
	{
		cout<<endl<<exc.what()<<endl;
		FCHECK(-1);
	}
}

//use
void arrUniEdge::copyDTH()
{
	try
	{
		if(noElem < 0) return;
		hUniEdge = nullptr;
		hUniEdge = (UniEdge*)malloc(sizeof(UniEdge)*noElem);
		if(hUniEdge == nullptr) {FCHECK(-1);}
		CUCHECK(cudaMemcpy(hUniEdge,dUniEdge,sizeof(UniEdge)*noElem,cudaMemcpyDeviceToHost));
	}
	catch (...)
	{
		FCHECK(-1)
	}
}

//use
void EXTk::findBoundary(unsigned int& maxOfVer,int*& dArrBoundaryScanResult)
{
	int *dArrBoundary=nullptr; 
	CUCHECK(cudaMalloc((void**)&dArrBoundary,sizeof(int)*noElem));
	CUCHECK(cudaMemset(dArrBoundary,0,sizeof(int)*noElem));

	dArrBoundaryScanResult=nullptr;
	CUCHECK(cudaMalloc((void**)&dArrBoundaryScanResult,sizeof(int)*noElem));
	CUCHECK(cudaMemset(dArrBoundaryScanResult,0,sizeof(int)*noElem));

	dim3 block(blocksize);
	dim3 grid((noElem+block.x-1)/block.x);
	

	kernelfindBoundary<<<grid,block>>>(dArrExt,noElem,dArrBoundary,maxOfVer);
	CUCHECK(cudaDeviceSynchronize());
	CUCHECK(cudaGetLastError());

	get_idx(dArrBoundary,noElem,dArrBoundaryScanResult);

	CUCHECK(cudaFree(dArrBoundary));
}

//use
void EXTk::findSupport(unsigned int& maxOfVer)
{
	try
	{
		int* dArrBoundaryIndex = nullptr;
		findBoundary(maxOfVer,dArrBoundaryIndex);

		int *dF=nullptr;
		int noElemdF = 0;
		CUCHECK(cudaMemcpy(&noElemdF,&dArrBoundaryIndex[noElem-1],sizeof(int),cudaMemcpyDeviceToHost));
		++noElemdF;

		CUCHECK(cudaMalloc((void**)&dF,sizeof(int)*noElemdF));
		CUCHECK(cudaMemset(dF,0,sizeof(int)*noElemdF));

		//Tính Support cho các mở rộng backward
		if(uniBE.noElem>0)
		{
			uniBE.copyDTH();
			uniBE.hSupport = nullptr;
			uniBE.hSupport = (int*)malloc(sizeof(int)*uniBE.noElem);
			if(uniBE.hSupport==nullptr) FCHECK(-1);
			memset(uniBE.hSupport,0,sizeof(int)*uniBE.noElem);
			for (int i = 0; i < uniBE.noElem; i++)
			{
				findSupportFW(dArrBoundaryIndex,uniBE.dUniEdge,i,dF,noElemdF,uniBE.hSupport[i]);
				//Mỗi lần lặp thì reset lại zerocho dF 
				CUCHECK(cudaMemset(dF,0,sizeof(int)*noElemdF));
			}
			//uniBE.showSupport();
		}

		//Tính Support cho các mở rộng forward
		if(uniFE.noElem>0)
		{
			uniFE.copyDTH();
			uniFE.hSupport = nullptr;
			uniFE.hSupport = (int*)malloc(sizeof(int)*uniFE.noElem);
			if(uniFE.hSupport==nullptr) FCHECK(-1);
			memset(uniFE.hSupport,0,sizeof(int)*uniFE.noElem);
			//Duyệt qua các phần tử duy nhất, tính support của chúng và lưu lại trong hSupport tại index tương ứng.
			for (int i = 0; i < uniFE.noElem; i++)
			{
				findSupportFW(dArrBoundaryIndex,uniFE.dUniEdge,i,dF,noElemdF,uniFE.hSupport[i]);
				//Mỗi lần lặp thì reset lại zerocho dF 
				CUCHECK(cudaMemset(dF,0,sizeof(int)*noElemdF));
			}
			//uniFE.showSupport();
		}

		CUCHECK(cudaFree(dArrBoundaryIndex));
		CUCHECK(cudaFree(dF));
	}
	catch(...)
	{
		FCHECK(-1);
	}
}

//use
void EXTk::findSupportFW(int*& dArrBoundaryIndex,UniEdge*& dArrUniEdge,int& idxUniEdge, int*& dF,int& noElemdF,int& support)
{
	try
	{
		int hSupport=0;
		dim3 block(blocksize);
		dim3 grid((noElem + block.x - 1)/block.x);
		kernelFilldF<<<grid,block>>>(dArrUniEdge,idxUniEdge,dArrExt,noElem,dArrBoundaryIndex,dF);
		CUCHECK(cudaDeviceSynchronize());
		CUCHECK(cudaGetLastError());

		//PMS_PRINT("\n**********dF****************\n");
		//FCHECK(displayDeviceArr(dF,noElemdF));

		myReduce(dF,noElemdF,hSupport);


		//PMS_PRINT("\n******support********");
		//PMS_PRINT("\n Support:%d",hSupport);

		support=hSupport;
	}
	catch(...)
	{
		FCHECK(-1);
	}
}

//use
void EXTk::extractStatisfyMinsup(unsigned int& minsup,arrUniEdge& uniEdge,UniEdgeStatisfyMinSup& uniES)
{
	try
	{
		//Trích các mở rộng forward thoả minsup
		if (uniEdge.noElem<=0) return;
		//1. Cấp phát mảng trên device có kích thước bằng noElemUniEdge
		int *dResultSup=nullptr; //cần được giải phóng ở cuối hàm
		CUCHECK(cudaMalloc((void**)&dResultSup,uniEdge.noElem*sizeof(int)));
		//Chép độ hỗ trợ từ host qua device để lọc song song 
		CUCHECK(cudaMemcpy(dResultSup,uniEdge.hSupport,uniEdge.noElem*sizeof(int),cudaMemcpyHostToDevice));

		//2. Đánh dấu 1 trên dV cho những phần tử thoả minsup
		int *dV=nullptr; //cần được giải phóng ở cuối hàm
		CUCHECK(cudaMalloc((void**)&dV,uniEdge.noElem*sizeof(int)));
		CUCHECK(cudaMemset(dV,0,sizeof(int)*uniEdge.noElem));

		dim3 block(blocksize);
		dim3 grid((uniEdge.noElem + block.x - 1)/block.x);
		kernelMarkUniEdgeSatisfyMinsup<<<grid,block>>>(dResultSup,uniEdge.noElem,dV,minsup);
		CUCHECK(cudaDeviceSynchronize());
		CUCHECK(cudaGetLastError());

		int *dVScanResult=nullptr; //cần được giải phóng ở cuối hàm
		CUCHECK(cudaMalloc((void**)&dVScanResult,uniEdge.noElem*sizeof(int)));

		get_idx(dV,uniEdge.noElem,dVScanResult);

		int noElemUniEdgeSatisfyMinSup = 0;
		get_noElem_valid(dV,dVScanResult,uniEdge.noElem,noElemUniEdgeSatisfyMinSup);
		//Nếu không có phần tử nào thoả minsup thì không khai thác nữa
		if(noElemUniEdgeSatisfyMinSup==0)
		{ 
			CUCHECK(cudaFree(dResultSup));
			CUCHECK(cudaFree(dV));
			CUCHECK(cudaFree(dVScanResult));
			return;
		}

		uniES.noElem = noElemUniEdgeSatisfyMinSup;
		CUCHECK(cudaMalloc((void**)&uniES.dArrUniEdge,uniES.noElem*sizeof(UniEdge)));

		uniES.hArrSupport = (int*)malloc(sizeof(int)*uniES.noElem);
		if (uniES.hArrSupport ==nullptr){FCHECK(-1);}

		int *dSup=nullptr; //cần được giải phóng ở cuối hàm
		CUCHECK(cudaMalloc((void**)&dSup,uniES.noElem*sizeof(int)));

		dim3 blocka(blocksize);
		dim3 grida((uniEdge.noElem + blocka.x -1)/blocka.x);
		kernelExtractUniEdgeSatifyMinsupV3<<<grida,blocka>>> ( \
			uniEdge.dUniEdge,dV,dVScanResult,uniEdge.noElem,uniES.dArrUniEdge,dSup,dResultSup);
		CUCHECK(cudaDeviceSynchronize());
		CUCHECK(cudaGetLastError());

		//PMS_PRINT("\n ********hUniEdgeSatisfyMinsup.dSup****************\n");
		//displayDeviceArr(dSup,uniES.noElem);

		CUCHECK(cudaMemcpy(uniES.hArrSupport,dSup,sizeof(int)*uniES.noElem,cudaMemcpyDeviceToHost));

		//uniES.show();
		CUCHECK(cudaFree(dResultSup));
		CUCHECK(cudaFree(dV));
		CUCHECK(cudaFree(dVScanResult));
		CUCHECK(cudaFree(dSup));
		uniEdge.ReleaseMemory();
	}
	catch(std::exception &exc)
	{
		cout<<endl<<exc.what()<<endl;
		FCHECK(-1);
	}
}

//use
void PMS::buildRMPLabel(int* &dRMP, int* &dRMPLabel,int &noElemMappingVj,int &_vi,int &_li)
{
	try
	{
		vector<int> RMP;
		vector<int> vertexLabel;
		int vi,vj;
		int preVj;
		int idxContinue=0;
		for(int i = DFS_CODE.size() - 1; i>=0;i--)
		{
			vi = DFS_CODE.at(i).from;
			vj = DFS_CODE.at(i).to;
			bool isForward = (vi<vj);
			if(isForward==true)
			{
				_vi = vj;
				_li = DFS_CODE.at(i).tolabel;
				idxContinue = i;
				preVj = vj;
				break;
			}
		}
		int i;
		for (i = idxContinue; i >= 0; i--)
		{
			vi = DFS_CODE.at(i).from;
			vj = DFS_CODE.at(i).to;
			if(vi<vj && preVj==vj)
			{
				RMP.push_back(vj);
				vertexLabel.push_back(DFS_CODE.at(i).tolabel);
				preVj = vi;
			}
		}
		RMP.push_back(vi);
		vertexLabel.push_back(DFS_CODE.at(++i).fromlabel);

		std::reverse(RMP.begin(),RMP.end());
		std::reverse(vertexLabel.begin(),vertexLabel.end());

		DFS_CODE.noElemOnRMP = RMP.size();
		int *hRMPLabel = nullptr;
		int *hRMP = nullptr;

		hRMP = (int*)malloc(sizeof(int)*(DFS_CODE.noElemOnRMP-2));
		if(hRMP == nullptr) {FCHECK(-1);}
		hRMPLabel = (int*)malloc(sizeof(int)*(DFS_CODE.noElemOnRMP-2));
		if(hRMPLabel == nullptr) {FCHECK(-1);}

		dRMPLabel = nullptr;
		dRMP = nullptr;
		CUCHECK(cudaMalloc((void**)&dRMPLabel,sizeof(int)*(DFS_CODE.noElemOnRMP-2)));
		CUCHECK(cudaMalloc((void**)&dRMP,sizeof(int)*(DFS_CODE.noElemOnRMP-2)));
		int idx;
		for (idx = 0; idx < RMP.size()-2; ++idx)
		{
			//std::printf("V[%d] Li[%d]; ",RMP[idx],vertexLabel[idx]);
			hRMP[idx] = RMP[idx];
			hRMPLabel[idx] = vertexLabel[idx];
		}

		CUCHECK(cudaMemcpy(dRMP,hRMP,(DFS_CODE.noElemOnRMP-2)*sizeof(int),cudaMemcpyHostToDevice));
		CUCHECK(cudaMemcpy(dRMPLabel,hRMPLabel,(DFS_CODE.noElemOnRMP-2)*sizeof(int),cudaMemcpyHostToDevice));
		int lastIdx = idx - 1;
		noElemMappingVj = hRMP[lastIdx] + 1;

		free(hRMP);
		free(hRMPLabel);
		RMP.clear();
		vertexLabel.clear();
	}
	catch(std::exception &exc)
	{
		cout<<endl<<exc.what()<<endl;
		FCHECK(-1);
	}
}

//use
void PMS::getVjBackwardDFSCODE(int* &dRMP,int &noElemOnRMP, \
							   int* &dVj,int &noElemdVj)
{
	try
	{
		//Chứa các Vj của các backward extension có thể có từ đỉnh cuối của DFS_CODE.
		vector<int> vertexVj;
		//Nếu cạnh cuối là backward thì xét cạnh kế cuối cho đến khi đó là forward.
		int vi,vj;
		vector<int> vjBackward;
		for(int i = DFS_CODE.size() - 1; i>=0;i--)
		{ 
			vi = DFS_CODE.at(i).from;
			vj = DFS_CODE.at(i).to;

			bool isForward = (vi<vj);
			//Nếu là forward thì không lấy Vj nữa
			if(isForward==true)
			{
				break;
			}
			else
			{
				vjBackward.push_back(vj);
			}
		}
		if (vjBackward.size() <=0) return;
		int* hValidBackward = (int*)malloc(vjBackward.size()*sizeof(int));
		if(hValidBackward == nullptr) {FCHECK(-1);}
		//Copy dữ liệu từ vjBackward sang mảng hValidBackward
		for (int i = 0; i < vjBackward.size(); i++)
		{
			hValidBackward[i] = vjBackward.at(i);
		}

		//update output
		noElemdVj = vjBackward.size();
		CUCHECK(cudaMalloc((void**)&dVj,sizeof(int)*noElemdVj));
		//Copy dữ liệu từ hValidBackward sang dVj
		CUCHECK(cudaMemcpy(dVj,hValidBackward,sizeof(int)*noElemdVj,cudaMemcpyHostToDevice));
		//giải phóng bộ nhớ
		free(hValidBackward);
	}
	catch(std::exception &exc)
	{
		cout<<endl<<exc.what()<<endl;
		FCHECK(-1);
	}
}

//use
void PMS::buildExistBackwardInfo(int* &dRMP,int &noElemOnRMP, \
								 int* &dValidBackward)
{
	try
	{
		if(noElemOnRMP<=0) return;
		CUCHECK(cudaMalloc((void**)&dValidBackward,noElemOnRMP*sizeof(int)));
		CUCHECK(cudaMemset(dValidBackward,-1,noElemOnRMP*sizeof(int))); //-1 được xem là dVj đó chưa tồn tại Backard link nào.

		if (hEm.back().hBackwardEmbedding.size()<=0) return; //Chưa có backward nào thì return;

		int* dVjBackward = nullptr;
		int noElemdVjBackward = 0;
		getVjBackwardDFSCODE(dRMP,noElemOnRMP, \
								dVjBackward,noElemdVjBackward);
		if (noElemdVjBackward == 0)
		{
			return;
		}
		
		//Thread i mang giá trị dVj[i] quét trong mảng dValidBackward
		//Thread i set 1 tại existBackwardInfo.dValidBackward[i] nếu dV[i] tồn tại trong dValidBackward
		dim3 block(blocksize);
		dim3 grid((DFS_CODE.noElemOnRMP+block.x-1)/block.x);
		kernelFillValidBackward<<<grid,block>>>( \
			dValidBackward,dRMP,noElemOnRMP, \
			dVjBackward,noElemdVjBackward);
		CUCHECK(cudaDeviceSynchronize());
		CUCHECK(cudaGetLastError());

		//cout<<endl<<"****dValidBackward*****"<<endl;
		//displayDeviceArr(dValidBackward,noElemOnRMP);
	}
	catch(const std::exception &exc)
	{
		std::cerr << exc.what();
		FCHECK(-1);
	}
}


//use
void PMS::getEmCol(Embedding** &dEmCol,int &noElemdEmCol)
{
	//Duyệt qua embedding columns và trích các Embedding Col từ hEm.
	//Nếu tại hEm.at(i) có backward col thì ưu tiên trích backward
	try
	{
		if (hEm.size()<=0) return;
		noElemdEmCol = hEm.size();
		size_t noBytedEmCol = hEm.size()*sizeof(Embedding**);
		CUCHECK(cudaMalloc((void**)&dEmCol,noBytedEmCol));
		for (int i = 0; i < hEm.size(); i++)
		{
			//Mỗi phần tử của mảng dArrPointerEmbedding chứa địa chỉ của dArrEmbedding
			if (hEm.at(i).hBackwardEmbedding.size()>0)
			{
				kernelGetPointerdArrEmbedding<<<1,1>>>(hEm.at(i).hBackwardEmbedding.back().dArrEmbedding, \
					dEmCol, \
					i);
			}
			else
			{
				kernelGetPointerdArrEmbedding<<<1,1>>>(hEm.at(i).dArrEmbedding, dEmCol, i);
			}
		}
		CUCHECK(cudaDeviceSynchronize());
		CUCHECK(cudaGetLastError());
		//-------------------------- đã có dEmCol chứa danh sách các pointer dArrEmbedding trên device.
	}
	catch(const std::exception &exc)
	{
		std::cerr << exc.what();
		FCHECK(-1);
	}
}

//use
void PMS::getEmColRMP(Embedding** &dEmCol,const int &noElemRMP)
{
	try
	{
		if (noElemRMP<=0) return;
		size_t noBytedEmCol = noElemRMP*sizeof(Embedding**);
		CUCHECK(cudaMalloc((void**)&dEmCol,noBytedEmCol));

		for (int prevCol = hEm.size()-1; prevCol != -1; prevCol = hEm.at(prevCol).prevCol)
		{
			kernelGetPointerdArrEmbedding<<<1,1>>>(hEm.at(prevCol).dArrEmbedding, dEmCol, prevCol);
		}
		CUCHECK(cudaDeviceSynchronize());
		CUCHECK(cudaGetLastError());
	}
	catch(const std::exception &exc)
	{
		std::cerr << exc.what();
		FCHECK(-1);
	}
}

//use
void PMS::createMarkEmColRMP(int* &dRMP,int &noElemdRMP,int* &dEmColRMP)
{
	try
	{
		if (hEm.size()<=0) return;

		size_t noBytesdEmRMP = hEm.size()*sizeof(int);
		CUCHECK(cudaMalloc((void**)&dEmColRMP,noBytesdEmRMP));
		CUCHECK(cudaMemset(dEmColRMP,0,noBytesdEmRMP));
		
		//kernel update dEmColRMP
		dim3 block(blocksize);
		dim3 grid((noElemdRMP + block.x -1)/block.x);
		kernelCreatedEmRMP<<<grid,block>>>(dRMP, dEmColRMP,noElemdRMP);

		CUCHECK(cudaDeviceSynchronize());
		CUCHECK(cudaGetLastError());
	}
	catch(const std::exception &exc)
	{
		std::cerr << exc.what();
		FCHECK(-1);
	}
}

//use
void PMS::createRMP(int* &dRMP,int &noElem)
{
	//example: v0-- v1--v5--v7 is on RMP; noElem =4 and dRMP=[0,1,5,7]
	try
	{
		if(hEm.size()<=0) return;
		//Build list RMP
		list<int> lstRMP;
		lstRMP.push_front(hEm.size() -1);
		int nextPrevCol = hEm.back().prevCol;
		while(nextPrevCol != -1 )
		{
			lstRMP.push_front(nextPrevCol);
			nextPrevCol = hEm.at(nextPrevCol).prevCol;
		}
		//update noElem
		noElem = lstRMP.size();
		DFS_CODE.noElemOnRMP = noElem;
		//copy lstRMP to host temp memory
		int *hArrRMP = nullptr;
		hArrRMP = (int*)malloc(sizeof(int)*noElem);
		if(hArrRMP == nullptr) FCHECK(-1);

		for (int idxLstRMP = 0; idxLstRMP < noElem; idxLstRMP++)
		{
			hArrRMP[idxLstRMP] = lstRMP.front();
			lstRMP.pop_front();
		}
		//clear lstRMP
		lstRMP.clear();

		//update dRMP base on host temp memory.
		CUCHECK(cudaMalloc((void**)&dRMP,sizeof(int)*noElem));
		CUCHECK(cudaMemcpy(dRMP,hArrRMP,sizeof(int)*noElem,cudaMemcpyHostToDevice));

		free(hArrRMP);
	}
	catch(const std::exception &exc)
	{
		std::cerr << exc.what();
		FCHECK(-1);
	}
}

//use
void PMS::getnoElemEmbedding(int &noElemEmbedding)
{
	try
	{
		noElemEmbedding = 0;
		if (hEm.size() <= 0) return;
		if(hEm.back().hBackwardEmbedding.size()>0)
		{
			noElemEmbedding = hEm.back().hBackwardEmbedding.back().noElem;
		}
		else
		{
			noElemEmbedding = hEm.back().noElem;
		}
	}
	catch(const std::exception &exc)
	{
		std::cerr << exc.what();
		FCHECK(-1);
	}
}