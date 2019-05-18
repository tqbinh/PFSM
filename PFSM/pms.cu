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
void write_minDFS_CODE(DFSCode dfscode){
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

PMS::PMS()
{
	Level=0;
	idxLevel=0;
	Lv=0;
	Le=0;
	maxOfVer=0;
	numberOfGraph=0;
	minLabel = -1;
	maxId = -1;	
	//std::cout<<" PMS initialized " << std::endl;
	//char* outfile;
	//outfile = "/result.graph";
	//fos.open(outfile);	
}
PMS::~PMS(){
	//std::cout<<" PMS terminated " << std::endl;
	//fos.close();

	if(hLevelUniEdge.size()>0){
		hLevelUniEdge.clear();
	}

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
			cudaFree(hExtension.at(i).dExtension);
		}
		hExtension.clear();
	}

	if(hUniEdge.size()!=0){
		for (int i = 0; i < hUniEdge.size(); i++)
		{
			cudaFree(hUniEdge.at(i).dUniEdge);
		}
		hUniEdge.clear();
	}

	if(hUniEdgeSatisfyMinsup.size()!=0){
		for (int i = 0; i < hUniEdgeSatisfyMinsup.size(); i++)
		{			
			cudaFree(hUniEdgeSatisfyMinsup.at(i).dUniEdge);
			free(hUniEdgeSatisfyMinsup.at(i).hArrSup);					
		}
		hUniEdgeSatisfyMinsup.clear();
	}
	//cudaDeviceReset();
}
bool fexists(const char *filename)
{
  ifstream ifile(filename);
  return ifile;
}
int PMS::prepareDataBase()
{
	int status =0;
	cudaError_t cudaStatus;
	//unsigned int minsup = 10;
	unsigned int minsup = 2;
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
	fname= "G0G1G2_custom1"; //Kết quả giống với gSpan
	//fname="Chemical_340Origin";

	////PMS pms;	
	ofstream fout("result.txt");
	char* minDFSCODE = "minDFSCODE.txt";
	if(fexists(minDFSCODE)==true){
		remove(minDFSCODE);
		cout<<"Xoa file minDFSCODE dang ton tai"<<endl;
	}
	//Chuyển dữ liệu từ fname sang TRANS
	//pms.run(fname,fout,minsup,maxpat,minnodes,enc,where,directed);
	run(fname,fout,minsup,maxpat,minnodes,enc,where,directed);
	//maxOfVer=pms.findMaxVertices();
	maxOfVer=findMaxVertices();
	numberOfGraph=noGraphs();
	int sizeOfarrayO=maxOfVer*numberOfGraph;
	int* arrayO = new int[sizeOfarrayO]; //Tạo mảng arrayO có kích thước D*m
	if(arrayO==NULL){
		PMS_PRINT("\n!!!Memory Problem ArrayO");
		exit(1);
	}else{
		memset(arrayO, -1, sizeOfarrayO*sizeof(int)); // gán giá trị cho các phần tử mảng bằng -1
	}
	unsigned int noDeg; //Tổng bậc của tất cả các đỉnh trong csdl đồ thị TRANS
	//noDeg = pms.sumOfDeg();
	noDeg = sumOfDeg();
	//cout<<noDeg;
	unsigned int sizeOfArrayN=noDeg;
	int* arrayN = new int[sizeOfArrayN]; //Mảng arrayN lưu trữ id của các đỉnh kề với đỉnh tương ứng trong mảng arrayO.
	if(arrayN==NULL){ //kiểm tra cấp phát bộ nhớ cho mảng có thành công hay không
		PMS_PRINT("\n!!!Memory Problem ArrayN");
		exit(1);
	}else
	{
		memset(arrayN, -1, noDeg*sizeof(int));
	}

	//
	int* arrayLO = new int[sizeOfarrayO]; //Mảng arrayLO lưu trữ label cho tất cả các đỉnh trong TRANS.
	if(arrayLO==NULL){ //kiểm tra cấp phát bộ nhớ cho mảng có thành công hay không
		PMS_PRINT("\n!!!Memory Problem ArrayLO");
		exit(1);
	}else
	{
		memset(arrayLO, -1, sizeOfarrayO*sizeof(int));
	}



	int* arrayLN = new int[noDeg]; //Mảng arrayLN lưu trữ label của tất cả các cạnh trong TRANS
	if(arrayLN==NULL){ //kiểm tra cấp phát bộ nhớ cho mảng có thành công hay không
		PMS_PRINT("\n!!!Memory Problem ArrayLN");
		exit(1);
	}else
	{
		memset(arrayLN, -1, noDeg*sizeof(int));
	}


	//pms.importDataToArray(arrayO,arrayLO,arrayN,arrayLN,sizeOfarrayO,noDeg,maxOfVer);
	importDataToArray(arrayO,arrayLO,arrayN,arrayLN,sizeOfarrayO,noDeg,maxOfVer);
	//cout<<"ArrayO:";
	//displayArray(arrayO,sizeOfarrayO);
	//cout<<"\nArrayLO:";
	//displayArray(arrayLO,sizeOfarrayO);
	//cout<<"\nArrayN:";
	//displayArray(arrayN,noDeg);
	//cout<<"\nArrayLN:";
	//displayArray(arrayLN,noDeg);

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

	CHECK(cudaStatus=cudaMalloc((void**)&graphdb.dO,nBytesO));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	//Cấp phát bộ nhớ trên GPU được quản lý bởi pointer dLO
	CHECK(cudaStatus=cudaMalloc((void**)&graphdb.dLO,nBytesO));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus=cudaMalloc((void**)&graphdb.dN,nBytesN));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus=cudaMalloc((void**)&graphdb.dLN,nBytesN));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	//Chép dữ liệu từ mảng arrayO trên CPU sang GPU được quản lý bởi pointer dO
	CHECK(cudaStatus=cudaMemcpy(graphdb.dO,arrayO,nBytesO,cudaMemcpyHostToDevice));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	//	delete(arrayO);

	CHECK(cudaStatus=cudaMemcpy(graphdb.dLO,arrayLO,nBytesO,cudaMemcpyHostToDevice));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	//delete(arrayLO);
	CHECK(cudaStatus=cudaMemcpy(graphdb.dN,arrayN,nBytesN,cudaMemcpyHostToDevice));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	//delete(arrayN);
	CHECK(cudaStatus=cudaMemcpy(graphdb.dLN,arrayLN,nBytesN,cudaMemcpyHostToDevice));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	//delete(arrayLN);
	//pms.db.push_back(graphdb); //Đưa cơ sở dữ liệu vào vector db
	//pms.countNumberOfDifferentValue(pms.db.at(0).dLO,pms.db.at(0).noElemdO,pms.Lv);
	//pms.countNumberOfDifferentValue(pms.db.at(0).dLN,pms.db.at(0).noElemdN,pms.Le);
	hdb.push_back(graphdb); //Đưa cơ sở dữ liệu vào vector db
	countNumberOfDifferentValue(hdb.at(0).dLO,hdb.at(0).noElemdO,Lv);
	countNumberOfDifferentValue(hdb.at(0).dLN,hdb.at(0).noElemdN,Le);
	//pms.printdb();
Error:
	return status;
}

bool PMS::checkArray(int *hostRef, int *gpuRef, const int N) {
	bool result=true;
	double epsilon = 1.0E-8;
	int match = 1;
	for (int i = 0; i < N; i++) {
		if ((float)(abs(hostRef[i] - gpuRef[i])) > epsilon) {
			match = 0;
			result=false;
			PMS_PRINT("Arrays do not match!\n");
			PMS_PRINT("host %5.2f gpu %5.2f at current %d\n",
				hostRef[i], gpuRef[i], i);
			break;
		}
	}
	if (match){
		PMS_PRINT("Arrays match.\n\n");		
	}

	return result;
}


void PMS::displayArray(int *p, const unsigned int pSize=0)
{
	for(int i=0;i<pSize;i++){
		PMS_PRINT("P[%d]:%d ",i,p[i]);
	}
	PMS_PRINT("\n");
	return;
}
//inuse
void PMS::displayHostArray(int *&p, const unsigned int pSize=0)
{
	std::printf("\n");
	for(int i=0;i<pSize;i++){
		PMS_PRINT("[%d]:%d ",i,p[i]);
	}
	PMS_PRINT("\n");
	return;
}

__global__ void kernelPrintdArr(int *dArr,unsigned int noElem){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i<noElem){
		PMS_PRINT("A[%d]:%d ",i,dArr[i]);
	}
}


void PMS::printdb(){
	PMS_PRINT("\n *********** Lv, Le **********\n");
	PMS_PRINT("\n Lv:%d",Lv);
	PMS_PRINT("\n Le:%d",Le);
	for (int i = 0; i < hdb.size(); i++)
	{
		unsigned int noElem =  hdb.at(i).noElemdO;	


		dim3 block(blocksize);
		dim3 grid((noElem + block.x -1)/block.x);
		PMS_PRINT("\n ********* dO *********\n");
		kernelPrintdArr<<<grid,block>>>(hdb.at(i).dO,noElem);
		cudaDeviceSynchronize();
		PMS_PRINT("\n");

		PMS_PRINT("\n ********* dLO *********\n");
		kernelPrintdArr<<<grid,block>>>(hdb.at(i).dLO,noElem);
		cudaDeviceSynchronize();
		PMS_PRINT("\n");

		unsigned int noElemdN = hdb.at(i).noElemdN;
		dim3 blocka(blocksize);
		dim3 grida((noElemdN + blocka.x -1)/blocka.x);

		PMS_PRINT("\n ********* dN *********\n");
		kernelPrintdArr<<<grida,blocka>>>(hdb.at(i).dN,noElemdN);
		cudaDeviceSynchronize();
		PMS_PRINT("\n");

		PMS_PRINT("\n ********* dLN *********\n");
		kernelPrintdArr<<<grida,blocka>>>(hdb.at(i).dLN,noElemdN);
		cudaDeviceSynchronize();
		PMS_PRINT("\n");
	}
}


__global__ void kernelMyScanV(int *dArrInput,int noElem,int *dResult){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElem){
		if(i==0){
			dResult[i]=0;
		}else
		{
			int temp=0;
			for (int j = 0; j <= (i-1); j++)
			{
				temp=temp + dArrInput[j];
			}
			dResult[i]=temp;
		}
	}
}

__global__ void kernelCopyDeviceArray(int *dArrInput,int *dResult,int noElem)
{
	int i = blockIdx.x  * blockDim.x + threadIdx.x;
	if(i<noElem)
	{
		dResult[i]=dArrInput[i];
	}
}

__global__ void kernelCopyDevice(int** dPointerArr,int* dArr,int at)
{
	dPointerArr[at] = dArr;
	//PMS_PRINT("\n dPointerArr:%d, dArr:%d",dPointerArr[at],dArr);
}

__global__ void kernelCopyDeviceEXT(EXT** dPointerArr,EXT* dArr,int at)
{
	dPointerArr[at] = dArr;
}



void  myReduce(int *dArrInput,int noElem,int &hResult)
{
	CudaContext& cdactx = *ctx;
	hResult = Reduce(dArrInput,noElem,cdactx);
	/*cout<<"reduce output: "<<hResult<<endl;*/
}

//in use
cudaError_t  myScanV(int *dArrInput,int noElem,int *&dResult)
{
	cudaError_t cudaStatus;
	dim3 block(blocksize);
	dim3 grid((noElem + block.x -1)/block.x);

	CHECK(cudaStatus=cudaMalloc((void**)&dResult,noElem * sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		goto Error;
	}
	//Copy dArrInput to dResult
	kernelCopyDeviceArray<<<grid,block>>>(dArrInput,dResult,noElem);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess)
	{
		goto Error;
	}

	//kernelMyScanV<<<grid,block>>>(dArrInput,noElem,dResult);
	CudaContext& cdactx = *ctx;
	mgpu::ScanExc(dResult, noElem,cdactx);

	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess)
	{
		goto Error;
	}

	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess)
	{
		goto Error;
	}
Error:
	return cudaStatus;
}
//inuse
void get_idx(int*& dArrInput,int& noElem,int*& dResult)
{
	dim3 block(blocksize);
	dim3 grid((noElem + block.x -1)/block.x);

	CUCHECK(cudaMalloc((void**)&dResult,noElem * sizeof(int)));

	//Copy dArrInput to dResult
	/*kernelCopyDeviceArray<<<grid,block>>>(dArrInput,dResult,noElem);
	CUCHECK(cudaDeviceSynchronize())
	CUCHECK(cudaGetLastError());*/
	CUCHECK(cudaMemcpy(dResult,dArrInput,noElem*sizeof(int),cudaMemcpyDeviceToDevice))

	//displayDeviceArr(dResult,noElem);
	//kernelMyScanV<<<grid,block>>>(dArrInput,noElem,dResult);
	CudaContext& cdactx = *ctx;
	mgpu::ScanExc(dResult, noElem,cdactx);
	//displayDeviceArr(dResult,noElem);
	CUCHECK(cudaDeviceSynchronize());
	CUCHECK(cudaGetLastError());
}

void  myReduction(int *dArrInput,int noElem,int &hResult){

	CudaContext& cdactx = *ctx;
	int total = Reduce(dArrInput, noElem, cdactx);
	//PMS_PRINT("Reduction total: %d\n\n", total);
	hResult = total;

	return;
}


cudaError_t  myScanV_beta()
{
	cudaError_t cudaStatus;

	int noElem = 5;
	int* ptr = (int*)malloc(sizeof(int)*noElem);
	for (int i = 0; i < noElem; i++)
	{
		ptr[i]=i;
		cout<<ptr[i]<<" ";
	}
	cout<<endl;
	int *p=nullptr;
	cudaMalloc((void**)&p,sizeof(int)*noElem);
	cudaMemcpy(p,ptr,noElem*sizeof(int),cudaMemcpyHostToDevice);
	cout<<"Input data"<<endl;
	kernelPrintdArr<<<1,100>>>(p,noElem);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess)
	{
		goto Error;
	}

	cout<<endl;
	//// int result = Reduce(p, noElem, *context);
	//// printf("Reduction total: %d\n\n", result);
	int result=0;
	//ScanExc(p, noElem, &result, *context);
	//CudaContext& cdactx=*ctx;
	CudaContext& cdactx = *ctx;
	ScanExc(p, noElem,cdactx);
	//   PrintArray(*data, "%4d", 10);
	kernelPrintdArr<<<1,100>>>(p,noElem);
	cudaDeviceSynchronize();
	//printf("Exclusive scan:\n");
	//printf("Scan total: %d\n", result);

	cudaFree(p);

	/*
	dim3 block(blocksize);
	dim3 grid((noElem + block.x -1)/block.x);

	CHECK(cudaStatus=cudaMalloc((void**)&dResult,noElem * sizeof(int)));
	if(cudaStatus!=cudaSuccess){
	goto Error;
	}
	kernelMyScanV<<<grid,block>>>(dArrInput,noElem,dResult);
	cudaDeviceSynchronize();
	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
	goto Error;
	}
	*/
Error:
	return cudaStatus;
}


__global__ void kernelCountNumberOfLabelVertex(int *d_LO,int *d_Lv,unsigned int sizeOfArrayLO){
	int i= blockDim.x*blockIdx.x + threadIdx.x;
	if(i<sizeOfArrayLO){
		if(d_LO[i]!=-1){
			d_Lv[d_LO[i]]=1;
		}
	}
}

void sumUntilReachZero(int *h_Lv,unsigned int n,int &result){
	for(int i=0;i<n && h_Lv[i]!=0;++i){
		++result;
	}
}

int  PMS::countNumberOfDifferentValue(int* d_LO,unsigned int sizeOfArrayLO, unsigned int &numberOfSaperateVertex){
	int status=0;
	cudaError_t cudaStatus;
	numberOfSaperateVertex=0;
	size_t nBytesLv = sizeOfArrayLO*sizeof(int);
	//cấp phát mảng d_Lv trên device
	int *d_Lv;
	cudaStatus=cudaMalloc((int**)&d_Lv,nBytesLv);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaMalloc d_Lv failed");
		goto Error;
	}
	else
	{
		cudaMemset(d_Lv,0,nBytesLv);
	}

	//Cấp phát threads
	dim3 block(blocksize); //current blocksize is 512
	dim3 grid((sizeOfArrayLO+block.x-1)/block.x);
	kernelCountNumberOfLabelVertex<<<grid,block>>>(d_LO,d_Lv,sizeOfArrayLO);

	cudaDeviceSynchronize();
	/*PMS_PRINT("\nElements of d_Lv:");
	kernelPrintdArr<<<grid,block>>>(d_Lv,sizeOfArrayLO);*/

	int* h_Lv=NULL;
	h_Lv=(int*)malloc(nBytesLv);
	if(h_Lv==NULL){
		PMS_PRINT("h_Lv malloc memory fail");
		exit(1);
	}
	cudaMemcpy(h_Lv,d_Lv,nBytesLv,cudaMemcpyDeviceToHost);
	cudaStatus=cudaDeviceSynchronize();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaDeviceSynchronize fail",cudaStatus);
		status = -1;
		goto Error;
	}
	int result=0;
	sumUntilReachZero(h_Lv,sizeOfArrayLO,result);
	numberOfSaperateVertex=result;	

Error:
	cudaFree(d_Lv);	
	return status;
}
//Kernel trích tất cả các mở rộng hợp lệ ban đầu vào mảng d_Extension
__global__ void kernelGetAndStoreExtension(int *d_O,int *d_LO,unsigned int numberOfElementd_O,int *d_N,int *d_LN,unsigned int numberOfElementd_N,Extension *d_Extension){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i<numberOfElementd_O){
		if (d_O[i]!=-1){
			int j;
			int ek;
			//PMS_PRINT("\nThread:%d",i);	
			for(j=i+1;j<numberOfElementd_O;++j){	
				if(d_O[j]!=-1) {break;}	
			}

			if (j==numberOfElementd_O) {
				ek=numberOfElementd_N;
			}
			else
			{
				ek=d_O[j];
			}
			//PMS_PRINT("\n[%d]:%d",i,ek);
			for(int k=d_O[i];k<ek;k++){
				//do something
				int index= k;
				d_Extension[index].vi=0; //không cần gán nữa vì đã khởi tạo trong định nghĩa cấu trúc Extension
				d_Extension[index].vj=1;
				d_Extension[index].li=d_LO[i];
				d_Extension[index].lij=d_LN[k];
				d_Extension[index].lj=d_LO[d_N[k]];
				d_Extension[index].vgi=i;
				d_Extension[index].vgj=d_N[k];
				//PMS_PRINT("\n[%d]:%d",i,index);
				/*PMS_PRINT("\n[%d]: DFS code:(%d,%d,%d,%d,%d)  (vgi,vgj):(%d,%d)\n",k,d_Extension[i].vi,d_Extension[i].vj,d_Extension[i].li,
				d_Extension[i].lij,d_Extension[i].lj,d_Extension[i].vgi,d_Extension[i].vgj);*/
			}
		}
	}
}


int PMS::getAndStoreExtension(Extension *&d_Extension){
	int status =0;
	cudaError_t cudaStatus;
	dim3 block(blocksize);
	unsigned int numberOfElementd_O = hdb.at(0).noElemdO;
	dim3 grid((numberOfElementd_O+block.x-1)/block.x);

	kernelGetAndStoreExtension<<<grid,block>>>(hdb.at(0).dO,hdb.at(0).dLO,numberOfElementd_O,hdb.at(0).dN,hdb.at(0).dLN,hdb.at(0).noElemdN,d_Extension);

	CHECK(cudaStatus=cudaDeviceSynchronize());
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaDeviceSynchronize kernelGetAndStoreExtension failed",cudaStatus);
		status =-1;
		goto Error;
	}


Error:
	return status;
}


int PMS::extractAllEdgeInDB(){
	int status = 0;
	cudaError_t cudaStatus;
	arrExtension arrE;
	//cấp phát bộ nhớ cho d_Extension
	arrE.noElem =hdb.at(0).noElemdN; //Lấy số lượng cạnh của tất cả các đồ thị
	size_t nBytesOfArrayExtension = arrE.noElem*sizeof(Extension); //Cấp phát bộ nhớ để lưu trữ tất cả các mở rộng ban đầu tương ứng với số lượng cạnh thu được;

	CHECK(cudaStatus=cudaMalloc((Extension**)&arrE.dExtension,nBytesOfArrayExtension));
	if(cudaStatus != cudaSuccess){
		status=-1;
		goto Error;
	}

	//Trích tất cả các cạnh từ database rồi lưu vào d_Extension
	FUNCHECK(status  = getAndStoreExtension(arrE.dExtension));
	if(status !=0){
		goto Error;
	}
	hExtension.push_back(arrE);
Error:
	return status;
}

__global__ void kernelPrintExtention(Extension *d_Extension,int n){
	int i=blockIdx.x*blockDim.x + threadIdx.x;
	if (i<n){		
		PMS_PRINT("\n[%d]: DFS code:(%d,%d,%d,%d,%d)  (vgi,vgj):(%d,%d)\n",i,d_Extension[i].vi,d_Extension[i].vj,d_Extension[i].li,d_Extension[i].lij,d_Extension[i].lj,d_Extension[i].vgi,d_Extension[i].vgj);
	}
}

int PMS::displayArrExtension(Extension *dExtension,int noElem){
	int status =0;
	cudaError_t cudaStatus;
	//dim3 block(blocksize);
	//dim3 grid((noElem + block.x - 1)/block.x);

	//kernelPrintExtention<<<grid,block>>>(dExtension,noElem);
	//cudaDeviceSynchronize();
	Extension *hExtension = (Extension*)malloc(sizeof(Extension)*noElem);
	if(hExtension==NULL){
		status=-1;
		PMS_PRINT("\n Malloc hExtension in displayArrExtension() failed");
		goto Error;
	}
	CHECK(cudaStatus=cudaMemcpy(hExtension,dExtension,sizeof(Extension)*noElem,cudaMemcpyDeviceToHost));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	for (int i = 0; i < noElem; i++)
	{
		PMS_PRINT("\n[%d]: DFS code:(%d,%d,%d,%d,%d)  (vgi,vgj):(%d,%d)\n",i,hExtension[i].vi,hExtension[i].vj,hExtension[i].li,hExtension[i].lij,hExtension[i].lj,hExtension[i].vgi,hExtension[i].vgj);
	}
Error:
	return status;
}

int PMS::displayArrUniEdge(UniEdge* dUniEdge,int noElem){
	cudaError_t cudaStatus;
	int status =0;
	UniEdge *hUniEdge = (UniEdge*)malloc(sizeof(UniEdge)*noElem);
	if(hUniEdge==NULL){
		status=-1;
		PMS_PRINT("\n malloc hUniEdge in displayArrUniEdge() failed");
		goto Error;
	}
	CHECK(cudaStatus=cudaMemcpy(hUniEdge,dUniEdge,sizeof(UniEdge)*noElem,cudaMemcpyDeviceToHost));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	for (int i = 0; i < noElem; i++)
	{
		PMS_PRINT("\n U[%d]: (vi vj):(%d %d) (li lij lj) = (%d %d %d)",i,hUniEdge[i].vi,hUniEdge[i].vj,hUniEdge[i].li,hUniEdge[i].lij,hUniEdge[i].lj);
	}
	//	free(hUniEdge);
Error:
	return status;
}


__global__ void	kernelValidEdge(Extension *d_Extension,int *dV,unsigned int numberElementd_Extension){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i<numberElementd_Extension){	
		if(d_Extension[i].li<=d_Extension[i].lj){
			dV[i]=1;
		}
	}
}


cudaError_t validEdge(Extension *d_Extension,int *&dV,unsigned int numberElementd_Extension){
	cudaError_t cudaStatus;

	dim3 block(blocksize);
	dim3 grid(numberElementd_Extension+block.x-1/block.x);

	kernelValidEdge<<<grid,block>>>(d_Extension,dV,numberElementd_Extension);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess){
		goto Error;
	}

	CHECK(cudaStatus=cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		goto Error;
	}
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaDeviceSynchronize kernelValidEdge failed",cudaStatus);
		goto Error;
	}

Error:
	return cudaStatus;
}



__global__ void kernelPrintdArr(float *dArr,int noElem){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElem){
		PMS_PRINT("A[%d]:%d   ",i,dArr[i]);
	}
}
__global__ void kernelPrintdArr(int *dArr,int noElem){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElem){
		PMS_PRINT("dArr[%d]:%d ",i,dArr[i]);
	}
}
int PMS::displayDeviceArr(int *dArr,int noElem){
	int status =0;
	//dim3 block(blocksize);
	//dim3 grid((noElem + block.x -1)/block.x);
	//kernelPrintdArr<<<grid,block>>>(dArr,noElem);
	//cudaDeviceSynchronize();
	//cudaError_t cudaStatus = cudaGetLastError();
	//if(cudaStatus!=cudaSuccess){
	//	fprintf(stderr,"\n kernelDisplayDeviceArr() in displayDeviceArr() failed",cudaStatus);
	//	status = -1;
	//	goto Error;
	//}

	int *temp = (int*)malloc(sizeof(int)*noElem);
	if(temp==NULL){
		PMS_PRINT("\n Malloc temp in displayDeviceArr() failed");
		status=-1;
		goto Error;	
	}

	CHECK(cudaMemcpy(temp,dArr,noElem*sizeof(int),cudaMemcpyDeviceToHost));

	for (int i = 0; i < noElem; i++)
	{
		PMS_PRINT(" A[%d]:%d  ",i,temp[i]);
	}

	free(temp);
Error:

	return 0;
}

int displayDeviceArr(int *dArr,int noElem){
	int status =0;
	int *temp = (int*)malloc(sizeof(int)*noElem);
	if(temp==NULL){
		PMS_PRINT("\n Malloc temp in displayDeviceArr() failed");
		status=-1;
		goto Error;
	}
	CHECK(cudaMemcpy(temp,dArr,noElem*sizeof(int),cudaMemcpyDeviceToHost));
	for (int i = 0; i < noElem; i++)
	{
		PMS_PRINT(" A[%d]:%d  ",i,temp[i]);
	}
	free(temp);
Error:
	return 0;
}


int PMS::displayDeviceArr(float *dArr,int noElem){
	int status =0;
	//dim3 block(blocksize);
	//dim3 grid((noElem + block.x -1)/block.x);
	//kernelPrintdArr<<<grid,block>>>(dArr,noElem);
	//cudaDeviceSynchronize();
	//cudaError_t cudaStatus = cudaGetLastError();
	//if(cudaStatus!=cudaSuccess){
	//	fprintf(stderr,"\n kernelDisplayDeviceArr() in displayDeviceArr() failed",cudaStatus);
	//	status = -1;
	//	goto Error;
	//}
	float *temp = (float*)malloc(sizeof(float)*noElem);
	if(temp==NULL){
		PMS_PRINT("\n Malloc temp in displayDeviceArr() failed");
		status=-1;
		goto Error;	
	}

	CHECK(cudaMemcpy(temp,dArr,noElem*sizeof(float),cudaMemcpyDeviceToHost));

	for (int i = 0; i < noElem; i++)
	{
		int a = (int)temp[i];
		PMS_PRINT(" A[%d]:%d  ",i,a);
	}

	free(temp);
Error:
	return status;
}


//in use
__global__ void kernelGetSize(int *dV,int *dVScanResult,int noElem,int *size)
{
	*size = dVScanResult[noElem-1];
	if(dV[noElem-1]==1)
	{
		*size = *size + 1;
	}
}


cudaError_t getSizeBaseOnScanResult(int *dV,int *dVScanResult,int noElem,int &output)
{
	cudaError_t cudaStatus;
	int temp=0;
	int *size=nullptr;
	CHECK(cudaStatus=cudaMalloc((void**)&size,sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		goto Error;
	}

	CHECK(cudaStatus=cudaMemset(size,0,sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		goto Error;
	}

	kernelGetSize<<<1,1>>>(dV,dVScanResult,noElem,size);
	CHECK(cudaStatus =cudaDeviceSynchronize());
	if(cudaStatus != cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize kernelGetSize() in getSizeBaseOnResult() failed",cudaStatus);
		goto Error;
	}
	CHECK(cudaStatus=cudaGetLastError());
	if(cudaStatus != cudaSuccess){
		fprintf(stderr,"\n kernelGetSize() in getSizeBaseOnResult() failed",cudaStatus);
		goto Error;
	}

	CHECK(cudaMemcpy(&temp,size,sizeof(int),cudaMemcpyDeviceToHost));
	output = (int)temp;

	cudaFree(size);
Error:

	return cudaStatus;
}

//inuse
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
}

//in use
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
	cudaFree(size);
}

int allocate_gpu_memory(EXT* &d_array,int noElem)
{
	cudaError_t cudaStatus;
	size_t n_bytes = sizeof(EXT)*noElem;
	CHECK(cudaStatus=cudaMalloc((void**)&d_array,n_bytes));
	if(cudaStatus!=cudaSuccess) return -1;
	return 0;
}

__global__ void kernelExtractValidExtension(Extension *d_Extension,int *dV,int *dVScanResult,int numberElementd_Extension,Extension *d_ValidExtension){
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
		}
	}
}
///<summary>	
///Trích các mở rộng duy nhất ban đầu
///</summary>
__global__ void kernelExtractValidExtension_pure(Extension *d_Extension,int *dV,int *dVScanResult,int numberElementd_Extension,EXT *d_ValidExtension){
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

cudaError_t extractValidExtension(Extension *d_Extension,int *dV,int *dVScanResult, int numberElementd_Extension,Extension *&d_ValidExtension){
	cudaError_t cudaStatus;

	//printfExtension(d_Extension,numberElementd_Extension);

	dim3 block(blocksize);
	dim3 grid((numberElementd_Extension+block.x)/block.x);

	kernelExtractValidExtension<<<grid,block>>>(d_Extension,dV,dVScanResult,numberElementd_Extension,d_ValidExtension);

	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if (cudaStatus != cudaSuccess){
		fprintf(stderr,"\nkernelGetValidExtension failed");
		goto Error;
	}

Error:
	return cudaStatus;
}

cudaError_t extractValidExtension_pure(Extension *d_Extension,int *dV,int *dVScanResult, int numberElementd_Extension,EXT *&d_ValidExtension)
{
	cudaError_t cudaStatus;

	//printfExtension(d_Extension,numberElementd_Extension);

	dim3 block(blocksize);
	dim3 grid((numberElementd_Extension+block.x)/block.x);

	kernelExtractValidExtension_pure<<<grid,block>>>(d_Extension,dV,dVScanResult,numberElementd_Extension,d_ValidExtension);

	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess)
	{
		goto Error;
	}

	CHECK(cudaStatus=cudaGetLastError());
	if (cudaStatus != cudaSuccess){
		fprintf(stderr,"\nkernelGetValidExtension failed");
		goto Error;
	}

Error:
	return cudaStatus;
}

int PMS::getValidExtension_pure()
{
	cout<<"Trich tat ca cac mo rong hop le ban dau"<<endl; //comment
	int status = 0;
	cudaError_t cudaStatus;
	//Phase 1: đánh dấu vị trí những cạnh hợp lệ (li<=lj)

	//int numberElementd_Extension = hExtension.at(0).noElem; //Lấy số lượng các phần tử trong dExtension.
	//hLevelEXT.at(objLevel.Level).noElem = hExtension.at(0).noElem; //Lấy số lượng các phần tử trong dExtension.
	int *dV;
	size_t nBytesdV= hExtension.at(0).noElem *sizeof(int);

	CHECK(cudaStatus=cudaMalloc((void**)&dV,nBytesdV));
	if (cudaStatus!= cudaSuccess){
		fprintf(stderr,"cudaMalloc array V failed",cudaStatus);
		status = -1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(dV,0,nBytesdV));
		if(cudaStatus!=cudaSuccess){
			status=-1;
			goto Error;
		}
	}
	//Đánh dấu các mở rộng hợp lệ trong hExtension.at(0).dExtension
	//cout<<"\tDanh dau cac mo rong hop le ban dau trong dExtension"<<endl; //comment
	CHECK(cudaStatus=validEdge(hExtension.at(0).dExtension,dV,hExtension.at(0).noElem));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaDeviceSynchronize validEdge failed",cudaStatus);
		status = -1;
		goto Error;
	}
	/*int(status=extractValidExtension(dV);
	if(cudaStatus!=cudaSuccess){
	fprintf(stderr,"cudaDeviceSynchronize validEdge failed",cudaStatus);
	status = -1;
	goto Error;
	}*/

	//Chép kết quả của dV sang hV để xem kết quả trong dV
	/*int *hV = (int*)malloc(sizeof(int)*hExtension.at(0).noElem);
	CHECK(cudaStatus=cudaMemcpy(hV,dV,sizeof(int)*hExtension.at(0).noElem,cudaMemcpyDeviceToHost));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}*/

	/*PMS_PRINT("\n ************ dV **************\n");
	for (int i = 0; i < hExtension.at(0).noElem; i++)
	{
		int temp = hV[i];
		PMS_PRINT("[%d]:%d ",i,temp);
	}*/

	int* dVScanResult;
	CHECK(cudaStatus=cudaMalloc((void**)&dVScanResult,hExtension.at(0).noElem*sizeof(int)));
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"Cuda Malloc failed",cudaStatus);
		goto Error;
	}	
	else
	{
		CHECK(cudaStatus=cudaMemset(dVScanResult,0,hExtension.at(0).noElem*sizeof(int)));
		if(cudaStatus!=cudaSuccess){
			status=-1;
			goto Error;
		}

	}
	//Exclusive scan mảng V và lưu kết quả scan vào mảng index
	//scanV(dV,numberElementd_Extension,dVScanResult);
	CHECK(cudaStatus=myScanV(dV,hExtension.at(0).noElem,dVScanResult));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	//Kiểm tra kết quả scan dV
	/*PMS_PRINT("\n ************ dVScanResult **************\n");
	CHECK(cudaStatus=cudaMemcpy(hV,dVScanResult,sizeof(int)*hExtension.at(0).noElem,cudaMemcpyDeviceToHost));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}*/



	/*for (int i = 0; i < hExtension.at(0).noElem; i++)
	{
		int temp = hV[i]; 
		PMS_PRINT("[%d]:%d ",i,temp);
	}*/

	
	

	//Phase 2: trích những cạnh hợp lệ sang một mảng khác dValidExtension
	//arrExtension arrValidExtension;
	//hValidExtension.resize(1);

	
	hLevelEXT.resize(1); 
	hLevelEXT.at(objLevel.Level).noElem=1;
	hLevelEXT.at(objLevel.Level).vE.resize(1); //Ban đầu chúng ta chỉ có 1 tập các mở rộng hợp lệ
	////Hàm getSizeBaseOnScanResult dùng để đếm xem có bao nhiêu cạnh mở rộng hợp lệ.
	////Có thể thay thế hàm này bởi hàm myReduction
	//timer.start();
	/*CHECK(cudaStatus=getSizeBaseOnScanResult(dV,dVScanResult,hExtension.at(0).noElem,\
		hLevelEXT.at(objLevel.Level).vE.at(0).noElem));

	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	std::printf("Time getSizeBaseOnScanResult: %f (ms)\n",timer.getTime());
	timer.reset();

	timer.start();*/
	myReduction(dV,hExtension.at(0).noElem,hLevelEXT.at(objLevel.Level).vE.at(0).noElem);

	/*timer.stop();
	std::printf("Time myReduction: %f (ms)\n",timer.getTime());
	timer.reset();*/
	//PMS_PRINT("\n arrValidExtension.noElem:%d",hLevelEXT.at(objLevel.Level).vE.at(0).noElem);
	////Việc cấp phát bộ nhớ trên device có thể dùng user function allocate_gpu_memory. Nếu có lỗi thì giải reset device và exit luôn.
	//CHECK(cudaStatus=cudaMalloc((void**)&(hLevelEXT.at(objLevel.Level).vE.at(0).dArrExt),sizeof(EXT)*hLevelEXT.at(objLevel.Level).vE.at(0).noElem));
	//if(cudaStatus!=cudaSuccess){
	//	status=-1;
	//	goto Error;
	//}
	
	//timer.start();
	FUNCHECK(status=allocate_gpu_memory(hLevelEXT.at(objLevel.Level).vE.at(0).dArrExt,hLevelEXT.at(objLevel.Level).vE.at(0).noElem));
	if (status != 0) goto Error;

	/*timer.stop();
	std::printf("Processing time: %f (ms)\n", timer.getTime());
	timer.reset();*/

	CHECK(cudaStatus=extractValidExtension_pure(hExtension.at(0).dExtension,dV,dVScanResult,hExtension.at(0).noElem,hLevelEXT.at(objLevel.Level).vE.at(0).dArrExt));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	//displayArrExtension(arrValidExtension.dExtension,arrValidExtension.noElem);

	//hValidExtension.push_back(arrValidExtension);
	/*PMS_PRINT("\n************hLevelEXT.at(0).vE.at(0).dArrExt***********\n");
	FUNCHECK(status=displaydArrEXT(hLevelEXT.at(objLevel.Level).vE.at(0).dArrExt,hLevelEXT.at(objLevel.Level).vE.at(0).noElem));
	if(status!=0){
		goto Error;
	}*/



	CHECK(cudaStatus=cudaFree(dV));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus=cudaFree(dVScanResult));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	//CHECK(cudaStatus=cudaFree(hExtension.at(0).dExtension)); //Chưa giải phóng vì cần phải tìm các đồ thị chứa các frequent subgraph
	//if(cudaStatus!=cudaSuccess){
	//	status=-1;
	//	goto Error;
	//}
	//hExtension.clear();


	//free(hV);
Error:	
	return status;
}

__global__ void kernelMarkLabelEdge(Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,unsigned int Lv,unsigned int Le,int *d_allPossibleExtension){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i<noElem_d_ValidExtension){
		int index=	d_ValidExtension[i].li*Lv*Le + d_ValidExtension[i].lij*Lv + d_ValidExtension[i].lj;
		d_allPossibleExtension[index]=1;
	}
}
__global__ void kernelMarkLabelEdge_pure(EXT *d_ValidExtension,unsigned int noElem_d_ValidExtension,unsigned int Lv,unsigned int Le,int *d_allPossibleExtension)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i<noElem_d_ValidExtension){
		int index=	d_ValidExtension[i].li*Lv*Le + d_ValidExtension[i].lij*Lv + d_ValidExtension[i].lj;
		d_allPossibleExtension[index]=1;
	}
}

cudaError_t markLabelEdge(Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,unsigned int Lv,unsigned int Le,int *&d_allPossibleExtension){
	cudaError_t cudaStatus;

	dim3 block(blocksize);
	dim3 grid((noElem_d_ValidExtension+block.x-1)/block.x);

	kernelMarkLabelEdge<<<grid,block>>>(d_ValidExtension,noElem_d_ValidExtension,Lv,Le,d_allPossibleExtension);
	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"kernelMarkLabelEdge in markLabelEdge() failed");
		goto Error;
	}

Error:
	return cudaStatus;
}
//Các cạnh mở rộng hợp lệ có thể giống nhau==> Hàm này sẽ đi ánh xạ chúng vào không gian d_allPossibleExtension.
cudaError_t markLabelEdge_pure(EXT *&d_ValidExtension,unsigned int noElem_d_ValidExtension,unsigned int Lv,unsigned int Le,int *&d_allPossibleExtension){
	cudaError_t cudaStatus;

	dim3 block(blocksize);
	dim3 grid((noElem_d_ValidExtension+block.x-1)/block.x);

	kernelMarkLabelEdge_pure<<<grid,block>>>(d_ValidExtension,noElem_d_ValidExtension,Lv,Le,d_allPossibleExtension);
	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"kernelMarkLabelEdge in markLabelEdge() failed");
		goto Error;
	}

Error:
	return cudaStatus;
}
//Ánh xạ từ vị trí trong d_allPossibleExtension sang cạnh tương ứng trong UniEdge 
__global__ void kernelCalcLabelAndStoreUniqueExtension(int *d_allPossibleExtension,int *d_allPossibleExtensionScanResult,unsigned int noElem_allPossibleExtension,UniEdge *d_UniqueExtension,unsigned int Le,unsigned int Lv){
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

__global__ void kernelReverseMappingToUniEdgeLabel(int *d_allPossibleExtension,int *d_allPossibleExtensionScanResult,unsigned int noElem_allPossibleExtension,UniEdge *d_UniqueExtension,unsigned int Le,unsigned int Lv){
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

//Ánh xạ và lưu cạnh vào dUniEdge từ vị trí có giá trị 1 trong d_allPossibleExtension
cudaError_t calcLabelAndStoreUniqueExtension(int *d_allPossibleExtension,int *d_allPossibleExtensionScanResult,unsigned int noElem_allPossibleExtension,UniEdge *&d_UniqueExtension,unsigned int noElem_d_UniqueExtension,unsigned int Le,unsigned int Lv){
	cudaError_t cudaStatus;

	dim3 block(blocksize);
	dim3 grid((noElem_allPossibleExtension+block.x-1)/block.x);
	kernelCalcLabelAndStoreUniqueExtension<<<grid,block>>>(d_allPossibleExtension,d_allPossibleExtensionScanResult,noElem_allPossibleExtension,d_UniqueExtension,Le,Lv);

	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"kernelCalcLabelAndStoreUniqueExtension in CalcLabelAndStoreUniqueExtension() failed");
		goto Error;
	}

Error:
	return cudaStatus;
}


cudaError_t reverseMappingToUniEdgeLabel(int *d_allPossibleExtension,int *d_allPossibleExtensionScanResult,unsigned int noElem_allPossibleExtension,UniEdge *&d_UniqueExtension,unsigned int noElem_d_UniqueExtension,unsigned int Le,unsigned int Lv){
	cudaError_t cudaStatus;

	dim3 block(blocksize);
	dim3 grid((noElem_allPossibleExtension+block.x-1)/block.x);
	kernelReverseMappingToUniEdgeLabel<<<grid,block>>>(d_allPossibleExtension,d_allPossibleExtensionScanResult,noElem_allPossibleExtension,d_UniqueExtension,Le,Lv);
	cudaDeviceSynchronize();
	CHECK(cudaStatus=cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		goto Error;
	}

Error:
	return cudaStatus;
}

//Tăng Level
void PMS::increaseLevel()
{
	objLevel.prevLevel=objLevel.Level;
	++objLevel.Level;
	objLevel.size=objLevel.Level+1;

	hLevelEXT.resize(objLevel.size);
	hLevelUniEdgeSatisfyMinsup.resize(objLevel.size);
	hLevelPtrEmbedding.resize(objLevel.size);
	hLevelRMP.resize(objLevel.size);
	hLevelListVerRMP.resize(objLevel.size);
	hLevelUniEdgeSatisfyMinDFSCODE.resize(objLevel.size); //Lưu các unique edge thoả minDFSCODE ở từng level
}
//Giảm Level
void PMS::decreaseLevel()
{
	objLevel.size=objLevel.Level;	
	--objLevel.Level;	
	objLevel.prevLevel=objLevel.Level-1;

	hLevelEXT.pop_back();
	hLevelUniEdgeSatisfyMinsup.pop_back();
	hLevelPtrEmbedding.pop_back();
	hLevelRMP.pop_back();
	hLevelListVerRMP.pop_back();
	hLevelUniEdgeSatisfyMinDFSCODE.pop_back();
}



//Trích các mở rộng duy nhất
int PMS::extractUniEdge(){
	int status=0;
	cudaError_t	cudaStatus;

	//Tính số lượng tất cả các cạnh có thể có dựa vào nhãn của chúng
	unsigned int noElem_dallPossibleExtension=Le*Lv*Lv; //(Mỗi một đỉnh sẽ có thể có Le*Lv mở rộng. Mà chúng ta có Lv đỉnh, nên ta có: Le*Lv*Lv mở rộng có thể có).
	int *d_allPossibleExtension;

	//cấp phát bộ nhớ cho mảng d_allPossibleExtension
	CHECK(cudaStatus=cudaMalloc((void**)&d_allPossibleExtension,noElem_dallPossibleExtension*sizeof(int)));
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaMalloc d_allPossibleExtension failed",cudaStatus);
		status = -1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(d_allPossibleExtension,0,noElem_dallPossibleExtension*sizeof(int)));
		if(cudaStatus!=cudaSuccess){
			goto Error;
		}
	}
	//Hàm markLabelEdge hoạt động theo nguyên tắc: "Mỗi mở rộng trong dExtension đều có 1 vị trí duy nhất trong d_allPossibleExtension. Và nhiệm vụ của hàm này là bậc giá trị 1 cho vị trí đó"
	//CHECK(cudaStatus=markLabelEdge(hValidExtension.at(0).dExtension,hValidExtension.at(0).noElem,Lv,Le,d_allPossibleExtension)); 
	CHECK(cudaStatus=markLabelEdge_pure(hLevelEXT.at(objLevel.Level).vE.at(0).dArrExt,hLevelEXT.at(objLevel.Level).vE.at(0).noElem,Lv,Le,d_allPossibleExtension)); 
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"getUniqueExtension failed",cudaStatus);
		status=-1;
		goto Error;
	}


	int *d_allPossibleExtensionScanResult;
	CHECK(cudaStatus=cudaMalloc((void**)&d_allPossibleExtensionScanResult,noElem_dallPossibleExtension*sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaMalloc d_allPossibleExtensionScanResult failed");
		status = -1;
		goto Error;
	}
	// PMS_PRINT("\n **************** hValidExtension ****************\n");
	//displayArrExtension(hValidExtension.at(0).dExtension,hValidExtension.at(0).noElem);

	//Exclusive scan mảng d_allPossibleExtension và lưu kết quả vào mảng d_allPossibleExtensionScanResult
	//cudaStatus = scanV(d_allPossibleExtension,noElem_dallPossibleExtension,d_allPossibleExtensionScanResult);
	//if(cudaStatus!=cudaSuccess){
	//	fprintf(stderr,"\n ScanV() in computeSupport() failed");
	//	status = -1;
	//	goto Error;
	//}
	CHECK(cudaStatus=myScanV(d_allPossibleExtension,noElem_dallPossibleExtension,d_allPossibleExtensionScanResult));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	//	 PMS_PRINT("\n **************** hValidExtension ****************\n");
	//displayArrExtension(hValidExtension.at(0).dExtension,hValidExtension.at(0).noElem);


	//PMS_PRINT("\n**********d_allPossibleExtension************\n");
	//displayDeviceArr(d_allPossibleExtension,noElem_dallPossibleExtension);


	arrUniEdge strUniEdge;
	int noElem_d_UniqueExtension=0;
	//Tính kích thước của mảng d_UniqueExtension dựa vào kết quả exclusive scan
	CHECK(cudaStatus=getSizeBaseOnScanResult(d_allPossibleExtension,d_allPossibleExtensionScanResult,noElem_dallPossibleExtension,noElem_d_UniqueExtension));
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"getLastElement() in extractUniEdge() failed",cudaStatus);
		status = -1;
		goto Error;
	}
	//PMS_PRINT("\n\nnoElem_d_UniqueExtension:%d",noElem_d_UniqueExtension);
	strUniEdge.noElem = noElem_d_UniqueExtension;
	//Tạo mảng d_UniqueExtension với kích thước mảng vừa tính được
	CHECK(cudaStatus=cudaMalloc((void**)&strUniEdge.dUniEdge,noElem_d_UniqueExtension*sizeof(UniEdge)));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaMalloc d_UniqueExtension in extractUniEdge() failed",cudaStatus);
		status = -1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(strUniEdge.dUniEdge,0,noElem_d_UniqueExtension*sizeof(UniEdge)));
		if(cudaStatus!=cudaSuccess){
			status = -1;
			goto Error;
		}
	}

	//Ánh xạ ngược lại từ vị trí trong d_allPossibleExtension thành cạnh và lưu kết quả vào d_UniqueExtension
	CHECK(cudaStatus=calcLabelAndStoreUniqueExtension(d_allPossibleExtension,d_allPossibleExtensionScanResult,noElem_dallPossibleExtension,strUniEdge.dUniEdge,noElem_d_UniqueExtension,Le,Lv));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n\ncalcLabelAndStoreUniqueExtension() in extractUniEdge() failed",cudaStatus);
		status = -1;
		goto Error;
	}

	hUniEdge.push_back(strUniEdge);
	/*PMS_PRINT("\n **************** hUniEdge ****************\n");
	FUNCHECK(status=displayArrUniEdge(hUniEdge.at(0).dUniEdge,hUniEdge.at(0).noElem));
	if(status!=0){
		goto Error;
	}*/

	CHECK(cudaStatus=cudaFree(d_allPossibleExtension));
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	CHECK(cudaStatus=cudaFree(d_allPossibleExtensionScanResult));
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
Error:
	return status;
}



__global__ void kernelCalcBoundary(Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,int *dB,unsigned int maxOfVer){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i<noElem_d_ValidExtension-1){
		unsigned int graphIdAfter=d_ValidExtension[i+1].vgi/maxOfVer;
		unsigned int graphIdCurrent=d_ValidExtension[i].vgi/maxOfVer;
		unsigned int resultDiff=graphIdAfter-graphIdCurrent;
		dB[i]=resultDiff;
	}
}
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


cudaError_t calcBoundary(Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,int *&dB,unsigned int maxOfVer){
	cudaError_t cudaStatus;

	dim3 block(blocksize);
	dim3 grid((noElem_d_ValidExtension+block.x)/block.x);

	kernelCalcBoundary<<<grid,block>>>(d_ValidExtension,noElem_d_ValidExtension,dB,maxOfVer);

	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\kernelCalcBoundary in calcBoundary() failed");
		goto Error;
	}

Error:
	return cudaStatus;
}
//Xây dựng boundary cho các mở rộng hợp lệ trong d_ValidExtension để tính support
cudaError_t calcBoundary_pure(EXT *&d_ValidExtension,unsigned int noElem_d_ValidExtension,int *&dB,unsigned int maxOfVer)
{
	cudaError_t cudaStatus;

	dim3 block(blocksize);
	dim3 grid((noElem_d_ValidExtension+block.x)/block.x);

	kernelCalcBoundary_pure<<<grid,block>>>(d_ValidExtension,noElem_d_ValidExtension,dB,maxOfVer);

	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess)
	{
		goto Error;
	}

	CHECK(cudaStatus=cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		//fprintf(stderr,"\kernelCalcBoundary in calcBoundary() failed");
		goto Error;
	}

Error:
	return cudaStatus;
}

//inuse
__global__ void kernelGetLastElement(int *dScanResult,unsigned int noElem,int *output)
{
	output[0]=dScanResult[noElem-1];
}

//inuse
cudaError_t getLastElement(int *dScanResult,unsigned int noElem,int &output)
{
	cudaError_t cudaStatus=cudaSuccess;
	dim3 block(blocksize);
	dim3 grid((noElem+block.x-1)/block.x);

	int *value;
	CUCHECK(cudaMalloc((int**)&value,sizeof(int)));

	kernelGetLastElement<<<1,1>>>(dScanResult,noElem,value);
	CUCHECK(cudaDeviceSynchronize());
	CUCHECK(cudaGetLastError());

	CUCHECK(cudaMemcpy(&output,value,sizeof(int),cudaMemcpyDeviceToHost));

	//PMS_PRINT("\n\nnumberElementd_UniqueExtension:%d",numberElementd_UniqueExtension);

	CUCHECK(cudaFree(value));

Error:
	return cudaStatus;
}

__global__ void kernelSetValuedF(UniEdge *dUniEdge,int noElemdUniEdge,Extension *dValidExtension,int noElemdValidExtension,int *dBScanResult,int *dF,int noElemF){
	int i = blockDim.x * blockIdx.x +threadIdx.x;
	if(i<noElemdValidExtension){
		for (int j = 0; j < noElemdUniEdge; j++)
		{
			if(dUniEdge[j].li==dValidExtension[i].li && dUniEdge[j].lij==dValidExtension[i].lij &&	dUniEdge[j].lj==dValidExtension[i].lj){
				dF[dBScanResult[i]+j*noElemF]=1;
			}
		}
	}
}

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


__global__ void kernelCopyFromdFtoTempF(int *d_F,int *tempF,int from,int noElemNeedToCopy){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElemNeedToCopy){
		int index = from*noElemNeedToCopy + i;
		tempF[i]=d_F[index];
	}
}

cudaError_t calcSupport(UniEdge *dUniEdge,int noElemdUniEdge,Extension *dValidExtension,int noElemdValidExtension,int *dBScanResult,int *dF,int noElemF,int *&hResultSup){
	cudaError_t cudaStatus;

	//Đánh dấu những đồ thị chứa embedding trong mảng d_F
	dim3 block(blocksize);
	dim3 grid((noElemdValidExtension+block.x - 1)/block.x);
	kernelSetValuedF<<<grid,block>>>(dUniEdge,noElemdUniEdge,dValidExtension,noElemdValidExtension,dBScanResult,dF,noElemF);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess){
		return cudaStatus;
	}

	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() of kernelComputeSupport in computeSupport failed",cudaStatus);
		return cudaStatus;
	}

	//Duyệt qua mảng d_UniqueExtension, tính reduction cho mỗi segment i*noElemF, kết quả của reduction là độ support của cạnh i trong d_UniqueExtension
	int *tempF;
	CHECK(cudaStatus = cudaMalloc((void**)&tempF,noElemF*sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n CudaMalloc tempF in calcSupport() failed",cudaStatus);
		return cudaStatus;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(tempF,0,noElemF*sizeof(int)));
		if(cudaStatus!=cudaSuccess){
			return cudaStatus;
		}
	}

	//float *resultSup; /* Lưu kết quả reduction */
	hResultSup = (int*)malloc(noElemdUniEdge*sizeof(int));
	if (hResultSup==NULL){
		PMS_PRINT("\n Malloc hResultSup in calcSupport() failed");
		exit(1);
	}

	dim3 blocka(blocksize);
	dim3 grida((noElemF+blocka.x-1)/blocka.x);
	/*int from =0;*/	
	for (int i = 0; i < noElemdUniEdge; i++)
	{		
		//chép dữ liệu d_F sang tempF ứng theo các phần tử lần lược là i*noElemF, copy đúng noElemF
		/*from =i;*/				
		kernelCopyFromdFtoTempF<<<grid,block>>>(dF,tempF,i,noElemF);
		CHECK(cudaStatus=cudaDeviceSynchronize());
		if(cudaStatus!=cudaSuccess){
			return cudaStatus;
		}

		//reduction(tempF,noElemF,hResultSup[i]);		
	}
	////In độ hỗ trợ cho các cạnh tương ứng trong mảng kết quả resultSup
	//for (int i = 0; i < noElemdUniEdge; i++)
	//{
	//	PMS_PRINT("\n resultSup[%d]:%d",i,hResultSup[i]);
	//}

	CHECK(cudaStatus=cudaFree(tempF));
	if(cudaStatus!=cudaSuccess){
		return cudaStatus;
	}

	return cudaStatus;
}
//Tính support cho các mở rộng duy nhất
cudaError_t calcSupport_pure(UniEdge *dUniEdge,int noElemdUniEdge,EXT *dValidExtension,int noElemdValidExtension,int *dBScanResult,int *dF,int noElemF,int *&hResultSup)
{
	cudaError_t cudaStatus=cudaSuccess;

	//Đánh dấu những đồ thị chứa embedding trong mảng d_F
	dim3 block(blocksize);
	dim3 grid((noElemdValidExtension+block.x - 1)/block.x);
	kernelSetValuedF_pure<<<grid,block>>>(dUniEdge,noElemdUniEdge,dValidExtension,noElemdValidExtension,dBScanResult,dF,noElemF);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess)
	{
		return cudaStatus;
	}

	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess)
	{
		fprintf(stderr,"\n cudaDeviceSynchronize() of kernelComputeSupport in computeSupport failed",cudaStatus);
		return cudaStatus;
	}

	//Duyệt qua mảng d_UniqueExtension, tính reduction cho mỗi segment i*noElemF, kết quả của reduction là độ support của cạnh i trong d_UniqueExtension
	//int *tempF;
	//CHECK(cudaStatus = cudaMalloc((void**)&tempF,noElemF*sizeof(int)));
	//if(cudaStatus!=cudaSuccess)
	//{
	//	fprintf(stderr,"\n CudaMalloc tempF in calcSupport() failed",cudaStatus);
	//	return cudaStatus;
	//}
	//else
	//{
	//	CHECK(cudaStatus=cudaMemset(tempF,0,noElemF*sizeof(int)));
	//	if(cudaStatus!=cudaSuccess)
	//	{
	//		return cudaStatus;
	//	}
	//}

	//float *resultSup; /* Lưu kết quả reduction */
	hResultSup = (int*)malloc(noElemdUniEdge*sizeof(int));
	if (hResultSup==NULL)
	{
		PMS_PRINT("\n Malloc hResultSup in calcSupport() failed");
		exit(0);
	}

	//dim3 blocka(blocksize);
	//dim3 grida((noElemF+blocka.x-1)/blocka.x);
	/*int from =0;*/	
	//timer.start();
	//for (int i = 0; i < noElemdUniEdge; i++)
	//{		
	//	//chép dữ liệu d_F sang tempF ứng theo các phần tử lần lược là i*noElemF, copy đúng noElemF
	//	/*from =i;*/				
	//	kernelCopyFromdFtoTempF<<<grid,block>>>(dF,tempF,i,noElemF);
	//	CHECK(cudaStatus=cudaDeviceSynchronize());
	//	if(cudaStatus!=cudaSuccess)
	//	{
	//		return cudaStatus;
	//	}

	//	//reduction(tempF,noElemF,hResultSup[i]);
	//	//myReduce(tempF,noElemF,hResultSup[i]);
	//	myReduction(tempF,noElemF,hResultSup[i]);
	//	//cout<<"support: "<<(hResultSup[i])<<endl;
	//}
	//timer.stop();
	//std::printf("Time myReduction for each unique Extension: %f (ms)\n",timer.getTime());
	//timer.reset();
	////In độ hỗ trợ cho các cạnh tương ứng trong mảng kết quả resultSup
	//for (int i = 0; i < noElemdUniEdge; i++)
	//{
	//	PMS_PRINT("\n resultSup[%d]:%d",i,hResultSup[i]);
	//}
	int *d_supports = nullptr;
	//timer.start();
	int status=0;
	FUNCHECK(status=SegReduce(dF,noElemdUniEdge,noElemF,d_supports));
	if(status!=0){
		goto Error;
	}
	/*timer.stop();
	std::printf("Time myReduction for Segmented Extension: %f (ms)\n",timer.getTime());
	timer.reset();*/
	//displayDeviceArr(d_supports,noElemdUniEdge);
	CHECK(cudaStatus=cudaMemcpy(hResultSup,d_supports,noElemdUniEdge*sizeof(int),cudaMemcpyDeviceToHost));
	if(cudaStatus!=cudaSuccess){
		goto Error;
	}
	//CHECK(cudaStatus=cudaFree(tempF));
	//if(cudaStatus!=cudaSuccess)
	//{
	//	return cudaStatus;
	//}
	cudaFree(d_supports);
	Error:
	return cudaStatus;
}

__global__ void	kernelMarkUniEdgeSatisfyMinsup(int *dResultSup,int noElemUniEdge,int *dV,unsigned int minsup){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElemUniEdge){
		int temp = dResultSup[i];
		if(temp >= minsup){
			dV[i]=1;
		}
	}
}

__global__ void	kernelExtractUniEdgeSatifyMinsup(UniEdge *dUniEdge,int *dV,int *dVScanResult,int noElemUniEdge,UniEdge *dUniEdgeSatisfyMinsup,int *dSup,int *dResultSup)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElemUniEdge){
		if(dV[i]==1){
			dUniEdgeSatisfyMinsup[dVScanResult[i]].li = dUniEdge[i].li;
			dUniEdgeSatisfyMinsup[dVScanResult[i]].lij = dUniEdge[i].lij;
			dUniEdgeSatisfyMinsup[dVScanResult[i]].lj=dUniEdge[i].lj;
			dSup[dVScanResult[i]]=dResultSup[i];
		}
	}
}

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

__global__ void	kernelExtractUniEdgeSatifyMinsupV3(UniEdge *dUniEdge,int *dV,int *dVScanResult,int noElemUniEdge,UniEdge *dUniEdgeSatisfyMinsup,int *dSup,int *dResultSup){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElemUniEdge){
		if(dV[i]==1){
			dUniEdgeSatisfyMinsup[dVScanResult[i]].vi = dUniEdge[i].vi;
			dUniEdgeSatisfyMinsup[dVScanResult[i]].vj = dUniEdge[i].vj;
			dUniEdgeSatisfyMinsup[dVScanResult[i]].li = dUniEdge[i].li;
			dUniEdgeSatisfyMinsup[dVScanResult[i]].lij = dUniEdge[i].lij;
			dUniEdgeSatisfyMinsup[dVScanResult[i]].lj=dUniEdge[i].lj;
			dSup[dVScanResult[i]]=dResultSup[i];
		}
	}
}



//Trích các mở rộng duy nhất thoả minsup
int PMS::extractUniEdgeSatisfyMinsup(int *hResultSup,int noElemUniEdge,unsigned int minsup){
	int status=0;
	cudaError_t cudaStatus=cudaSuccess;
	//1. Cấp phát mảng trên device có kích thước bằng noElemUniEdge
	int *dResultSup=nullptr;
	CHECK(cudaStatus=cudaMalloc((void**)&dResultSup,noElemUniEdge*sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus=cudaMemcpy(dResultSup,hResultSup,noElemUniEdge*sizeof(int),cudaMemcpyHostToDevice));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	//PMS_PRINT("\n *******dResultSup********\n");
	//FUNCHECK(status=displayDeviceArr(dResultSup,noElemUniEdge));
	//if(status!=0){
	//	goto Error;
	//}


	//2. Đánh dấu 1 trên dV cho những phần tử thoả minsup
	int *dV=nullptr;
	CHECK(cudaStatus=cudaMalloc((void**)&dV,noElemUniEdge*sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus=cudaMemset(dV,0,sizeof(int)*noElemUniEdge));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	dim3 block(blocksize);
	dim3 grid((noElemUniEdge + block.x - 1)/block.x);
	kernelMarkUniEdgeSatisfyMinsup<<<grid,block>>>(dResultSup,noElemUniEdge,dV,minsup);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus=cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n kernelMarkUniEdgeSatisfyMinsup in extractUniEdgeSatisfyMinsup() failed",cudaStatus);
		status = -1;
		goto Error;
	}

	//PMS_PRINT("\n ***********dV**********\n");
	//FUNCHECK(status=displayDeviceArr(dV,noElemUniEdge));
	//if(cudaStatus!=cudaSuccess){
	//	status=-1;
	//	goto Error;
	//}
	int *dVScanResult=nullptr;
	CHECK(cudaStatus=cudaMalloc((void**)&dVScanResult,noElemUniEdge*sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	//CHECK(scanV(dV,noElemUniEdge,dVScanResult));
	CHECK(cudaStatus=myScanV(dV,noElemUniEdge,dVScanResult));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	//PMS_PRINT("\n ***********dVScanResult**********\n");
	//FUNCHECK(status=displayDeviceArr(dVScanResult,noElemUniEdge));
	//if(status!=0){
	//	goto Error;
	//}

	//hUniEdgeSatisfyMinsup.resize(1);
	hLevelUniEdgeSatisfyMinsup.resize(1);
	hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.resize(1);

	CHECK(cudaStatus=getSizeBaseOnScanResult(dV,dVScanResult,noElemUniEdge,hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(0).noElem));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus=cudaMalloc((void**)&hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(0).dArrUniEdge,hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(0).noElem*sizeof(UniEdge)));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(0).hArrSupport = (int*)malloc(sizeof(int)*hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(0).noElem);
	if (hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(0).hArrSupport ==NULL){
		PMS_PRINT("\n malloc hArrSup of hUniEdgeSatisfyMinsup failed()");
		exit(0);
	}


	int *dSup=nullptr;
	CHECK(cudaStatus=cudaMalloc((void**)&dSup,hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(0).noElem*sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}



	dim3 blocka(blocksize);
	dim3 grida((noElemUniEdge + blocka.x -1)/blocka.x);
	kernelExtractUniEdgeSatifyMinsup_pure<<<grida,blocka>>>(hUniEdge.at(0).dUniEdge,dV,dVScanResult,noElemUniEdge,hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(0).dArrUniEdge,dSup,dResultSup);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		status = -1;
		fprintf(stderr,"\n kernelExtractUniEdgeSatisfyMinsup() in extractUniEdgeSatisfyMinsup() failed",cudaStatus);
		goto Error;
	}
	//PMS_PRINT("\n ********hUniEdgeSatisfyMinsup.dUniEdge****************\n");
	//FUNCHECK(status=displayArrUniEdge(hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(0).dArrUniEdge,hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(0).noElem));
	//if(status!=0){
	//	goto Error;
	//}

	//PMS_PRINT("\n ********hUniEdgeSatisfyMinsup.dSup****************\n");
	//FUNCHECK(status=displayDeviceArr(dSup,hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(0).noElem));
	//if(status!=0){
	//	goto Error;
	//}


	CHECK(cudaStatus=cudaMemcpy(hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(0).hArrSupport,dSup,sizeof(int)*hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(0).noElem,cudaMemcpyDeviceToHost));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	//for (int i = 0; i < hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(0).noElem; ++i)
	//{
	//	PMS_PRINT("\n hArrSup:%d ",hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(0).hArrSupport[i]);
	//}

	CHECK(cudaStatus=cudaFree(dResultSup));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus=cudaFree(dV));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus=cudaFree(dVScanResult));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus=cudaFree(dSup));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	if(hUniEdge.at(0).noElem>0)
	{
		CHECK(cudaStatus=cudaFree(hUniEdge.at(0).dUniEdge));
		if(cudaStatus!=cudaSuccess)
		{
			status=-1;
			goto Error;
		}
		hUniEdge.clear();
	}

Error:
	return status;
}


int PMS::computeSupport()
{
	cout<<"Tinh support cho cac mo rong ban dau"<<endl; //comment
	int status=0;
	cudaError_t cudaStatus;
	/* Xây dựng Boundary cho mảng d_ValidExtension */
	//1. Cấp phát một mảng d_B và gán các giá trị 0 cho mọi phần tử của d_B
	//unsigned int noElement_dB=hValidExtension.at(0).noElem;
	unsigned int noElement_dB=hLevelEXT.at(objLevel.Level).vE.at(0).noElem;
	int* dB;
	CHECK(cudaStatus=cudaMalloc((int**)&dB,noElement_dB*sizeof(int)));
	if (cudaStatus!=cudaSuccess)
	{
		fprintf(stderr,"cudaMalloc dB in computeSupport() failed",cudaStatus);
		status = -1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(dB,0,noElement_dB*sizeof(int)));
		if(cudaStatus!=cudaSuccess)
		{
			status=-1;
			goto Error;
		}
	}
	//PMS_PRINT("\n**********dValidExtension*************\n");
	//displayArrExtension(hValidExtension.at(0).dExtension,noElement_dB);
	//PMS_PRINT("\n*********dB********\n");
	//displayDeviceArr(dB,noElement_dB);


	//Gián giá trị boundary cho d_B
	//CHECK(cudaStatus=calcBoundary(hValidExtension.at(0).dExtension,noElement_dB,dB,maxOfVer));
	CHECK(cudaStatus=calcBoundary_pure(hLevelEXT.at(objLevel.Level).vE.at(0).dArrExt,noElement_dB,dB,maxOfVer));
	if (cudaStatus!=cudaSuccess)
	{
		fprintf(stderr,"calcBoundary() in computeSupport() failed",cudaStatus);
		status=-1;
		goto Error;
	}

	//PMS_PRINT("\n**********dValidExtension*************\n");
	//FUNCHECK(status=displayArrExtension(hValidExtension.at(0).dExtension,noElement_dB));
	//FUNCHECK(status=displaydArrEXT(hLevelEXT.at(objLevel.Level).vE.at(0).dArrExt,hLevelEXT.at(objLevel.Level).vE.at(0).noElem));
	//if(status!=0)
	//{
	//	goto Error;
	//}

	//PMS_PRINT("\n*********dB********\n");
	//FUNCHECK(status=displayDeviceArr(dB,noElement_dB));
	//if(status!=0)
	//{
	//	goto Error;
	//}
	//2. Exclusive Scan mảng d_B
	int* dBScanResult;
	CHECK(cudaStatus=cudaMalloc((int**)&dBScanResult,noElement_dB*sizeof(int)));
	if(cudaStatus!=cudaSuccess)
	{
		fprintf(stderr,"cudaMalloc dBScanResult in computeSupport() failed",cudaStatus);
		status = -1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(dBScanResult,0,noElement_dB*sizeof(int)));
		if(cudaStatus!=cudaSuccess)
		{
			status=-1;
			goto Error;
		}

	}

	//cudaStatus=scanV(dB,noElement_dB,dBScanResult);
	//if(cudaStatus!=cudaSuccess){
	//	fprintf(stderr,"\nscanB function failed",cudaStatus);
	//	status =-1;
	//	goto Error;
	//}
	CHECK(cudaStatus=myScanV(dB,noElement_dB,dBScanResult));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	//PMS_PRINT("\n\n*******dBScanResult***********\n");
	//FUNCHECK(status=displayDeviceArr(dBScanResult,noElement_dB));
	//if(status!=0)
	//{
	//	goto Error;
	//}

	//3. Tính độ hỗ trợ cho các mở rộng trong d_UniqueExtension
	//3.1 Tạo mảng d_F có số lượng phần tử bằng với giá trị cuối cùng của mảng d_scanB_Result cộng 1 và gán giá trị 0 cho các phần tử.
	int noElemF=0;
	CHECK(cudaStatus=getLastElement(dBScanResult,noElement_dB,noElemF));
	if(cudaStatus!=cudaSuccess)
	{
		fprintf(stderr,"\ngetLastElement function failed",cudaStatus);
		status=-1;
		goto Error;
	}

	++noElemF;
	/*noElemGraphInExt=noElemF;*/

	PMS_PRINT("\n\n noElement_F:%d",noElemF);
	int noElem_d_UniqueExtension= hUniEdge.at(0).noElem;
	int *dF;
	CHECK(cudaStatus=cudaMalloc((int**)&dF,noElem_d_UniqueExtension*noElemF*sizeof(int)));
	if (cudaStatus!=cudaSuccess)
	{
		fprintf(stderr,"\ncudaMalloc dF in computeSupport() failed",cudaStatus);
		status =-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(dF,0,noElem_d_UniqueExtension*noElemF*sizeof(int)));
		if(cudaStatus!=cudaSuccess)
		{
			status=-1;
			goto Error;
		}
	}
	int *hResultSup=nullptr;
	CHECK(cudaStatus=calcSupport_pure(hUniEdge.at(0).dUniEdge,\
		hUniEdge.at(0).noElem,hLevelEXT.at(objLevel.Level).vE.at(0).dArrExt,\
		hLevelEXT.at(objLevel.Level).vE.at(0).noElem,dBScanResult,dF,noElemF,hResultSup));
	if(cudaStatus!=cudaSuccess)
	{
		fprintf(stderr,"\n calcSupport() in computeSupport() failed",cudaStatus);
		status =-1;
		goto Error;
	}

	////In độ hỗ trợ cho các cạnh tương ứng trong mảng kết quả h_resultSup
	//for (int i = 0; i < noElem_d_UniqueExtension; i++)
	//{
	//	PMS_PRINT("\n resultSup[%d]:%d",i,hResultSup[i]);
	//}
	//
	FUNCHECK(status=extractUniEdgeSatisfyMinsup(hResultSup,noElem_d_UniqueExtension,minsup));
	if(status!=0)
	{
		goto Error;
	}
	cout<<"\nDa trich duoc cac mo rong ban dau thoa minsub:"<<minsup<<endl; //comment

	CHECK(cudaStatus=cudaFree(dBScanResult));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus=cudaFree(dB));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

Error:
	return status;
}
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
int generate_segment_index(int noElem_of_graph_per_unique_ext,\
						   int number_unique_extension,\
						   int *&SegmentStarts)
{
	cudaError_t cudaStatus=cudaSuccess;
	int status =0;
	int noElem_segment = number_unique_extension;
	dim3 block(blocksize);
	dim3 grid((noElem_segment + block.x -1)/block.x);
	CHECK(cudaStatus= cudaMalloc((void**)&SegmentStarts,noElem_segment*sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	kernel_generate_segment_index<<<grid,block>>>(SegmentStarts, noElem_segment, noElem_of_graph_per_unique_ext);
	CHECK(cudaStatus = cudaDeviceSynchronize());
	
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus=cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	Error:
	return status;
}
int SegReduce(int* dF,int number_unique_extension,int noElem_of_graph_per_unique_ext,int *&resultsDevice) 
{
	int status = 0;
	cudaError_t cudaStatus = cudaSuccess;
	CudaContext& context = *ctx;
	int count = number_unique_extension*noElem_of_graph_per_unique_ext;
	int *SegmentStarts = nullptr;
	FUNCHECK(status=generate_segment_index(noElem_of_graph_per_unique_ext,number_unique_extension,SegmentStarts));
	if(status!=0){
		goto Error;
	}
	const int NumSegments = number_unique_extension;
	CHECK(cudaStatus = cudaMalloc((void**)&resultsDevice,number_unique_extension*sizeof(int)));
	if(cudaStatus!=cudaSuccess)
	{
		status = -1;
		goto Error;
	}
	SegReduceCsr(dF, SegmentStarts, count, number_unique_extension,\
				false, resultsDevice, (int)0, mgpu::plus<int>(), context);

	cudaFree(SegmentStarts);

	Error:
	return status;
}


__global__ void kernelGetGraphIdContainEmbedding(int li,int lij,int lj,\
												 Extension *d_ValidExtension,\
												 int noElem_d_ValidExtension,\
												 int *dV,unsigned int maxOfVer)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i<noElem_d_ValidExtension){
		if(	d_ValidExtension[i].li == li && d_ValidExtension[i].lij == lij && 	d_ValidExtension[i].lj == lj){
			int graphid = (d_ValidExtension[i].vgi/maxOfVer);
			dV[graphid]=1;
		}
	}
}

__global__ void kernelGetGraphIdContainEmbedding_pure(int li,int lij,int lj,EXT *d_ValidExtension,int noElem_d_ValidExtension,int *dV,unsigned int maxOfVer)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i<noElem_d_ValidExtension)
	{
		if(	d_ValidExtension[i].li == li && d_ValidExtension[i].lij == lij && 	d_ValidExtension[i].lj == lj)
		{
			int graphid = (d_ValidExtension[i].vgi/maxOfVer);
			dV[graphid]=1;
		}
	}
}


__global__ void kernelGetLastElementExtension(Extension *inputArray,unsigned int noEleInputArray,int *value,unsigned int maxOfVer){
	value[0] = inputArray[noEleInputArray-1].vgi/maxOfVer; /*Lấy global vertex id chia cho tổng số đỉnh của đồ thị (maxOfVer). Ở đây các đồ thị luôn có số lượng đỉnh bằng nhau (maxOfVer) */
}


cudaError_t getLastElementExtension(Extension* inputArray,unsigned int numberElementOfInputArray,int &outputValue,unsigned int maxOfVer){
	cudaError_t cudaStatus;

	int *temp=nullptr;
	CHECK(cudaMalloc((int**)&temp,sizeof(int)));
	//kernelPrintExtention<<<1,512>>>(inputArray,numberElementOfInputArray);
	//cudaDeviceSynchronize();
	//cudaStatus= cudaGetLastError();
	//if(cudaStatus != cudaSuccess){
	//	fprintf(stderr,"cudaDeviceSynchronize failed",cudaStatus);
	//	goto Error;
	//}

	/* Lấy graphId chứa embedding cuối cùng */
	kernelGetLastElementExtension<<<1,1>>>(inputArray,numberElementOfInputArray,temp,maxOfVer);
	cudaDeviceSynchronize();
	cudaStatus= cudaGetLastError();
	if(cudaStatus != cudaSuccess){
		fprintf(stderr,"cudaDeviceSynchronize failed",cudaStatus);
		goto Error;
	}

	CHECK(cudaMemcpy(&outputValue,temp,sizeof(int),cudaMemcpyDeviceToHost));
	//PMS_PRINT("\n\nnumberElementd_UniqueExtension:%d",numberElementd_UniqueExtension);

	cudaFree(temp);
Error:	
	return cudaStatus;	
}

cudaError_t getLastElementExtension_pure(EXT* inputArray,unsigned int numberElementOfInputArray,int &outputValue,unsigned int maxOfVer)
{
	cudaError_t cudaStatus;

	int *temp=nullptr;
	CHECK(cudaStatus=cudaMalloc((void**)&temp,sizeof(int)));
	if(cudaStatus!=cudaSuccess)
	{
		goto Error;
	}

	//kernelPrintExtention<<<1,512>>>(inputArray,numberElementOfInputArray);
	//cudaDeviceSynchronize();
	//cudaStatus= cudaGetLastError();
	//if(cudaStatus != cudaSuccess){
	//	fprintf(stderr,"cudaDeviceSynchronize failed",cudaStatus);
	//	goto Error;
	//}

	/* Lấy graphId chứa embedding cuối cùng */
	kernelGetLastElementEXT<<<1,1>>>(inputArray,numberElementOfInputArray,temp,maxOfVer);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess)
	{
		goto Error;
	}

	CHECK(cudaStatus= cudaGetLastError());
	if(cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"cudaDeviceSynchronize failed",cudaStatus);
		goto Error;
	}
	//Chép kết quả từ device sang host
	CHECK(cudaStatus = cudaMemcpy(&outputValue,temp,sizeof(int),cudaMemcpyDeviceToHost));
	if(cudaStatus!=cudaSuccess)
	{
		goto Error;
	}

	//PMS_PRINT("\n\nnumberElementd_UniqueExtension:%d",numberElementd_UniqueExtension);

	CHECK(cudaStatus=cudaFree(temp));
	if(cudaStatus!=cudaSuccess)
	{
		goto Error;
	}

Error:	
	return cudaStatus;	
}
//inuse
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
//Lấy global vertex id chia cho tổng số đỉnh của đồ thị (maxOfVer). 
//Ở đây các đồ thị luôn có số lượng đỉnh bằng nhau (maxOfVer)
__global__ void kernelGetLastElementEXT(EXT *inputArray,int noEleInputArray,int *value,unsigned int maxOfVer)
{
	*value = inputArray[noEleInputArray-1].vgi/maxOfVer; 
}

cudaError_t getLastElementEXT(EXT *inputArray,int numberElementOfInputArray,int &outputValue,unsigned int maxOfVer)
{
	cudaError_t cudaStatus=cudaSuccess;

	int *temp=nullptr;
	CUCHECK(cudaMalloc((int**)&temp,sizeof(int)));

	/* Lấy graphId chứa embedding cuối cùng */
	kernelGetLastElementEXT<<<1,1>>>(inputArray,numberElementOfInputArray,temp,maxOfVer);
	CUCHECK(cudaDeviceSynchronize());
	CUCHECK(cudaStatus= cudaGetLastError());

	CUCHECK(cudaMemcpy(&outputValue,temp,sizeof(int),cudaMemcpyDeviceToHost));
	//PMS_PRINT("\n\nnumberElementd_UniqueExtension:%d",numberElementd_UniqueExtension);

	if(temp!=nullptr) cudaFree(temp);

	return cudaStatus;
}

//inuse
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

__global__ void kernelGetGraphIdContainEmbeddingBW(int vj,EXT *d_ValidExtension,int noElem_d_ValidExtension,int *dV,unsigned int maxOfVer){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i<noElem_d_ValidExtension){
		if(	d_ValidExtension[i].vj == vj){
			int graphid = (d_ValidExtension[i].vgi/maxOfVer);
			dV[graphid]=1;
		}
	}
}

//Get Graph IDs from hLevelEXT at current objLevel of unique edge.
int PMS::WriteResult(UniEdge &edge, EXTk &ext,int &supportvalue)
{
	int status =0;
	int backward =0;

	edge.print();
	int *hArrGraphId = nullptr;
	int noElemhArrGraphId=0;
	
	//if(edge.vi > edge.vj)
	//{
	//	//if edge is a backward.
	//	FUNCHECK(status=getGraphIdContainEmbeddingBW(edge,hArrGraphId,noElemhArrGraphId,ext.dArrExt,ext.noElem));
	//	if(status!=0){goto Error;}
	//}
	//else
	//{
	//	//if edge is a forward
	//	FUNCHECK(status=getGraphIdContainEmbeddingFW(edge,hArrGraphId,noElemhArrGraphId,ext.dArrExt,ext.noElem));
	//	if(status!=0){goto Error;}
	//}
	FUNCHECK(status=getGraphId(edge,hArrGraphId,noElemhArrGraphId,ext.dArrExt,ext.noElem));
	if(status!=0){goto Error;}
	
	report(hArrGraphId,noElemhArrGraphId,supportvalue);
	free(hArrGraphId);

Error:
	return status;
}
int PMS::getGraphIdContainEmbeddingv2(UniEdge edge,int *&hArrGraphId,int &noElemhArrGraphId,EXT *dArrEXT,int noElemdArrEXT)
{
	int status =0;
	int backward =0;
	if(edge.vi > edge.vj)
	{
		backward =1;
	}

	if(backward==1){
		FUNCHECK(status=getGraphIdContainEmbeddingBW(edge,hArrGraphId,noElemhArrGraphId,dArrEXT,noElemdArrEXT));
		if(status!=0)
		{
			goto Error;
		}
	}
	else
	{
		FUNCHECK(status=getGraphIdContainEmbeddingFW(edge,hArrGraphId,noElemhArrGraphId,dArrEXT,noElemdArrEXT));
		if(status!=0)
		{
			goto Error;
		}
	}

Error:
	return status;
}
void PMS::get_graphid(UniEdge &edge,int *&hArrGraphId,int &noElemhArrGraphId,EXT *dArrEXT,int noElemdArrEXT)
{
	dim3 block(blocksize);
	dim3 grid((noElemdArrEXT+block.x-1)/block.x);

	int *d_graphid=nullptr; //1. need cudaFree
	int noElem_d_graphid=0;
	//How many graphs contains embeddings of DFS_CODE?
	CUCHECK(getLastElementEXT(dArrEXT,noElemdArrEXT,noElem_d_graphid,maxOfVer));
	++noElem_d_graphid;

	CUCHECK(cudaMalloc((void**)&d_graphid,noElem_d_graphid*sizeof(int)));
	CUCHECK(cudaMemset(d_graphid,0,noElem_d_graphid*sizeof(int)));

	kernelGetGraphIdContainEmbeddingv2<<<grid,block>>>(edge.vi,edge.vj,edge.li,edge.lij,edge.lj,dArrEXT,noElemdArrEXT,d_graphid,maxOfVer);
	CUCHECK(cudaDeviceSynchronize());
	CUCHECK(cudaGetLastError());

	int *dVScanResult=nullptr; //2. need cudaFree
	CUCHECK(cudaMalloc((void**)&dVScanResult,noElem_d_graphid*sizeof(int)));
	CHECK(cudaMemset(dVScanResult,0,noElem_d_graphid*sizeof(int)));

	//scanV(dV,noElemdV,dVScanResult);
	CUCHECK(myScanV(d_graphid,noElem_d_graphid,dVScanResult));

	//PMS_PRINT("\n ************* dVScanResult *************\n");
	//FCHECK(displayDeviceArr(dVScanResult,noElem_d_graphid));

	int noElem_kq=0;
	//CUCHECK(getLastElement(dVScanResult,noElem_d_graphid,noElem_kq));
	getSizeBaseOnScanResultv2(d_graphid,dVScanResult,noElem_d_graphid,noElem_kq);
	//++noElem_kq;

	int *d_kq; //3. need cudaFree
	CUCHECK(cudaMalloc((void**)&d_kq,sizeof(int)*noElem_kq));

	dim3 blocka(blocksize);
	dim3 grida((noElem_d_graphid + blocka.x -1)/blocka.x);

	kernelGetGraph<<<grida,blocka>>>(d_graphid,noElem_d_graphid,d_kq,dVScanResult);
	CUCHECK(cudaDeviceSynchronize());
	CUCHECK(cudaGetLastError());

	/*PMS_PRINT("\n*********** d_kq ***********\n");
	FUNCHECK(status=displayDeviceArr(d_kq,noElem_kq));
	if(status!=0)
	{
		goto Error;
	}*/


	hArrGraphId=(int*)malloc(sizeof(int)*noElem_kq);
	if(hArrGraphId==nullptr) {FCHECK(-1);}
	noElemhArrGraphId=noElem_kq;

	CUCHECK(cudaMemcpy(hArrGraphId,d_kq,sizeof(int)*noElem_kq,cudaMemcpyDeviceToHost));
	//free memory
	CUCHECK(cudaFree(d_kq)); //3.
	CUCHECK(cudaFree(d_graphid)); //1.
	CUCHECK(cudaFree(dVScanResult));//2.
}

int PMS::getGraphIdContainEmbeddingBW(UniEdge edge,int *&hArrGraphId,int &noElemhArrGraphId,EXT *dArrEXT,int noElemdArrEXT)
{
	int status =0;
	cudaError_t cudaStatus;
	//int li,lij,lj;
	//li = edge.li;
	//lij = edge.lij;
	//lj = edge.lj;
	int vj;
	vj=edge.vj;
	dim3 block(blocksize);
	dim3 grid((noElemdArrEXT+block.x-1)/block.x);

	int *dV=nullptr; //1. need cudaFree
	int noElemdV=0;

	//displayArrExtension(hValidExtension.at(0).dExtension,hValidExtension.at(0).noElem);

	CHECK(cudaStatus =getLastElementEXT(dArrEXT,noElemdArrEXT,noElemdV,maxOfVer));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	++noElemdV;

	CHECK(cudaStatus=cudaMalloc((void**)&dV,noElemdV*sizeof(int)));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(dV,0,noElemdV*sizeof(int)));
		if(cudaStatus!=cudaSuccess)
		{
			status=-1;
			goto Error;
		}
	}

	kernelGetGraphIdContainEmbeddingBW<<<grid,block>>>(vj,dArrEXT,noElemdArrEXT,dV,maxOfVer);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}

	int *dVScanResult=nullptr; //2. need cudaFree
	CHECK(cudaStatus=cudaMalloc((void**)&dVScanResult,noElemdV*sizeof(int)));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus = cudaMemset(dVScanResult,0,noElemdV*sizeof(int)));
		if(cudaStatus!=cudaSuccess)
		{
			status=-1;
			goto Error;
		}
	}


	//scanV(dV,noElemdV,dVScanResult);
	CHECK(cudaStatus=myScanV(dV,noElemdV,dVScanResult));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	/*PMS_PRINT("\n ************* dVScanResult *************\n");
	FUNCHECK(status=displayDeviceArr(dVScanResult,noElemdV));
	if(status!=0)
	{
		goto Error;
	}*/

	int noElem_kq;	
	CHECK(cudaStatus=getLastElement(dVScanResult,noElemdV,noElem_kq));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	++noElem_kq;

	int *d_kq; //3. need cudaFree
	CHECK(cudaStatus=cudaMalloc((void**)&d_kq,sizeof(int)*noElem_kq));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	dim3 blocka(blocksize);
	dim3 grida((noElemdV + blocka.x -1)/blocka.x);

	kernelGetGraph<<<grida,blocka>>>(dV,noElemdV,d_kq,dVScanResult);
	cudaDeviceSynchronize();
	CHECK(cudaStatus=cudaGetLastError());
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	/*PMS_PRINT("\n*********** d_kq ***********\n");
	FUNCHECK(status=displayDeviceArr(d_kq,noElem_kq));
	if(status!=0)
	{
		goto Error;
	}*/


	hArrGraphId=(int*)malloc(sizeof(int)*noElem_kq);
	if(hArrGraphId==NULL)
	{
		PMS_PRINT("\nMalloc hArrGraphId in getGraphIdContainEmbedding() failed");
		exit(1);
	}
	noElemhArrGraphId=noElem_kq;

	CHECK(cudaStatus=cudaMemcpy(hArrGraphId,d_kq,sizeof(int)*noElem_kq,cudaMemcpyDeviceToHost));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	//free memory
	CHECK(cudaStatus=cudaFree(d_kq)); //3.
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus=cudaFree(dV)); //1.
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus=cudaFree(dVScanResult));//2.
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
Error:
	return status;
}


int PMS::getGraphIdContainEmbeddingFW(UniEdge edge,int *&hArrGraphId,int &noElemhArrGraphId,EXT *dArrEXT,int noElemdArrEXT)
{
	int status =0;
	cudaError_t cudaStatus;
	int vi,vj,li,lij,lj;
	vi = edge.vi;
	vj = edge.vj;
	li = edge.li;
	lij = edge.lij;
	lj = edge.lj;

	dim3 block(blocksize);
	dim3 grid((noElemdArrEXT+block.x-1)/block.x);

	int *dV=nullptr; //1. need cudaFree
	int noElemdV=0;

	//displayArrExtension(hValidExtension.at(0).dExtension,hValidExtension.at(0).noElem);

	CHECK(cudaStatus =getLastElementEXT(dArrEXT,noElemdArrEXT,noElemdV,maxOfVer));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	++noElemdV; //Số lượng đồ thị chứa graph_ID của mở rộng đang xét

	CHECK(cudaStatus=cudaMalloc((void**)&dV,noElemdV*sizeof(int)));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(dV,0,noElemdV*sizeof(int)));
		if(cudaStatus!=cudaSuccess)
		{
			status=-1;
			goto Error;
		}
	}

	kernelGetGraphIdContainEmbeddingv2<<<grid,block>>>(vi,vj,li,lij,lj,dArrEXT,noElemdArrEXT,dV,maxOfVer);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}

	int *dVScanResult=nullptr; //2. need cudaFree
	CHECK(cudaStatus=cudaMalloc((void**)&dVScanResult,noElemdV*sizeof(int)));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus = cudaMemset(dVScanResult,0,noElemdV*sizeof(int)));
		if(cudaStatus!=cudaSuccess){
			status=-1;
			goto Error;
		}
	}


	//scanV(dV,noElemdV,dVScanResult);
	CHECK(cudaStatus=myScanV(dV,noElemdV,dVScanResult));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	/*PMS_PRINT("\n ************* dVScanResult *************\n");
	FUNCHECK(status=displayDeviceArr(dVScanResult,noElemdV));
	if(status!=0)
	{
		goto Error;
	}*/

	int noElem_kq;	
	//CHECK(cudaStatus=getLastElement(dVScanResult,noElemdV,noElem_kq));
	CHECK(cudaStatus=getSizeBaseOnScanResult(dV,dVScanResult,noElemdV,noElem_kq));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	//++noElem_kq;

	int *d_kq; //3. need cudaFree
	CHECK(cudaStatus=cudaMalloc((void**)&d_kq,sizeof(int)*noElem_kq));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	dim3 blocka(blocksize);
	dim3 grida((noElemdV + blocka.x -1)/blocka.x);

	kernelGetGraph<<<grida,blocka>>>(dV,noElemdV,d_kq,dVScanResult);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus=cudaGetLastError());
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	/*PMS_PRINT("\n*********** d_kq ***********\n");
	FUNCHECK(status=displayDeviceArr(d_kq,noElem_kq));
	if(status!=0)
	{
		goto Error;
	}*/


	hArrGraphId=(int*)malloc(sizeof(int)*noElem_kq);
	if(hArrGraphId==NULL)
	{
		PMS_PRINT("\nMalloc hArrGraphId in getGraphIdContainEmbedding() failed");
		exit(1);
	}
	noElemhArrGraphId=noElem_kq;

	CHECK(cudaStatus=cudaMemcpy(hArrGraphId,d_kq,sizeof(int)*noElem_kq,cudaMemcpyDeviceToHost));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	//free memory
	CHECK(cudaStatus=cudaFree(d_kq)); //3.
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus=cudaFree(dV)); //1.
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus=cudaFree(dVScanResult));//2.
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
Error:
	return status;
}

int PMS::getGraphId(UniEdge &edge,int *&hArrGraphId,int &noElemhArrGraphId,EXT *&dArrEXT,int &noElemdArrEXT)
{
	int status =0;
	cudaError_t cudaStatus;
	int vi,vj,li,lij,lj;
	vi = edge.vi;
	vj = edge.vj;
	li = edge.li;
	lij = edge.lij;
	lj = edge.lj;

	dim3 block(blocksize);
	dim3 grid((noElemdArrEXT+block.x-1)/block.x);

	int *dV=nullptr; //1. need cudaFree
	int noElemdV=0;

	//displayArrExtension(hValidExtension.at(0).dExtension,hValidExtension.at(0).noElem);

	CHECK(cudaStatus =getLastElementEXT(dArrEXT,noElemdArrEXT,noElemdV,maxOfVer));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	++noElemdV; //Số lượng đồ thị chứa graph_ID của mở rộng đang xét

	CHECK(cudaStatus=cudaMalloc((void**)&dV,noElemdV*sizeof(int)));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(dV,0,noElemdV*sizeof(int)));
		if(cudaStatus!=cudaSuccess)
		{
			status=-1;
			goto Error;
		}
	}

	kernelGetGraphIdContainEmbeddingv2<<<grid,block>>>(vi,vj,li,lij,lj,dArrEXT,noElemdArrEXT,dV,maxOfVer);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}

	int *dVScanResult=nullptr; //2. need cudaFree
	CHECK(cudaStatus=cudaMalloc((void**)&dVScanResult,noElemdV*sizeof(int)));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus = cudaMemset(dVScanResult,0,noElemdV*sizeof(int)));
		if(cudaStatus!=cudaSuccess){
			status=-1;
			goto Error;
		}
	}


	//scanV(dV,noElemdV,dVScanResult);
	CHECK(cudaStatus=myScanV(dV,noElemdV,dVScanResult));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	/*PMS_PRINT("\n ************* dVScanResult *************\n");
	FUNCHECK(status=displayDeviceArr(dVScanResult,noElemdV));
	if(status!=0)
	{
		goto Error;
	}*/

	int noElem_kq;	
	//CHECK(cudaStatus=getLastElement(dVScanResult,noElemdV,noElem_kq));
	CHECK(cudaStatus=getSizeBaseOnScanResult(dV,dVScanResult,noElemdV,noElem_kq));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	//++noElem_kq;

	int *d_kq; //3. need cudaFree
	CHECK(cudaStatus=cudaMalloc((void**)&d_kq,sizeof(int)*noElem_kq));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	dim3 blocka(blocksize);
	dim3 grida((noElemdV + blocka.x -1)/blocka.x);

	kernelGetGraph<<<grida,blocka>>>(dV,noElemdV,d_kq,dVScanResult);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus=cudaGetLastError());
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	/*PMS_PRINT("\n*********** d_kq ***********\n");
	FUNCHECK(status=displayDeviceArr(d_kq,noElem_kq));
	if(status!=0)
	{
		goto Error;
	}*/


	hArrGraphId=(int*)malloc(sizeof(int)*noElem_kq);
	if(hArrGraphId==NULL)
	{
		PMS_PRINT("\nMalloc hArrGraphId in getGraphIdContainEmbedding() failed");
		exit(1);
	}
	noElemhArrGraphId=noElem_kq;

	CHECK(cudaStatus=cudaMemcpy(hArrGraphId,d_kq,sizeof(int)*noElem_kq,cudaMemcpyDeviceToHost));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	//free memory
	CHECK(cudaStatus=cudaFree(d_kq)); //3.
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus=cudaFree(dV)); //1.
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus=cudaFree(dVScanResult));//2.
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
Error:
	return status;
}


int PMS::getGraphIdContainEmbedding(UniEdge edge,int *&hArrGraphId,int &noElemhArrGraphId){
	int status =0;
	cudaError_t cudaStatus;
	int noElemdValidExtension = hLevelEXT.at(0).vE.at(0).noElem;

	int li,lij,lj;
	li = edge.li;
	lij = edge.lij;
	lj = edge.lj;
	dim3 block(blocksize);
	dim3 grid((noElemdValidExtension+block.x-1)/block.x);

	int *dV=nullptr;
	int noElemdV=0;

	//displayArrExtension(hValidExtension.at(0).dExtension,hValidExtension.at(0).noElem);

	CHECK(getLastElementExtension(hValidExtension.at(0).dExtension,hValidExtension.at(0).noElem,noElemdV,maxOfVer));
	noElemdV++;

	CHECK(cudaStatus=cudaMalloc((void**)&dV,noElemdV*sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		fprintf(stderr,"\n cudaMalloc dV in getGraphIdContainEmbedding() failed");
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(dV,0,noElemdV*sizeof(int)));
		if(cudaStatus!=cudaSuccess){
			status=-1;
			goto Error;
		}
	}

	kernelGetGraphIdContainEmbedding<<<grid,block>>>(li,lij,lj,hValidExtension.at(0).dExtension,hValidExtension.at(0).noElem,dV,maxOfVer);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		status =-1;
		fprintf(stderr,"\n kernelGetGraphIdContainEmbedding() in getGraphIdContainEmbedding() failed",cudaStatus);
		goto Error;
	}

	int *dVScanResult=nullptr;
	CHECK(cudaStatus=cudaMalloc((void**)&dVScanResult,noElemdV*sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMalloc dVScanResult in getGraphIdContainEmbedding() failed");
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(dVScanResult,0,noElemdV*sizeof(int)));
		if(cudaStatus!=cudaSuccess){
			status=-1;
			goto Error;
		}
	}


	//scanV(dV,noElemdV,dVScanResult);
	CHECK(cudaStatus=myScanV(dV,noElemdV,dVScanResult));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	/*PMS_PRINT("\n ************* dVScanResult *************\n");
	displayDeviceArr(dVScanResult,noElemdV);*/
	int noElem_kq;	
	CHECK(cudaStatus=getLastElement(dVScanResult,noElemdV,noElem_kq));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	++noElem_kq;

	int *d_kq;
	CHECK(cudaStatus=cudaMalloc((void**)&d_kq,sizeof(int)*noElem_kq));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	dim3 blocka(blocksize);
	dim3 grida((noElemdV + blocka.x -1)/blocka.x);

	kernelGetGraph<<<grida,blocka>>>(dV,noElemdV,d_kq,dVScanResult);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	PMS_PRINT("\n*********** d_kq ***********\n");
	FUNCHECK(status=displayDeviceArr(d_kq,noElem_kq));
	if(status!=0){
		goto Error;
	}


	hArrGraphId=(int*)malloc(sizeof(int)*noElem_kq);
	if(hArrGraphId==NULL){
		PMS_PRINT("\nMalloc hArrGraphId in getGraphIdContainEmbedding() failed");
		exit(1);
	}
	noElemhArrGraphId=noElem_kq;

	CHECK(cudaStatus=cudaMemcpy(hArrGraphId,d_kq,sizeof(int)*noElem_kq,cudaMemcpyDeviceToHost));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus=cudaFree(d_kq));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus=cudaFree(dV));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus=cudaFree(dVScanResult));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
Error:
	return status;
}
//Trích các GraphID chứa các embeddings của DFS_CODE thoả minDFS_CODE.
int PMS::getGraphIdContainEmbedding_pure(UniEdge edge,int *&hArrGraphId,int &noElemhArrGraphId)
{
	int status =0;
	cudaError_t cudaStatus;
	int noElemdValidExtension = hLevelEXT.at(objLevel.Level).vE.at(0).noElem;

	int li,lij,lj;
	li = edge.li;
	lij = edge.lij;
	lj = edge.lj;
	dim3 block(blocksize);
	dim3 grid((noElemdValidExtension+block.x-1)/block.x);

	int *dV=nullptr;
	int noElemdV=0;

	/*PMS_PRINT("\n************hLevelEXT.at(0).vE.at(0).dArrExt***********\n");
	FUNCHECK(status=displaydArrEXT(hLevelEXT.at(objLevel.Level).vE.at(0).dArrExt,hLevelEXT.at(objLevel.Level).vE.at(0).noElem));
	if(status!=0){
		goto Error;
	}*/
	//vì các phần tử trong dArrExt là có thứ tự từ đồ thị có ID bé --> ID lớn.
	CHECK(cudaStatus=getLastElementExtension_pure ( \
		hLevelEXT.at(objLevel.Level).vE.at(0).dArrExt, \
		hLevelEXT.at(objLevel.Level).vE.at(0).noElem, \
		noElemdV, \
		maxOfVer));
	if(cudaStatus!=cudaSuccess)
	{
		goto Error;
	}
	//Mỗi phần tử của dV được ví như là một graph_id. Ví dụ, dV[0] thì 0 vừa là index vừa là graph_id luôn.
	++noElemdV; 

	CHECK(cudaStatus=cudaMalloc((void**)&dV,noElemdV*sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		fprintf(stderr,"\n cudaMalloc dV in getGraphIdContainEmbedding() failed");
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(dV,0,noElemdV*sizeof(int)));
		if(cudaStatus!=cudaSuccess){
			status=-1;
			goto Error;
		}
	}
	kernelGetGraphIdContainEmbedding_pure<<<grid,block>>>(li,lij,lj,hLevelEXT.at(objLevel.Level).vE.at(0).dArrExt,hLevelEXT.at(objLevel.Level).vE.at(0).noElem,dV,maxOfVer);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		fprintf(stderr,"\n kernelGetGraphIdContainEmbedding() in getGraphIdContainEmbedding() failed",cudaStatus);
		goto Error;
	}
	//PMS_PRINT("\n*********** dV ***********\n"); //dV[i] =1 có nghĩa là graph_id thứ i là đồ thị cần tìm. Loại bỏ những dV[i]=0.
	//FUNCHECK(status=displayDeviceArr(dV,noElemdV));
	//if(status!=0)
	//{
	//	goto Error;
	//}
	int *dVScanResult=nullptr;
	CHECK(cudaStatus=cudaMalloc((void**)&dVScanResult,noElemdV*sizeof(int)));
	if(cudaStatus!=cudaSuccess)
	{
		fprintf(stderr,"\n cudaMalloc dVScanResult in getGraphIdContainEmbedding() failed");
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(dVScanResult,0,noElemdV*sizeof(int)));
		if(cudaStatus!=cudaSuccess)
		{
			status=-1;
			goto Error;
		}
	}


	CHECK(cudaStatus=myScanV(dV,noElemdV,dVScanResult));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	/*PMS_PRINT("\n ************* dVScanResult *************\n");
	FUNCHECK(status=displayDeviceArr(dVScanResult,noElemdV));
	if(status!=0)
	{
		goto Error;
	}*/

	int noElem_kq;	
	//CHECK(cudaStatus=getLastElement(dVScanResult,noElemdV,noElem_kq));
	CHECK(cudaStatus = getSizeBaseOnScanResult(dV,dVScanResult,noElemdV,noElem_kq));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	//++noElem_kq;

	int *d_kq;
	CHECK(cudaStatus=cudaMalloc((void**)&d_kq,sizeof(int)*noElem_kq));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	dim3 blocka(blocksize);
	dim3 grida((noElemdV + blocka.x -1)/blocka.x);

	kernelGetGraph<<<grida,blocka>>>(dV,noElemdV,d_kq,dVScanResult);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	/*PMS_PRINT("\n*********** d_kq ***********\n");
	FUNCHECK(status=displayDeviceArr(d_kq,noElem_kq));
	if(status!=0)
	{
		goto Error;
	}*/


	hArrGraphId=(int*)malloc(sizeof(int)*noElem_kq);
	if(hArrGraphId==NULL)
	{
		PMS_PRINT("\nMalloc hArrGraphId in getGraphIdContainEmbedding() failed");
		exit(1);
	}
	noElemhArrGraphId=noElem_kq;

	CHECK(cudaStatus=cudaMemcpy(hArrGraphId,d_kq,sizeof(int)*noElem_kq,cudaMemcpyDeviceToHost));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus=cudaFree(d_kq));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus=cudaFree(dV));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus=cudaFree(dVScanResult));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
Error:
	return status;
}



//Allocate Device Memory: Hàm ADM dùng để cấp phát bộ nhớ nBytes trên device. Kết quả được trỏ đến bởi devicePointer.
cudaError_t ADM(int *&devicePointer,size_t nBytes){
	cudaError_t cudaStatus;
	cudaStatus= cudaMalloc((void**)&devicePointer,nBytes);
	return cudaStatus;
}

//Input: Các unique edge satisfy minsup ban đầu.
//Need to do: 
//1. Xây dựng right most path ban đầu (Ở đây: RMP là giống nhau cho tất cả các UniEdge Satisfy minSup ban đầu)
//-> 2.Duyệt qua từng UniEdge Satisfy minsup -> 3.Xây dựng DFS_CODE ban đầu 
//-> 3.1 Kiểm tra is_min() (nếu thoả) -> 3.2. Ghi nhận kết quả ->3.3. Xây dựng Embedding column ban đầu 
//-> 3.4. Gọi hàm FSMining() -> 3.6 Gỡ bỏ Embedding Column ->3.7. Gỡ bỏ DFS_CODE ban đầu
//->1.1. Gỡ bỏ right most path.

//int PMS::Mining(){
//	int status = 0;
//	cudaError_t cudaStatus;
//
//	FUNCHECK(status=buildRMP()); //Xây dựng RMP ban đầu
//	if(status!=0){
//		goto Error;
//	}
//	#pragma region "build RMP on device"
//	//Xây dựng right most path từ vector<int> hRMP
//	int noElemVerOnRMP = hRMP.at(0).noElem; //right most path chứa bao nhiêu đỉnh
//	int *rmp = nullptr; //rigt most path trên bộ nhớ device
//	CHECK(cudaStatus = cudaMalloc((void**)&rmp,noElemVerOnRMP*sizeof(int))); //cấp phát bộ nhớ trên device cho rmp
//	if(cudaStatus!=cudaSuccess){
//		status =-1;
//		goto Error;
//	}
//	int *tempRMP=(int*)malloc(sizeof(int)*noElemVerOnRMP); //dùng để chứa dữ liệu từ vector hRMP
//	if(tempRMP==NULL){
//		status =-1;
//		goto Error;
//	}
//	//chép dữ liệu từ hRMP sang bộ nhớ temp
//	for (int i = 0; i < noElemVerOnRMP; i++)
//	{
//		tempRMP[i] = hRMP.at(0).hArrRMP.at(i);
//	}
//	//Chép dữ liệu từ temp trên host sang rmp trên device. //ở bước này không cần phải làm phức tạp như thế. Chỉ cần khởi tạo bộ nhớ trên Device và gán cho nó giá trị {1,0} cho nó là được
//	CHECK(cudaStatus =cudaMemcpy(rmp,tempRMP,sizeof(int)*noElemVerOnRMP,cudaMemcpyHostToDevice)); //chép dữ liệu từ temp ở host sang rmp trên device
//	if(cudaStatus!=cudaSuccess){
//		status =-1;
//		goto Error;
//	}
//
//	std::free(tempRMP);
//
//	PMS_PRINT("\n\n ******* rmp *********\n");
//	FUNCHECK(status=displayDeviceArr(rmp,noElemVerOnRMP));
//	if(status!=0){
//		goto Error;
//	}
//#pragma endregion
//	//Nên kiểm tra cái này ở đầu hàm Mining(), vì không có mở rộng thoả minsup nào thì chúng ta return về status liền. Không phải mất công làm các bước trên.
//	if(hUniEdgeSatisfyMinsup.at(0).noElem<=0){
//		PMS_PRINT("\n There no any edge in hUniEdgeSatisfyMinsup\n");
//		return status;
//	}
//
//	int noElemtemp = hUniEdgeSatisfyMinsup.at(0).noElem;
//	if (noElemtemp==0){
//		PMS_PRINT("\n No any extension that satisfy minsup. \n Mining has been stopped\n");
//		goto Error;
//	}
//	//Cấp phát một mảng tạm ở host để chép dữ liệu từ device sang host lưu giữ các cạnh duy nhất thoả minsup
//	UniEdge *temp=(UniEdge*)malloc(sizeof(UniEdge)*noElemtemp);
//	if(temp==NULL){
//		PMS_PRINT("\n malloc temp failed");
//		status =-1;
//		goto Error;
//	}
//	//chép dữ liệu từ device sang host
//	CHECK(cudaStatus=cudaMemcpy(temp,hUniEdgeSatisfyMinsup.at(0).dUniEdge,noElemtemp*sizeof(UniEdge),cudaMemcpyDeviceToHost));
//	if(cudaStatus!=cudaSuccess){
//		status=-1;
//		goto Error;
//	}
//
//	for (int i = 0; i < noElemtemp; i++) //Duyệt qua các UniEdge thoả minSup để kiểm tra minDFS_CODE, nếu thoả thì ghi kết quả vào result và xây dựng embedding
//	{
//		int li,lij,lj;
//		li = temp[i].li;
//		lij= temp[i].lij;
//		lj=temp[i].lj;
//
//		DFS_CODE.push(0,1,temp[i].li,temp[i].lij,temp[i].lj);//xây dựng DFS_CODE
//		minLabel = temp[i].li;
//		maxId = 1;
//
//		if(is_min()){ //Nếu DFS_CODE là min thì tìm các graphid chứa embedding của DFS_CODE
//			PMS_PRINT("\n This is minDFSCODE\n");
//
//			int *hArrGraphId; //Mảng chứa các graphID có embedding của DFS_Code.
//			int noElemhArrGraphId=0;
//			/* Trước khi ghi kết quả thì phải biết đồ thị phổ biến đó tồn tại ở những graphId nào. Hàm getGraphIdContainEmbedding dùng để làm việc này
//			* 3 tham số đầu tiên của hàm là nhãn cạnh của phần tử d_UniqueExtension đang xét */
//			FUNCHECK(status =getGraphIdContainEmbedding(temp[i],hArrGraphId,noElemhArrGraphId));
//			if (status!=0){
//				PMS_PRINT("\n\n getGraphIdContainEmbedding() in Mining() failed");
//				goto Error;
//			}
//
//			//In nội dung mảng hArrGraphId
//
//			PMS_PRINT("\n ************** hArrGraphId ****************\n");
//			for (int j = 0; j < noElemhArrGraphId; j++)
//			{
//				PMS_PRINT("%d ",hArrGraphId[j]);
//			}
//
//			/*	Ghi kết quả DFS_CODE vào file result.txt ************************************************************
//			*	Hàm report sẽ chuyển DFS_CODE pattern sang dạng đồ thị, sau đó sẽ ghi đồ thị đó xuống file result.txt
//			*	Hàm report gồm 3 tham số:
//			*	Tham số thứ 1: mảng chứa danh sách các graphID chứa DFS_CODE pattern
//			*	Tham số thứ 2: số lượng mảng
//			*	Tham số thứ 3: độ hỗ trợ của DFS_CODE pattern *******************************************************/
//
//			report(hArrGraphId,noElemhArrGraphId,hUniEdgeSatisfyMinsup.at(0).hArrSup[i]);
//			//Giải phóng bộ nhớ 
//			std::free(hArrGraphId);
//
//			//Xây dựng Embedding cho DFS_Code rồi gọi hàm GraphMining để khai thác
//			//Trong GraphMining sẽ gọi GraphMining khác để thực hiện khai thác đệ quy
//
//			FUNCHECK(status=buildFirstEmbedding(temp[i])); //Xây dựng 2 cột embedding ban đầu.
//			if(status!=0){
//				goto Error;
//			}
//			FUNCHECK(status=FSMining(rmp,noElemVerOnRMP)); //Gọi FSMining.( Hàm này thực hiện theo tuần tự (1. Find Extension -> 2. Extract UniEdge -> 3.Compute & CHECK Support -> 4. CHECK minDFS_CODE -> 5. BuildEmbedding -> 6.Find RMP -> 1.)
//			if(status!=0){
//				goto Error;
//			}
//			if(hEmbedding.size()!=0){ //3. giải phóng embedding
//				for (int j = 0; j < hEmbedding.size(); j++)
//				{
//					CHECK(cudaStatus=cudaFree(hEmbedding.at(j).dArrEmbedding));
//					if(cudaStatus!=cudaSuccess){
//						status=-1;
//						goto Error;
//					}
//
//				}
//				hEmbedding.clear();
//			}
//			DFS_CODE.pop(); //2. gỡ bỏ DFS_CODE
//		} //kết thúc is_min()
//	} //kết thúc for: việc duyệt qua tất cả các satisfied minsup unique edge	
//	//Gỡ bỏ right most path ban đầu. Ở đây Right most path cho trường hợp này là không thay đổi rmp luôn = {1,0}. Do đó chúng ta không cần phải xoá rmp chi cho mất công phải xây dựng lại để khai thác cho các cạnh khác.
//	if(hRMP.size()>0){ 
//		for (int j = 0; j < hRMP.size(); j++)
//		{
//			hRMP.at(j).hArrRMP.clear();
//		}
//		hRMP.clear();
//		CHECK(cudaStatus=cudaFree(rmp));
//		if(cudaStatus!=cudaSuccess){
//			status=-1;
//			goto Error;
//		}
//	}
//
//	std::free(temp);
//Error:
//	return status;
//}






int PMS::Mining()
{
	int status = 0;
	cudaError_t cudaStatus;

	//1. Tăng Level
	//cout<<"\t-----------------------------------"<<endl;
	//cout<<"\t\t Increase Level"<<endl;
	increaseLevel();
	//std::printf("Level: %d\n",objLevel.Level);
	//cout<<"\t\t Tim mo rong hop le tu cac RMP cua embeddings"<<endl;
	//Tìm các mở rộng hợp lệ từ RMP của các embeddings.
	//2. Duyệt qua các embedding columns thuộc RMP để tìm các mở rộng hợp lệ
	//2.1. Chuẩn bị bộ nhớ lưu trữ các mở rộng hợp lệ
	int noElemEXTk = hLevelRMP.at(objLevel.prevLevel).noElem;
	//Chuẩn bị các bộ nhớ hLevelExt, hLevelUniEdgeSatisfyMinsup, hLevelListVerRMP để lưu dữ liệu khai thác ở new level.
	//Lưu trữ các mở rộng hợp lệ của từng đỉnh thuộc RMP của embeddings
	hLevelEXT.at(objLevel.Level).noElem= noElemEXTk; 
	hLevelEXT.at(objLevel.Level).vE.resize(noElemEXTk);
	
	//2.2. Chuẩn bị bộ nhớ Lưu trữ các mở rộng thoả minsup
	hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).noElem = noElemEXTk;  //nên xoá noElem vì không cần
	hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.resize(noElemEXTk);

	//Lưu trữ danh sách các đỉnh thuộc RMP của DFS_CODE ở từng level
	//hLevelListVerRMP.resize(objLevel.size); //Đã được gọi trong hàm increaseLevel()

	int *tempListVerCol = nullptr; //chứa danh sách các đỉnh cần mở rộng thuộc một embedding column cụ thể.
	CHECK(cudaStatus = cudaMalloc((void**)&tempListVerCol,hLevelPtrEmbedding.at(objLevel.prevLevel).noElemEmbedding * sizeof(int)));
	if(cudaStatus!=cudaSuccess)
	{
		status = -1;
		goto Error;
	}
	//lưu trữ các global vertex id thuộc RMP của các Embeddings, có kích thước nhỏ hơn  đỉnh so với RMP.
	int *dArrVidOnRMP = nullptr;
	FUNCHECK(status = findVerOnRMPForBWCheckv2( \
		hLevelPtrEmbedding.at(objLevel.prevLevel), \
		hLevelRMP.at(objLevel.prevLevel).dRMP, \
		hLevelRMP.at(objLevel.prevLevel).noElem, \
		dArrVidOnRMP));
	if(status!=0)
	{
		goto Error;
	}
	/*FUNCHECK(status=displaydArrPointerEmbedding ( \
		hLevelPtrEmbedding.at(objLevel.Level).dArrPointerEmbedding, \
		hLevelPtrEmbedding.at(objLevel.Level).noElem, \
		hLevelPtrEmbedding.at(objLevel.Level).noElemEmbedding));
	if(status!=0){
		goto Error;
	}*/
	//2.2 Tìm và trích các mở rộng thoả minsup từ các đỉnh thuộc RMP của các embeddings lưu kết quả vào hLevelUniEdgeSatisfyMinsup.

#pragma region "2.2. find and extract uniEdge that satisfy minsup to hLevelUniEdgeSatisfyMinsup"

	nthDFSCODE;
	if(hLevelRMP.at(objLevel.prevLevel).noElem>2)
	{ //Nếu số lượng đỉnh trên RMP lớn hơn 2 thì mới tồn tại backward. Vì ở đây chỉ xét đơn đồ thị vô hướng
		//dArrVidOnRMP: chứa các đỉnh thuộc RMP của mỗi Embedding. Dùng để kiểm tra sự tồn tại của đỉnh trên right most path
		//khi tìm các mở rộng backward. Chỉ dùng tới khi right most path có 3 đỉnh trở lên (chỉ xét trong đơn đồ thị vô hướng).
		//int *dArrVidOnRMP = nullptr; //7. need cudaFree //lưu trữ các đỉnh trên RMP của Embedding, có kích thước nhỏ hơn  đỉnh so với RMP
		//FUNCHECK(status = findVerOnRMPForBWCheckv2(hLevelPtrEmbeddingv2.at(idxLevel),rmp,noElemVerOnRMP,dArrVidOnRMP));
		//if(status!=0){
		//	goto Error;
		//}
		//Duyệt qua các cột của Embedding để tìm các mở rộng hợp lệ, Xét embedding column cuối trước.
		for (int i = 0; i < hLevelRMP.at(objLevel.prevLevel).noElem ; i++)
		{
			int colEmbedding = hLevelRMP.at(objLevel.prevLevel).hArrRMP.at(i);//Tìm mở rộng cho các đỉnh tại vị trí colEmbedding trong vector hEmbedding
			//cout<<endl<<"Extend from vertex: "<<colEmbedding <<" at idxLevel "<<objLevel.prevLevel;

			currentColEmbedding=colEmbedding; //Đang mở rộng từ cột nào của Embedding. Được dùng để cập nhật prevCol, phục vụ cho việc xây dựng Right Most Path
			int k = i; //lưu vào Extk với k = i; K=0 đại diện cho EXT0: là EXT cuối
			dim3 block(blocksize);
			dim3 grid((hLevelPtrEmbedding.at(objLevel.prevLevel).noElemEmbedding + block.x -1)/block.x);
			//Trích các đỉnh cần mở rộng của cột Embedding đang xét
			//Consider 1
			//trích các đỉnh từ listVer, từ vị trí i*noElemEmbedding,trích noElemEmbedding phần tử, bỏ vào tempListVerCol
			kernelExtractFromListVer<<<grid,block>>>( \
				hLevelListVerRMP.at(objLevel.Level).dListVer, \
				i*hLevelPtrEmbedding.at(objLevel.Level).noElemEmbedding, \
				hLevelPtrEmbedding.at(objLevel.Level).noElemEmbedding,\
				tempListVerCol);
			CHECK(cudaStatus=cudaDeviceSynchronize());
			if(cudaStatus!=cudaSuccess)
			{
				status=-1;
				goto Error;
			}

			CHECK(cudaStatus = cudaGetLastError());
			if(cudaStatus!=cudaSuccess)
			{
				status=-1;
				goto Error;
			}

			PMS_PRINT("\n ****** tempListVerCol ***********\n");
			FUNCHECK(status=displayDeviceArr(tempListVerCol,hLevelPtrEmbedding.at(objLevel.Level).noElemEmbedding));
			if(status!=0)
			{
				goto Error;
			}

			//Gọi hàm tìm các mở rộng hợp lệ từ đỉnh cuối i=0
			if(i==0)
			{ //khai thác forward lẫn backward. với các tham số:
				//1. Mở rộng từ danh sách đỉnh nào, 
				//2. embedding trên device,
				//3. trên EXTk nào,
				//4.colembedding: để cập nhật vi cho mở rộng mới.
				FUNCHECK(status = findValidFBExtensionv2 ( \
					tempListVerCol, \
					hLevelPtrEmbedding.at(objLevel.prevLevel), \
					k, \
					colEmbedding, \
					dArrVidOnRMP, \
					hLevelRMP.at(objLevel.prevLevel).dRMP));
				if(status!=0)
				{
					goto Error;
				}				
			}
			else
			{
				//chỉ khai thác forward, need to complete this function now.
				FUNCHECK(status = findForwardExtension( \
					tempListVerCol, \
					hLevelPtrEmbedding.at(objLevel.prevLevel), \
					k, \
					colEmbedding, \
					dArrVidOnRMP, \
					hLevelRMP.at(objLevel.prevLevel).dRMP));
				if(status!=0)
				{
					goto Error;
				}				
			}
		}
	}
	else //Ngược lại, chỉ khai thác forward khi số lượng đỉnh trên RMP chỉ bằng 2.
	{
		for (int i = 0; i < hLevelRMP.at(objLevel.prevLevel).noElem ; i++)
		{
			int colEmbedding = hLevelRMP.at(objLevel.prevLevel).hArrRMP.at(i); //Tìm mở rộng cho các đỉnh tại vị trí colEmbedding trong vector hEmbedding
			currentColEmbedding=colEmbedding; //Đang mở rộng từ cột nào của Embedding. Được dùng để cập nhật prevCol, phục vụ cho việc xây dựng Right Most Path
			int k = i; //lưu vào Extk với k = i; K=0 đại diện cho EXT0: là EXT cuối
			dim3 block(blocksize);
			dim3 grid((hLevelPtrEmbedding.at(objLevel.prevLevel).noElemEmbedding + block.x -1)/block.x);
			//Trích các đỉnh cần mở rộng của cột Embedding đang xét
			//trích các đỉnh từ listVer, từ vị trí i*noElemEmbedding,trích noElemEmbedding phần tử, bỏ vào tempListVerCol
			kernelExtractFromListVer<<<grid,block>>>( \
				hLevelListVerRMP.at(objLevel.prevLevel).dListVer, \
				i*hLevelPtrEmbedding.at(objLevel.prevLevel).noElemEmbedding, \
				hLevelPtrEmbedding.at(objLevel.prevLevel).noElemEmbedding, \
				tempListVerCol);
			CHECK(cudaStatus=cudaDeviceSynchronize());
			if(cudaStatus!=cudaSuccess)
			{
				status=-1;
				goto Error;
			}

			CHECK(cudaStatus = cudaGetLastError());
			if(cudaStatus!=cudaSuccess)
			{
				status=-1;
				goto Error;
			}

			/*PMS_PRINT("\n ****** tempListVerCol ***********\n");
			FUNCHECK(status=displayDeviceArr(tempListVerCol,hLevelPtrEmbedding.at(objLevel.prevLevel).noElemEmbedding));
			if(status!=0)
			{
				goto Error;
			}*/

			FUNCHECK(status = findForwardExtension( \
				tempListVerCol, \
				hLevelPtrEmbedding.at(objLevel.prevLevel), \
				k, \
				colEmbedding, \
				dArrVidOnRMP, \
				hLevelRMP.at(objLevel.prevLevel).dRMP));
			if(status!=0)
			{
				goto Error;
			}				
		}
	}

#pragma endregion

	//Hiển thị nội dung của hLevelUniEdgeSatisfyMinsup tại Level đang xét.
	/*for (int i = 0; i < hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).noElem; i++)
	{
		PMS_PRINT("\n ************ dArrUniEdgeStatisfy minSup ***************\n");
		FUNCHECK(status=displayArrUniEdge(hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(i).dArrUniEdge,hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(i).noElem)); 
		if(status!=0) 
		{
			goto Error;
		}
		cout<<endl<<"Support: "<<endl;

		for (int j = 0; j < hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(i).noElem; j++)
		{
			cout<<hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(i).hArrSupport[j]<<" ";
		}

	} */

	//3. Duyệt qua các mở rộng duy nhất thoả minsup
	//3.1 Xây dựng DFS_CODE cho mở rộng
	//3.2 Kiểm tra minDFS_CODE nếu là nhỏ nhất thì tiếp tục mục 3.3
	//3.3 Ghi nhận kết quả vào result.txt
	//3.4 Xây dựng Embedding cho pattern (Tuỳ vào cạnh đang xét là forward hay backward để xây dựng embedding cho đúng
	//3.5 Tìm danh sách các đỉnh của các Embedding Column thuộc Right Most Path cần mở rộng.
	//3.6 Gọi hàm Mining() để khai thác đệ quy cho Level mới.
	//3.7 Dọn dẹp bộ nhớ khi cần thiết.
#pragma region "3. Recursion of Mining for every satisfied unique edge"

	//3. Duyệt qua các mở rộng duy nhất thoả minsup
	for (int i = 0; i < hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).noElem; i++)
	{
		/*PMS_PRINT("\n ************ dArrUniEdgeStatisfy minSup ***************\n");
		FUNCHECK(status=displayArrUniEdge(hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(i).dArrUniEdge,hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(i).noElem));
		if(status!=0) 
		{
			goto Error;
		}*/
		if(hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(i).noElem<=0)
		{
			//PMS_PRINT("\n There no any edge in hUniEdgeSatisfyMinsup at Level: %d, at vecUES:  ",objLevel.Level);
			//goto Error;
			continue; //quay lên xét mở rộng
		}

		int noElemtemp = hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(i).noElem;

		//Cấp phát một mảng tạm ở host để chép dữ liệu từ device sang host lưu giữ các cạnh duy nhất thoả minsup
		UniEdge *temp=(UniEdge*)malloc(sizeof(UniEdge)*noElemtemp);
		if(temp==NULL)
		{
			PMS_PRINT("\n malloc temp failed");
			status =-1;
			goto Error;
		}
		//Chép dữ liệu từ device sang host
		CHECK(cudaStatus=cudaMemcpy(temp,hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(i).dArrUniEdge,noElemtemp*sizeof(UniEdge),cudaMemcpyDeviceToHost));
		if(cudaStatus!=cudaSuccess)
		{
			status=-1;
			goto Error;
		}
		//Duyệt qua các UniEdge thoả minSup để kiểm tra minDFS_CODE, nếu thoả thì ghi kết quả vào result và xây dựng embedding
		for (int j = 0; j < noElemtemp; j++) 
		{
			int backward=0;
			int li,lij,lj,vi,vj;
			li = temp[j].li;
			lij= temp[j].lij;
			lj=temp[j].lj;
			vi=temp[j].vi;
			vj=temp[j].vj;
			//PMS_PRINT("\n Extent from DFS code:(%d %d %d %d %d):",vi,vj,li,lij,lj);
			if(vi>vj)
			{
				backward=1;
			}

			//Nếu là mở rộng forward thì cập nhật lại maxId bằng vj;
			if(backward!=1)
			{
				DFS_CODE.push(vi,vj,-1,temp[j].lij,temp[j].lj);//xây dựng DFS_CODE forward
				maxId=vj;
			}
			else
			{
				DFS_CODE.push(vi,vj,-1,temp[j].lij,-1);//xây dựng DFS_CODE backward
			}
			currentColEmbedding=vi;
			//3.2 Kiểm tra minDFS_CODE nếu là nhỏ nhất thì tiếp tục mục 3.3
			if(is_min())
			{ 
				//PMS_PRINT("\n This is minDFSCODE\n");
				//3.3 Ghi nhận kết quả vào result.txt
				int *hArrGraphId; //2. need free //Mảng chứa các graphID có embedding của DFS_Code.
				int noElemhArrGraphId=0;
				FUNCHECK(status =getGraphIdContainEmbeddingv2(temp[j],hArrGraphId,noElemhArrGraphId,hLevelEXT.at(objLevel.Level).vE.at(i).dArrExt,hLevelEXT.at(objLevel.Level).vE.at(i).noElem));
				if (status!=0)
				{
					PMS_PRINT("\n\n getGraphIdContainEmbedding() in Mining() failed");
					goto Error;
				}
				////In nội dung mảng hArrGraphId
				/*PMS_PRINT("\n ************** hArrGraphId ****************\n");
				for (int k = 0; k < noElemhArrGraphId; k++)
				{
					PMS_PRINT("%d ",hArrGraphId[k]);
				}*/

				report(hArrGraphId,noElemhArrGraphId,hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(i).hArrSupport[j]); //ghi kết quả vào tập tin result.txt
				write_minDFS_CODE(DFS_CODE);
				free(hArrGraphId);

				//3.4 Xây dựng Embedding cho pattern (Tuỳ vào cạnh đang xét là forward hay backward để xây dựng embedding cho đúng
				if(backward!=1)
				{
					//thực hiện mở rộng forward: là hoạt động thêm một embedding column
					//FUNCHECK(status=extendEmbedding(temp[i],idxExt)); //mở rộng embedding. 
					//if(status!=0){
					//	goto Error;
					//}
					FUNCHECK(status=extendEmbeddingv2(temp[j],hLevelEXT.at(objLevel.Level).vE.at(i).dArrExt,hLevelEXT.at(objLevel.Level).vE.at(i).noElem));
					if(status!=0)
					{
						goto Error;
					}
				}
				else
				{
					//Thực hiện mở rộng backward: là hoạt động giữa là các row pointer backward của embedding column và loại bỏ các forward row pointer.
					//Thêm cột backward embedding khi đã có bwembedding
					int lastColEmbedding = hEmbedding.size()-1;
					if (hEmbedding.at(lastColEmbedding).hBackwardEmbedding.size()>0)
					{
						int idxLastBWEC = hEmbedding.at(lastColEmbedding).hBackwardEmbedding.size()-1;
						FUNCHECK(status = extendEmbeddingBW2(temp[j],hEmbedding.at(lastColEmbedding),hEmbedding.at(lastColEmbedding).hBackwardEmbedding.at(idxLastBWEC).dArrEmbedding,hLevelEXT.at(objLevel.Level).vE.at(i).dArrExt,hLevelEXT.at(objLevel.Level).vE.at(i).noElem));
					}
					else
					{//Thêm cột backward embedding mới khi chưa có embedding nào được khai thác.
						FUNCHECK(status = extendEmbeddingBW(temp[j],hEmbedding.at(lastColEmbedding),hLevelEXT.at(objLevel.Level).vE.at(i).dArrExt,hLevelEXT.at(objLevel.Level).vE.at(i).noElem));
					}

					if(status!=0)
					{
						goto Error;
					}
				}
				//3.5 Xây dựng RMP cho embedding columns ở bước trên.
				//Nếu mở rộng đang xét là backward thì right most path giống Level trước.
				//(we dont need to update right most path for backward extension).
				//Chúng ta chỉ tìm một right most path mới khi mở rộng đang xét là forward.
				//(We only need to upate the right most path for the forward extension.
				if(backward!=1)
				{ //Nếu mở rộng đang xét không phải là backward extension thì phải tìm right most path cho forward extension đó.
					FUNCHECK(status = updateRMP()); //Cập nhật cho vector hRMPv2. Nếu là backward thì vẫn đúng. Nhưng vẫn phải tạo Level mới. 
					if(status!=0)
					{
						goto Error;
					}
				}
				else
				{//Có RMP trên host lẫn trên device. ==> có cần thiết hay không?
					FUNCHECK(status = updateRMPBW()); //Cập nhật cho vector hRMPv2. Nếu là backward thì vẫn đúng. Nhưng vẫn phải tạo Level mới. 
					if(status!=0)
					{
						goto Error;
					}
				}
				//Xây dựng embedding trên device tại level hiện tại
				FUNCHECK(status=buildArrPointerEmbeddingv3());
				if(status!=0)
				{
					goto Error;
				}
				//write_embedding_column();
				if(nthDFSCODE==10)
				{
					write_embedding_column();
				}


				//3.6 Tìm danh sách các đỉnh của các Embedding Column thuộc Right Most Path cần mở rộng.
				//FUNCHECK(status = findListVer(hLevelPtrEmbeddingv2.at(idxLevel).dArrPointerEmbedding,hLevelPtrEmbeddingv2.at(idxLevel).noElemEmbedding,rmp,noElemVerOnRMP,hListVerv2));
				FUNCHECK(status = findListVer ( \
					hLevelPtrEmbedding.at(objLevel.Level).dArrPointerEmbedding, \
					hLevelPtrEmbedding.at(objLevel.Level).noElemEmbedding, \
					hLevelRMP.at(objLevel.Level).dRMP, \
					hLevelRMP.at(objLevel.Level).noElem));
				if(status !=0){
					goto Error;
				}
				//hiển thị danh sách đỉnh
				PMS_PRINT("\n\n ********* listVer *********\n");
				FUNCHECK(status= displayDeviceArr( \
					hLevelListVerRMP.at(objLevel.Level).dListVer, \
					hLevelListVerRMP.at(objLevel.Level).noElem)); //5. need cudaFree and pop_back
				if(status!=0){
					goto Error;
				}
				//3.7 Gọi hàm Mining() để khai thác đệ quy cho Level mới.
				FUNCHECK(status=Mining());
				if(status!=0)
				{
					goto Error;
				}

				//Gỡ bỏ embedding
				int lastCol = hEmbedding.size()-1;
				int bwEmbeddingSize = hEmbedding.at(lastCol).hBackwardEmbedding.size();
				if(bwEmbeddingSize!=0)
				{
					int lastBWCol= hEmbedding.at(lastCol).hBackwardEmbedding.size()-1;
					CHECK(cudaStatus=cudaFree(hEmbedding.at(lastCol).hBackwardEmbedding.at(lastBWCol).dArrEmbedding)); //phải phóng bộ nhớ backward embedding column trên device được quản lý bởi phần tử của vector
					if(cudaStatus!=cudaSuccess)
					{
						status=-1;
						goto Error;
					}
					hEmbedding.at(lastCol).hBackwardEmbedding.pop_back(); //gỡ bỏ phần tử của vector quản lý backward embedding column trên device.
				}
				else
				{
					CHECK(cudaStatus = cudaFree(hEmbedding.at(lastCol).dArrEmbedding)); //xoá phần tử cuối của Embedding
					if(cudaStatus!=cudaSuccess)
					{
						status=-1;
						goto Error;
					}

					hEmbedding.pop_back();
				}
				//Gỡ bỏ hLevelPtrEmbedding ở Level hiện tại
				if(hLevelPtrEmbedding.at(objLevel.Level).noElem>0)
				{
					CHECK(cudaStatus = cudaFree(hLevelPtrEmbedding.at(objLevel.Level).dArrPointerEmbedding));
					if(cudaStatus!=cudaSuccess)
					{
						status=-1;
						goto Error;
					}
					//hLevelPtrEmbedding.pop_back();
				}

				//Gỡ bỏ RMP ở Level hiện tại
				if(hLevelRMP.at(objLevel.Level).noElem>0)
				{
					CHECK(cudaStatus=cudaFree(hLevelRMP.at(objLevel.Level).dRMP));
					if(cudaStatus!=cudaSuccess)
					{
						status=-1;
						goto Error;
					}
					hLevelRMP.at(objLevel.Level).hArrRMP.clear();
					//hLevelRMP.pop_back();
				}
			}
			else
			{
				//cout<<endl<<"This is not satisfy minsup"<<endl;
			}

			//Sau khi khai thác xong thì gỡ bỏ DFS_CODE
			DFS_CODE.pop_back(); //Xoá phần tử cuối của DFS_CODE. Sự khác nhau giữa pop() và pop_back() là gì?
			if(backward!=1)
			{ //Nếu pop() một forward thì phải giảm maxId
				--maxId;
			}	
		}
		free(temp);
	} 

#pragma endregion

	//Giải phóng bộ nhớ không cần thiết
	//Gỡ bỏ hLevelEXT ở Level hiện tại
	if(hLevelEXT.at(objLevel.Level).vE.empty() == false)
	{
		for (int i = 0; i < hLevelEXT.at(objLevel.Level).noElem; i++)
		{
			if(hLevelEXT.at(objLevel.Level).vE.at(i).noElem>0)
			{
				CHECK(cudaStatus= cudaFree(hLevelEXT.at(objLevel.Level).vE.at(i).dArrExt));
				if(cudaStatus!=cudaSuccess)
				{
					status =-1;
					goto Error;
				}
			}
		}
		hLevelEXT.at(objLevel.Level).vE.clear();
		//hLevelEXT.pop_back();
	}
	//Gỡ bỏ hLevelUniEdgeSatisfyMinsup ở Level hiện tại
	if(hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.empty() == false)
	{
		for (int i = 0; i < hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).noElem; i++)
		{
			if(hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(i).noElem>0)
			{
				CHECK(cudaStatus= cudaFree(hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(i).dArrUniEdge));
				if(cudaStatus!=cudaSuccess)
				{
					status =-1;
					goto Error;
				}
				free(hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(i).hArrSupport);
			}
		}
		hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.clear();
		//hLevelUniEdgeSatisfyMinsup.pop_back();
	}

	//Gỡ bỏ hLevelListVerRMP ở Level hiện tại
	if (hLevelListVerRMP.at(objLevel.Level).noElem>0)
	{
		CHECK(cudaStatus = cudaFree(hLevelListVerRMP.at(objLevel.Level).dListVer));
		if(cudaStatus != cudaSuccess)
		{
			status=-1;
			goto Error;
		}
	}

	CHECK(cudaStatus = cudaFree(tempListVerCol));
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}
	CHECK(cudaStatus = cudaFree(dArrVidOnRMP));
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}

	//The last step. Giảm Level
	decreaseLevel();

Error:
	return status;
}


int PMS::MiningDeeper(EXTk &ext,UniEdgeStatisfyMinSup &UES)
{
	if (UES.noElem <= 0) return 0;
	UES.hArrUniEdge = (UniEdge*)malloc(sizeof(UniEdge)*UES.noElem);
	
	for(int idx_ues = 0; idx_ues < UES.noElem ; ++idx_ues)
	{
		if(UES.hArrUniEdge==nullptr) {return 0;}
		CUCHECK(cudaMemcpy(UES.hArrUniEdge,UES.dArrUniEdge,sizeof(UniEdge)*UES.noElem,cudaMemcpyDeviceToHost));
		DFS_CODE.add(UES.hArrUniEdge[idx_ues].vi,UES.hArrUniEdge[idx_ues].vj, \
			UES.hArrUniEdge[idx_ues].li,UES.hArrUniEdge[idx_ues].lij,UES.hArrUniEdge[idx_ues].lj);
		//Check minDFSCode
		if(is_min())
		{
			//Đánh tất cả các embedding của unique edge trong UES.hArrUniEdge
			//Kết quả đánh dấu của lưu vào dValid
			int *dValid = nullptr;
			//Tại sao mark_edge không truyền vào posRow? Điều này có ảnh hưởng gì không?
			ext.mark_edge(UES.hArrUniEdge[idx_ues].vi,UES.hArrUniEdge[idx_ues].vj, \
				UES.hArrUniEdge[idx_ues].li,UES.hArrUniEdge[idx_ues].lij,UES.hArrUniEdge[idx_ues].lj,dValid);
			//Hiển thị các phần tử trong ext.
			std::printf("\n********ext.show()************\n");
			ext.show();
			//FCHECK(displayDeviceArr(dValid,ext.noElem));

			//Tạo mảng index của các phần tử hợp lệ trong mảng mới.
			int *dIdx = nullptr;
			get_idx(dValid,ext.noElem,dIdx);
			FCHECK(displayDeviceArr(dIdx,ext.noElem));

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
			for (int i = 0; i < hEm.size(); i++)
			{
				cout<<endl;
				hEm.at(i).show();
			}

			vector<EXTk> vecValidEXTk;
			//Find valid extension and return vector EXTk<i>
			findValidExtension(vecValidEXTk);
			//Trích các Unique Forward Extension ở tất cả các EXTk
			for (int idxEXTk = 0; idxEXTk < vecValidEXTk.size(); idxEXTk++)
			{
				if(vecValidEXTk.at(idxEXTk).noElem>0)
				{
					vecValidEXTk.at(idxEXTk).show();
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
					vecValidEXTk.at(lastIdxEXTk).show();
					//Trích các unique edge backward.
					int* dRMP = nullptr;
					int* dRMPLabel = nullptr;
					int noElemMappingVj = 0;
					buildRMPLabel(dRMP,dRMPLabel,noElemMappingVj);
					vecValidEXTk.at(lastIdxEXTk).extractUniBackwardExtension(Lv,Le,DFS_CODE.noElemOnRMP,dRMP,dRMPLabel,noElemMappingVj,DFS_CODE.back().to,DFS_CODE.back().tolabel);
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
					}
					if(vecValidEXTk.at(idxvecValidEXTk).uniFES.noElem>0)
					{
						MiningDeeper(vecValidEXTk.at(idxvecValidEXTk),vecValidEXTk.at(idxvecValidEXTk).uniFES);
					}
					//Giải phóng bộ nhớ cho EXTk đã khai thác xong
					//vecValidEXTk.at(idxvecValidEXTk);
				}
			}
			


			//Release memory in HEAP and device
			for (int i = 0; i < vecValidEXTk.size(); i++)
			{
				if(vecValidEXTk.at(i).noElem>0)
				{
					vecValidEXTk.at(i).ReleaseMemory();
				}
			}
			vecValidEXTk.clear();
			CUCHECK(cudaFree(dValid));
			CUCHECK(cudaFree(dIdx));
			removeEmbedding();
		}
		//Khi khai thác xong thì gỡ bỏ cạnh vừa thêm ra khỏi DFS_CODE
		DFS_CODE.remove(UES.hArrUniEdge[idx_ues].vi,UES.hArrUniEdge[idx_ues].vj);
		if(DFS_CODE.dRMP!=nullptr) CUCHECK(cudaFree(DFS_CODE.dRMP));
	}
	FCHECK(ext.ReleaseMemory());
	FCHECK(UES.ReleaseMemory());
	return 0;
}


int PMS::initialize()
{
	int status = 0;
	cudaError_t cudaStatus;

	//Nên kiểm tra cái này ở đầu hàm Mining(), vì không có mở rộng thoả minsup nào thì chúng ta return về status liền. Không phải mất công làm các bước trên.
	/*if(hUniEdgeSatisfyMinsup.at(0).noElem<=0){
	PMS_PRINT("\n There no any edge in hUniEdgeSatisfyMinsup\n");
	return status;
	}*/
	//Nếu không có mở rộng nào thoả minsup thì return không khai thác nhánh đó nữa
	if(hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(0).noElem<=0)
	{
		PMS_PRINT("\n There no any edge in hUniEdgeSatisfyMinsup at Level: %d, at vecUES: 0 ",objLevel.Level);
		goto Error;
	}

	int noElemtemp = hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(0).noElem;
	if (noElemtemp==0)
	{
		PMS_PRINT("\n No any extension that satisfy minsup. \n Mining has been stopped\n");
		goto Error;
	}
	//Cấp phát một mảng tạm ở host để chép dữ liệu từ device sang host lưu giữ các cạnh duy nhất thoả minsup
	UniEdge *temp=(UniEdge*)malloc(sizeof(UniEdge)*noElemtemp);
	if(temp==NULL)
	{
		PMS_PRINT("\n malloc temp failed");
		status =-1;
		goto Error;
	}
	//chép dữ liệu từ device sang host
	CHECK(cudaStatus=cudaMemcpy(temp,hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(0).dArrUniEdge,noElemtemp*sizeof(UniEdge),cudaMemcpyDeviceToHost));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	FUNCHECK(status=buildRMP()); //Xây dựng RMP ban đầu trên host và device
	if(status!=0)
	{
		goto Error;
	}

	//Duyệt qua các UniEdge thoả minSup để kiểm tra minDFS_CODE, nếu thoả thì ghi kết quả vào result và xây dựng embedding
	/*cout<<"***************************"<<cout<<endl;
	cout<<"Duyet qua cac unique edges thoa minsup --> check minDSF_CODE --> build Embedding columns "<<endl;*/
	for (int i = 0; i < noElemtemp; i++) 
	{
		//cout<<"\t Duyet qua unique edges thu: "<<i<<endl; //comment
		int li,lij,lj;
		li = temp[i].li;
		lij= temp[i].lij;
		lj=temp[i].lj;

		DFS_CODE.push(0,1,temp[i].li,temp[i].lij,temp[i].lj);//xây dựng DFS_CODE
		minLabel = temp[i].li;
		maxId = 1;

		//Ban đầu pattern chỉ có một cạnh nên DFS_CODE của nó luôn nhỏ nhất. Do đó, chúng ta không cần kiểm tra minDFS_Code ở bước này.
		//Mà tiến hành ghi kết quả vào file result.txt luôn.
		
		//Mảng chứa các graphID có embedding của DFS_Code.
		int *hArrGraphId;
		//Số lượng GraphID chứa các embeddings của minDFSCODE ban đầu.
		int noElemhArrGraphId=0;
		/* Trước khi ghi kết quả thì phải biết đồ thị phổ biến đó tồn tại ở những graphId nào. Hàm getGraphIdContainEmbedding dùng để làm việc này
		* 3 tham số đầu tiên của hàm là nhãn cạnh của phần tử d_UniqueExtension đang xét */
		//cout<<"\tThu thap cac GraphID"<<endl; //comment
		FUNCHECK(status =getGraphIdContainEmbedding_pure(temp[i],hArrGraphId,noElemhArrGraphId));
		if (status!=0)
		{
			PMS_PRINT("\n\n getGraphIdContainEmbedding() in Mining() failed");
			goto Error;
		}

		//In nội dung mảng hArrGraphId

		/*PMS_PRINT("\n ************** hArrGraphId ****************\n");
		for (int j = 0; j < noElemhArrGraphId; j++)
		{
			PMS_PRINT("%d ",hArrGraphId[j]);
		}*/

		/*	Ghi kết quả DFS_CODE vào file result.txt ************************************************************
		*	Hàm report sẽ chuyển DFS_CODE pattern sang dạng đồ thị, sau đó sẽ ghi đồ thị đó xuống file result.txt
		*	Hàm report gồm 3 tham số:
		*	Tham số thứ 1: mảng chứa danh sách các graphID chứa DFS_CODE pattern
		*	Tham số thứ 2: số lượng mảng
		*	Tham số thứ 3: độ hỗ trợ của DFS_CODE pattern *******************************************************/

		//report(hArrGraphId,noElemhArrGraphId,hUniEdgeSatisfyMinsup.at(0).hArrSup[i]);
		//cout<<"\t Ghi ket qua subgraph xuong result.txt"<<endl; //comment
		report(hArrGraphId,noElemhArrGraphId,hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(0).hArrSupport[i]);
		//cout<<"\t Ghi ket qua mindfscode xuong dfscodemin.txt"<<endl; //comment
		write_minDFS_CODE(DFS_CODE);
		//Giải phóng bộ nhớ hArrGraphId sau khi ghi kết quả
		//cout<<"\t Giải phóng hArrGraphId"<<endl; //comment
		if(hArrGraphId!=NULL)
		{
			free(hArrGraphId);
		}
		//Xây dựng Embedding cho DFS_Code rồi gọi hàm GraphMining để khai thác
		//Trong GraphMining sẽ gọi GraphMining khác để thực hiện khai thác đệ quy
		//cout<<"\t Xay dung embedding columns cho mindfscode"<<endl; //comment
		FUNCHECK(status=buildEmbedding_pure(temp[i])); //Xây dựng 2 cột embedding ban đầu trên host và device.
		if(status!=0)
		{
			goto Error;
		}
		//Tìm các đỉnh thuộc RMP
		//cout<<"\t Tim cac dinh thuoc RMP"<<endl; //comment
		FUNCHECK(status=findListVerOnRMP());
		if(status!=0)
		{
			goto Error;
		}
		//cout<<"\t Goi ham Mining()"<<endl; //comment
		FUNCHECK(status = Mining());
		if(status!=0)
		{
			goto Error;
		}

		//FUNCHECK(status=FSMining(rmp,noElemVerOnRMP)); //Gọi FSMining.( Hàm này thực hiện theo tuần tự (1. Find Extension -> 2. Extract UniEdge -> 3.Compute & CHECK Support -> 4. CHECK minDFS_CODE -> 5. BuildEmbedding -> 6.Find RMP -> 1.)
		//if(status!=0){
		//	goto Error;
		//}

		//Giải phóng bộ nhớ 
		//Giải phóng bộ nhớ chứa các đỉnh cần mở rộng thuộc embedding columns
		CHECK(cudaStatus=cudaFree(hLevelListVerRMP.at(objLevel.Level).dListVer));
		if(cudaStatus!=cudaSuccess)
		{
			status=-1;
			goto Error;
		}
		hLevelListVerRMP.pop_back();

		//3. Giải phóng embedding
		if(hEmbedding.size()>0)
		{ 
			for (int j = hEmbedding.size()-1; j>=0; j--)
			{
				CHECK(cudaStatus=cudaFree(hEmbedding.at(j).dArrEmbedding));
				if(cudaStatus!=cudaSuccess)
				{
					status=-1;
					goto Error;
				}
				hEmbedding.pop_back();
			}
		}
		DFS_CODE.pop(); //2. Gỡ bỏ mở rộng vừa thêm vào DFS_CODE sau khi đã khai thác xong
	} //kết thúc for: việc duyệt qua tất cả các satisfied minsup unique edge	
	//Giải phóng bộ nhớ level
	if(hLevelRMP.at(objLevel.Level).noElem>0)
	{ 
		hLevelRMP.at(objLevel.Level).hArrRMP.clear();

		CHECK(cudaStatus=cudaFree(hLevelRMP.at(objLevel.Level).dRMP));
		if(cudaStatus!=cudaSuccess)
		{
			status=-1;
			goto Error;
		}

		hLevelRMP.clear();
	}

	CHECK(cudaStatus = cudaFree(hLevelEXT.at(objLevel.Level).vE.at(0).dArrExt));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	hLevelEXT.at(objLevel.Level).vE.clear();

	CHECK(cudaStatus=cudaFree(hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(0).dArrUniEdge));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	free(hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(0).hArrSupport);
	hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.clear();
	hLevelUniEdgeSatisfyMinsup.clear();

	hLevelListVerRMP.clear();
	if(temp!=NULL)
	{
		std::free(temp); 
	}

Error:
	return status;
}


__global__ void	kernelGetvivj(EXT *dArrEXT,int noElemdArrEXT,int li,int lij,int lj,int *dvi,int *dvj){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElemdArrEXT){
		if(dArrEXT[i].li == li && dArrEXT[i].lij == lij && dArrEXT[i].lj == lj){
			*dvi=dArrEXT[i].vi;
			*dvj=dArrEXT[i].vj;
			//PMS_PRINT("\n Thread:%d (dvi dvj):(%d %d)",i,*dvi,*dvj);
		}
	}
}

int displayDeviceEXT(EXT *dArrEXT,int noElemdArrEXT){
	int status =0;
	cudaError_t cudaStatus;

	EXT *hArrEXT = (EXT*)malloc(sizeof(EXT)*noElemdArrEXT);
	if(hArrEXT == NULL){
		PMS_PRINT("\n malloc hArrEXT failed");
		status =-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaMemcpy(hArrEXT,dArrEXT,noElemdArrEXT*sizeof(EXT),cudaMemcpyDeviceToHost));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	PMS_PRINT("\n************** EXT **************\n");
	for (int i = 0; i < noElemdArrEXT; i++)
	{
		PMS_PRINT("\n (vi vj):(%d %d) (li lij lj):(%d %d %d) (vgi vgj):(%d %d) (RowPointer:%d)",hArrEXT[i].vi,hArrEXT[i].vj,hArrEXT[i].li,hArrEXT[i].lij,hArrEXT[i].lj,hArrEXT[i].vgi,hArrEXT[i].vgj,hArrEXT[i].posRow);
	}

	std::free(hArrEXT);
Error:
	return status;
}



int PMS::getvivj(EXT *dArrEXT,int noElemdArrEXT,int li,int lij,int lj,int &vi,int &vj){
	int status=0;
	cudaError_t cudaStatus;

	int *dvi=nullptr;
	int *dvj=nullptr;
	size_t nBytesvi=sizeof(int);

	CHECK(cudaStatus = ADM(dvi,nBytesvi));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus = ADM(dvj,nBytesvi));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	//cudaStatus = cudaMalloc((void**)&dvi,sizeof(int));
	//CHECK(cudaStatus);
	//if(cudaStatus !=cudaSuccess){
	//	status =-1;
	//	goto Error;
	//}

	//cudaStatus = cudaMalloc((void**)&dvj,sizeof(int));
	//CHECK(cudaStatus);
	//if(cudaStatus !=cudaSuccess){
	//	status =-1;
	//	goto Error;
	//}
	dim3 block(blocksize);
	dim3 grid((noElemdArrEXT+block.x-1)/block.x);

	FUNCHECK(status =displayDeviceEXT(dArrEXT,noElemdArrEXT));
	if(status!=0){
		goto Error;
	}

	kernelGetvivj<<<grid,block>>>(dArrEXT,noElemdArrEXT,li,lij,lj,dvi,dvj);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus=cudaStatus=cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus=cudaMemcpy(&vi,dvi,nBytesvi,cudaMemcpyDeviceToHost));
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}

	CHECK(cudaStatus=cudaMemcpy(&vj,dvj,nBytesvi,cudaMemcpyDeviceToHost));
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}


	CHECK(cudaStatus = cudaFree(dvi));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus =cudaFree(dvj));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
Error:
	return status;
}

//int PMS::updateRMPBW_old(){
//	int status=0;
//	cudaError_t cudaStatus;
//	hRMPv2.resize(Level);//Tạo Level mới cho hRMPv2
//	//int noEC = hEmbedding.size(); //noEC is number of embedding column
//	//int lastIdxEC = noEC-1;
//	//for (int i = lastIdxEC; i != -1;)
//	//{
//	//	hRMPv2.at(idxLevel).hArrRMP.push_back(i);
//	//	i=hEmbedding.at(i).prevCol;		 //Dựa vào hEmbedding để tìm right most path và lưu chúng trên bộ nhớ ở host
//	//}
//	//hRMPv2.at(idxLevel).noElem = hRMPv2.at(idxLevel).hArrRMP.size(); //Cập nhật số lượng phần tử của vector hArrRMP
//	////In RMP
//	for (int i = 0; i < hRMPv2.at(idxLevel-1).noElem; i++) //Duyệt qua số lượng phần tử ở Level trước
//	{
//		//PMS_PRINT("\n RMPv2[%d]:%d",i,hRMPv2.at(idxLevel).hArrRMP.at(i));
//		hRMPv2.at(idxLevel).hArrRMP.push_back(hRMPv2.at(idxLevel-1).hArrRMP.at(i)); //gán giá trị của level trước cho level hiện tại
//		PMS_PRINT("\n RMPv2[%d]:%d",i,hRMPv2.at(idxLevel).hArrRMP.at(i));
//	}
//	hRMPv2.at(idxLevel).noElem=hRMPv2.at(idxLevel-1).noElem;
//
//Error:
//	return status;
//}

int PMS::updateRMPBW()
{
	int status=0;
	cudaError_t cudaStatus;

	//hRMPv2.resize(Level);//Tạo Level mới cho hRMPv2
	//int noEC = hEmbedding.size(); //noEC is number of embedding column
	//int lastIdxEC = noEC-1;
	//for (int i = lastIdxEC; i != -1;)
	//{
	//	hRMPv2.at(idxLevel).hArrRMP.push_back(i);
	//	i=hEmbedding.at(i).prevCol;		 //Dựa vào hEmbedding để tìm right most path và lưu chúng trên bộ nhớ ở host
	//}
	//hRMPv2.at(idxLevel).noElem = hRMPv2.at(idxLevel).hArrRMP.size(); //Cập nhật số lượng phần tử của vector hArrRMP
	////In RMP

	for (int i = 0; i < hLevelRMP.at(objLevel.prevLevel).noElem; i++) //Duyệt qua số lượng phần tử ở Level trước
	{
		//PMS_PRINT("\n RMPv2[%d]:%d",i,hRMPv2.at(idxLevel).hArrRMP.at(i));
		hLevelRMP.at(objLevel.Level).hArrRMP.push_back(hLevelRMP.at(objLevel.prevLevel).hArrRMP.at(i)); //gán giá trị của level trước cho level hiện tại
		//PMS_PRINT("\n RMPv2[%d]:%d",i,hLevelRMP.at(objLevel.Level).hArrRMP.at(i));
	}
	hLevelRMP.at(objLevel.Level).noElem=hLevelRMP.at(objLevel.prevLevel).noElem;

#pragma region "build RMP on device"
	//Xây dựng right most path từ vector<int> hRMP
	int noElemVerOnRMP = hLevelRMP.at(objLevel.Level).noElem; //right most path chứa bao nhiêu đỉnh
	hLevelRMP.at(objLevel.Level).dRMP = nullptr; //rigt most path trên bộ nhớ device
	CHECK(cudaStatus = cudaMalloc((void**)&hLevelRMP.at(objLevel.Level).dRMP,noElemVerOnRMP*sizeof(int))); //cấp phát bộ nhớ trên device cho rmp
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}
	int *tempRMP=(int*)malloc(sizeof(int)*noElemVerOnRMP); //dùng để chứa dữ liệu từ vector hRMP
	if(tempRMP==NULL)
	{
		status =-1;
		goto Error;
	}
	////chép dữ liệu từ hRMP sang bộ nhớ temp
	for (int i = 0; i < noElemVerOnRMP; i++)
	{
		tempRMP[i] = hLevelRMP.at(objLevel.Level).hArrRMP.at(i);
	}

	//Chép dữ liệu từ temp trên host sang rmp trên device. //ở bước này không cần phải làm phức tạp như thế. Chỉ cần khởi tạo bộ nhớ trên Device và gán cho nó giá trị {1,0} cho nó là được
	CHECK(cudaStatus =cudaMemcpy(hLevelRMP.at(objLevel.Level).dRMP,tempRMP,sizeof(int)*noElemVerOnRMP,cudaMemcpyHostToDevice)); //chép dữ liệu từ temp ở host sang rmp trên device
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}

	if(tempRMP!=NULL)
	{
		free(tempRMP);
	}

	/*PMS_PRINT("\n\n ******* rmp *********\n");
	FUNCHECK(status=displayDeviceArr(hLevelRMP.at(objLevel.Level).dRMP,noElemVerOnRMP));
	if(status!=0)
	{
		goto Error;
	}*/

#pragma endregion


Error:
	return status;
}


//int PMS::updateRMP_old(){
//	int status=0;
//	cudaError_t cudaStatus;
//	hRMPv2.resize(Level);//Tạo Level mới cho hRMPv2
//	int noEC = hEmbedding.size(); //noEC is number of embedding column
//	int lastIdxEC = noEC-1;
//	for (int i = lastIdxEC; i != -1;)
//	{
//		hRMPv2.at(idxLevel).hArrRMP.push_back(i);
//		i=hEmbedding.at(i).prevCol;		 //Dựa vào hEmbedding để tìm right most path và lưu chúng trên bộ nhớ ở host
//	}
//	hRMPv2.at(idxLevel).noElem = hRMPv2.at(idxLevel).hArrRMP.size(); //Cập nhật số lượng phần tử của vector hArrRMP
//	//In RMP
//	for (int i = 0; i < hRMPv2.at(idxLevel).noElem; i++)
//	{
//		PMS_PRINT("\n RMPv2[%d]:%d",i,hRMPv2.at(idxLevel).hArrRMP.at(i));
//	}
//
//Error:
//	return status;
//}

int PMS::updateRMP()
{
	int status=0;
	cudaError_t cudaStatus;

	//1.Create new level for hLevelRMP
	hLevelRMP.resize(objLevel.Level + 1);

	//2.Get number of embedding columns.
	int noEC = hEmbedding.size(); 
	int lastIdxEC = noEC-1;
	int Qi = lastIdxEC;
	//3. Dựa vào hEmbedding để tìm right most path và lưu chúng trên bộ nhớ ở host
	do 
	{
		hLevelRMP.at(objLevel.Level).hArrRMP.push_back(Qi);
		Qi=hEmbedding.at(Qi).prevCol; 
	} while(Qi!=-1);

	hLevelRMP.at(objLevel.Level).noElem = hLevelRMP.at(objLevel.Level).hArrRMP.size();

	//4. Display hArrRMP at hLevelRMP of current Level.
	/*for (int i = 0; i < hLevelRMP.at(objLevel.Level).noElem; i++)
	{
		PMS_PRINT("\n RMPv2[%d]:%d",i,hLevelRMP.at(objLevel.Level).hArrRMP.at(i));
	}*/

#pragma region "build RMP on device"
	//Xây dựng right most path trên device dựa vào vector<int> hRMP
	int noElemVerOnRMP = hLevelRMP.at(objLevel.Level).noElem;

	//1. Cấp phát bộ nhớ trên device cho dRMP tại current Level bằng với số lượng phần tử của mảng hRMP.
	hLevelRMP.at(objLevel.Level).dRMP = nullptr; //rigt most path trên bộ nhớ device
	CHECK(cudaStatus = cudaMalloc((void**)&hLevelRMP.at(objLevel.Level).dRMP,noElemVerOnRMP*sizeof(int)));
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}

	//Cấp phát bộ nhớ tạm trên host để chép RMP từ vector hRMP sang.
	int *tempRMP=(int*)malloc(sizeof(int)*noElemVerOnRMP);
	if(tempRMP==NULL)
	{
		status =-1;
		goto Error;
	}
	////chép dữ liệu từ hRMP sang bộ nhớ temp
	for (int i = 0; i < noElemVerOnRMP; i++)
	{
		tempRMP[i] = hLevelRMP.at(objLevel.Level).hArrRMP.at(i);
	}

	//Chép dữ liệu từ temp trên host sang rmp trên device.
	CHECK(cudaStatus =cudaMemcpy( \
		hLevelRMP.at(objLevel.Level).dRMP, \
		tempRMP, \
		sizeof(int)*noElemVerOnRMP, \
		cudaMemcpyHostToDevice));
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}

	if(tempRMP!=NULL)
	{
		free(tempRMP);
	}
	//Hiển thị RMP
	/*PMS_PRINT("\n\n ******* hLevelRMP.at(%d).dRMP *********\n",objLevel.Level);
	FUNCHECK(status=displayDeviceArr(hLevelRMP.at(objLevel.Level).dRMP,noElemVerOnRMP));
	if(status!=0){goto Error;}*/
#pragma endregion
Error:
	return status;
}

int PMS::updateRMP_DFSCODE()
{
	int status=0;
	cudaError_t cudaStatus;
	//1.Get number of embedding columns.
	int noEC = hEmbedding.size(); 

	//2. Allocation host memory to store RMP of DFSCODE.
	DFS_CODE.hRMP = (int*)malloc(sizeof(int)*noEC);
	
	if(DFS_CODE.hRMP == NULL) {status = -1; exit(0);}

	int lastIdxEC = noEC-1;
	int Qi = lastIdxEC;
	//3. Dựa vào hEmbedding để tìm right most path và lưu chúng trên bộ nhớ ở host
	int i =0;
	do 
	{
		
		DFS_CODE.hRMP[i] = Qi;
		Qi=hEmbedding.at(Qi).prevCol; 
		++i;
	} while(Qi!=-1);
	DFS_CODE.noElemOnRMP = i;

#pragma region "build RMP on device"

	//1. Cấp phát bộ nhớ trên device cho dRMP tại current Level bằng với số lượng phần tử của mảng hRMP.
	DFS_CODE.dRMP = nullptr; //rigt most path trên bộ nhớ device
	CHECK(cudaStatus = cudaMalloc((void**)&DFS_CODE.dRMP,noEC*sizeof(int)));
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}

	
	//Chép dữ liệu từ temp trên host sang rmp trên device.
	CHECK(cudaStatus =cudaMemcpy( \
		DFS_CODE.dRMP, \
		DFS_CODE.hRMP, \
		sizeof(int)*noEC, \
		cudaMemcpyHostToDevice));
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}
#pragma endregion
Error:
	return status;
}

//Input: các unique edge thoả minsup
//Need to do: 0. Duyệt qua các unique edge satified minsup -> 1. mở rộng DFS_Code -> 2. Cần kiểm tra minDFS_code (nếu thoả) -> 3. Ghi nhận kết quả -> 4. Mở rộng embedding column -> 5. Gọi Miningv2()
//6. xoá embedding column và hLevelPtrEmbedding tại Level đang xét -> 7.xoá DFS_Code -> 8.giảm Level
int PMS::Miningv2(int noElem,UniEdge *dArrUniEdgeSatisfyMinSup,int *hArrSupport,EXT *dArrEXT,int noElemdArrEXT,int idxExt){
	int status = 0;
	cudaError_t cudaStatus;
	//1.Tăng Level
	Level++;
	idxLevel=Level-1;

	int vi,vj,backward;
	vi=vj=-1;
	backward=0;
	int noElemtemp = noElem;
	UniEdge *temp=(UniEdge*)malloc(sizeof(UniEdge)*noElemtemp);
	if(temp==NULL){
		PMS_PRINT("\n malloc temp failed");
		status =-1;
		goto Error;
	}

	CHECK(cudaStatus=cudaMemcpy(temp,dArrUniEdgeSatisfyMinSup,noElemtemp*sizeof(UniEdge),cudaMemcpyDeviceToHost));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	//Duyệt qua các UniEdge thoả minSup để kiểm tra minDFS_CODE, nếu thoả thì ghi kết quả vào result và xây dựng embedding
	for (int i = 0; i < noElemtemp; i++) 
	{
		int li,lij,lj;
		li = temp[i].li;
		lij= temp[i].lij;
		lj=temp[i].lj;

		FUNCHECK(status = getvivj(dArrEXT,noElemdArrEXT,li,lij,lj,vi,vj));
		if(status!=0){
			goto Error;
		}
		if(vi>vj){
			backward=1;
		}

		//Nếu là mở rộng forward thì cập nhật lại maxId bằng vj;
		if(backward!=1){
			DFS_CODE.push(vi,vj,-1,temp[i].lij,temp[i].lj);//xây dựng DFS_CODE forward
			maxId=vj;
		}
		else
		{
			DFS_CODE.push(vi,vj,-1,temp[i].lij,-1);//xây dựng DFS_CODE backward
		}

		if(is_min()){ //Nếu DFS_CODE là min thì tìm các graphid chứa embedding của DFS_CODE
			PMS_PRINT("\n This is minDFSCODE\n");

			int *hArrGraphId; //Mảng chứa các graphID có embedding của DFS_Code.
			int noElemhArrGraphId=0;
			FUNCHECK(status =getGraphIdContainEmbeddingFW(temp[i],hArrGraphId,noElemhArrGraphId,dArrEXT,noElemdArrEXT));
			if (status!=0){
				goto Error;
			}
			////In nội dung mảng hArrGraphId
			/*PMS_PRINT("\n ************** hArrGraphId ****************\n");
			for (int j = 0; j < noElemhArrGraphId; j++)
			{
				PMS_PRINT("%d ",hArrGraphId[j]);
			}*/

			report(hArrGraphId,noElemhArrGraphId,hArrSupport[i]);
			write_minDFS_CODE(DFS_CODE);
			//Giải phóng bộ nhớ 
			std::free(hArrGraphId);

			//Xây dựng Embedding cho DFS_Code rồi gọi hàm GraphMining để khai thác
			//Trong GraphMining sẽ gọi GraphMining khác để thực hiện khai thác đệ quy

			//Hàm extendEmbedding chỉ xây dựng embedding column mới cho unique edge that satisfy minsup
			FUNCHECK(status=extendEmbedding(temp[i],idxExt));
			if(status!=0){
				goto Error;
			}
			//hLevelPtrEmbeddingv2.resize(Level);
			//hLevelPtrEmbeddingv2.at(idxLevel).noElem=hEmbedding.size();
			//int lastCol = hEmbedding.size()-1;
			//hLevelPtrEmbeddingv2.at(idxLevel).noElemEmbedding=hEmbedding.at(lastCol).noElem;
			//CHECK(cudaStatus = cudaMalloc((void**)&hLevelPtrEmbeddingv2.at(idxLevel).dArrPointerEmbedding,hLevelPtrEmbeddingv2.at(idxLevel).noElem*sizeof(Embedding**))); //Cấp phát bộ nhớ cho mảng dArrPointerEmbedding.
			//if(cudaStatus!=cudaSuccess){
			//	status = -1;
			//	std::PMS_PRINT("\n cudaMalloc dArrPointerEmbedding failed()");
			//	goto Error;
			//}
			//for (int i = 0; i < hEmbedding.size(); i++)
			//{		
			//	kernelGetPointerdArrEmbedding<<<1,1>>>(hEmbedding.at(i).dArrEmbedding,hLevelPtrEmbeddingv2.at(idxLevel).dArrPointerEmbedding,i); //Mỗi phần tử của mảng dArrPointerEmbedding chứa địa chỉ của dArrEmbedding
			//}
			//cudaDeviceSynchronize();
			//cudaStatus = cudaGetLastError();
			//CHECK(cudaStatus);
			//if(cudaStatus!=cudaSuccess){
			//	status = -1;
			//	PMS_PRINT("\n kernelGetPointerdArrEmbedding failed");
			//	goto Error;
			//}
			//FUNCHECK(status = updateRMP());
			//if(status!=0){
			//	goto Error;
			//}
			FUNCHECK(status=FSMiningv2()); //Gọi FSMining.( Hàm này thực hiện theo tuần tự (1. Find Extension -> 2. Extract UniEdge -> 3.Compute & CHECK Support -> 4. CHECK minDFS_CODE -> 5. BuildEmbedding -> 6.Find RMP -> 1.)
			if(status!=0){
				goto Error;
			}
			//Giải phóng bộ nhớ
			//Lấy chỉ số phần tử cuối của embedding
			int lastCol = hRMPv2.at(idxLevel).hArrRMP[0];
			//xoá phần tử cuối của Embedding column trên device
			CHECK(cudaStatus=cudaFree(hEmbedding.at(lastCol).dArrEmbedding)); 
			if(cudaStatus!=cudaSuccess){
				status=-1;
				goto Error;
			}

			//xoá hLevelPtrEmbeddingv2 ở device tại Level hiện tại
			CHECK(cudaStatus=cudaFree(hLevelPtrEmbeddingv2.at(idxLevel).dArrPointerEmbedding)); //xoá dArrPointerEmbedding tại Level đang xét
			if(cudaStatus!=cudaSuccess){
				status=-1;
				goto Error;
			}
			//xoá phần tử cuối của vector quản lý embedding column trên host
			hEmbedding.pop_back();

			//xoá vector quản lý hLevelPtrEmbeddingv2 ở host tại Level hiện tại
			hLevelPtrEmbeddingv2.pop_back(); //Xoá phần tử cuối của Level pointerEmbeeding đang xét.
			//xoá right most path ở level hiện tại
			hRMPv2.at(idxLevel).hArrRMP.clear(); //Xoá  RightMostPath của phần tử Embedding tại Level tương ứng.
			hRMPv2.pop_back();

		} //kết thúc kiểm tra is_min()
		//Xoá phần tử cuối của DFS_CODE
		DFS_CODE.pop(); 
		//Nếu pop() một forward thì phải giảm maxId
		if(backward!=1){ 
			--maxId;
		}
		hLevelPtrEmbeddingv2.pop_back();
	}// kết thúc duyệt qua các uniEdge satisfy minsup tại một EXT<i> của Level. Chúng ta chỉ giảm Level khi đã duyệt qua tất cả các EXT<i>
	Level--;
	idxLevel=Level-1;
	std::free(temp);
Error:
	return status;
}

//Đã tính độ hỗ trợ xong. Cần kiểm tra minDFS_code
//status=Miningv3(hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).noElem,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).dArrUniEdge,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).hArrSupport,hLevelEXTv2.at(idxLevel).vE.at(idxEXT).dArrExt,hLevelEXTv2.at(idxLevel).vE.at(idxEXT).noElem,idxEXT);
int PMS::Miningv3(int noElemUniEdgeSatisfyMinSup,UniEdge *dArrUniEdgeSatisfyMinSup,int *hArrSupport,EXT *dArrEXT,int noElemdArrEXT,int idxExt){
	int status = 0;
	cudaError_t cudaStatus;

	int noElemtemp = noElemUniEdgeSatisfyMinSup;
	UniEdge *temp=(UniEdge*)malloc(sizeof(UniEdge)*noElemtemp); //1. need free
	if(temp==NULL){
		PMS_PRINT("\n malloc temp failed");
		status =-1;
		goto Error;
	}

	CHECK(cudaStatus=cudaMemcpy(temp,dArrUniEdgeSatisfyMinSup,noElemtemp*sizeof(UniEdge),cudaMemcpyDeviceToHost));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	for (int i = 0; i < noElemtemp; i++) //Duyệt qua các UniEdge thoả minSup để kiểm tra minDFS_CODE, nếu thoả thì ghi kết quả vào result và xây dựng embedding
	{
		int backward=0;
		int li,lij,lj,vi,vj;
		li = temp[i].li;
		lij= temp[i].lij;
		lj=temp[i].lj;
		vi=temp[i].vi;
		vj=temp[i].vj;
		PMS_PRINT("\n Extent from DFS code:(%d %d %d %d %d):",vi,vj,li,lij,lj);
		if(vi>vj){
			backward=1;
		}

		//Nếu là mở rộng forward thì cập nhật lại maxId bằng vj;
		if(backward!=1){
			DFS_CODE.push(vi,vj,-1,temp[i].lij,temp[i].lj);//xây dựng DFS_CODE forward
			maxId=vj;
		}
		else
		{
			DFS_CODE.push(vi,vj,-1,temp[i].lij,-1);//xây dựng DFS_CODE backward
		}
		currentColEmbedding=vi;
		if(is_min()){ //Nếu DFS_CODE là min thì tìm các graphid chứa embedding của DFS_CODE
			PMS_PRINT("\n This is minDFSCODE\n");
			//1.Tăng Level
			Level++;
			idxLevel=Level-1;

			int *hArrGraphId; //2. need free //Mảng chứa các graphID có embedding của DFS_Code.
			int noElemhArrGraphId=0;
			FUNCHECK(status =getGraphIdContainEmbeddingv2(temp[i],hArrGraphId,noElemhArrGraphId,dArrEXT,noElemdArrEXT));
			if (status!=0){
				PMS_PRINT("\n\n getGraphIdContainEmbedding() in Mining() failed");
				goto Error;
			}
			////In nội dung mảng hArrGraphId
			/*PMS_PRINT("\n ************** hArrGraphId ****************\n");
			for (int j = 0; j < noElemhArrGraphId; j++)
			{
				PMS_PRINT("%d ",hArrGraphId[j]);
			}*/

			report(hArrGraphId,noElemhArrGraphId,hArrSupport[i]); //ghi kết quả vào tập tin result.txt
			write_minDFS_CODE(DFS_CODE);
			free(hArrGraphId);
			//FUNCHECK(status=extendEmbedding(temp[i],idxExt)); //mở rộng embedding. 
			//if(status!=0){
			//	goto Error;
			//}

			//Nếu là backward thì có các mở rộng embedding khác với forward
			if(backward!=1){
				//thực hiện mở rộng forward: là hoạt động thêm một embedding column
				//FUNCHECK(status=extendEmbedding(temp[i],idxExt)); //mở rộng embedding. 
				//if(status!=0){
				//	goto Error;
				//}
				FUNCHECK(status=extendEmbeddingv2(temp[i],dArrEXT,noElemdArrEXT)); //mở rộng embedding forward.
				if(status!=0){
					goto Error;
				}
			}
			else
			{
				//Thực hiện mở rộng backward: là hoạt động giữa là các row pointer backward của embedding column và loại bỏ các forward row pointer.
				//Thêm cột backward embedding khi đã có bwembedding
				int lastColEmbedding = hEmbedding.size()-1;
				if (hEmbedding.at(lastColEmbedding).hBackwardEmbedding.size()>0){//sai, vì currentColEmbedding
					int idxLastBWEC = hEmbedding.at(lastColEmbedding).hBackwardEmbedding.size()-1;
					FUNCHECK(status = extendEmbeddingBW2(temp[i],hEmbedding.at(lastColEmbedding),hEmbedding.at(lastColEmbedding).hBackwardEmbedding.at(idxLastBWEC).dArrEmbedding,dArrEXT,noElemdArrEXT));
				}
				else
				{//Thêm cột backward embedding mới
					FUNCHECK(status = extendEmbeddingBW(temp[i],hEmbedding.at(lastColEmbedding),dArrEXT,noElemdArrEXT));
				}

				if(status!=0){
					goto Error;
				}
			}
			FUNCHECK(FSMiningv4(backward)); //Gọi FSMining.( Hàm này thực hiện theo tuần tự (1. Find Extension -> 2. Extract UniEdge -> 3.Compute & CHECK Support -> 4. CHECK minDFS_CODE -> 5. BuildEmbedding -> 6.Find RMP -> 1.)
			//Giải phóng embedding column khi đã khai thác xong
			int lastCol = hEmbedding.size()-1;
			int bwEmbeddingSize = hEmbedding.at(lastCol).hBackwardEmbedding.size();
			if(bwEmbeddingSize!=0){
				int lastBWCol= hEmbedding.at(lastCol).hBackwardEmbedding.size()-1;
				CHECK(cudaStatus=cudaFree(hEmbedding.at(lastCol).hBackwardEmbedding.at(lastBWCol).dArrEmbedding)); //phải phóng bộ nhớ backward embedding column trên device được quản lý bởi phần tử của vector
				if(cudaStatus!=cudaSuccess){
					status=-1;
					goto Error;
				}
				hEmbedding.at(lastCol).hBackwardEmbedding.pop_back(); //gỡ bỏ phần tử của vector quản lý backward embedding column trên device.
			}
			else
			{
				CHECK(cudaStatus = cudaFree(hEmbedding.at(lastCol).dArrEmbedding)); //xoá phần tử cuối của Embedding
				if(cudaStatus!=cudaSuccess){
					status=-1;
					goto Error;
				}

				hEmbedding.pop_back();
			}
		}
		else{cout<<endl<<"NOT Satisfy minsup";}
		DFS_CODE.pop(); //Xoá phần tử cuối của DFS_CODE
		if(backward!=1){ //Nếu pop() một forward thì phải giảm maxId
			--maxId;
		}	

	}	
	std::free(temp);
Error:
	return status;
}

int PMS::extendEmbeddingBW2(UniEdge ue,EmbeddingColumn& EC,Embedding *fromdArrEmbedding,EXT* dArrExt,int noElemdArrExt)
{
	int status =0;
	cudaError_t cudaStatus;

	//1. Khởi tạo một mảng <int> có số lượng phần tử bằng với số lượng Embedding gọi là dV và đánh dấu các posRow chứa backward extension
	int *dV=nullptr;
	int noElemdV=hLevelPtrEmbedding.at(objLevel.prevLevel).noElemEmbedding; //ở đây mình dựa vào pointer embedding column nên vẫn đúng. Nhưng cần phải suy nghĩ thêm, nếu dựa vào hEmbedding Column thì phải làm thế nào?
	size_t nBytedV=noElemdV*sizeof(int);
	CHECK(cudaStatus=cudaMalloc((void**)&dV,nBytedV));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	else
	{
		cudaMemset(dV,0,nBytedV);
	}
	dim3 block(blocksize);
	dim3 grid((noElemdArrExt+block.x-1)/block.x);
	kernelExtractRowFromEXT<<<grid,block>>>(dArrExt,noElemdArrExt,dV,ue.vj);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	/*PMS_PRINT("\n********dV**********\n");
	FUNCHECK(status=displayDeviceArr(dV,noElemdV));
	if(status!=0)
	{
		goto Error;
	}*/
	//1.1 scan dV để biết kích thước của backward column embedding
	int *dVScanResult = nullptr;
	CHECK(cudaStatus=cudaMalloc((void**)&dVScanResult,noElemdV*sizeof(int)));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus=myScanV(dV,noElemdV,dVScanResult));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	int noElemBW=0;
	CHECK(cudaStatus=getSizeBaseOnScanResult(dV,dVScanResult,noElemdV,noElemBW));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	//2.Dựa vào dV để trích các embedding chứa backward extension sang một embedding column mới.
	//2.1 Tạo mới phần tử vector hBackwardEmbedding
	//int currentSize = hEmbedding.at(currentColEmbedding).hBackwardEmbedding.size();
	//int newsize = currentSize+1;
	//hEmbedding.at(currentColEmbedding).hBackwardEmbedding.resize(newsize);
	//hEmbedding.at(currentColEmbedding).hBackwardEmbedding.at(currentSize).noElem = noElemBW; //update noElem backward embedding
	//hEmbedding.at(currentColEmbedding).hBackwardEmbedding.at(currentSize).prevCol = hEmbedding.at(currentColEmbedding).prevCol;//update preCol BW
	//CHECK(cudaStatus=cudaMalloc((void**)&hEmbedding.at(currentColEmbedding).hBackwardEmbedding.at(currentSize).dArrEmbedding,noElemBW*sizeof(Embedding)));
	//if(cudaStatus!=cudaSuccess){
	//	status = -1;
	//	goto Error;
	//}

	int currentSize = EC.hBackwardEmbedding.size();
	int newsize = currentSize+1;
	EC.hBackwardEmbedding.resize(newsize);
	EC.hBackwardEmbedding.at(currentSize).noElem = noElemBW; //update noElem backward embedding
	EC.hBackwardEmbedding.at(currentSize).prevCol = EC.prevCol;//update preCol BW
	CHECK(cudaStatus=cudaMalloc((void**)&EC.hBackwardEmbedding.at(currentSize).dArrEmbedding,noElemBW*sizeof(Embedding)));
	if(cudaStatus!=cudaSuccess)
	{
		status = -1;
		goto Error;
	}

	dim3 blocka(blocksize);
	dim3 grida((noElemdV + blocka.x -1)/blocka.x);
	kernelExtractBWEmbeddingRow<<<grida,blocka>>>(EC.hBackwardEmbedding.at(currentSize).dArrEmbedding,dV,dVScanResult,noElemdV,fromdArrEmbedding);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus=cudaGetLastError());
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	/*PMS_PRINT("\n ************* dArrBWEmbeddingCol ***********\n");
	FUNCHECK(status=displayBWEmbeddingCol(EC.hBackwardEmbedding.at(currentSize).dArrEmbedding,EC.hBackwardEmbedding.at(currentSize).noElem));
	if(status!=0)
	{
		goto Error;
	}*/

	//free memory
	CHECK(cudaStatus=cudaFree(dVScanResult));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

Error:
	return cudaStatus;
}

int PMS::extendEmbeddingBW(UniEdge ue,EmbeddingColumn& EC,EXT* dArrExt,int noElemdArrExt)
{
	int status =0;
	cudaError_t cudaStatus;

	//1. Khởi tạo một mảng <int> có số lượng phần tử bằng với số lượng Embedding gọi là dV và đánh dấu các posRow chứa backward extension
	int *dV=nullptr;
	int noElemdV=hLevelPtrEmbedding.at(objLevel.prevLevel).noElemEmbedding; //ở đây mình dựa vào pointer embedding column nên vẫn đúng. Nhưng cần phải suy nghĩ thêm, nếu dựa vào hEmbedding Column thì phải làm thế nào?
	size_t nBytedV=noElemdV*sizeof(int);
	CHECK(cudaStatus=cudaMalloc((void**)&dV,nBytedV));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	else
	{
		cudaMemset(dV,0,nBytedV);
	}
	dim3 block(blocksize);
	dim3 grid((noElemdArrExt+block.x-1)/block.x);
	kernelExtractRowFromEXT<<<grid,block>>>(dArrExt,noElemdArrExt,dV,ue.vj);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess)
	{
		status = -1;
		goto Error;
	}

	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	/*PMS_PRINT("\n********dV**********\n");
	FUNCHECK(status=displayDeviceArr(dV,noElemdV));
	if(status!=0)
	{
		goto Error;
	}*/
	//1.1 scan dV để biết kích thước của backward column embedding
	int *dVScanResult = nullptr;
	CHECK(cudaStatus=cudaMalloc((void**)&dVScanResult,noElemdV*sizeof(int)));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus=myScanV(dV,noElemdV,dVScanResult));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	int noElemBW=0;
	CHECK(cudaStatus=getSizeBaseOnScanResult(dV,dVScanResult,noElemdV,noElemBW));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	//2.Dựa vào dV để trích các embedding chứa backward extension sang một embedding column mới.
	//2.1 Tạo mới phần tử vector hBackwardEmbedding
	//int currentSize = hEmbedding.at(currentColEmbedding).hBackwardEmbedding.size();
	//int newsize = currentSize+1;
	//hEmbedding.at(currentColEmbedding).hBackwardEmbedding.resize(newsize);
	//hEmbedding.at(currentColEmbedding).hBackwardEmbedding.at(currentSize).noElem = noElemBW; //update noElem backward embedding
	//hEmbedding.at(currentColEmbedding).hBackwardEmbedding.at(currentSize).prevCol = hEmbedding.at(currentColEmbedding).prevCol;//update preCol BW
	//CHECK(cudaStatus=cudaMalloc((void**)&hEmbedding.at(currentColEmbedding).hBackwardEmbedding.at(currentSize).dArrEmbedding,noElemBW*sizeof(Embedding)));
	//if(cudaStatus!=cudaSuccess){
	//	status = -1;
	//	goto Error;
	//}

	int currentSize = EC.hBackwardEmbedding.size();
	int newsize = currentSize+1;
	EC.hBackwardEmbedding.resize(newsize);
	EC.hBackwardEmbedding.at(currentSize).noElem = noElemBW; //update noElem backward embedding
	EC.hBackwardEmbedding.at(currentSize).prevCol = EC.prevCol;//update preCol BW //Chưa có thông tin nào ở BW embedding column cho biết cạnh backward trong column kết nôí với đỉnh nào trên RMP.
	CHECK(cudaStatus=cudaMalloc((void**)&EC.hBackwardEmbedding.at(currentSize).dArrEmbedding,noElemBW*sizeof(Embedding)));
	if(cudaStatus!=cudaSuccess)
	{
		status = -1;
		goto Error;
	}

	dim3 blocka(blocksize);
	dim3 grida((noElemdV + blocka.x -1)/blocka.x);
	kernelExtractBWEmbeddingRow<<<grida,blocka>>>(EC.hBackwardEmbedding.at(currentSize).dArrEmbedding,dV,dVScanResult,noElemdV,EC.dArrEmbedding);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus=cudaGetLastError());
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	//Hiển thị embedding column backward
	PMS_PRINT("\n ************* dArrBWEmbeddingCol ***********\n");
	FUNCHECK(status=displayBWEmbeddingCol(EC.hBackwardEmbedding.at(currentSize).dArrEmbedding,EC.hBackwardEmbedding.at(currentSize).noElem));
	if(status!=0)
	{
		goto Error;
	}

	//free memory
	CHECK(cudaStatus=cudaFree(dVScanResult));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

Error:
	return cudaStatus;
}

int PMS::displayBWEmbeddingCol(Embedding* dArrBWEmbeddingCol,int noElem){
	int status =0;
	cudaError_t cudaStatus;
	Embedding *hArrBWEmbeddingCol = (Embedding*)malloc(sizeof(Embedding)*noElem);
	if(hArrBWEmbeddingCol==NULL){
		status=-1;
		PMS_PRINT("\n malloc hArrBWEmbeddingCol failed\n");
		goto Error;
	}

	CHECK(cudaStatus=cudaMemcpy(hArrBWEmbeddingCol,dArrBWEmbeddingCol,noElem*sizeof(Embedding),cudaMemcpyDeviceToHost));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	for (int i = 0; i < noElem; i++)
	{
		PMS_PRINT("\n %d: (idx vid):(%d %d)",i,hArrBWEmbeddingCol[i].idx,hArrBWEmbeddingCol[i].vid);
	}

	free(hArrBWEmbeddingCol);

Error:
	return status;
}


//inuse
__global__ void kernelExtractBWEmbeddingRow(Embedding* dArrBWEmbedding,int *dV, \
											int *dVScanResult,int noElemdV,Embedding *dArrEmbedding){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElemdV){
		if(dV[i]==1){
			dArrBWEmbedding[dVScanResult[i]].idx = dArrEmbedding[i].idx;
			dArrBWEmbedding[dVScanResult[i]].vid = dArrEmbedding[i].vid;
		}
	}
}



//Đánh dấu những dòng nào (embeddings nào) trong Embedding column có mở rộng backward.
__global__ void	kernelExtractRowFromEXT(EXT *dArrExt,int noElemdArrExt,int *dV,int vj)
{
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



__global__ void kernelMarkExtension(const Extension *d_ValidExtension,int noElem_d_ValidExtension,int *dV,int li,int lij,int lj){
	int i= blockIdx.x*blockDim.x + threadIdx.x;
	if(i<noElem_d_ValidExtension){
		if(d_ValidExtension[i].li==li && d_ValidExtension[i].lij==lij && d_ValidExtension[i].lj==lj){
			dV[i]=1;
		}		
	}
}


__global__ void kernelMarkExtension_pure(const EXT *d_ValidExtension,int noElem_d_ValidExtension,int *dV,int li,int lij,int lj)
{
	int i= blockIdx.x*blockDim.x + threadIdx.x;
	if(i<noElem_d_ValidExtension){
		if(d_ValidExtension[i].li==li && d_ValidExtension[i].lij==lij && d_ValidExtension[i].lj==lj){
			dV[i]=1;
		}		
	}
}

//Đánh dấu các mở rộng có cạnh băng (li,lij,lj)
__global__ void kernelMarkEXT(const EXT *d_ValidExtension,int noElem_d_ValidExtension,int *dV,int vi, int vj, int li,int lij,int lj){
	int i= blockIdx.x*blockDim.x + threadIdx.x;
	if(i<noElem_d_ValidExtension){
		if(d_ValidExtension[i].li==li && 
			d_ValidExtension[i].lij==lij && 
			d_ValidExtension[i].lj==lj &&
			d_ValidExtension[i].vi==vi&&
			d_ValidExtension[i].vj==vj){
			dV[i]=1;
		}		
	}
}

//inuse
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



__global__ void	kernelPrintEmbedding(Embedding *dArrEmbedding,int noElem){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElem){
		PMS_PRINT("\n Thread:%d address:%p (idx vid):(%d %d)",i,dArrEmbedding,dArrEmbedding[i].idx,dArrEmbedding[i].vid);
	}
}
//inuse
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

//Mở rộng embedding column forward
int PMS::extendEmbedding(UniEdge ue,int idxExt){
	int status=0;
	cudaError_t cudaStatus;
	//Lấy nhãn cạnh
	
	int li,lij,lj;
	li=ue.li;
	lij=ue.lij;
	lj=ue.lj;
	int vi=ue.vi;
	int vj=ue.vj;
	//Lấy số lượng embedding columns hiện có để tạo embedding column mới. Đồng thời lấy chỉ số truy xuất embedding column mới.
	int currentSize= hEmbedding.size();
	int newSize = currentSize+1;
	int lastEC =newSize-1; //lastEC is last Embedding Column or index of last element hEmbedding vector.

	//Tạo embedding column mới
	hEmbedding.resize(newSize); //Mỗi phần tử của Vector sẽ quản lý 1 dArrEmbedding trên device. Khi cần thiết có thể tập hợp chúng lại thành 1 mảng trên device.
	//hEmbedding.at(0).noElem;

	int *dV=nullptr;
	int noElemdV = hLevelEXT.at(idxLevel).vE.at(idxExt).noElem;
	CHECK(cudaStatus=cudaMalloc((void**)&dV, sizeof(int)*noElemdV));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus=cudaMemset(dV,0,sizeof(int)*noElemdV));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	FUNCHECK(status=displayDeviceEXT( hLevelEXT.at(idxLevel).vE.at(idxExt).dArrExt, hLevelEXT.at(idxLevel).vE.at(idxExt).noElem));
	if(status!=0){
		goto Error;
	}


	dim3 block(blocksize);
	dim3 grid((noElemdV+block.x-1)/block.x);

	//Đánh dấu các mở rộng cần rút trích để xây dựng embedding column
	kernelMarkEXT<<<grid,block>>>(hLevelEXT.at(idxLevel).vE.at(idxExt).dArrExt,noElemdV,dV,vi,vj,li,lij,lj);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		status = -1;
		fprintf(stderr,"\n kernelMarkExtension failed",cudaStatus);
		goto Error;
	}

	int* dVScanResult;
	CHECK(cudaStatus=cudaMalloc((int**)&dVScanResult,noElemdV*sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus=cudaMemset(dVScanResult,0,noElemdV*sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	//CHECK(scanV(dV,noElemdV,dVScanResult));
	CHECK(cudaStatus=myScanV(dV,noElemdV,dVScanResult));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}


	//Tính số lượng embedding của embedding column mới.
	int noElemOfdArEmbedding=0;
	CHECK(cudaStatus=getSizeBaseOnScanResult(dV,dVScanResult,noElemdV,noElemOfdArEmbedding));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	//Cập nhật số lượng Embedding của embedding column mới.
	hEmbedding.at(lastEC).noElem=noElemOfdArEmbedding;
	//Cấp phát bộ nhớ trên device cho embedding column mới tương ứng với số lượng embedding đã tính ở trên.
	CHECK(cudaStatus=cudaMalloc((void**)&hEmbedding.at(lastEC).dArrEmbedding,noElemOfdArEmbedding*sizeof(Embedding)));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	//Gán giá trị cho Embedding column mới.
	kernelSetValueForEmbeddingColumn<<<grid,block>>>(hLevelEXT.at(idxLevel).vE.at(idxExt).dArrExt,hLevelEXT.at(idxLevel).vE.at(idxExt).noElem,hEmbedding.at(lastEC).dArrEmbedding,dV,dVScanResult);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus !=cudaSuccess){
		fprintf(stderr,"\n kernelSetValueForEmbeddingColumn in failed",cudaStatus);
		status = -1;
		goto Error;
	}
	//Cập nhật prevCol của embedding column mới. 
	hEmbedding.at(lastEC).prevCol=currentColEmbedding; 
	//Hiển thị nội dung của embedding column mới.
	for (int i = 0; i < hEmbedding.size(); i++)
	{
		PMS_PRINT("\n\n Q[%d] prevCol:%d ",i,hEmbedding.at(i).prevCol);		
		kernelPrintEmbedding<<<1,512>>>(hEmbedding.at(i).dArrEmbedding,hEmbedding.at(i).noElem);
		CHECK(cudaStatus=cudaDeviceSynchronize());
		if(cudaStatus!=cudaSuccess){
			status=-1;
			goto Error;
		}

		CHECK(cudaStatus = cudaGetLastError());
		if(cudaStatus!=cudaSuccess){
			status =-1;
			PMS_PRINT("kernelPrintEmbedding failed");
			goto Error;
		}
	}

	CHECK(cudaStatus=cudaFree(dV));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus=cudaFree(dVScanResult));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
Error:
	return status;
}

//Mở rộng Embedding trên host và device.
int PMS::ExtendEmbedding(UniEdge &ue,EXT *&dArrExt,int &noElemdArrExt)
{
	int status = 0;
	cudaError_t cudaStatus = cudaSuccess;

	//Mở rộng khi ue là một backward unique edge.
	if (ue.vi > ue.vj)
	{
		//Thực hiện mở rộng backward: là hoạt động giữa là các row pointer backward của embedding column và loại bỏ các forward row pointer.
		//Thêm cột backward embedding khi đã có bwembedding
		int lastColEmbedding = hEmbedding.size()-1;
		if (hEmbedding.at(lastColEmbedding).hBackwardEmbedding.size()>0)
		{
			int idxLastBWEC = hEmbedding.at(lastColEmbedding).hBackwardEmbedding.size()-1;
			FUNCHECK(status = extendEmbeddingBW2( \
				ue, \
				hEmbedding.at(lastColEmbedding), \
				hEmbedding.at(lastColEmbedding).hBackwardEmbedding.at(idxLastBWEC).dArrEmbedding, \
				dArrExt, \
				noElemdArrExt \
				));
		}
		else
		{//Thêm cột backward embedding mới khi chưa có embedding nào được khai thác.
			FUNCHECK(status = extendEmbeddingBW( \
				ue, \
				hEmbedding.at(lastColEmbedding), \
				dArrExt, \
				noElemdArrExt \
				));
		}
		if(status!=0){goto Error;}
	}
	//Mở rộng khi ue là một forward unique edge.
	else
	{
		FUNCHECK(status=extendEmbeddingv2( \
			ue, \
			dArrExt, \
			noElemdArrExt \
			));
		if(status!=0){goto Error;}
	}
Error:
	return status;
}

//Trích các frequent embeddings từ dArrExt xây dựng embedding column.
int PMS::extendEmbeddingv2(UniEdge ue,EXT *dArrExt,int noElemdArrExt)
{
	int li,lij,lj;
	int vi=ue.vi;
	int vj=ue.vj;
	li=ue.li;
	lij=ue.lij;
	lj=ue.lj;
	int status =0;
	cudaError_t cudaStatus;
	//1. Cần biết có bao nhiêu embeddings của new embedding columns
	int currentSize= hEmbedding.size(); //Số lượng embedding columns hiện tại
	int newSize = currentSize+1; //Số lượng embedding mở rộng mới
	int lastEC =newSize-1; //lastEC is last Embedding Column or index of last element hEmbedding vector.

	hEmbedding.resize(newSize); //Mỗi phần tử của Vector sẽ quản lý 1 dArrEmbedding trên device. Khi cần thiết có thể tập hợp chúng lại thành 1 mảng trên device.
	//hEmbedding.at(0).noElem;

	int *dV=nullptr; //1. need cudaFree
	int noElemdV = noElemdArrExt;
	CHECK(cudaStatus=cudaMalloc((void**)&dV, sizeof(int)*noElemdV));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus=cudaMemset(dV,0,sizeof(int)*noElemdV));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	/*PMS_PRINT("\n ************** dArrExt **************\n");
	FUNCHECK(status=displayDeviceEXT(dArrExt, noElemdArrExt));
	if(status!=0)
	{
		goto Error;
	}*/


	dim3 block(blocksize);
	dim3 grid((noElemdV+block.x-1)/block.x);


	kernelMarkEXT<<<grid,block>>>(dArrExt,noElemdV,dV,vi,vj,li,lij,lj);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess)
	{
		status = -1;
		goto Error;
	}

	int* dVScanResult; //2. need cudaFree
	CHECK(cudaStatus=cudaMalloc((int**)&dVScanResult,noElemdV*sizeof(int)));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus=cudaMemset(dVScanResult,0,noElemdV*sizeof(int)));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	//CHECK(scanV(dV,noElemdV,dVScanResult));
	CHECK(cudaStatus=myScanV(dV,noElemdV,dVScanResult));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}



	int noElemOfdArEmbedding=0;
	CHECK(cudaStatus = getSizeBaseOnScanResult(dV,dVScanResult,noElemdV,noElemOfdArEmbedding));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	//Cập nhật số lượng embedding ở cột mới thêm.
	hEmbedding.at(lastEC).noElem=noElemOfdArEmbedding; 

	//Cấp phát bộ nhớ trên device cho embedding column mới.
	CHECK(cudaStatus=cudaMalloc((void**)&hEmbedding.at(lastEC).dArrEmbedding,noElemOfdArEmbedding*sizeof(Embedding))); 
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	kernelSetValueForEmbeddingColumn<<<grid,block>>>(dArrExt,noElemdArrExt,hEmbedding.at(lastEC).dArrEmbedding,dV,dVScanResult);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus !=cudaSuccess)
	{
		status = -1;
		goto Error;
	}
	//Cập nhật previous column, phải có prevCol để tìm thông tin right most path. Có cách nào để
	//tìm right most path mà không cần đến thông tin prevCol hay không?
	//Hiển thị các embedding column
	hEmbedding.at(lastEC).prevCol=currentColEmbedding;
	//for (int i = 0; i < hEmbedding.size(); i++)
	//{
	//	//PMS_PRINT("\n\n Q[%d] prevCol:%d ",i,hEmbedding.at(i).prevCol);
	//	kernelPrintEmbedding<<<1,512>>>(hEmbedding.at(i).dArrEmbedding,hEmbedding.at(i).noElem);
	//	CHECK(cudaStatus=cudaDeviceSynchronize());
	//	if(cudaStatus!=cudaSuccess)
	//	{
	//		status=-1;
	//		goto Error;
	//	}

	//	CHECK(cudaStatus = cudaGetLastError());
	//	if(cudaStatus!=cudaSuccess)
	//	{
	//		status =-1;
	//		PMS_PRINT("kernelPrintEmbedding failed");
	//		goto Error;
	//	}
	//}

	//Free memory
	CHECK(cudaStatus=cudaFree(dV)); //1.
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus=cudaFree(dVScanResult)); //2.
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

Error:
	return status;
}
/*
//Xây dựng Embedding ban đầu
int PMS::buildFirstEmbedding(UniEdge ue){
int li,lij,lj;
li=ue.li;
lij=ue.lij;
lj=ue.lj;
int status =0;
cudaError_t cudaStatus;
hEmbedding.resize(2); //Mỗi phần tử của Vector sẽ quản lý 1 dArrEmbedding trên device. Khi cần thiết có thể tập hợp chúng lại thành 1 mảng trên device.
hEmbedding.at(0).noElem;

int *dV=nullptr;
int noElemdV = hValidExtension.at(0).noElem; //các mở rộng hợp lệ của cạnh đầu tiên được quản lý bởi hValidExtension.
CHECK(cudaStatus = cudaMalloc((void**)&dV, sizeof(int)*noElemdV));
if(cudaStatus!=cudaSuccess){
status=-1;
PMS_PRINT("\n cudaMalloc dV failed\n");
goto Error;
}

CHECK(cudaStatus = cudaMemset(dV,0,sizeof(int)*noElemdV));
if(cudaStatus!=cudaSuccess)
{
status=-1;
goto Error;
}

dim3 block(blocksize);
dim3 grid((noElemdV+block.x-1)/block.x);

//kernelPrintExtention<<<1,512>>>(hValidExtension.at(0).dExtension,hValidExtension.at(0).noElem);
//cudaDeviceSynchronize();
//CHECK(cudaGetLastError());
//if(cudaGetLastError() !=cudaSuccess){
//	PMS_PRINT("Error here");
//	goto Error;
//}

kernelMarkExtension<<<grid,block>>>(hValidExtension.at(0).dExtension,noElemdV,dV,li,lij,lj);
CHECK(cudaStatus=cudaDeviceSynchronize());
if(cudaStatus!=cudaSuccess)
{
status=-1;
goto Error;
}

CHECK(cudaStatus = cudaGetLastError());
//CHECK(cudaStatus);
if(cudaStatus!=cudaSuccess){
status = -1;
fprintf(stderr,"\n kernelMarkExtension failed",cudaStatus);
goto Error;
}

int* dVScanResult;
CHECK(cudaStatus=cudaMalloc((int**)&dVScanResult,noElemdV*sizeof(int)));
if(cudaStatus!=cudaSuccess){
status=-1;
PMS_PRINT("\n cudamalloc dVScanResult failed\n");
goto Error;
}

CHECK(cudaStatus=cudaMemset(dVScanResult,0,noElemdV*sizeof(int)));
if(cudaStatus!=cudaSuccess){
status=-1;
PMS_PRINT("\n cudamemset dVScanResult failed\n");
goto Error;
}
//CHECK(scanV(dV,noElemdV,dVScanResult));
FUNCHECK(status=myScanV(dV,noElemdV,dVScanResult));
if(status!=0)
{
goto Error;
}
int noElemOfdArEmbedding=0;

CHECK(cudaStatus=getSizeBaseOnScanResult(dV,dVScanResult,noElemdV,noElemOfdArEmbedding));
if(cudaStatus!=cudaSuccess)
{
status=-1;
goto Error;
}

hEmbedding.at(0).noElem=hEmbedding.at(1).noElem=noElemOfdArEmbedding;

CHECK(cudaStatus=cudaMalloc((void**)&hEmbedding.at(0).dArrEmbedding,noElemOfdArEmbedding*sizeof(Embedding)));
if(cudaStatus!=cudaSuccess)
{
status=-1;
PMS_PRINT("\n cudaMalloc hEmbedding.at(0).dArrEmbedding failed\n");
goto Error;
}

CHECK(cudaStatus=cudaMalloc((void**)&hEmbedding.at(1).dArrEmbedding,noElemOfdArEmbedding*sizeof(Embedding)));
if(cudaStatus!=cudaSuccess)
{
status=-1;
PMS_PRINT("\n cudaMalloc hEmbedding.at(1).dArrEmbedding failed\n");
goto Error;
}
//kernelSetValueForFirstTwoEmbeddingColumn<<<grid,block>>>(hValidExtension.at(0).dExtension,hValidExtension.at(0).noElem,hEmbedding.at(0).dArrEmbedding,hEmbedding.at(1).dArrEmbedding,dVScanResult,li,lij,lj);
CHECK(cudaStatus=cudaDeviceSynchronize());
if(cudaStatus!=cudaSuccess)
{
status=-1;
goto Error;
}

CHECK(cudaStatus = cudaGetLastError());
if(cudaStatus !=cudaSuccess)
{
fprintf(stderr,"\n kernelSetValueForFirstTwoEmbeddingColumn in failed",cudaStatus);
status = -1;
goto Error;
}

hEmbedding.at(0).prevCol=-1;
hEmbedding.at(1).prevCol=0;

for (int i = 0; i < hEmbedding.size(); i++)
{
PMS_PRINT("\n\n Q[%d] prevCol:%d ",i,hEmbedding.at(i).prevCol);		
kernelPrintEmbedding<<<1,512>>>(hEmbedding.at(i).dArrEmbedding,hEmbedding.at(i).noElem);
CHECK(cudaStatus=cudaDeviceSynchronize());
if(cudaStatus!=cudaSuccess)
{
status=-1;
PMS_PRINT("kernelPrintEmbedding cudaDeviceSynchronize failed");
goto Error;
}

CHECK(cudaStatus = cudaGetLastError());
if(cudaStatus!=cudaSuccess)
{
status =-1;
PMS_PRINT("kernelPrintEmbedding failed");
goto Error;
}
}
Error:
return status;
}
*/


int PMS::buildEmbedding_pure(UniEdge &ue)
{
	int li,lij,lj;
	li=ue.li;
	lij=ue.lij;
	lj=ue.lj;

	int status =0;
	cudaError_t cudaStatus;
	//Mỗi phần tử của Vector sẽ quản lý 1 dArrEmbedding trên device. 
	//Khi cần thiết có thể tập hợp chúng lại thành 1 mảng trên device.
	hEmbedding.resize(2);
	hEmbedding.at(0).noElem;

	int *dV=nullptr;
	//Các mở rộng hợp lệ của cạnh đầu tiên được quản lý bởi hValidExtension.
	//int noElemdV = hValidExtension.at(0).noElem;
	int noElemdV = hLevelEXT.at(objLevel.Level).vE.at(0).noElem;
	CHECK(cudaStatus = cudaMalloc((void**)&dV, sizeof(int)*noElemdV));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaMemset(dV,0,sizeof(int)*noElemdV));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	dim3 block(blocksize);
	dim3 grid((noElemdV+block.x-1)/block.x);

	//kernelPrintExtention<<<1,512>>>(hValidExtension.at(0).dExtension,hValidExtension.at(0).noElem);
	//cudaDeviceSynchronize();
	//CHECK(cudaGetLastError());
	//if(cudaGetLastError() !=cudaSuccess){
	//	PMS_PRINT("Error here");
	//	goto Error;
	//}

	/*kernelMarkExtension<<<grid,block>>>(hValidExtension.at(0).dExtension,noElemdV,dV,li,lij,lj);*/
	kernelMarkExtension_pure<<<grid,block>>>(hLevelEXT.at(objLevel.Level).vE.at(0).dArrExt,noElemdV,dV,li,lij,lj);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaGetLastError());
	//CHECK(cudaStatus);
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}

	int* dVScanResult;
	CHECK(cudaStatus=cudaMalloc((int**)&dVScanResult,noElemdV*sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus=cudaMemset(dVScanResult,0,noElemdV*sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	//CHECK(scanV(dV,noElemdV,dVScanResult));
	FUNCHECK(status=myScanV(dV,noElemdV,dVScanResult));
	if(status!=0)
	{
		goto Error;
	}
	int noElemOfdArEmbedding=0;

	CHECK(cudaStatus=getSizeBaseOnScanResult(dV,dVScanResult,noElemdV,noElemOfdArEmbedding));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	hEmbedding.at(0).noElem=hEmbedding.at(1).noElem=noElemOfdArEmbedding;

	CHECK(cudaStatus=cudaMalloc((void**)&hEmbedding.at(0).dArrEmbedding,noElemOfdArEmbedding*sizeof(Embedding)));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		PMS_PRINT("\n cudaMalloc hEmbedding.at(0).dArrEmbedding failed\n");
		goto Error;
	}

	CHECK(cudaStatus=cudaMalloc((void**)&hEmbedding.at(1).dArrEmbedding,noElemOfdArEmbedding*sizeof(Embedding)));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		PMS_PRINT("\n cudaMalloc hEmbedding.at(1).dArrEmbedding failed\n");
		goto Error;
	}
	//kernelSetValueForFirstTwoEmbeddingColumn<<<grid,block>>>(hValidExtension.at(0).dExtension,hValidExtension.at(0).noElem,hEmbedding.at(0).dArrEmbedding,hEmbedding.at(1).dArrEmbedding,dVScanResult,li,lij,lj);
	kernelSetValueForFirstTwoEmbeddingColumn<<<grid,block>>>( \
		hLevelEXT.at(objLevel.Level).vE.at(0).dArrExt, \
		hLevelEXT.at(objLevel.Level).vE.at(0).noElem, \
		hEmbedding.at(0).dArrEmbedding, \
		hEmbedding.at(1).dArrEmbedding, \
		dVScanResult, \
		li,lij,lj);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus !=cudaSuccess)
	{
		status = -1;
		goto Error;
	}
	//Gán giá trị prevCol cho các embedding columns. 
	hEmbedding.at(0).prevCol=-1;
	hEmbedding.at(1).prevCol=0;
	//Duyệt và hiển thị nội dung của các embedding columns
	//for (int i = 0; i < hEmbedding.size(); i++)
	//{
	//	PMS_PRINT("\n\n Q[%d] prevCol:%d ",i,hEmbedding.at(i).prevCol);		
	//	kernelPrintEmbedding<<<1,512>>>(hEmbedding.at(i).dArrEmbedding,hEmbedding.at(i).noElem);
	//	CHECK(cudaStatus=cudaDeviceSynchronize());
	//	if(cudaStatus!=cudaSuccess)
	//	{
	//		status=-1;
	//		PMS_PRINT("kernelPrintEmbedding cudaDeviceSynchronize failed");
	//		goto Error;
	//	}

	//	CHECK(cudaStatus = cudaGetLastError());
	//	if(cudaStatus!=cudaSuccess)
	//	{
	//		status =-1;
	//		PMS_PRINT("kernelPrintEmbedding failed");
	//		goto Error;
	//	}
	//}

#pragma region "build dArrPointerEmbedding on device in buildEmbedding_pure"
	hLevelPtrEmbedding.resize(1); //Tạo level ban đầu (level 0). Mỗi level lưu trữ nhiều embedding column.
	hLevelPtrEmbedding.at(objLevel.Level).noElem=hEmbedding.size(); //Cập nhật số lượng embedding columns ở Level này
	hLevelPtrEmbedding.at(objLevel.Level).noElemEmbedding= hEmbedding.at(hEmbedding.size()-1).noElem; //Cập nhật số lượng embedding	

	//Cấp phát bộ nhớ cho dArrPointerEmbedding, đây là con trỏ cấp 2 trên device, 
	//trỏ đến bộ embedding column trên device (con trỏ cấp 1 trên device).
	CHECK(cudaStatus = cudaMalloc(
									(void**)&hLevelPtrEmbedding.at(0).dArrPointerEmbedding, \
									hEmbedding.size()*sizeof(Embedding**) \
								)
		); //Cấp phát bộ nhớ cho mảng dArrPointerEmbedding tại Level tương ứng.
	if(cudaStatus!=cudaSuccess){
		status = -1;
		PMS_PRINT("\n cudaMalloc dArrPointerEmbedding failed()");
		goto Error;
	}
	//Duyệt qua hEmbedding để lấy địa chỉ của các embedding column rồi lưu vào dArrPointerEmbedding của hLevelPtrEmbedding
	for (int i = 0; i < hEmbedding.size(); i++)
	{		
		kernelGetPointerdArrEmbedding<<<1,1>>>( \
			hEmbedding.at(i).dArrEmbedding, \
			hLevelPtrEmbedding.at(objLevel.Level).dArrPointerEmbedding, \
			i); //Mỗi phần tử của mảng dArrPointerEmbedding chứa địa chỉ của dArrEmbedding
	}
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		status = -1;
		PMS_PRINT("\n kernelGetPointerdArrEmbedding failed");
		goto Error;
	}
	//Hiển thị nội dung của dArrPointerEmbedding (địa chỉ của embedding columns trên device).
	//FUNCHECK(status=displaydArrPointerEmbedding(hLevelPtrEmbedding.at(objLevel.Level).dArrPointerEmbedding,hLevelPtrEmbedding.at(objLevel.Level).noElem,hLevelPtrEmbedding.at(objLevel.Level).noElemEmbedding));
	//if(status!=0){
	//	goto Error;
	//}

	/*std::printf("address of hLevelPtrEmbedding Column: %d",&hLevelPtrEmbedding);
	FUNCHECK(status=displayEmbeddingColumn(hLevelPtrEmbedding));
	std::printf("\nAddress of first EC: %d\n",hEmbedding.at(0).dArrEmbedding);
	std::printf("\nAddress of last EC: %d\n",hEmbedding.at(1).dArrEmbedding);*/
	//FUNCHECK(status=saveEmbeddingColumn(hLevelPtrEmbedding));

#pragma endregion 


	//Giải phóng bộ nhớ
	CHECK(cudaStatus=cudaFree(dV));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus=cudaFree(dVScanResult));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}


Error:
	return status;
}

//inuse
void PMS::removeEmbedding()
{
	if(DFS_CODE.size() == 1)
	{
		removeFirstEmbedding();
		
	}
	else
	{
		//what notes when extending embedding columns?
	}
}
void PMS::removeFirstEmbedding()
{
	hEm.at(1).ReleaseMemory();
	hEm.at(0).ReleaseMemory();
	hEm.pop_back();
	hEm.pop_back();
}
//inuse
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
//inuse
int PMS::buildFirstEmbedding(UniEdge &ue,EXTk &ext,int*&dValid,int*&dIdx)
{
	int status =0;
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
}

//inuse
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
}

//inuse
void PMS::buildBackwardEmbedding(UniEdge& ue,EXTk& ext,int*& dValid,int*& dIdx)
{
	try
	{
		//Trích các row trong Embedding column cuối 
		cout<<endl;
		ue.print();
		cout<<endl;
		ext.show();
		cout<<endl;
		displayDeviceArr(dValid,ext.noElem);
		cout<<endl;
		displayDeviceArr(dIdx,ext.noElem);
		cout<<endl;
		for (int i = 0; i < hEm.size(); i++)
		{
			hEm.at(i).show();
		}

		//1. Khởi tạo một mảng <int> có số lượng phần tử bằng với số lượng Embedding gọi là dV 
		//	và đánh dấu các posRow chứa backward extension
		int *dV=nullptr;
		int noElemdV = 0;
		//Lấy số lượng Embedding dựa vào Embedding Column cuối.
		//noElemEmbedding tuỳ thuộc vào Embedding Colmn đó đã có backward embedding column nào hay chưa?
		EmCol* sourceEmCol= new EmCol;
		if(hEm.back().hBackwardEmbedding.size()>0)
		{
			noElemdV = hEm.back().hBackwardEmbedding.back().noElem;
			sourceEmCol = &hEm.back().hBackwardEmbedding.back();
		}
		else
		{
			noElemdV = hEm.back().noElem;
			sourceEmCol = &hEm.back();
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

		PMS_PRINT("\n********dV**********\n");
		FCHECK(displayDeviceArr(dV,noElemdV));

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
		
		kernelExtractBWEmbeddingRow<<<grida,blocka>>>(hEm.back().hBackwardEmbedding.back().dArrEmbedding, \
			dV,dVScanResult,noElemdV,sourceEmCol->dArrEmbedding);
		CUCHECK(cudaDeviceSynchronize());
		CUCHECK(cudaGetLastError());
		//Cập nhật số lượng phần tử và preCol cho backward EmCol vừa mới thêm vào.
		hEm.back().hBackwardEmbedding.back().noElem = noElemBW;
		hEm.back().hBackwardEmbedding.back().prevCol = sourceEmCol->prevCol;
		PMS_PRINT("\n ************* dArrBWEmbeddingCol ***********\n");
		hEm.back().hBackwardEmbedding.back().show();
		//free memory
		CUCHECK(cudaFree(dVScanResult));
		free(sourceEmCol);
	}
	catch(...)
	{
		FCHECK(-1);
	}
}


//Tìm các đỉnh cần mở rộng thuộc right most path của embedding.
int PMS::findListVerOnRMP()
{
	int status=0;
	cudaError_t cudaStatus;

#pragma region "cudaMalloc for listVer to find listVer On All EmbeddingColumn that belong to RMP"
	//Tìm danh sách các đỉnh thuộc right most path của các embedding
	//Kết quả lưu vào các vector tương ứng
	//int lastCol = hEmbedding.size() - 1; //cột cuối của embedding

	//số lượng phần tử của listVer bằng số lượng đỉnh trên right most path nhân với số lượng embedding
	int noElemListVer= hLevelRMP.at(objLevel.Level).noElem * hLevelPtrEmbedding.at(objLevel.Level).noElemEmbedding;
	//Tạo hListVer để lưu trữ các đỉnh thuộc Embedding column
	hLevelListVerRMP.resize(objLevel.size);

	//cấp phát bộ nhớ cho listVer
	hLevelListVerRMP.at(objLevel.Level).noElem=noElemListVer;
	CHECK(cudaStatus = cudaMalloc((void**)&hLevelListVerRMP.at(objLevel.Level).dListVer,sizeof(int)*noElemListVer));
	if(cudaStatus!=cudaSuccess)
	{
		PMS_PRINT("\n CudaMalloc dListVer failed");
		status =-1;
		goto Error;
	}

#pragma endregion

#pragma region "find listVer from All EmbeddingColumn"

	//Tìm danh sách các đỉnh thuộc right most path ở các cột embedding để thực hiện mở rộng
	dim3 block(blocksize);
	dim3 grid((hLevelPtrEmbedding.at(objLevel.Level).noElemEmbedding + block.x -1)/block.x);
	//Kernel tìm các đỉnh thuộc embedding và lưu chúng vào hListVer
	//int noElemVerOnRMP = hLevelRMP.at(objLevel.Level).noElem; //right most path chứa bao nhiêu đỉnh

	kernelFindListVer<<<grid,block>>>( \
		hLevelPtrEmbedding.at(objLevel.Level).dArrPointerEmbedding, \
		hLevelPtrEmbedding.at(objLevel.Level).noElemEmbedding, \
		hLevelRMP.at(objLevel.Level).dRMP, \
		hLevelRMP.at(objLevel.Level).noElem, \
		hLevelListVerRMP.at(objLevel.Level).dListVer); //tìm listVer
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}
	//hiển thị danh sách đỉnh thuộc right most path cần mở rộng
	//PMS_PRINT("\n\n ********* listVer *********\n");
	//FUNCHECK(status=displayDeviceArr(hLevelListVerRMP.at(objLevel.Level).dListVer,noElemListVer));
	//if(status!=0){
	//	goto Error;
	//}
#pragma endregion
Error:
	return status;
}


//Why do this snippet face the error: Invalid device pointer

//void PMS::displayEmbeddingColumn(EmbeddingColumn ec){
//	PMS_PRINT("\n noElem:%d prevCol:%d",ec.noElem,ec.prevCol);
//	
//	Embedding *hArrEmbeddingt = (Embedding*)malloc(sizeof(Embedding)*ec.noElem);
//	if(hArrEmbeddingt==NULL){
//		PMS_PRINT("\n malloc hArrEmbeddingt in displayEmbeddingColumn() failed");
//		exit(1);
//	}
//
//	CHECK(cudaMemcpy(hArrEmbeddingt,ec.dArrEmbedding,sizeof(Embedding)*ec.noElem,cudaMemcpyDeviceToHost));
//	for (int i = 0; i < ec.noElem; i++)
//	{
//		PMS_PRINT("\n A[%d]: (idx, vid):(%d, %d)",i,hArrEmbeddingt[i].idx,hArrEmbeddingt[i].vid);
//	}
//
//	cudaFree(hArrEmbeddingt);
//}

//Tìm các đỉnh thuộc RMP của mỗi embedding (ngoại từ đỉnh cuối của mỗi embedding)
//Lưu kết quả vào dArrVidOnRMP để phục vụ cho việc kiểm tra xem một mở rộng backward có phải hợp lệ không.
__global__ void kernelFindVidOnRMP(Embedding **dArrPointerEmbedding, \
								   int noElemEmbedding, \
								   int *rmp, \
								   int noElemVerOnRMP, \
								   int *dArrVidOnRMP, \
								   int step)
{
	//Mỗi một Thread sẽ xử lý cho 1 embedding. 
	//(Thread thứ i cho embedding thứ i).
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElemEmbedding)
	{
		int index,end;
		int start=rmp[0]; //Bắt đầu từ embedding column cuối
		//Lưu trữ địa chỉ của embedding column cần xử lý.
		Embedding * dArrEmbedding; 

		//Duyệt qua các đỉnh thuộc RMP của embeddings, ngoại trừ đỉnh cuối.
		for (int k = 0; k < (noElemVerOnRMP-1); ++k) 
		{
			dArrEmbedding = dArrPointerEmbedding[k];
			int index_in_dArrVidOnRMP = i*(noElemVerOnRMP-1)+k;
			dArrVidOnRMP[index_in_dArrVidOnRMP] = dArrEmbedding[i].vid;
			/*
			int prevRow = i;
			// = 0*3+0 = 0;
			//Lưu trữ đỉnh thuộc RMP thứ k của embedding thứ i với mỗi embedding 
			//có số đỉnh RMP cần lấy là (noElemEmbedding-1).
			index = i*step+k; 
			int j;
			k++; //k=1;
			if(k==noElemVerOnRMP) break;

			end = rmp[k]; //Lưu index của embeddings column đầu tiên. //Sao không gán line này ở trước vòng lặp for?
			//Từ cột start sẽ trích ra được vid và prevRow;
			for (j = start; j >end; j--) //Duyệt qua các cột của embedding. Từ embedding column cuối đến column đầu. Mỗi k thì phải lặp lại việc tìm prevRow ==> Cần tối ưu?
			{
				dArrEmbedding = dArrPointerEmbedding[j]; //Lấy địa chỉ của embedding column j.
				prevRow= dArrEmbedding[prevRow].idx; //Lấy index của dòng ở phía trước mà embedding kết nối ở column liền trước.
				//PMS_PRINT("\n j:%d, end:%d, prevRow:%d\n",j,end,prevRow);
			}
			//PMS_PRINT("\n j:%d, end:%d, prevRow:%d\n",j,end,prevRow);
			dArrEmbedding = dArrPointerEmbedding[j]; //j=2
			dArrVidOnRMP[index]=dArrEmbedding[prevRow].vid;
			//prevRow= dArrEmbedding[prevRow].idx; //update row
			//std::PMS_PRINT("\n thread:%d start:%d end:%d index:%d vid:%d",i,start,end,index,dArrVidOnRMP[index]);
			*/

		}
	}
}

__global__ void kernelFindVidOnRMPv2(Embedding **dArrPointerEmbedding,int noElemEmbedding,int *rmp,int noElemVerOnRMP,int *dArrVidOnRMP,int step){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElemEmbedding){	
		int index;
		int start=rmp[0];
		int prevRow = i;
		int end=0;
		Embedding * dArrEmbedding;
		for (int k = 0; k < noElemVerOnRMP-1; )//k<3
		{
			index = i*step+k; // = 0*3+0 = 0
			int j;
			end = rmp[++k]; //rmp[1]=2;
			//Từ cột start sẽ trích ra được vid và prevRow;
			for (j = start; j >end; j--) //start=3, end=2;
			{
				dArrEmbedding = dArrPointerEmbedding[j]; //j=3;
				prevRow= dArrEmbedding[prevRow].idx; //update row
			}
			dArrEmbedding = dArrPointerEmbedding[j]; //j=2
			dArrVidOnRMP[index]=dArrEmbedding[prevRow].vid;
			//prevRow= dArrEmbedding[prevRow].idx; //update row
			//PMS_PRINT("\n thread:%d start:%d end:%d index:%d vid:%d",i,start,end,index,dArrVidOnRMP[index]);
		}

	}
}
//Trích các vid cần mở rộng trên embedding column thuộc RMP.
//Các noElemEmbedding vid của embedding column cuối được lưu ở phía đầu của listVer.
__global__ void kernelFindListVer(Embedding **dArrPointerEmbedding,int noElemEmbedding,int *rmp,int noElemVerOnRMP,int *listVer)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	//Mỗi một thread xử lý 1 embeddings.
	if(i<noElemEmbedding)
	{
		//Duyệt qua các embedding column và trích các mở rộng thuộc embedding column đó để lưu vào listVer.
		//Các đỉnh thuộc embedding column cuối thì được lưu ở đầu của mảng listVer và ngược lại.
		for(int j = 0; j < noElemVerOnRMP; ++j)
		{
			int idxListVer = abs(j-(noElemVerOnRMP-1))*noElemEmbedding + i;
			Embedding *dArrEmbedding = dArrPointerEmbedding[j];
			listVer[idxListVer] = dArrEmbedding[i].vid;
		}
//		int j =0;
//		int posCol = rmp[j]; //bắt đầu từ cột cuối của Embedding.
//		int posRow = i; //Embedding đang xét thuộc dòng posRow.
//		Embedding *dArrEmbedding = dArrPointerEmbedding[posCol]; //Lấy địa chỉ của embedding column dựa vào posCol.
//		int idxListVer = j*noElemEmbedding + i;
//		listVer[idxListVer] = dArrEmbedding[posRow].vid; //Trích vid lưu vào mảng listVer tại vị trí tương ứng.
//loop:
//		j=j+1; //tăng chỉ số j của rmp
//		if(j==noElemVerOnRMP) return;
//		int loopTimes = posCol - rmp[j];
//		for (int k = 0; k < loopTimes; k++)
//		{
//			posRow = dArrEmbedding[posRow].idx;
//			posCol = posCol-1;
//			dArrEmbedding = dArrPointerEmbedding[posCol];
//			//PMS_PRINT("\nThread %d j:%d k:%d posCol:%d posRow:%d",i,j,k,posCol,posRow);
//		}
//		idxListVer = j*noElemEmbedding + i;
//		listVer[idxListVer] = dArrEmbedding[posRow].vid; //Trích vid lưu vào mảng listVer tại vị trí tương ứng.
//		//PMS_PRINT("\n Thread %d j:%d vid:%d idxListVer:%d posCol:%e posRow:%d",i,j,dArrEmbedding[posRow].vid, idxListVer,posCol,posRow);
//		goto loop;
	}
}
//Hiển thị embedding column
__global__ void kernelDisplaydArrPointerEmbedding(Embedding **dArrPointerEmbedding,int noElemEmbeddingCol,int noElemEmbedding){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElemEmbedding){
		int lastCol = noElemEmbeddingCol-1;
		Embedding *dArrEmbedding;
		PMS_PRINT("\n Last Embedding column:%d Element:%d (idx vid):(%d %d)",lastCol,i,dArrEmbedding[i].idx,dArrEmbedding[i].vid);
		int prevRow=i;
		for (int j = lastCol; j>=0; j--)
		{
			dArrEmbedding= dArrPointerEmbedding[j];
			prevRow=dArrEmbedding[prevRow].idx;
			PMS_PRINT("\n Embedding column:%d Element:%d (idx vid):(%d %d)",lastCol,i,dArrEmbedding[i].idx,dArrEmbedding[i].vid);
		}
	}
}
__global__ void kernelDisplaydArrPointerEmbedding2(Embedding **dArrPointerEmbedding,int noElemEmbeddingCol,int noElemEmbedding){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElemEmbedding){
		PMS_PRINT("\n Thread: %d",i);
		Embedding *dArrEmbedding=nullptr;
		int lastCol = noElemEmbeddingCol-1;
		dArrEmbedding = dArrPointerEmbedding[lastCol];
		PMS_PRINT("\n Embedding column:%d Element:%d (idx vid):(%d %d)",lastCol,i,dArrEmbedding[i].idx,dArrEmbedding[i].vid);
	}
}

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

__global__ void kernel_GetEmbeddings(Embedding **dArrPointerEmbedding,int noElemEmbeddingCol,int noElemEmbedding){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	int lastColumn = noElemEmbeddingCol -1;
	if(i<noElemEmbedding){
		Embedding* EC = nullptr;
		EC = dArrPointerEmbedding[lastColumn];
		PMS_PRINT("EC: %d (idx:%d,vid:%d)\n",lastColumn,EC[i].idx,EC[i].vid);
		EC = dArrPointerEmbedding[0];
		PMS_PRINT("EC: %d (idx:%d,vid:%d)\n",0,EC[i].idx,EC[i].vid);
	}
}
//Duyệt qua các embedding column
//Cần biết địa chỉ của con trỏ trên device và số lượng phần tử trong embedding column đó
//Tạo bộ nhớ trên host để lưu trữ embedding column từ device
//Chép embedding column từ device sang bộ nhớ host.
int PMS::displayEmbeddingColumn(const vector<ptrArrEmbedding> &hEmbeddingColumn){
	int status =0;
	/*cudaError_t cudaStatus;
	dim3 block(blocksize);
	dim3 grid((noElemEmbedding + block.x - 1)/block.x);
	PMS_PRINT("\n************ Embedding dArrPointerEmbedding ************\n");
	kernelDisplaydArrPointerEmbedding2<<<grid,block>>>(dArrPointerEmbedding,noElemEmbeddingCol,noElemEmbedding);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}
	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}*/
	std::printf("\nAddress of Embedding Column inside function: %d",&hEmbeddingColumn);


Error:
	return status;
}

//Save embedding column to csv file.
int PMS::saveEmbeddingColumn(vector<ptrArrEmbedding> &hLevelPtrEmbedding){
	int status =0;
	cudaError_t cudaStatus;
	std::printf("\nAddress of Embedding Column inside function: %d",&hLevelPtrEmbedding);


	cout<< "Number of Embedding column: "<< hLevelPtrEmbedding.at(objLevel.Level).noElem<<endl;
	cout << "Number of embeddings: " << hLevelPtrEmbedding.at(objLevel.Level).noElemEmbedding <<endl;
	
	//Save các embeddings vào file csv.
	//step 1. Tạo số lượng Threads bằng với số lượng embeddings.
	dim3 block(blocksize);
	dim3 grid((hLevelPtrEmbedding.at(objLevel.Level).noElemEmbedding + block.x - 1)/block.x);
	//PMS_PRINT("\n************ Embedding dArrPointerEmbedding ************\n");
	
	kernel_GetEmbeddings<<<grid,block>>>(hLevelPtrEmbedding.at(objLevel.Level).dArrPointerEmbedding, \
		hLevelPtrEmbedding.at(objLevel.Level).noElem, \
		hLevelPtrEmbedding.at(objLevel.Level).noElemEmbedding);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}
	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}
Error:
	return status;
}
int PMS::displaydArrPointerEmbedding(Embedding **dArrPointerEmbedding,int noElemEmbeddingCol,int noElemEmbedding){
	int status =0;
	cudaError_t cudaStatus;
	dim3 block(blocksize);
	dim3 grid((noElemEmbedding + block.x - 1)/block.x);
	//PMS_PRINT("\n************ Embedding dArrPointerEmbedding ************\n");
	kernelDisplaydArrPointerEmbedding2<<<grid,block>>>(dArrPointerEmbedding,noElemEmbeddingCol,noElemEmbedding);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}
	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}
Error:
	return status;
}


//Xây dựng Right Most Path cho mở rộng đầu tiên
int PMS::buildRMP(){
	int status = 0;
	cudaError_t cudaStatus;
	//hRMP.resize(1);
	//int noElem = hEmbedding.size();
	//int index = noElem - 1;
	//for (int i = index ; i != -1 ; i = hEmbedding.at(i).prevCol)
	//{
	//	hRMP.at(0).hArrRMP.push_back(i);		
	//}
	/*hRMP.at(0).hArrRMP.push_back(1);
	hRMP.at(0).hArrRMP.push_back(0);		
	hRMP.at(0).noElem = 2;*/

	//Xây dựng right most path ban đầu của các mở rộng luôn là 0 và 1.
	hLevelRMP.resize(objLevel.size);
	hLevelRMP.at(objLevel.Level).noElem=2;
	hLevelRMP.at(objLevel.Level).hArrRMP.push_back(1);
	hLevelRMP.at(objLevel.Level).hArrRMP.push_back(0);

#pragma region "build RMP on device"
	//Xây dựng right most path từ vector<int> hRMP
	int noElemVerOnRMP = hLevelRMP.at(objLevel.Level).noElem; //right most path chứa bao nhiêu đỉnh
	hLevelRMP.at(objLevel.Level).dRMP = nullptr; //rigt most path trên bộ nhớ device
	CHECK(cudaStatus = cudaMalloc((void**)&hLevelRMP.at(objLevel.Level).dRMP,noElemVerOnRMP*sizeof(int))); //cấp phát bộ nhớ trên device cho rmp
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}
	int *tempRMP=(int*)malloc(sizeof(int)*noElemVerOnRMP); //dùng để chứa dữ liệu từ vector hRMP
	if(tempRMP==NULL)
	{
		status =-1;
		goto Error;
	}
	////chép dữ liệu từ hRMP sang bộ nhớ temp
	for (int i = 0; i < noElemVerOnRMP; i++)
	{
		tempRMP[i] = hLevelRMP.at(objLevel.Level).hArrRMP.at(i);
	}

	//Chép dữ liệu từ temp trên host sang rmp trên device. //ở bước này không cần phải làm phức tạp như thế. Chỉ cần khởi tạo bộ nhớ trên Device và gán cho nó giá trị {1,0} cho nó là được
	CHECK(cudaStatus =cudaMemcpy(hLevelRMP.at(objLevel.Level).dRMP,tempRMP,sizeof(int)*noElemVerOnRMP,cudaMemcpyHostToDevice)); //chép dữ liệu từ temp ở host sang rmp trên device
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}

	if(tempRMP!=NULL)
	{
		free(tempRMP);
	}

	//PMS_PRINT("\n\n ******* rmp *********\n");
	//FUNCHECK(status=displayDeviceArr(hLevelRMP.at(objLevel.Level).dRMP,noElemVerOnRMP));
	//if(status!=0){
	//	goto Error;
	//}


#pragma endregion

Error:
	return status;
}
//Hàm này thực hiện theo tuần tự (1. Find Extension -> 2. Extract UniEdge -> 3.Compute & CHECK Support -> 4. CHECK minDFS_CODE -> 5. BuildEmbedding -> 6.Find RMP -> Miningv2
//Input: inlucding RMP, DFS_CODE, hEmbedding Columns ban đầu
//Need to do: 
//1. Xây dựng hLevelPtrEmbedding ở Level đầu tiên -> 2. Xây dựng hListVer để lưu trữ các đỉnh thuộc embedding column
//-> 3. Trích các đỉnh thuộc embedding column và lưu vào hListVer -> 3.1. Tạo tempListVer
//-> 4. Duyệt các các embedding column -> 4.1. Trích các đỉnh đã tìm được từ hListVer vào bộ nhớ tempListVer 
//-> 4.2 Gọi hàm forwardExtension() để tìm các mở rộng hợp lệ từ các đỉnh đã trích trong tempListVer
// Lưu các mở rộng hợp lệ (nếu có) vào trong EXT<k> tương ứng (với k=0 dành cho last embedding column ngược về embedding column trước). 
// Sau đó khai thác sâu xuống. (gọi hàm MiningV3() trong forwardExtension).
// -> 4.3. Sau khi khai thác xong thì phải xoá bộ nhớ trên device của hLevelEXTk ở Level đang xét
// Xoá bộ nhớ trên device của hLevelUniEdge và hLevelUniEdgeSatisfyminSup
//-> 3.2 Xoá bộ nhớ tempListVer 
// Clear bộ nhớ hLevelUniEdge,hLevelUniEdgeSatisfyminSup, hLevelEXTk
int PMS::FSMining(int *rmp,int noElemVerOnRMP)
{
	int status = 0;
	cudaError_t cudaStatus;
	//1. Tìm mở rộng từ các đỉnh thuộc right most path của các embedding
	//2. Trích ra các mở rộng hợp lệ và lưu chúng vào EXTk tương ứng
	//3. Duyệt qua các EXTk trích các mở rộng duy nhất
	//4. Tính độ hỗ trợ cho các mở rộng duy nhất trong EXTk
	//5. Loại bỏ những mở rộng không thoả mãn độ hỗ trợ do người dùng chỉ định
	//6. Kiểm tra minDFS_CODE --> ghi nhận kết quả và tiếp bước 7
	//7. Mở rộng Embedding cho các DFS_CODE thoả minSup
	//8. Lặp lại bước 1.

	//1. Để tìm các mở rộng cho các Embedding từ các đỉnh thuộc RMP, thì chúng ta cần xây dựng Embedding trên device
	//Hàm kernelGetPointerdArrEmbedding giúp chúng ta xây dựng mảng dArrPointerEmbedding chứa các pointer của các mảng dArrEmbedding trên device hiện
	//đang được quản lý bởi các phần tử của vector hEmbedding ở bộ nhớ host.
	//Kernel có thể đọc dữ liệu trực tiếp từ mảng dArrPointerEmbedding.
	//1. Xây dựng hLevelPtrEmbedding ở Level đầu tiên
	hLevelPtrEmbedding.resize(1);
	hLevelPtrEmbedding.at(0).noElem=hEmbedding.size(); //Cập nhật số lượng embedding column ở Level này
#pragma region "build dArrPointerEmbedding on device"
	CHECK(cudaStatus = cudaMalloc((void**)&hLevelPtrEmbedding.at(0).dArrPointerEmbedding,hEmbedding.size()*sizeof(Embedding**))); //Cấp phát bộ nhớ cho mảng dArrPointerEmbedding tại Level tương ứng.
	if(cudaStatus!=cudaSuccess){
		status = -1;
		PMS_PRINT("\n cudaMalloc dArrPointerEmbedding failed()");
		goto Error;
	}
	//Duyệt qua hEmbedding để lấy địa chỉ của các embedding column rồi lưu vào dArrPointerEmbedding của hLevelPtrEmbedding
	for (int i = 0; i < hEmbedding.size(); i++)
	{		
		kernelGetPointerdArrEmbedding<<<1,1>>>(hEmbedding.at(i).dArrEmbedding,hLevelPtrEmbedding.at(0).dArrPointerEmbedding,i); //Mỗi phần tử của mảng dArrPointerEmbedding chứa địa chỉ của dArrEmbedding
	}
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		status = -1;
		PMS_PRINT("\n kernelGetPointerdArrEmbedding failed");
		goto Error;
	}
#pragma endregion 


#pragma region "cudaMalloc for listVer to find listVer On All EmbeddingColumn that belong to RMP"

	//Tìm danh sách các đỉnh thuộc right most path của các embedding
	//Kết quả lưu vào các vector tương ứng
	//int lastCol = hEmbedding.size() - 1; //cột cuối của embedding
	int lastCol=1; //Ở đây chúng ta có thể chỉ định cột cuối của embedding column có index là 1.
	hLevelPtrEmbedding.at(0).noElemEmbedding= hEmbedding.at(lastCol).noElem; //Cập nhật số lượng embedding	
	int noElemListVer= hRMP.at(0).noElem * hLevelPtrEmbedding.at(0).noElemEmbedding; //số lượng phần tử của listVer bằng số lượng đỉnh trên right most path nhân với số lượng embedding
	hListVer.resize(1); //Tạo hListVer để lưu trữ các đỉnh thuộc Embedding column
	hListVer.at(0).noElem=noElemListVer;
	CHECK(cudaStatus = cudaMalloc((void**)&hListVer.at(0).dListVer,sizeof(int)*noElemListVer)); //cấp phát bộ nhớ cho listVer
	if(cudaStatus!=cudaSuccess){
		PMS_PRINT("\n CudaMalloc dListVer failed");
		status =-1;
		goto Error;
	}

#pragma endregion

	FUNCHECK(status=displaydArrPointerEmbedding(hLevelPtrEmbedding.at(0).dArrPointerEmbedding,hLevelPtrEmbedding.at(0).noElem,hLevelPtrEmbedding.at(0).noElemEmbedding));
	if(status!=0){
		goto Error;
	}


	//#pragma region "build RMP on device"
	//
	//	//Xây dựng right most path từ vector<int> hRMP
	//	int noElemVerOnRMP = hRMP.at(0).noElem; //right most path chứa bao nhiêu đỉnh
	//	int *rmp = nullptr; //rigt most path trên bộ nhớ device
	//	CHECK(cudaStatus = cudaMalloc((void**)&rmp,noElemVerOnRMP*sizeof(int))); //cấp phát bộ nhớ trên device cho rmp
	//	if(cudaStatus!=cudaSuccess){
	//		status =-1;
	//		goto Error;
	//	}
	//	int *temp=(int*)malloc(sizeof(int)*noElemVerOnRMP); //dùng để chứa dữ liệu từ vector hRMP
	//	if(temp==NULL){
	//		status =-1;
	//		goto Error;
	//	}
	//	//chép dữ liệu từ hRMP sang bộ nhớ temp
	//	for (int i = 0; i < noElemVerOnRMP; i++)
	//	{
	//		temp[i] = hRMP.at(0).hArrRMP.at(i);
	//	}
	//	//Chép dữ liệu từ temp trên host sang rmp trên device
	//	CHECK(cudaStatus =cudaMemcpy(rmp,temp,sizeof(int)*noElemVerOnRMP,cudaMemcpyHostToDevice)); //chép dữ liệu từ temp ở host sang rmp trên device
	//	if(cudaStatus!=cudaSuccess){
	//		status =-1;
	//		goto Error;
	//	}
	//
	//	std::free(temp);
	//
	//	PMS_PRINT("\n\n ******* rmp *********\n");
	//	FUNCHECK(status=displayDeviceArr(rmp,noElemVerOnRMP));
	//	if(status!=0){
	//		goto Error;
	//	}
	//#pragma endregion

#pragma region "find listVer from All EmbeddingColumn"

	//Tìm danh sách các đỉnh thuộc right most path ở các cột embedding để thực hiện mở rộng
	dim3 block(blocksize);
	dim3 grid((hLevelPtrEmbedding.at(0).noElemEmbedding + block.x -1)/block.x);
	//Kernel tìm các đỉnh thuộc embedding và lưu chúng vào hListVer
	kernelFindListVer<<<block,grid>>>(hLevelPtrEmbedding.at(0).dArrPointerEmbedding,hLevelPtrEmbedding.at(0).noElemEmbedding,rmp,noElemVerOnRMP,hListVer.at(0).dListVer); //tìm listVer
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}
	//hiển thị danh sách đỉnh
	PMS_PRINT("\n\n ********* listVer *********\n");
	FUNCHECK(status=displayDeviceArr(hListVer.at(0).dListVer,noElemListVer));
	if(status!=0){
		goto Error;
	}


#pragma endregion


	//dArrVidOnRMP: chứa các đỉnh thuộc RMP của mỗi Embedding. Dùng để kiểm tra sự tồn tại của đỉnh trên right most path
	//khi tìm các mở rộng backward. Chỉ dùng tới khi right most path có 3 đỉnh trở lên (chỉ xét trong đơn đồ thị vô hướng).
	//int *dArrVidOnRMP = nullptr; //lưu trữ các đỉnh trên RMP của Embedding, có kích thước nhỏ hơn 2 đỉnh so với RMP
	//int noElemdArrVidOnRMP= hRMP.at(0).noElem - 2;
	//int *fromPosCol=nullptr; //lưu trữ các cột của Embedding mà tại đó thuộc right most path. Thật ra mình có thể suy luận được từ rmp
	//if (noElemdArrVidOnRMP >0){
	//	cudaStatus = cudaMalloc((void**)&dArrVidOnRMP,hLevelPtrEmbedding.at(idxLevel).noElemEmbedding*noElemdArrVidOnRMP*sizeof(int));
	//	CHECK(cudaStatus);
	//	if(cudaStatus!=cudaSuccess){
	//		status =-1;
	//		goto Error;
	//	}
	//	//cudaStatus = cudaMalloc((void**)&fromPosCol,hLevelPtrEmbedding.at(idxLevel).noElemEmbedding*noElemdArrVidOnRMP*sizeof(int));
	//	//CHECK(cudaStatus);
	//	//if(cudaStatus!=cudaSuccess){
	//	//	status =-1;
	//	//	goto Error;
	//	//}
	//}
	//if(hRMP.at(0).noElem>2){ //Nếu số lượng đỉnh trên RMP lớn hơn 2 thì mới tồn tại backward. Vì ở đây chỉ xét đơn đồ thị vô hướng
	//	FUNCHECK(status = displaydArrPointerEmbedding(hLevelPtrEmbedding.at(idxLevel).dArrPointerEmbedding,hLevelPtrEmbedding.at(idxLevel).noElem,hLevelPtrEmbedding.at(idxLevel).noElemEmbedding));
	//	if(status!=0){
	//		goto Error;
	//	}
	//	kernelFindVidOnRMP<<<grid,block>>>(hLevelPtrEmbedding.at(idxLevel).dArrPointerEmbedding,hLevelPtrEmbedding.at(idxLevel).noElemEmbedding,rmp,noElemVerOnRMP,dArrVidOnRMP,noElemdArrVidOnRMP);
	//	cudaDeviceSynchronize();
	//	cudaStatus = cudaGetLastError();
	//	CHECK(cudaStatus);
	//	if(cudaStatus!=cudaSuccess){
	//		status =-1;
	//		goto Error;
	//	}
	//	PMS_PRINT("\n ******** dArrVidOnRMP *******\n");
	//	FUNCHECK(status = displayDeviceArr(dArrVidOnRMP,noElemdArrVidOnRMP*hLevelPtrEmbedding.at(idxLevel).noElemEmbedding));
	//	if(status!=0){
	//		goto Error;
	//	}
	//	//PMS_PRINT("\n ******** fromPosCol *******\n");
	//	//displayDeviceArr(fromPosCol,noElemdArrVidOnRMP);
	//}


	int noElemEXTk =noElemVerOnRMP; //Số lượng phần tử EXTk bằng số lượng đỉnh trên right most path
	//hEXTk.resize(noElemEXTk); //Các mở rộng hợp lệ từ đỉnh k sẽ được lưu trữ vào EXTk tương ứng.

	//Quản lý theo Level phục vụ cho khai thác đệ quy
	hLevelEXT.resize(1); //Khởi tạo vector quản lý bộ nhớ cho level
	hLevelEXT.at(0).noElem = noElemVerOnRMP; //Cập nhật số lượng phần tử vector vE bằng số lượng đỉnh trên RMP
	hLevelEXT.at(0).vE.resize(noElemVerOnRMP); //Cấp phát bộ nhớ cho vector vE. Có thể code theo cách khác:hLevelEXT.at(0).vE.resize(hLevelEXT.at(0).noElem). Vì fermi không hỗ trợ đệ quy nên chúng ta cũng không thể xử lý song song các VE. Do đó, việc tạo nhiều VE có vẻ dư thừa. Chúng ta chỉ cần 1 VE và giải phóng nó để sử dụng cho lần lặp tiếp theo.

	hLevelUniEdge.resize(1); //Số lượng phần tử UniEdge cũng giống với EXT
	hLevelUniEdge.at(0).noElem=noElemVerOnRMP;
	hLevelUniEdge.at(0).vUE.resize(noElemVerOnRMP);

	hLevelUniEdgeSatisfyMinsup.resize(1);
	hLevelUniEdgeSatisfyMinsup.at(0).noElem= noElemVerOnRMP;
	hLevelUniEdgeSatisfyMinsup.at(0).vecUES.resize(noElemVerOnRMP);

	int *tempListVerCol = nullptr; //chứa danh sách các đỉnh cần mở rộng thuộc một embedding column cụ thể.
	CHECK(cudaStatus = cudaMalloc((void**)&tempListVerCol,hLevelPtrEmbedding.at(0).noElemEmbedding * sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	//3.Duyệt qua các Embedding column, trích các đỉnh cần mở rộng ở cột đó để đi mở rộng
	//Ở đây chỉ có các mở rộng forward, chưa có mở rộng backward vì chúng ta đang mở rộng subgraph chỉ có 1 cạnh.
	//Do đó, chúng ta chỉ cần duyệt qua các embedding columns và tìm các mở rộng forward từ các đỉnh thuộc embedding column.
	//if(noElemVerOnRMP == 2){
	for (int i = 0; i < noElemVerOnRMP ; i++)
	{
		int colEmbedding = hRMP.at(0).hArrRMP.at(i); //Tìm mở rộng cho các đỉnh tại vị trí cột colEmbedding trong vector hEmbedding //nên sửa lại là hLevelRMP.at(0).hArrRMP.at(i);
		currentColEmbedding=colEmbedding; //Đang mở rộng từ cột nào của Embedding. Được dùng để cập nhật prevCol, phục vụ cho việc xây dựng Right Most Path
		int k = i; //lưu vào Extk với k = i; K=0 đại diện cho EXT0: là EXT cuối
		kernelExtractFromListVer<<<grid,block>>>(hListVer.at(0).dListVer,i*hLevelPtrEmbedding.at(0).noElemEmbedding,hLevelPtrEmbedding.at(0).noElemEmbedding,tempListVerCol);//trích các đỉnh từ listVer, từ vị trí i*noElemEmbedding,trích noElemEmbedding phần tử, bỏ vào tempListVerCol
		CHECK(cudaStatus=cudaDeviceSynchronize());
		if(cudaStatus!=cudaSuccess){
			status=-1;
			goto Error;
		}

		CHECK(cudaStatus = cudaGetLastError());
		if(cudaStatus!=cudaSuccess){
			status=-1;
			goto Error;
		}
		PMS_PRINT("\n ****** tempListVerCol ***********\n");
		FUNCHECK(status=displayDeviceArr(tempListVerCol,hLevelPtrEmbedding.at(0).noElemEmbedding));
		if(status!=0){
			goto Error;
		}

		//gọi hàm forwardExtension để tìm các mở rộng forward từ cột colEmbedding, lưu kết quả vào hEXTk tại vị trí k, với các đỉnh
		// cần mở rộng là tempListVerCol, thuộc righ most path
		////Hàm này cũng đồng thời trích các mở rộng duy nhất từ các EXT và lưu vào UniEdge
		//Hàm này cũng gọi đệ quy FSMining bên trong

		//Tìm các mở rộng forward từ các đỉnh thuộc right most path và lưu chúng vào EXT<k> tương ứng
		FUNCHECK(status = forwardExtension(k,tempListVerCol,hLevelPtrEmbedding.at(0).noElemEmbedding,hRMP.at(0).hArrRMP.at(i)));
		if(status!=0){
			goto Error;
		}
		//FSMining();
		//Khi đã khai thác xong một embedding column thì phải giải phóng bộ nhớ để chuẩn bị cho khai thác embedding column khác.
		if(hLevelEXT.at(0).vE.at(i).noElem>0){ //Nếu số lượng phần tử của mảng dArrExt = 0 thì chúng ta không giải phóng bộ nhớ dArrExt vì nó chưa được cấp phát.
			CHECK(cudaStatus=cudaFree(hLevelEXT.at(0).vE.at(i).dArrExt));
			if(cudaStatus!=cudaSuccess){
				status=-1;
				goto Error;
			}

			CHECK(cudaStatus=cudaFree(hLevelUniEdge.at(0).vUE.at(i).dArrUniEdge));
			if(cudaStatus!=cudaSuccess){
				status=-1;
				goto Error;
			}
		}
		if(hLevelUniEdgeSatisfyMinsup.at(0).vecUES.at(i).noElem>0){
			CHECK(cudaStatus=cudaFree(hLevelUniEdgeSatisfyMinsup.at(0).vecUES.at(i).dArrUniEdge));
			if(cudaStatus!=cudaSuccess){
				status=-1;
				goto Error;
			}
			free(hLevelUniEdgeSatisfyMinsup.at(0).vecUES.at(i).hArrSupport);
		}

	} //kết thúc for duyệt qua các đỉnh của embedding column để tìm các mở rộng forward hợp lệ
	if(hLevelEXT.at(0).noElem>0){
		hLevelEXT.at(0).vE.clear();
	}
	if(hLevelEXT.size()>0){
		hLevelEXT.clear();
	}

	if(hLevelUniEdge.at(0).noElem>0){
		hLevelUniEdge.at(0).vUE.clear();
	}
	if(hLevelUniEdge.size()>0){
		hLevelUniEdge.clear();
	}


	if(hLevelUniEdgeSatisfyMinsup.at(0).noElem>0){
		hLevelUniEdgeSatisfyMinsup.at(0).vecUES.clear();
	}

	if(hLevelUniEdgeSatisfyMinsup.size()>0){
		hLevelUniEdgeSatisfyMinsup.clear();
	}

	CHECK(cudaStatus=cudaFree(tempListVerCol));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	if(hListVer.at(0).noElem>0){
		CHECK(cudaStatus=cudaFree(hListVer.at(0).dListVer));
		if(cudaStatus!=cudaSuccess){
			status=-1;
			goto Error;
		}
		hListVer.clear();
	}
Error:
	return status;
}

int PMS::buildArrPointerEmbedding(vector<EmbeddingColumn> hEmbedding,vector<ptrArrEmbedding>& hLevelPtrEmbedding){
	int status = 0;
	cudaError_t cudaStatus;
	//3.1. Cấp phát bộ nhớ cho dArrPointerEmbedding và vector hLevelPtrEmbedding để lưu kết quả
	int lastCol = hEmbedding.size() - 1; //cột cuối của embedding
	hLevelPtrEmbedding.resize(Level);
	hLevelPtrEmbedding.at(idxLevel).noElem=hEmbedding.size(); //số lượng embedding column
	hLevelPtrEmbedding.at(idxLevel).noElemEmbedding= hEmbedding.at(lastCol).noElem; //số lượng embedding
	CHECK(cudaStatus = cudaMalloc((void**)&hLevelPtrEmbedding.at(idxLevel).dArrPointerEmbedding,hEmbedding.size()*sizeof(Embedding**))); //Cấp phát bộ nhớ cho mảng dArrPointerEmbedding tại Level tương ứng.
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	//3.2 Xây dựng mảng dArrPointerEmbedding chứa địa chỉ của các embedding column trên device
	for (int i = 0; i < hEmbedding.size(); i++)
	{		
		kernelGetPointerdArrEmbedding<<<1,1>>>(hEmbedding.at(i).dArrEmbedding,hLevelPtrEmbedding.at(idxLevel).dArrPointerEmbedding,i); //Mỗi phần tử của mảng dArrPointerEmbedding chứa địa chỉ của dArrEmbedding
	}
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}

Error:
	return status;
}


int PMS::buildArrPointerEmbeddingv2(vector<EmbeddingColumn> hEmbedding,vector<ptrArrEmbedding>& hLevelPtrEmbedding)
{
	int status = 0;
	cudaError_t cudaStatus;

	//3.1. Cấp phát bộ nhớ cho dArrPointerEmbedding và vector hLevelPtrEmbedding để lưu kết quả
	int lastCol = hEmbedding.size() - 1; //cột cuối của embedding
	hLevelPtrEmbedding.resize(Level);
	hLevelPtrEmbedding.at(idxLevel).noElem=hEmbedding.size(); //số lượng embedding column
	if(hEmbedding.at(lastCol).hBackwardEmbedding.size()>0){
		int lastcolbw = hEmbedding.at(lastCol).hBackwardEmbedding.size()-1;
		hLevelPtrEmbedding.at(idxLevel).noElemEmbedding= hEmbedding.at(lastCol).hBackwardEmbedding.at(lastcolbw).noElem;
	}
	else
	{
		hLevelPtrEmbedding.at(idxLevel).noElemEmbedding= hEmbedding.at(lastCol).noElem; //số lượng embedding
	}
	CHECK(cudaStatus = cudaMalloc((void**)&hLevelPtrEmbedding.at(idxLevel).dArrPointerEmbedding,hEmbedding.size()*sizeof(Embedding**))); //Cấp phát bộ nhớ cho mảng dArrPointerEmbedding tại Level tương ứng.
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	//3.2 Xây dựng mảng dArrPointerEmbedding chứa địa chỉ của các embedding column trên device
	for (int i = 0; i < hEmbedding.size(); i++)
	{		
		if(hEmbedding.at(i).hBackwardEmbedding.size()>0){
			int idxLastBackwardEmbeddingCol = hEmbedding.at(i).hBackwardEmbedding.size()-1;
			kernelGetPointerdArrEmbedding<<<1,1>>>(hEmbedding.at(i).hBackwardEmbedding.at(idxLastBackwardEmbeddingCol).dArrEmbedding,hLevelPtrEmbedding.at(idxLevel).dArrPointerEmbedding,i); //Mỗi phần tử của mảng dArrPointerEmbedding chứa địa chỉ của dArrEmbedding
		}
		else
		{
			kernelGetPointerdArrEmbedding<<<1,1>>>(hEmbedding.at(i).dArrEmbedding,hLevelPtrEmbedding.at(idxLevel).dArrPointerEmbedding,i); //Mỗi phần tử của mảng dArrPointerEmbedding chứa địa chỉ của dArrEmbedding
		}
	}
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}

Error:
	return status;
}

int PMS::buildArrPointerEmbeddingv3()
{
	int status = 0;
	cudaError_t cudaStatus;
	
	//3.1. Cấp phát bộ nhớ cho dArrPointerEmbedding và vector hLevelPtrEmbedding để lưu kết quả
	int lastCol = hEmbedding.size() - 1; //cột cuối của embedding
	//hLevelPtrEmbedding.at(objLevel.Level).noElem=hEmbedding.size(); //số lượng embedding column
	hLevelPtrEmbedding.at(objLevel.Level).noElem=hLevelRMP.at(objLevel.Level).noElem;
	//Cập nhật số lượng embedding cho level mới tuỳ vào embedding column đang xét có backward embedding column hay không
	if(hEmbedding.at(lastCol).hBackwardEmbedding.size()>0)
	{
		int lastcolbw = hEmbedding.at(lastCol).hBackwardEmbedding.size()-1;
		hLevelPtrEmbedding.at(objLevel.Level).noElemEmbedding= hEmbedding.at(lastCol).hBackwardEmbedding.at(lastcolbw).noElem;
	}
	else
	{
		hLevelPtrEmbedding.at(objLevel.Level).noElemEmbedding= hEmbedding.at(lastCol).noElem; //số lượng embedding
	}
	
	//Cấp phát bộ nhớ cho mảng dArrPointerEmbedding tại Level tương ứng.
	CHECK(cudaStatus = cudaMalloc((void**)&hLevelPtrEmbedding.at(objLevel.Level).dArrPointerEmbedding,hLevelRMP.at(objLevel.Level).noElem*sizeof(Embedding**))); 
	if(cudaStatus!=cudaSuccess)
	{
		status = -1;
		goto Error;
	}
	
	//3.2 Xây dựng mảng dArrPointerEmbedding chứa địa chỉ của các embedding column trên device
	//int prevCol = lastCol;
	for (int i =0; i<hLevelRMP.at(objLevel.Level).noElem;++i)
	{
		int j = abs(i-(hLevelRMP.at(objLevel.Level).noElem-1));
		int col = hLevelRMP.at(objLevel.Level).hArrRMP.at(j);
		if(hEmbedding.at(col).hBackwardEmbedding.size()>0)
		{ 
			int idxLastBackwardEmbeddingCol = hEmbedding.at(col).hBackwardEmbedding.size()-1;
			//Mỗi phần tử của mảng dArrPointerEmbedding chứa địa chỉ của dArrEmbedding
			kernelGetPointerdArrEmbedding<<<1,1>>>( \
				hEmbedding.at(col).hBackwardEmbedding.at(idxLastBackwardEmbeddingCol).dArrEmbedding, \
				hLevelPtrEmbedding.at(objLevel.Level).dArrPointerEmbedding,i);
		}
		else
		{
			//Mỗi phần tử của mảng dArrPointerEmbedding chứa địa chỉ của dArrEmbedding
			kernelGetPointerdArrEmbedding<<<1,1>>>( \
				hEmbedding.at(col).dArrEmbedding, \
				hLevelPtrEmbedding.at(objLevel.Level).dArrPointerEmbedding,i); 
		}
	}
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess)
	{
		status = -1;
		goto Error;
	}

Error:
	return status;
}

void PMS::write_embedding_column()
{
	/*
	fstream of;
	char *filename ="EC.csv";
	of.open(filename,ios::out|ios::app);
	//3.1. Cấp phát bộ nhớ cho dArrPointerEmbedding và vector hLevelPtrEmbedding để lưu kết quả
	int lastCol = hEmbedding.size() - 1; //cột cuối của embedding
	int prevCol = lastCol;
	while(prevCol!=-1) //Trong khi chưa ở Q cuối thì lặp lại khối lệnh while
	{
		if(hEmbedding.at(prevCol).hBackwardEmbedding.size()>0)
		{ //Nếu tại EmbeddingColumn đang xét tồn tại các BW Embedding Column thì ưu tiên lấy BW EC cuối cùng.
			int idxLastBackwardEmbeddingCol = hEmbedding.at(prevCol).hBackwardEmbedding.size()-1;
			Embedding *hArrEmbedding = (Embedding*)malloc(sizeof(Embedding)*hEmbedding.at(prevCol).hBackwardEmbedding.at(idxLastBackwardEmbeddingCol).noElem);
			cudaMemcpy(hArrEmbedding, \
				hEmbedding.at(prevCol).hBackwardEmbedding.at(idxLastBackwardEmbeddingCol).dArrEmbedding, \
				hEmbedding.at(prevCol).hBackwardEmbedding.at(idxLastBackwardEmbeddingCol).noElem * sizeof(Embedding), \
				cudaMemcpyDeviceToHost);
			//std::printf("\nprevQ: %d",hEmbedding.at(i).hBackwardEmbedding.at(idxLastBackwardEmbeddingCol).prevCol);
			of<<"prevQ: "<<hEmbedding.at(prevCol).hBackwardEmbedding.at(idxLastBackwardEmbeddingCol).prevCol<<endl;
			for(int m=0;m<hEmbedding.at(prevCol).hBackwardEmbedding.at(idxLastBackwardEmbeddingCol).noElem;++m)
			{
				//std::printf("\nColumn: %d, (%d,%d)",i,hArrEmbedding[m].idx,hArrEmbedding[m].vid);
				of<<hArrEmbedding[m].idx<<","<<hArrEmbedding[m].vid<<endl;
			}

			prevCol = hEmbedding.at(prevCol).hBackwardEmbedding.at(idxLastBackwardEmbeddingCol).prevCol;
		}
		else
		{
			//Mỗi phần tử của mảng dArrPointerEmbedding chứa địa chỉ của dArrEmbedding
			//kernelGetPointerdArrEmbedding<<<1,1>>>(hEmbedding.at(i).dArrEmbedding,hLevelPtrEmbedding.at(objLevel.Level).dArrPointerEmbedding,i); 
			Embedding *hArrEmbedding = (Embedding*)malloc(sizeof(Embedding)*hEmbedding.at(prevCol).noElem);
			cudaMemcpy(hArrEmbedding, \
			hEmbedding.at(prevCol).dArrEmbedding, \
			hEmbedding.at(prevCol).noElem * sizeof(Embedding), \
			cudaMemcpyDeviceToHost);
			//std::printf("\nprevQ: %d",hEmbedding.at(i).prevCol);
			of<<"prevQ: "<<hEmbedding.at(prevCol).prevCol<<endl;
			for(int m=0;m<hEmbedding.at(prevCol).noElem;++m)
			{
				//std::printf("\nColumn: %d, (%d,%d)",i,hArrEmbedding[m].idx,hArrEmbedding[m].vid);
				of<<hArrEmbedding[m].idx<<","<<hArrEmbedding[m].vid<<endl;
			}
			prevCol = hEmbedding.at(prevCol).prevCol;
		}
	}
	of.flush();
	of.close();
	*/
	fstream of;
	char *filename ="EC.csv";
	of.open(filename,ios::out|ios::app);
	//Duyệt qua các phần tử trên RMP tại Level đang xét
	for (int i =0; i<hLevelRMP.at(objLevel.Level).noElem;++i)
	{
		int j = abs(i-(hLevelRMP.at(objLevel.Level).noElem-1));
		int col = hLevelRMP.at(objLevel.Level).hArrRMP.at(j);
		if(hEmbedding.at(col).hBackwardEmbedding.size()>0)
		{ 
			int idxLastBackwardEmbeddingCol = hEmbedding.at(col).hBackwardEmbedding.size()-1;
			Embedding *hArrEmbedding = (Embedding*)malloc(sizeof(Embedding)*hEmbedding.at(col).hBackwardEmbedding.at(idxLastBackwardEmbeddingCol).noElem);
			cudaMemcpy(hArrEmbedding, \
				hEmbedding.at(col).hBackwardEmbedding.at(idxLastBackwardEmbeddingCol).dArrEmbedding, \
				hEmbedding.at(col).hBackwardEmbedding.at(idxLastBackwardEmbeddingCol).noElem * sizeof(Embedding), \
				cudaMemcpyDeviceToHost);
			//std::printf("\nprevQ: %d",hEmbedding.at(i).hBackwardEmbedding.at(idxLastBackwardEmbeddingCol).prevCol);
			of<<"prevQ: "<<hEmbedding.at(col).hBackwardEmbedding.at(idxLastBackwardEmbeddingCol).prevCol<<endl;
			for(int m=0;m<hEmbedding.at(col).hBackwardEmbedding.at(idxLastBackwardEmbeddingCol).noElem;++m)
			{
				//std::printf("\nColumn: %d, (%d,%d)",i,hArrEmbedding[m].idx,hArrEmbedding[m].vid);
				of<<hArrEmbedding[m].idx<<","<<hArrEmbedding[m].vid<<endl;
			}
			free(hArrEmbedding);
		}
		else
		{
			//Mỗi phần tử của mảng dArrPointerEmbedding chứa địa chỉ của dArrEmbedding
			//kernelGetPointerdArrEmbedding<<<1,1>>>(hEmbedding.at(i).dArrEmbedding,hLevelPtrEmbedding.at(objLevel.Level).dArrPointerEmbedding,i); 
			Embedding *hArrEmbedding = (Embedding*)malloc(sizeof(Embedding)*hEmbedding.at(col).noElem);
			cudaMemcpy(hArrEmbedding, \
			hEmbedding.at(col).dArrEmbedding, \
			hEmbedding.at(col).noElem * sizeof(Embedding), \
			cudaMemcpyDeviceToHost);
			//std::printf("\nprevQ: %d",hEmbedding.at(i).prevCol);
			of<<"prevQ: "<<hEmbedding.at(col).prevCol<<endl;
			for(int m=0;m<hEmbedding.at(col).noElem;++m)
			{
				//std::printf("\nColumn: %d, (%d,%d)",i,hArrEmbedding[m].idx,hArrEmbedding[m].vid);
				of<<hArrEmbedding[m].idx<<","<<hArrEmbedding[m].vid<<endl;
			}
			
			free(hArrEmbedding);
		}
	}
	of.flush();
	of.close();
	return;
}

//Tại mỗi Level có thể có nhiều phần tử Pointer Embedding tuỳ thuộc vào số lượng backward extension phổ biến ở Level đó.
int PMS::buildArrPointerEmbeddingbw(vector<EmbeddingColumn> hEmbedding,vector<ptrArrEmbedding>& hLevelPtrEmbedding){ 
	int status = 0;
	cudaError_t cudaStatus;
	//3.1. Cấp phát bộ nhớ cho dArrPointerEmbedding và vector hLevelPtrEmbedding để lưu kết quả
	//int lastCol = hEmbedding.size() - 1; //cột cuối của embedding. Nếu mình lấy cột cuối thì không đúng. Phải lấy cột hiện tại. Vì khi quay lui mà vẫn lấy cột cuối là sai.
	hLevelPtrEmbedding.resize(Level);
	hLevelPtrEmbedding.at(idxLevel).noElem=hEmbedding.size(); //số lượng embedding column

	int hbwLast = hEmbedding.at(currentColEmbedding).hBackwardEmbedding.size() - 1;//lấy index của backward embedding column cuối cùng trong cột Embedding
	hLevelPtrEmbedding.at(idxLevel).noElemEmbedding= hEmbedding.at(currentColEmbedding).hBackwardEmbedding.at(hbwLast).noElem; //số lượng embedding
	CHECK(cudaStatus = cudaMalloc((void**)&hLevelPtrEmbedding.at(idxLevel).dArrPointerEmbedding,hEmbedding.size()*sizeof(Embedding**))); //Cấp phát bộ nhớ cho mảng dArrPointerEmbedding tại Level tương ứng.
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	//Khi nào giải phóng bộ nhớ của hLevelPtrEmbedding.at(idxLevel).dArrPointerEmbedding? ==> Khi khai thác xong trên nhánh này, sau đó quay lui để khai thác trên nhánh khác.
	//3.2 Xây dựng mảng dArrPointerEmbedding chứa địa chỉ của các embedding column trên device
	for (int i = 0; i < hEmbedding.size(); i++)
	{		
		if(i==currentColEmbedding){

			kernelGetPointerdArrEmbedding<<<1,1>>>(hEmbedding.at(i).hBackwardEmbedding.at(hbwLast).dArrEmbedding,hLevelPtrEmbedding.at(idxLevel).dArrPointerEmbedding,i); //Mỗi phần tử của mảng dArrPointerEmbedding chứa địa chỉ của dArrEmbedding
		}
		else //Nếu tại hEmbedding[i] mà có backward embedding column thì chúng ta copy địa chỉ của backward embedding column cuối cùng đó.
		{
			if(hEmbedding.at(i).hBackwardEmbedding.size()>0){
				int idxLastBackwardEmbeddingCol = hEmbedding.at(i).hBackwardEmbedding.size()-1;
				kernelGetPointerdArrEmbedding<<<1,1>>>(hEmbedding.at(i).hBackwardEmbedding.at(idxLastBackwardEmbeddingCol).dArrEmbedding,hLevelPtrEmbedding.at(idxLevel).dArrPointerEmbedding,i); //Mỗi phần tử của mảng dArrPointerEmbedding chứa địa chỉ của dArrEmbedding
			}
			else
			{
				kernelGetPointerdArrEmbedding<<<1,1>>>(hEmbedding.at(i).dArrEmbedding,hLevelPtrEmbedding.at(idxLevel).dArrPointerEmbedding,i); //Mỗi phần tử của mảng dArrPointerEmbedding chứa địa chỉ của dArrEmbedding
			}
		}
	}

	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}

Error:
	return status;
}


int PMS::buildrmpOnDevice(RMP hRMPatALevel,int *&rmp){
	int status = 0;
	cudaError_t cudaStatus;
	//cần có rmp trên device
	int noElemVerOnRMP= hRMPatALevel.noElem;
	CHECK(cudaStatus = cudaMalloc((void**)&rmp,noElemVerOnRMP*sizeof(int))); //cấp phát bộ nhớ trên device cho rmp
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}
	int *temp=(int*)malloc(sizeof(int)*noElemVerOnRMP); //dùng để chứa dữ liệu từ vector hRMP
	if(temp==NULL){
		status =-1;
		FUNCHECK(status);
		goto Error;
	}
	//chép dữ liệu từ hRMP sang bộ nhớ temp
	for (int i = 0; i < noElemVerOnRMP; i++)
	{
		temp[i] = hRMPv2.at(idxLevel).hArrRMP.at(i);
	}
	//Chép dữ liệu từ temp trên host sang rmp trên device
	CHECK(cudaStatus =cudaMemcpy(rmp,temp,sizeof(int)*noElemVerOnRMP,cudaMemcpyHostToDevice)); //chép dữ liệu từ temp ở host sang rmp trên device
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}

	std::free(temp);

	//PMS_PRINT("\n\n ******* rmp *********\n");
	//displayDeviceArr(rmp,noElemVerOnRMP);
Error:
	return status;
}
//Trích các vid cần mở rộng trên embedding column.
int PMS::findListVer(Embedding **dArrPointerEmbedding,int noElemEmbedding,int *rmp,int noElemVerOnRMP)
{
	int status = 0;
	cudaError_t cudaStatus;
	//Kiểm tra giá trị rmp trên device đã đúng chưa. vì RMP trên device sai thì sẽ kéo theo sai việc tìm các vid thuộc RMP của embeddings.
	int *hrmp = (int*)malloc(sizeof(int)*noElemVerOnRMP);
	cudaMemcpy(hrmp,rmp,sizeof(int)*noElemVerOnRMP,cudaMemcpyDeviceToHost);
	for(int i = 0; i < noElemVerOnRMP; ++i)
	{
		std::printf("\n hrmp[%d]:%d ",i,hrmp[i]);
	}


	//số lượng phần tử của listVer bằng số lượng đỉnh trên right most path nhân với số lượng embedding
	int noElemListVer= noElemVerOnRMP * noElemEmbedding; 

	hLevelListVerRMP.at(objLevel.Level).noElem=noElemListVer;
	 //cấp phát bộ nhớ cho dListVer tại current level.
	CHECK(cudaStatus = cudaMalloc((void**)&hLevelListVerRMP.at(objLevel.Level).dListVer,sizeof(int)*noElemListVer));
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}
	
	//Tìm danh sách các đỉnh thuộc right most path ở các cột embedding để thực hiện mở rộng
	//Danh sách các đỉnh của một Embedding Column sẽ được lưu vào dListVer.
	//Thứ tự lưu dListVer như thế nào?
	dim3 block(blocksize);
	dim3 grid((noElemEmbedding + block.x -1)/block.x);
	kernelFindListVer<<<grid,block>>>( \
		dArrPointerEmbedding, \
		noElemEmbedding, \
		rmp, \
		noElemVerOnRMP, \
		hLevelListVerRMP.at(objLevel.Level).dListVer);
	CHECK(cudaStatus = cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}
	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}
	////hiển thị danh sách đỉnh
	/*PMS_PRINT("\n\n ********* listVer *********\n");
	FUNCHECK(status=displayDeviceArr(hLevelListVerRMP.at(objLevel.Level).dListVer,hLevelListVerRMP.at(objLevel.Level).noElem));
	if(status!=0)
	{
	goto Error;
	}*/

Error:
	return status;
}

int PMS::findVerOnRMPForBWCheck(ptrArrEmbedding hLevelPtrEmbeddingatALevel,int* rmp,int noElemVerOnRMP,int *&dArrVidOnRMP){
	int status = 0;
	cudaError_t cudaStatus;
	int noElemdArrVidOnRMP= noElemVerOnRMP - 1;
	//int *fromPosCol=nullptr; //lưu trữ các cột của Embedding mà tại đó thuộc right most path. Thật ra mình có thể suy luận được từ rmp

	cudaStatus = cudaMalloc((void**)&dArrVidOnRMP,hLevelPtrEmbeddingatALevel.noElemEmbedding*noElemdArrVidOnRMP*sizeof(int));
	CHECK(cudaStatus);
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}

	FUNCHECK(status = displaydArrPointerEmbedding(hLevelPtrEmbeddingatALevel.dArrPointerEmbedding,hLevelPtrEmbeddingatALevel.noElem,hLevelPtrEmbeddingatALevel.noElemEmbedding));
	if(status!=0){
		goto Error;
	}

	dim3 block(blocksize);
	dim3 grid((hLevelPtrEmbeddingatALevel.noElemEmbedding + block.x -1)/block.x);

	//Hàm này tìm các vid thuộc right most path và lưu vào mảng dArrVidOnRMP. Mảng này dùng để tìm các valid backward edge.
	kernelFindVidOnRMP<<<grid,block>>>(hLevelPtrEmbeddingatALevel.dArrPointerEmbedding,hLevelPtrEmbeddingatALevel.noElemEmbedding,rmp,noElemVerOnRMP,dArrVidOnRMP,noElemdArrVidOnRMP);
	//kernelFindVidOnRMPv2<<<grid,block>>>(hLevelPtrEmbeddingatALevel.dArrPointerEmbedding,hLevelPtrEmbeddingatALevel.noElemEmbedding,rmp,noElemVerOnRMP,dArrVidOnRMP,noElemdArrVidOnRMP);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}
	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}

	PMS_PRINT("\n ******** dArrVidOnRMP *******\n");
	FUNCHECK(status=displayDeviceArr(dArrVidOnRMP,noElemdArrVidOnRMP*hLevelPtrEmbeddingatALevel.noElemEmbedding));
	if(status!=0){
		goto Error;
	}

	//PMS_PRINT("\n ******** fromPosCol *******\n");
	//displayDeviceArr(fromPosCol,noElemdArrVidOnRMP);
Error:
	return status;
}
/*Tìm các đỉnh thuộc RMP của các embeddings, mục đích là để tìm xem có mở rộng backward nào trỏ đến những đỉnh này không.
 - Input:
	+Embedding columns
	+rmp: RMP trên device.
	+noElemVerOnRMP: Số lượng đỉnh thuôc RMP của DFS CODE.
- Output:
	+dArrVidOnRMP:
*/
int PMS::findVerOnRMPForBWCheckv2(ptrArrEmbedding hLevelPtrEmbeddingatALevel,int* rmp,int noElemVerOnRMP,int *&dArrVidOnRMP)
{
	int status = 0;
	cudaError_t cudaStatus;
	int noElemdArrVidOnRMP= noElemVerOnRMP - 1; //chứa các đỉnh thuộc RMP của mỗi embedding của DFS_Code.
	//int *fromPosCol=nullptr; //lưu trữ các cột của Embedding mà tại đó thuộc right most path. Thật ra mình có thể suy luận được từ rmp
	//Giả sử có 2 embeddings của DFSCode có số lượng đỉnh thuộc RMP = 5 
	//thì khả năng 4 đỉnh khác đỉnh cuối của RMP sẽ là mở rộng backward. 
	//Khi xét các mở rộng backward, thì đỉnh nào đã thuộc RMP rồi thì không có khả năng thuộc mở rộng forward được nữa 
	//==> Nhưng mình chỉ lấy (noElemVerOnRMP - 1) thuộc RMP thì liệu có lỗ hỏng nào không khi tìm các forward?
	CHECK(cudaStatus = cudaMalloc( \
		(void**)&dArrVidOnRMP, \
		hLevelPtrEmbeddingatALevel.noElemEmbedding*noElemdArrVidOnRMP*sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}

	FUNCHECK(status = displaydArrPointerEmbedding( \
	hLevelPtrEmbeddingatALevel.dArrPointerEmbedding, \
	hLevelPtrEmbeddingatALevel.noElem,hLevelPtrEmbeddingatALevel.noElemEmbedding));
	if(status!=0){
	goto Error;
	}

	//Hàm này tìm các vid thuộc right most path và lưu vào mảng dArrVidOnRMP.
	//Mảng này dùng để tìm các valid backward edge.
	dim3 block(blocksize);
	dim3 grid((hLevelPtrEmbeddingatALevel.noElemEmbedding + block.x -1)/block.x);
	kernelFindVidOnRMP<<<grid,block>>>( \
		hLevelPtrEmbeddingatALevel.dArrPointerEmbedding, \
		hLevelPtrEmbeddingatALevel.noElemEmbedding, \
		rmp, \
		noElemVerOnRMP, \
		dArrVidOnRMP, \
		noElemdArrVidOnRMP);

	//kernelFindVidOnRMPv2<<<grid,block>>>(hLevelPtrEmbeddingatALevel.dArrPointerEmbedding,hLevelPtrEmbeddingatALevel.noElemEmbedding,rmp,noElemVerOnRMP,dArrVidOnRMP,noElemdArrVidOnRMP);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}

	/*PMS_PRINT("\n ******** dArrVidOnRMP *******\n");
	FUNCHECK(status=displayDeviceArr(dArrVidOnRMP,noElemdArrVidOnRMP*hLevelPtrEmbeddingatALevel.noElemEmbedding));
	if(status!=0){
		goto Error;
	}*/

	//PMS_PRINT("\n ******** fromPosCol *******\n");
	//displayDeviceArr(fromPosCol,noElemdArrVidOnRMP);
Error:
	return status;
}

//Tìm các mở rộng hợp lệ (forward & backward) từ EXT cuối cùng
int PMS::findValidFBExtension(int *listOfVer,ptrArrEmbedding hLevelPtrEmbeddingAtALevel,int k,int fromColumEmbedding,int *dArrVidOnRMP,int *rmp){
	int status =0;
	cudaError_t cudaStatus;

	//Tìm bậc lớn nhất của các đỉnh cần mở rộng trong listOfVer
	int maxDegreeOfVer=0;
	float *dArrDegreeOfVid=nullptr; // 1.Need cudaFree //chứa cậc của các đỉnh trong listOfVer, dùng để duyệt qua các đỉnh lân cận trong DB
	FUNCHECK(status=findMaxDegreeOfVer(listOfVer,maxDegreeOfVer,dArrDegreeOfVid,hLevelPtrEmbeddingAtALevel.noElemEmbedding)); //tìm bậc lớn nhất
	if(status==-1){
		PMS_PRINT("\n findMaxDegreeOfVer() in forwardExtension() failed");
		goto Error;
	}
	//Tạo mảng dArrV để ghi nhận những mở rộng hợp lệ. 
	V *dArrV=nullptr; //cần đổi tên biến thành hArrV, vì đây là bộ nhớ ở host chứ không phải device.
	dArrV = (V*)malloc(sizeof(V)); //4. Need free

	dArrV->noElem =maxDegreeOfVer*hLevelPtrEmbeddingAtALevel.noElemEmbedding;
	CHECK(cudaStatus=cudaMalloc((void**)&(dArrV->valid),(dArrV->noElem)*sizeof(int))); //2. Need cudaFree
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus = cudaMemset(dArrV->valid,0,(dArrV->noElem)*sizeof(int)));
		if(cudaStatus !=cudaSuccess){
			status =-1;
			goto Error;
		}
	}
	CHECK(cudaStatus=cudaMalloc((void**)&(dArrV->backward),(dArrV->noElem)*sizeof(int))); //3. Need cudaFree
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(dArrV->backward,0,(dArrV->noElem)*sizeof(int)));
		if(cudaStatus !=cudaSuccess){
			status =-1;
			goto Error;
		}
	}


	//////Các mở rộng hợp lệ sẽ được ghi nhận vào mảng dArrV, đồng thời thông tin của cạnh mở rộng gồm dfscode, vgi, vgj và row pointer của nó cũng được xây dựng
	//////và lưu trữ trong mảng EXT *dExtensionTemp, mảng này có số lượng phần tử bằng với mảng dArrV. Sau đó chúng ta sẽ rút trích những mở rộng hợp lệ này và lưu vào dExt. 
	//////Để xây dựng dfscode (vi,vj,li,lij,lj) thì chúng ta cần:
	////// - Dựa vào giá trị của right most path để xác định vi
	////// - Dựa vào maxid để xác định vj
	////// - Dựa vào CSDL để xác định các thành phần còn lại.
	//////Chúng ta có thể giải phóng bộ nhớ của dExtensionTemp sau khi đã trích các mở rộng hợp lệ thành công.


	EXT *dArrExtensionTemp= nullptr; //Nơi lưu trữ tạm thời tất cả các cạnh mở rộng. Sau đó, chúng sẽ được lọc ra các mở rộng hợp lệ sang EXTk tương ứng.
	CHECK(cudaStatus = cudaMalloc((void**)&dArrExtensionTemp,(dArrV->noElem)*sizeof(EXT))); //5. Need cudaFree
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(dArrExtensionTemp,0,dArrV->noElem*sizeof(EXT)));
		if(cudaStatus !=cudaSuccess){
			status =-1;
			goto Error;
		}
	}

	PMS_PRINT("\n\n noElem_dArrV:%d",dArrV->noElem );



	////Gọi kernel với các đối số: CSDL, bậc của các đỉnh, dArrV, dArrExtension,noElem_Embedding,maxDegreeOfVer,idxQ,dArrPointerEmbedding,minLabel,maxid
	dim3 block(blocksize);
	dim3 grid((hLevelPtrEmbeddingAtALevel.noElemEmbedding+block.x - 1)/block.x);
	//kernelFindValidForwardExtension<<<grid,block>>>(hLevelPtrEmbeddingAtALevel.dArrPointerEmbedding,hLevelPtrEmbeddingAtALevel.noElem,hLevelPtrEmbeddingAtALevel.noElemEmbedding,hdb.at(0).dO,hdb.at(0).dLO,hdb.at(0).dN,hdb.at(0).dLN,dArrDegreeOfVid,maxDegreeOfVer,dArrExtensionTemp,listOfVer,minLabel,maxId,fromColumEmbedding,dArrV->valid,dArrV->backward);
	kernelFindValidFBExtension<<<grid,block>>>(hLevelPtrEmbeddingAtALevel.dArrPointerEmbedding,hLevelPtrEmbeddingAtALevel.noElem,hLevelPtrEmbeddingAtALevel.noElemEmbedding,hdb.at(0).dO,hdb.at(0).dLO,hdb.at(0).dN,hdb.at(0).dLN,dArrDegreeOfVid,maxDegreeOfVer,dArrV->valid,dArrV->backward,dArrExtensionTemp,listOfVer,minLabel,maxId,fromColumEmbedding,dArrVidOnRMP,hRMPv2.at(idxLevel).noElem-1,rmp);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus=cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	////In mảng dArrV để kiểm tra thử
	PMS_PRINT("\n****************dArrV_valid*******************\n");
	FUNCHECK(status=displayDeviceArr(dArrV->valid,dArrV->noElem));
	if(status!=0){
		goto Error;
	}

	PMS_PRINT("\n****************dArrV_backward*******************\n");
	FUNCHECK(status=displayDeviceArr(dArrV->backward,dArrV->noElem));
	if(status!=0){
		goto Error;
	}

	////Chép kết quả từ dArrExtensionTemp sang dExt
	//chúng ta cần có mảng dArrV để trích các mở rộng duy nhất
	FUNCHECK(status = displayDeviceEXT(dArrExtensionTemp,dArrV->noElem));
	if(status!=0){
		goto Error;
	}

	//Hàm này cũng gọi hàm trích các mở rộng duy nhất và tính độ hỗ trợ của chúng
	FUNCHECK(status = extractValidExtensionTodExtv2(dArrExtensionTemp,dArrV,dArrV->noElem,k));
	if(status!=0){
		goto Error;
	}

	//Giải phóng bộ nhớ

	CHECK(cudaStatus = cudaFree(dArrDegreeOfVid)); //1.
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaFree(dArrV->valid)); //2.
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaFree(dArrV->backward));//3.
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}

	free(dArrV);//4.

	CHECK(cudaStatus = cudaFree(dArrExtensionTemp)); //5.
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}

Error:
	return status;
}
//đang xét
int PMS::findValidFBExtensionv2(int *listOfVer,ptrArrEmbedding hLevelPtrEmbeddingAtALevel,int k,int fromColumEmbedding,int *dArrVidOnRMP,int *rmp)
{
	int status =0;
	cudaError_t cudaStatus;

	//Tìm bậc lớn nhất của các đỉnh cần mở rộng trong listOfVer
	int maxDegreeOfVer=0;
	float *dArrDegreeOfVid=nullptr; // 1.Need cudaFree //chứa cậc của các đỉnh trong listOfVer, dùng để duyệt qua các đỉnh lân cận trong DB
	FUNCHECK(status=findMaxDegreeOfVer(listOfVer,maxDegreeOfVer,dArrDegreeOfVid,hLevelPtrEmbeddingAtALevel.noElemEmbedding)); //tìm bậc lớn nhất
	if(status==-1)
	{
		PMS_PRINT("\n findMaxDegreeOfVer() in forwardExtension() failed");
		goto Error;
	}
	//Tạo mảng dArrV để ghi nhận những mở rộng hợp lệ. 
	V *dArrV=nullptr;
	dArrV = (V*)malloc(sizeof(V)); //4. Need free

	dArrV->noElem =maxDegreeOfVer*hLevelPtrEmbeddingAtALevel.noElemEmbedding;
	CHECK(cudaStatus=cudaMalloc((void**)&(dArrV->valid),(dArrV->noElem)*sizeof(int))); //2. Need cudaFree
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus = cudaMemset(dArrV->valid,0,(dArrV->noElem)*sizeof(int)));
		if(cudaStatus !=cudaSuccess)
		{
			status =-1;
			goto Error;
		}
	}
	CHECK(cudaStatus=cudaMalloc((void**)&(dArrV->backward),(dArrV->noElem)*sizeof(int))); //3. Need cudaFree
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(dArrV->backward,0,(dArrV->noElem)*sizeof(int)));
		if(cudaStatus !=cudaSuccess)
		{
			status =-1;
			goto Error;
		}
	}


	//////Các mở rộng hợp lệ sẽ được ghi nhận vào mảng dArrV, đồng thời thông tin của cạnh mở rộng gồm dfscode, vgi, vgj và row pointer của nó cũng được xây dựng
	//////và lưu trữ trong mảng EXT *dExtensionTemp, mảng này có số lượng phần tử bằng với mảng dArrV. Sau đó chúng ta sẽ rút trích những mở rộng hợp lệ này và lưu vào dExt. 
	//////Để xây dựng dfscode (vi,vj,li,lij,lj) thì chúng ta cần:
	////// - Dựa vào giá trị của right most path để xác định vi
	////// - Dựa vào maxid để xác định vj
	////// - Dựa vào CSDL để xác định các thành phần còn lại.
	//////Chúng ta có thể giải phóng bộ nhớ của dExtensionTemp sau khi đã trích các mở rộng hợp lệ thành công.


	EXT *dArrExtensionTemp= nullptr; //Nơi lưu trữ tạm thời tất cả các cạnh mở rộng.
	//Sau đó, chúng sẽ được lọc ra các mở rộng hợp lệ sang EXTk tương ứng.
	CHECK(cudaStatus = cudaMalloc((void**)&dArrExtensionTemp,(dArrV->noElem)*sizeof(EXT))); //5. Need cudaFree
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(dArrExtensionTemp,0,dArrV->noElem*sizeof(EXT)));
		if(cudaStatus !=cudaSuccess)
		{
			status =-1;
			goto Error;
		}
	}

	//PMS_PRINT("\n\n noElem_dArrV:%d",dArrV->noElem );
	////Gọi kernel với các đối số: CSDL, bậc của các đỉnh, dArrV, dArrExtension,noElem_Embedding,maxDegreeOfVer,idxQ,dArrPointerEmbedding,minLabel,maxid
	dim3 block(blocksize);
	dim3 grid((hLevelPtrEmbeddingAtALevel.noElemEmbedding+block.x - 1)/block.x);
	//kernelFindValidForwardExtension<<<grid,block>>>(hLevelPtrEmbeddingAtALevel.dArrPointerEmbedding,hLevelPtrEmbeddingAtALevel.noElem,hLevelPtrEmbeddingAtALevel.noElemEmbedding,hdb.at(0).dO,hdb.at(0).dLO,hdb.at(0).dN,hdb.at(0).dLN,dArrDegreeOfVid,maxDegreeOfVer,dArrExtensionTemp,listOfVer,minLabel,maxId,fromColumEmbedding,dArrV->valid,dArrV->backward);
	int noElemdArrVj=hEmbedding.at(currentColEmbedding).hBackwardEmbedding.size(); //Số lượng Backward embedding đã có.
	//int noElemOnRMP = hRMPv2.at(idxLevel).noElem;
	int noElemOnRMP = hLevelRMP.at(objLevel.prevLevel).noElem;
	//Question 1: Display 4 arrays of database related to Graph_ID = 256? không cần thiết.
	//Question 2: How many embeddings belongs to Graph_ID = 256?
	
	//Question 3: Dipslay all embeddings of DFS_Code of Graph_ID = 256?
	//Question 4: Display all valid Extension of Each Embedding related to Graph_ID = 256?
	if( noElemdArrVj==0) //Nếu noElemdArrVj=0: nghĩa là tại đỉnh có của RMP chưa có mở rộng backwark nào nên mình sẽ tìm mở rộng forward và backward tại đỉnh đó.
	{ //tồn tại mở rộng forward lẫn backward, không cần kiểm tra các backward sẵn có là cạnh nào.
		kernelFindValidFBExtension<<<grid,block>>>(hLevelPtrEmbeddingAtALevel.dArrPointerEmbedding,hLevelPtrEmbeddingAtALevel.noElem,hLevelPtrEmbeddingAtALevel.noElemEmbedding,hdb.at(0).dO,hdb.at(0).dLO,hdb.at(0).dN,hdb.at(0).dLN,dArrDegreeOfVid,maxDegreeOfVer,dArrV->valid,dArrV->backward,dArrExtensionTemp,listOfVer,minLabel,maxId,fromColumEmbedding,dArrVidOnRMP,hLevelRMP.at(objLevel.prevLevel).noElem-1,hLevelRMP.at(objLevel.prevLevel).dRMP);
		CHECK(cudaStatus=cudaDeviceSynchronize());
		if(cudaStatus!=cudaSuccess)
		{
			status=-1;
			goto Error;
		}

		CHECK(cudaStatus=cudaGetLastError());
		if(cudaStatus!=cudaSuccess)
		{
			status=-1;
			goto Error;
		}
	} /*else if(noElemdArrVj == (noElemOnRMP-2)){ //chỉ tồn tại mở rộng forward, vì đã mở rộng hết tất cả các backward có thể có rồi.

	  }*/else //tồn tại mở rộng forward lẫn backward nhưng phải kiểm tra sự tồn tại của các mở rộng backward đã có
	{
		int *dArrVj = nullptr;
		FUNCHECK(status=getVjFromDFSCODE(dArrVj,noElemdArrVj)); //Nếu backward chưa được khai thác thì noElemdArrVj=0, ==> chúng ta không được gọi hàm getVjFromDFSCODE, và chúng ta không được gọi hàm kernelFindValidFBExtensionv2 với tham số dArrVj
		if(status!=0)
		{
			goto Error;
		}
		kernelFindValidFBExtensionv3<<<grid,block>>>(hLevelPtrEmbeddingAtALevel.dArrPointerEmbedding, \
			hLevelPtrEmbeddingAtALevel.noElem, \
			hLevelPtrEmbeddingAtALevel.noElemEmbedding, \
			hdb.at(0).dO,hdb.at(0).dLO,hdb.at(0).dN,hdb.at(0).dLN, \
			dArrDegreeOfVid, \
			maxDegreeOfVer, \
			dArrV->valid,dArrV->backward,dArrExtensionTemp, \
			listOfVer,minLabel,maxId, \
			fromColumEmbedding, \
			dArrVidOnRMP, \
			hLevelRMP.at(objLevel.prevLevel).noElem-1, \
			hLevelRMP.at(objLevel.prevLevel).dRMP, \
			dArrVj, \
			noElemdArrVj);
		CHECK(cudaStatus=cudaDeviceSynchronize());
		if(cudaStatus!=cudaSuccess)
		{
			status=-1;
			goto Error;
		}

		CHECK(cudaStatus=cudaGetLastError());
		if(cudaStatus!=cudaSuccess)
		{
			status=-1;
			goto Error;
		}

		CHECK(cudaStatus=cudaFree(dArrVj));
		if(cudaStatus!=cudaSuccess)
		{
			status=-1;
			goto Error;
		}
	}

	//////In mảng dArrV để kiểm tra thử
	//PMS_PRINT("\n****************dArrV_valid*******************\n");
	//FUNCHECK(status=displayDeviceArr(dArrV->valid,dArrV->noElem));
	//if(status!=0)
	//{
	//	goto Error;
	//}

	//PMS_PRINT("\n****************dArrV_backward*******************\n");
	//FUNCHECK(status=displayDeviceArr(dArrV->backward,dArrV->noElem));
	//if(status!=0)
	//{
	//	goto Error;
	//}

	////Chép kết quả từ dArrExtensionTemp sang dExt
	//chúng ta cần có mảng dArrV để trích các mở rộng duy nhất
	/*FUNCHECK(status = displayDeviceEXT(dArrExtensionTemp,dArrV->noElem));
	if(status!=0)
	{
		goto Error;
	}*/

	//Hàm này cũng gọi hàm trích các mở rộng duy nhất và tính độ hỗ trợ của chúng
	//FUNCHECK(status = extractValidExtensionTodExtv2(dArrExtensionTemp,dArrV,dArrV->noElem,k));
	//if(status!=0){
	//	goto Error;
	//}

	FUNCHECK(status = extractValidExtensionTodExtv3(dArrExtensionTemp,dArrV,dArrV->noElem,k));
	if(status!=0)
	{
		goto Error;
	}

	//Giải phóng bộ nhớ

	CHECK(cudaStatus = cudaFree(dArrDegreeOfVid)); //1.
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaFree(dArrV->valid)); //2.
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaFree(dArrV->backward));//3.
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}

	free(dArrV);//4.

	CHECK(cudaStatus = cudaFree(dArrExtensionTemp)); //5.
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}
Error:
	return status;
}

//inuse
void PMS::findValidExtension(vector<EXTk> &vecValidEXTk)
{
	try
	{
		existBackwardInfo bwInfo;
		buildExistBackwardInfo(bwInfo);
		//1. Trích địa chỉ của các embedding column thuộc RMP.
		//Dựa vào thứ tự threads để biết vid mà thread đang xử lý thuộc RMP nào.
		if(hEm.size()==0) return;

		//Chứa các embedding column thuộc RMP
		list<int> lstRMP;
		lstRMP.push_front(hEm.size() -1);
		int nextPrevCol = hEm.back().prevCol;
		while(nextPrevCol != -1 )
		{
			lstRMP.push_front(nextPrevCol);
			nextPrevCol = hEm.at(nextPrevCol).prevCol;
		}
		//noElem column thuộc RMP chính bằng hEm.size()
		int noElemRMP = lstRMP.size();
		//DFS_CODE.noElemOnRMP = noElemRMP;
		//Get address of dArrEmbedding of Embedding Columns that belong to RMP
		Embedding** dPointerdArrEmbedding = nullptr;
		CUCHECK(cudaMalloc((void**)&dPointerdArrEmbedding,noElemRMP*sizeof(Embedding**)));
		
		//Tạo device RMP từ lstRMP
		//DFS_CODE.dRMP = nullptr;
		int *dArrRMP = nullptr;
		int *hArrRMP = nullptr;
		hArrRMP = (int*)malloc(sizeof(int)*noElemRMP);
		if(hArrRMP == nullptr) FCHECK(-1);
		for (int idxLstRMP = 0; idxLstRMP < noElemRMP; idxLstRMP++)
		{
			hArrRMP[idxLstRMP] = lstRMP.front();
			lstRMP.pop_front();
		}

		for (int rmpCol = 0; rmpCol < noElemRMP; rmpCol++)
		{
			//Mỗi phần tử của mảng dArrPointerEmbedding chứa địa chỉ của dArrEmbedding
			int idx=hArrRMP[rmpCol];
			if (hEm.at(idx).hBackwardEmbedding.size()>0)
			{
				kernelGetPointerdArrEmbedding<<<1,1>>>(hEm.at(idx).hBackwardEmbedding.back().dArrEmbedding, dPointerdArrEmbedding, rmpCol);
			}
			else
			{
				kernelGetPointerdArrEmbedding<<<1,1>>>(hEm.at(idx).dArrEmbedding, dPointerdArrEmbedding, rmpCol);
			}
		}
		CUCHECK(cudaDeviceSynchronize());
		CUCHECK(cudaGetLastError());

		

		//CUCHECK(cudaMalloc((void**)&DFS_CODE.dRMP,sizeof(int)*noElemRMP));
		CUCHECK(cudaMalloc((void**)&dArrRMP,sizeof(int)*noElemRMP));
		CUCHECK(cudaMemcpy(dArrRMP,hArrRMP,sizeof(int)*noElemRMP,cudaMemcpyHostToDevice));
		//CUCHECK(cudaMemcpy(DFS_CODE.dRMP,hArrRMP,sizeof(int)*noElemRMP,cudaMemcpyHostToDevice));

		// Tìm bậc đỉnh cao nhất ==> Chuẩn bị được bộ nhớ lưu trữ các mở rộng có thể có
		int noElemEmbedding = 0;
		if(hEm.back().hBackwardEmbedding.size()>0)
		{
			noElemEmbedding = hEm.back().hBackwardEmbedding.back().noElem;
		}
		else
		{
			noElemEmbedding = hEm.back().noElem;
		}
		//Tìm bậc của các đỉnh vid của các embeding thuộc RMP
		int noElemVid = noElemRMP*noElemEmbedding;
		float *dArrDegreeOfVid = nullptr;
		CUCHECK(cudaMalloc((void**)&dArrDegreeOfVid,noElemVid*sizeof(float)));
		dim3 block(blocksize);
		dim3 grid((noElemVid + block.x -1)/block.x);
		kernelCalDegreeOfVidOnEmbeddingColumnv2<<<grid,block>>>(dPointerdArrEmbedding, \
			hdb.at(0).dO,hdb.at(0).noElemdO,noElemRMP, noElemEmbedding, \
			hdb.at(0).noElemdN, maxOfVer,dArrDegreeOfVid);
		CUCHECK(cudaDeviceSynchronize());
		CUCHECK(cudaGetLastError());
		//FCHECK(displayDeviceArr(dArrDegreeOfVid,noElemVid));
		float *h_max;
		h_max = (float*)malloc(sizeof(float));
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
		int maxDegreeOfVer = (int)*h_max;

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
		
		kernelFindValidExtension<<<grid,block>>>( \
			dPointerdArrEmbedding, \
			dArrRMP,
			noElemRMP, \
			noElemEmbedding, \
			hdb.at(0).dO,hdb.at(0).dLO,hdb.at(0).dN,hdb.at(0).dLN, \
			dArrDegreeOfVid, \
			maxDegreeOfVer, \
			dPointerArrValid,dPointerArrEXT, \
			DFS_CODE.minLabel,DFS_CODE.maxId, \
			bwInfo.dValidBackward,bwInfo.dVj);
		CUCHECK(cudaDeviceSynchronize());
		CUCHECK(cudaGetLastError());

		//Hiển thị thông tin của vecValid
		/*for (int i = 0; i < noElemRMP; i++)
		{
			vecValid.at(i).show();
		}*/
		//6. Trích được các mở rộng FW/BW từ vecValid lưu vào vecValidEXTk
		//Khởi tạo số lượng phần tử của vecValidEXTk bằng số lượng phần tử của vecValid
		vecValidEXTk.resize(vecValid.size());
		//Duyệt qua các vecValid và trích các mở rộng hợp lệ sang vecValidEXTk tương ứng.
		for (int idxVecValid = 0; idxVecValid < vecValid.size(); idxVecValid++)
		{
			std::printf("\nMark the valid backward/forward extentions:");
			vecValid.at(idxVecValid).show();
			vecValid.at(idxVecValid).extractValid(vecValidEXTk.at(idxVecValid));
			vecValid.at(idxVecValid).ReleaseMemory();
			std::printf("\nShow the valid backward/forward extentions:");
			vecValidEXTk.at(idxVecValid).show();
			//vecValidEXTk.at(idxVecValid).ReleaseMemory();
		}
		CUCHECK(cudaDeviceSynchronize());
		CUCHECK(cudaGetLastError());
		//Chưa dùng dArrBackward của vecValid.
		vecValid.clear();
		//7. Release memory
		//if(vecValid.size()>0) {vecValid.clear();}
		CUCHECK(cudaFree(dPointerdArrEmbedding));
		CUCHECK(cudaFree(dArrDegreeOfVid));
		CUCHECK(cudaFree(d_max));
		CUCHECK(cudaFree(d_mutex));
		CUCHECK(cudaFree(dArrRMP));
		CUCHECK(cudaFree(dPointerArrValid));
		CUCHECK(cudaFree(dPointerArrEXT));
		free(hArrRMP);
		free(h_max);
		lstRMP.clear();
		CUCHECK(cudaFree(bwInfo.dValidBackward));
		CUCHECK(cudaFree(bwInfo.dVj));
	} catch (...)
	{
		cout<<endl<<"Somethings Error"<<endl;
		FCHECK(-1);
	}
}

int PMS::findValidForwardExtensionForNonLastSegment(int *listOfVer,ptrArrEmbedding hLevelPtrEmbeddingAtALevel,int k,int fromColumEmbedding,int *dArrVidOnRMP,int *rmp){
	int status =0;
	cudaError_t cudaStatus;

	//Tìm bậc lớn nhất của các đỉnh cần mở rộng trong listOfVer
	int maxDegreeOfVer=0;
	float *dArrDegreeOfVid=nullptr; // 1.Need cudaFree //chứa cậc của các đỉnh trong listOfVer, dùng để duyệt qua các đỉnh lân cận trong DB
	FUNCHECK(status=findMaxDegreeOfVer(listOfVer,maxDegreeOfVer,dArrDegreeOfVid,hLevelPtrEmbeddingAtALevel.noElemEmbedding)); //tìm bậc lớn nhất
	if(status==-1)
	{
		PMS_PRINT("\n findMaxDegreeOfVer() in forwardExtension() failed");
		goto Error;
	}
	//Tạo mảng dArrV để ghi nhận những mở rộng hợp lệ. 
	V *dArrV=nullptr;
	dArrV = (V*)malloc(sizeof(V)); //4. Need free

	dArrV->noElem =maxDegreeOfVer*hLevelPtrEmbeddingAtALevel.noElemEmbedding;
	CHECK(cudaStatus=cudaMalloc((void**)&(dArrV->valid),(dArrV->noElem)*sizeof(int))); //2. Need cudaFree
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus = cudaMemset(dArrV->valid,0,(dArrV->noElem)*sizeof(int)));
		if(cudaStatus !=cudaSuccess)
		{
			status =-1;
			goto Error;
		}
	}
	CHECK(cudaStatus=cudaMalloc((void**)&(dArrV->backward),(dArrV->noElem)*sizeof(int))); //3. Need cudaFree
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(dArrV->backward,0,(dArrV->noElem)*sizeof(int)));
		if(cudaStatus !=cudaSuccess)
		{
			status =-1;
			goto Error;
		}
	}


	//////Các mở rộng hợp lệ sẽ được ghi nhận vào mảng dArrV, đồng thời thông tin của cạnh mở rộng gồm dfscode, vgi, vgj và row pointer của nó cũng được xây dựng
	//////và lưu trữ trong mảng EXT *dExtensionTemp, mảng này có số lượng phần tử bằng với mảng dArrV. Sau đó chúng ta sẽ rút trích những mở rộng hợp lệ này và lưu vào dExt. 
	//////Để xây dựng dfscode (vi,vj,li,lij,lj) thì chúng ta cần:
	////// - Dựa vào giá trị của right most path để xác định vi
	////// - Dựa vào maxid để xác định vj
	////// - Dựa vào CSDL để xác định các thành phần còn lại.
	//////Chúng ta có thể giải phóng bộ nhớ của dExtensionTemp sau khi đã trích các mở rộng hợp lệ thành công.


	EXT *dArrExtensionTemp= nullptr; //Nơi lưu trữ tạm thời tất cả các cạnh mở rộng. Sau đó, chúng sẽ được lọc ra các mở rộng hợp lệ sang EXTk tương ứng.
	CHECK(cudaStatus = cudaMalloc((void**)&dArrExtensionTemp,(dArrV->noElem)*sizeof(EXT))); //5. Need cudaFree
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(dArrExtensionTemp,0,dArrV->noElem*sizeof(EXT)));
		if(cudaStatus !=cudaSuccess)
		{
			status =-1;
			goto Error;
		}
	}

	PMS_PRINT("\n\n noElem_dArrV:%d",dArrV->noElem );
	////Gọi kernel với các đối số: CSDL, bậc của các đỉnh, dArrV, dArrExtension,noElem_Embedding,maxDegreeOfVer,idxQ,dArrPointerEmbedding,minLabel,maxid
	dim3 block(blocksize);
	dim3 grid((hLevelPtrEmbeddingAtALevel.noElemEmbedding+block.x - 1)/block.x);
	//kernelFindValidForwardExtension<<<grid,block>>>(hLevelPtrEmbeddingAtALevel.dArrPointerEmbedding,hLevelPtrEmbeddingAtALevel.noElem,hLevelPtrEmbeddingAtALevel.noElemEmbedding,hdb.at(0).dO,hdb.at(0).dLO,hdb.at(0).dN,hdb.at(0).dLN,dArrDegreeOfVid,maxDegreeOfVer,dArrExtensionTemp,listOfVer,minLabel,maxId,fromColumEmbedding,dArrV->valid,dArrV->backward);
	//int noElemdArrVj=hEmbedding.at(currentColEmbedding).hBackwardEmbedding.size(); 
	//int noElemOnRMP = hRMPv2.at(idxLevel).noElem;
	//if( noElemdArrVj==0){ //tồn tại mở rộng forward lẫn backward, không cần kiểm tra các backward sẵn có là cạnh nào.
	//	kernelFindValidFBExtension<<<grid,block>>>(hLevelPtrEmbeddingAtALevel.dArrPointerEmbedding,hLevelPtrEmbeddingAtALevel.noElem,hLevelPtrEmbeddingAtALevel.noElemEmbedding,hdb.at(0).dO,hdb.at(0).dLO,hdb.at(0).dN,hdb.at(0).dLN,dArrDegreeOfVid,maxDegreeOfVer,dArrV->valid,dArrV->backward,dArrExtensionTemp,listOfVer,minLabel,maxId,fromColumEmbedding,dArrVidOnRMP,hRMPv2.at(idxLevel).noElem-1,rmp);
	//	CHECK(cudaStatus=cudaDeviceSynchronize());
	//	if(cudaStatus!=cudaSuccess){
	//		status=-1;
	//		goto Error;
	//	}

	//	CHECK(cudaStatus=cudaGetLastError());
	//	if(cudaStatus!=cudaSuccess){
	//		status=-1;
	//		goto Error;
	//	}
	//} /*else if(noElemdArrVj == (noElemOnRMP-2)){ //chỉ tồn tại mở rộng forward, vì đã mở rộng hết tất cả các backward có thể có rồi.

	//}*/else //tồn tại mở rộng forward lẫn backward nhưng phải kiểm tra sự tồn tại của các mở rộng backward đã có
	//{
	//	int *dArrVj = nullptr;
	//	FUNCHECK(status=getVjFromDFSCODE(dArrVj,noElemdArrVj)); //Nếu backward chưa được khai thác thì noElemdArrVj=0, ==> chúng ta không được gọi hàm getVjFromDFSCODE, và chúng ta không được gọi hàm kernelFindValidFBExtensionv2 với tham số dArrVj
	//	if(status!=0){
	//		goto Error;
	//	}
	//	kernelFindValidFBExtensionv3<<<grid,block>>>(hLevelPtrEmbeddingAtALevel.dArrPointerEmbedding,hLevelPtrEmbeddingAtALevel.noElem,hLevelPtrEmbeddingAtALevel.noElemEmbedding,hdb.at(0).dO,hdb.at(0).dLO,hdb.at(0).dN,hdb.at(0).dLN,dArrDegreeOfVid,maxDegreeOfVer,dArrV->valid,dArrV->backward,dArrExtensionTemp,listOfVer,minLabel,maxId,fromColumEmbedding,dArrVidOnRMP,hRMPv2.at(idxLevel).noElem-1,rmp,dArrVj,noElemdArrVj);
	//	CHECK(cudaStatus=cudaDeviceSynchronize());
	//	if(cudaStatus!=cudaSuccess){
	//		status=-1;
	//		goto Error;
	//	}

	//	CHECK(cudaStatus=cudaGetLastError());
	//	if(cudaStatus!=cudaSuccess){
	//		status=-1;
	//		goto Error;
	//	}

	//CHECK(cudaStatus=cudaFree(dArrVj));
	//if(cudaStatus!=cudaSuccess){
	//	status=-1;
	//	goto Error;
	//}
	//}

	kernelFindValidForwardExtensionv3<<<grid,block>>>(hLevelPtrEmbeddingAtALevel.dArrPointerEmbedding,hLevelPtrEmbeddingAtALevel.noElem,hLevelPtrEmbeddingAtALevel.noElemEmbedding,hdb.at(0).dO,hdb.at(0).dLO,hdb.at(0).dN,hdb.at(0).dLN,dArrDegreeOfVid,maxDegreeOfVer,dArrV->valid,dArrV->backward,dArrExtensionTemp,listOfVer,minLabel,maxId,fromColumEmbedding,dArrVidOnRMP,hLevelRMP.at(objLevel.prevLevel).noElem - 1,rmp);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus=cudaGetLastError());
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	////In mảng dArrV để kiểm tra thử
	PMS_PRINT("\n****************dArrV_valid*******************\n");
	FUNCHECK(status=displayDeviceArr(dArrV->valid,dArrV->noElem));
	if(status!=0)
	{
		goto Error;
	}

	PMS_PRINT("\n****************dArrV_backward*******************\n");
	FUNCHECK(status=displayDeviceArr(dArrV->backward,dArrV->noElem));
	if(status!=0)
	{
		goto Error;
	}

	////Chép kết quả từ dArrExtensionTemp sang dExt
	//chúng ta cần có mảng dArrV để trích các mở rộng duy nhất
	FUNCHECK(status = displayDeviceEXT(dArrExtensionTemp,dArrV->noElem));
	if(status!=0)
	{
		goto Error;
	}

	//Hàm này cũng gọi hàm trích các mở rộng duy nhất và tính độ hỗ trợ của chúng
	//FUNCHECK(status = extractValidExtensionTodExtv2(dArrExtensionTemp,dArrV,dArrV->noElem,k));
	//if(status!=0){
	//	goto Error;
	//}

	/*FUNCHECK(status = extractValidExtensionTodExtv3(dArrExtensionTemp,dArrV,dArrV->noElem,k));
	if(status!=0)
	{
	goto Error;
	}*/

	//Giải phóng bộ nhớ

	CHECK(cudaStatus = cudaFree(dArrDegreeOfVid)); //1.
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaFree(dArrV->valid)); //2.
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaFree(dArrV->backward));//3.
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}

	free(dArrV);//4.

	CHECK(cudaStatus = cudaFree(dArrExtensionTemp)); //5.
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}

Error:
	return status;
}

int PMS::findForwardExtension(int *listOfVer, \
							  ptrArrEmbedding hLevelPtrEmbeddingAtALevel, \
							  int k,int fromColumEmbedding, \
							  int *dArrVidOnRMP, \
							  int *rmp)
{
	int status =0;
	cudaError_t cudaStatus;

	//Tìm bậc lớn nhất của các đỉnh cần mở rộng trong listOfVer
	// 1.Need cudaFree //chứa cậc của các đỉnh trong listOfVer, dùng để duyệt qua các đỉnh lân cận trong DB
	int maxDegreeOfVer=0;
	float *dArrDegreeOfVid=nullptr; 
	//Tìm bậc lớn nhất.
	FUNCHECK(status=findMaxDegreeOfVer(listOfVer, \
		maxDegreeOfVer, \
		dArrDegreeOfVid, \
		hLevelPtrEmbeddingAtALevel.noElemEmbedding)); 
	if(status==-1)
	{
		PMS_PRINT("\n findMaxDegreeOfVer() in forwardExtension() failed");
		goto Error;
	}
	//Tạo mảng dArrV để ghi nhận những mở rộng hợp lệ. 
	V *dArrV=nullptr;
	dArrV = (V*)malloc(sizeof(V)); //4. Need free

	dArrV->noElem =maxDegreeOfVer*hLevelPtrEmbeddingAtALevel.noElemEmbedding;
	CHECK(cudaStatus=cudaMalloc((void**)&(dArrV->valid),(dArrV->noElem)*sizeof(int))); //2. Need cudaFree
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus = cudaMemset(dArrV->valid,0,(dArrV->noElem)*sizeof(int)));
		if(cudaStatus !=cudaSuccess)
		{
			status =-1;
			goto Error;
		}
	}
	CHECK(cudaStatus=cudaMalloc((void**)&(dArrV->backward),(dArrV->noElem)*sizeof(int))); //3. Need cudaFree
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(dArrV->backward,0,(dArrV->noElem)*sizeof(int)));
		if(cudaStatus !=cudaSuccess)
		{
			status =-1;
			goto Error;
		}
	}


	//////Các mở rộng hợp lệ sẽ được ghi nhận vào mảng dArrV, đồng thời thông tin của cạnh mở rộng gồm dfscode, vgi, vgj và row pointer của nó cũng được xây dựng
	//////và lưu trữ trong mảng EXT *dExtensionTemp, mảng này có số lượng phần tử bằng với mảng dArrV. Sau đó chúng ta sẽ rút trích những mở rộng hợp lệ này và lưu vào dExt. 
	//////Để xây dựng dfscode (vi,vj,li,lij,lj) thì chúng ta cần:
	////// - Dựa vào giá trị của right most path để xác định vi
	////// - Dựa vào maxid để xác định vj
	////// - Dựa vào CSDL để xác định các thành phần còn lại.
	//////Chúng ta có thể giải phóng bộ nhớ của dExtensionTemp sau khi đã trích các mở rộng hợp lệ thành công.

	 //Nơi lưu trữ tạm thời tất cả các cạnh mở rộng. Sau đó, chúng sẽ được lọc ra các mở rộng hợp lệ sang EXTk tương ứng.
	EXT *dArrExtensionTemp= nullptr;
	CHECK(cudaStatus = cudaMalloc((void**)&dArrExtensionTemp,(dArrV->noElem)*sizeof(EXT))); //5. Need cudaFree
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(dArrExtensionTemp,0,dArrV->noElem*sizeof(EXT)));
		if(cudaStatus !=cudaSuccess)
		{
			status =-1;
			goto Error;
		}
	}

	//PMS_PRINT("\n\n noElem_dArrV:%d",dArrV->noElem );
	////Gọi kernel với các đối số: CSDL, bậc của các đỉnh, dArrV, dArrExtension,noElem_Embedding,maxDegreeOfVer,idxQ,dArrPointerEmbedding,minLabel,maxid
	dim3 block(blocksize);
	dim3 grid((hLevelPtrEmbeddingAtALevel.noElemEmbedding+block.x - 1)/block.x);

	kernelFindValidForwardExtensionv3<<<grid,block>>>( \
													  hLevelPtrEmbeddingAtALevel.dArrPointerEmbedding,\
													  hLevelPtrEmbeddingAtALevel.noElem, \
													  hLevelPtrEmbeddingAtALevel.noElemEmbedding, \
													  hdb.at(0).dO,hdb.at(0).dLO,hdb.at(0).dN,hdb.at(0).dLN, \
													  dArrDegreeOfVid, \
													  maxDegreeOfVer, \
													  dArrV->valid, \
													  dArrV->backward,\
													  dArrExtensionTemp, \
													  listOfVer, \
													  minLabel, \
													  maxId, \
													  fromColumEmbedding, \
													  dArrVidOnRMP, \
													  hLevelRMP.at(objLevel.prevLevel).noElem - 1, \
													  rmp);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus=cudaGetLastError());
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	////In mảng dArrV để kiểm tra thử
	/*PMS_PRINT("\n****************dArrV_valid*******************\n");
	FUNCHECK(status=displayDeviceArr(dArrV->valid,dArrV->noElem));
	if(status!=0)
	{
		goto Error;
	}

	PMS_PRINT("\n****************dArrV_backward*******************\n");
	FUNCHECK(status=displayDeviceArr(dArrV->backward,dArrV->noElem));
	if(status!=0)
	{
		goto Error;
	}*/

	////Chép kết quả từ dArrExtensionTemp sang dExt
	//chúng ta cần có mảng dArrV để trích các mở rộng duy nhất
	/*FUNCHECK(status = displayDeviceEXT(dArrExtensionTemp,dArrV->noElem));
	if(status!=0)
	{
		goto Error;
	}*/

	//Hàm này cũng gọi hàm trích các mở rộng duy nhất và tính độ hỗ trợ của chúng
	//FUNCHECK(status = extractValidExtensionTodExtv2(dArrExtensionTemp,dArrV,dArrV->noElem,k));
	//if(status!=0){
	//	goto Error;
	//}

	FUNCHECK(status = extractValidExtensionTodExtv3(dArrExtensionTemp,dArrV,dArrV->noElem,k));
	if(status!=0)
	{
		goto Error;
	}

	//Giải phóng bộ nhớ

	CHECK(cudaStatus = cudaFree(dArrDegreeOfVid)); //1.
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaFree(dArrV->valid)); //2.
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaFree(dArrV->backward));//3.
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}

	free(dArrV);//4.

	CHECK(cudaStatus = cudaFree(dArrExtensionTemp)); //5.
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}

Error:
	return status;
}


int PMS::findValidForwardExtensionv2(int *listOfVer,ptrArrEmbedding hLevelPtrEmbeddingAtALevel,int k,int fromColumEmbedding,int *dArrVidOnRMP,int *rmp){
	int status =0;
	cudaError_t cudaStatus;

	//Tìm bậc lớn nhất của các đỉnh cần mở rộng trong listOfVer
	int maxDegreeOfVer=0;
	float *dArrDegreeOfVid=nullptr; // 1.Need cudaFree //chứa cậc của các đỉnh trong listOfVer, dùng để duyệt qua các đỉnh lân cận trong DB
	FUNCHECK(status=findMaxDegreeOfVer(listOfVer,maxDegreeOfVer,dArrDegreeOfVid,hLevelPtrEmbeddingAtALevel.noElemEmbedding)); //tìm bậc lớn nhất
	if(status!=0){
		PMS_PRINT("\n findMaxDegreeOfVer() in forwardExtension() failed");
		goto Error;
	}
	//Tạo mảng dArrV để ghi nhận những mở rộng hợp lệ. 
	V *dArrV=nullptr;
	dArrV = (V*)malloc(sizeof(V)); //4. Need free

	dArrV->noElem =maxDegreeOfVer*hLevelPtrEmbeddingAtALevel.noElemEmbedding;
	CHECK(cudaStatus=cudaMalloc((void**)&(dArrV->valid),(dArrV->noElem)*sizeof(int))); //2. Need cudaFree
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus = cudaMemset(dArrV->valid,0,(dArrV->noElem)*sizeof(int)));
		if(cudaStatus !=cudaSuccess){
			status =-1;
			goto Error;
		}
	}
	CHECK(cudaStatus=cudaMalloc((void**)&(dArrV->backward),(dArrV->noElem)*sizeof(int))); //3. Need cudaFree
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(dArrV->backward,0,(dArrV->noElem)*sizeof(int)));
		if(cudaStatus !=cudaSuccess){
			status =-1;
			goto Error;
		}
	}

	//////Các mở rộng hợp lệ sẽ được ghi nhận vào mảng dArrV, đồng thời thông tin của cạnh mở rộng gồm dfscode, vgi, vgj và row pointer của nó cũng được xây dựng
	//////và lưu trữ trong mảng EXT *dExtensionTemp, mảng này có số lượng phần tử bằng với mảng dArrV. Sau đó chúng ta sẽ rút trích những mở rộng hợp lệ này và lưu vào dExt. 
	//////Để xây dựng dfscode (vi,vj,li,lij,lj) thì chúng ta cần:
	////// - Dựa vào giá trị của right most path để xác định vi
	////// - Dựa vào maxid để xác định vj
	////// - Dựa vào CSDL để xác định các thành phần còn lại.
	//////Chúng ta có thể giải phóng bộ nhớ của dExtensionTemp sau khi đã trích các mở rộng hợp lệ thành công.

	EXT *dArrExtensionTemp= nullptr; //Nơi lưu trữ tạm thời tất cả các cạnh mở rộng. Sau đó, chúng sẽ được lọc ra các mở rộng hợp lệ sang EXTk tương ứng.
	CHECK(cudaStatus = cudaMalloc((void**)&dArrExtensionTemp,(dArrV->noElem)*sizeof(EXT))); //5. Need cudaFree
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(dArrExtensionTemp,0,dArrV->noElem*sizeof(EXT)));
		if(cudaStatus !=cudaSuccess){
			status =-1;
			goto Error;
		}
	}

	PMS_PRINT("\n\n noElem_dArrV:%d",dArrV->noElem );

	////Gọi kernel với các đối số: CSDL, bậc của các đỉnh, dArrV, dArrExtension,noElem_Embedding,maxDegreeOfVer,idxQ,dArrPointerEmbedding,minLabel,maxid
	dim3 block(blocksize);
	dim3 grid((hLevelPtrEmbeddingAtALevel.noElemEmbedding+block.x - 1)/block.x);
	kernelFindValidForwardExtension<<<grid,block>>>(hLevelPtrEmbeddingAtALevel.dArrPointerEmbedding,hLevelPtrEmbeddingAtALevel.noElem,hLevelPtrEmbeddingAtALevel.noElemEmbedding,hdb.at(0).dO,hdb.at(0).dLO,hdb.at(0).dN,hdb.at(0).dLN,dArrDegreeOfVid,maxDegreeOfVer,dArrExtensionTemp,listOfVer,minLabel,maxId,fromColumEmbedding,dArrV->valid,dArrV->backward);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus=cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	////In mảng dArrV để kiểm tra thử
	PMS_PRINT("\n****************dArrV_valid*******************\n");
	FUNCHECK(status=displayDeviceArr(dArrV->valid,dArrV->noElem));
	if(status!=0){
		goto Error;
	}

	//PMS_PRINT("\n****************dArrV_backward*******************\n");
	//FUNCHECK(status=displayDeviceArr(dArrV->backward,dArrV->noElem));
	//if(status!=0){
	//	goto Error;
	//}

	////Chép kết quả từ dArrExtensionTemp sang dExt
	//chúng ta cần có mảng dArrV để trích các mở rộng duy nhất
	FUNCHECK(status = displayDeviceEXT(dArrExtensionTemp,dArrV->noElem));
	if(status!=0){
		goto Error;
	}

	//Hàm này cũng gọi hàm trích các mở rộng duy nhất và tính độ hỗ trợ của chúng
	//FUNCHECK(status = extractValidExtensionTodExtv2(dArrExtensionTemp,dArrV,dArrV->noElem,k));
	//if(status!=0){
	//	goto Error;
	//}

	FUNCHECK(status = extractValidExtensionTodExtv3(dArrExtensionTemp,dArrV,dArrV->noElem,k));
	if(status!=0){
		goto Error;
	}

	//Giải phóng bộ nhớ

	CHECK(cudaStatus = cudaFree(dArrDegreeOfVid)); //1.
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaFree(dArrV->valid)); //2.
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaFree(dArrV->backward));//3.
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}

	free(dArrV);//4.

	CHECK(cudaStatus = cudaFree(dArrExtensionTemp)); //5.
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}

Error:
	return status;
}


int PMS::getVjFromDFSCODE(int *&dArrVj,int noElemdArrVj){ //Lấy vj từ DFS_code để làm gì? ==>> để dựa vào đó tìm các mở rộng backward hợp lệ. 
	int status = 0; //Nhưng cụ thể làm như thế nào?
	cudaError_t cudaStatus;
	CHECK(cudaStatus = cudaMalloc((void**)&dArrVj,noElemdArrVj*sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(dArrVj,0,noElemdArrVj*sizeof(int)));
		if(cudaStatus!=cudaSuccess){
			status=-1;
			goto Error;
		}
	}

	int *vj = (int*)malloc(noElemdArrVj*sizeof(int));
	if(vj==NULL){
		status=-1;
		PMS_PRINT("\n Malloc vj failed\n");
		goto Error;
	}


	int idx=DFS_CODE.size()-1;
	//Dựa vào số lượng phần tử hBackwardEmbedding.size() hiện có để duyệt và lấy ra vj của DFS_CODE
	for (int i = 0; i < noElemdArrVj; i++,idx--)
	{
		vj[i]=DFS_CODE.at(idx).to; //Lấy vi để làm gì? cách làm này có đúng không? Giải thuật cụ thể là gì?
	}
	//Chép vj từ host sang device. s
	CHECK(cudaStatus=cudaMemcpy(dArrVj,vj,sizeof(int)*noElemdArrVj,cudaMemcpyHostToDevice));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	PMS_PRINT("\n******* dArrVj ********\n");
	FUNCHECK(status=displayDeviceArr(dArrVj,noElemdArrVj));
	if(status!=0){
		goto Error;
	}

	free(vj);

Error:
	return status;
}

int PMS::FSMiningv2() //đã có Embedding mới và RMP tương ứng với nó. Khai thác các mở rộng
{
	int status = 0;
	cudaError_t cudaStatus;
	FUNCHECK(status = updateRMP()); //Cập nhật cho vector hRMPv2
	if(status!=0){
		goto Error;
	}
	//2. Lấy số lượng đỉnh trên right most path
	int noElemVerOnRMP = hRMPv2.at(idxLevel).noElem; //1.need pop_back this level //right most path chứa bao nhiêu đỉnh 

	//3.Tìm danh sách các đỉnh thuộc right most path của các embedding
	//3.1 xây dựng dArrPointerEmbedding. hEmbedding là bộ nhớ trên host. Muốn khai thác trên device thì cần phải xây dựng một bộ nhớ trên device.
	FUNCHECK(status = buildArrPointerEmbedding(hEmbedding,hLevelPtrEmbeddingv2)); //4. need cudaFree and pop_back Level//xây dựng dArrPointerEmbedding dựa vào hEmbedding
	if(status!=0){
		goto Error;
	}
	//3.2 xây dựng rmp on device dựa vào hRMPv2
	int *rmp = nullptr; //3. need cudaFree //rigt most path trên bộ nhớ device
	FUNCHECK(status = buildrmpOnDevice(hRMPv2.at(idxLevel),rmp));
	if(status!=0){
		goto Error;
	}
	PMS_PRINT("\n\n ******* rmp *********\n");
	FUNCHECK(status=displayDeviceArr(rmp,noElemVerOnRMP));
	if(status!=0){
		goto Error;
	}

	//3.3. Dựa vào dArrPointerEmbedding và rmp để tìm các đỉnh cần mở rộng
	//	FUNCHECK(status = findListVer(hLevelPtrEmbeddingv2.at(idxLevel).dArrPointerEmbedding,hLevelPtrEmbeddingv2.at(idxLevel).noElemEmbedding,rmp,noElemVerOnRMP,hListVerv2));
	if(status !=0){
		goto Error;
	}
	//hiển thị danh sách đỉnh
	PMS_PRINT("\n\n ********* listVer *********\n");
	FUNCHECK(status=displayDeviceArr(hListVerv2.at(idxLevel).dListVer,hListVerv2.at(idxLevel).noElem)); //5. need cudaFree and pop_back
	if(status!=0){
		goto Error;
	}



	//Chuẩn bị bộ nhớ ở Level mới
	int noElemEXTk =noElemVerOnRMP; //Số lượng phần tử EXTk bằng số lượng đỉnh trên right most path
	//hEXTk.resize(noElemEXTk); //Các mở rộng hợp lệ từ đỉnh k sẽ được lưu trữ vào EXTk tương ứng.

	//Quản lý theo Level phục vụ cho khai thác đệ quy
	hLevelEXTv2.resize(Level); //8. clear and pop_back //Khởi tạo vector quản lý bộ nhớ cho level
	hLevelEXTv2.at(idxLevel).noElem = noElemVerOnRMP; //Cập nhật số lượng phần tử vector vE bằng số lượng đỉnh trên RMP
	hLevelEXTv2.at(idxLevel).vE.resize(noElemVerOnRMP); //Cấp phát bộ nhớ cho vector vE.

	hLevelUniEdgev2.resize(Level); //Số lượng phần tử UniEdge cũng giống với EXT
	hLevelUniEdgev2.at(idxLevel).noElem=noElemVerOnRMP;
	hLevelUniEdgev2.at(idxLevel).vUE.resize(noElemVerOnRMP);

	hLevelUniEdgeSatisfyMinsupv2.resize(Level);
	hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).noElem= noElemVerOnRMP;
	hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.resize(noElemVerOnRMP);


	int *tempListVerCol = nullptr; //6. need cudaFree //chứa danh sách các đỉnh cần mở rộng thuộc một embedding column cụ thể.
	CHECK(cudaStatus = cudaMalloc((void**)&tempListVerCol,hLevelPtrEmbeddingv2.at(idxLevel).noElemEmbedding * sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	int *dArrVidOnRMP = nullptr; //7. need cudaFree //lưu trữ các đỉnh trên RMP của Embedding, có kích thước nhỏ hơn  đỉnh so với RMP

	if(hRMPv2.at(idxLevel).noElem>2)
	{ //Nếu số lượng đỉnh trên RMP lớn hơn 2 thì mới tồn tại backward. Vì ở đây chỉ xét đơn đồ thị vô hướng
		//dArrVidOnRMP: chứa các đỉnh thuộc RMP của mỗi Embedding. Dùng để kiểm tra sự tồn tại của đỉnh trên right most path
		//khi tìm các mở rộng backward. Chỉ dùng tới khi right most path có 3 đỉnh trở lên (chỉ xét trong đơn đồ thị vô hướng).

		FUNCHECK(status = findVerOnRMPForBWCheck(hLevelPtrEmbeddingv2.at(idxLevel),rmp,noElemVerOnRMP,dArrVidOnRMP));
		if(status!=0){
			goto Error;
		}
		//Duyệt qua các cột của Embedding để tìm các mở rộng hợp lệ
		for (int i = 0; i < noElemVerOnRMP ; i++)
		{
			int colEmbedding = hRMPv2.at(idxLevel).hArrRMP.at(i); //Tìm mở rộng cho các đỉnh tại vị trí colEmbedding trong vector hEmbedding
			currentColEmbedding=colEmbedding; //Đang mở rộng từ cột nào của Embedding. Được dùng để cập nhật prevCol, phục vụ cho việc xây dựng Right Most Path
			int k = i; //lưu vào Extk với k = i; K=0 đại diện cho EXT0: là EXT cuối
			dim3 block(blocksize);
			dim3 grid((hLevelPtrEmbeddingv2.at(idxLevel).noElemEmbedding + block.x -1)/block.x);
			//Trích các đỉnh cần mở rộng của cột Embedding đang xét
			kernelExtractFromListVer<<<grid,block>>>(hListVerv2.at(idxLevel).dListVer,i*hLevelPtrEmbeddingv2.at(idxLevel).noElemEmbedding,hLevelPtrEmbeddingv2.at(idxLevel).noElemEmbedding,tempListVerCol);//trích các đỉnh từ listVer, từ vị trí i*noElemEmbedding,trích noElemEmbedding phần tử, bỏ vào tempListVerCol
			CHECK(cudaStatus=cudaDeviceSynchronize());
			if(cudaStatus!=cudaSuccess){
				status=-1;
				goto Error;
			}
			CHECK(cudaStatus = cudaGetLastError());
			if(cudaStatus!=cudaSuccess){
				status=-1;
				goto Error;
			}
			PMS_PRINT("\n ****** tempListVerCol ***********\n");
			FUNCHECK(status=displayDeviceArr(tempListVerCol,hLevelPtrEmbeddingv2.at(idxLevel).noElemEmbedding));
			if(status!=0){
				goto Error;
			}
			//Gọi hàm tìm các mở rộng hợp lệ từ đỉnh cuối i=0
			if(i==0){ //khai thác forward lẫn backward. với các tham số:
				//1. Mở rộng từ danh sách đỉnh nào, 2. embedding trên device,3. trên EXTk nào,4.colembedding: để cập nhật vi cho mở rộng mới
				FUNCHECK(status = findValidFBExtension(tempListVerCol,hLevelPtrEmbeddingv2.at(idxLevel),k,colEmbedding,dArrVidOnRMP,rmp));
				if(status!=0){
					goto Error;
				}				
			}
			else
			{
				//chỉ khai thác forward, cần bổ sung hàm khai thác forward ở đây. Thử gọi luôn hàm ở trên để tìm luôn.
				FUNCHECK(status = findValidFBExtension(tempListVerCol,hLevelPtrEmbeddingv2.at(idxLevel),k,colEmbedding,dArrVidOnRMP,rmp));
				if(status!=0){
					goto Error;
				}	
			}
		}
		//free memory
		CHECK(cudaStatus=cudaFree(dArrVidOnRMP)); //7.
		if(cudaStatus!=cudaSuccess){
			status=-1;
			goto Error;
		}
	} //Kết thúc việc xử lý các mở rộng có 3 đỉnh trong RMP (vì nó có backward extension).
	else //Ngược lại, chỉ khai thác forward khi số lượng đỉnh trên RMP chỉ bằng 2.
	{
		for (int i = 0; i < noElemVerOnRMP ; i++)
		{
			int colEmbedding = hRMPv2.at(idxLevel).hArrRMP.at(i); //Tìm mở rộng cho các đỉnh tại vị trí colEmbedding trong vector hEmbedding
			currentColEmbedding=colEmbedding; //Đang mở rộng từ cột nào của Embedding. Được dùng để cập nhật prevCol, phục vụ cho việc xây dựng Right Most Path
			int k = i; //lưu vào Extk với k = i; K=0 đại diện cho EXT0: là EXT cuối
			dim3 block(blocksize);
			dim3 grid((hLevelPtrEmbeddingv2.at(idxLevel).noElemEmbedding + block.x -1)/block.x);
			//Trích các đỉnh cần mở rộng của cột Embedding đang xét
			kernelExtractFromListVer<<<grid,block>>>(hListVerv2.at(idxLevel).dListVer,i*hLevelPtrEmbeddingv2.at(idxLevel).noElemEmbedding,hLevelPtrEmbeddingv2.at(idxLevel).noElemEmbedding,tempListVerCol);//trích các đỉnh từ listVer, từ vị trí i*noElemEmbedding,trích noElemEmbedding phần tử, bỏ vào tempListVerCol
			CHECK(cudaStatus=cudaDeviceSynchronize());
			if(cudaStatus!=cudaSuccess){
				status=-1;
				goto Error;
			}
			CHECK(cudaStatus = cudaGetLastError());
			if(cudaStatus!=cudaSuccess){
				status=-1;
				goto Error;
			}
			PMS_PRINT("\n ****** tempListVerCol ***********\n");
			FUNCHECK(status=displayDeviceArr(tempListVerCol,hLevelPtrEmbeddingv2.at(idxLevel).noElemEmbedding));
			if(status!=0){
				goto Error;
			}
			FUNCHECK(status = findValidFBExtension(tempListVerCol,hLevelPtrEmbeddingv2.at(idxLevel),k,colEmbedding,dArrVidOnRMP,rmp));
			if(status!=0){
				goto Error;
			}				
		}
		//free memory
		CHECK(cudaStatus=cudaFree(dArrVidOnRMP)); //7.
		if(cudaStatus!=cudaSuccess){
			status=-1;
			goto Error;
		}
	}

	//Giải phóng bộ nhớ
	hRMPv2.at(idxLevel).hArrRMP.clear(); //1.xoá bộ nhớ vector hArrRMP
	hRMPv2.pop_back(); //2.xoá phần tử cuối của vector hRMPv2

	CHECK(cudaStatus=cudaFree(rmp)); //3. xoá right most path trên device.
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus=cudaFree(hLevelPtrEmbeddingv2.at(idxLevel).dArrPointerEmbedding)); //4. cudaFree
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	hLevelPtrEmbeddingv2.pop_back(); //4. pop_back

	CHECK(cudaStatus=cudaFree(hListVerv2.at(idxLevel).dListVer)); //5. cudaFree
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	hListVerv2.pop_back(); //5.pop_back

	CHECK(cudaStatus=cudaFree(tempListVerCol)); //6.
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	if(hLevelEXTv2.at(idxLevel).noElem>0){ //8. clear and pop_back
		hLevelEXTv2.at(idxLevel).vE.clear();
		hLevelEXTv2.pop_back();
	}
	if(hLevelUniEdgev2.at(idxLevel).noElem>0){
		hLevelUniEdgev2.at(idxLevel).vUE.clear();
		hLevelUniEdgev2.pop_back();
	}
	if(hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).noElem>0){
		hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.clear();
		hLevelUniEdgeSatisfyMinsupv2.pop_back();
	}
Error:
	return status;
}

int PMS::FSMiningv3(int bw) //đã có Embedding mới và RMP tương ứng với nó. Khai thác các mở rộng
{
	int status = 0;
	cudaError_t cudaStatus;
	FUNCHECK(status = updateRMP()); //Cập nhật cho vector hRMPv2. Nếu là backward thì vẫn đúng. Nhưng vẫn phải tạo Level mới. 
	if(status!=0){
		goto Error;
	}
	//2. Lấy số lượng đỉnh trên right most path
	int noElemVerOnRMP = hRMPv2.at(idxLevel).noElem; //1.need pop_back this level //right most path chứa bao nhiêu đỉnh 

	//3.Tìm danh sách các đỉnh thuộc right most path của các embedding
	//3.1 xây dựng dArrPointerEmbedding. hEmbedding là bộ nhớ trên host. Muốn khai thác trên device thì cần phải xây dựng một bộ nhớ trên device.
	if(bw!=1){
		FUNCHECK(status = buildArrPointerEmbedding(hEmbedding,hLevelPtrEmbeddingv2)); //4. need cudaFree and pop_back Level//xây dựng dArrPointerEmbedding dựa vào hEmbedding
		if(status!=0){
			goto Error;
		}
	}
	else
	{
		FUNCHECK(status = buildArrPointerEmbeddingbw(hEmbedding,hLevelPtrEmbeddingv2)); //4. need cudaFree and pop_back Level//xây dựng dArrPointerEmbedding dựa vào hEmbedding
		if(status!=0){
			goto Error;
		}
	}

	//3.2 xây dựng rmp on device dựa vào hRMPv2
	int *rmp = nullptr; //3. need cudaFree //rigt most path trên bộ nhớ device
	FUNCHECK(status = buildrmpOnDevice(hRMPv2.at(idxLevel),rmp));
	if(status!=0){
		goto Error;
	}
	PMS_PRINT("\n\n ******* rmp *********\n");
	FUNCHECK(status=displayDeviceArr(rmp,noElemVerOnRMP));
	if(status!=0){
		goto Error;
	}

	//3.3. Dựa vào dArrPointerEmbedding và rmp để tìm các đỉnh cần mở rộng
	//	FUNCHECK(status = findListVer(hLevelPtrEmbeddingv2.at(idxLevel).dArrPointerEmbedding,hLevelPtrEmbeddingv2.at(idxLevel).noElemEmbedding,rmp,noElemVerOnRMP,hListVerv2));
	if(status !=0){
		goto Error;
	}
	//hiển thị danh sách đỉnh
	PMS_PRINT("\n\n ********* listVer *********\n");
	FUNCHECK(status= displayDeviceArr(hListVerv2.at(idxLevel).dListVer,hListVerv2.at(idxLevel).noElem)); //5. need cudaFree and pop_back
	if(status!=0){
		goto Error;
	}


	//Chuẩn bị bộ nhớ ở Level mới
	int noElemEXTk =noElemVerOnRMP; //Số lượng phần tử EXTk bằng số lượng đỉnh trên right most path
	//hEXTk.resize(noElemEXTk); //Các mở rộng hợp lệ từ đỉnh k sẽ được lưu trữ vào EXTk tương ứng.

	//Quản lý theo Level phục vụ cho khai thác đệ quy
	hLevelEXTv2.resize(Level); //8. clear and pop_back //Khởi tạo vector quản lý bộ nhớ cho level
	hLevelEXTv2.at(idxLevel).noElem = noElemVerOnRMP; //Cập nhật số lượng phần tử vector vE bằng số lượng đỉnh trên RMP
	hLevelEXTv2.at(idxLevel).vE.resize(noElemVerOnRMP); //Cấp phát bộ nhớ cho vector vE.

	hLevelUniEdgev2.resize(Level); //Số lượng phần tử UniEdge cũng giống với EXT
	hLevelUniEdgev2.at(idxLevel).noElem=noElemVerOnRMP;
	hLevelUniEdgev2.at(idxLevel).vUE.resize(noElemVerOnRMP);

	hLevelUniEdgeSatisfyMinsupv2.resize(Level);
	hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).noElem= noElemVerOnRMP;
	hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.resize(noElemVerOnRMP);


	int *tempListVerCol = nullptr; //6. need cudaFree //chứa danh sách các đỉnh cần mở rộng thuộc một embedding column cụ thể.
	CHECK(cudaStatus = cudaMalloc((void**)&tempListVerCol,hLevelPtrEmbeddingv2.at(idxLevel).noElemEmbedding * sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}

	if(hRMPv2.at(idxLevel).noElem>2){ //Nếu số lượng đỉnh trên RMP lớn hơn 2 thì mới tồn tại backward. Vì ở đây chỉ xét đơn đồ thị vô hướng
		//dArrVidOnRMP: chứa các đỉnh thuộc RMP của mỗi Embedding. Dùng để kiểm tra sự tồn tại của đỉnh trên right most path
		//khi tìm các mở rộng backward. Chỉ dùng tới khi right most path có 3 đỉnh trở lên (chỉ xét trong đơn đồ thị vô hướng).
		int *dArrVidOnRMP = nullptr; //7. need cudaFree //lưu trữ các đỉnh trên RMP của Embedding, có kích thước nhỏ hơn  đỉnh so với RMP
		FUNCHECK(status = findVerOnRMPForBWCheck(hLevelPtrEmbeddingv2.at(idxLevel),rmp,noElemVerOnRMP,dArrVidOnRMP));
		if(status!=0){
			goto Error;
		}
		//Duyệt qua các cột của Embedding để tìm các mở rộng hợp lệ
		for (int i = 0; i < noElemVerOnRMP ; i++)
		{
			int colEmbedding = hRMPv2.at(idxLevel).hArrRMP.at(i); //Tìm mở rộng cho các đỉnh tại vị trí colEmbedding trong vector hEmbedding
			currentColEmbedding=colEmbedding; //Đang mở rộng từ cột nào của Embedding. Được dùng để cập nhật prevCol, phục vụ cho việc xây dựng Right Most Path
			int k = i; //lưu vào Extk với k = i; K=0 đại diện cho EXT0: là EXT cuối
			dim3 block(blocksize);
			dim3 grid((hLevelPtrEmbeddingv2.at(idxLevel).noElemEmbedding + block.x -1)/block.x);
			//Trích các đỉnh cần mở rộng của cột Embedding đang xét
			kernelExtractFromListVer<<<grid,block>>>(hListVerv2.at(idxLevel).dListVer,i*hLevelPtrEmbeddingv2.at(idxLevel).noElemEmbedding,hLevelPtrEmbeddingv2.at(idxLevel).noElemEmbedding,tempListVerCol);//trích các đỉnh từ listVer, từ vị trí i*noElemEmbedding,trích noElemEmbedding phần tử, bỏ vào tempListVerCol
			cudaDeviceSynchronize();
			CHECK(cudaStatus = cudaGetLastError());
			if(cudaStatus!=cudaSuccess){
				status=-1;
				goto Error;
			}

			PMS_PRINT("\n ****** tempListVerCol ***********\n");
			FUNCHECK(status=displayDeviceArr(tempListVerCol,hLevelPtrEmbeddingv2.at(idxLevel).noElemEmbedding));
			if(status!=0){
				goto Error;
			}

			//Gọi hàm tìm các mở rộng hợp lệ từ đỉnh cuối i=0
			if(i==0){ //khai thác forward lẫn backward. với các tham số:
				//1. Mở rộng từ danh sách đỉnh nào, 2. embedding trên device,3. trên EXTk nào,4.colembedding: để cập nhật vi cho mở rộng mới
				FUNCHECK(status = findValidFBExtensionv2(tempListVerCol,hLevelPtrEmbeddingv2.at(idxLevel),k,colEmbedding,dArrVidOnRMP,rmp));
				if(status!=0){
					goto Error;
				}				
			}
			else
			{
				//chỉ khai thác forward. Cần phải code cho phần này.
			}
		}

		//free memory
		CHECK(cudaStatus=cudaFree(dArrVidOnRMP)); //7.
		if(cudaStatus!=cudaSuccess){
			status=-1;
			goto Error;
		}
	}
	else //Ngược lại, chỉ khai thác forward khi số lượng đỉnh trên RMP chỉ bằng 2.
	{
		//for (int i = 0; i < noElemVerOnRMP ; i++)
		//{
		//	int colEmbedding = hRMPv2.at(idxLevel).hArrRMP.at(i); //Tìm mở rộng cho các đỉnh tại vị trí colEmbedding trong vector hEmbedding
		//	currentColEmbedding=colEmbedding; //Đang mở rộng từ cột nào của Embedding. Được dùng để cập nhật prevCol, phục vụ cho việc xây dựng Right Most Path
		//	int k = i; //lưu vào Extk với k = i; K=0 đại diện cho EXT0: là EXT cuối
		//	dim3 block(blocksize);
		//	dim3 grid((hLevelPtrEmbeddingv2.at(idxLevel).noElemEmbedding + block.x -1)/block.x);
		//	kernelExtractFromListVer<<<grid,block>>>(hListVerv2.at(idxLevel).dListVer,i*hLevelPtrEmbeddingv2.at(idxLevel).noElemEmbedding,hLevelPtrEmbeddingv2.at(idxLevel).noElemEmbedding,tempListVerCol);//trích các đỉnh từ listVer, từ vị trí i*noElemEmbedding,trích noElemEmbedding phần tử, bỏ vào tempListVerCol
		//	cudaDeviceSynchronize();
		//	CHECK(cudaStatus = cudaGetLastError());
		//	if(cudaStatus!=cudaSuccess){
		//		status=-1;
		//		goto Error;
		//	}
		//	PMS_PRINT("\n ****** tempListVerCol ***********\n");
		//	displayDeviceArr(tempListVerCol,hLevelPtrEmbedding.at(idxLevel).noElemEmbedding);
		//	//gọi hàm forwardExtension để tìm các mở rộng forward từ cột colEmbedding, lưu kết quả vào hEXTk tại vị trí k, với các đỉnh
		//	// cần mở rộng là tempListVerCol, thuộc righ most path
		//	////Hàm này cũng đồng thời trích các mở rộng duy nhất từ các EXT và lưu vào UniEdge
		//	//Hàm này cũng gọi đệ quy FSMining bên trong
		//	FUNCHECK(status = forwardExtension(k,tempListVerCol,hLevelPtrEmbeddingv2.at(idxLevel).noElemEmbedding,hRMPv2.at(idxLevel).hArrRMP.at(i)));
		//	if(status ==-1){
		//		goto Error;
		//	}
		//}
	}

	//Giải phóng bộ nhớ
	hRMPv2.at(idxLevel).hArrRMP.clear(); //1.xoá bộ nhớ vector hArrRMP
	hRMPv2.pop_back(); //2.xoá phần tử cuối của vector hRMPv2

	CHECK(cudaStatus=cudaFree(rmp)); //3. xoá right most path trên device.
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus=cudaFree(hLevelPtrEmbeddingv2.at(idxLevel).dArrPointerEmbedding)); //4. cudaFree
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	hLevelPtrEmbeddingv2.pop_back(); //4. pop_back

	CHECK(cudaStatus=cudaFree(hListVerv2.at(idxLevel).dListVer)); //5. cudaFree
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	hListVerv2.pop_back(); //5.pop_back

	CHECK(cudaStatus=cudaFree(tempListVerCol)); //6.
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	if(hLevelEXTv2.at(idxLevel).noElem>0){ //8. clear and pop_back
		hLevelEXTv2.at(idxLevel).vE.clear();
		hLevelEXTv2.pop_back();
	}
	if(hLevelUniEdgev2.at(idxLevel).noElem>0){
		hLevelUniEdgev2.at(idxLevel).vUE.clear();
		hLevelUniEdgev2.pop_back();
	}
	if(hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).noElem>0){
		hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.clear();
		hLevelUniEdgeSatisfyMinsupv2.pop_back();
	}


	--Level; //Giảm Level
	idxLevel=Level-1;	//Cập nhật lại idxLevel 

Error:
	return status;
}

int PMS::FSMiningv4(int bw) //đã có Embedding mới và RMP tương ứng với nó. Khai thác các mở rộng
{
	int status = 0;
	cudaError_t cudaStatus;

	//Nếu mở rộng đang xét là backward thì right most path giống Level trước.
	//(we dont need to update right most path for backward extension).
	//Chúng ta chỉ tìm một right most path mới khi mở rộng đang xét là forward.
	//(We only need to upate the right most path for the forward extension.
	if(bw!=1){ //Nếu mở rộng đang xét không phải là backward extension thì phải tìm right most path cho forward extension đó.
		FUNCHECK(status = updateRMP()); //Cập nhật cho vector hRMPv2. Nếu là backward thì vẫn đúng. Nhưng vẫn phải tạo Level mới. 
		if(status!=0){
			goto Error;
		}
	}
	else
	{
		FUNCHECK(status = updateRMPBW()); //Cập nhật cho vector hRMPv2. Nếu là backward thì vẫn đúng. Nhưng vẫn phải tạo Level mới. 
		if(status!=0){
			goto Error;
		}
	}


	//2. Lấy số lượng đỉnh trên right most path
	int noElemVerOnRMP = hRMPv2.at(idxLevel).noElem; //1.need pop_back this level //right most path chứa bao nhiêu đỉnh 

	//3.Tìm danh sách các đỉnh thuộc right most path của các embedding
	//3.1 xây dựng dArrPointerEmbedding. hEmbedding là bộ nhớ trên host. Muốn khai thác trên device thì cần phải xây dựng một bộ nhớ trên device.

	//if(bw!=1){
	FUNCHECK(status = buildArrPointerEmbeddingv2(hEmbedding,hLevelPtrEmbeddingv2)); //4. need cudaFree and pop_back Level//xây dựng dArrPointerEmbedding dựa vào hEmbedding
	if(status!=0){
		goto Error;
	}
	//Kiểm tra xem việc xây dựng embedding trên device đã đúng chưa
	FUNCHECK(status=displaydArrPointerEmbedding(hLevelPtrEmbeddingv2.at(idxLevel).dArrPointerEmbedding,hLevelPtrEmbeddingv2.at(idxLevel).noElem,hLevelPtrEmbeddingv2.at(idxLevel).noElemEmbedding));
	if(status!=0){
		goto Error;
	}
	//}
	//else
	//{
	//FUNCHECK(status = buildArrPointerEmbeddingbw(hEmbedding,hLevelPtrEmbeddingv2)); //4. need cudaFree and pop_back Level//xây dựng dArrPointerEmbedding dựa vào hEmbedding
	//if(status!=0){
	//	goto Error;
	//}
	//}

	//3.2 xây dựng rmp on device dựa vào hRMPv2
	int *rmp = nullptr; //3. need cudaFree //rigt most path trên bộ nhớ device
	FUNCHECK(status = buildrmpOnDevice(hRMPv2.at(idxLevel),rmp));
	if(status!=0){
		goto Error;
	}
	PMS_PRINT("\n\n ******* rmp *********\n");
	FUNCHECK(status=displayDeviceArr(rmp,noElemVerOnRMP));
	if(status!=0){
		goto Error;
	}

	//3.3. Dựa vào dArrPointerEmbedding và rmp để tìm các đỉnh cần mở rộng
	//	FUNCHECK(status = findListVer(hLevelPtrEmbeddingv2.at(idxLevel).dArrPointerEmbedding,hLevelPtrEmbeddingv2.at(idxLevel).noElemEmbedding,rmp,noElemVerOnRMP,hListVerv2));
	if(status !=0){
		goto Error;
	}
	//hiển thị danh sách đỉnh
	PMS_PRINT("\n\n ********* listVer *********\n");
	FUNCHECK(status= displayDeviceArr(hListVerv2.at(idxLevel).dListVer,hListVerv2.at(idxLevel).noElem)); //5. need cudaFree and pop_back
	if(status!=0){
		goto Error;
	}


	//Chuẩn bị bộ nhớ ở Level mới
	int noElemEXTk =noElemVerOnRMP; //Số lượng phần tử EXTk bằng số lượng đỉnh trên right most path
	//hEXTk.resize(noElemEXTk); //Các mở rộng hợp lệ từ đỉnh k sẽ được lưu trữ vào EXTk tương ứng.

	//Quản lý theo Level phục vụ cho khai thác đệ quy
	hLevelEXTv2.resize(Level); //8. clear and pop_back //Khởi tạo vector quản lý bộ nhớ cho level
	hLevelEXTv2.at(idxLevel).noElem = noElemVerOnRMP; //Cập nhật số lượng phần tử vector vE bằng số lượng đỉnh trên RMP
	hLevelEXTv2.at(idxLevel).vE.resize(noElemVerOnRMP); //Cấp phát bộ nhớ cho vector vE.

	hLevelUniEdgev2.resize(Level); //Số lượng phần tử UniEdge cũng giống với EXT
	hLevelUniEdgev2.at(idxLevel).noElem=noElemVerOnRMP;
	hLevelUniEdgev2.at(idxLevel).vUE.resize(noElemVerOnRMP);

	hLevelUniEdgeSatisfyMinsupv2.resize(Level);
	hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).noElem= noElemVerOnRMP;
	hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.resize(noElemVerOnRMP);


	int *tempListVerCol = nullptr; //6. need cudaFree //chứa danh sách các đỉnh cần mở rộng thuộc một embedding column cụ thể.
	CHECK(cudaStatus = cudaMalloc((void**)&tempListVerCol,hLevelPtrEmbeddingv2.at(idxLevel).noElemEmbedding * sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	int *dArrVidOnRMP = nullptr; //7. need cudaFree //lưu trữ các đỉnh trên RMP của Embedding, có kích thước nhỏ hơn  đỉnh so với RMP
	FUNCHECK(status = findVerOnRMPForBWCheckv2(hLevelPtrEmbeddingv2.at(idxLevel),rmp,noElemVerOnRMP,dArrVidOnRMP));
	if(status!=0){
		goto Error;
	}

	if(hRMPv2.at(idxLevel).noElem>2){ //Nếu số lượng đỉnh trên RMP lớn hơn 2 thì mới tồn tại backward. Vì ở đây chỉ xét đơn đồ thị vô hướng
		//dArrVidOnRMP: chứa các đỉnh thuộc RMP của mỗi Embedding. Dùng để kiểm tra sự tồn tại của đỉnh trên right most path
		//khi tìm các mở rộng backward. Chỉ dùng tới khi right most path có 3 đỉnh trở lên (chỉ xét trong đơn đồ thị vô hướng).
		//int *dArrVidOnRMP = nullptr; //7. need cudaFree //lưu trữ các đỉnh trên RMP của Embedding, có kích thước nhỏ hơn  đỉnh so với RMP
		//FUNCHECK(status = findVerOnRMPForBWCheckv2(hLevelPtrEmbeddingv2.at(idxLevel),rmp,noElemVerOnRMP,dArrVidOnRMP));
		//if(status!=0){
		//	goto Error;
		//}
		//Duyệt qua các cột của Embedding để tìm các mở rộng hợp lệ
		for (int i = 0; i < noElemVerOnRMP ; i++)
		{
			int colEmbedding = hRMPv2.at(idxLevel).hArrRMP.at(i);//Tìm mở rộng cho các đỉnh tại vị trí colEmbedding trong vector hEmbedding
			cout<<endl<<"Extend from vertex: "<<colEmbedding <<" at idxLevel "<<idxLevel;

			currentColEmbedding=colEmbedding; //Đang mở rộng từ cột nào của Embedding. Được dùng để cập nhật prevCol, phục vụ cho việc xây dựng Right Most Path
			int k = i; //lưu vào Extk với k = i; K=0 đại diện cho EXT0: là EXT cuối
			dim3 block(blocksize);
			dim3 grid((hLevelPtrEmbeddingv2.at(idxLevel).noElemEmbedding + block.x -1)/block.x);
			//Trích các đỉnh cần mở rộng của cột Embedding đang xét
			kernelExtractFromListVer<<<grid,block>>>(hListVerv2.at(idxLevel).dListVer,i*hLevelPtrEmbeddingv2.at(idxLevel).noElemEmbedding,hLevelPtrEmbeddingv2.at(idxLevel).noElemEmbedding,tempListVerCol);//trích các đỉnh từ listVer, từ vị trí i*noElemEmbedding,trích noElemEmbedding phần tử, bỏ vào tempListVerCol
			CHECK(cudaStatus=cudaDeviceSynchronize());
			if(cudaStatus!=cudaSuccess){
				status=-1;
				goto Error;
			}

			CHECK(cudaStatus = cudaGetLastError());
			if(cudaStatus!=cudaSuccess){
				status=-1;
				goto Error;
			}

			PMS_PRINT("\n ****** tempListVerCol ***********\n");
			FUNCHECK(status=displayDeviceArr(tempListVerCol,hLevelPtrEmbeddingv2.at(idxLevel).noElemEmbedding));
			if(status!=0){
				goto Error;
			}

			//Gọi hàm tìm các mở rộng hợp lệ từ đỉnh cuối i=0
			if(i==0){ //khai thác forward lẫn backward. với các tham số:
				//1. Mở rộng từ danh sách đỉnh nào, 2. embedding trên device,3. trên EXTk nào,4.colembedding: để cập nhật vi cho mở rộng mới
				FUNCHECK(status = findValidFBExtensionv2(tempListVerCol,hLevelPtrEmbeddingv2.at(idxLevel),k,colEmbedding,dArrVidOnRMP,rmp));
				if(status!=0){
					goto Error;
				}				
			}
			else
			{
				//chỉ khai thác forward, need to complete this function now.
				FUNCHECK(status = findValidForwardExtensionForNonLastSegment(tempListVerCol,hLevelPtrEmbeddingv2.at(idxLevel),k,colEmbedding,dArrVidOnRMP,rmp)); //ze lui tới k=2 thì bị lỗi
				if(status!=0){
					goto Error;
				}				
			}
		}

		////free memory
		//CHECK(cudaStatus=cudaFree(dArrVidOnRMP)); //7.
		//if(cudaStatus!=cudaSuccess){
		//	status=-1;
		//	goto Error;
		//}
	}
	else //Ngược lại, chỉ khai thác forward khi số lượng đỉnh trên RMP chỉ bằng 2.
	{
		for (int i = 0; i < noElemVerOnRMP ; i++)
		{
			int colEmbedding = hRMPv2.at(idxLevel).hArrRMP.at(i); //Tìm mở rộng cho các đỉnh tại vị trí colEmbedding trong vector hEmbedding
			currentColEmbedding=colEmbedding; //Đang mở rộng từ cột nào của Embedding. Được dùng để cập nhật prevCol, phục vụ cho việc xây dựng Right Most Path
			int k = i; //lưu vào Extk với k = i; K=0 đại diện cho EXT0: là EXT cuối
			dim3 block(blocksize);
			dim3 grid((hLevelPtrEmbeddingv2.at(idxLevel).noElemEmbedding + block.x -1)/block.x);
			//Trích các đỉnh cần mở rộng của cột Embedding đang xét
			kernelExtractFromListVer<<<grid,block>>>(hListVerv2.at(idxLevel).dListVer,i*hLevelPtrEmbeddingv2.at(idxLevel).noElemEmbedding,hLevelPtrEmbeddingv2.at(idxLevel).noElemEmbedding,tempListVerCol);//trích các đỉnh từ listVer, từ vị trí i*noElemEmbedding,trích noElemEmbedding phần tử, bỏ vào tempListVerCol
			CHECK(cudaStatus=cudaDeviceSynchronize());
			if(cudaStatus!=cudaSuccess){
				status=-1;
				goto Error;
			}

			CHECK(cudaStatus = cudaGetLastError());
			if(cudaStatus!=cudaSuccess){
				status=-1;
				goto Error;
			}

			PMS_PRINT("\n ****** tempListVerCol ***********\n");
			FUNCHECK(status=displayDeviceArr(tempListVerCol,hLevelPtrEmbeddingv2.at(idxLevel).noElemEmbedding));
			if(status!=0){
				goto Error;
			}

			FUNCHECK(status = findValidForwardExtensionForNonLastSegment(tempListVerCol,hLevelPtrEmbeddingv2.at(idxLevel),k,colEmbedding,dArrVidOnRMP,rmp));
			if(status!=0){
				goto Error;
			}				
		}
	}
	//free memory
	CHECK(cudaStatus=cudaFree(dArrVidOnRMP)); //7.
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	//Giải phóng bộ nhớ
	hRMPv2.at(idxLevel).hArrRMP.clear(); //1.xoá bộ nhớ right most path tại level đó
	hRMPv2.pop_back(); //2.xoá phần tử cuối của vector hRMPv2

	CHECK(cudaStatus=cudaFree(rmp)); //3. Giải phóng bộ nhớ right most path trên device
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus=cudaFree(hLevelPtrEmbeddingv2.at(idxLevel).dArrPointerEmbedding)); //4. cudaFree. Giải phóng bộ nhớ các pointer trỏ đến các embedding columns trên device
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	hLevelPtrEmbeddingv2.pop_back(); //4. pop_back, gỡ bỏ phần tử quản lý pointer embedding ở level cuối

	CHECK(cudaStatus=cudaFree(hListVerv2.at(idxLevel).dListVer)); //5. cudaFree. Gỡ bỏ danh sách các đỉnh thuộc trên các embedding column trên device
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	hListVerv2.pop_back(); //5.pop_back. Gỡ bỏ phần tử quản lý các đỉnh thuộc embedding column trên device

	CHECK(cudaStatus=cudaFree(tempListVerCol)); //6.Giải phóng bộ nhớ tạm lưu trữ các đỉnh của một embedding column cần mở rộng
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	if(hLevelEXTv2.at(idxLevel).noElem>0){ //8. clear and pop_back. Vì đã khai thác tất cả các đỉnh của DFS_CODE tại Level này.
		hLevelEXTv2.at(idxLevel).vE.clear(); //Xoá bộ nhớ EXT và gỡ bỏ vectỏ quản lý tại level hiện tại. Mảng device đã được giải phóng của vE đã được giải phóng rồi, nên chúng ta không sợ.
		hLevelEXTv2.pop_back();
	}
	if(hLevelUniEdgev2.at(idxLevel).noElem>0){
		hLevelUniEdgev2.at(idxLevel).vUE.clear();
		hLevelUniEdgev2.pop_back();
	}
	if(hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).noElem>0){
		hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.clear();
		hLevelUniEdgeSatisfyMinsupv2.pop_back();
	}

	cout<<endl<<"****** Decrease Level ****";
	--Level; //Giảm Level
	idxLevel=Level-1;	//Cập nhật lại idxLevel 

Error:
	return status;
}


__global__ void kernelExtractFromListVer(int *listVer,int from,int noElemEmbedding,int *tempListVerCol){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElemEmbedding){
		tempListVerCol[i] = listVer[from+i];
	}
}

//kernel tìm các mở rộng hợp lệ và ghi nhận vào mảng dArrV và dArrExtension tương ứng.
__global__ void kernelFindValidForwardExtension(Embedding **dArrPointerEmbedding,int noElem_dArrPointerEmbedding,int noElem_Embedding,int *d_O,int *d_LO,int *d_N,int *d_LN,float *dArrDegreeOfVid,int maxDegreeOfVer,EXT *dArrExtension,int *listOfVer,int minLabel,int maxId,int fromRMP,int *dArrV_valid,int *dArrV_backward){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	//Duyệt qua các Embedding và xét các mở rộng cho đỉnh tại vị trí idxQ
	if(i<noElem_Embedding){
		int posColumn =noElem_dArrPointerEmbedding-1;
		int posRow=i;
		int col = posColumn;
		int row = posRow;
		int vid = listOfVer[i];
		int degreeVid=__float2int_rn(dArrDegreeOfVid[i]);
		//Duyệt qua các đỉnh kề với đỉnh vid dựa vào số lần duyệt là bậc
		int indexToVidIndN=d_O[vid];
		int labelFromVid = d_LO[vid];
		int toVid;
		int labelToVid;
		bool b=true;
		for (int j = 0; j < degreeVid; j++,indexToVidIndN++) //Duyệt qua tất cả các đỉnh kề với đỉnh vid, nếu đỉnh không thuộc embedding thì --> cạnh cũng không thuộc embedding vì đây là Q cuối
		{			
			toVid=d_N[indexToVidIndN]; //Lấy vid của đỉnh cần kiểm tra
			labelToVid = d_LO[toVid]; //lấy label của đỉnh cần kiểm tra
			posColumn=col;
			posRow=row;
			Embedding *Q=dArrPointerEmbedding[posColumn];
			//PMS_PRINT("\nThread %d, j: %d has ToVidLabel:%d",i,j,labelToVid);
			//1. Trước tiên kiểm tra nhãn của labelToVid có nhỏ hơn minLabel hay không. Nếu nhỏ hơn thì return
			if(labelToVid<minLabel) continue;
			//2. kiểm tra xem đỉnh toVid có tồn tại trong embedding hay không nếu tồn tại thì return
			//Duyệt qua embedding column từ Q cuối đến Q đầu, lần lượt lấy vid so sánh với toVid

			//PMS_PRINT("\n Q[%d] Row[%d] (idx:%d vid:%d)",posColumn,posRow,Q[posRow].idx,Q[posRow].vid);//Q[1][0]
			if(toVid==Q[posRow].vid) continue;
			//PMS_PRINT("\nj:%d toVid:%d Q.vid:%d",j,toVid,Q[posRow].vid);

			while (true)
			{
				posRow = Q[posRow].idx;//0
				posColumn=posColumn-1;		//0
				Q=dArrPointerEmbedding[posColumn];
				//PMS_PRINT("\n posColumn[%d] Row[%d] (idx:%d vid:%d)",posColumn,posRow,Q[posRow].idx,Q[posRow].vid);//Q[0][0]
				//PMS_PRINT("\nj:%d toVid:%d Q.vid:%d",j,toVid,Q[posRow].vid);
				if(toVid==Q[posRow].vid) {
					b=false; break;
				}
				posRow=Q[posRow].idx;//-1
				//PMS_PRINT("\nposRow:%d",posRow);
				if(posRow==-1) break;
			}
			if (b==false){b=true; continue;}
			int indexOfd_arr_V=i*maxDegreeOfVer+j;
			//PMS_PRINT("\nThread %d: m:%d",i,maxDegreeOfVer);
			int indexOfd_LN=indexToVidIndN;
			//dArrV[indexOfd_arr_V].valid=1;
			dArrV_valid[indexOfd_arr_V]=1;
			dArrV_backward[indexOfd_arr_V]=0;
			//PMS_PRINT("\ndArrV[%d].valid:%d",indexOfd_arr_V,dArrV[indexOfd_arr_V].valid);
			//cập nhật dữ liệu cho mảng dArrExtension
			dArrExtension[indexOfd_arr_V].vgi=vid;
			dArrExtension[indexOfd_arr_V].vgj=toVid;
			dArrExtension[indexOfd_arr_V].lij=d_LN[indexOfd_LN];
			//PMS_PRINT("\n");
			//PMS_PRINT("d_LN[%d]:%d ",indexOfd_LN,d_LN[indexOfd_LN]);
			dArrExtension[indexOfd_arr_V].li=labelFromVid;
			dArrExtension[indexOfd_arr_V].lj=labelToVid;
			dArrExtension[indexOfd_arr_V].vi=fromRMP;
			dArrExtension[indexOfd_arr_V].vj=maxId+1;
			//dArrExtension[indexOfd_arr_V].posColumn=idxQ;
			dArrExtension[indexOfd_arr_V].posRow=row;
		}
	}
}

//consider 2
//kernel tìm các mở rộng hợp lệ và ghi nhận vào mảng dArrV và dArrExtension tương ứng.
//fromRMP: dùng để cập nhật vi
//rmp: dùng để cập nhật vj
__global__ void kernelFindValidFBExtension(Embedding **dArrPointerEmbedding,int noElem_dArrPointerEmbedding,int noElem_Embedding,int *d_O,int *d_LO,int *d_N,int *d_LN,float *dArrDegreeOfVid,int maxDegreeOfVer,int *dArrV_valid,int *dArrV_backward,EXT *dArrExtension,int *listOfVer,int minLabel,int maxId,int fromRMP, int *dArrVidOnRMP,int segdArrVidOnRMP,int *rmp){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElem_Embedding){ //Mỗi một thread i sẽ tìm mở rộng cho một đỉnh của Embedding.
		//if(i==0){
		//int fromPosCol;
		//int idxRMP=0;
		//int noELemVerOnRMP = segdArrVidOnRMP +1;
		int posColumn =noElem_dArrPointerEmbedding-1; 
		int posRow=i; //Phải giữ lại posColumn và posRow để cập nhật thông tin trong EXT<k>. Khi nhìn vào EXT<k> thì sẽ biết được mở rộng đó là từ embedding nào, ở dòng nào trong Embedding Column cuối cùng.
		int col = posColumn;
		int row = posRow;
		//Embedding *Q=dArrPointerEmbedding[idxQ];
		int vid = listOfVer[i]; //Lấy đỉnh cần mở rộng.
		int degreeVid=__float2int_rn(dArrDegreeOfVid[i]); //Lấy bậc của đỉnh đang xét.
		//Duyệt qua các đỉnh kề với đỉnh vid dựa vào số lần duyệt là bậc
		int indexToVidIndN=d_O[vid]; //Lấy index trong mảng nhãn cạnh
		int labelFromVid = d_LO[vid]; //Lấy nhãn của đỉnh được mở rộng
		int toVid;
		int labelToVid;
		bool b;
		for (int j = 0; j < degreeVid; ++j,++indexToVidIndN) //Duyệt qua tất cả các đỉnh kề với đỉnh vid, nếu đỉnh không thuộc embedding thì --> cạnh cũng không thuộc embedding vì đây là Q cuối, Nếu đỉnh không thuộc Embedding thì nó cũng không phải là backward
		{		
			b=false;
			//1.Kiểm tra forward
			toVid=d_N[indexToVidIndN]; //Lấy vid của đỉnh cần kiểm tra
			labelToVid = d_LO[toVid]; //lấy label của đỉnh cần kiểm tra
			posColumn=col;
			posRow=row;
			Embedding *Q=dArrPointerEmbedding[posColumn];
			//PMS_PRINT("\nThread %d, j: %d has ToVidLabel:%d",i,j,labelToVid);
			//1. Trước tiên kiểm tra nhãn của labelToVid có nhỏ hơn minLabel hay không. Nếu nhỏ hơn thì return. Vì nó cũng không có khả năng là backward extension
			if(labelToVid<minLabel) 
				continue;
			//goto backward;
			//2. kiểm tra xem đỉnh toVid có tồn tại trong embedding hay không, nếu tồn tại thì nó không là forward extension --> có khả năng nó là backward
			//			__device__ bool IsVertexOnEmbedding(int vertex,Embedding **dArrPointerEmbedding,int noElem_dArrPointerEmbedding,int row){
			b=IsVertexOnEmbedding(toVid,dArrPointerEmbedding,noElem_dArrPointerEmbedding,i);

			/*
			if(b==true){
			PMS_PRINT("\nThread %d Exist:%d",i,toVid);
			}
			else
			{
			PMS_PRINT("\nThread %d NonExist:%d",i,toVid);
			}
			*/

			if(b==true){
				goto backward;
			}

			int indexOfd_arr_V=i*maxDegreeOfVer+j;
			//PMS_PRINT("\nThread %d: m:%d",i,maxDegreeOfVer);
			int indexOfd_LN=indexToVidIndN;
			dArrV_valid[indexOfd_arr_V]=1;
			//PMS_PRINT("\ndArrV[%d].valid:%d",indexOfd_arr_V,dArrV_valid[indexOfd_arr_V]);
			//cập nhật dữ liệu cho mảng dArrExtension
			dArrExtension[indexOfd_arr_V].vgi=vid;
			dArrExtension[indexOfd_arr_V].vgj=toVid;
			dArrExtension[indexOfd_arr_V].lij=d_LN[indexOfd_LN];
			//PMS_PRINT("\n");
			//PMS_PRINT("d_LN[%d]:%d ",indexOfd_LN,d_LN[indexOfd_LN]);
			//PMS_PRINT("Graph_ID:%d",__float2int_rn(toVid/maxDegreeOfVer));
			dArrExtension[indexOfd_arr_V].li=labelFromVid;
			dArrExtension[indexOfd_arr_V].lj=labelToVid;
			dArrExtension[indexOfd_arr_V].vi=fromRMP;
			dArrExtension[indexOfd_arr_V].vj=maxId+1;
			//dArrExtension[indexOfd_arr_V].posColumn=idxQ;
			dArrExtension[indexOfd_arr_V].posRow=row;
			continue;
backward:
			//2. Kiểm tra backward
			for (int k = 1; k < segdArrVidOnRMP; k++)
			{
				if(toVid == dArrVidOnRMP[i*segdArrVidOnRMP+k]){ //Nếu đỉnh tồn tại trên dArrVidOnRMP thì nó thoả backward
					int indexOfd_arr_V=i*maxDegreeOfVer+j;
					//PMS_PRINT("\nThread %d: m:%d",i,maxDegreeOfVer);
					int indexOfd_LN=indexToVidIndN;
					dArrV_valid[indexOfd_arr_V] = 1;
					dArrV_backward[indexOfd_arr_V]=1;
					//PMS_PRINT("\ndArrV[%d].valid:%d backward:%d",indexOfd_arr_V,dArrV_valid[indexOfd_arr_V],dArrV_backward[indexOfd_arr_V]);
					//cập nhật dữ liệu cho mảng dArrExtension
					dArrExtension[indexOfd_arr_V].vgi=vid;
					dArrExtension[indexOfd_arr_V].vgj=toVid;
					dArrExtension[indexOfd_arr_V].lij=d_LN[indexOfd_LN];
					/*PMS_PRINT("\n");
					PMS_PRINT("d_LN[%d]:%d ",indexOfd_LN,d_LN[indexOfd_LN]);*/
					dArrExtension[indexOfd_arr_V].li=labelFromVid;
					dArrExtension[indexOfd_arr_V].lj=labelToVid;
					dArrExtension[indexOfd_arr_V].vi=maxId;
					//dArrExtension[indexOfd_arr_V].vj=fromPosCol[i*segdArrVidOnRMP+k];
					dArrExtension[indexOfd_arr_V].vj=rmp[k+1];
					//dArrExtension[indexOfd_arr_V].posColumn=idxQ;
					dArrExtension[indexOfd_arr_V].posRow=row;
					break; //thoát khỏi vòng lặp hiện tại
				}
			}
		}
	}
}

__global__ void kernelFindValidFBExtensionv2(Embedding **dArrPointerEmbedding,int noElem_dArrPointerEmbedding,int noElem_Embedding,int *d_O,int *d_LO,int *d_N,int *d_LN,float *dArrDegreeOfVid,int maxDegreeOfVer,int *dArrV_valid,int *dArrV_backward,EXT *dArrExtension,int *listOfVer,int minLabel,int maxId,int fromRMP, int *dArrVidOnRMP,int segdArrVidOnRMP,int *rmp,int *dArrVj,int noElemdArrVj){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	//Duyệt qua các Embedding và xét các mở rộng cho đỉnh tại vị trí idxQ
	if(i<noElem_Embedding){
		//	int fromPosCol;
		//int idxRMP=0;
		//int noELemVerOnRMP = segdArrVidOnRMP +1;
		int posColumn =noElem_dArrPointerEmbedding-1;
		int posRow=i;
		int col = posColumn;
		int row = posRow;
		//Embedding *Q=dArrPointerEmbedding[idxQ];
		int vid = listOfVer[i];
		int degreeVid=__float2int_rn(dArrDegreeOfVid[i]);
		//Duyệt qua các đỉnh kề với đỉnh vid dựa vào số lần duyệt là bậc
		int indexToVidIndN=d_O[vid];
		int labelFromVid = d_LO[vid];
		int toVid;
		int labelToVid;
		bool b=true;
		for (int j = 0; j < degreeVid; j++,indexToVidIndN++) //Duyệt qua tất cả các đỉnh kề với đỉnh vid, nếu đỉnh không thuộc embedding thì --> cạnh cũng không thuộc embedding vì đây là Q cuối, Nếu đỉnh không thuộc Embedding thì nó cũng không phải là backward
		{			
			//1.Kiểm tra forward
			toVid=d_N[indexToVidIndN]; //Lấy vid của đỉnh cần kiểm tra
			labelToVid = d_LO[toVid]; //lấy label của đỉnh cần kiểm tra
			posColumn=col;
			posRow=row;
			Embedding *Q=dArrPointerEmbedding[posColumn];
			//PMS_PRINT("\nThread %d, j: %d has ToVidLabel:%d",i,j,labelToVid);
			//1. Trước tiên kiểm tra nhãn của labelToVid có nhỏ hơn minLabel hay không. Nếu nhỏ hơn thì return. Vì nó cũng không có khả năng là backward extension
			if(labelToVid<minLabel) 
				return;
			//goto backward;
			//2. kiểm tra xem đỉnh toVid có tồn tại trong embedding hay không, nếu tồn tại thì nó không là forward extension --> có khả năng nó là backward
			//Duyệt qua embedding column từ Q cuối đến Q đầu, lần lượt lấy vid so sánh với toVid

			//PMS_PRINT("\n Q[%d] Row[%d] (idx:%d vid:%d)",posColumn,posRow,Q[posRow].idx,Q[posRow].vid);//Q[1][0]
			if(toVid==Q[posRow].vid) 
				goto backward;

			//PMS_PRINT("\nj:%d toVid:%d Q.vid:%d",j,toVid,Q[posRow].vid);

			while (true)
			{
				posRow = Q[posRow].idx;//0
				posColumn=posColumn-1;		//0
				Q=dArrPointerEmbedding[posColumn];
				//PMS_PRINT("\n posColumn[%d] Row[%d] (idx:%d vid:%d)",posColumn,posRow,Q[posRow].idx,Q[posRow].vid);//Q[0][0]
				//PMS_PRINT("\nj:%d toVid:%d Q.vid:%d",j,toVid,Q[posRow].vid);
				if(toVid==Q[posRow].vid) {
					b=false; break;
				}
				posRow=Q[posRow].idx;//-1
				//PMS_PRINT("\nposRow:%d",posRow);
				if(posRow==-1) break;
			}
			if (b==false){
				b=true; 
				goto backward;
			}
			int indexOfd_arr_V=i*maxDegreeOfVer+j;
			//PMS_PRINT("\nThread %d: m:%d",i,maxDegreeOfVer);
			int indexOfd_LN=indexToVidIndN;
			dArrV_valid[indexOfd_arr_V]=1;
			//PMS_PRINT("\ndArrV[%d].valid:%d",indexOfd_arr_V,dArrV_valid[indexOfd_arr_V]);
			//cập nhật dữ liệu cho mảng dArrExtension
			dArrExtension[indexOfd_arr_V].vgi=vid;
			dArrExtension[indexOfd_arr_V].vgj=toVid;
			dArrExtension[indexOfd_arr_V].lij=d_LN[indexOfd_LN];
			//PMS_PRINT("\n");
			//PMS_PRINT("d_LN[%d]:%d ",indexOfd_LN,d_LN[indexOfd_LN]);
			dArrExtension[indexOfd_arr_V].li=labelFromVid;
			dArrExtension[indexOfd_arr_V].lj=labelToVid;
			dArrExtension[indexOfd_arr_V].vi=fromRMP;
			dArrExtension[indexOfd_arr_V].vj=maxId+1;
			//dArrExtension[indexOfd_arr_V].posColumn=idxQ;
			dArrExtension[indexOfd_arr_V].posRow=row;
backward:
			//2. Kiểm tra backward. Vấn đề là khi mở rộng backward từ một backward embedding thì làm sao biết backward extension đó đã có rồi.
			//==> giải pháp là dựa vào các vj của backward extension trong DFS_CODE.
			//Nếu nobwWeHave = 2 thì chúng ta phải lấy 2 vj của DFS_CODE của embedding column hiện tại

			for (int k = 1; k < segdArrVidOnRMP; k++) //tại sao k lại bắt đầu từ 1, vì k=0 là node kế trước của node đang mở rộng, nên sẽ không tồn tại backward với node kế trước.
			{
				int agreeK = 0; //backward extension chưa tồn tại. Ở đây có thể cải tiến song song cho việc kiểm tra tồn tại backward extension
				for (int m = 0; m < noElemdArrVj; m++)
				{
					if(k == ((segdArrVidOnRMP-1)-dArrVj[m])){
						agreeK=-1; //backward extension đã tồn tại
						break;
					}
				}
				if(agreeK==-1){
					//PMS_PRINT("\n Thread:%d agreek:%d",i,agreeK);
					continue;
				}
				if(toVid == dArrVidOnRMP[i*segdArrVidOnRMP+k]){
					int indexOfd_arr_V=i*maxDegreeOfVer+j;
					//PMS_PRINT("\nThread %d: m:%d",i,maxDegreeOfVer);
					int indexOfd_LN=indexToVidIndN;
					dArrV_valid[indexOfd_arr_V] = 1;
					dArrV_backward[indexOfd_arr_V]=1;
					//PMS_PRINT("\ndArrV[%d].valid:%d backward:%d",indexOfd_arr_V,dArrV_valid[indexOfd_arr_V],dArrV_backward[indexOfd_arr_V]);
					//cập nhật dữ liệu cho mảng dArrExtension
					dArrExtension[indexOfd_arr_V].vgi=vid;
					dArrExtension[indexOfd_arr_V].vgj=toVid;
					dArrExtension[indexOfd_arr_V].lij=d_LN[indexOfd_LN];
					//PMS_PRINT("\n");
					//PMS_PRINT("d_LN[%d]:%d ",indexOfd_LN,d_LN[indexOfd_LN]);
					dArrExtension[indexOfd_arr_V].li=labelFromVid;
					dArrExtension[indexOfd_arr_V].lj=labelToVid;
					dArrExtension[indexOfd_arr_V].vi=maxId;
					//dArrExtension[indexOfd_arr_V].vj=fromPosCol[i*segdArrVidOnRMP+k];
					dArrExtension[indexOfd_arr_V].vj=rmp[k+1];
					//dArrExtension[indexOfd_arr_V].posColumn=idxQ;
					dArrExtension[indexOfd_arr_V].posRow=row;
					break; //thoát khỏi vòng lặp hiện tại
				}
			}
		}
	}
}
//Cần tạo một device_function để kiểm tra xem đỉnh đã tồn tại hay chưa
__device__ bool IsVertexOnEmbedding(int vertex,Embedding **dArrPointerEmbedding,int noElem_dArrPointerEmbedding,int row){
	bool bExist = false; //Ban đầu giả sử đỉnh chưa thuộc embedding.
	int lastCol = noElem_dArrPointerEmbedding-1; //index của embedding column cuối cùng
	Embedding *dArrEmbedding;
	//PMS_PRINT("\n Last Embedding column:%d Element:%d (idx vid):(%d %d)",lastCol,i,dArrEmbedding[i].idx,dArrEmbedding[i].vid);
	int prevRow=row; //index của dòng đang xét trong embedding column.
	for (int j = lastCol; j>=0; j--) //Duyệt từ embedding column cuối ngược lên embedding đầu (Duyệt từ ngọn đến gốc của embedding đang xét).
	{
		dArrEmbedding= dArrPointerEmbedding[j]; //Lấy embedding đang xét trong embedding column.
		//std::PMS_PRINT("\n Last Embedding column:%d Element:%d (idx vid):(%d %d)",lastCol,i,dArrEmbedding[prevRow].idx,dArrEmbedding[prevRow].vid);
		if(vertex==dArrEmbedding[prevRow].vid){ //Nếu đỉnh tồn tại trong embedding thì bậc cờ true và return về
			bExist=true; break; 
		}
		prevRow=dArrEmbedding[prevRow].idx; //Ngược lại, xét đỉnh khác trên embedding.
	}
	return bExist;
}
//inuse
__global__ void kernelFillValidBackward(int* dValidBackward,int* dVj,int noElem, int* dLookupArrVj,int noElemLookup)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElem)
	{
		int vj = dVj[i];
		for (int k = 0; k < noElemLookup; k++)
		{
			if(vj == dLookupArrVj[k])
			{
				dValidBackward[i] = 1;
				break;
			}
		}
	}
}
//inuse
__global__ void kernelFindValidExtension(Embedding **dPointerdArrEmbedding,int* dArrRMP, int noElemRMP,int noElemEmbedding, \
										 int *dO,int *dLO,int *dN,int *dLN, float *dArrDegreeOfVid, \
										 int maxDegreeOfVer,int** dPointerArrValid, \
										 EXT** dPointerArrEXT, int minLabel,int maxId, int* dValidBackward,int* dVj)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int noElemVid = noElemEmbedding*noElemRMP;
	if(i<noElemVid)
	{
		int idxLastCol = noElemRMP-1;
		//Bắt đầu từ embedding column cuối cùng.
		//Phần dư của thread i sẽ cho biết Thread đó đang xử lý cho embedding nào trong Embedding column cuối.
		int idxRow = i % noElemEmbedding;
		//Phần nguyên là embedding column index của vid
		int idxCol = i / noElemEmbedding;
		int vid=-1;
		//PMS_PRINT("\n Thread: %d idxCol:%d idxRow:%d idxLastCol:%d ",i,idxCol,idxRow,idxLastCol);
		for(int s = idxLastCol,int idxRowTemp=idxRow; s>=0; s--)
		{
			Embedding* dArrEmbeddingTemp = dPointerdArrEmbedding[s];
			//Lấy vid mà thread cần xử lý tại embedding column ở index: idxCol
			if(s>=idxCol)
			{
				vid = dArrEmbeddingTemp[idxRowTemp].vid;
			}
			//Đưa các đỉnh thuộc RMP vào danh sách
			idxRowTemp = dArrEmbeddingTemp[idxRowTemp].idx;
		}

		//Duyệt qua các đỉnh kề với vid trong dN dựa vào bậc của vid.
		int degreeOfVid = __float2int_rn(dArrDegreeOfVid[i]); //Lấy bậc của vid
		int idxRMP = -1;
		for (int idxToVid = 0; idxToVid < degreeOfVid; idxToVid++)
		{
			int indexToVidIndN=dO[vid]+idxToVid; //Lấy index trong mảng nhãn cạnh.
			int labelFromVid = dLO[vid]; //Lấy nhãn của đỉnh được mở rộng.
			int toVid=dN[indexToVidIndN]; //vid của đỉnh kề.
			int labelToVid = dLO[toVid]; //nhãn của đỉnh kề

			//Xét đỉnh kề có thoả các điều kiện của mở rộng forward hay không.
			//đk1: nếu nhãn đỉnh kề nhỏ hơn minLabel của DFS_CODE thì continue xét đỉnh kề tiếp theo.
			if(labelToVid<minLabel) continue;
			//đk2: nếu đỉnh kề đã thuộc RMP của embedding rồi thì xét xem nó có là backward hay không.
			//Nếu đỉnh kề tồn tại trong lstVidOnRMP thì xem như nó đã thuộc embedding, tiếp tục xét đỉnh kề khác
			bool isExist = false;
			int existButSureNotValidBackward = 0;
			for(int s = idxLastCol,int idxRowTemp=idxRow; s>=0; s--)
			{
				Embedding* dArrEmbeddingTemp = dPointerdArrEmbedding[s];
				existButSureNotValidBackward = existButSureNotValidBackward + 1;
				if(toVid == dArrEmbeddingTemp[idxRowTemp].vid)
				{
					isExist = true;
					idxRMP = s;
					//PMS_PRINT("\n Thread: %d Goodbye:%d",i,toVid);
					break;
				}
				idxRowTemp = dArrEmbeddingTemp[idxRowTemp].idx;
			}
			//PMS_PRINT("\n Thread: %d ,toVid:%d isExist:%d ,existButSureNotValidBackward:%d \n",i,toVid,isExist,existButSureNotValidBackward);
			//vid có tối đa là maxDegreeOfVer mở rộng hợp lệ được lưu trữ trong dPointerArrEXT tương ứng tại idxCol, idxRow.
			EXT* dArrEXT = dPointerArrEXT[idxCol];
			int* dArrValid = dPointerArrValid[idxCol];
			int idxRowEXT = idxRow*maxDegreeOfVer + idxToVid;
			if (isExist == true ) 
			{
				//Nếu Thread đang xử lý cho vid thuộc Embedding Column cuối và 
				//nhiều hơn 2 đỉnh thuộc RMP thì mới xét mở rộng backard và
				//Đỉnh kề phải cách đỉnh cuối ít nhất 1 đỉnh.
				if (idxCol == idxLastCol && noElemRMP>2 && \
					(idxLastCol - idxRMP)>=1 && existButSureNotValidBackward != 2) 
				{
					goto considerBackward;
				}

				continue;
			};
			//Lưu mở rộng hợp lệ forward
			//Lưu mở rộng forward hợp lệ
			dArrValid[idxRowEXT] = 1;
			dArrEXT[idxRowEXT].vi = dArrRMP[idxCol];
			dArrEXT[idxRowEXT].vj = maxId + 1;
			dArrEXT[idxRowEXT].li = labelFromVid;
			dArrEXT[idxRowEXT].lij = dLN[indexToVidIndN];
			dArrEXT[idxRowEXT].lj = labelToVid;
			dArrEXT[idxRowEXT].vgi = vid; 
			dArrEXT[idxRowEXT].vgj = toVid;
			dArrEXT[idxRowEXT].posRow = idxRow;
			//lưu xong thì continue xét đỉnh kề khác
			continue;

			considerBackward:
			//Kiểm tra backward có hợp lệ hay không
			//Nếu backward đã tồn tại rồi thì continue xét đỉnh kề khác.
			if(1==dValidBackward[idxRMP]) continue;
			//Lưu lại mở rộng backward hợp lệ.
			dArrValid[idxRowEXT] = 1;
			dArrEXT[idxRowEXT].vi = maxId;
			dArrEXT[idxRowEXT].vj = dVj[idxRMP];
			dArrEXT[idxRowEXT].li = labelFromVid;
			dArrEXT[idxRowEXT].lij = dLN[indexToVidIndN];
			dArrEXT[idxRowEXT].lj = labelToVid;
			dArrEXT[idxRowEXT].vgi = vid; 
			dArrEXT[idxRowEXT].vgj = toVid;
			dArrEXT[idxRowEXT].posRow = idxRow;
		}
	}
}

__global__ void kernelFindValidFBExtensionv3(Embedding **dArrPointerEmbedding, \
											 int noElem_dArrPointerEmbedding, int noElem_Embedding, \
											 int *d_O,int *d_LO,int *d_N,int *d_LN, \
											 float *dArrDegreeOfVid, \
											 int maxDegreeOfVer, \
											 int *dArrV_valid,int *dArrV_backward,EXT *dArrExtension, \
											 int *listOfVer, \
											 int minLabel,int maxId, \
											 int fromRMP, \
											 int *dArrVidOnRMP, \
											 int segdArrVidOnRMP, \
											 int *rmp, \
											 int *dArrVj, \
											 int noElemdArrVj){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElem_Embedding){ //Mỗi một thread i sẽ tìm mở rộng cho một đỉnh của Embedding Column
		//if(i==0){
		//int fromPosCol;
		//int idxRMP=0;
		//int noELemVerOnRMP = segdArrVidOnRMP +1;
		int posColumn =noElem_dArrPointerEmbedding-1; 
		int posRow=i; //Phải giữ lại posColumn và posRow để cập nhật thông tin trong EXT<k>.
		int col = posColumn;
		int row = posRow;
		//Embedding *Q=dArrPointerEmbedding[idxQ];
		int vid = listOfVer[i]; //Lấy đỉnh cần mở rộng.
		int degreeVid=__float2int_rn(dArrDegreeOfVid[i]); //Lấy bậc của đỉnh đang xét.
		//Duyệt qua các đỉnh kề với đỉnh vid dựa vào số lần duyệt là bậc
		int indexToVidIndN=d_O[vid]; //Lấy index trong mảng nhãn cạnh
		int labelFromVid = d_LO[vid]; //Lấy nhãn của đỉnh được mở rộng
		int toVid;
		int labelToVid;
		bool b;
		for (int j = 0; j < degreeVid; ++j,++indexToVidIndN) //Duyệt qua tất cả các đỉnh kề với đỉnh vid, nếu đỉnh không thuộc embedding thì --> cạnh cũng không thuộc embedding vì đây là Q cuối, Nếu đỉnh không thuộc Embedding thì nó cũng không phải là backward
		{		
			b=false;
			//1.Kiểm tra forward
			toVid=d_N[indexToVidIndN]; //Lấy vid của đỉnh cần kiểm tra
			labelToVid = d_LO[toVid]; //lấy label của đỉnh cần kiểm tra
			posColumn=col;
			posRow=row;
			Embedding *Q=dArrPointerEmbedding[posColumn];
			//PMS_PRINT("\nThread %d, j: %d has ToVidLabel:%d",i,j,labelToVid);
			//1. Trước tiên kiểm tra nhãn của labelToVid có nhỏ hơn minLabel hay không. Nếu nhỏ hơn thì return. Vì nó cũng không có khả năng là backward extension
			if(labelToVid<minLabel) 
				continue;
			//goto backward;
			//2. kiểm tra xem đỉnh toVid có tồn tại trong embedding hay không, nếu tồn tại thì nó không là forward extension --> có khả năng nó là backward
			//			__device__ bool IsVertexOnEmbedding(int vertex,Embedding **dArrPointerEmbedding,int noElem_dArrPointerEmbedding,int row){
			int qi=-1;
			//b=IsVertexOnEmbedding(toVid,dArrPointerEmbedding,noElem_dArrPointerEmbedding,i);
			//check toVid is on RMP of embedding or not -->
			int lastCol = noElem_dArrPointerEmbedding-1; //index của embedding column cuối cùng
			Embedding *dArrEmbedding;
			//PMS_PRINT("\n Last Embedding column:%d Element:%d (idx vid):(%d %d)",lastCol,i,dArrEmbedding[i].idx,dArrEmbedding[i].vid);
			int prevRow=row; //index của dòng đang xét trong embedding column.
			for (int j = lastCol; j>=0; j--) //Duyệt từ embedding column cuối ngược lên embedding đầu (Duyệt từ ngọn đến gốc của embedding đang xét).
			{
				//PMS_PRINT("\nThread %d rmp[%d]:%d",i,j,rmp[abs(j-lastCol)]);
				dArrEmbedding= dArrPointerEmbedding[j]; //Lấy embedding đang xét trong embedding column.
				//std::PMS_PRINT("\n Last Embedding column:%d Element:%d (idx vid):(%d %d)",lastCol,i,dArrEmbedding[prevRow].idx,dArrEmbedding[prevRow].vid);
				if(toVid==dArrEmbedding[prevRow].vid){ //Nếu đỉnh tồn tại trong embedding thì bậc cờ true và return về
					b=true;
					qi = rmp[abs(j-lastCol)];
					break; 
				}
				prevRow=dArrEmbedding[prevRow].idx; //Ngược lại, xét đỉnh khác trên embedding.
			}
			//check toVid is on RMP of embedding or not <--
			
			if(b==true){
			PMS_PRINT("\nThread %d Exist:%d, rmp:%d",i,toVid,qi);
			}
			/*else
			{
			PMS_PRINT("\nThread %d NonExist:%d",i,toVid);
			}*/


			if(b==true){
				goto backward;
			}

			int indexOfd_arr_V=i*maxDegreeOfVer+j;
			//PMS_PRINT("\nThread %d: m:%d",i,maxDegreeOfVer);
			int indexOfd_LN=indexToVidIndN;
			dArrV_valid[indexOfd_arr_V]=1;
			//PMS_PRINT("\ndArrV[%d].valid:%d",indexOfd_arr_V,dArrV_valid[indexOfd_arr_V]);
			//cập nhật dữ liệu cho mảng dArrExtension
			dArrExtension[indexOfd_arr_V].vgi=vid;
			dArrExtension[indexOfd_arr_V].vgj=toVid;
			dArrExtension[indexOfd_arr_V].lij=d_LN[indexOfd_LN];
			//PMS_PRINT("\n");
			//PMS_PRINT("d_LN[%d]:%d ",indexOfd_LN,d_LN[indexOfd_LN]);
			dArrExtension[indexOfd_arr_V].li=labelFromVid;
			dArrExtension[indexOfd_arr_V].lj=labelToVid;
			dArrExtension[indexOfd_arr_V].vi=fromRMP;
			dArrExtension[indexOfd_arr_V].vj=maxId+1;
			//dArrExtension[indexOfd_arr_V].posColumn=idxQ;
			dArrExtension[indexOfd_arr_V].posRow=row;
			continue;
			backward:
			//2. Kiểm tra backward. Vấn đề là khi mở rộng backward từ một backward embedding thì làm sao biết backward extension đó đã có rồi.
			//==> giải pháp là dựa vào các vj của backward extension trong DFS_CODE.
			//Nếu nobwWeHave = 2 thì chúng ta phải lấy 2 vj của DFS_CODE của embedding column hiện tại
			/*
			//Check valid backward
			//K =1 và k++ có vẻ sai rồi
			//Duyệt qua các phần từ của RMP
			for (int k = 1; k < segdArrVidOnRMP; k++) //tại sao k lại bắt đầu từ 1, vì k=0 là node kế trước của node đang mở rộng, nên sẽ không tồn tại backward với node kế trước.
			{
				int agreeK = 0; //backward extension chưa tồn tại. Ở đây có thể cải tiến song song cho việc kiểm tra tồn tại backward extension
				for (int m = 0; m < noElemdArrVj; m++)
				{
					if(k == ((segdArrVidOnRMP-1)-dArrVj[m])){ //Code có vẻ bị sai logic rồi :(
						agreeK=-1; //backward extension đã tồn tại
						break;
					}
				}
				if(agreeK==-1){
					//PMS_PRINT("\n Thread:%d agreek:%d",i,agreeK);
					continue;
				}
				if(toVid == dArrVidOnRMP[i*segdArrVidOnRMP+k]){
					int indexOfd_arr_V=i*maxDegreeOfVer+j;
					//PMS_PRINT("\nThread %d: m:%d",i,maxDegreeOfVer);
					int indexOfd_LN=indexToVidIndN;
					dArrV_valid[indexOfd_arr_V] = 1;
					dArrV_backward[indexOfd_arr_V]=1;
					//PMS_PRINT("\ndArrV[%d].valid:%d backward:%d",indexOfd_arr_V,dArrV_valid[indexOfd_arr_V],dArrV_backward[indexOfd_arr_V]);
					//cập nhật dữ liệu cho mảng dArrExtension
					dArrExtension[indexOfd_arr_V].vgi=vid;
					dArrExtension[indexOfd_arr_V].vgj=toVid;
					dArrExtension[indexOfd_arr_V].lij=d_LN[indexOfd_LN];
					//PMS_PRINT("\n");
					//PMS_PRINT("d_LN[%d]:%d ",indexOfd_LN,d_LN[indexOfd_LN]);
					dArrExtension[indexOfd_arr_V].li=labelFromVid;
					dArrExtension[indexOfd_arr_V].lj=labelToVid;
					dArrExtension[indexOfd_arr_V].vi=maxId;
					//dArrExtension[indexOfd_arr_V].vj=fromPosCol[i*segdArrVidOnRMP+k];
					dArrExtension[indexOfd_arr_V].vj=rmp[k+1];
					//dArrExtension[indexOfd_arr_V].posColumn=idxQ;
					dArrExtension[indexOfd_arr_V].posRow=row;
					break; //thoát khỏi vòng lặp hiện tại
				}
			} //end For check valid backward
			*/
			//Kiểm tra xem qi (là một số nguyên chỉ embedding column thuộc RMP) đã thuộc dArrVj hay chưa
			//Nếu qi đã tồn tại trong dArrVj ==> invalid BW Extension. Ngược lại, thì ghi nhận mở rộng backward đó là hợp lệ
			int agreeK = 0;
			for (int m = 0; m < noElemdArrVj; m++)
			{
				if(qi == dArrVj[m]){ //Code có vẻ bị sai logic rồi :(
					agreeK=-1; //backward extension đã tồn tại
					break;
				}
			}
			//Nếu agreeK = 0 thì xem như là hợp lệ
			if(agreeK == 0)
			{
				int indexOfd_arr_V=i*maxDegreeOfVer+j;
				//PMS_PRINT("\nThread %d: m:%d",i,maxDegreeOfVer);
				int indexOfd_LN=indexToVidIndN;
				dArrV_valid[indexOfd_arr_V] = 1;
				dArrV_backward[indexOfd_arr_V]=1;
				//PMS_PRINT("\ndArrV[%d].valid:%d backward:%d",indexOfd_arr_V,dArrV_valid[indexOfd_arr_V],dArrV_backward[indexOfd_arr_V]);
				//cập nhật dữ liệu cho mảng dArrExtension
				dArrExtension[indexOfd_arr_V].vgi=vid;
				dArrExtension[indexOfd_arr_V].vgj=toVid;
				dArrExtension[indexOfd_arr_V].lij=d_LN[indexOfd_LN];
				//PMS_PRINT("\n");
				//PMS_PRINT("d_LN[%d]:%d ",indexOfd_LN,d_LN[indexOfd_LN]);
				dArrExtension[indexOfd_arr_V].li=labelFromVid;
				dArrExtension[indexOfd_arr_V].lj=labelToVid;
				dArrExtension[indexOfd_arr_V].vi=maxId;
				//dArrExtension[indexOfd_arr_V].vj=fromPosCol[i*segdArrVidOnRMP+k];
				dArrExtension[indexOfd_arr_V].vj=qi;
				//dArrExtension[indexOfd_arr_V].posColumn=idxQ;
				dArrExtension[indexOfd_arr_V].posRow=row;
				//break; //thoát khỏi vòng lặp hiện tại
			}
		} //end For duyệt qua các bậc của đỉnh.
	}
}
//Consider current
__global__ void kernelFindValidForwardExtensionv3(Embedding **dArrPointerEmbedding,int noElem_dArrPointerEmbedding,int noElem_Embedding,int *d_O,int *d_LO,int *d_N,int *d_LN,float *dArrDegreeOfVid,int maxDegreeOfVer,int *dArrV_valid,int *dArrV_backward,EXT *dArrExtension,int *listOfVer,int minLabel,int maxId,int fromRMP, int *dArrVidOnRMP,int segdArrVidOnRMP,int *rmp){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElem_Embedding){ //Mỗi một thread i sẽ tìm mở rộng cho một đỉnh của Embedding.
		//if(i==0){
		//int fromPosCol;
		//int idxRMP=0;
		//int noELemVerOnRMP = segdArrVidOnRMP +1;
		int posColumn =noElem_dArrPointerEmbedding-1; 
		int posRow=i; //Phải giữ lại posColumn và posRow để cập nhật thông tin trong EXT<k>. Nó giúp xác định row index của embedding trong embedding column của cạnh mở rộng được lưu trữ trong EXTk
		int col = posColumn;
		int row = posRow;
		//Embedding *Q=dArrPointerEmbedding[idxQ];
		int vid = listOfVer[i]; //Lấy đỉnh cần mở rộng.
		int degreeVid=__float2int_rn(dArrDegreeOfVid[i]); //Lấy bậc của đỉnh đang xét.
		//Duyệt qua các đỉnh kề với đỉnh vid dựa vào số lần duyệt là bậc
		int indexToVidIndN=d_O[vid]; //Lấy index trong mảng nhãn cạnh
		int labelFromVid = d_LO[vid]; //Lấy nhãn của đỉnh được mở rộng
		int toVid;
		int labelToVid;
		bool b;
		for (int j = 0; j < degreeVid; j++,indexToVidIndN++) //Duyệt qua tất cả các đỉnh kề với đỉnh vid, nếu đỉnh không thuộc embedding thì --> cạnh cũng không thuộc embedding vì đây là Q cuối, Nếu đỉnh không thuộc Embedding thì nó cũng không phải là backward
		{			
			b=false;
			//1.Kiểm tra forward
			toVid=d_N[indexToVidIndN]; //Lấy vid của đỉnh cần kiểm tra
			labelToVid = d_LO[toVid]; //lấy label của đỉnh cần kiểm tra
			posColumn=col;
			posRow=row;
			Embedding *Q=dArrPointerEmbedding[posColumn];
			//PMS_PRINT("\nThread %d, j: %d has ToVidLabel:%d",i,j,labelToVid);
			//1. Trước tiên kiểm tra nhãn của labelToVid có nhỏ hơn minLabel hay không. Nếu nhỏ hơn thì continue xét đỉnh kề khác. Vì nó cũng không có khả năng là backward extension
			if(labelToVid<minLabel) 
				continue;
			//goto backward;
			//2. kiểm tra xem đỉnh toVid có tồn tại trong embedding hay không, nếu tồn tại thì nó không là forward extension --> có khả năng nó là backward
			//			__device__ bool IsVertexOnEmbedding(int vertex,Embedding **dArrPointerEmbedding,int noElem_dArrPointerEmbedding,int row){
			b=IsVertexOnEmbedding(toVid,dArrPointerEmbedding,noElem_dArrPointerEmbedding,i);

			/*
			if(b==true){
			PMS_PRINT("\nThread %d Exist:%d",i,toVid);
			}
			else
			{
			PMS_PRINT("\nThread %d NonExist:%d",i,toVid);
			}
			*/

			if(b==true){ //Nếu đỉnh mở rộng thuộc embedding thì chắc chắn nó không phải là mở rộng forward. Tiếp tục xét đỉnh khác.
				continue;
			}
			//Vì đỉnh kề không thuộc embedding và nhãn đỉnh kề lớn hơn hoặc bằng minlabel nên nó thoả mở rộng forward ==> Bậc cờ 1 trong mảng dArrV.
			int indexOfd_arr_V=i*maxDegreeOfVer+j; //Tính index của mở rộng hợp lệ trong mảng dArrV.
			//PMS_PRINT("\nThread %d: m:%d",i,maxDegreeOfVer);
			int indexOfd_LN=indexToVidIndN;
			dArrV_valid[indexOfd_arr_V]=1;
			//PMS_PRINT("\ndArrV[%d].valid:%d",indexOfd_arr_V,dArrV_valid[indexOfd_arr_V]);
			//PMS_PRINT("\nGraphID:%d",__float2int_rn(toVid/maxDegreeOfVer));
			//cập nhật dữ liệu cho mảng dArrExtension
			dArrExtension[indexOfd_arr_V].vgi=vid;
			dArrExtension[indexOfd_arr_V].vgj=toVid;
			dArrExtension[indexOfd_arr_V].lij=d_LN[indexOfd_LN];
			//PMS_PRINT("\n");
			//PMS_PRINT("d_LN[%d]:%d ",indexOfd_LN,d_LN[indexOfd_LN]);
			dArrExtension[indexOfd_arr_V].li=labelFromVid;
			dArrExtension[indexOfd_arr_V].lj=labelToVid;
			dArrExtension[indexOfd_arr_V].vi=fromRMP;
			dArrExtension[indexOfd_arr_V].vj=maxId+1;
			//dArrExtension[indexOfd_arr_V].posColumn=idxQ;
			dArrExtension[indexOfd_arr_V].posRow=row;
		} //end For duyệt qua các bậc của đỉnh.
	}
}

//Copy the address of dArrEmbedding into dArrPointerEmbedding
__global__ void	kernelGetPointerdArrEmbedding(Embedding *dArrEmbedding,Embedding **dArrPointerEmbedding,int idx)
{
	dArrPointerEmbedding[idx]=dArrEmbedding;
	//PMS_PRINT("\n PointerdArrEmbedding:%p, PointerdArrPointerEmbedding:%p",dArrEmbedding,dArrPointerEmbedding[idx]);
}

__global__ void kernelPrintdArrPointerEmbedding(Embedding **dArrPointerEmbedding,int noElem,int sizeArr){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i<noElem){
		Embedding *E = dArrPointerEmbedding[i];
		for (int j = 0; j < sizeArr; j++)
		{
			PMS_PRINT("\n Thread %d pointer:%p (idx vid):(%d %d)",i,E,E[j].idx,E[j].vid);
		}
	}
}

//kernel in mảng struct_V *dArrV trên device
__global__ void kernelprintdArrV(V *dArrV,int noElem_dArrV,EXT *dArrExtension){
	int i = blockDim.x *blockIdx.x + threadIdx.x;
	if(i<noElem_dArrV){
		int vi = dArrExtension[i].vi;
		int vj = dArrExtension[i].vj;
		int li = dArrExtension[i].li;
		int lij = dArrExtension[i].lij;
		int lj = dArrExtension[i].lj;
		//PMS_PRINT("\n dArrV[%d].backward:%d ,dArrV[%d].valid:%d Extension:(vgi:%d,vgj:%d) (vi:%d vj:%d li:%d lij:%d lj:%d)",i,dArrV[i].backward,i,dArrV[i].valid,dArrExtension[i].vgi,dArrExtension[i].vgj,vi,vj,li,lij,lj);
	}

}

cudaError_t printdArrV(V *dArrV,int noElem_dArrV,EXT *dArrExtension){
	cudaError_t cudaStatus;
	dim3 block(blocksize);
	dim3 grid((noElem_dArrV + block.x -1 )/block.x);
	kernelprintdArrV<<<grid,block>>>(dArrV,noElem_dArrV,dArrExtension);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() in printdArrV() failed", cudaStatus);
		goto Error;
	}
Error:
	return cudaStatus;
}
//Kernel in nội dung mảng EXT *dExt
__global__ void kernelPrintdExt(EXT *dExt,int noElem_dExt){
	int i = blockDim.x *blockIdx.x + threadIdx.x;
	if(i<noElem_dExt){		
		int vi=dExt[i].vi;
		int vj=dExt[i].vj;
		int li= dExt[i].li;
		int lij=dExt[i].lij;
		int lj=dExt[i].lj;
		int vgi=dExt[i].vgi;
		int vgj=dExt[i].vgj;
		//		int posColumn= dExt[i].posColumn;
		int posRow=dExt[i].posRow;
		PMS_PRINT("\n Thread %d (vi:%d vj:%d li:%d lij:%d lj:%d) (vgi:%d vgj:%d) ( posRow:%d)",i,vi,vj,li,lij,lj,vgi,vgj,posRow);
	}

}

//Hàm in dExt
inline cudaError_t printdExt(EXT *dExt,int noElem_dExt){
	cudaError_t cudaStatus;

	dim3 block(blocksize);
	dim3 grid((noElem_dExt+block.x -1)/block.x);
	kernelPrintdExt<<<grid,block>>>(dExt,noElem_dExt);
	cudaDeviceSynchronize();
	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() kernelPrintdExt in printdExt() failed", cudaStatus);
		goto Error;
	}

Error:
	return cudaStatus;
}
//consider 3
//kernel trích các mở rộng hợp lệ từ mảng dArrExtension sang mảng dExt
__global__ void kernelExtractValidExtensionTodExt(EXT *dArrExtension,int *dArrValid,int *dArrValidScanResult,int noElem_dArrV,EXT *dExt,int noElem_dExt)
{
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
			//dExt[dArrValidScanResult[i]].posColumn = dArrExtension[i].posColumn;
			dExt[dArrValidScanResult[i]].posRow = dArrExtension[i].posRow;
		}
	}
}

////kernel trích phần tử valid từ mảng dArrV và lưu vào mảng dArrValid
//__global__ void kernelExtractValidFromdArrV(V *dArrV,int noElem_dArrV,int *dArrValid){
//	int i = threadIdx.x + blockDim.x*blockIdx.x;
//	if(i<noElem_dArrV){
//		dArrValid[i]=dArrV[i].valid;
//	}
//}

__global__ void	kernelForwardPossibleExtension_NonLast(EXT *dArrExt,int noElem,int Lv,int *dArrAllPossibleExtension){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i<noElem){
		int lij,lj;
		lij=dArrExt[i].lij;
		lj=dArrExt[i].lj;
		int idx=lij*Lv+lj;
		dArrAllPossibleExtension[idx]=1;
	}
}

//kernel lấy nhãn from Li
__global__ void kernelGetFromLabel(EXT *dArrExt,int *dFromLi){
	*dFromLi	= dArrExt[0].li;
}

//kernelGetFromLabelv3<<<grida,blocka>>>(ext.dArrExt,dFromVi,dFromLi);
__global__ void kernelGetFromLabelv3(EXT *dArrExt,int *dFromVi,int *dFromLi){
	*dFromVi = dArrExt[0].vi; //trích ra vi của DFS_CODE
	*dFromLi = dArrExt[0].li; //trích ra li nhãn của đỉnh to của DFS_CODE
}


__global__ void kernelGetFromLabelv2(EXT *dArrExt,int noElem,int *dFromVi,int *dFromLi){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElem){
		//PMS_PRINT("\n Thread %d",i);
		if(dArrExt[i].vi<dArrExt[i].vj){ //Nếu là forward Extension thì trích ra Vi và Li
			*dFromVi = dArrExt[i].vi; //trích ra vi của DFS_CODE
			*dFromLi = dArrExt[i].li; //trích ra li nhãn của đỉnh to của DFS_CODE
			//PMS_PRINT("\ndFromVi:%d",*dFromVi);
			//PMS_PRINT("\ndFromVi:%d",*dFromLi);
		}
	}
}

__global__ void kernelFilldArrUniEdge(int *dArrAllPossibleExtension,int *dArrAllPossibleExtensionScanResult,int noElem_dArrAllPossibleExtension,UniEdge *dArrUniEdge,int Lv,int *dFromLi){
	int i = blockDim.x*blockIdx.x +threadIdx.x;
	if(i<noElem_dArrAllPossibleExtension){
		if(dArrAllPossibleExtension[i]==1){
			int li,lij,lj;
			li=*dFromLi;
			lij = i/Lv;
			lj=i%Lv;
			dArrUniEdge[dArrAllPossibleExtensionScanResult[i]].li=li;
			dArrUniEdge[dArrAllPossibleExtensionScanResult[i]].lij=lij;
			dArrUniEdge[dArrAllPossibleExtensionScanResult[i]].lj=lj;
		}
	}
}

__global__ void kernelFilldArrUniEdgev2(int *dArrAllPossibleExtension,int *dArrAllPossibleExtensionScanResult,int noElem_dArrAllPossibleExtension,UniEdge *dArrUniEdge,int Lv,int *dFromLi,int *dFromVi,int maxId){
	int i = blockDim.x*blockIdx.x +threadIdx.x;
	if(i<noElem_dArrAllPossibleExtension){
		if(dArrAllPossibleExtension[i]==1){
			int li,lij,lj;
			li=*dFromLi;
			lij = i/Lv;
			lj=i%Lv;
			dArrUniEdge[dArrAllPossibleExtensionScanResult[i]].vi=*dFromVi;
			dArrUniEdge[dArrAllPossibleExtensionScanResult[i]].vj=maxId + 1;
			dArrUniEdge[dArrAllPossibleExtensionScanResult[i]].li=li;
			dArrUniEdge[dArrAllPossibleExtensionScanResult[i]].lij=lij;
			dArrUniEdge[dArrAllPossibleExtensionScanResult[i]].lj=lj;
		}
	}
}

__global__ void kernelGet_vivjlj(EXT* dArrExt,int* dvi,int* dvj,int* dli,int maxId)
{
	*dvi = dArrExt[0].vi;
	*dvj = maxId+1;
	*dli = dArrExt[0].li;
}
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
//inuse
__global__ void kernelMarkUniBE(int* dMappingVj,int* dAllExtension,int Lv,int noElem,EXT* dArrEXT)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i<noElem)
	{
		int vj = dArrEXT[i].vj;
		int lij = dArrEXT[i].lij;

		int idxVj = dMappingVj[vj];

		int idxAllExtension = lij*Lv + idxVj;
		dAllExtension[idxAllExtension] = 1;
	}
}

//inuse
__global__ void kernelFilldMappingVj(int noElemBW,int* dMappingVj,int* dRMP)
{
	int i = blockDim.x*blockIdx.x +threadIdx.x;
	if(i<noElemBW)
	{
		int vj = dRMP[i];
		dMappingVj[vj] = i;
	}
}
//inuse
//Kernel fill unique forward extension
__global__ void kernelFillUniFE( int *dArrAllPossibleExtension, \
								int *dArrAllPossibleExtensionScanResult, \
								int noElem_dArrAllPossibleExtension, \
								UniEdge *dArrUniEdge, \
								int Lv,int *dvi, \
								int *dvj,int *dli)
{
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
int displaydArrUniEdge(UniEdge *dArrUniEdge,int noElem_dArrUniEdge){
	cudaError_t cudaStatus;
	int status =0;
	UniEdge *hArrUniEdge = (UniEdge*)malloc(sizeof(UniEdge) * noElem_dArrUniEdge);
	if(hArrUniEdge == NULL){
		PMS_PRINT("\n malloc hArrUniEde failed");
		goto Error;
	}

	cudaStatus = cudaMemcpy(hArrUniEdge,dArrUniEdge,sizeof(UniEdge)*noElem_dArrUniEdge,cudaMemcpyDeviceToHost);
	CHECK(cudaStatus);
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}

	for (int i = 0; i < noElem_dArrUniEdge; i++)
	{
		PMS_PRINT("\n U[%d] (li lij lj):(%d %d %d)",i,hArrUniEdge[i].li,hArrUniEdge[i].lij,hArrUniEdge[i].lj);
	}

	std::free(hArrUniEdge);
Error:
	return status;
}



//Hàm trích các mở rộng hợp lệ từ mảng dArrExtension sang mảng dArrExt của hLevelExt
int PMS::extractValidExtensionTodExt(EXT *dArrExtension,V *dArrV,int noElem_dArrV,int idxEXT){
	cudaError_t cudaStatus;

	int status =0;
	//2. Scan mảng dArrValid để lấy kích thước của mảng cần tạo
	int *dArrValidScanResult = nullptr;

	CHECK(cudaStatus = cudaMalloc((void**)&dArrValidScanResult,sizeof(int)*noElem_dArrV));
	if (cudaStatus!=cudaSuccess){
		status = -1;
		fprintf(stderr,"\n CudaMalloc dArrValidScanResult in extractValidExtensionToExt() failed");
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(dArrValidScanResult,0,sizeof(int)*noElem_dArrV));
		if(cudaStatus!=cudaSuccess){
			status=-1;
			goto Error;
		}
	}


	//cudaStatus = scanV(dArrV->valid,noElem_dArrV,dArrValidScanResult); //hàm scan này có vấn đề. Nó làm thay đổi giá trị đầu vào.
	//if (cudaStatus!=cudaSuccess){
	//	status = -1;
	//	fprintf(stderr,"\n scanV dArrValid in extractValidExtensionToExt() failed");
	//	goto Error;
	//}
	CHECK(cudaStatus=myScanV(dArrV->valid,noElem_dArrV,dArrValidScanResult));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}


	////In nội dung kết quả dArrValidScanResult
	PMS_PRINT("\n********dArrValid******\n");
	FUNCHECK(status=displayDeviceArr(dArrV->valid,noElem_dArrV));
	if(status!=0){
		goto Error;
	}

	PMS_PRINT("\n********dArrValidScanResult******\n");
	FUNCHECK(status=displayDeviceArr(dArrValidScanResult,noElem_dArrV));
	if(status!=0){
		goto Error;
	}

	//3. Lấy kích thước của mảng dArrExt;
	int noElem_dExt=0;
	CHECK(cudaStatus=getSizeBaseOnScanResult(dArrV->valid,dArrValidScanResult,noElem_dArrV,noElem_dExt));
	if (cudaStatus!=cudaSuccess){
		status = -1;
		fprintf(stderr,"\n getSizeBaseOnScanResult in extractValidExtensionToExt() failed");
		goto Error;
	}

	//In nội dung noElem_dExt
	PMS_PRINT("\n******** noElem In dArrExt ******\n");
	PMS_PRINT("\n noElem_dExt:%d",noElem_dExt);
	hLevelEXT.at(0).vE.at(idxEXT).noElem = noElem_dExt;
	/**************** Nếu không tìm được mở rộng nào thì return *************/
	if (noElem_dExt == 0) 
	{
		CHECK(cudaStatus=cudaFree(dArrValidScanResult));
		if(cudaStatus!=cudaSuccess){
			status=-1;
			goto Error;
		}
		return status;
	}
	//Nếu tìm được mở rộng thì xây dựng EXTk, rồi trích các mở rộng duy nhất và tính độ hỗ trợ của chúng. Đồng thời
	//lọc ra các độ hỗ trợ thoả minsup
	//Quản lý theo Level
	//4. Khởi tạo mảng dArrExt có kích thước noElem_dExt rồi trích dữ liệu từ dArrExtension sang dựa vào dArrValid.
	//hLevelEXT.at(0).vE.at(idxEXT).noElem = noElem_dExt;
	CHECK(cudaStatus = cudaMalloc((void**)&hLevelEXT.at(0).vE.at(idxEXT).dArrExt,noElem_dExt*sizeof(EXT)));
	if(cudaStatus!=cudaSuccess){
		status =-1;
		fprintf(stderr,"\n cudaMalloc dExt in extractValidExtensionTodExt() failed", cudaStatus);
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(hLevelEXT.at(0).vE.at(idxEXT).dArrExt,0,sizeof(EXT)*noElem_dExt));
		if(cudaStatus!=cudaSuccess){
			status=-1;
			goto Error;
		}
	}
	dim3 blockb(blocksize);
	dim3 gridb((noElem_dArrV+blockb.x -1)/blockb.x);
	kernelExtractValidExtensionTodExt<<<gridb,blockb>>>(dArrExtension,dArrV->valid,dArrValidScanResult,noElem_dArrV,hLevelEXT.at(0).vE.at(idxEXT).dArrExt,noElem_dExt);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		status = -1;
		fprintf(stderr,"\n cudaDeviceSynchronize() kernelExtractValidExtensionTodExt in extractValidExtensionTodExt() failed", cudaStatus);
		goto Error;
	}
	//In mảng dExt;
	PMS_PRINT("\n********** dArrExt **********\n");
	CHECK(cudaStatus =printdExt(hLevelEXT.at(0).vE.at(idxEXT).dArrExt,noElem_dExt));
	if(cudaStatus!=cudaSuccess){
		status =-1;
		fprintf(stderr,"\n printdExt() in extractValidExtensionTodExt() failed", cudaStatus);
		goto Error;
	}

	//kernelGetvivj<<<1,100>>>(hLevelEXT.at(0).vE.at(idxEXT).dArrExt,noElem_dExt);
	/*cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
	status=-1;
	goto Error;
	}*/


	//Dựa vào dArrV để trích các cạnh duy nhất từ dArrExt và lưu vào dArrUniEdge tại vị trí tương ứng
	//Ở đây chỉ trích các mở rộng forward, vì nó chưa tồn tại mở rộng backward.
	int *dArrAllPossibleExtension =nullptr;
	int noElem_dArrAllPossibleExtension = Lv*Le;
	int noElem_dArrUniEdge=0;
	CHECK(cudaStatus=cudaMalloc((void**)&dArrAllPossibleExtension,noElem_dArrAllPossibleExtension*sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		status =-1;
		fprintf(stderr,"\n cudaMalloc((void**)&dArrAllPossibleExtension in extractUniExtension() failed",cudaStatus);
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(dArrAllPossibleExtension,0,noElem_dArrAllPossibleExtension*sizeof(int)));
		if(cudaStatus!=cudaSuccess){
			status=-1;
			goto Error;
		}
	}

	dim3 blockc(blocksize);
	dim3 gridc((hLevelEXT.at(0).vE.at(idxEXT).noElem + blockc.x -1)/blockc.x);
	kernelForwardPossibleExtension_NonLast<<<gridc,blockc>>>(hLevelEXT.at(0).vE.at(idxEXT).dArrExt,hLevelEXT.at(0).vE.at(idxEXT).noElem,Lv,dArrAllPossibleExtension);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus !=cudaSuccess){
		status =-1;
		fprintf(stderr,"\n kernelForwardPossibleExtension_NonLast  failed",cudaStatus);
		goto Error;
	}

	//Scan mảng dArrAllPossibleExtension để biết kích thước của mảng dArrUniEdge và ánh xạ từ vị trí trong dArrAllPossibleExtension thành nhãn để lưu vào dArrUniEdge được quản lý bởi hLevelUniEdge
	int *dArrAllPossibleExtensionScanResult =nullptr;
	CHECK(cudaStatus = cudaMalloc((void**)&dArrAllPossibleExtensionScanResult,noElem_dArrAllPossibleExtension*sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		status = -1;
		fprintf(stderr,"\n cudaMalloc dArrAllPossibleExtensionScanResult  failed",cudaStatus);
		goto Error;
	}
	//cudaStatus = scanV(dArrAllPossibleExtension,noElem_dArrAllPossibleExtension,dArrAllPossibleExtensionScanResult);
	//CHECK(cudaStatus);
	//if(cudaStatus!=cudaSuccess){
	//	status = -1;
	//	fprintf(stderr,"\n scanV dArrAllPossibleExtension failed",cudaStatus);
	//	goto Error;
	//}
	CHECK(cudaStatus=myScanV(dArrAllPossibleExtension,noElem_dArrAllPossibleExtension,dArrAllPossibleExtensionScanResult));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	//Tính kích thước của dArrUniEdge và lưu vào noElem_dArrUniEdge
	CHECK(cudaStatus =getSizeBaseOnScanResult(dArrAllPossibleExtension,dArrAllPossibleExtensionScanResult,noElem_dArrAllPossibleExtension,noElem_dArrUniEdge));
	if(cudaStatus!=cudaSuccess){
		status = -1;
		fprintf(stderr,"\n scanV dArrAllPossibleExtension in extractUniExtension() failed",cudaStatus);
		goto Error;
	}

	//Hiển thị giá trị của noElem_dArrUniEdge
	PMS_PRINT("\n******noElem_dArrUniEdge************\n");
	PMS_PRINT("\n noElem_dArrUniEdge:%d",noElem_dArrUniEdge);

	hLevelUniEdge.at(0).vUE.at(idxEXT).noElem=noElem_dArrUniEdge;

	//Cấp phát bộ nhớ cho dArrUniEdge
	CHECK(cudaStatus = cudaMalloc((void**)&hLevelUniEdge.at(0).vUE.at(idxEXT).dArrUniEdge,noElem_dArrUniEdge*sizeof(UniEdge)));
	if(cudaStatus!=cudaSuccess){
		status =-1;
		fprintf(stderr,"\n cudaMalloc dArrUniEdge  failed",cudaStatus);
		goto Error;
	}

	//lấy nhãn Li lưu vào biến dFromLi	
	int *dFromLi=nullptr;
	CHECK(cudaStatus = cudaMalloc((void**)&dFromLi,sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		status =-1;
		PMS_PRINT("\n cudaMalloc dFromLi failed");
		goto Error;
	}

	kernelGetFromLabel<<<1,1>>>(hLevelEXT.at(0).vE.at(idxEXT).dArrExt,dFromLi);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus=cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		status =-1;
		fprintf(stderr,"\n  kernelGetFromLabel  failed");
		goto Error;
	}
	//Gọi hàm để ánh xạ dữ liệu từ dArrAllPossibleExtension sang mảng dArrUniEdge
	/* Input Data:	dArrAllPossibleExtension, dArrAllPossibleExtensionScanResult,  */
	dim3 blockd(blocksize);
	dim3 gridd((noElem_dArrAllPossibleExtension + blockd.x -1)/blockd.x);
	kernelFilldArrUniEdge<<<gridd,blockd>>>(dArrAllPossibleExtension,dArrAllPossibleExtensionScanResult,noElem_dArrAllPossibleExtension,hLevelUniEdge.at(0).vUE.at(idxEXT).dArrUniEdge,Lv,dFromLi);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		status =-1;
		fprintf(stderr,"\n kernelFilldArrUniEdge failed",cudaStatus);
		goto Error;
	}

	//In nội dung mảng dArrUniEdge
	PMS_PRINT("\n**********dArrUniEdge************");
	FUNCHECK(status=displaydArrUniEdge(hLevelUniEdge.at(0).vUE.at(idxEXT).dArrUniEdge,noElem_dArrUniEdge));
	if(status!=0){
		goto Error;
	}
	//Duyệt qua các cạnh duy nhất tính và lưu trữ độ hỗ trợ của chúng vào một mảng tạm nào đó
	//Sau đó trích những độ hỗ trợ thoả minsup vào lưu vào hLevelUniEdgeSatisfyMinsup
	//Chỉ cần quan tâm kết quả trả về gồm số lượng cạnh thoả minsup, cạnh đó là gì và độ hỗ trợ là bao nhiêu.
	FUNCHECK(status = computeSupportv2(hLevelEXT.at(0).vE.at(idxEXT).dArrExt,hLevelEXT.at(0).vE.at(idxEXT).noElem,hLevelUniEdge.at(0).vUE.at(idxEXT).dArrUniEdge,hLevelUniEdge.at(0).vUE.at(idxEXT).noElem,hLevelUniEdgeSatisfyMinsup.at(0).vecUES.at(idxEXT).noElem,hLevelUniEdgeSatisfyMinsup.at(0).vecUES.at(idxEXT).dArrUniEdge,hLevelUniEdgeSatisfyMinsup.at(0).vecUES.at(idxEXT).hArrSupport));
	if(status!=0){
		goto Error;
	}
	//PMS_PRINT("\n************ dArrUniEdgeSatisfyMinSup*********\n");
	//PMS_PRINT("\n noElem:%d",hLevelUniEdgeSatisfyMinsup.at(0).vecUES.at(idxEXT).noElem);
	//displaydArrUniEdge(hLevelUniEdgeSatisfyMinsup.at(0).vecUES.at(idxEXT).dArrUniEdge,hLevelUniEdgeSatisfyMinsup.at(0).vecUES.at(idxEXT).noElem);
	//for (int j = 0; j < hLevelUniEdgeSatisfyMinsup.at(0).vecUES.at(idxEXT).noElem; j++)
	//{
	//	PMS_PRINT("\n Support: %d ",hLevelUniEdgeSatisfyMinsup.at(0).vecUES.at(idxEXT).hArrSupport[j]);
	//}
	if(hLevelUniEdgeSatisfyMinsup.at(0).vecUES.at(idxEXT).noElem>0){ //Nếu tồn tại mở rộng thoả minSup thì mới gọi hàm Miningv2() để tiếp tục khai thác.
		FUNCHECK(status=Miningv2(hLevelUniEdgeSatisfyMinsup.at(0).vecUES.at(idxEXT).noElem,hLevelUniEdgeSatisfyMinsup.at(0).vecUES.at(idxEXT).dArrUniEdge,hLevelUniEdgeSatisfyMinsup.at(0).vecUES.at(idxEXT).hArrSupport,hLevelEXT.at(0).vE.at(idxEXT).dArrExt,hLevelEXT.at(0).vE.at(idxEXT).noElem,idxEXT));
		if(status!=0){
			goto Error;
		}
	}
	//Giải phóng bộ nhớ trên device
	CHECK(cudaStatus = cudaFree(dFromLi));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus = cudaFree(dArrAllPossibleExtension));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus = cudaFree(dArrValidScanResult));	
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	if(hLevelUniEdge.at(0).vUE.at(idxEXT).noElem>0){
		CHECK(cudaStatus=cudaFree(hLevelUniEdge.at(0).vUE.at(idxEXT).dArrUniEdge));
		if(cudaStatus!=cudaSuccess){
			status=-1;
			goto Error;
		}
	}

	if(hLevelUniEdgeSatisfyMinsup.at(0).vecUES.at(idxEXT).noElem>0){
		CHECK(cudaStatus=cudaFree(hLevelUniEdgeSatisfyMinsup.at(0).vecUES.at(idxEXT).dArrUniEdge));
		if(cudaStatus!=cudaSuccess){
			status=-1;
			goto Error;
		}
	}

	if(hLevelEXT.at(0).vE.at(idxEXT).noElem>0){
		CHECK(cudaStatus = cudaFree(hLevelEXT.at(0).vE.at(idxEXT).dArrExt));
		if(cudaStatus!=cudaSuccess){
			status=-1;
			goto Error;
		}
	}
Error:
	return cudaStatus;
}
//in use
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

//__global__ void kernelmarkValidBackwardEdge_LastExt(EXT* dArrExt, int noElemdArrExt,unsigned int Lv,int *dAllPossibleExtensionBW){
//	int i = blockIdx.x*blockDim.x + threadIdx.x;
//	if(i<noElemdArrExt){
//		if(dArrExt[i].vi > dArrExt[i].vj){ //Chỉ xét các backward
//			//int index=	dArrExt[i].lij*Lv + dArrExt[i].lj;
//			//dAllPossibleExtensionBW[index]=1;
//			dAllPossibleExtensionBW[dArrExt[i].vj]=1;
//		}
//	}
//}


int PMS::markValidForwardEdge(EXT* dArrExt,int noElemdArrExt,unsigned int _Lv,int* dAllPossibleExtension){
	cudaError_t cudaStatus;
	int status =0;

	dim3 block(blocksize);
	dim3 grid((noElemdArrExt+block.x-1)/block.x);

	kernelmarkValidForwardEdge_LastExt<<<grid,block>>>(dArrExt,noElemdArrExt,_Lv,dAllPossibleExtension);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus=cudaGetLastError());
	if (cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

Error:
	return status;
}

//int PMS::markValidBackwardEdge(EXT* dArrExt,int noElemdArrExt,unsigned int _Lv,int* dAllPossibleExtension){
//	cudaError_t cudaStatus;
//	int status =0;
//
//	dim3 block(blocksize);
//	dim3 grid((noElemdArrExt+block.x-1)/block.x);
//
//	kernelmarkValidBackwardEdge_LastExt<<<grid,block>>>(dArrExt,noElemdArrExt,_Lv,dAllPossibleExtension);
//	cudaDeviceSynchronize();
//	CHECK(cudaStatus=cudaGetLastError());
//	if (cudaStatus!=cudaSuccess){
//		status=-1;
//		goto Error;
//	}
//
//Error:
//	return status;
//}


int PMS::extractUniqueForwardBackwardEdge_LastExt(EXTk ext,UniEdgek& ue)
{
	cudaError_t cudaStatus;
	int status =0;

	UniEdgek fwEdgeTemp; //1. need cudaFree
	UniEdgek bwEdgeTemp; //2. need cudaFree

	//Input: EXTk, dTempAllBWExtension,d_allPossibleExtensionBW
	FUNCHECK(status = extractAllBWExtension(bwEdgeTemp,ext)); //Hàm trích tất cả các bacward uniEdge hợp lệ
	if(status!=0){
		goto Error;
	}

	//Input: EXTk, dTempAllBWExtension,d_allPossibleExtensionBW
	FUNCHECK(status = extractAllFWExtension(fwEdgeTemp,ext)); //Hàm trích tất cả các forward uniEdge hợp lệ
	if(status!=0){
		goto Error;
	}

	bwEdgeTemp.Li=fwEdgeTemp.Li; //trích dFromLi
	int *dFromLi=nullptr;
	CHECK(cudaStatus=cudaMalloc((void**)&dFromLi,sizeof(int))); //3. need cudaFree
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaMemcpy(dFromLi,&bwEdgeTemp.Li,sizeof(int),cudaMemcpyHostToDevice));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	/*PMS_PRINT("\n *********** fwEdgeTemp.dArrUniEdge **********\n");
	FUNCHECK(status=displayArrUniEdge(fwEdgeTemp.dArrUniEdge,fwEdgeTemp.noElem));
	if(status!=0){
		goto Error;
	}*/

	/*PMS_PRINT("\n *********** bwEdgeTemp.dArrUniEdge **********\n");
	FUNCHECK(status=displayArrUniEdge(bwEdgeTemp.dArrUniEdge,fwEdgeTemp.noElem));
	if(status!=0){
		goto Error;
	}*/

	//chép kết quả fwEdgeTemp và bwEdgeTemp sang ue.
	FUNCHECK(status = cpResultToUE(fwEdgeTemp,bwEdgeTemp,dFromLi,ue));
	if(status!=0){
		goto Error;
	}

	ue.firstIndexForwardExtension = bwEdgeTemp.noElem;

	PMS_PRINT("\n *********** ue.dArrUniEdge **********\n");
	PMS_PRINT("\n ue.noElem:%d",ue.noElem);
	PMS_PRINT("\n ue.Li:%d",ue.Li);
	PMS_PRINT("\n ue.firstIndexForwardExtension:%d",ue.firstIndexForwardExtension);
	FUNCHECK(status=displayArrUniEdge(ue.dArrUniEdge,ue.noElem));
	if(status!=0){
		goto Error;
	}

	//free memory
	CHECK(cudaStatus=cudaFree(fwEdgeTemp.dArrUniEdge)); //1.
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus=cudaFree(bwEdgeTemp.dArrUniEdge)); //2.
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus=cudaFree(dFromLi)); //3.
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

Error:
	return status;
}
//Trích các mở rộng ở EXTk cuối cùng. Vì ở EXTk cuối tồn tại cả mở rộng forward và backward.
int PMS::extractUniqueForwardBackwardEdge_LastExtv2(EXTk ext,UniEdgek& ue)
{
	cudaError_t cudaStatus;
	int status =0;

	UniEdgek fwEdgeTemp; //1. need cudaFree
	UniEdgek bwEdgeTemp; //2. need cudaFree

	//Nếu số lượng đỉnh trên Right most path trừ 2 (chính là số lượng backward extension có thể có) bằng với số lượng hEmbeddingBWExtension thì sẽ không tồn tại backward extension
	int noBWCurrent = hEmbedding.at(currentColEmbedding).hBackwardEmbedding.size();
	//int noBWCanHave = hRMPv2.at(idxLevel).noElem - 2;
	int noBWCanHave = hLevelRMP.at(objLevel.prevLevel).noElem - 2;

	if(noBWCanHave>noBWCurrent)
	{
		//Input: EXTk, dTempAllBWExtension,d_allPossibleExtensionBW
		FUNCHECK(status = extractAllBWExtensionv2(bwEdgeTemp,ext)); //Hàm trích tất cả các backward uniEdge hợp lệ
		if(status!=0)
		{
			goto Error;
		}
	}

	//Input: EXTk, dTempAllBWExtension,d_allPossibleExtensionBW
	FUNCHECK(status = extractAllFWExtension(fwEdgeTemp,ext)); //Hàm trích tất cả các forward uniEdge hợp lệ //Nếu không có fw thì không trích
	if(status!=0)
	{
		goto Error;
	}

	int *dFromLi=nullptr; //trích nhãn của đỉnh được mở rộng
	CHECK(cudaStatus=cudaMalloc((void**)&dFromLi,sizeof(int))); //3. need cudaFree
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	if (bwEdgeTemp.noElem>0 && fwEdgeTemp.noElem>0 )
	{ //nếu tồn tại backward và forward extension 
		CHECK(cudaStatus = cudaMemcpy(dFromLi,&bwEdgeTemp.Li,sizeof(int),cudaMemcpyHostToDevice)); //trích dFromLi từ cái nào cũng được
		if(cudaStatus!=cudaSuccess)
		{
			status=-1;
			goto Error;
		}
		//chép kết quả fwEdgeTemp và bwEdgeTemp sang ue.
		FUNCHECK(status = cpResultToUE(fwEdgeTemp,bwEdgeTemp,dFromLi,ue));
		if(status!=0)
		{
			goto Error;
		}

		//free memory
		CHECK(cudaStatus=cudaFree(fwEdgeTemp.dArrUniEdge)); //1.
		if(cudaStatus!=cudaSuccess)
		{
			status=-1;
			goto Error;
		}
		CHECK(cudaStatus=cudaFree(bwEdgeTemp.dArrUniEdge)); //2.
		if(cudaStatus!=cudaSuccess)
		{
			status=-1;
			goto Error;
		}
		CHECK(cudaStatus=cudaFree(dFromLi)); //3.
		if(cudaStatus!=cudaSuccess)
		{
			status=-1;
			goto Error;
		}

	}
	else if (fwEdgeTemp.noElem>0)
	{
		CHECK(cudaStatus = cudaMemcpy(dFromLi,&fwEdgeTemp.Li,sizeof(int),cudaMemcpyHostToDevice));
		if(cudaStatus!=cudaSuccess)
		{
			status=-1;
			goto Error;
		}
		FUNCHECK(status = cpResultToUEfw(fwEdgeTemp,dFromLi,ue));
		if(status!=0)
		{
			goto Error;
		}
		//free memory
		CHECK(cudaStatus=cudaFree(fwEdgeTemp.dArrUniEdge)); //1.
		if(cudaStatus!=cudaSuccess)
		{
			status=-1;
			goto Error;
		}
		CHECK(cudaStatus=cudaFree(dFromLi)); //3.
		if(cudaStatus!=cudaSuccess)
		{
			status=-1;
			goto Error;
		}

	}
	else if (bwEdgeTemp.noElem>0)
	{
		CHECK(cudaStatus = cudaMemcpy(dFromLi,&bwEdgeTemp.Li,sizeof(int),cudaMemcpyHostToDevice));
		if(cudaStatus!=cudaSuccess)
		{
			status=-1;
			goto Error;
		}
		FUNCHECK(status = cpResultToUEbw(bwEdgeTemp,dFromLi,ue));
		if(status!=0)
		{
			goto Error;
		}
		CHECK(cudaStatus=cudaFree(bwEdgeTemp.dArrUniEdge)); //2.
		if(cudaStatus!=cudaSuccess)
		{
			status=-1;
			goto Error;
		}
		CHECK(cudaStatus=cudaFree(dFromLi)); //3.
		if(cudaStatus!=cudaSuccess)
		{
			status=-1;
			goto Error;
		}

	}
	else
	{
		goto Error;
	}

	/*PMS_PRINT("\n *********** ue.dArrUniEdge **********\n");
	PMS_PRINT("\n ue.noElem:%d",ue.noElem);
	PMS_PRINT("\n ue.Li:%d",ue.Li);
	PMS_PRINT("\n ue.firstIndexForwardExtension:%d",ue.firstIndexForwardExtension);
	FUNCHECK(status=displayArrUniEdge(ue.dArrUniEdge,ue.noElem));
	if(status!=0)
	{
		goto Error;
	}*/

Error:
	return status;
}
//Đối với các EXTk không ở cuối thì không tồn tại mở rộng backward. Nên hàm này chỉ trích các mở rộng forward hợp lệ.
int PMS::extractUniqueForwardEdge_NonLastExtv2(EXTk ext,UniEdgek& ue){
	cudaError_t cudaStatus;
	int status =0;

	UniEdgek fwEdgeTemp; //1. need cudaFree
	//UniEdgek bwEdgeTemp; //2. need cudaFree

	//Nếu số lượng đỉnh trên Right most path trừ 2 (chính là số lượng backward extension có thể có) bằng với số lượng hEmbeddingBWExtension thì sẽ không tồn tại backward extension
	//int noBWCurrent = hEmbedding.at(currentColEmbedding).hBackwardEmbedding.size();
	//int noBWCanHave = hRMPv2.at(idxLevel).noElem - 2;
	//if(noBWCanHave>noBWCurrent){
	////Input: EXTk, dTempAllBWExtension,d_allPossibleExtensionBW
	//FUNCHECK(status = extractAllBWExtensionv2(bwEdgeTemp,ext)); //Hàm trích tất cả các backward uniEdge hợp lệ
	//if(status!=0){
	//		goto Error;
	//	}
	//}

	//Input: EXTk, dTempAllBWExtension,d_allPossibleExtensionBW
	FUNCHECK(status = extractAllFWExtension(fwEdgeTemp,ext)); //Hàm trích tất cả các forward uniEdge hợp lệ //Nếu không có fw thì không trích
	if(status!=0){
		goto Error;
	}

	int *dFromLi=nullptr; //trích nhãn của đỉnh được mở rộng
	CHECK(cudaStatus=cudaMalloc((void**)&dFromLi,sizeof(int))); //3. need cudaFree
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaMemcpy(dFromLi,&fwEdgeTemp.Li,sizeof(int),cudaMemcpyHostToDevice));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	FUNCHECK(status = cpResultToUEfw(fwEdgeTemp,dFromLi,ue));
	if(status!=0){
		goto Error;
	}
	//free memory
	CHECK(cudaStatus=cudaFree(fwEdgeTemp.dArrUniEdge)); //1.
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus=cudaFree(dFromLi)); //3.
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	//if (bwEdgeTemp.noElem>0 && fwEdgeTemp.noElem>0 ){ //nếu tồn tại backward và forward extension 
	//	CHECK(cudaStatus = cudaMemcpy(dFromLi,&bwEdgeTemp.Li,sizeof(int),cudaMemcpyHostToDevice)); //trích dFromLi từ cái nào cũng được
	//	if(cudaStatus!=cudaSuccess){
	//		status=-1;
	//		goto Error;
	//	}
	//	//chép kết quả fwEdgeTemp và bwEdgeTemp sang ue.
	//	FUNCHECK(status = cpResultToUE(fwEdgeTemp,bwEdgeTemp,dFromLi,ue));
	//	if(status!=0){
	//		goto Error;
	//	}
	//	//free memory
	//	CHECK(cudaStatus=cudaFree(fwEdgeTemp.dArrUniEdge)); //1.
	//	if(cudaStatus!=cudaSuccess){
	//		status=-1;
	//		goto Error;
	//	}
	//	CHECK(cudaStatus=cudaFree(bwEdgeTemp.dArrUniEdge)); //2.
	//	if(cudaStatus!=cudaSuccess){
	//		status=-1;
	//		goto Error;
	//	}
	//	CHECK(cudaStatus=cudaFree(dFromLi)); //3.
	//	if(cudaStatus!=cudaSuccess){
	//		status=-1;
	//		goto Error;
	//	}
	//}
	//else if (fwEdgeTemp.noElem>0){
	//	CHECK(cudaStatus = cudaMemcpy(dFromLi,&fwEdgeTemp.Li,sizeof(int),cudaMemcpyHostToDevice));
	//	if(cudaStatus!=cudaSuccess){
	//		status=-1;
	//		goto Error;
	//	}
	//	FUNCHECK(status = cpResultToUEfw(fwEdgeTemp,dFromLi,ue));
	//	if(status!=0){
	//		goto Error;
	//	}
	//	//free memory
	//	CHECK(cudaStatus=cudaFree(fwEdgeTemp.dArrUniEdge)); //1.
	//	if(cudaStatus!=cudaSuccess){
	//		status=-1;
	//		goto Error;
	//	}
	//	CHECK(cudaStatus=cudaFree(dFromLi)); //3.
	//	if(cudaStatus!=cudaSuccess){
	//		status=-1;
	//		goto Error;
	//	}
	//}
	//else if (bwEdgeTemp.noElem>0){
	//	CHECK(cudaStatus = cudaMemcpy(dFromLi,&bwEdgeTemp.Li,sizeof(int),cudaMemcpyHostToDevice));
	//	if(cudaStatus!=cudaSuccess){
	//		status=-1;
	//		goto Error;
	//	}
	//	FUNCHECK(status = cpResultToUEbw(bwEdgeTemp,dFromLi,ue));
	//	if(status!=0){
	//		goto Error;
	//	}
	//	CHECK(cudaStatus=cudaFree(bwEdgeTemp.dArrUniEdge)); //2.
	//	if(cudaStatus!=cudaSuccess){
	//		status=-1;
	//		goto Error;
	//	}
	//	CHECK(cudaStatus=cudaFree(dFromLi)); //3.
	//	if(cudaStatus!=cudaSuccess){
	//		status=-1;
	//		goto Error;
	//	}
	//}
	//else
	//{
	//	goto Error;
	//}

	PMS_PRINT("\n *********** ue.dArrUniEdge **********\n");
	PMS_PRINT("\n ue.noElem:%d",ue.noElem);
	PMS_PRINT("\n ue.Li:%d",ue.Li);
	PMS_PRINT("\n ue.firstIndexForwardExtension:%d",ue.firstIndexForwardExtension);
	FUNCHECK(status=displayArrUniEdge(ue.dArrUniEdge,ue.noElem));
	if(status!=0){
		goto Error;
	}

Error:
	return status;
}


int PMS::extractAllFWExtension(UniEdgek &ue ,EXTk ext)
{
	int status =0;
	cudaError_t cudaStatus;

	int *dFromLi=nullptr;
	CHECK(cudaStatus = cudaMalloc((void**)&dFromLi,sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	int *dFromVi=nullptr;
	CHECK(cudaStatus = cudaMalloc((void**)&dFromVi,sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	/*PMS_PRINT("\n **********dArrExt*************\n");
	FUNCHECK(status=displayDeviceEXT(ext.dArrExt,ext.noElem));
	if(status!=0){
		goto Error;
	}*/

	dim3 blocka(blocksize);
	dim3 grida((ext.noElem + blocka.x -1)/blocka.x);
	//Cập nhật Nhãn đỉnh được mở rộng
	//kernelGetFromLabelv2<<<grida,blocka>>>(ext.dArrExt,ext.noElem,dFromVi,dFromLi);
	kernelGetFromLabelv3<<<grida,blocka>>>(ext.dArrExt,dFromVi,dFromLi);

	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus=cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaMemcpy(&ue.Li,dFromLi,sizeof(int),cudaMemcpyDeviceToHost));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	//Tính số lượng tất cả các cạnh có thể có dựa vào nhãn của chúng
	int noElem_dallPossibleExtension=Le*Lv;
	int *d_allPossibleExtensionFW=nullptr; //cần được giải phóng cuối hàm
	//cấp phát bộ nhớ cho mảng d_allPossibleExtension
	CHECK(cudaStatus=cudaMalloc((void**)&d_allPossibleExtensionFW,noElem_dallPossibleExtension*sizeof(int)));
	if (cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(d_allPossibleExtensionFW,0,noElem_dallPossibleExtension*sizeof(int)));
		if(cudaStatus!=cudaSuccess){
			status=-1;
			goto Error;
		}
	}

	int *d_allPossibleExtensionScanResultFW=nullptr; //cần được giải phóng cuối hàm
	CHECK(cudaStatus=cudaMalloc((void**)&d_allPossibleExtensionScanResultFW,noElem_dallPossibleExtension*sizeof(int)));
	if (cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}

	//Đánh dấu vị trí các mở rộng forward hợp lệ là 1 tại vị trí d_allPossibleExtension tương ứng
	FUNCHECK(status=markValidForwardEdge(ext.dArrExt,ext.noElem,Lv,d_allPossibleExtensionFW));
	if(status!=0){
		goto Error;
	}

	//FUNCHECK(status=markValidBackwardEdge(ext.dArrExt,ext.noElem,Lv,d_allPossibleExtensionBW)); //nên xoá hàm này, vì không dùng
	//if(status!=0){
	//	goto Error;
	//}


	/*PMS_PRINT("\n************* d_AllPossibleExtensionFW ************\n");
	FUNCHECK(status=displayDeviceArr(d_allPossibleExtensionFW,noElem_dallPossibleExtension));
	if(status!=0){
		goto Error;
	}*/

	CHECK(cudaStatus=myScanV(d_allPossibleExtensionFW,noElem_dallPossibleExtension,d_allPossibleExtensionScanResultFW));
	if(cudaStatus!=cudaSuccess){
		goto Error;
	}

	/*PMS_PRINT("\n************* d_AllPossibleExtensionResultFW ************\n");
	FUNCHECK(status=displayDeviceArr(d_allPossibleExtensionScanResultFW,noElem_dallPossibleExtension));
	if(status!=0){
		goto Error;
	}*/
	int noElem_d_UniqueExtensionFW=0;
	////Tính kích thước của mảng d_UniqueExtension dựa vào kết quả exclusive scan
	CHECK(cudaStatus=getSizeBaseOnScanResult(d_allPossibleExtensionFW,d_allPossibleExtensionScanResultFW,noElem_dallPossibleExtension,noElem_d_UniqueExtensionFW));
	if (cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}

	//PMS_PRINT("\n\nnoElem_d_UniqueExtension:%d",noElem_d_UniqueExtensionFW);
	ue.noElem = noElem_d_UniqueExtensionFW;
	if(ue.noElem==0){
		//giải phóng bộ nhớ và return
		CHECK(cudaStatus=cudaFree(d_allPossibleExtensionFW));
		if(cudaStatus!=cudaSuccess){
			status=-1;
			goto Error;
		}

		CHECK(cudaStatus=cudaFree(d_allPossibleExtensionScanResultFW));
		if(cudaStatus!=cudaSuccess){
			status=-1;
			goto Error;
		}

		CHECK(cudaStatus=cudaFree(dFromLi));
		if(cudaStatus!=cudaSuccess){
			status=-1;
			goto Error;
		}

		CHECK(cudaStatus=cudaFree(dFromVi));
		if(cudaStatus!=cudaSuccess){
			status=-1;
			goto Error;
		}

		return status;
	}

	//Tạo mảng dArrUniEdge với kích thước mảng vừa tính được
	CHECK(cudaStatus=cudaMalloc((void**)&ue.dArrUniEdge,ue.noElem*sizeof(UniEdge)));
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(ue.dArrUniEdge,0,ue.noElem*sizeof(UniEdge)));
		if(cudaStatus!=cudaSuccess){
			status = -1;
			goto Error;
		}
	}

	////Ánh xạ ngược lại từ vị trí trong d_allPossibleExtension thành cạnh và lưu kết quả vào d_UniqueExtension
	dim3 block(blocksize);
	dim3 grid((noElem_dallPossibleExtension + block.x -1)/block.x);
	kernelFilldArrUniEdgev2<<<grid,block>>>(d_allPossibleExtensionFW,d_allPossibleExtensionScanResultFW,noElem_dallPossibleExtension,ue.dArrUniEdge,Lv,dFromLi,dFromVi,maxId);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}	

	/*PMS_PRINT("\n ***************ue.dArrUniEdge **************\n");
	FUNCHECK(status=displayArrUniEdge(ue.dArrUniEdge,ue.noElem));
	if(status!=0){
		goto Error;
	}*/


	//Giải phóng bộ nhớ
	CHECK(cudaStatus=cudaFree(d_allPossibleExtensionFW));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus=cudaFree(d_allPossibleExtensionScanResultFW));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus=cudaFree(dFromLi));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus=cudaFree(dFromVi));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

Error:
	return status;
}





int PMS::extractAllBWExtension(UniEdgek &ue ,EXTk ext){
	int status =0;
	cudaError_t cudaStatus;

	int noElem_dallPossibleExtensionBW=hRMPv2.at(idxLevel).hArrRMP.at(2) + 1;
	int *d_allPossibleExtensionBW=nullptr; //cần được giải phóng ở cuối hàm
	CHECK(cudaStatus=cudaMalloc((void**)&d_allPossibleExtensionBW,noElem_dallPossibleExtensionBW*sizeof(int)));
	if (cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(d_allPossibleExtensionBW,0,noElem_dallPossibleExtensionBW*sizeof(int)));
		if(cudaStatus!=cudaSuccess){
			status=-1;
			goto Error;
		}
	}
	UniEdgek dTempAllBWExtension; //cần được giải phóng ở cuối hàm
	dTempAllBWExtension.noElem=noElem_dallPossibleExtensionBW;
	CHECK(cudaStatus = cudaMalloc((void**)&dTempAllBWExtension.dArrUniEdge,dTempAllBWExtension.noElem*sizeof(UniEdge)));
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}
	//Trích các backward extension duy nhất và lưu vào mảng dTempAllBWExtension.dArrUniEdge
	dim3 block(blocksize);
	dim3 grid((ext.noElem + block.x -1)/block.x);
	kernelextractAllBWExtension<<<grid,block>>>(ext.dArrExt,ext.noElem,dTempAllBWExtension.dArrUniEdge,d_allPossibleExtensionBW);
	cudaDeviceSynchronize();
	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus !=cudaSuccess){
		status =-1;
		goto Error;
	}

	PMS_PRINT("\n *********** dTempAllBWExtension.dArrUniEdge**************\n");
	displayArrUniEdge(dTempAllBWExtension.dArrUniEdge,dTempAllBWExtension.noElem);
	PMS_PRINT("\n *********** dAllPossibleExtensionBW**************\n");
	displayDeviceArr(d_allPossibleExtensionBW,noElem_dallPossibleExtensionBW);
	//Exclusive scan d_allPossibleExtensionBW để biết số lượng các Valid UniEdge
	int *d_allPossibleExtensionScanResultBW=nullptr; //cần được giải phóng ở cuối hàm
	CHECK(cudaStatus=cudaMalloc((void**)&d_allPossibleExtensionScanResultBW,noElem_dallPossibleExtensionBW*sizeof(int)));
	if (cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	CHECK(cudaStatus=myScanV(d_allPossibleExtensionBW,noElem_dallPossibleExtensionBW,d_allPossibleExtensionScanResultBW));
	if(cudaStatus!=cudaSuccess){
		goto Error;
	}

	PMS_PRINT("\n************* d_AllPossibleExtensionResultBW ************\n");
	FUNCHECK(status=displayDeviceArr(d_allPossibleExtensionScanResultBW,noElem_dallPossibleExtensionBW));
	if(status!=0){
		goto Error;
	}
	int noElem_d_UniqueExtensionBW=0;
	CHECK(cudaStatus=getSizeBaseOnScanResult(d_allPossibleExtensionBW,d_allPossibleExtensionScanResultBW,noElem_dallPossibleExtensionBW,noElem_d_UniqueExtensionBW));
	if (cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	PMS_PRINT("\n\nnoElem_d_UniqueExtension:%d",noElem_d_UniqueExtensionBW);
	ue.noElem = noElem_d_UniqueExtensionBW;
	CHECK(cudaStatus=cudaMalloc((void**)&ue.dArrUniEdge,ue.noElem*sizeof(UniEdge)));
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(ue.dArrUniEdge,0,ue.noElem*sizeof(UniEdge)));
		if(cudaStatus!=cudaSuccess){
			status = -1;
			goto Error;
		}
	}

	//Lọc các dòng hợp lệ từ dTempAllBWExtension sang ue
	FUNCHECK(status=extractValidBWExtension(dTempAllBWExtension.dArrUniEdge,dTempAllBWExtension.noElem,ue.dArrUniEdge,d_allPossibleExtensionBW,d_allPossibleExtensionScanResultBW));
	if(status!=0){
		goto Error;
	}

	PMS_PRINT("\n **************ue.dArrUniEdge*****************\n");
	FUNCHECK(status=displayArrUniEdge(ue.dArrUniEdge,ue.noElem));
	if(status!=0){
		goto Error;
	}

	//Giải phóng bộ nhớ tạm
	CHECK(cudaStatus=cudaFree(dTempAllBWExtension.dArrUniEdge));
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}
	CHECK(cudaStatus=cudaFree(d_allPossibleExtensionBW));
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}
	CHECK(cudaStatus=cudaFree(d_allPossibleExtensionScanResultBW));
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}

Error:
	return status;
}
//Trích tất cả các mở rộng backward nếu có trong EXT<k>
int PMS::extractAllBWExtensionv2(UniEdgek &ue ,EXTk ext)
{
	int status =0;
	cudaError_t cudaStatus;

	//int noElem_dallPossibleExtensionBW=hRMPv2.at(idxLevel).hArrRMP.at(2) + 1;
	//Tại sao lại chỉ định là hằng số at.(2)?
	int noElem_dallPossibleExtensionBW = hLevelRMP.at(objLevel.prevLevel).noElem - 2; //Số lượng các mở rộng backward có thể có chính bằng số lượng đỉnh thuộc right most path trừ 2;
	int *d_allPossibleExtensionBW=nullptr; //cần được giải phóng ở cuối hàm
	CHECK(cudaStatus=cudaMalloc((void**)&d_allPossibleExtensionBW,noElem_dallPossibleExtensionBW*sizeof(int)));
	if (cudaStatus!=cudaSuccess)
	{
		status = -1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(d_allPossibleExtensionBW,0,noElem_dallPossibleExtensionBW*sizeof(int)));
		if(cudaStatus!=cudaSuccess)
		{
			status=-1;
			goto Error;
		}
	}
	UniEdgek dTempAllBWExtension; //cần được giải phóng ở cuối hàm
	dTempAllBWExtension.noElem=noElem_dallPossibleExtensionBW;
	CHECK(cudaStatus = cudaMalloc((void**)&dTempAllBWExtension.dArrUniEdge,dTempAllBWExtension.noElem*sizeof(UniEdge)));
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}
	//Trích các backward extension duy nhất và lưu vào mảng dTempAllBWExtension.dArrUniEdge
	dim3 block(blocksize);
	dim3 grid((ext.noElem + block.x -1)/block.x);
	kernelextractAllBWExtension<<<grid,block>>>(ext.dArrExt,ext.noElem,dTempAllBWExtension.dArrUniEdge,d_allPossibleExtensionBW);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus !=cudaSuccess)
	{
		status =-1;
		goto Error;
	}

	/*PMS_PRINT("\n *********** dTempAllBWExtension.dArrUniEdge**************\n");
	FUNCHECK(status=displayArrUniEdge(dTempAllBWExtension.dArrUniEdge,dTempAllBWExtension.noElem));
	if(status!=0)
	{
		goto Error;
	}

	PMS_PRINT("\n *********** dAllPossibleExtensionBW**************\n");
	FUNCHECK(status=displayDeviceArr(d_allPossibleExtensionBW,noElem_dallPossibleExtensionBW));
	if(status!=0)
	{
		goto Error;
	}*/
	//Exclusive scan d_allPossibleExtensionBW để biết số lượng các Valid UniEdge
	int *d_allPossibleExtensionScanResultBW=nullptr; //cần được giải phóng ở cuối hàm
	CHECK(cudaStatus=cudaMalloc((void**)&d_allPossibleExtensionScanResultBW,noElem_dallPossibleExtensionBW*sizeof(int)));
	if (cudaStatus!=cudaSuccess)
	{
		status = -1;
		goto Error;
	}
	CHECK(cudaStatus=myScanV(d_allPossibleExtensionBW,noElem_dallPossibleExtensionBW,d_allPossibleExtensionScanResultBW));
	if(cudaStatus!=cudaSuccess)
	{
		goto Error;
	}

	/*PMS_PRINT("\n************* d_AllPossibleExtensionResultBW ************\n");
	FUNCHECK(status=displayDeviceArr(d_allPossibleExtensionScanResultBW,noElem_dallPossibleExtensionBW));
	if(status!=0)
	{
		goto Error;
	}*/
	int noElem_d_UniqueExtensionBW=0;
	CHECK(cudaStatus=getSizeBaseOnScanResult(d_allPossibleExtensionBW,d_allPossibleExtensionScanResultBW,noElem_dallPossibleExtensionBW,noElem_d_UniqueExtensionBW));
	if (cudaStatus!=cudaSuccess)
	{
		status = -1;
		goto Error;
	}
	//PMS_PRINT("\n\nnoElem_d_UniqueExtension:%d",noElem_d_UniqueExtensionBW);
	ue.noElem=noElem_d_UniqueExtensionBW;
	if(noElem_d_UniqueExtensionBW==0)
	{ //Nếu không có backward extension nào thì giải phóng bộ nhớ tạm và return
		CHECK(cudaStatus=cudaFree(dTempAllBWExtension.dArrUniEdge));
		if(cudaStatus!=cudaSuccess)
		{
			status =-1;
			goto Error;
		}
		CHECK(cudaStatus=cudaFree(d_allPossibleExtensionBW));
		if(cudaStatus!=cudaSuccess)
		{
			status =-1;
			goto Error;
		}
		CHECK(cudaStatus=cudaFree(d_allPossibleExtensionScanResultBW));
		if(cudaStatus!=cudaSuccess)
		{
			status =-1;
			goto Error;
		}

		return status;
	}

	ue.noElem = noElem_d_UniqueExtensionBW;
	CHECK(cudaStatus=cudaMalloc((void**)&ue.dArrUniEdge,ue.noElem*sizeof(UniEdge)));
	if(cudaStatus!=cudaSuccess)
	{
		status = -1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(ue.dArrUniEdge,0,ue.noElem*sizeof(UniEdge)));
		if(cudaStatus!=cudaSuccess)
		{
			status = -1;
			goto Error;
		}
	}

	//Lọc các dòng hợp lệ từ dTempAllBWExtension sang ue
	FUNCHECK(status=extractValidBWExtension(dTempAllBWExtension.dArrUniEdge,dTempAllBWExtension.noElem,ue.dArrUniEdge,d_allPossibleExtensionBW,d_allPossibleExtensionScanResultBW));
	if(status!=0)
	{
		goto Error;
	}

	int *dFromLi=nullptr;
	CHECK(cudaStatus = cudaMalloc((void**)&dFromLi,sizeof(int)));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	int *dFromVi=nullptr;
	CHECK(cudaStatus = cudaMalloc((void**)&dFromVi,sizeof(int)));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	//PMS_PRINT("\n **********dArrExt*************\n");
	//FUNCHECK(status=displayDeviceEXT(ext.dArrExt,ext.noElem));
	//if(status!=0){
	//	goto Error;
	//}

	dim3 blocka(blocksize);
	dim3 grida((ext.noElem + blocka.x -1)/blocka.x);
	/*FUNCHECK(status=displaydArrEXT(ext.dArrExt,ext.noElem));
	if(status!=0)
	{
		goto Error;
	}*/

	//Cập nhật Nhãn đỉnh được mở rộng
	//kernelGetFromLabelv2<<<grida,blocka>>>(ext.dArrExt,ext.noElem,dFromVi,dFromLi);
	kernelGetFromLabelv3<<<grida,blocka>>>(ext.dArrExt,dFromVi,dFromLi);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus=cudaGetLastError());
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaMemcpy(&ue.Li,dFromLi,sizeof(int),cudaMemcpyDeviceToHost));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}


	/*PMS_PRINT("\n **************ue.dArrUniEdge*****************\n");
	FUNCHECK(status=displayArrUniEdge(ue.dArrUniEdge,ue.noElem));
	if(status!=0)
	{
		goto Error;
	}*/

	//Giải phóng bộ nhớ tạm
	CHECK(cudaStatus=cudaFree(dTempAllBWExtension.dArrUniEdge));
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}
	CHECK(cudaStatus=cudaFree(d_allPossibleExtensionBW));
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}
	CHECK(cudaStatus=cudaFree(d_allPossibleExtensionScanResultBW));
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}

	CHECK(cudaStatus=cudaFree(dFromLi));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
Error:
	return status;
}
//Hiển thị nội dung device array. Chép qua host rồi mới hiển thị nội dung.
int PMS::displaydArrEXT(EXT* dArrEXT,int noElem){
	int status= 0;
	cudaError_t cudaStatus;
	//Cấp phát bộ nhớ ở host
	EXT* hArrEXT = (EXT*)malloc(noElem *sizeof(EXT));
	if(hArrEXT==NULL){
		status=-1;
		PMS_PRINT("\n malloc hArrEXT failed\n");
		goto Error;
	} //Chép dữ liệu từ GPU sang HOST
	CHECK(cudaStatus=cudaMemcpy(hArrEXT,dArrEXT,sizeof(EXT)*noElem,cudaMemcpyDeviceToHost));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	//Duyệt và hiển thị từng phần tử. Mỗi phần tử là một cạnh của DFS Code
	for (int i = 0; i < noElem; i++)
	{
		PMS_PRINT("\n dArrEXT[%d]:(vi,vj) (%d %d) (li,lij,lj) (%d,%d,%d) (vgi, vgj) (%d, %d) posRow:%d",i,hArrEXT[i].vi,hArrEXT[i].vj,hArrEXT[i].li,hArrEXT[i].lij,hArrEXT[i].lj,hArrEXT[i].vgi,hArrEXT[i].vgj,hArrEXT[i].posRow);
	}

Error:
	return status;
}


int PMS::extractValidBWExtension(UniEdge* dsrcUniEdge,int noElem,UniEdge*& ddstUniEdge,int *dAllPossibleExtension,int *dAllPossibleExtensionScanResult)
{
	int status = 0;
	cudaError_t cudaStatus;

	dim3 block(blocksize);
	dim3 grid((noElem + block.x -1)/block.x);
	kernelextractValidBWExtension<<<grid,block>>>(dsrcUniEdge,ddstUniEdge,noElem,dAllPossibleExtension,dAllPossibleExtensionScanResult);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}
	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}
Error:
	return status;
}

__global__ void kernelextractValidBWExtension(UniEdge *dsrcUniEdge,UniEdge *ddstUniEdge,int noElem,int *dAllPossibleExtension,int *dAllPossibleExtensionScanResult){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElem){
		if(dAllPossibleExtension[i]==1){
			ddstUniEdge[dAllPossibleExtensionScanResult[i]].vi=dsrcUniEdge[i].vi;
			ddstUniEdge[dAllPossibleExtensionScanResult[i]].vj=dsrcUniEdge[i].vj;
			ddstUniEdge[dAllPossibleExtensionScanResult[i]].li=dsrcUniEdge[i].li;
			ddstUniEdge[dAllPossibleExtensionScanResult[i]].lij=dsrcUniEdge[i].lij;
			ddstUniEdge[dAllPossibleExtensionScanResult[i]].lj=dsrcUniEdge[i].lj;
		}
	}
}




__global__ void kernelextractAllBWExtension(EXT *dArrExt,int noElemdArrExt,UniEdge* dArrUniEdge,int *dAllPossibelExtension){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElemdArrExt){
		if(dArrExt[i].vi>dArrExt[i].vj){
			dAllPossibelExtension[dArrExt[i].vj]=1;
			dArrUniEdge[dArrExt[i].vj].vi=dArrExt[i].vi;
			dArrUniEdge[dArrExt[i].vj].vj=dArrExt[i].vj;
			dArrUniEdge[dArrExt[i].vj].li=dArrExt[i].li;
			dArrUniEdge[dArrExt[i].vj].lj=dArrExt[i].lj;
			dArrUniEdge[dArrExt[i].vj].lij=dArrExt[i].lij;
		}
	}
}


int PMS::cpResultToUE(UniEdgek fwEdgeTemp,UniEdgek bwEdgeTemp,int *dFromLi,UniEdgek& ue){
	int status = 0;
	cudaError_t cudaStatus;

	ue.noElem = fwEdgeTemp.noElem + bwEdgeTemp.noElem;

	CHECK(cudaStatus = cudaMalloc((void**)&ue.dArrUniEdge,sizeof(UniEdge)*ue.noElem));
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}

	CHECK(cudaStatus = cudaMemcpy(&ue.Li,dFromLi,sizeof(int),cudaMemcpyDeviceToHost));
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	/*PMS_PRINT("\n *********** fwEdgeTemp.dArrUniEdge **********\n");
	FUNCHECK(status=displayArrUniEdge(fwEdgeTemp.dArrUniEdge,fwEdgeTemp.noElem));
	if(status!=0){
		goto Error;
	}*/

	/*PMS_PRINT("\n *********** bwEdgeTemp.dArrUniEdge **********\n");
	FUNCHECK(status=displayArrUniEdge(bwEdgeTemp.dArrUniEdge,bwEdgeTemp.noElem));
	if(status!=0){
		goto Error;
	}*/



	dim3 block(blocksize);
	dim3 grid((ue.noElem + block.x -1)/block.x);
	kernelCopyResultToUE<<<grid,block>>>(fwEdgeTemp.dArrUniEdge,bwEdgeTemp.dArrUniEdge,bwEdgeTemp.noElem,ue.dArrUniEdge,ue.noElem);
	cudaDeviceSynchronize();
	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	ue.firstIndexForwardExtension = bwEdgeTemp.noElem; //Trong mảng ue sẽ chứa các mở rộng backward trước và forward sau.
//Nếu ue.firstIndexForwardExtension =-1 ý muốn nói bwEdgeTemp.noElem =-1, tức là trong mảng ue không có mở rộng nào là backward.
Error:
	return status;
}

int PMS::cpResultToUEbw(UniEdgek fwEdgeTemp,int *dFromLi,UniEdgek& ue){
	int status = 0;
	cudaError_t cudaStatus;

	ue.noElem = fwEdgeTemp.noElem;

	CHECK(cudaStatus = cudaMalloc((void**)&ue.dArrUniEdge,sizeof(UniEdge)*ue.noElem));
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}

	CHECK(cudaStatus = cudaMemcpy(&ue.Li,dFromLi,sizeof(int),cudaMemcpyDeviceToHost));
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	/*PMS_PRINT("\n *********** fwEdgeTemp.dArrUniEdge **********\n");
	FUNCHECK(status=displayArrUniEdge(fwEdgeTemp.dArrUniEdge,fwEdgeTemp.noElem));
	if(status!=0){
		goto Error;
	}*/

	dim3 block(blocksize);
	dim3 grid((ue.noElem + block.x -1)/block.x);
	kernelCopyResultToUE<<<grid,block>>>(fwEdgeTemp.dArrUniEdge,ue.dArrUniEdge,ue.noElem);
	cudaDeviceSynchronize();
	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	//ue.firstIndexForwardExtension = 0;

Error:
	return status;
}

int PMS::cpResultToUEfw(UniEdgek bwEdgeTemp,int *dFromLi,UniEdgek& ue){
	int status = 0;
	cudaError_t cudaStatus;

	ue.noElem = bwEdgeTemp.noElem;

	CHECK(cudaStatus = cudaMalloc((void**)&ue.dArrUniEdge,sizeof(UniEdge)*ue.noElem));
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}

	CHECK(cudaStatus = cudaMemcpy(&ue.Li,dFromLi,sizeof(int),cudaMemcpyDeviceToHost));
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	/*PMS_PRINT("\n *********** fwEdgeTemp.dArrUniEdge **********\n");
	FUNCHECK(status=displayArrUniEdge(bwEdgeTemp.dArrUniEdge,bwEdgeTemp.noElem));
	if(status!=0){
		goto Error;
	}*/

	dim3 block(blocksize);
	dim3 grid((ue.noElem + block.x -1)/block.x);
	kernelCopyResultToUE<<<grid,block>>>(bwEdgeTemp.dArrUniEdge,ue.dArrUniEdge,ue.noElem);
	cudaDeviceSynchronize();
	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	ue.firstIndexForwardExtension = 0;

Error:
	return status;
}


__global__ void kernelCopyResultToUE(UniEdge *fwdArrUniEdge,UniEdge *bwdArrUniEdge,int bwnoElem,UniEdge *uedArrUniEdge,int uenoElem){
	int i = blockDim.x * blockIdx.x  + threadIdx.x;
	if(i<uenoElem){
		if(i<bwnoElem){
			uedArrUniEdge[i].vi=bwdArrUniEdge[i].vi;
			uedArrUniEdge[i].vj=bwdArrUniEdge[i].vj;
			uedArrUniEdge[i].li=bwdArrUniEdge[i].li;
			uedArrUniEdge[i].lij=bwdArrUniEdge[i].lij;
			uedArrUniEdge[i].lj=bwdArrUniEdge[i].lj;
		}
		else
		{
			uedArrUniEdge[i].vi=fwdArrUniEdge[i-bwnoElem].vi;
			uedArrUniEdge[i].vj=fwdArrUniEdge[i-bwnoElem].vj;
			uedArrUniEdge[i].li=fwdArrUniEdge[i-bwnoElem].li;
			uedArrUniEdge[i].lij=fwdArrUniEdge[i-bwnoElem].lij;
			uedArrUniEdge[i].lj=fwdArrUniEdge[i-bwnoElem].lj;
		}
	}
}

__global__ void kernelCopyResultToUE(UniEdge *fwdArrUniEdge,UniEdge *uedArrUniEdge,int uenoElem){
	int i = blockDim.x * blockIdx.x  + threadIdx.x;
	if(i<uenoElem){
		uedArrUniEdge[i].vi=fwdArrUniEdge[i].vi;
		uedArrUniEdge[i].vj=fwdArrUniEdge[i].vj;
		uedArrUniEdge[i].li=fwdArrUniEdge[i].li;
		uedArrUniEdge[i].lij=fwdArrUniEdge[i].lij;
		uedArrUniEdge[i].lj=fwdArrUniEdge[i].lj;
	}
}

int PMS::extractValidExtensionTodExtv2(EXT *dArrExtension,V *dArrV,int noElem_dArrV,int idxEXT){
	cudaError_t cudaStatus;
	int status =0;
	// Scan mảng dArrValid để lấy kích thước của mảng cần tạo
	int *dArrValidScanResult = nullptr; //1. Cần được giải phóng ở cuối hàm

	CHECK(cudaStatus = cudaMalloc((void**)&dArrValidScanResult,sizeof(int)*noElem_dArrV));
	if (cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus = cudaMemset(dArrValidScanResult,0,sizeof(int)*noElem_dArrV));
		if (cudaStatus!=cudaSuccess){
			status = -1;
			goto Error;
		}
	}

	//cudaStatus = scanV(dArrV->valid,noElem_dArrV,dArrValidScanResult); //hàm scan này có vấn đề. Nó làm thay đổi giá trị đầu vào.
	//if (cudaStatus!=cudaSuccess){
	//	status = -1;
	//	fprintf(stderr,"\n scanV dArrValid in extractValidExtensionToExt() failed");
	//	goto Error;
	//}

	CHECK(cudaStatus = myScanV(dArrV->valid,noElem_dArrV,dArrValidScanResult));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	//In nội dung kết quả dArrValidScanResult
	PMS_PRINT("\n********dArrValid******\n");
	FUNCHECK(status=displayDeviceArr(dArrV->valid,noElem_dArrV));
	if(status!=0){
		goto Error;
	}


	PMS_PRINT("\n********dArrValidScanResult******\n");
	FUNCHECK(status=displayDeviceArr(dArrValidScanResult,noElem_dArrV));
	if(status!=0){
		goto Error;
	}
	////3. Lấy kích thước của mảng dArrExt;
	int noElem_dExt=0;
	CHECK(cudaStatus=getSizeBaseOnScanResult(dArrV->valid,dArrValidScanResult,noElem_dArrV,noElem_dExt));
	if (cudaStatus!=cudaSuccess){
		status = -1;
		fprintf(stderr,"\n getSizeBaseOnScanResult in extractValidExtensionToExt() failed");
		goto Error;
	}

	//In nội dung noElem_dExt
	PMS_PRINT("\n******** noElem In dArrExt ******\n");
	PMS_PRINT("\n noElem_dExt:%d",noElem_dExt);

	//**************** Nếu không tìm được mở rộng nào thì return *************/
	if (noElem_dExt == 0) 
	{
		CHECK(cudaStatus=cudaFree(dArrValidScanResult));
		if(cudaStatus!=cudaSuccess){
			status =-1;
			goto Error;
		}
		return status;
	}
	//Nếu tìm được mở rộng thì xây dựng EXTk, rồi trích các mở rộng duy nhất và tính độ hỗ trợ của chúng. Đồng thời
	//lọc ra các độ hỗ trợ thoả minsup
	//Quản lý theo Level
	//4. Khởi tạo mảng dArrExt có kích thước noElem_dExt rồi trích dữ liệu từ dArrExtension sang dựa vào dArrValid.
	hLevelEXTv2.at(idxLevel).vE.at(idxEXT).noElem = noElem_dExt;	//Cập nhật số lượng mở rộng hợp lệ cho EXT của 
	CHECK(cudaStatus = cudaMalloc((void**)&hLevelEXTv2.at(idxLevel).vE.at(idxEXT).dArrExt,noElem_dExt*sizeof(EXT))); //2. Mảng dArrExt này phải được giải phóng ở cuối hàm
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus = cudaMemset(hLevelEXTv2.at(idxLevel).vE.at(idxEXT).dArrExt,0,sizeof(EXT)*noElem_dExt));
		if(cudaStatus!=cudaSuccess){
			status =-1;
			goto Error;
		}
	}
	dim3 block(blocksize);
	dim3 grid((noElem_dArrV+block.x -1)/block.x);
	kernelExtractValidExtensionTodExt<<<grid,block>>>(dArrExtension,dArrV->valid,dArrValidScanResult,noElem_dArrV,hLevelEXTv2.at(idxLevel).vE.at(idxEXT).dArrExt,noElem_dExt);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	////In mảng dExt;
	PMS_PRINT("\n********** dArrExt **********\n");
	CHECK(cudaStatus =printdExt(hLevelEXTv2.at(idxLevel).vE.at(idxEXT).dArrExt,noElem_dExt));
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}

	//Chuẩn bị bộ nhớ để trích các cạnh duy nhất
	//Trích các mở rộng duy nhất forward: lưu chúng vào mảng dUniEdgeForwardTemp
	//Các tham số:1. dArrExt: để trích các forward uniedge extension
	//hLevelUniEdgev2.at(idxLevel).vUE.at(0); //chúng ta lưu vào đây, vì sao? vì đây là hàm khai thác các backward và forward edge ở EXT cuối.
	FUNCHECK(status=extractUniqueForwardBackwardEdge_LastExt(hLevelEXTv2.at(idxLevel).vE.at(idxEXT),hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT)));
	if(status!=0){
		goto Error;
	}

	PMS_PRINT("\n ************ dArrUniEdge from hLevelUniEdgev2 ***************\n");
	PMS_PRINT("\n noElem:%d",hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT).noElem);
	PMS_PRINT("\n Li:%d",hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT).Li);
	PMS_PRINT("\n firstIndex of forward extension:%d",hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT).firstIndexForwardExtension);
	FUNCHECK(status=displayArrUniEdge(hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT).dArrUniEdge,hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT).noElem)); //3. bộ nhớ dArrUniEdge phỉ được giải phóng ở cuối hàm
	if(status!=0){
		goto Error;
	}


	//Duyệt qua mảng dUniEdge để tính độ hỗ trợ. 
	//Như vậy, trong trường hợp này chúng ta sẽ tính độ hỗ trợ cho các mở rộng backward trước nếu có, rồi mới đến forward.
	//vì backward nằm trước forward extension trong mảng dArrUniEdge
	////Sau đó trích những độ hỗ trợ thoả minsup vào lưu vào hLevelUniEdgeSatisfyMinsup

	//FUNCHECK(status = computeSupportv2(hLevelEXTv2.at(idxLevel).vE.at(idxEXT).dArrExt,hLevelEXTv2.at(idxLevel).vE.at(idxEXT).noElem,hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT).dArrUniEdge,hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT).noElem,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).noElem,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).dArrUniEdge,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).hArrSupport));
	//if(status!=0){
	//	goto Error;
	//}
	FUNCHECK(status = computeSupportv3(hLevelEXTv2.at(idxLevel).vE.at(idxEXT).dArrExt,hLevelEXTv2.at(idxLevel).vE.at(idxEXT).noElem,hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT).dArrUniEdge,hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT).noElem,hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT).firstIndexForwardExtension,hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT).Li,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).noElem,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).dArrUniEdge,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).hArrSupport));
	if(status!=0){
		goto Error;
	}

	//Hiển thị nội dung mảng dArrUniEde
	PMS_PRINT("\n ************ dArrUniEdgeStatisfy minSup ***************\n");
	FUNCHECK(status=displayArrUniEdge(hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).dArrUniEdge,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).noElem)); //4. dArrUniEdge phải được giải phóng ở cuối hàm
	if(status!=0){
		goto Error;
	}

	PMS_PRINT("\n");
	for (int i = 0; i < hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).noElem; i++)
	{
		PMS_PRINT("\n Sup[%d]:%d",i,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).hArrSupport[i]);
	}

	//status=Miningv2(hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).noElem,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).dArrUniEdge,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).hArrSupport,hLevelEXTv2.at(idxLevel).vE.at(idxEXT).dArrExt,hLevelEXTv2.at(idxLevel).vE.at(idxEXT).noElem,idxEXT);
	//FUNCHECK(status);
	//if(status!=0){
	//	goto Error;
	//}
	//Nếu không có phần tử nào thoả minsup thì không khai thác nữa mà quay lui để xét segment EXT kế tiếp
	if(hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).noElem>0){
		FUNCHECK(status=Miningv3(hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).noElem,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).dArrUniEdge,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).hArrSupport,hLevelEXTv2.at(idxLevel).vE.at(idxEXT).dArrExt,hLevelEXTv2.at(idxLevel).vE.at(idxEXT).noElem,idxEXT));
		if(status!=0){
			goto Error;
		}
	}
	else
	{
		PMS_PRINT("\n No any extension that satisfy minsup at Level: %d of segment EXT: %d",Level,idxEXT);
	}

	//Giải phóng bộ nhớ trên device
	CHECK(cudaStatus = cudaFree(dArrValidScanResult));	//1.
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	if(hLevelEXTv2.at(idxLevel).vE.at(idxEXT).noElem>0){
		CHECK(cudaStatus = cudaFree(hLevelEXTv2.at(idxLevel).vE.at(idxEXT).dArrExt));//2.
		if(cudaStatus!=cudaSuccess){
			status=-1;
			goto Error;
		}
	}
	if(hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT).noElem>0){
		CHECK(cudaStatus = cudaFree(hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT).dArrUniEdge));//3.
		if(cudaStatus!=cudaSuccess){
			status=-1;
			goto Error;
		}
	}
	if(hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).noElem>0){
		CHECK(cudaStatus = cudaFree(hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).dArrUniEdge));//4.
		if(cudaStatus!=cudaSuccess){
			status=-1;
			goto Error;
		}
	}
Error:
	return cudaStatus;
}

int PMS::extractValidExtensionTodExtv3(EXT *dArrExtension,V *dArrV,int noElem_dArrV,int idxEXT)
{
	cudaError_t cudaStatus;
	int status =0;
	// Scan mảng dArrValid để lấy kích thước của mảng cần tạo
	int *dArrValidScanResult = nullptr; //1. Cần được giải phóng ở cuối hàm

	CHECK(cudaStatus = cudaMalloc((void**)&dArrValidScanResult,sizeof(int)*noElem_dArrV));
	if (cudaStatus!=cudaSuccess)
	{
		status = -1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus = cudaMemset(dArrValidScanResult,0,sizeof(int)*noElem_dArrV));
		if (cudaStatus!=cudaSuccess)
		{
			status = -1;
			goto Error;
		}
	}

	//cudaStatus = scanV(dArrV->valid,noElem_dArrV,dArrValidScanResult); //hàm scan này có vấn đề. Nó làm thay đổi giá trị đầu vào.
	//if (cudaStatus!=cudaSuccess){
	//	status = -1;
	//	fprintf(stderr,"\n scanV dArrValid in extractValidExtensionToExt() failed");
	//	goto Error;
	//}

	CHECK(cudaStatus = myScanV(dArrV->valid,noElem_dArrV,dArrValidScanResult));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	//In nội dung kết quả dArrValidScanResult
	/*PMS_PRINT("\n********dArrValid******\n");
	FUNCHECK(status=displayDeviceArr(dArrV->valid,noElem_dArrV));
	if(status!=0)
	{
		goto Error;
	}*/


	/*PMS_PRINT("\n********dArrValidScanResult******\n");
	FUNCHECK(status=displayDeviceArr(dArrValidScanResult,noElem_dArrV));
	if(status!=0)
	{
		goto Error;
	}*/
	////3. Lấy kích thước của mảng dArrExt;
	int noElem_dExt=0;
	CHECK(cudaStatus=getSizeBaseOnScanResult(dArrV->valid,dArrValidScanResult,noElem_dArrV,noElem_dExt));
	if (cudaStatus!=cudaSuccess)
	{
		status = -1;
		fprintf(stderr,"\n getSizeBaseOnScanResult in extractValidExtensionToExt() failed");
		goto Error;
	}

	//In nội dung noElem_dExt
	/*PMS_PRINT("\n******** noElem In dArrExt ******\n");
	PMS_PRINT("\n noElem_dExt:%d",noElem_dExt);*/

	//**************** Nếu không tìm được mở rộng nào thì return *************/
	if (noElem_dExt == 0) 
	{
		CHECK(cudaStatus=cudaFree(dArrValidScanResult));
		if(cudaStatus!=cudaSuccess)
		{
			status =-1;
			goto Error;
		}
		return status;
	}
	//Nếu tìm được mở rộng thì xây dựng EXTk, rồi trích các mở rộng duy nhất và tính độ hỗ trợ của chúng. Đồng thời
	//lọc ra các độ hỗ trợ thoả minsup
	//Quản lý theo Level
	//4. Khởi tạo mảng dArrExt có kích thước noElem_dExt rồi trích dữ liệu từ dArrExtension sang dựa vào dArrValid.
	//hLevelEXTv2.at(idxLevel).vE.at(idxEXT).noElem = noElem_dExt;	//Cập nhật số lượng mở rộng hợp lệ cho EXT của 
	hLevelEXT.at(objLevel.Level).vE.at(idxEXT).noElem= noElem_dExt;
	CHECK(cudaStatus = cudaMalloc((void**)&hLevelEXT.at(objLevel.Level).vE.at(idxEXT).dArrExt,noElem_dExt*sizeof(EXT))); //2. Mảng dArrExt này phải được giải phóng ở cuối hàm
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus = cudaMemset(hLevelEXT.at(objLevel.Level).vE.at(idxEXT).dArrExt,0,sizeof(EXT)*noElem_dExt));
		if(cudaStatus!=cudaSuccess)
		{
			status =-1;
			goto Error;
		}
	}
	dim3 block(blocksize);
	dim3 grid((noElem_dArrV+block.x -1)/block.x);
	kernelExtractValidExtensionTodExt<<<grid,block>>>(dArrExtension,dArrV->valid,dArrValidScanResult,noElem_dArrV,hLevelEXT.at(objLevel.Level).vE.at(idxEXT).dArrExt,noElem_dExt);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess)
	{
		status = -1;
		goto Error;
	}
	////In mảng dExt;
	/*PMS_PRINT("\n********** dArrExt **********\n");
	CHECK(cudaStatus =printdExt(hLevelEXT.at(objLevel.Level).vE.at(idxEXT).dArrExt,noElem_dExt));
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}*/



	//Chuẩn bị bộ nhớ để trích các cạnh duy nhất
	//Trích các mở rộng duy nhất forward: lưu chúng vào mảng dUniEdgeForwardTemp
	//Các tham số:1. dArrExt: để trích các forward uniedge extension
	//hLevelUniEdgev2.at(idxLevel).vUE.at(0); //chúng ta lưu vào đây, vì sao? vì đây là hàm khai thác các backward và forward edge ở EXT cuối.
	//FUNCHECK(status=extractUniqueForwardBackwardEdge_LastExtv2(hLevelEXTv2.at(idxLevel).vE.at(idxEXT),hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT)));
	hLevelUniEdge.resize(1);
	hLevelUniEdge.at(0).noElem=1;
	hLevelUniEdge.at(0).vUE.resize(1);

	FUNCHECK(status=extractUniqueForwardBackwardEdge_LastExtv2(hLevelEXT.at(objLevel.Level).vE.at(idxEXT),hLevelUniEdge.at(0).vUE.at(0)));
	if(status!=0)
	{
		goto Error;
	}
	//? extractUniqueForwardEdge_NonLastExt đã viết
	//PMS_PRINT("\n ************ dArrUniEdge from hLevelUniEdgev2 ***************\n");
	//PMS_PRINT("\n noElem:%d",hLevelUniEdge.at(0).vUE.at(0).noElem);
	//PMS_PRINT("\n Li:%d",hLevelUniEdge.at(0).vUE.at(0).Li);
	//PMS_PRINT("\n firstIndex of forward extension:%d",hLevelUniEdge.at(0).vUE.at(0).firstIndexForwardExtension);
	//FUNCHECK(status=displayArrUniEdge(hLevelUniEdge.at(0).vUE.at(0).dArrUniEdge,hLevelUniEdge.at(0).vUE.at(0).noElem)); //3. bộ nhớ dArrUniEdge phỉ được giải phóng ở cuối hàm
	//if(status!=0)
	//{
	//	goto Error;
	//}



	//Duyệt qua mảng dUniEdge để tính độ hỗ trợ. 
	//Như vậy, trong trường hợp này chúng ta sẽ tính độ hỗ trợ cho các mở rộng backward trước nếu có, rồi mới đến forward.
	//vì backward nằm trước forward extension trong mảng dArrUniEdge
	////Sau đó trích những độ hỗ trợ thoả minsup vào lưu vào hLevelUniEdgeSatisfyMinsup

	//FUNCHECK(status = computeSupportv2(hLevelEXTv2.at(idxLevel).vE.at(idxEXT).dArrExt,hLevelEXTv2.at(idxLevel).vE.at(idxEXT).noElem,hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT).dArrUniEdge,hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT).noElem,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).noElem,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).dArrUniEdge,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).hArrSupport));
	//if(status!=0){
	//	goto Error;
	//}
	//FUNCHECK(status = computeSupportv3(hLevelEXTv2.at(idxLevel).vE.at(idxEXT).dArrExt,hLevelEXTv2.at(idxLevel).vE.at(idxEXT).noElem,hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT).dArrUniEdge,hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT).noElem,hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT).firstIndexForwardExtension,hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT).Li,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).noElem,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).dArrUniEdge,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).hArrSupport));
	//if(status!=0){
	//	goto Error;
	//}
	FUNCHECK(status = computeSupportv4(hLevelEXT.at(objLevel.Level).vE.at(idxEXT).dArrExt,hLevelEXT.at(objLevel.Level).vE.at(idxEXT).noElem,hLevelUniEdge.at(0).vUE.at(0).dArrUniEdge,hLevelUniEdge.at(0).vUE.at(0).noElem,hLevelUniEdge.at(0).vUE.at(0).firstIndexForwardExtension,hLevelUniEdge.at(0).vUE.at(0).Li,hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(idxEXT).noElem,hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(idxEXT).dArrUniEdge,hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(idxEXT).hArrSupport));
	if(status!=0)
	{
		goto Error;
	}

	//Hiển thị nội dung mảng dArrUniEdeSatisfyMinSup
	//PMS_PRINT("\n ************ dArrUniEdgeStatisfy minSup ***************\n");
	//FUNCHECK(status=displayArrUniEdge(hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(idxEXT).dArrUniEdge,hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(idxEXT).noElem)); //4. dArrUniEdge phải được giải phóng ở cuối hàm
	//if(status!=0) 
	//{
	//	goto Error;
	//}
	//PMS_PRINT("\n");
	//for (int i = 0; i < hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(idxEXT).noElem; i++)
	//{
	//	PMS_PRINT("\n Sup[%d]:%d",i,hLevelUniEdgeSatisfyMinsup.at(objLevel.Level).vecUES.at(idxEXT).hArrSupport[i]);
	//}
	/*
	//status=Miningv2(hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).noElem,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).dArrUniEdge,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).hArrSupport,hLevelEXTv2.at(idxLevel).vE.at(idxEXT).dArrExt,hLevelEXTv2.at(idxLevel).vE.at(idxEXT).noElem,idxEXT);
	//FUNCHECK(status);
	//if(status!=0){
	//	goto Error;
	//}

	if(hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).noElem>0){
	FUNCHECK(status=Miningv3(hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).noElem,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).dArrUniEdge,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).hArrSupport,hLevelEXTv2.at(idxLevel).vE.at(idxEXT).dArrExt,hLevelEXTv2.at(idxLevel).vE.at(idxEXT).noElem,idxEXT));
	if(status!=0)
	{
	goto Error;
	}
	}

	*/
	//Giải phóng bộ nhớ trên device
	if(hLevelUniEdge.at(0).vUE.at(0).noElem>0)
	{
		CHECK(cudaStatus=cudaFree(hLevelUniEdge.at(0).vUE.at(0).dArrUniEdge));
		if(cudaStatus != cudaSuccess)
		{
			status=-1;
			goto Error;
		}
		hLevelUniEdge.at(0).vUE.clear();
		hLevelUniEdge.clear();
	}

	CHECK(cudaStatus = cudaFree(dArrValidScanResult));	//1.
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

Error:
	return status;
}

int PMS::extractValidExtensionTodExtv4(EXT *dArrExtension,V *dArrV,int noElem_dArrV,int idxEXT){
	cudaError_t cudaStatus;
	int status =0;
	// Scan mảng dArrValid để lấy kích thước của mảng cần tạo
	int *dArrValidScanResult = nullptr; //1. Cần được giải phóng ở cuối hàm

	CHECK(cudaStatus = cudaMalloc((void**)&dArrValidScanResult,sizeof(int)*noElem_dArrV));
	if (cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus = cudaMemset(dArrValidScanResult,0,sizeof(int)*noElem_dArrV));
		if (cudaStatus!=cudaSuccess){
			status = -1;
			goto Error;
		}
	}

	//cudaStatus = scanV(dArrV->valid,noElem_dArrV,dArrValidScanResult); //hàm scan này có vấn đề. Nó làm thay đổi giá trị đầu vào.
	//if (cudaStatus!=cudaSuccess){
	//	status = -1;
	//	fprintf(stderr,"\n scanV dArrValid in extractValidExtensionToExt() failed");
	//	goto Error;
	//}

	CHECK(cudaStatus = myScanV(dArrV->valid,noElem_dArrV,dArrValidScanResult)); //Sccan để biết có bao nhiêu mở rộng hợp lệ
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	//In nội dung kết quả dArrValidScanResult
	PMS_PRINT("\n********dArrValid******\n");
	FUNCHECK(status=displayDeviceArr(dArrV->valid,noElem_dArrV));
	if(status!=0){
		goto Error;
	}


	PMS_PRINT("\n********dArrValidScanResult******\n");
	FUNCHECK(status=displayDeviceArr(dArrValidScanResult,noElem_dArrV));
	if(status!=0){
		goto Error;
	}
	////3. Lấy kích thước của mảng dArrExt;
	int noElem_dExt=0;
	CHECK(cudaStatus=getSizeBaseOnScanResult(dArrV->valid,dArrValidScanResult,noElem_dArrV,noElem_dExt));
	if (cudaStatus!=cudaSuccess){
		status = -1;
		fprintf(stderr,"\n getSizeBaseOnScanResult in extractValidExtensionToExt() failed");
		goto Error;
	}

	//In nội dung noElem_dExt
	PMS_PRINT("\n******** noElem In dArrExt ******\n");
	PMS_PRINT("\n noElem_dExt:%d",noElem_dExt);

	//**************** Nếu không tìm được mở rộng nào thì return *************/
	if (noElem_dExt == 0) 
	{
		CHECK(cudaStatus=cudaFree(dArrValidScanResult));
		if(cudaStatus!=cudaSuccess){
			status =-1;
			goto Error;
		}
		return status;
	}
	//Nếu tìm được mở rộng thì xây dựng EXTk, rồi trích các mở rộng duy nhất và tính độ hỗ trợ của chúng. Đồng thời
	//lọc ra các độ hỗ trợ thoả minsup
	//Quản lý theo Level
	//4. Khởi tạo mảng dArrExt có kích thước noElem_dExt rồi trích dữ liệu từ dArrExtension sang dựa vào dArrValid.
	hLevelEXTv2.at(idxLevel).vE.at(idxEXT).noElem = noElem_dExt;	//Cập nhật số lượng mở rộng hợp lệ cho EXT của 
	CHECK(cudaStatus = cudaMalloc((void**)&hLevelEXTv2.at(idxLevel).vE.at(idxEXT).dArrExt,noElem_dExt*sizeof(EXT))); //2. Mảng dArrExt này phải được giải phóng ở cuối hàm
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus = cudaMemset(hLevelEXTv2.at(idxLevel).vE.at(idxEXT).dArrExt,0,sizeof(EXT)*noElem_dExt));
		if(cudaStatus!=cudaSuccess){
			status =-1;
			goto Error;
		}
	}
	dim3 block(blocksize);
	dim3 grid((noElem_dArrV+block.x -1)/block.x);
	kernelExtractValidExtensionTodExt<<<grid,block>>>(dArrExtension,dArrV->valid,dArrValidScanResult,noElem_dArrV,hLevelEXTv2.at(idxLevel).vE.at(idxEXT).dArrExt,noElem_dExt);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	////In mảng dExt;
	PMS_PRINT("\n********** dArrExt **********\n");
	CHECK(cudaStatus =printdExt(hLevelEXTv2.at(idxLevel).vE.at(idxEXT).dArrExt,noElem_dExt));
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}

	//Chuẩn bị bộ nhớ để trích các cạnh duy nhất
	//Trích các mở rộng duy nhất forward: lưu chúng vào mảng dUniEdgeForwardTemp
	//Các tham số:1. dArrExt: để trích các forward uniedge extension
	//hLevelUniEdgev2.at(idxLevel).vUE.at(0); //chúng ta lưu vào đây, vì sao? vì đây là hàm khai thác các backward và forward edge ở EXT cuối.
	//FUNCHECK(status=extractUniqueForwardBackwardEdge_LastExtv2(hLevelEXTv2.at(idxLevel).vE.at(idxEXT),hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT)));
	//if(status!=0){
	//	goto Error;
	//}
	FUNCHECK(status=extractUniqueForwardEdge_NonLastExtv2(hLevelEXTv2.at(idxLevel).vE.at(idxEXT),hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT)));
	if(status!=0){
		goto Error;
	}

	PMS_PRINT("\n ************ dArrUniEdge from hLevelUniEdgev2 ***************\n");
	PMS_PRINT("\n noElem:%d",hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT).noElem);
	PMS_PRINT("\n Li:%d",hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT).Li);
	PMS_PRINT("\n firstIndex of forward extension:%d",hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT).firstIndexForwardExtension);
	FUNCHECK(status=displayArrUniEdge(hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT).dArrUniEdge,hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT).noElem)); //3. bộ nhớ dArrUniEdge phỉ được giải phóng ở cuối hàm
	if(status!=0){
		goto Error;
	}


	//Duyệt qua mảng dUniEdge để tính độ hỗ trợ. 
	//Như vậy, trong trường hợp này chúng ta sẽ tính độ hỗ trợ cho các mở rộng backward trước nếu có, rồi mới đến forward.
	//vì backward nằm trước forward extension trong mảng dArrUniEdge
	////Sau đó trích những độ hỗ trợ thoả minsup vào lưu vào hLevelUniEdgeSatisfyMinsup

	//FUNCHECK(status = computeSupportv2(hLevelEXTv2.at(idxLevel).vE.at(idxEXT).dArrExt,hLevelEXTv2.at(idxLevel).vE.at(idxEXT).noElem,hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT).dArrUniEdge,hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT).noElem,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).noElem,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).dArrUniEdge,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).hArrSupport));
	//if(status!=0){
	//	goto Error;
	//}
	//FUNCHECK(status = computeSupportv3(hLevelEXTv2.at(idxLevel).vE.at(idxEXT).dArrExt,hLevelEXTv2.at(idxLevel).vE.at(idxEXT).noElem,hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT).dArrUniEdge,hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT).noElem,hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT).firstIndexForwardExtension,hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT).Li,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).noElem,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).dArrUniEdge,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).hArrSupport));
	//if(status!=0){
	//	goto Error;
	//}
	FUNCHECK(status = computeSupportv4(hLevelEXTv2.at(idxLevel).vE.at(idxEXT).dArrExt,hLevelEXTv2.at(idxLevel).vE.at(idxEXT).noElem,hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT).dArrUniEdge,hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT).noElem,hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT).firstIndexForwardExtension,hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT).Li,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).noElem,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).dArrUniEdge,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).hArrSupport));
	if(status!=0){
		goto Error;
	}

	//Hiển thị nội dung mảng dArrUniEde
	PMS_PRINT("\n ************ dArrUniEdgeStatisfy minSup ***************\n");
	FUNCHECK(status=displayArrUniEdge(hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).dArrUniEdge,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).noElem)); //4. dArrUniEdge phải được giải phóng ở cuối hàm
	if(status!=0){
		goto Error;
	}
	PMS_PRINT("\n");
	for (int i = 0; i < hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).noElem; i++)
	{
		PMS_PRINT("\n Sup[%d]:%d",i,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).hArrSupport[i]);
	}

	//status=Miningv2(hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).noElem,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).dArrUniEdge,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).hArrSupport,hLevelEXTv2.at(idxLevel).vE.at(idxEXT).dArrExt,hLevelEXTv2.at(idxLevel).vE.at(idxEXT).noElem,idxEXT);
	//FUNCHECK(status);
	//if(status!=0){
	//	goto Error;
	//}

	if(hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).noElem>0){
		FUNCHECK(status=Miningv3(hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).noElem,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).dArrUniEdge,hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).hArrSupport,hLevelEXTv2.at(idxLevel).vE.at(idxEXT).dArrExt,hLevelEXTv2.at(idxLevel).vE.at(idxEXT).noElem,idxEXT));
		if(status!=0){
			goto Error;
		}
	}
	//Giải phóng bộ nhớ trên device
	CHECK(cudaStatus = cudaFree(dArrValidScanResult));	//1.
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus = cudaFree(hLevelEXTv2.at(idxLevel).vE.at(idxEXT).dArrExt));//2.
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaFree(hLevelUniEdgev2.at(idxLevel).vUE.at(idxEXT).dArrUniEdge));//3.
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaFree(hLevelUniEdgeSatisfyMinsupv2.at(idxLevel).vecUES.at(idxEXT).dArrUniEdge));//4.
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

Error:
	return cudaStatus;
}


int PMS::computeSupportv3(EXT *dArrExt,int noElemdArrExt,UniEdge *dArrUniEdge,int noElemdArrUniEdge,int firstIdxOfFW,int hFromLi,int &noElem,UniEdge *&dArrUniEdgeSup,int *&hArrSupport){
	int status=0;
	cudaError_t cudaStatus;

#pragma region "find Boundary and scan Boundary"
	int *dArrBoundary=nullptr; //1.cần được giải phóng ở cuối hàm
	int noElemdArrBoundary = noElemdArrExt;
	CHECK(cudaStatus=cudaMalloc((void**)&dArrBoundary,sizeof(int)*noElemdArrBoundary));
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(dArrBoundary,0,sizeof(int)*noElemdArrBoundary));
		if(cudaStatus!=cudaSuccess){
			status =-1;
			goto Error;
		}
	}

	int *dArrBoundaryScanResult=nullptr; //2.cần được giải phóng ở cuối hàm
	CHECK(cudaStatus=cudaMalloc((void**)&dArrBoundaryScanResult,sizeof(int)*noElemdArrBoundary));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(dArrBoundaryScanResult,0,sizeof(int)*noElemdArrBoundary));
		if(cudaStatus!=cudaSuccess){
			status=-1;
			goto Error;
		}
	}

	//Tìm boundary của EXTk và lưu kết quả vào mảng dArrBoundary
	FUNCHECK(status = findBoundary(dArrExt,noElemdArrExt,dArrBoundary));
	if(status!=0){
		goto Error;
	}

	PMS_PRINT("\n ************* dArrBoundary ************\n");
	FUNCHECK(status=displayDeviceArr(dArrBoundary,noElemdArrExt));
	if(status!=0){
		goto Error;
	}

	//Scan dArrBoundary lưu kết quả vào dArrBoundaryScanResult
	//cudaStatus=scanV(dArrBoundary,noElemdArrBoundary,dArrBoundaryScanResult);
	//CHECK(cudaStatus);
	//if(cudaStatus!=cudaSuccess){
	//	status = -1;
	//	fprintf(stderr,"\n Exclusive scan dArrBoundary in computeSupportv2() failed",cudaStatus);
	//	goto Error;
	//}

	CHECK(cudaStatus = myScanV(dArrBoundary,noElemdArrBoundary,dArrBoundaryScanResult));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	PMS_PRINT("\n**************dArrBoundaryScanResult****************\n");
	FUNCHECK(status=displayDeviceArr(dArrBoundaryScanResult,noElemdArrBoundary));
	if(status!=0){
		goto Error;
	}


	//Tính support của cạnh duy nhất.
	int *dF=nullptr; //khai báo mảng dF, 
	int noElemdF = 0; //Số lượng phần tử của mảng dF

	CHECK(cudaStatus = cudaMemcpy(&noElemdF,&dArrBoundaryScanResult[noElemdArrBoundary-1],sizeof(int),cudaMemcpyDeviceToHost));
	if(cudaStatus !=cudaSuccess){
		status =-1;
		goto Error;
	}
	noElemdF++; //Phải tăng lên 1 vì giá trị hiện tại chỉ là chỉ số của mảng
	PMS_PRINT("\n*****noElemdF******\n");
	PMS_PRINT("noElemdF:%d",noElemdF);

	//Cấp phát bộ nhớ trên device cho mảng dF
	CHECK(cudaStatus = cudaMalloc((void**)&dF,sizeof(int)*noElemdF)); //3.Cần được giải phóng ở cuối hàm
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus = cudaMemset(dF,0,sizeof(int)*noElemdF));
		if(cudaStatus!=cudaSuccess){
			status =-1;
			goto Error;
		}
	}
#pragma endregion "end of finding Boundary"

	//Tạm thời chứa độ hỗ trợ của tất cả các cạnh duy nhất.
	//Sau đó, trích những cạnh và độ hỗ trợ thoả minsup vào hLevelUniEdgeSatisfyMinsup tại level tương ứng
	int *hArrSupportTemp = (int*)malloc(sizeof(int)*noElemdArrUniEdge); //4. Cần được giải phóng ở cuối hàm
	if(hArrSupportTemp==NULL){
		status =-1;
		goto Error;
	}
	else
	{
		memset(hArrSupportTemp,0,sizeof(int)*noElemdArrUniEdge);
	}
	////Duyệt và tính độ hỗ trợ của các cạnh. Còn một cách khác là tính độ hỗ trợ cho tất cả các cạnh trong UniEdge cùng một lục
	//dim3 blocke(blocksize);
	//dim3 gride((noElemdArrExt+blocke.x-1)/blocke.x);

	//PMS_PRINT("\n**********dArrUniEdge************");				
	//displaydArrUniEdge(dArrUniEdge,noElemdArrUniEdge);

	for (int i = 0; i < noElemdArrUniEdge; i++)
	{			
		if(i<firstIdxOfFW){
			//Compute support for backward extension in uniedge
			FUNCHECK(status = computeSupportBW(dArrExt,dArrBoundaryScanResult,noElemdArrBoundary,dArrUniEdge,i,dF,noElemdF,hArrSupportTemp[i]));
			if(status!=0){
				goto Error;
			}
		}
		else
		{
			//compute support for forward extension in uniedge
			FUNCHECK(status = computeSupportFW(dArrExt,dArrBoundaryScanResult,noElemdArrBoundary,dArrUniEdge,i,dF,noElemdF,hArrSupportTemp[i]));
			if(status!=0){
				goto Error;
			}
		}
	}
	PMS_PRINT("\n************hArrSupportTemp**********\n");
	for (int j = 0; j < noElemdArrUniEdge; j++)
	{
		PMS_PRINT("j[%d]:%d ",j,hArrSupportTemp[j]);
	}

	//Tiếp theo là lọc giữ lại cạnh và độ hỗ trợ thoả minsup
	//FUNCHECK(status = extractUniEdgeSatisfyMinsupV2(hArrSupportTemp,dArrUniEdge,noElemdArrUniEdge,minsup,noElem,dArrUniEdgeSup,hArrSupport));
	//if(status!=0){
	//	goto Error;
	//}

	FUNCHECK(status = extractUniEdgeSatisfyMinsupV3(hArrSupportTemp,dArrUniEdge,noElemdArrUniEdge,minsup,noElem,dArrUniEdgeSup,hArrSupport));
	if(status!=0){
		goto Error;
	}

	//Giải phóng bộ nhớ
	free(hArrSupportTemp);	//4.

	CHECK(cudaStatus =cudaFree(dArrBoundary)); //1.
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus =cudaFree(dArrBoundaryScanResult));//2.
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus =cudaFree(dF)); //3.
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
Error:	
	return status;
}
//consider 4
int PMS::computeSupportv4(EXT *dArrExt,int noElemdArrExt,UniEdge *dArrUniEdge,int noElemdArrUniEdge, \
						  int firstIdxOfFW,int hFromLi,int &noElem,UniEdge *&dArrUniEdgeSup,int *&hArrSupport)
{
	int status=0;
	cudaError_t cudaStatus;

#pragma region "find Boundary and scan Boundary"
	int *dArrBoundary=nullptr; //1.cần được giải phóng ở cuối hàm
	int noElemdArrBoundary = noElemdArrExt;
	CHECK(cudaStatus=cudaMalloc((void**)&dArrBoundary,sizeof(int)*noElemdArrBoundary));
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(dArrBoundary,0,sizeof(int)*noElemdArrBoundary));
		if(cudaStatus!=cudaSuccess)
		{
			status =-1;
			goto Error;
		}
	}

	int *dArrBoundaryScanResult=nullptr; //2.cần được giải phóng ở cuối hàm
	CHECK(cudaStatus=cudaMalloc((void**)&dArrBoundaryScanResult,sizeof(int)*noElemdArrBoundary));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(dArrBoundaryScanResult,0,sizeof(int)*noElemdArrBoundary));
		if(cudaStatus!=cudaSuccess)
		{
			status=-1;
			goto Error;
		}
	}

	//Tìm boundary của EXTk và lưu kết quả vào mảng dArrBoundary
	FUNCHECK(status = findBoundary(dArrExt,noElemdArrExt,dArrBoundary));
	if(status!=0){
		goto Error;
	}

	/*PMS_PRINT("\n ************* dArrBoundary ************\n");
	FUNCHECK(status=displayDeviceArr(dArrBoundary,noElemdArrExt));
	if(status!=0)
	{
		goto Error;
	}*/

	//Scan dArrBoundary lưu kết quả vào dArrBoundaryScanResult
	//cudaStatus=scanV(dArrBoundary,noElemdArrBoundary,dArrBoundaryScanResult);
	//CHECK(cudaStatus);
	//if(cudaStatus!=cudaSuccess){
	//	status = -1;
	//	fprintf(stderr,"\n Exclusive scan dArrBoundary in computeSupportv2() failed",cudaStatus);
	//	goto Error;
	//}

	CHECK(cudaStatus = myScanV(dArrBoundary,noElemdArrBoundary,dArrBoundaryScanResult));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	/*PMS_PRINT("\n**************dArrBoundaryScanResult****************\n");
	FUNCHECK(status=displayDeviceArr(dArrBoundaryScanResult,noElemdArrBoundary));
	if(status!=0)
	{
		goto Error;
	}*/


	//Tính support của cạnh duy nhất.
	int *dF=nullptr; //khai báo mảng dF, 
	int noElemdF = 0; //Số lượng phần tử của mảng dF

	CHECK(cudaStatus = cudaMemcpy(&noElemdF,&dArrBoundaryScanResult[noElemdArrBoundary-1],sizeof(int),cudaMemcpyDeviceToHost));
	if(cudaStatus !=cudaSuccess)
	{
		status =-1;
		goto Error;
	}
	noElemdF++; //Phải tăng lên 1 vì giá trị hiện tại chỉ là chỉ số của mảng
	/*PMS_PRINT("\n*****noElemdF******\n");
	PMS_PRINT("noElemdF:%d",noElemdF);*/

	//Cấp phát bộ nhớ trên device cho mảng dF
	CHECK(cudaStatus = cudaMalloc((void**)&dF,sizeof(int)*noElemdF)); //3.Cần được giải phóng ở cuối hàm
	if(cudaStatus!=cudaSuccess)
	{
		status =-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus = cudaMemset(dF,0,sizeof(int)*noElemdF));
		if(cudaStatus!=cudaSuccess)
		{
			status =-1;
			goto Error;
		}
	}
#pragma endregion "end of finding Boundary"

	//Tạm thời chứa độ hỗ trợ của tất cả các cạnh duy nhất.
	//Sau đó, trích những cạnh và độ hỗ trợ thoả minsup vào hLevelUniEdgeSatisfyMinsup tại level tương ứng
	int *hArrSupportTemp = (int*)malloc(sizeof(int)*noElemdArrUniEdge); //4. Cần được giải phóng ở cuối hàm
	if(hArrSupportTemp==NULL)
	{
		status =-1;
		goto Error;
	}
	else
	{
		memset(hArrSupportTemp,0,sizeof(int)*noElemdArrUniEdge);
	}
	////Duyệt và tính độ hỗ trợ của các cạnh. Còn một cách khác là tính độ hỗ trợ cho tất cả các cạnh trong UniEdge cùng một lục
	//dim3 blocke(blocksize);
	//dim3 gride((noElemdArrExt+blocke.x-1)/blocke.x);

	//PMS_PRINT("\n**********dArrUniEdge************");				
	//displaydArrUniEdge(dArrUniEdge,noElemdArrUniEdge);
	if(firstIdxOfFW>0)
	{
		for (int i = 0; i < noElemdArrUniEdge; i++) //Nếu ở đây chỉ có backward thì phải làm sao?
		{											//Nếu chỉ có backward thì firstIdxOfFW = -1;
			//Phát biểu: Nếu firstIdxOfFW >=0 thì khai thác forward, ngược lại thì khai thác backward.
			if(i<firstIdxOfFW)
			{
				//Compute support for backward extension in uniedge
				FUNCHECK(status = computeSupportBW(dArrExt,dArrBoundaryScanResult,noElemdArrBoundary, \
					dArrUniEdge,i,dF,noElemdF,hArrSupportTemp[i]));//consider 5: có embedding mà support =0 là sai rồi:(
				if(status!=0)
				{
					goto Error;
				}
			}
			else
			{
				//compute support for forward extension in uniedge
				FUNCHECK(status = computeSupportFW(dArrExt,dArrBoundaryScanResult,noElemdArrBoundary,dArrUniEdge,i, \
					dF,noElemdF,hArrSupportTemp[i]));
				if(status!=0)
				{
					goto Error;
				}
			}
		}
	}
	else if(firstIdxOfFW==0)
	{ //chỉ có forward
		for(int i=0;i<noElemdArrUniEdge;i++)
		{
			FUNCHECK(status = computeSupportFW(dArrExt,dArrBoundaryScanResult,noElemdArrBoundary, \
				dArrUniEdge,i,dF,noElemdF,hArrSupportTemp[i]));
			if(status!=0)
			{
				goto Error;
			}
		}
	}
	else
	{
		for(int i=0;i<noElemdArrUniEdge;i++)
		{ //chỉ có backward
			FUNCHECK(status = computeSupportBW(dArrExt,dArrBoundaryScanResult,noElemdArrBoundary,dArrUniEdge,i, \
				dF,noElemdF,hArrSupportTemp[i]));
			if(status!=0)
			{
				goto Error;
			}
		}
	}

	/*PMS_PRINT("\n************hArrSupportTemp**********\n");
	for (int j = 0; j < noElemdArrUniEdge; j++)
	{
		PMS_PRINT("j[%d]:%d ",j,hArrSupportTemp[j]);
	}*/

	//Tiếp theo là lọc giữ lại cạnh và độ hỗ trợ thoả minsup
	//FUNCHECK(status = extractUniEdgeSatisfyMinsupV2(hArrSupportTemp,dArrUniEdge,noElemdArrUniEdge,minsup,noElem,dArrUniEdgeSup,hArrSupport));
	//if(status!=0){
	//	goto Error;
	//}

	FUNCHECK(status = extractUniEdgeSatisfyMinsupV3(hArrSupportTemp,dArrUniEdge,noElemdArrUniEdge, \
		minsup,noElem,dArrUniEdgeSup,hArrSupport));
	if(status!=0)
	{
		goto Error;
	}

	//Giải phóng bộ nhớ
	free(hArrSupportTemp);	//4.

	CHECK(cudaStatus =cudaFree(dArrBoundary)); //1.
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus =cudaFree(dArrBoundaryScanResult));//2.
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus =cudaFree(dF)); //3.
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
Error:	
	return status;
}


int PMS::computeSupportBW(EXT* dArrExt,int* dArrBoundaryScanResult,int noElemdArrExt,UniEdge* dArrUniEdge,int pos,int* dF,int noElemdF,int& supportOutPut)
{
	int status=0;
	cudaError_t cudaStatus;
	int hSupport=0;
	dim3 block(blocksize);
	dim3 grid((noElemdArrExt + block.x - 1)/block.x);
	kernelFilldFbw<<<grid,block>>>(dArrUniEdge,pos,dArrExt,noElemdArrExt,dArrBoundaryScanResult,dF);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus !=cudaSuccess)
	{
		status =-1;
		goto Error;
	}
	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus !=cudaSuccess)
	{
		status =-1;
		goto Error;
	}				

	/*PMS_PRINT("\n**********dF****************\n");
	displayDeviceArr(dF,noElemdF);*/

	/*CHECK(cudaStatus = reduction(dF,noElemdF,support));
	if(cudaStatus!=cudaSuccess)
	{
	status=-1;
	goto Error;
	}*/
	myReduce(dF,noElemdF,hSupport);

	/*PMS_PRINT("\n******support********");
	PMS_PRINT("\n Support:%f",hSupport);*/

	CHECK(cudaStatus = cudaMemset(dF,0,noElemdF*sizeof(int)));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	supportOutPut=hSupport;
Error:
	return status;
}
//consider 6
int PMS::computeSupportFW(EXT* dArrExt,int* dArrBoundaryScanResult,int noElemdArrExt,UniEdge* dArrUniEdge,int pos,int* dF,int noElemdF,int& supportOutPut)
{
	int status=0;
	cudaError_t cudaStatus;
	int hSupport=0;
	dim3 block(blocksize);
	dim3 grid((noElemdArrExt + block.x - 1)/block.x);
	kernelFilldF<<<grid,block>>>(dArrUniEdge,pos,dArrExt,noElemdArrExt,dArrBoundaryScanResult,dF);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus !=cudaSuccess)
	{
		status =-1;
		goto Error;
	}
	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus !=cudaSuccess)
	{
		status =-1;
		goto Error;
	}				

	/*PMS_PRINT("\n**********dF****************\n");
	FUNCHECK(status=displayDeviceArr(dF,noElemdF));
	if(status!=0)
	{
		goto Error;
	}*/


	/*CHECK(cudaStatus = reduction(dF,noElemdF,support));
	if(cudaStatus!=cudaSuccess)
	{
	status=-1;
	goto Error;
	}*/

	myReduce(dF,noElemdF,hSupport);


	/*PMS_PRINT("\n******support********");
	PMS_PRINT("\n Support:%d",hSupport);*/

	CHECK(cudaStatus = cudaMemset(dF,0,noElemdF*sizeof(int)));
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}

	supportOutPut=hSupport;

Error:
	return status;
}



int PMS::computeSupportv2(EXT *dArrExt,int noElemdArrExt,UniEdge *dArrUniEdge,int noElemdArrUniEdge,int &noElem,UniEdge *&dArrUniEdgeSup,int *&hArrSupport){
	int status=0;
	cudaError_t cudaStatus;

#pragma region "find Boundary and scan Boundary"
	int *dArrBoundary=nullptr;
	int noElemdArrBoundary = noElemdArrExt;
	CHECK(cudaStatus=cudaMalloc((void**)&dArrBoundary,sizeof(int)*noElemdArrBoundary));
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(dArrBoundary,0,sizeof(int)*noElemdArrBoundary));
		if(cudaStatus!=cudaSuccess){
			status =-1;
			goto Error;
		}
	}

	int *dArrBoundaryScanResult=nullptr;
	CHECK(cudaStatus=cudaMalloc((void**)&dArrBoundaryScanResult,sizeof(int)*noElemdArrBoundary));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(dArrBoundaryScanResult,0,sizeof(int)*noElemdArrBoundary));
		if(cudaStatus!=cudaSuccess){
			status=-1;
			goto Error;
		}
	}

	//Tìm boundary của EXTk và lưu kết quả vào mảng dArrBoundary
	FUNCHECK(status = findBoundary(dArrExt,noElemdArrExt,dArrBoundary));
	if(status!=0){
		goto Error;
	}

	PMS_PRINT("\n ************* dArrBoundary ************\n");
	FUNCHECK(status=displayDeviceArr(dArrBoundary,noElemdArrExt));
	if(status!=0){
		goto Error;
	}


	//Scan dArrBoundary lưu kết quả vào dArrBoundaryScanResult
	//cudaStatus=scanV(dArrBoundary,noElemdArrBoundary,dArrBoundaryScanResult);
	//CHECK(cudaStatus);
	//if(cudaStatus!=cudaSuccess){
	//	status = -1;
	//	fprintf(stderr,"\n Exclusive scan dArrBoundary in computeSupportv2() failed",cudaStatus);
	//	goto Error;
	//}

	CHECK(cudaStatus = myScanV(dArrBoundary,noElemdArrBoundary,dArrBoundaryScanResult));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	PMS_PRINT("\n**************dArrBoundaryScanResult****************\n");
	FUNCHECK(status=displayDeviceArr(dArrBoundaryScanResult,noElemdArrBoundary));
	if(status!=0){
		goto Error;
	}


	//Tính support của cạnh duy nhất.
	int *dF=nullptr; //khai báo mảng dF
	int noElemdF = 0; //Số lượng phần tử của mảng dF

	CHECK(cudaStatus = cudaMemcpy(&noElemdF,&dArrBoundaryScanResult[noElemdArrBoundary-1],sizeof(int),cudaMemcpyDeviceToHost));
	if(cudaStatus !=cudaSuccess){
		status =-1;
		goto Error;
	}
	noElemdF++; //Phải tăng lên 1 vì giá trị hiện tại chỉ là chỉ số của mảng
	PMS_PRINT("\n*****noElemdF******\n");
	PMS_PRINT("noElemdF:%d",noElemdF);

	//Cấp phát bộ nhớ trên device cho mảng dF
	CHECK(cudaStatus = cudaMalloc((void**)&dF,sizeof(int)*noElemdF));
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus = cudaMemset(dF,0,sizeof(int)*noElemdF));
		if(cudaStatus!=cudaSuccess){
			status =-1;
			goto Error;
		}
	}
#pragma endregion "end of finding Boundary"

	//Tạm thời chứa độ hỗ trợ của tất cả các cạnh duy nhất.
	//Sau đó, trích những cạnh và độ hỗ trợ thoả minsup vào hLevelUniEdgeSatisfyMinsup tại level tương ứng
	int *hArrSupportTemp = (int*)malloc(sizeof(int)*noElemdArrUniEdge);
	if(hArrSupportTemp==NULL){
		status =-1;
		goto Error;
	}
	else
	{
		memset(hArrSupportTemp,0,sizeof(unsigned int)*noElemdArrUniEdge);
	}
	//		//Duyệt và tính độ hỗ trợ của các cạnh. Còn một cách khác là tính độ hỗ trợ cho tất cả các cạnh trong UniEdge cùng một lục
	dim3 blocke(blocksize);
	dim3 gride((noElemdArrExt+blocke.x-1)/blocke.x);

	//PMS_PRINT("\n**********dArrUniEdge************");				
	//displaydArrUniEdge(dArrUniEdge,noElemdArrUniEdge);
	//Duyệt qua các cạnh duy nhất và tính độ hỗ trợ
	for (int i = 0; i < noElemdArrUniEdge; i++)
	{					
		float support=0;
		kernelFilldF<<<gride,blocke>>>(dArrUniEdge,i,dArrExt,noElemdArrExt,dArrBoundaryScanResult,dF);

		CHECK(cudaStatus=cudaDeviceSynchronize());
		if(cudaStatus!=cudaSuccess){
			status=-1;
			goto Error;
		}

		CHECK(cudaStatus = cudaGetLastError());
		if(cudaStatus !=cudaSuccess){
			status =-1;
			goto Error;
		}				

		PMS_PRINT("\n**********dF****************\n");
		displayDeviceArr(dF,noElemdF);

		/*CHECK(cudaStatus = reduction(dF,noElemdF,support));
		if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
		}*/

		PMS_PRINT("\n******support********");
		PMS_PRINT("\n Support:%f",support);

		CHECK(cudaStatus = cudaMemset(dF,0,noElemdF*sizeof(float)));
		if(cudaStatus!=cudaSuccess){
			status=-1;
			goto Error;
		}

		hArrSupportTemp[i]=support;
	}
	PMS_PRINT("\n************hArrSupportTemp**********\n");
	for (int j = 0; j < noElemdArrUniEdge; j++)
	{
		PMS_PRINT("j[%d]:%d ",j,hArrSupportTemp[j]);
	}

	//Tiếp theo là lọc giữ lại cạnh và độ hỗ trợ thoả minsup
	FUNCHECK(status = extractUniEdgeSatisfyMinsupV2(hArrSupportTemp,dArrUniEdge,noElemdArrUniEdge,minsup,noElem,dArrUniEdgeSup,hArrSupport));
	if(status!=0){
		goto Error;
	}

	free(hArrSupportTemp);	

	CHECK(cudaStatus =cudaFree(dArrBoundary));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus =cudaFree(dArrBoundaryScanResult));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

Error:	
	return status;
}

int PMS::extractUniEdgeSatisfyMinsupV2(int *hResultSup,UniEdge *dArrUniEdge,int noElemUniEdge,unsigned int minsup,int &noElem,UniEdge *&dArrUniEdgeSup,int *&hArrSupport){
	int status=0;
	cudaError_t cudaStatus;
	//1. Cấp phát mảng trên device có kích thước bằng noElemUniEdge
	int *dResultSup=nullptr;
	CHECK(cudaStatus =cudaMalloc((void**)&dResultSup,noElemUniEdge*sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	CHECK(cudaStatus = cudaMemcpy(dResultSup,hResultSup,noElemUniEdge*sizeof(int),cudaMemcpyHostToDevice));
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}

	PMS_PRINT("\n *******dResultSup********\n");
	FUNCHECK(status = displayDeviceArr(dResultSup,noElemUniEdge));
	if(status != 0){
		goto Error;
	}


	//2. Đánh dấu 1 trên dV cho những phần tử thoả minsup
	int *dV=nullptr;
	CHECK(cudaStatus =cudaMalloc((void**)&dV,noElemUniEdge*sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	CHECK(cudaStatus =cudaMemset(dV,0,sizeof(int)*noElemUniEdge));
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}

	dim3 block(blocksize);
	dim3 grid((noElemUniEdge + block.x - 1)/block.x);
	kernelMarkUniEdgeSatisfyMinsup<<<grid,block>>>(dResultSup,noElemUniEdge,dV,minsup);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus=cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n kernelMarkUniEdgeSatisfyMinsup in extractUniEdgeSatisfyMinsup() failed",cudaStatus);
		status = -1;
		goto Error;
	}

	/*PMS_PRINT("\n ***********dV**********\n");
	FUNCHECK(status = displayDeviceArr(dV,noElemUniEdge));
	if(status!=0){
		goto Error;
	}*/


	int *dVScanResult=nullptr;
	CHECK(cudaStatus=cudaMalloc((void**)&dVScanResult,noElemUniEdge*sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	//CHECK(cudaStatus = scanV(dV,noElemUniEdge,dVScanResult));
	//if(cudaStatus!=cudaSuccess){
	//	status=-1;
	//	goto Error;
	//}

	CHECK(cudaStatus = myScanV(dV,noElemUniEdge,dVScanResult));
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}


	/*PMS_PRINT("\n ***********dVScanResult**********\n");
	FUNCHECK(status=displayDeviceArr(dVScanResult,noElemUniEdge));
	if(status!=0){
		goto Error;
	}*/


	CHECK(cudaStatus=getSizeBaseOnScanResult(dV,dVScanResult,noElemUniEdge,noElem));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus=cudaMalloc((void**)&dArrUniEdgeSup,noElem*sizeof(UniEdge)));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	hArrSupport = (int*)malloc(sizeof(int)*noElem);
	if (hArrSupport ==NULL){
		status =-1;
		PMS_PRINT("\n malloc hArrSup of hUniEdgeSatisfyMinsup failed()");
		goto Error;
	}


	int *dSup=nullptr;
	CHECK(cudaStatus=cudaMalloc((void**)&dSup,noElem*sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	dim3 blocka(blocksize);
	dim3 grida((noElemUniEdge + blocka.x -1)/blocka.x);
	kernelExtractUniEdgeSatifyMinsup<<<grida,blocka>>>(dArrUniEdge,dV,dVScanResult,noElemUniEdge,dArrUniEdgeSup,dSup,dResultSup);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		status = -1;
		fprintf(stderr,"\n kernelExtractUniEdgeSatisfyMinsup() in extractUniEdgeSatisfyMinsup() failed",cudaStatus);
		goto Error;
	}
	PMS_PRINT("\n ********hUniEdgeSatisfyMinsup.dUniEdge****************\n");
	FUNCHECK(status=displayArrUniEdge(dArrUniEdgeSup,noElem));
	if(status!=0){
		goto Error;
	}

	PMS_PRINT("\n ********hUniEdgeSatisfyMinsup.dSup****************\n");
	FUNCHECK(status=displayDeviceArr(dSup,noElem));
	if(status!=0){
		goto Error;
	}


	CHECK(cudaStatus = cudaMemcpy(hArrSupport,dSup,sizeof(int)*noElem,cudaMemcpyDeviceToHost));
	if(cudaStatus!=cudaSuccess){
		status = -1;
		fprintf(stderr,"\n kernelExtractUniEdgeSatisfyMinsup() in extractUniEdgeSatisfyMinsup() failed",cudaStatus);
		goto Error;
	}

	for (int i = 0; i < noElem; i++)
	{
		PMS_PRINT("\n hArrSupport:%d ",hArrSupport[i]);
	}

	CHECK(cudaStatus = cudaFree(dResultSup));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaFree(dV));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus = cudaFree(dVScanResult));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus = cudaFree(dSup));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

Error:
	return status;
}

int PMS::extractUniEdgeSatisfyMinsupV3(int *hResultSup,UniEdge *dArrUniEdge,int noElemUniEdge,unsigned int minsup,int &noElemUniEdgeSatisfyMinSup,UniEdge *&dArrUniEdgeSup,int *&hArrSupport){
	int status=0;
	cudaError_t cudaStatus;
	//1. Cấp phát mảng trên device có kích thước bằng noElemUniEdge
	int *dResultSup=nullptr; //cần được giải phóng ở cuối hàm
	CHECK(cudaStatus =cudaMalloc((void**)&dResultSup,noElemUniEdge*sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	//Chép độ hỗ trợ từ host qua device để lọc song song 
	CHECK(cudaStatus = cudaMemcpy(dResultSup,hResultSup,noElemUniEdge*sizeof(int),cudaMemcpyHostToDevice));
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}

	/*PMS_PRINT("\n *******dResultSup********\n");
	FUNCHECK(status = displayDeviceArr(dResultSup,noElemUniEdge));
	if(status != 0){
		goto Error;
	}*/


	//2. Đánh dấu 1 trên dV cho những phần tử thoả minsup
	int *dV=nullptr; //cần được giải phóng ở cuối hàm
	CHECK(cudaStatus =cudaMalloc((void**)&dV,noElemUniEdge*sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	CHECK(cudaStatus =cudaMemset(dV,0,sizeof(int)*noElemUniEdge));
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}

	dim3 block(blocksize);
	dim3 grid((noElemUniEdge + block.x - 1)/block.x);
	kernelMarkUniEdgeSatisfyMinsup<<<grid,block>>>(dResultSup,noElemUniEdge,dV,minsup);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus=cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}

	/*PMS_PRINT("\n ***********dV**********\n");
	FUNCHECK(status = displayDeviceArr(dV,noElemUniEdge));
	if(status!=0){
		goto Error;
	}*/


	int *dVScanResult=nullptr; //cần được giải phóng ở cuối hàm
	CHECK(cudaStatus=cudaMalloc((void**)&dVScanResult,noElemUniEdge*sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	//CHECK(cudaStatus = scanV(dV,noElemUniEdge,dVScanResult));
	//if(cudaStatus!=cudaSuccess){
	//	status=-1;
	//	goto Error;
	//}

	CHECK(cudaStatus = myScanV(dV,noElemUniEdge,dVScanResult));
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}


	/*PMS_PRINT("\n ***********dVScanResult**********\n");
	FUNCHECK(status=displayDeviceArr(dVScanResult,noElemUniEdge));
	if(status!=0){
		goto Error;
	}*/


	CHECK(cudaStatus=getSizeBaseOnScanResult(dV,dVScanResult,noElemUniEdge,noElemUniEdgeSatisfyMinSup));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	//Nếu không có phần tử nào thoả minsup thì không khai thác nữa
	if(noElemUniEdgeSatisfyMinSup==0){ 
		CHECK(cudaStatus = cudaFree(dResultSup));
		if(cudaStatus!=cudaSuccess){
			status = -1;
			goto Error;
		}
		CHECK(cudaStatus = cudaFree(dV));
		if(cudaStatus!=cudaSuccess){
			status = -1;
			goto Error;
		}
		CHECK(cudaStatus =cudaFree(dVScanResult));
		if(cudaStatus!=cudaSuccess){
			status = -1;
			goto Error;
		}
		goto Error;
	}

	CHECK(cudaStatus=cudaMalloc((void**)&dArrUniEdgeSup,noElemUniEdgeSatisfyMinSup*sizeof(UniEdge)));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	hArrSupport = (int*)malloc(sizeof(int)*noElemUniEdgeSatisfyMinSup);
	if (hArrSupport ==NULL){
		status =-1;
		goto Error;
	}


	int *dSup=nullptr; //cần được giải phóng ở cuối hàm
	CHECK(cudaStatus=cudaMalloc((void**)&dSup,noElemUniEdgeSatisfyMinSup*sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	dim3 blocka(blocksize);
	dim3 grida((noElemUniEdge + blocka.x -1)/blocka.x);
	//kernelExtractUniEdgeSatifyMinsup<<<grida,blocka>>>(dArrUniEdge,dV,dVScanResult,noElemUniEdge,dArrUniEdgeSup,dSup,dResultSup);
	kernelExtractUniEdgeSatifyMinsupV3<<<grida,blocka>>>(dArrUniEdge,dV,dVScanResult,noElemUniEdge,dArrUniEdgeSup,dSup,dResultSup);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(status!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	/*PMS_PRINT("\n ********hUniEdgeSatisfyMinsup.dUniEdge****************\n");
	FUNCHECK(status=displayArrUniEdge(dArrUniEdgeSup,noElemUniEdgeSatisfyMinSup));
	if(status!=0){
		goto Error;
	}

	PMS_PRINT("\n ********hUniEdgeSatisfyMinsup.dSup****************\n");
	FUNCHECK(status=displayDeviceArr(dSup,noElemUniEdgeSatisfyMinSup));
	if(status!=0){
		goto Error;
	}*/

	CHECK(cudaStatus = cudaMemcpy(hArrSupport,dSup,sizeof(int)*noElemUniEdgeSatisfyMinSup,cudaMemcpyDeviceToHost));
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}

	/*for (int i = 0; i < noElemUniEdgeSatisfyMinSup; i++)
	{
		PMS_PRINT("\n hArrSupport:%d ",hArrSupport[i]);
	}*/

	CHECK(cudaStatus = cudaFree(dResultSup));
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	CHECK(cudaStatus = cudaFree(dV));
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	CHECK(cudaStatus =cudaFree(dVScanResult));
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	CHECK(cudaStatus =cudaFree(dSup));
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
Error:
	return status;
}


__global__ void printdArrUniEdge(UniEdge *dArrUniEdge,int pos){
	PMS_PRINT("\n d[%d]: (li,lij,lj):(%d %d %d)",pos,dArrUniEdge[pos].li,dArrUniEdge[pos].lij,dArrUniEdge[pos].lj);
}


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

__global__ void kernelFilldFbw(UniEdge *dArrUniEdge,int pos,EXT *dArrExt,int noElemdArrExt,int *dArrBoundaryScanResult,int *dF)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i<noElemdArrExt){
		if(dArrUniEdge[pos].vj==dArrExt[i].vj)
		{
			dF[dArrBoundaryScanResult[i]]=1;
		}
	}
}


int PMS::findBoundary(EXT *dArrExt,int noElemdArrExt,int *&dArrBoundary){
	int status =0;
	cudaError_t cudaStatus;
	dim3 block(blocksize);
	dim3 grid((noElemdArrExt+block.x-1)/block.x);

	kernelfindBoundary<<<grid,block>>>(dArrExt,noElemdArrExt,dArrBoundary,maxOfVer);

	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		status = -1;
		fprintf(stderr,"\n kernelfindBoundary in findBoundary() failed",cudaStatus);
		goto Error;
	}
Error:	
	return status;
}


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

//Input: Danh sách các đỉnh của embedding column cần mở rộng, vị trí EXT<k>, số lượng embedding và phát triển từ đỉnh nào của RMP
int PMS::forwardExtension(int idxhEXTk,int *listOfVer,int noElemEmbedding,int fromRMP){
	int status = 0;
	cudaError_t cudaStatus;
	//int lastCol = hEmbedding.size() - 1;
	int lastCol =1; //ở đây chúng ta biết cột cuối của embedding là 1

	dim3 block(blocksize);
	dim3 grid((noElemEmbedding + block.x -1)/block.x);

	//Tìm bậc lớn nhất của các đỉnh cần mở rộng trong listOfVer
	int maxDegreeOfVer=0;
	float *dArrDegreeOfVid=nullptr; //chứa bậc của các đỉnh trong listOfVer, dùng để duyệt qua các đỉnh lân cận trong csdl
	FUNCHECK(status=findMaxDegreeOfVer(listOfVer,maxDegreeOfVer,dArrDegreeOfVid,noElemEmbedding)); //tìm bậc lớn nhất
	if(status!=0){
		PMS_PRINT("\n findMaxDegreeOfVer() failed");
		goto Error;
	}
	//Tạo mảng dArrV có số lượng phần tử bằng số lượng embedding nhân với bậc lớn nhất của các vid vừa tìm được
	//Tạo mảng d_arr_V có kích thước: maxDegree_vid_Q * |Q|
	//	Lưu ý, mảng d_arr_V phải có dạng cấu trúc đủ thể hiện cạnh mở rộng có hợp lệ hay không và là forward extension hay backward extension.
	//	struct struct_V
	//	{
	//		int valid; //default: 0, valid: 1
	//		int backward; //default: 0- forward; backward: 1
	//	}

	V *dArrV=nullptr;
	dArrV = (V*)malloc(sizeof(V));
	if(dArrV==NULL){
		PMS_PRINT("\n Malloc dArrV failed\n");
		status=-1;
		goto Error;
	}
	dArrV->noElem =maxDegreeOfVer*noElemEmbedding;
	CHECK(cudaStatus=cudaMalloc((void**)&(dArrV->valid),(dArrV->noElem)*sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMalloc dArrV in  failed");
		status =-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus = cudaMemset(dArrV->valid,0,(dArrV->noElem)*sizeof(int)));
		if(cudaStatus !=cudaSuccess){
			status =-1;
			goto Error;
		}
	}
	CHECK(cudaStatus=cudaMalloc((void**)&(dArrV->backward),(dArrV->noElem)*sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMalloc dArrV in  failed");
		status =-1;
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(dArrV->backward,0,(dArrV->noElem)*sizeof(int)));
		if(cudaStatus !=cudaSuccess){
			status =-1;
			goto Error;
		}
	}


	////Các mở rộng hợp lệ sẽ được ghi nhận vào mảng dArrV, đồng thời thông tin của cạnh mở rộng gồm dfscode, vgi, vgj và row pointer của nó cũng được xây dựng
	////và lưu trữ trong mảng EXT *dExtensionTemp, mảng này có số lượng phần tử bằng với mảng dArrV. Sau đó chúng ta sẽ rút trích những mở rộng hợp lệ này và lưu vào dExt. 
	////Để xây dựng dfscode (vi,vj,li,lij,lj) thì chúng ta cần:
	//// - Dựa vào giá trị của right most path để xác định vi
	//// - Dựa vào maxid để xác định vj
	//// - Dựa vào CSDL để xác định các thành phần còn lại.
	////Chúng ta có thể giải phóng bộ nhớ của dExtensionTemp sau khi đã trích các mở rộng hợp lệ thành công.


	EXT *dArrExtensionTemp= nullptr; //Nơi lưu trữ tạm thời tất cả các cạnh mở rộng. Sau đó, chúng sẽ được lọc ra các mở rộng hợp lệ sang EXTk tương ứng.
	CHECK(cudaStatus = cudaMalloc((void**)&dArrExtensionTemp,(dArrV->noElem)*sizeof(EXT)));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		fprintf(stderr,"\n cudaMalloc dArrExtensionTemp forwardExtensionQ() failed",cudaStatus);
		goto Error;
	}
	else
	{
		CHECK(cudaStatus=cudaMemset(dArrExtensionTemp,0,dArrV->noElem*sizeof(EXT)));
		if(cudaStatus !=cudaSuccess){
			status =-1;
			goto Error;
		}
	}

	PMS_PRINT("\nnoElem_dArrV:%d",dArrV->noElem );


	////Gọi kernel với các đối số: CSDL, bậc của các đỉnh, dArrV, dArrExtension,noElem_Embedding,maxDegreeOfVer,idxQ,dArrPointerEmbedding,minLabel,maxid
	dim3 blocka(blocksize);
	dim3 grida((noElemEmbedding+block.x - 1)/blocka.x);
	//hdb.at(0).dN;
	//int noElemdArrPointerEmbedding = lastCol+1;

	//kernel tìm các mở rộng forward hợp lệ	
	kernelFindValidForwardExtension<<<grida,blocka>>>(hLevelPtrEmbedding.at(idxLevel).dArrPointerEmbedding,hLevelPtrEmbedding.at(idxLevel).noElem,hLevelPtrEmbedding.at(idxLevel).noElemEmbedding,hdb.at(0).dO,hdb.at(0).dLO,hdb.at(0).dN,hdb.at(0).dLN,dArrDegreeOfVid,maxDegreeOfVer,dArrExtensionTemp,listOfVer,minLabel,maxId,fromRMP,dArrV->valid,dArrV->backward);
	CHECK(cudaStatus=cudaDeviceSynchronize());
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus=cudaGetLastError());
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() kernelFindValidForwardExtension in forwardExtensionQ() failed",cudaStatus);
		status=-1;
		goto Error;
	}
	//In mảng dArrV để kiểm tra thử
	PMS_PRINT("\n****************dArrV_valid*******************\n");
	FUNCHECK(status=displayDeviceArr(dArrV->valid,dArrV->noElem));
	if(status!=0){
		goto Error;
	}

	////Chép kết quả từ dArrExtension sang dExt
	//chúng ta cần có mảng dArrV để trích các mở rộng duy nhất
	//Hàm này cũng gọi hàm trích các mở rộng duy nhất và tính độ hỗ trợ của chúng
	FUNCHECK(status=displayDeviceEXT(dArrExtensionTemp,dArrV->noElem));
	if(status!=0){
		goto Error;
	}

	FUNCHECK(status = extractValidExtensionTodExt(dArrExtensionTemp,dArrV,dArrV->noElem,idxhEXTk));
	if(status!=0){
		fprintf(stderr,"\n extractValidExtensionTodExt() in forwardExtensionQ() failed");
		goto Error;
	}

	CHECK(cudaStatus=cudaFree(dArrV->valid));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	CHECK(cudaStatus=cudaFree(dArrV->backward));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	free(dArrV);

	CHECK(cudaStatus=cudaFree(dArrDegreeOfVid));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	if(dArrExtensionTemp!=nullptr){
		CHECK(cudaStatus=cudaFree(dArrExtensionTemp));
		if(cudaStatus!=cudaSuccess){
			status=-1;
			goto Error;
		}
	}


Error:
	return status;
}



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

int PMS::findMaxDegreeOfVerEmbeddingColumn(int &idx_EC,int &maxDegreeOfVer,float *&dArrDegreeOfVid)
{
	int status = 0;
	int noElem = 0;
	FUNCHECK(status = findDegreeOfVerEmbeddingColumn(idx_EC,dArrDegreeOfVid,noElem));
	if(status == -1){
		PMS_PRINT("\n findDegreeOfVer() in findMaxDegreeOfVer() faild");
		goto Error;
	}

	/*PMS_PRINT("\n*******dArrDegreeOfVid*************\n");
	displayDeviceArr(dArrDegreeOfVid,noElem);*/

	//Tìm bậc lớn nhất và lưu kết quả vào biến maxDegreeOfVer
	float *h_max;
	h_max = (float*)malloc(sizeof(float));
	if(h_max==NULL){
		PMS_PRINT("\nMalloc h_max failed");
		status = -1;
		FUNCHECK(status);
		goto Error;
	}

	float *d_max;
	int *d_mutex;
	cudaError_t cudaStatus=cudaMalloc((void**)&d_max,sizeof(float));
	CHECK(cudaStatus);
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaMalloc d_max failed",cudaStatus);
		status =-1;
		goto Error;
	}
	else
	{
		CHECK(cudaMemset(d_max,0,sizeof(float)));
	}

	cudaStatus=cudaMalloc((void**)&d_mutex,sizeof(int));
	CHECK(cudaStatus);
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaMalloc d_mutex failed");
		status = -1;
		goto Error;
	}
	else
	{
		cudaMemset(d_mutex,0,sizeof(int));
	}

	dim3 gridSize = 256;
	dim3 blockSize = 256;
	find_maximum_kernel<<<gridSize, blockSize>>>(dArrDegreeOfVid, d_max, d_mutex, noElem);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	CHECK(cudaStatus);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n find_maximum_kernel in findMaxDegreeOfVer() failed");
		status =-1;
		goto Error;
	}

	// copy from device to host
	CHECK(cudaMemcpy(h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost));

	//report results
	maxDegreeOfVer = (int)(*h_max); //bậc lớn nhất của các đỉnh trong 1 cột Q
	//PMS_PRINT("\nMax degree of vid in Q column is: %d",maxDegreeOfVer);

	cudaFree(d_max);
	cudaFree(d_mutex);
	free(h_max);
Error:
	return status;
}

//Tìm bậc lớn nhất của các đỉnh cần mở rộng thuộc một Embedding Column.
int PMS::findMaxDegreeOfVer(int *&listOfVer,int &maxDegreeOfVer,float *&dArrDegreeOfVid,int &noElem)
{
	int status = 0;
	FUNCHECK(status = findDegreeOfVer(listOfVer,dArrDegreeOfVid,noElem));
	if(status == -1){
		PMS_PRINT("\n findDegreeOfVer() in findMaxDegreeOfVer() faild");
		goto Error;
	}

	/*PMS_PRINT("\n*******dArrDegreeOfVid*************\n");
	displayDeviceArr(dArrDegreeOfVid,noElem);*/

	//Tìm bậc lớn nhất và lưu kết quả vào biến maxDegreeOfVer
	float *h_max;
	h_max = (float*)malloc(sizeof(float));
	if(h_max==NULL){
		PMS_PRINT("\nMalloc h_max failed");
		status = -1;
		FUNCHECK(status);
		goto Error;
	}

	float *d_max;
	int *d_mutex;
	cudaError_t cudaStatus=cudaMalloc((void**)&d_max,sizeof(float));
	CHECK(cudaStatus);
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaMalloc d_max failed",cudaStatus);
		status =-1;
		goto Error;
	}
	else
	{
		CHECK(cudaMemset(d_max,0,sizeof(float)));
	}

	cudaStatus=cudaMalloc((void**)&d_mutex,sizeof(int));
	CHECK(cudaStatus);
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaMalloc d_mutex failed");
		status = -1;
		goto Error;
	}
	else
	{
		cudaMemset(d_mutex,0,sizeof(int));
	}

	dim3 gridSize = 256;
	dim3 blockSize = 256;
	find_maximum_kernel<<<gridSize, blockSize>>>(dArrDegreeOfVid, d_max, d_mutex, noElem);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	CHECK(cudaStatus);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n find_maximum_kernel in findMaxDegreeOfVer() failed");
		status =-1;
		goto Error;
	}

	// copy from device to host
	CHECK(cudaMemcpy(h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost));

	//report results
	maxDegreeOfVer = (int)(*h_max); //bậc lớn nhất của các đỉnh trong 1 cột Q
	//PMS_PRINT("\nMax degree of vid in Q column is: %d",maxDegreeOfVer);

	cudaFree(d_max);
	cudaFree(d_mutex);
	free(h_max);
Error:
	return status;
}
//__global__ void kernelCalDegreeOfVid(Embedding *dArrEmbedding,int *d_O, int numberOfElementd_O,int noElem_Embedding,int numberOfElementd_N,unsigned int maxOfVer,float *dArrDegreeOfVid){
//	int i = blockDim.x * blockIdx.x + threadIdx.x;
//	if(i<noElem_Embedding){
//		int vid = dArrEmbedding[i].vid;
//		float degreeOfV =0;
//		int nextVid;
//		int graphid;
//		int lastGraphId=(numberOfElementd_O-1)/maxOfVer;
//		if (vid==numberOfElementd_O-1){ //nếu như đây là đỉnh cuối cùng trong d_O
//			degreeOfV=numberOfElementd_N-d_O[vid]; //thì bậc của đỉnh vid chính bằng tổng số cạnh trừ cho giá trị của d_O[vid].
//		}
//		else
//		{
//			nextVid = vid+1; //xét đỉnh phía sau có khác 1 hay không?
//			graphid=vid/maxOfVer;
//			if(d_O[nextVid]==-1 && graphid==lastGraphId){
//				degreeOfV=numberOfElementd_N-d_O[vid];
//			}
//			else if(d_O[nextVid]==-1 && graphid!=lastGraphId){
//				nextVid=(graphid+1)*maxOfVer;
//				degreeOfV=d_O[nextVid]-d_O[vid];
//			}
//			else
//			{
//				degreeOfV=d_O[nextVid]-d_O[vid];
//			}							
//		}
//		dArrDegreeOfVid[i]=degreeOfV;
//	}
//}

__global__ void kernelCalDegreeOfVid(int *listOfVer, \
									 int *d_O, int numberOfElementd_O,int noElem_Embedding, \
									 int numberOfElementd_N,unsigned int maxOfVer,float *dArrDegreeOfVid)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElem_Embedding)
	{
		int vid = listOfVer[i];
		float degreeOfV =0;
		int nextVid;
		int graphid;
		int lastGraphId=(numberOfElementd_O-1)/maxOfVer;
		if (vid==numberOfElementd_O-1){ //nếu như đây là đỉnh cuối cùng trong d_O
			degreeOfV=numberOfElementd_N-d_O[vid]; //thì bậc của đỉnh vid chính bằng tổng số cạnh trừ cho giá trị của d_O[vid].
		}
		else
		{
			nextVid = vid+1; //xét đỉnh phía sau có khác 1 hay không?
			graphid=vid/maxOfVer;
			if(d_O[nextVid]==-1 && graphid==lastGraphId){
				degreeOfV=numberOfElementd_N-d_O[vid];
			}
			else if(d_O[nextVid]==-1 && graphid!=lastGraphId){
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


__global__ void kernelCalDegreeOfVidOnEmbeddingColumn(Embedding* dArrEmbedding, \
									 int *d_O, int numberOfElementd_O,int noElem_Embedding, \
									 int numberOfElementd_N,unsigned int maxOfVer,float *dArrDegreeOfVid)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElem_Embedding)
	{
		int vid = dArrEmbedding[i].vid;
		float degreeOfV =0;
		int nextVid=-1;
		int graphid=-1;
		int lastGraphId=(numberOfElementd_O-1)/maxOfVer;
		if (vid==numberOfElementd_O-1)
		{ //nếu như đây là đỉnh cuối cùng trong d_O
			degreeOfV=numberOfElementd_N-d_O[vid]; //thì bậc của đỉnh vid chính bằng tổng số cạnh trừ cho giá trị của d_O[vid].
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

__global__ void kernelCalDegreeOfVidOnEmbeddingColumnv2(Embedding** dPointerEmbedding, \
									 int *d_O, int numberOfElementd_O,int noElemRMP, int noElemEmbedding, \
									 int numberOfElementd_N,unsigned int maxOfVer,float *dArrDegreeOfVid)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int noElemVid = noElemRMP * noElemEmbedding;
	if(i<noElemVid)
	{
		int idxLastCol = noElemRMP-1;
		//Bắt đầu từ embedding column cuối cùng.
		//Phần dư của thread i sẽ cho biết Thread đó đang xử lý cho embedding nào trong Embedding column cuối.
		int idxRow = i % noElemEmbedding;
		//Phần nguyên là embedding column index của vid
		int idxCol = i / noElemEmbedding;
		int vid=-1;
		//PMS_PRINT("\n Thread: %d idxCol:%d idxRow:%d idxLastCol:%d ",i,idxCol,idxRow,idxLastCol);
		for(int s = idxLastCol; s>=idxCol; s--)
		{
			Embedding* dArrEmbedding = dPointerEmbedding[s];
			vid = dArrEmbedding[idxRow].vid;
			idxRow = dArrEmbedding[idxRow].idx;
		}

		float degreeOfV =0;
		int nextVid=-1;
		int graphid=-1;
		int lastGraphId=(numberOfElementd_O-1)/maxOfVer;
		if (vid==numberOfElementd_O-1)
		{ //nếu như đây là đỉnh cuối cùng trong d_O
			degreeOfV=numberOfElementd_N-d_O[vid]; //thì bậc của đỉnh vid chính bằng tổng số cạnh trừ cho giá trị của d_O[vid].
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

//Tìm bậc của một list các đỉnh.
int PMS::findDegreeOfVer(int *&listOfVer,float *&dArrDegreeOfVid,int &noElem_Embedding){
	int status = 0;
	cudaError_t cudaStatus;
	Embedding *EC = nullptr; //giả mạo
	CHECK(cudaStatus =cudaMalloc((void**)&dArrDegreeOfVid,noElem_Embedding*sizeof(float)));
	if(cudaStatus !=cudaSuccess){
		status =-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaMemset(dArrDegreeOfVid,0,noElem_Embedding*sizeof(float)));
	if(cudaStatus !=cudaSuccess){
		status =-1;
		goto Error;
	}
	for(int i = 0; i<noElem_Embedding;i++)
		{
			std::printf("i:%d - listofver:%d:",i,listOfVer[i]);
		}

	dim3 block(blocksize);
	dim3 grid((noElem_Embedding + block.x -1)/block.x);
	//Đầu vào của kernelCalDegreeOfVid là một tập đỉnh trên RMP kèm theo Embedding Header của nó.
	kernelCalDegreeOfVidOnEmbeddingColumn<<<grid,block>>>( \
		EC, \
		hdb.at(0).dO, hdb.at(0).noElemdO, \
		noElem_Embedding, \
		hdb.at(0).noElemdN, \
		maxOfVer, \
		dArrDegreeOfVid);
	CHECK(cudaStatus=cudaDeviceSynchronize()); 
	if(cudaStatus!=cudaSuccess)
	{
		status=-1;
		goto Error;
	}
	CHECK(cudaStatus=cudaGetLastError());	
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}

Error:
	return status;
}

//Tìm bậc của vid thuộc embedding column.
int PMS::findDegreeOfVerEmbeddingColumn(int &idx_EC,float *&dArrDegreeOfVid,int &noElem_Embedding)
{
	int status = 0;
	cudaError_t cudaStatus;
	//Nếu tồn tại backward thì lấy embedding column của backward cuối cùng
	if (hEmbedding.at(idx_EC).hBackwardEmbedding.size() != 0)
	{
		//Lấy index của embedding column cuối
		int last_idx = hEmbedding.at(idx_EC).hBackwardEmbedding.size() -1;
		noElem_Embedding=hEmbedding.at(idx_EC).hBackwardEmbedding.at(last_idx).noElem;
		dim3 block(blocksize);
		dim3 grid((noElem_Embedding + block.x -1)/block.x);
		//Đầu vào của kernelCalDegreeOfVid là một tập đỉnh trên RMP kèm theo Embedding Header của nó.
		kernelCalDegreeOfVidOnEmbeddingColumn<<<grid,block>>>( \
			hEmbedding.at(idx_EC).hBackwardEmbedding.at(last_idx).dArrEmbedding, \
			hdb.at(0).dO, hdb.at(0).noElemdO, \
			noElem_Embedding, \
			hdb.at(0).noElemdN, \
			maxOfVer, \
			dArrDegreeOfVid);
		CHECK(cudaStatus=cudaDeviceSynchronize()); 
		if(cudaStatus!=cudaSuccess)
		{
			status=-1;
			goto Error;
		}
		CHECK(cudaStatus=cudaGetLastError());	
		if(cudaStatus!=cudaSuccess){
			status =-1;
			goto Error;
		}
	}
	else
	{
		noElem_Embedding=hEmbedding.at(idx_EC).noElem;
		dim3 block(blocksize);
		dim3 grid((noElem_Embedding + block.x -1)/block.x);
		//Đầu vào của kernelCalDegreeOfVid là một tập đỉnh trên RMP kèm theo Embedding Header của nó.
		kernelCalDegreeOfVidOnEmbeddingColumn<<<grid,block>>>( \
			hEmbedding.at(idx_EC).dArrEmbedding, \
			hdb.at(0).dO, hdb.at(0).noElemdO, \
			noElem_Embedding, \
			hdb.at(0).noElemdN, \
			maxOfVer, \
			dArrDegreeOfVid);
		CHECK(cudaStatus=cudaDeviceSynchronize()); 
		if(cudaStatus!=cudaSuccess)
		{
			status=-1;
			goto Error;
		}
		CHECK(cudaStatus=cudaGetLastError());	
		if(cudaStatus!=cudaSuccess){
			status =-1;
			goto Error;
		}
	}
	CHECK(cudaStatus =cudaMalloc((void**)&dArrDegreeOfVid,noElem_Embedding*sizeof(float)));
	if(cudaStatus !=cudaSuccess){
		status =-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaMemset(dArrDegreeOfVid,0,noElem_Embedding*sizeof(float)));
	if(cudaStatus !=cudaSuccess){
		status =-1;
		goto Error;
	}



Error:
	return status;
}

int EXTk::mark_edge(int vi,int vj,int li,int lij,int lj,int *&dValid)
{
	CUCHECK(cudaMalloc((void**)&dValid,sizeof(int)*noElem));
	CUCHECK(cudaMemset(dValid,0,sizeof(int)*noElem));

	dim3 block(blocksize);
	dim3 grid((noElem+block.x-1)/block.x);
	//GridBlockCompute(noElem,block,grid);

	kernel_mark_edge<<<grid,block>>>(vi,vj,li,lij,lj,dArrExt,dValid,noElem);
	CUCHECK(cudaDeviceSynchronize());
	CUCHECK(cudaGetLastError());
	return 0;
}
//inuse
void structValid::extractValid(EXTk &outputEXT)
{
	//doing somethings here
	//1. Scan on dArrValid to get index
	int *dArrValidScanResult=nullptr;
	CUCHECK(cudaMalloc((void**)&dArrValidScanResult,sizeof(int)*noElem));
	CUCHECK(cudaMemset(dArrValidScanResult,0,sizeof(int)*noElem));

	CUCHECK(myScanV(dArrValid,noElem,dArrValidScanResult));
	//displayDeviceArr(dArrValid,noElem);
	//displayDeviceArr(dArrValidScanResult,noElem);
	int noElem_dExt=0;
	get_noElem_valid(dArrValid,dArrValidScanResult,noElem,noElem_dExt);
	if (noElem_dExt == 0) 
	{
		CUCHECK(cudaFree(dArrValidScanResult));
		return;
	}
	//std::printf("\n noElem Valid:%d",noElem_dExt);

	outputEXT.noElem = noElem_dExt;

	CUCHECK(cudaMalloc((void**)&outputEXT.dArrExt,sizeof(EXT)*outputEXT.noElem));
	dim3 block(blocksize);
	dim3 grid((noElem+block.x -1)/block.x);
	kernelExtractValidExtensionTodExt<<<grid,block>>>(dArrEXT,dArrValid,dArrValidScanResult,noElem,outputEXT.dArrExt,noElem_dExt);


	CUCHECK(cudaFree(dArrValidScanResult));
}

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

		displayDeviceArr(dvi,1);
		displayDeviceArr(dvj,1);
		displayDeviceArr(dli,1);

		dim3 block1(blocksize);
		dim3 grid1((noElem_dallPossibleExtension + block1.x -1)/block1.x);
		//kernelFilldArrUniEdgev2<<<grid,block>>>( d_allPossibleExtension,d_allPossibleExtensionScanResult, \
		//	noElem_dallPossibleExtension,uniFE.dUniEdge,Lv,dFromLi,dFromVi,maxId);

		kernelFillUniFE<<<grid1,block1>>>( d_allPossibleExtension, \
								d_allPossibleExtensionScanResult, \
								noElem_dallPossibleExtension, \
								uniFE.dUniEdge, \
								Lv, dvi, dvj, dli);

		CUCHECK(cudaDeviceSynchronize());
		CUCHECK(cudaGetLastError());

		uniFE.show();

	}

	if (d_allPossibleExtension != nullptr) CUCHECK(cudaFree(d_allPossibleExtension));
	if (d_allPossibleExtensionScanResult != nullptr) CUCHECK(cudaFree(d_allPossibleExtensionScanResult));
	return;
}
//inuse
void EXTk::extractUniBackwardExtension(unsigned int& Lv,unsigned int& Le,int& noElemRMP, \
									   int*& dRMP,int*& dRMPLabel, int& noElemMappingVj,int& vi,int& li)
{
	int noElemdAllExtension = Le * (noElemRMP-2);
	//Tính số lượng tất cả các cạnh có thể có dựa vào nhãn của chúng
	int noElemBW = noElemRMP -2;
	displayDeviceArr(dRMP,noElemBW);
	displayDeviceArr(dRMPLabel,noElemBW);
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
	cout<<endl<<"dMappingVj:"<<endl;
	displayDeviceArr(dMappingVj,noElemMappingVj);
	//Bật 1 cho các unique backward extension trong dAllExtension
	dim3 block1(blocksize);
	dim3 grid1((noElem + block.x - 1)/block.x);

	kernelMarkUniBE<<<grid1,block1>>>(dMappingVj,dAllExtension,Lv,noElem,dArrExt);
	CUCHECK(cudaDeviceSynchronize());
	CUCHECK(cudaGetLastError());

	cout<<endl<<"dAllExtension:"<<endl;
	displayDeviceArr(dAllExtension,noElemdAllExtension);

	//Scan on dAllExtension to get index, noElem and cudamalloc
	get_idx(dAllExtension,noElemdAllExtension,dAllExtensionIdx);
	get_noElem_valid(dAllExtension,dAllExtensionIdx,noElemdAllExtension,uniBE.noElem);
	CUCHECK(cudaMalloc((void**)&uniBE.dUniEdge,sizeof(UniEdge)*uniBE.noElem));

	//Ánh xạ ngược từ dAllExtension sang UniEdge Backward
	dim3 block2(blocksize);
	dim3 grid2((noElemdAllExtension + block2.x -1)/block2.x);
	kernelExtractUniBE<<<grid2,block2>>>(dAllExtension,noElemdAllExtension,dRMP,dRMPLabel,Lv,uniBE.dUniEdge,dAllExtensionIdx,vi,li);
	CUCHECK(cudaDeviceSynchronize());
	CUCHECK(cudaGetLastError());

	uniBE.show();


	if (dAllExtension != nullptr) CUCHECK(cudaFree(dAllExtension));
	if (dAllExtensionIdx != nullptr) CUCHECK(cudaFree(dAllExtensionIdx));
	if (dMappingVj != nullptr) CUCHECK(cudaFree(dMappingVj));
	if (dRMPLabel != nullptr) CUCHECK(cudaFree(dRMPLabel));
	if (dRMP != nullptr) CUCHECK(cudaFree(dRMP));
	return;
}

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

int UniEdge::calcSupport()
{
	//don't use
	return 0;
}

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
			uniBE.showSupport();
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
			uniFE.showSupport();
		}

		CUCHECK(cudaFree(dArrBoundaryIndex));
		CUCHECK(cudaFree(dF));
	}
	catch(...)
	{
		FCHECK(-1);
	}
}

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
		CHECK(cudaMalloc((void**)&dV,uniEdge.noElem*sizeof(int)));
		CHECK(cudaMemset(dV,0,sizeof(int)*uniEdge.noElem));

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

		/*PMS_PRINT("\n ********hUniEdgeSatisfyMinsup.dUniEdge****************\n");
		FUNCHECK(status=displnoElemUniEdgeSatisfyMinSupayArrUniEdge(dArrUniEdgeSup,));
		if(status!=0){
		goto Error;
		}*/

		PMS_PRINT("\n ********hUniEdgeSatisfyMinsup.dSup****************\n");
		displayDeviceArr(dSup,uniES.noElem);

		CUCHECK(cudaMemcpy(uniES.hArrSupport,dSup,sizeof(int)*uniES.noElem,cudaMemcpyDeviceToHost));

		uniES.show();
		CUCHECK(cudaFree(dResultSup));
		CUCHECK(cudaFree(dV));
		CUCHECK(cudaFree(dVScanResult));
		CUCHECK(cudaFree(dSup));
		uniEdge.ReleaseMemory();
	}
	catch(...)
	{
		FCHECK(-1);
	}
}

void PMS::buildRMPLabel(int*& dRMP, int*& dRMPLabel,int& noElemMappingVj)
{
	vector<int> RMP;
	vector<int> vertexLabel;
	int vi,vj;
	int preVj;
	vi = DFS_CODE.back().from;
	vj = DFS_CODE.back().to;
	preVj = vi;
	RMP.push_back(vj);
	vertexLabel.push_back(DFS_CODE.back().tolabel);
	for(int i = DFS_CODE.size() - 2; i>=0;i--)
	{
		vi = DFS_CODE.at(i).from;
		vj = DFS_CODE.at(i).to;
		if(vi<vj && preVj==vj)
		{
			RMP.push_back(vj);
			vertexLabel.push_back(DFS_CODE.at(i).tolabel);
			preVj = vi;
			RMP.push_back(preVj);
			vertexLabel.push_back(DFS_CODE.at(i).fromlabel);
		}
	}
	std::reverse(RMP.begin(),RMP.end());
	std::reverse(vertexLabel.begin(),vertexLabel.end());

	DFS_CODE.noElemOnRMP = RMP.size();
	int *hRMPLabel = nullptr;
	int *hRMP = nullptr;

	hRMP = (int*)malloc(sizeof(int)*DFS_CODE.noElemOnRMP);
	if(hRMP == nullptr) {FCHECK(-1);}
	hRMPLabel = (int*)malloc(sizeof(int)*DFS_CODE.noElemOnRMP);
	if(hRMPLabel == nullptr) {FCHECK(-1);}

	dRMPLabel = nullptr;
	dRMP = nullptr;
	CUCHECK(cudaMalloc((void**)&dRMPLabel,sizeof(int)*DFS_CODE.noElemOnRMP));
	CUCHECK(cudaMalloc((void**)&dRMP,sizeof(int)*DFS_CODE.noElemOnRMP));

	for (int i = 0; i < RMP.size()-2; i++)
	{
		std::printf("V[%d] Li[%d]; ",RMP[i],vertexLabel[i]);
		hRMP[i] = RMP[i];
		hRMPLabel[i] = vertexLabel[i];
	}
	
	CUCHECK(cudaMemcpy(dRMP,hRMP,DFS_CODE.noElemOnRMP*sizeof(int),cudaMemcpyHostToDevice));
	CUCHECK(cudaMemcpy(dRMPLabel,hRMPLabel,DFS_CODE.noElemOnRMP*sizeof(int),cudaMemcpyHostToDevice));
	noElemMappingVj = hRMPLabel[RMP.size()-3] + 1;

	free(hRMP);
	free(hRMPLabel);
	RMP.clear();
	vertexLabel.clear();
}

void PMS::buildExistBackwardInfo(existBackwardInfo& existBackwardInfo)
{
	try
	{
		//Duyệt qua RMP để biết có bao nhiêu đỉnh
		//Chứa các Vj của các backward extension có thể có từ đỉnh cuối của DFS_CODE.
		vector<int> vertexVj;
		//Lấy vi và vj của cạnh cuối. Nếu cạnh cuối là backward thì sao?
		//Nếu cạnh cuối là backward thì xét cạnh kế cuối cho đến khi đó là forward.
		int vi,vj;
		int preVj=-1;
		//int idx = -1;
		//bool getVjBackward = true;
		vector<int> vjBackward;
		int idxContinue=0;
		for(int i = DFS_CODE.size() - 1; i>=0;i--)
		{ 
			vi = DFS_CODE.at(i).from;
			vj = DFS_CODE.at(i).to;

			bool isForward = (vi<vj);
			//Nếu là forward thì không lấy Vj nữa
			if(isForward==true)
			{
				idxContinue = i;
				preVj = vj;
				break;
			}
			else
			{
				vjBackward.push_back(vj);
			}
		}
		for (int i = idxContinue; i >= 0; i--)
		{
			vi = DFS_CODE.at(i).from;
			vj = DFS_CODE.at(i).to;
			if(vi<vj && preVj==vj)
			{
				//idx+=1;
				//RMP.push_back(idx);
				vertexVj.push_back(vj);
				if(i==0)
				{
					//idx+=1;
					//RMP.push_back(idx);
					vertexVj.push_back(vi);
					break;
				}
				preVj = vi;
			}
		}
		std::reverse(vertexVj.begin(),vertexVj.end());
		//Cập nhật số lượng đỉnh thuộc RMP
		DFS_CODE.noElemOnRMP = vertexVj.size();
		CUCHECK(cudaMalloc((void**)&existBackwardInfo.dVj,sizeof(int)*DFS_CODE.noElemOnRMP));


		CUCHECK(cudaMalloc((void**)&existBackwardInfo.dValidBackward,sizeof(int)*DFS_CODE.noElemOnRMP));
		CUCHECK(cudaMemset(existBackwardInfo.dValidBackward,-1,sizeof(int)*DFS_CODE.noElemOnRMP));
		
		//Nếu có backward
		if(vjBackward.size() > 0)
		{
			int* hValidBackward = (int*)malloc(vjBackward.size()*sizeof(int));
			if(hValidBackward == nullptr) {FCHECK(-1);}

			int* dValidBackward = nullptr;
			CUCHECK(cudaMalloc((void**)&dValidBackward,sizeof(int)*vjBackward.size()));
			//Copy dữ liệu từ vjBackward sang mảng hValidBackward
			for (int i = 0; i < vjBackward.size(); i++)
			{
				hValidBackward[i] = vjBackward.at(i);
			}
			//Copy dữ liệu từ hValidBackward sang dValidBackward
			CUCHECK(cudaMemcpy(dValidBackward,hValidBackward,sizeof(int)*vjBackward.size(),cudaMemcpyHostToDevice));
			free(hValidBackward);

			//Thread i mang giá trị dVj[i] quét trong mảng dValidBackward
			//Thread i set 1 tại existBackwardInfo.dValidBackward[i] nếu dV[i] tồn tại trong dValidBackward
			dim3 block(blocksize);
			dim3 grid((DFS_CODE.noElemOnRMP+block.x-1)/block.x);
			kernelFillValidBackward<<<grid,block>>>( \
				existBackwardInfo.dValidBackward,existBackwardInfo.dVj,DFS_CODE.noElemOnRMP, \
				dValidBackward,vjBackward.size());
			CUCHECK(cudaDeviceSynchronize());
			CUCHECK(cudaGetLastError());
		}
		cout<<endl<<"****dValidBackward*****"<<endl;
		displayDeviceArr(existBackwardInfo.dValidBackward,DFS_CODE.noElemOnRMP);
		cout<<endl<<"****dVj*****"<<endl;
		displayDeviceArr(existBackwardInfo.dVj,DFS_CODE.noElemOnRMP);
		vjBackward.clear();
		vertexVj.clear();
	}
	catch(...)
	{
		FCHECK(-1);
	}
}