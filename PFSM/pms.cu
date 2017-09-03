#pragma once
#include "pms.cuh"


float hTime=0.0;
float dTime=0.0;

PMS::PMS(){
	Level=0;
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
	cudaDeviceReset();
}

void PMS::prepareDataBase(){
	//unsigned int minsup = 34;
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
	fname= "G0G1G2_custom";
	//fname="Chemical_340Origin";

	////PMS pms;	
	ofstream fout("result.txt");

	//Chuyển dữ liệu từ fname sang TRANS
	//pms.run(fname,fout,minsup,maxpat,minnodes,enc,where,directed);
	run(fname,fout,minsup,maxpat,minnodes,enc,where,directed);
	//maxOfVer=pms.findMaxVertices();
	maxOfVer=findMaxVertices();	
	numberOfGraph=noGraphs();
	int sizeOfarrayO=maxOfVer*numberOfGraph;
	int* arrayO = new int[sizeOfarrayO]; //Tạo mảng arrayO có kích thước D*m
	if(arrayO==NULL){
		printf("\n!!!Memory Problem ArrayO");
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
		printf("\n!!!Memory Problem ArrayN");
		exit(1);
	}else
	{
		memset(arrayN, -1, noDeg*sizeof(int));
	}

	//
	int* arrayLO = new int[sizeOfarrayO]; //Mảng arrayLO lưu trữ label cho tất cả các đỉnh trong TRANS.
	if(arrayLO==NULL){ //kiểm tra cấp phát bộ nhớ cho mảng có thành công hay không
		printf("\n!!!Memory Problem ArrayLO");
		exit(1);
	}else
	{
		memset(arrayLO, -1, sizeOfarrayO*sizeof(int));
	}



	int* arrayLN = new int[noDeg]; //Mảng arrayLN lưu trữ label của tất cả các cạnh trong TRANS
	if(arrayLN==NULL){ //kiểm tra cấp phát bộ nhớ cho mảng có thành công hay không
		printf("\n!!!Memory Problem ArrayLN");
		exit(1);
	}else
	{
		memset(arrayLN, -1, noDeg*sizeof(int));
	}


	//pms.importDataToArray(arrayO,arrayLO,arrayN,arrayLN,sizeOfarrayO,noDeg,maxOfVer);
	importDataToArray(arrayO,arrayLO,arrayN,arrayLN,sizeOfarrayO,noDeg,maxOfVer);
	cout<<"ArrayO:";
	displayArray(arrayO,sizeOfarrayO);
	cout<<"\nArrayLO:";
	displayArray(arrayLO,sizeOfarrayO);
	cout<<"\nArrayN:";
	displayArray(arrayN,noDeg);
	cout<<"\nArrayLN:";
	displayArray(arrayLN,noDeg);
	//kích thước của dữ liệu
	size_t nBytesO = sizeOfarrayO*sizeof(int);
	size_t nBytesN = noDeg*sizeof(int);

	DB graphdb;
	graphdb.noElemdO = sizeOfarrayO;
	graphdb.noElemdN = noDeg;

	CHECK(cudaMalloc((void**)&graphdb.dO,nBytesO));
	CHECK(cudaMalloc((void**)&graphdb.dLO,nBytesO));
	CHECK(cudaMalloc((void**)&graphdb.dN,nBytesN));
	CHECK(cudaMalloc((void**)&graphdb.dLN,nBytesN));

	CHECK(cudaMemcpy(graphdb.dO,arrayO,nBytesO,cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(graphdb.dLO,arrayLO,nBytesO,cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(graphdb.dN,arrayN,nBytesN,cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(graphdb.dLN,arrayLN,nBytesN,cudaMemcpyHostToDevice));

	//pms.db.push_back(graphdb); //Đưa cơ sở dữ liệu vào vector db
	//pms.countNumberOfDifferentValue(pms.db.at(0).dLO,pms.db.at(0).noElemdO,pms.Lv);
	//pms.countNumberOfDifferentValue(pms.db.at(0).dLN,pms.db.at(0).noElemdN,pms.Le);
	hdb.push_back(graphdb); //Đưa cơ sở dữ liệu vào vector db
	countNumberOfDifferentValue(hdb.at(0).dLO,hdb.at(0).noElemdO,Lv);
	countNumberOfDifferentValue(hdb.at(0).dLN,hdb.at(0).noElemdN,Le);
	//pms.printdb();
}

bool PMS::checkArray(int *hostRef, int *gpuRef, const int N) {
	bool result=true;
	double epsilon = 1.0E-8;
	int match = 1;
	for (int i = 0; i < N; i++) {
		if ((float)(abs(hostRef[i] - gpuRef[i])) > epsilon) {
			match = 0;
			result=false;
			printf("Arrays do not match!\n");
			printf("host %5.2f gpu %5.2f at current %d\n",
				hostRef[i], gpuRef[i], i);
			break;
		}
	}
	if (match){
		printf("Arrays match.\n\n");		
	}

	return result;
}


void PMS::displayArray(int *p, const unsigned int pSize=0)
{
	for(int i=0;i<pSize;i++){
		printf("P[%d]:%d ",i,p[i]);
	}
	printf("\n");
	return;
}

__global__ void kernelPrintdArr(int *dArr,unsigned int noElem){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i<noElem){
		printf("A[%d]:%d ",i,dArr[i]);
	}
}


void PMS::printdb(){
	printf("\n *********** Lv, Le **********\n");
	printf("\n Lv:%d",Lv);
	printf("\n Le:%d",Le);
	for (int i = 0; i < hdb.size(); i++)
	{
		unsigned int noElem =  hdb.at(i).noElemdO;	


		dim3 block(blocksize);
		dim3 grid((noElem + block.x -1)/block.x);
		printf("\n ********* dO *********\n");
		kernelPrintdArr<<<grid,block>>>(hdb.at(i).dO,noElem);
		cudaDeviceSynchronize();
		printf("\n");

		printf("\n ********* dLO *********\n");
		kernelPrintdArr<<<grid,block>>>(hdb.at(i).dLO,noElem);
		cudaDeviceSynchronize();
		printf("\n");

		unsigned int noElemdN = hdb.at(i).noElemdN;
		dim3 blocka(blocksize);
		dim3 grida((noElemdN + blocka.x -1)/blocka.x);

		printf("\n ********* dN *********\n");
		kernelPrintdArr<<<grida,blocka>>>(hdb.at(i).dN,noElemdN);
		cudaDeviceSynchronize();
		printf("\n");

		printf("\n ********* dLN *********\n");
		kernelPrintdArr<<<grida,blocka>>>(hdb.at(i).dLN,noElemdN);
		cudaDeviceSynchronize();
		printf("\n");
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


void  myScanV(int *dArrInput,int noElem,int *&dResult){
	cudaError_t cudaStatus;
	dim3 block(blocksize);
	dim3 grid((noElem + block.x -1)/block.x);

	CHECK(cudaMalloc((void**)&dResult,noElem * sizeof(int)));

	kernelMyScanV<<<grid,block>>>(dArrInput,noElem,dResult);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	CHECK(cudaStatus);
	return;
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
	dim3 block(blocksize);
	dim3 grid((sizeOfArrayLO+block.x-1)/block.x);
	kernelCountNumberOfLabelVertex<<<grid,block>>>(d_LO,d_Lv,sizeOfArrayLO);

	cudaDeviceSynchronize();
	printf("\nElements of d_Lv:");
	kernelPrintdArr<<<grid,block>>>(d_Lv,sizeOfArrayLO);

	int* h_Lv=NULL;
	h_Lv=(int*)malloc(nBytesLv);
	if(h_Lv==NULL){
		printf("h_Lv malloc memory fail");
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

__global__ void kernelGetAndStoreExtension(int *d_O,int *d_LO,unsigned int numberOfElementd_O,int *d_N,int *d_LN,unsigned int numberOfElementd_N,Extension *d_Extension){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i<numberOfElementd_O){
		if (d_O[i]!=-1){
			int j;
			int ek;
			//printf("\nThread:%d",i);	
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
			//printf("\n[%d]:%d",i,ek);
			for(int k=d_O[i];k<ek;k++){
				//do something
				int index= k;
				d_Extension[index].vi=0;
				d_Extension[index].vj=0;
				d_Extension[index].li=d_LO[i];
				d_Extension[index].lij=d_LN[k];
				d_Extension[index].lj=d_LO[d_N[k]];
				d_Extension[index].vgi=i;
				d_Extension[index].vgj=d_N[k];
				//printf("\n[%d]:%d",i,index);
				/*printf("\n[%d]: DFS code:(%d,%d,%d,%d,%d)  (vgi,vgj):(%d,%d)\n",k,d_Extension[i].vi,d_Extension[i].vj,d_Extension[i].li,
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

	cudaStatus=cudaDeviceSynchronize();
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
	arrExtension arrE;
	//cấp phát bộ nhớ cho d_Extension
	arrE.noElem =hdb.at(0).noElemdN;
	size_t nBytesOfArrayExtension = arrE.noElem*sizeof(Extension);

	CHECK(cudaMalloc((Extension**)&arrE.dExtension,nBytesOfArrayExtension));
	//Trích tất cả các cạnh từ database rồi lưu vào d_Extension

	status  = getAndStoreExtension(arrE.dExtension);
	if(status ==-1){
		printf("\n getAndStoreExtension(arrE.dExtension) in extractAllEdgeInDB() failed");
		goto Error;
	}

	hExtension.push_back(arrE);
Error:
	return status;
}

__global__ void kernelPrintExtention(Extension *d_Extension,int n){
	int i=blockIdx.x*blockDim.x + threadIdx.x;
	if (i<n){		
		printf("\n[%d]: DFS code:(%d,%d,%d,%d,%d)  (vgi,vgj):(%d,%d)\n",i,d_Extension[i].vi,d_Extension[i].vj,d_Extension[i].li,d_Extension[i].lij,d_Extension[i].lj,d_Extension[i].vgi,d_Extension[i].vgj);
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
		printf("\n Malloc hExtension in displayArrExtension() failed");
		goto Error;
	}
	CHECK(cudaStatus=cudaMemcpy(hExtension,dExtension,sizeof(Extension)*noElem,cudaMemcpyDeviceToHost));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	for (int i = 0; i < noElem; i++)
	{
		printf("\n[%d]: DFS code:(%d,%d,%d,%d,%d)  (vgi,vgj):(%d,%d)\n",i,hExtension[i].vi,hExtension[i].vj,hExtension[i].li,hExtension[i].lij,hExtension[i].lj,hExtension[i].vgi,hExtension[i].vgj);
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
		printf("\n malloc hUniEdge in displayArrUniEdge() failed");
		goto Error;
	}
	CHECK(cudaStatus=cudaMemcpy(hUniEdge,dUniEdge,sizeof(UniEdge)*noElem,cudaMemcpyDeviceToHost));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	for (int i = 0; i < noElem; i++)
	{
		printf("\n U[%d]: (li lij lj) = (%d %d %d)",i,hUniEdge[i].li,hUniEdge[i].lij,hUniEdge[i].lj);
	}
	free(hUniEdge);
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
	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
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
		printf("A[%d]:%d   ",i,dArr[i]);
	}
}
__global__ void kernelPrintdArr(int *dArr,int noElem){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElem){
		printf("dArr[%d]:%d ",i,dArr[i]);
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
		printf("\n Malloc temp in displayDeviceArr() failed");
		status=-1;
		goto Error;	
	}

	CHECK(cudaMemcpy(temp,dArr,noElem*sizeof(int),cudaMemcpyDeviceToHost));

	for (int i = 0; i < noElem; i++)
	{
		printf(" A[%d]:%d  ",i,temp[i]);
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
		printf("\n Malloc temp in displayDeviceArr() failed");
		status=-1;
		goto Error;	
	}

	CHECK(cudaMemcpy(temp,dArr,noElem*sizeof(float),cudaMemcpyDeviceToHost));

	for (int i = 0; i < noElem; i++)
	{
		int a = (int)temp[i];
		printf(" A[%d]:%d  ",i,a);
	}

	free(temp);
Error:
	return status;
}



__global__ void kernelGetSize(int *dV,int *dVScanResult,int noElem,int *size){
	*size = dVScanResult[noElem-1];
	if(dV[noElem-1]==1){
		*size = *size + 1;
	}
}


cudaError_t getSizeBaseOnScanResult(int *dV,int *dVScanResult,int noElem,int &output){
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
	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
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

__global__ void kernelExtractValidExtension(Extension *d_Extension,int *dV,int *dVScanResult,int numberElementd_Extension,Extension *d_ValidExtension){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i<numberElementd_Extension){
		if(dV[i]==1){
			int index = dVScanResult[i];
			//printf("\nV[%d]:%d, index[%d]:%d,d_Extension[%d], d_Extension[%d]:%d\n",i,V[i],i,index[i],i,i,d_Extension[i].vgi);
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

int PMS::getValidExtension(){
	int status = 0;


	//Phase 1: đánh dấu vị trí những cạnh hợp lệ (li<=lj)

	int numberElementd_Extension = hExtension.at(0).noElem;
	int *dV;
	size_t nBytesdV= numberElementd_Extension*sizeof(int);

	cudaError_t cudaStatus=cudaMalloc((void**)&dV,nBytesdV);
	if (cudaStatus!= cudaSuccess){
		fprintf(stderr,"cudaMalloc array V failed",cudaStatus);
		status = -1;
		goto Error;
	}
	else
	{
		CHECK(cudaMemset(dV,0,nBytesdV));
	}

	cudaStatus=validEdge(hExtension.at(0).dExtension,dV,hExtension.at(0).noElem);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaDeviceSynchronize validEdge failed",cudaStatus);
		status = -1;
		goto Error;
	}
	//
	int *hV = (int*)malloc(sizeof(int)*numberElementd_Extension);
	cudaMemcpy(hV,dV,sizeof(int)*numberElementd_Extension,cudaMemcpyDeviceToHost);
	printf("\n ************ dV **************\n");
	for (int i = 0; i < numberElementd_Extension; i++)
	{
		int temp = hV[i];
		printf("[%d]:%d ",i,temp);
	}

	int* dVScanResult;
	cudaStatus=cudaMalloc((void**)&dVScanResult,numberElementd_Extension*sizeof(int));
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"Cuda Malloc failed",cudaStatus);
		goto Error;
	}	
	else
	{
		cudaMemset(dVScanResult,0,numberElementd_Extension*sizeof(int));
	}
	//Exclusive scan mảng V và lưu kết quả scan vào mảng index
	//scanV(dV,numberElementd_Extension,dVScanResult);
	myScanV(dV,numberElementd_Extension,dVScanResult);

	printf("\n ************ dVScanResult **************\n");
	cudaMemcpy(hV,dVScanResult,sizeof(int)*numberElementd_Extension,cudaMemcpyDeviceToHost);
	for (int i = 0; i < numberElementd_Extension; i++)
	{
		int temp = hV[i]; 
		printf("[%d]:%d ",i,temp);
	}

	//Phase 2: trích những cạnh hợp lệ sang một mảng khác dValidExtension
	//arrExtension arrValidExtension;
	hValidExtension.resize(1);
	getSizeBaseOnScanResult(dV,dVScanResult,numberElementd_Extension,hValidExtension.at(0).noElem);
	printf("\n arrValidExtension.noElem:%d",hValidExtension.at(0).noElem);
	CHECK(cudaMalloc((void**)&(hValidExtension.at(0).dExtension),sizeof(Extension)*hValidExtension.at(0).noElem));
	CHECK(extractValidExtension(hExtension.at(0).dExtension,dV,dVScanResult,numberElementd_Extension,hValidExtension.at(0).dExtension));
	//displayArrExtension(arrValidExtension.dExtension,arrValidExtension.noElem);

	//hValidExtension.push_back(arrValidExtension);
	printf("\n************hValidExtension***********\n");
	displayArrExtension(hValidExtension.at(0).dExtension,hValidExtension.at(0).noElem);

	cudaFree(dV);
	cudaFree(dVScanResult);
	free(hV);
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

__global__ void kernelCalcLabelAndStoreUniqueExtension(int *d_allPossibleExtension,int *d_allPossibleExtensionScanResult,unsigned int noElem_allPossibleExtension,UniEdge *d_UniqueExtension,unsigned int Le,unsigned int Lv){
	int i=blockIdx.x*blockDim.x + threadIdx.x;	
	if(i<noElem_allPossibleExtension && d_allPossibleExtension[i]==1){
		int li,lj,lij;
		li=i/(Le*Lv);
		lij=(i%(Le*Lv))/Lv;
		lj=(i%(Le*Lv))-((i%(Le*Lv))/Lv)*Lv;
		int index = d_allPossibleExtensionScanResult[i];
		//printf("\n[%d]:%d li:%d lij:%d lj:%d",i,d_allPossibleExtensionScanResult[i],li,lij,lj);
		d_UniqueExtension[index].li=li;
		d_UniqueExtension[index].lij=lij;
		d_UniqueExtension[index].lj=lj;
	}
}

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

int PMS::extractUniEdge(){
	int status=0;


	//Tính số lượng tất cả các cạnh có thể có dựa vào nhãn của chúng
	unsigned int noElem_dallPossibleExtension=Le*Lv*Lv;
	int *d_allPossibleExtension;

	//cấp phát bộ nhớ cho mảng d_allPossibleExtension
	cudaError_t	cudaStatus=cudaMalloc((void**)&d_allPossibleExtension,noElem_dallPossibleExtension*sizeof(int));
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaMalloc d_allPossibleExtension failed",cudaStatus);
		status = -1;
		goto Error;
	}
	else
	{
		CHECK(cudaMemset(d_allPossibleExtension,0,noElem_dallPossibleExtension*sizeof(int)));
	}

	cudaStatus=markLabelEdge(hValidExtension.at(0).dExtension,hValidExtension.at(0).noElem,Lv,Le,d_allPossibleExtension);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"getUniqueExtension failed",cudaStatus);
		status=-1;
		goto Error;
	}


	int *d_allPossibleExtensionScanResult;
	cudaStatus=cudaMalloc((void**)&d_allPossibleExtensionScanResult,noElem_dallPossibleExtension*sizeof(int));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaMalloc d_allPossibleExtensionScanResult failed");
		status = -1;
		goto Error;
	}
	// printf("\n **************** hValidExtension ****************\n");
	//displayArrExtension(hValidExtension.at(0).dExtension,hValidExtension.at(0).noElem);

	//Exclusive scan mảng d_allPossibleExtension và lưu kết quả vào mảng d_allPossibleExtensionScanResult
	//cudaStatus = scanV(d_allPossibleExtension,noElem_dallPossibleExtension,d_allPossibleExtensionScanResult);
	//if(cudaStatus!=cudaSuccess){
	//	fprintf(stderr,"\n ScanV() in computeSupport() failed");
	//	status = -1;
	//	goto Error;
	//}
	myScanV(d_allPossibleExtension,noElem_dallPossibleExtension,d_allPossibleExtensionScanResult);
	//	 printf("\n **************** hValidExtension ****************\n");
	//displayArrExtension(hValidExtension.at(0).dExtension,hValidExtension.at(0).noElem);


	//printf("\n**********d_allPossibleExtension************\n");
	//displayDeviceArr(d_allPossibleExtension,noElem_dallPossibleExtension);


	arrUniEdge strUniEdge;
	int noElem_d_UniqueExtension=0;
	//Tính kích thước của mảng d_UniqueExtension dựa vào kết quả exclusive scan
	cudaStatus=getSizeBaseOnScanResult(d_allPossibleExtension,d_allPossibleExtensionScanResult,noElem_dallPossibleExtension,noElem_d_UniqueExtension);
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"getLastElement() in extractUniEdge() failed",cudaStatus);
		status = -1;
		goto Error;
	}
	//printf("\n\nnoElem_d_UniqueExtension:%d",noElem_d_UniqueExtension);
	strUniEdge.noElem = noElem_d_UniqueExtension;



	//Tạo mảng d_UniqueExtension với kích thước mảng vừa tính được
	cudaStatus=cudaMalloc((void**)&strUniEdge.dUniEdge,noElem_d_UniqueExtension*sizeof(UniEdge));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaMalloc d_UniqueExtension in extractUniEdge() failed",cudaStatus);
		status = -1;
		goto Error;
	}
	else
	{
		CHECK(cudaMemset(strUniEdge.dUniEdge,0,noElem_d_UniqueExtension*sizeof(UniEdge)));
	}




	//Ánh xạ ngược lại từ vị trí trong d_allPossibleExtension thành cạnh và lưu kết quả vào d_UniqueExtension
	cudaStatus=calcLabelAndStoreUniqueExtension(d_allPossibleExtension,d_allPossibleExtensionScanResult,noElem_dallPossibleExtension,strUniEdge.dUniEdge,noElem_d_UniqueExtension,Le,Lv);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n\ncalcLabelAndStoreUniqueExtension() in extractUniEdge() failed",cudaStatus);
		status = -1;
		goto Error;
	}



	hUniEdge.push_back(strUniEdge);
	printf("\n **************** hUniEdge ****************\n");
	displayArrUniEdge(hUniEdge.at(0).dUniEdge,hUniEdge.at(0).noElem);


	cudaFree(d_allPossibleExtension);
	cudaFree(d_allPossibleExtensionScanResult);
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
__global__ void kernelGetLastElement(int *dScanResult,unsigned int noElem,int *output){
	output[0]=dScanResult[noElem-1];
}


cudaError_t getLastElement(int *dScanResult,unsigned int noElem,int &output){
	cudaError_t cudaStatus;
	dim3 block(blocksize);
	dim3 grid((noElem+block.x-1)/block.x);

	int *value;
	cudaMalloc((int**)&value,sizeof(int));

	kernelGetLastElement<<<1,1>>>(dScanResult,noElem,value);
	cudaDeviceSynchronize();
	cudaStatus= cudaGetLastError();
	if(cudaStatus != cudaSuccess){
		fprintf(stderr,"cudaDeviceSynchronize failed",cudaStatus);
		goto Error;
	}

	CHECK(cudaMemcpy(&output,value,sizeof(int),cudaMemcpyDeviceToHost));
	//printf("\n\nnumberElementd_UniqueExtension:%d",numberElementd_UniqueExtension);

	cudaFree(value);
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
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() of kernelComputeSupport in computeSupport failed",cudaStatus);
		goto Error;
	}



	//Duyệt qua mảng d_UniqueExtension, tính reduction cho mỗi segment i*noElemF, kết quả của reduction là độ support của cạnh i trong d_UniqueExtension
	int *tempF;
	cudaStatus = cudaMalloc((void**)&tempF,noElemF*sizeof(int));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n CudaMalloc tempF in calcSupport() failed",cudaStatus);
		goto Error;
	}
	else
	{
		CHECK(cudaMemset(tempF,0,noElemF*sizeof(int)));
	}

	//float *resultSup; /* Lưu kết quả reduction */
	hResultSup = (int*)malloc(noElemdUniEdge*sizeof(int));
	if (hResultSup==NULL){
		printf("\n Malloc hResultSup in calcSupport() failed");
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
		cudaDeviceSynchronize();
		reduction(tempF,noElemF,hResultSup[i]);		
	}
	////In độ hỗ trợ cho các cạnh tương ứng trong mảng kết quả resultSup
	//for (int i = 0; i < noElemdUniEdge; i++)
	//{
	//	printf("\n resultSup[%d]:%d",i,hResultSup[i]);
	//}

	cudaFree(tempF);

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

__global__ void	kernelExtractUniEdgeSatifyMinsup(UniEdge *dUniEdge,int *dV,int *dVScanResult,int noElemUniEdge,UniEdge *dUniEdgeSatisfyMinsup,int *dSup,int *dResultSup){
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



int PMS::extractUniEdgeSatisfyMinsup(int *hResultSup,int noElemUniEdge,unsigned int minsup){
	int status=0;
	cudaError_t cudaStatus;
	//1. Cấp phát mảng trên device có kích thước bằng noElemUniEdge
	int *dResultSup=nullptr;
	CHECK(cudaMalloc((void**)&dResultSup,noElemUniEdge*sizeof(int)));
	CHECK(cudaMemcpy(dResultSup,hResultSup,noElemUniEdge*sizeof(int),cudaMemcpyHostToDevice));

	printf("\n *******dResultSup********\n");
	displayDeviceArr(dResultSup,noElemUniEdge);

	//2. Đánh dấu 1 trên dV cho những phần tử thoả minsup
	int *dV=nullptr;
	CHECK(cudaMalloc((void**)&dV,noElemUniEdge*sizeof(int)));
	CHECK(cudaMemset(dV,0,sizeof(int)*noElemUniEdge));

	dim3 block(blocksize);
	dim3 grid((noElemUniEdge + block.x - 1)/block.x);
	kernelMarkUniEdgeSatisfyMinsup<<<grid,block>>>(dResultSup,noElemUniEdge,dV,minsup);
	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n kernelMarkUniEdgeSatisfyMinsup in extractUniEdgeSatisfyMinsup() failed",cudaStatus);
		status = -1;
		goto Error;
	}

	printf("\n ***********dV**********\n");
	displayDeviceArr(dV,noElemUniEdge);

	int *dVScanResult=nullptr;
	CHECK(cudaMalloc((void**)&dVScanResult,noElemUniEdge*sizeof(int)));
	//CHECK(scanV(dV,noElemUniEdge,dVScanResult));
	myScanV(dV,noElemUniEdge,dVScanResult);
	printf("\n ***********dVScanResult**********\n");
	displayDeviceArr(dVScanResult,noElemUniEdge);

	hUniEdgeSatisfyMinsup.resize(1);
	CHECK(getSizeBaseOnScanResult(dV,dVScanResult,noElemUniEdge,hUniEdgeSatisfyMinsup.at(0).noElem));
	CHECK(cudaMalloc((void**)&hUniEdgeSatisfyMinsup.at(0).dUniEdge,hUniEdgeSatisfyMinsup.at(0).noElem*sizeof(UniEdge)));
	hUniEdgeSatisfyMinsup.at(0).hArrSup = (int*)malloc(sizeof(int)*hUniEdgeSatisfyMinsup.at(0).noElem);
	if (hUniEdgeSatisfyMinsup.at(0).hArrSup ==NULL){
		printf("\n malloc hArrSup of hUniEdgeSatisfyMinsup failed()");
		exit(1);
	}


	int *dSup=nullptr;
	CHECK(cudaMalloc((void**)&dSup,hUniEdgeSatisfyMinsup.at(0).noElem*sizeof(int)));


	dim3 blocka(blocksize);
	dim3 grida((noElemUniEdge + blocka.x -1)/blocka.x);
	kernelExtractUniEdgeSatifyMinsup<<<grida,blocka>>>(hUniEdge.at(0).dUniEdge,dV,dVScanResult,noElemUniEdge,hUniEdgeSatisfyMinsup.at(0).dUniEdge,dSup,dResultSup);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		status = -1;
		fprintf(stderr,"\n kernelExtractUniEdgeSatisfyMinsup() in extractUniEdgeSatisfyMinsup() failed",cudaStatus);
		goto Error;
	}
	printf("\n ********hUniEdgeSatisfyMinsup.dUniEdge****************\n");
	displayArrUniEdge(hUniEdgeSatisfyMinsup.at(0).dUniEdge,hUniEdgeSatisfyMinsup.at(0).noElem);
	printf("\n ********hUniEdgeSatisfyMinsup.dSup****************\n");
	displayDeviceArr(dSup,hUniEdgeSatisfyMinsup.at(0).noElem);

	CHECK(cudaMemcpy(hUniEdgeSatisfyMinsup.at(0).hArrSup,dSup,sizeof(int)*hUniEdgeSatisfyMinsup.at(0).noElem,cudaMemcpyDeviceToHost));

	for (int i = 0; i < hUniEdgeSatisfyMinsup.at(0).noElem; i++)
	{
		printf("\n hArrSup:%d ",hUniEdgeSatisfyMinsup.at(0).hArrSup[i]);
	}

	cudaFree(dResultSup);
	cudaFree(dV);
	cudaFree(dVScanResult);
	cudaFree(dSup);
Error:
	return status;
}


int PMS::computeSupport(){
	int status=0;
	/* Xây dựng Boundary cho mảng d_ValidExtension */
	//1. Cấp phát một mảng d_B và gán các giá trị 0 cho mọi phần tử của d_B
	unsigned int noElement_dB=hValidExtension.at(0).noElem;
	int* dB;
	cudaError_t cudaStatus=cudaMalloc((int**)&dB,noElement_dB*sizeof(int));
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaMalloc dB in computeSupport() failed",cudaStatus);
		status = -1;
		goto Error;
	}
	else
	{
		CHECK(cudaMemset(dB,0,noElement_dB*sizeof(int)));
	}
	//printf("\n**********dValidExtension*************\n");
	//displayArrExtension(hValidExtension.at(0).dExtension,noElement_dB);
	//printf("\n*********dB********\n");
	//displayDeviceArr(dB,noElement_dB);


	//Gián giá trị boundary cho d_B
	cudaStatus=calcBoundary(hValidExtension.at(0).dExtension,noElement_dB,dB,maxOfVer);
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"calcBoundary() in computeSupport() failed",cudaStatus);
		return 1;
	}

	printf("\n**********dValidExtension*************\n");
	displayArrExtension(hValidExtension.at(0).dExtension,noElement_dB);
	printf("\n*********dB********\n");
	displayDeviceArr(dB,noElement_dB);


	//2. Exclusive Scan mảng d_B
	int* dBScanResult;
	cudaStatus=cudaMalloc((int**)&dBScanResult,noElement_dB*sizeof(int));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaMalloc dBScanResult in computeSupport() failed",cudaStatus);
		status = -1;
		goto Error;
	}
	else
	{
		cudaMemset(dBScanResult,0,noElement_dB*sizeof(int));
	}

	//cudaStatus=scanV(dB,noElement_dB,dBScanResult);
	//if(cudaStatus!=cudaSuccess){
	//	fprintf(stderr,"\nscanB function failed",cudaStatus);
	//	status =-1;
	//	goto Error;
	//}
	myScanV(dB,noElement_dB,dBScanResult);
	printf("\n\n*******dBScanResult***********\n");
	displayDeviceArr(dBScanResult,noElement_dB);

	//3. Tính độ hỗ trợ cho các mở rộng trong d_UniqueExtension
	//3.1 Tạo mảng d_F có số lượng phần tử bằng với giá trị cuối cùng của mảng d_scanB_Result cộng 1 và gán giá trị 0 cho các phần tử.
	int noElemF=0;
	cudaStatus=getLastElement(dBScanResult,noElement_dB,noElemF);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ngetLastElement function failed",cudaStatus);
		return 1;
	}

	noElemF++;
	/*noElemGraphInExt=noElemF;*/

	printf("\n\n noElement_F:%d",noElemF);
	int noElem_d_UniqueExtension= hUniEdge.at(0).noElem;
	int *dF;
	cudaStatus=cudaMalloc((int**)&dF,noElem_d_UniqueExtension*noElemF*sizeof(int));
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaMalloc dF in computeSupport() failed",cudaStatus);
		status =-1;
		goto Error;
	}
	else
	{
		CHECK(cudaMemset(dF,0,noElem_d_UniqueExtension*noElemF*sizeof(int)));
	}
	int *hResultSup=nullptr;
	cudaStatus=calcSupport(hUniEdge.at(0).dUniEdge,hUniEdge.at(0).noElem,hValidExtension.at(0).dExtension,hValidExtension.at(0).noElem,dBScanResult,dF,noElemF,hResultSup);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n calcSupport() in computeSupport() failed",cudaStatus);
		status =-1;
		goto Error;
	}

	////In độ hỗ trợ cho các cạnh tương ứng trong mảng kết quả h_resultSup
	//for (int i = 0; i < noElem_d_UniqueExtension; i++)
	//{
	//	printf("\n resultSup[%d]:%d",i,hResultSup[i]);
	//}
	//
	extractUniEdgeSatisfyMinsup(hResultSup,noElem_d_UniqueExtension,minsup);

	cudaFree(dBScanResult);
	cudaFree(dB);
Error:
	return status;
}

__global__ void kernelGetGraphIdContainEmbedding(int li,int lij,int lj,Extension *d_ValidExtension,int noElem_d_ValidExtension,int *dV,unsigned int maxOfVer){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i<noElem_d_ValidExtension){
		if(	d_ValidExtension[i].li == li && d_ValidExtension[i].lij == lij && 	d_ValidExtension[i].lj == lj){
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
	//printf("\n\nnumberElementd_UniqueExtension:%d",numberElementd_UniqueExtension);

	cudaFree(temp);
Error:	
	return cudaStatus;	
}

__global__ void kernelGetGraph(int *dV,int noElemdV,int *d_kq,int *dVScanResult){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElemdV){
		if(dV[i]!=0){
			d_kq[dVScanResult[i]]=i;
		}
	}
}

__global__ void kernelGetLastElementEXT(EXT *inputArray,int noEleInputArray,int *value,unsigned int maxOfVer){
	*value = inputArray[noEleInputArray-1].vgi/maxOfVer; /*Lấy global vertex id chia cho tổng số đỉnh của đồ thị (maxOfVer). Ở đây các đồ thị luôn có số lượng đỉnh bằng nhau (maxOfVer) */
}

cudaError_t getLastElementEXT(EXT *inputArray,int numberElementOfInputArray,int &outputValue,unsigned int maxOfVer){
	cudaError_t cudaStatus;

	int *temp=nullptr;
	CHECK(cudaStatus=cudaMalloc((int**)&temp,sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		goto Error;
	}

	/* Lấy graphId chứa embedding cuối cùng */
	kernelGetLastElementEXT<<<1,1>>>(inputArray,numberElementOfInputArray,temp,maxOfVer);
	cudaDeviceSynchronize();
	cudaStatus= cudaGetLastError();
	if(cudaStatus != cudaSuccess){
		fprintf(stderr,"cudaDeviceSynchronize failed",cudaStatus);
		goto Error;
	}

	CHECK(cudaMemcpy(&outputValue,temp,sizeof(int),cudaMemcpyDeviceToHost));
	//printf("\n\nnumberElementd_UniqueExtension:%d",numberElementd_UniqueExtension);

	cudaFree(temp);
Error:	
	return cudaStatus;	
}

__global__ void kernelGetGraphIdContainEmbeddingv2(int li,int lij,int lj,EXT *d_ValidExtension,int noElem_d_ValidExtension,int *dV,unsigned int maxOfVer){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i<noElem_d_ValidExtension){
		if(	d_ValidExtension[i].li == li && d_ValidExtension[i].lij == lij && d_ValidExtension[i].lj == lj){
			int graphid = (d_ValidExtension[i].vgi/maxOfVer);
			dV[graphid]=1;
		}
	}
}


int PMS::getGraphIdContainEmbeddingv2(UniEdge edge,int *&hArrGraphId,int &noElemhArrGraphId,EXT *dArrEXT,int noElemdArrEXT){
	int status =0;
	cudaError_t cudaStatus;
	int li,lij,lj;
	li = edge.li;
	lij = edge.lij;
	lj = edge.lj;
	dim3 block(blocksize);
	dim3 grid((noElemdArrEXT+block.x-1)/block.x);

	int *dV=nullptr;
	int noElemdV=0;

	//displayArrExtension(hValidExtension.at(0).dExtension,hValidExtension.at(0).noElem);

	CHECK(cudaStatus =getLastElementEXT(dArrEXT,noElemdArrEXT,noElemdV,maxOfVer));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	++noElemdV;

	CHECK(cudaStatus=cudaMalloc((void**)&dV,noElemdV*sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		fprintf(stderr,"\n cudaMalloc dV in getGraphIdContainEmbedding() failed");
		goto Error;
	}
	else
	{
		CHECK(cudaMemset(dV,0,noElemdV*sizeof(int)));
	}

	kernelGetGraphIdContainEmbeddingv2<<<grid,block>>>(li,lij,lj,dArrEXT,noElemdArrEXT,dV,maxOfVer);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	CHECK(cudaStatus);
	if(cudaStatus!=cudaSuccess){
		status =-1;
		fprintf(stderr,"\n kernelGetGraphIdContainEmbedding() in getGraphIdContainEmbedding() failed",cudaStatus);
		goto Error;
	}

	int *dVScanResult=nullptr;
	CHECK(cudaStatus=cudaMalloc((void**)&dVScanResult,noElemdV*sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		fprintf(stderr,"\n cudaMalloc dVScanResult in getGraphIdContainEmbedding() failed");
		goto Error;
	}
	else
	{
		CHECK(cudaMemset(dVScanResult,0,noElemdV*sizeof(int)));
	}


	//scanV(dV,noElemdV,dVScanResult);
	myScanV(dV,noElemdV,dVScanResult);

	printf("\n ************* dVScanResult *************\n");
	displayDeviceArr(dVScanResult,noElemdV);
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
	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	CHECK(cudaStatus);
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}


	printf("\n*********** d_kq ***********\n");
	displayDeviceArr(d_kq,noElem_kq);

	hArrGraphId=(int*)malloc(sizeof(int)*noElem_kq);
	if(hArrGraphId==NULL){
		printf("\nMalloc hArrGraphId in getGraphIdContainEmbedding() failed");
		exit(1);
	}
	noElemhArrGraphId=noElem_kq;

	CHECK(cudaMemcpy(hArrGraphId,d_kq,sizeof(int)*noElem_kq,cudaMemcpyDeviceToHost));

	cudaFree(d_kq);
	cudaFree(dV);
	cudaFree(dVScanResult);
Error:
	return status;
}


int PMS::getGraphIdContainEmbedding(UniEdge edge,int *&hArrGraphId,int &noElemhArrGraphId){
	int status =0;
	int noElemdValidExtension = hExtension.at(0).noElem;

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

	cudaError_t cudaStatus=cudaMalloc((void**)&dV,noElemdV*sizeof(int));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMalloc dV in getGraphIdContainEmbedding() failed");
		goto Error;
	}
	else
	{
		CHECK(cudaMemset(dV,0,noElemdV*sizeof(int)));
	}

	kernelGetGraphIdContainEmbedding<<<grid,block>>>(li,lij,lj,hValidExtension.at(0).dExtension,hValidExtension.at(0).noElem,dV,maxOfVer);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		status =-1;
		fprintf(stderr,"\n kernelGetGraphIdContainEmbedding() in getGraphIdContainEmbedding() failed",cudaStatus);
		goto Error;
	}

	int *dVScanResult=nullptr;
	cudaStatus=cudaMalloc((void**)&dVScanResult,noElemdV*sizeof(int));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMalloc dVScanResult in getGraphIdContainEmbedding() failed");
		goto Error;
	}
	else
	{
		CHECK(cudaMemset(dVScanResult,0,noElemdV*sizeof(int)));
	}


	//scanV(dV,noElemdV,dVScanResult);
	myScanV(dV,noElemdV,dVScanResult);
	printf("\n ************* dVScanResult *************\n");
	displayDeviceArr(dVScanResult,noElemdV);
	int noElem_kq;	
	CHECK(getLastElement(dVScanResult,noElemdV,noElem_kq));
	noElem_kq++;

	int *d_kq;
	cudaMalloc((void**)&d_kq,sizeof(int)*noElem_kq);

	dim3 blocka(blocksize);
	dim3 grida((noElemdV + blocka.x -1)/blocka.x);

	kernelGetGraph<<<grida,blocka>>>(dV,noElemdV,d_kq,dVScanResult);
	cudaDeviceSynchronize();

	printf("\n*********** d_kq ***********\n");
	displayDeviceArr(d_kq,noElem_kq);

	hArrGraphId=(int*)malloc(sizeof(int)*noElem_kq);
	if(hArrGraphId==NULL){
		printf("\nMalloc hArrGraphId in getGraphIdContainEmbedding() failed");
		exit(1);
	}
	noElemhArrGraphId=noElem_kq;

	CHECK(cudaMemcpy(hArrGraphId,d_kq,sizeof(int)*noElem_kq,cudaMemcpyDeviceToHost));

	cudaFree(d_kq);
	cudaFree(dV);
	cudaFree(dVScanResult);
Error:
	return status;
}

cudaError_t ADM(int *&devicePointer,size_t nBytes){
	cudaError_t cudaStatus;
	cudaStatus= cudaMalloc((void**)&devicePointer,nBytes);
	return cudaStatus;
}


int PMS::Mining(){
	int status = 0;
	cudaError_t cudaStatus;
	int noElemtemp = hUniEdgeSatisfyMinsup.at(0).noElem;
	UniEdge *temp=(UniEdge*)malloc(sizeof(UniEdge)*noElemtemp);
	if(temp==NULL){
		printf("\n malloc temp failed");
		status =-1;
		goto Error;
	}

	CHECK(cudaStatus=cudaMemcpy(temp,hUniEdgeSatisfyMinsup.at(0).dUniEdge,noElemtemp*sizeof(UniEdge),cudaMemcpyDeviceToHost));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	for (int i = 0; i < noElemtemp; i++) //Duyệt qua các UniEdge thoả minSup để kiểm tra minDFS_CODE, nếu thoả thì ghi kết quả vào result và xây dựng embedding
	{
		int li,lij,lj;
		li = temp[i].li;
		lij= temp[i].lij;
		lj=temp[i].lj;

		DFS_CODE.push(0,1,temp[i].li,temp[i].lij,temp[i].lj);//xây dựng DFS_CODE
		minLabel = temp[i].li;
		maxId = 1;

		if(is_min()){ //Nếu DFS_CODE là min thì tìm các graphid chứa embedding của DFS_CODE
			printf("\n This is minDFSCODE\n");

			int *hArrGraphId; //Mảng chứa các graphID có embedding của DFS_Code.
			int noElemhArrGraphId=0;
			/* Trước khi ghi kết quả thì phải biết đồ thị phổ biến đó tồn tại ở những graphId nào. Hàm getGraphIdContainEmbedding dùng để làm việc này
			* 3 tham số đầu tiên của hàm là nhãn cạnh của phần tử d_UniqueExtension đang xét */
			status =getGraphIdContainEmbedding(temp[i],hArrGraphId,noElemhArrGraphId);
			if (status!=0){
				printf("\n\n getGraphIdContainEmbedding() in Mining() failed");
				goto Error;
			}

			//In nội dung mảng hArrGraphId

			printf("\n ************** hArrGraphId ****************\n");
			for (int j = 0; j < noElemhArrGraphId; j++)
			{
				printf("%d ",hArrGraphId[j]);
			}

			/*	Ghi kết quả DFS_CODE vào file result.txt ************************************************************
			*	Hàm report sẽ chuyển DFS_CODE pattern sang dạng đồ thị, sau đó sẽ ghi đồ thị đó xuống file result.txt
			*	Hàm report gồm 3 tham số:
			*	Tham số thứ 1: mảng chứa danh sách các graphID chứa DFS_CODE pattern
			*	Tham số thứ 2: số lượng mảng
			*	Tham số thứ 3: độ hỗ trợ của DFS_CODE pattern *******************************************************/

			report(hArrGraphId,noElemhArrGraphId,hUniEdgeSatisfyMinsup.at(0).hArrSup[i]);
			//Giải phóng bộ nhớ 
			std::free(hArrGraphId);

			//Xây dựng Embedding cho DFS_Code rồi gọi hàm GraphMining để khai thác
			//Trong GraphMining sẽ gọi GraphMining khác để thực hiện khai thác đệ quy

			FUNCHECK(buildFirstEmbedding(temp[i])); //Xây dựng 2 cột embedding ban đầu.
			FUNCHECK(buildRMP()); //Xây dựng RMP ban đầu
			FUNCHECK(FSMining()); //Gọi FSMining.( Hàm này thực hiện theo tuần tự (1. Find Extension -> 2. Extract UniEdge -> 3.Compute & CHECK Support -> 4. CHECK minDFS_CODE -> 5. BuildEmbedding -> 6.Find RMP -> 1.)
		}
		 //Giải phóng bộ nhớ
			if(hRMP.size()>0){
				for (int j = 0; j < hRMP.size(); j++)
				{
					hRMP.at(j).hArrRMP.clear();
				}
				hRMP.clear();
			}

			DFS_CODE.pop();
			if(hEmbedding.size()!=0){
				for (int j = 0; j < hEmbedding.size(); j++)
				{
					cudaFree(hEmbedding.at(j).dArrEmbedding);
				}
				hEmbedding.clear();
			}
	}	
	std::free(temp);
Error:
	return status;
}

__global__ void	kernelGetvivj(EXT *dArrEXT,int noElemdArrEXT,int li,int lij,int lj,int *dvi,int *dvj){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElemdArrEXT){
		if(dArrEXT[i].li == li && dArrEXT[i].lij == lij && dArrEXT[i].lj == lj){
			*dvi=dArrEXT[i].vi;
			*dvj=dArrEXT[i].vj;
			printf("\n Thread:%d (dvi dvj):(%d %d)",i,*dvi,*dvj);
		}
	}
}

int displayDeviceEXT(EXT *dArrEXT,int noElemdArrEXT){
	int status =0;
	cudaError_t cudaStatus;

	EXT *hArrEXT = (EXT*)malloc(sizeof(EXT)*noElemdArrEXT);
	if(hArrEXT == NULL){
		printf("\n malloc hArrEXT failed");
		status =-1;
		goto Error;
	}

	CHECK(cudaStatus = cudaMemcpy(hArrEXT,dArrEXT,noElemdArrEXT*sizeof(EXT),cudaMemcpyDeviceToHost));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	printf("\n************** EXT **************\n");
	for (int i = 0; i < noElemdArrEXT; i++)
	{
		printf("\n (vi vj):(%d %d) (li lij lj):(%d %d %d) (vgi vgj):(%d %d) (RowPointer:%d)",hArrEXT[i].vi,hArrEXT[i].vj,hArrEXT[i].li,hArrEXT[i].lij,hArrEXT[i].lj,hArrEXT[i].vgi,hArrEXT[i].vgj,hArrEXT[i].posRow);
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
	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	CHECK(cudaStatus);
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

int PMS::updateRMP(){
	int status=0;
	cudaError_t cudaStatus;

	//int cSize = hRMP.size();
	//int nSize = cSize +1;
	//hRMP.resize(nSize);
	//int lastIdx = nSize-1;

	hRMPv2.resize(Level);


	RMP *dRMP = nullptr;
	CHECK(cudaStatus=cudaMalloc((void**)&dRMP,sizeof(RMP)));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}

	int noEC = hEmbedding.size(); //noEC is number of embedding column
	int lastIdxEC = noEC-1;

	for (int i = lastIdxEC; i != -1;)
	{
		hRMPv2.at(idxLevel).hArrRMP.push_back(i);
		i=hEmbedding.at(i).prevCol;		
	}
	hRMPv2.at(idxLevel).noElem = hRMPv2.at(idxLevel).hArrRMP.size();

	//In RMP
	for (int i = 0; i < hRMPv2.at(idxLevel).noElem; i++)
	{
		printf("\n RMPv2[%d]:%d",i,hRMPv2.at(idxLevel).hArrRMP.at(i));
	}

Error:
	return status;
}

//Đã tính độ hỗ trợ xong. Cần kiểm tra minDFS_code
int PMS::Miningv2(int noElem,UniEdge *dArrUniEdgeSatisfyMinSup,int *hArrSupport,EXT *dArrEXT,int noElemdArrEXT,int idxExt){
	int status = 0;
	cudaError_t cudaStatus;

	int vi,vj,backward;
	vi=vj=-1;
	backward=0;
	int noElemtemp = noElem;
	UniEdge *temp=(UniEdge*)malloc(sizeof(UniEdge)*noElemtemp);
	if(temp==NULL){
		printf("\n malloc temp failed");
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
		int li,lij,lj;
		li = temp[i].li;
		lij= temp[i].lij;
		lj=temp[i].lj;

		status = getvivj(dArrEXT,noElemdArrEXT,li,lij,lj,vi,vj);
		FUNCHECK(status);
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
			printf("\n This is minDFSCODE\n");

			int *hArrGraphId; //Mảng chứa các graphID có embedding của DFS_Code.
			int noElemhArrGraphId=0;
			status =getGraphIdContainEmbeddingv2(temp[i],hArrGraphId,noElemhArrGraphId,dArrEXT,noElemdArrEXT);
			if (status!=0){
				printf("\n\n getGraphIdContainEmbedding() in Mining() failed");
				goto Error;
			}
			////In nội dung mảng hArrGraphId
			printf("\n ************** hArrGraphId ****************\n");
			for (int j = 0; j < noElemhArrGraphId; j++)
			{
				printf("%d ",hArrGraphId[j]);
			}

			report(hArrGraphId,noElemhArrGraphId,hArrSupport[i]);

				//Xây dựng Embedding cho DFS_Code rồi gọi hàm GraphMining để khai thác
				//Trong GraphMining sẽ gọi GraphMining khác để thực hiện khai thác đệ quy

				FUNCHECK(status=extendEmbedding(temp[i],idxExt));
				if(status!=0){
					goto Error;
				}

				hLevelPtrEmbedding.resize(Level);
				hLevelPtrEmbedding.at(idxLevel).noElem=hEmbedding.size();
				int lastCol = hEmbedding.size()-1;
				hLevelPtrEmbedding.at(idxLevel).noElemEmbedding=hEmbedding.at(lastCol).noElem;
				CHECK(cudaStatus = cudaMalloc((void**)&hLevelPtrEmbedding.at(idxLevel).dArrPointerEmbedding,hLevelPtrEmbedding.at(idxLevel).noElem*sizeof(Embedding**))); //Cấp phát bộ nhớ cho mảng dArrPointerEmbedding.
				if(cudaStatus!=cudaSuccess){
					status = -1;
					std::printf("\n cudaMalloc dArrPointerEmbedding failed()");
					goto Error;
				}

				for (int i = 0; i < hEmbedding.size(); i++)
				{		
					kernelGetPointerdArrEmbedding<<<1,1>>>(hEmbedding.at(i).dArrEmbedding,hLevelPtrEmbedding.at(idxLevel).dArrPointerEmbedding,i); //Mỗi phần tử của mảng dArrPointerEmbedding chứa địa chỉ của dArrEmbedding
				}
				cudaDeviceSynchronize();
				cudaStatus = cudaGetLastError();
				CHECK(cudaStatus);
				if(cudaStatus!=cudaSuccess){
					status = -1;
					printf("\n kernelGetPointerdArrEmbedding failed");
					goto Error;
				}

				FUNCHECK(status = updateRMP()); //Xây dựng RMP ban đầu
				if(status!=0){
					goto Error;
				}

				FUNCHECK(FSMiningv2()); //Gọi FSMining.( Hàm này thực hiện theo tuần tự (1. Find Extension -> 2. Extract UniEdge -> 3.Compute & CHECK Support -> 4. CHECK minDFS_CODE -> 5. BuildEmbedding -> 6.Find RMP -> 1.)

			//Giải phóng bộ nhớ 
			std::free(hArrGraphId);

		}
		DFS_CODE.pop(); //Xoá phần tử cuối của DFS_CODE
		if(backward!=1){ //Nếu pop() một forward thì phải giảm maxId
			--maxId;
		}

		int lastCol = hRMPv2.at(idxLevel).hArrRMP[0];

		cudaFree(hEmbedding.at(lastCol).dArrEmbedding); //xoá phần tử cuối của Embedding
		hEmbedding.pop_back();

		cudaFree(hLevelPtrEmbedding.at(idxLevel).dArrPointerEmbedding); //xoá dArrPointerEmbedding tại Level đang xét
		hLevelPtrEmbedding.pop_back(); //Xoá phần tử cuối của Level pointerEmbeeding đang xét.

		hRMPv2.at(idxLevel).hArrRMP.clear(); //Xoá  RightMostPath của phần tử Embedding tại Level tương ứng.
		hRMPv2.pop_back();
	}	
	std::free(temp);
Error:
	return status;
}

int PMS::Miningv3(int noElem,UniEdge *dArrUniEdgeSatisfyMinSup,int *hArrSupport,EXT *dArrEXT,int noElemdArrEXT,int idxExt){
	int status = 0;
	cudaError_t cudaStatus;
	

	Level++;
	idxLevel=Level-1;
	hLevelPtrEmbedding.at(idxLevel).noElem=hEmbedding.size();
	cudaStatus = cudaMalloc((void**)&hLevelPtrEmbedding.at(idxLevel).dArrPointerEmbedding,hEmbedding.size()*sizeof(Embedding**)); //Cấp phát bộ nhớ cho mảng dArrPointerEmbedding tại Level tương ứng.
	CHECK(cudaStatus);
	if(cudaStatus!=cudaSuccess){
		status = -1;
		std::printf("\n cudaMalloc dArrPointerEmbedding failed()");
		goto Error;
	}

	for (int i = 0; i < hEmbedding.size(); i++)
	{		
		kernelGetPointerdArrEmbedding<<<1,1>>>(hEmbedding.at(i).dArrEmbedding,hLevelPtrEmbedding.at(idxLevel).dArrPointerEmbedding,i); //Mỗi phần tử của mảng dArrPointerEmbedding chứa địa chỉ của dArrEmbedding
	}
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	CHECK(cudaStatus);
	if(cudaStatus!=cudaSuccess){
		status = -1;
		printf("\n kernelGetPointerdArrEmbedding failed");
		goto Error;
	}

	int vi,vj,backward;
	vi=vj=-1;
	backward=0;
	int noElemtemp = noElem;
	UniEdge *temp=(UniEdge*)malloc(sizeof(UniEdge)*noElemtemp);
	if(temp==NULL){
		printf("\n malloc temp failed");
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
		int li,lij,lj;
		li = temp[i].li;
		lij= temp[i].lij;
		lj=temp[i].lj;

		status = getvivj(dArrEXT,noElemdArrEXT,li,lij,lj,vi,vj);
		FUNCHECK(status);
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
			printf("\n This is minDFSCODE\n");

			int *hArrGraphId; //Mảng chứa các graphID có embedding của DFS_Code.
			int noElemhArrGraphId=0;
			status =getGraphIdContainEmbeddingv2(temp[i],hArrGraphId,noElemhArrGraphId,dArrEXT,noElemdArrEXT);
			if (status!=0){
				printf("\n\n getGraphIdContainEmbedding() in Mining() failed");
				goto Error;
			}
			////In nội dung mảng hArrGraphId
			printf("\n ************** hArrGraphId ****************\n");
			for (int j = 0; j < noElemhArrGraphId; j++)
			{
				printf("%d ",hArrGraphId[j]);
			}

			report(hArrGraphId,noElemhArrGraphId,hArrSupport[i]);

				//Xây dựng Embedding cho DFS_Code rồi gọi hàm GraphMining để khai thác
				//Trong GraphMining sẽ gọi GraphMining khác để thực hiện khai thác đệ quy

				FUNCHECK(status=extendEmbedding(temp[i],idxExt)); //Xây dựng 2 cột embedding ban đầu.
				if(status!=0){
					goto Error;
				}

				hLevelPtrEmbedding.resize(Level);
				hLevelPtrEmbedding.at(idxLevel).noElem=hEmbedding.size();
				int lastCol = hEmbedding.size()-1;
				hLevelPtrEmbedding.at(idxLevel).noElemEmbedding=hEmbedding.at(lastCol).noElem;
				CHECK(cudaStatus = cudaMalloc((void**)&hLevelPtrEmbedding.at(idxLevel).dArrPointerEmbedding,hLevelPtrEmbedding.at(idxLevel).noElem*sizeof(Embedding**))); //Cấp phát bộ nhớ cho mảng dArrPointerEmbedding.
				if(cudaStatus!=cudaSuccess){
					status = -1;
					std::printf("\n cudaMalloc dArrPointerEmbedding failed()");
					goto Error;
				}

				for (int i = 0; i < hEmbedding.size(); i++)
				{		
					kernelGetPointerdArrEmbedding<<<1,1>>>(hEmbedding.at(i).dArrEmbedding,hLevelPtrEmbedding.at(idxLevel).dArrPointerEmbedding,i); //Mỗi phần tử của mảng dArrPointerEmbedding chứa địa chỉ của dArrEmbedding
				}
				cudaDeviceSynchronize();
				cudaStatus = cudaGetLastError();
				CHECK(cudaStatus);
				if(cudaStatus!=cudaSuccess){
					status = -1;
					printf("\n kernelGetPointerdArrEmbedding failed");
					goto Error;
				}

				FUNCHECK(status = updateRMP()); //Xây dựng RMP ban đầu
				if(status!=0){
					goto Error;
				}

				FUNCHECK(FSMiningv2()); //Gọi FSMining.( Hàm này thực hiện theo tuần tự (1. Find Extension -> 2. Extract UniEdge -> 3.Compute & CHECK Support -> 4. CHECK minDFS_CODE -> 5. BuildEmbedding -> 6.Find RMP -> 1.)

			//Giải phóng bộ nhớ 
			std::free(hArrGraphId);

		}
		DFS_CODE.pop(); //Xoá phần tử cuối của DFS_CODE
		if(backward!=1){ //Nếu pop() một forward thì phải giảm maxId
			--maxId;
		}

		int lastCol = hRMPv2.at(idxLevel).hArrRMP[0];

		cudaFree(hEmbedding.at(lastCol).dArrEmbedding); //xoá phần tử cuối của Embedding
		hEmbedding.pop_back();

		cudaFree(hLevelPtrEmbedding.at(idxLevel).dArrPointerEmbedding); //xoá dArrPointerEmbedding tại Level đang xét
		hLevelPtrEmbedding.pop_back(); //Xoá phần tử cuối của Level pointerEmbeeding đang xét.

		hRMPv2.at(idxLevel).hArrRMP.clear(); //Xoá  RightMostPath của phần tử Embedding tại Level tương ứng.
		hRMPv2.pop_back();
	}	
	std::free(temp);
Error:
	return status;
}


__global__ void kernelMarkExtension(const Extension *d_ValidExtension,int noElem_d_ValidExtension,int *dV,int li,int lij,int lj){
	int i= blockIdx.x*blockDim.x + threadIdx.x;
	if(i<noElem_d_ValidExtension){
		if(d_ValidExtension[i].li==li && d_ValidExtension[i].lij==lij && d_ValidExtension[i].lj==lj){
			dV[i]=1;
		}		
	}
}

__global__ void kernelMarkEXT(const EXT *d_ValidExtension,int noElem_d_ValidExtension,int *dV,int li,int lij,int lj){
	int i= blockIdx.x*blockDim.x + threadIdx.x;
	if(i<noElem_d_ValidExtension){
		if(d_ValidExtension[i].li==li && d_ValidExtension[i].lij==lij && d_ValidExtension[i].lj==lj){
			dV[i]=1;
		}		
	}
}


__global__ void kernelSetValueForFirstTwoEmbeddingColumn(const Extension *d_ValidExtension,int noElem_d_ValidExtension,Embedding *dQ1,Embedding *dQ2,int *d_scanResult,int li,int lij,int lj){
	int i = blockDim.x *blockIdx.x +threadIdx.x;
	if(i<noElem_d_ValidExtension){
		if(d_ValidExtension[i].li==li && d_ValidExtension[i].lij == lij && d_ValidExtension[i].lj==lj){
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
		printf("\n Thread:%d address:%p (idx vid):(%d %d)",i,dArrEmbedding,dArrEmbedding[i].idx,dArrEmbedding[i].vid);
	}
}

__global__ void kernelSetValueForEmbeddingColumn(EXT *dArrExt,int noElemInArrExt,Embedding *dArrQ,int *dM,int *dMScanResult){
	int i = blockDim.x *blockIdx.x + threadIdx.x;
	if(i<noElemInArrExt){		
		if(dM[i]==1){
			
			int posRow = dArrExt[i].posRow;
			int vgj =dArrExt[i].vgj;
			dArrQ[dMScanResult[i]].idx=posRow;
			dArrQ[dMScanResult[i]].vid=vgj;
		}
	}
}


int PMS::extendEmbedding(UniEdge ue,int idxExt){
	
	int li,lij,lj;
	li=ue.li;
	lij=ue.lij;
	lj=ue.lj;
	int status =0;
	cudaError_t cudaStatus;

	int currentSize= hEmbedding.size();
	int newSize = currentSize+1;
	int lastEC =newSize-1; //lastEC is last Embedding Column or index of last element hEmbedding vector.

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

	displayDeviceEXT( hLevelEXT.at(idxLevel).vE.at(idxExt).dArrExt, hLevelEXT.at(idxLevel).vE.at(idxExt).noElem);

	dim3 block(blocksize);
	dim3 grid((noElemdV+block.x-1)/block.x);

	
	kernelMarkEXT<<<grid,block>>>(hLevelEXT.at(idxLevel).vE.at(idxExt).dArrExt,noElemdV,dV,li,lij,lj);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	CHECK(cudaStatus);
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
	myScanV(dV,noElemdV,dVScanResult);


	int noElemOfdArEmbedding=0;
	CHECK(getSizeBaseOnScanResult(dV,dVScanResult,noElemdV,noElemOfdArEmbedding));
	hEmbedding.at(lastEC).noElem=noElemOfdArEmbedding;

	CHECK(cudaStatus=cudaMalloc((void**)&hEmbedding.at(lastEC).dArrEmbedding,noElemOfdArEmbedding*sizeof(Embedding)));
	if(cudaStatus!=cudaSuccess){
		status=-1;
		goto Error;
	}
	kernelSetValueForEmbeddingColumn<<<grid,block>>>(hLevelEXT.at(idxLevel).vE.at(idxExt).dArrExt,hLevelEXT.at(idxLevel).vE.at(idxExt).noElem,hEmbedding.at(lastEC).dArrEmbedding,dV,dVScanResult);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	CHECK(cudaStatus);
	if(cudaStatus !=cudaSuccess){
		fprintf(stderr,"\n kernelSetValueForEmbeddingColumn in failed",cudaStatus);
		status = -1;
		goto Error;
	}

	hEmbedding.at(lastEC).prevCol=currentColEmbedding; 

	for (int i = 0; i < hEmbedding.size(); i++)
	{
		printf("\n\n Q[%d] prevCol:%d ",i,hEmbedding.at(i).prevCol);		
		kernelPrintEmbedding<<<1,512>>>(hEmbedding.at(i).dArrEmbedding,hEmbedding.at(i).noElem);
		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		CHECK(cudaStatus);
		if(cudaStatus!=cudaSuccess){
			status =-1;
			printf("kernelPrintEmbedding failed");
			goto Error;
		}
	}
Error:
	return status;
}

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
	int noElemdV = hValidExtension.at(0).noElem;
	CHECK(cudaMalloc((void**)&dV, sizeof(int)*noElemdV));
	CHECK(cudaMemset(dV,0,sizeof(int)*noElemdV));
	dim3 block(blocksize);
	dim3 grid((noElemdV+block.x-1)/block.x);

	//kernelPrintExtention<<<1,512>>>(hValidExtension.at(0).dExtension,hValidExtension.at(0).noElem);
	//cudaDeviceSynchronize();
	//CHECK(cudaGetLastError());
	//if(cudaGetLastError() !=cudaSuccess){
	//	printf("Error here");
	//	goto Error;
	//}

	kernelMarkExtension<<<grid,block>>>(hValidExtension.at(0).dExtension,noElemdV,dV,li,lij,lj);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	CHECK(cudaStatus);
	if(cudaStatus!=cudaSuccess){
		status = -1;
		fprintf(stderr,"\n kernelMarkExtension failed",cudaStatus);
		goto Error;
	}

	int* dVScanResult;
	CHECK(cudaMalloc((int**)&dVScanResult,noElemdV*sizeof(int)));
	CHECK(cudaMemset(dVScanResult,0,noElemdV*sizeof(int)));

	//CHECK(scanV(dV,noElemdV,dVScanResult));
	myScanV(dV,noElemdV,dVScanResult);


	int noElemOfdArEmbedding=0;
	CHECK(getSizeBaseOnScanResult(dV,dVScanResult,noElemdV,noElemOfdArEmbedding));
	hEmbedding.at(0).noElem=hEmbedding.at(1).noElem=noElemOfdArEmbedding;

	CHECK(cudaMalloc((void**)&hEmbedding.at(0).dArrEmbedding,noElemOfdArEmbedding*sizeof(Embedding)));
	CHECK(cudaMalloc((void**)&hEmbedding.at(1).dArrEmbedding,noElemOfdArEmbedding*sizeof(Embedding)));


	kernelSetValueForFirstTwoEmbeddingColumn<<<grid,block>>>(hValidExtension.at(0).dExtension,hValidExtension.at(0).noElem,hEmbedding.at(0).dArrEmbedding,hEmbedding.at(1).dArrEmbedding,dVScanResult,li,lij,lj);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	CHECK(cudaStatus);
	if(cudaStatus !=cudaSuccess){
		fprintf(stderr,"\n kernelSetValueForFirstTwoEmbeddingColumn in failed",cudaStatus);
		status = -1;
		goto Error;
	}

	hEmbedding.at(0).prevCol=-1;
	hEmbedding.at(1).prevCol=0;

	for (int i = 0; i < hEmbedding.size(); i++)
	{
		printf("\n\n Q[%d] prevCol:%d ",i,hEmbedding.at(i).prevCol);		
		kernelPrintEmbedding<<<1,512>>>(hEmbedding.at(i).dArrEmbedding,hEmbedding.at(i).noElem);
		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		CHECK(cudaStatus);
		if(cudaStatus!=cudaSuccess){
			status =-1;
			printf("kernelPrintEmbedding failed");
			goto Error;
		}
	}


Error:
	return status;
}

//Why do this snippet face the error: Invalid device pointer

//void PMS::displayEmbeddingColumn(EmbeddingColumn ec){
//	printf("\n noElem:%d prevCol:%d",ec.noElem,ec.prevCol);
//	
//	Embedding *hArrEmbeddingt = (Embedding*)malloc(sizeof(Embedding)*ec.noElem);
//	if(hArrEmbeddingt==NULL){
//		printf("\n malloc hArrEmbeddingt in displayEmbeddingColumn() failed");
//		exit(1);
//	}
//
//	CHECK(cudaMemcpy(hArrEmbeddingt,ec.dArrEmbedding,sizeof(Embedding)*ec.noElem,cudaMemcpyDeviceToHost));
//	for (int i = 0; i < ec.noElem; i++)
//	{
//		printf("\n A[%d]: (idx, vid):(%d, %d)",i,hArrEmbeddingt[i].idx,hArrEmbeddingt[i].vid);
//	}
//
//	cudaFree(hArrEmbeddingt);
//}

__global__ void kernelFindVidOnRMP(Embedding **dArrPointerEmbedding,int noElemEmbedding,int *rmp,int noElemVerOnRMP,int *dArrVidOnRMP,int step){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElemEmbedding){	
		int index;
		int start=0;
		int prevRow = i;
		int end=0;
		Embedding * dArrEmbedding;
		for (int k = 0; k < noElemVerOnRMP; )
		{
			index = i*step+k;
			int j;
			start = rmp[k];
			k++;
			if(k==noElemVerOnRMP) break;
			end = rmp[k];
			//Từ cột start sẽ trích ra được vid và prevRow;
			for (j = start; j >end; j--)
			{
				dArrEmbedding = dArrPointerEmbedding[j];
				prevRow= dArrEmbedding[prevRow].idx; //update row
			}
			dArrEmbedding = dArrPointerEmbedding[j];
			dArrVidOnRMP[index]=dArrEmbedding[prevRow].vid;
			prevRow= dArrEmbedding[prevRow].idx; //update row
			printf("\n thread:%d start:%d end:%d index:%d vid:%d",i,start,end,index,dArrVidOnRMP[index]);
		}

	}
}


__global__ void kernelFindListVer(Embedding **dArrPointerEmbedding,int noElemEmbedding,int *rmp,int noElemVerOnRMP,int *listVer){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElemEmbedding){	
		int j =0;
		int posCol = rmp[j]; //bắt đầu từ cột cuối của Embedding
		int posRow = i;
		Embedding *dArrEmbedding = dArrPointerEmbedding[posCol];
		int idxListVer = j*noElemEmbedding + i;
		listVer[idxListVer] = dArrEmbedding[posRow].vid; //Trích vid lưu vào mảng listVer tại vị trí tương ứng.
loop:
		j=j+1; //tăng chỉ số j của rmp
		if(j==noElemVerOnRMP) return;
		int loopTimes = posCol - rmp[j];
		for (int k = 0; k < loopTimes; k++)
		{
			posRow = dArrEmbedding[posRow].idx;
			posCol = posCol-1;
			dArrEmbedding = dArrPointerEmbedding[posCol];
			//printf("\nThread %d j:%d k:%d posCol:%d posRow:%d",i,j,k,posCol,posRow);
		}
		idxListVer = j*noElemEmbedding + i;
		listVer[idxListVer] = dArrEmbedding[posRow].vid; //Trích vid lưu vào mảng listVer tại vị trí tương ứng.
		//printf("\n Thread %d j:%d vid:%d idxListVer:%d posCol:%e posRow:%d",i,j,dArrEmbedding[posRow].vid, idxListVer,posCol,posRow);
		goto loop;
	}
}

__global__ void kernelDisplaydArrPointerEmbedding(Embedding **dArrPointerEmbedding,int noElemEmbeddingCol,int noElemEmbedding){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElemEmbedding){
		int lastCol = noElemEmbeddingCol-1;
		Embedding *dArrEmbedding;
		//printf("\n Last Embedding column:%d Element:%d (idx vid):(%d %d)",lastCol,i,dArrEmbedding[i].idx,dArrEmbedding[i].vid);
		int prevRow=i;
		for (int j = lastCol; j>=0; j--)
		{
			dArrEmbedding= dArrPointerEmbedding[j];
			printf("\n Last Embedding column:%d Element:%d (idx vid):(%d %d)",lastCol,i,dArrEmbedding[prevRow].idx,dArrEmbedding[prevRow].vid);
			prevRow=dArrEmbedding[prevRow].idx;
		}
	}
}


int PMS::displaydArrPointerEmbedding(Embedding **dArrPointerEmbedding,int noElemEmbeddingCol,int noElemEmbedding){
	int status =0;
	cudaError_t cudaStatus;
	dim3 block(blocksize);
	dim3 grid((noElemEmbedding + block.x - 1)/block.x);
	printf("\n************ Embedding dArrPointerEmbedding ************\n");
	kernelDisplaydArrPointerEmbedding<<<grid,block>>>(dArrPointerEmbedding,noElemEmbeddingCol,noElemEmbedding);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	CHECK(cudaStatus);
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}
Error:
	return status;
}



int PMS::buildRMP(){
	int status = 0;
	hRMP.resize(1);
	int noElem = hEmbedding.size();
	int index = noElem - 1;

	for (int i = index ; i != -1 ; i = hEmbedding.at(i).prevCol)
	{
		hRMP.at(0).hArrRMP.push_back(i);		
	}
	hRMP.at(0).noElem = 2;
Error:
	return status;
}
//Hàm này thực hiện theo tuần tự (1. Find Extension -> 2. Extract UniEdge -> 3.Compute & CHECK Support -> 4. CHECK minDFS_CODE -> 5. BuildEmbedding -> 6.Find RMP -> Miningv2
int PMS::FSMining()
{
	int status = 0;
	cudaError_t cudaStatus;

	//Thiết lập điều kiện dừng (return) khi không tồn tại mở rộng
	/* somethings code here */

	Level++;
	idxLevel=Level-1;
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
	hLevelPtrEmbedding.resize(Level);
	hLevelPtrEmbedding.at(idxLevel).noElem=hEmbedding.size();
#pragma region "build dArrPointerEmbedding on device"
	cudaStatus = cudaMalloc((void**)&hLevelPtrEmbedding.at(idxLevel).dArrPointerEmbedding,hEmbedding.size()*sizeof(Embedding**)); //Cấp phát bộ nhớ cho mảng dArrPointerEmbedding tại Level tương ứng.
	CHECK(cudaStatus);
	if(cudaStatus!=cudaSuccess){
		status = -1;
		std::printf("\n cudaMalloc dArrPointerEmbedding failed()");
		goto Error;
	}

	for (int i = 0; i < hEmbedding.size(); i++)
	{		
		kernelGetPointerdArrEmbedding<<<1,1>>>(hEmbedding.at(i).dArrEmbedding,hLevelPtrEmbedding.at(idxLevel).dArrPointerEmbedding,i); //Mỗi phần tử của mảng dArrPointerEmbedding chứa địa chỉ của dArrEmbedding
	}
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	CHECK(cudaStatus);
	if(cudaStatus!=cudaSuccess){
		status = -1;
		printf("\n kernelGetPointerdArrEmbedding failed");
		goto Error;
	}
#pragma endregion 


#pragma region "cudaMalloc for listVer to find listVer On All EmbeddingColumn that belong to RMP"

	//Tìm danh sách các đỉnh thuộc right most path của các embedding
	//Kết quả lưu vào các vector tương ứng
	int lastCol = hEmbedding.size() - 1; //cột cuối của embedding
	hLevelPtrEmbedding.at(idxLevel).noElemEmbedding= hEmbedding.at(lastCol).noElem; //số lượng embedding	
	int noElemListVer= hRMP.at(0).noElem * hLevelPtrEmbedding.at(idxLevel).noElemEmbedding; //số lượng phần tử của listVer bằng số lượng đỉnh trên right most path nhân với số lượng embedding
	hListVer.resize(Level);
	hListVer.at(idxLevel).noElem=noElemListVer;
	CHECK(cudaStatus = cudaMalloc((void**)&hListVer.at(idxLevel).dListVer,sizeof(int)*noElemListVer)); //cấp phát bộ nhớ cho listVer
	if(cudaStatus!=cudaSuccess){
		printf("\n CudaMalloc dListVer failed");
		status =-1;
		goto Error;
	}

#pragma endregion

	FUNCHECK(status=displaydArrPointerEmbedding(hLevelPtrEmbedding.at(idxLevel).dArrPointerEmbedding,hLevelPtrEmbedding.at(idxLevel).noElem,hLevelPtrEmbedding.at(idxLevel).noElemEmbedding));
	if(status!=0){
		goto Error;
	}


#pragma region "build RMP on device"

	//Xây dựng right most path từ vector<int> hRMP
	int noElemVerOnRMP = hRMP.at(0).noElem; //right most path chứa bao nhiêu đỉnh
	int *rmp = nullptr; //rigt most path trên bộ nhớ device
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
		temp[i] = hRMP.at(0).hArrRMP.at(i);
	}
	//Chép dữ liệu từ temp trên host sang rmp trên device
	CHECK(cudaStatus =cudaMemcpy(rmp,temp,sizeof(int)*noElemVerOnRMP,cudaMemcpyHostToDevice)); //chép dữ liệu từ temp ở host sang rmp trên device
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}

	std::free(temp);

	printf("\n\n ******* rmp *********\n");
	displayDeviceArr(rmp,noElemVerOnRMP);

#pragma endregion

#pragma region "find listVer from All EmbeddingColumn"

	//Tìm danh sách các đỉnh thuộc right most path ở các cột embedding để thực hiện mở rộng
	dim3 block(blocksize);
	dim3 grid((hLevelPtrEmbedding.at(idxLevel).noElemEmbedding + block.x -1)/block.x);
	
	kernelFindListVer<<<block,grid>>>(hLevelPtrEmbedding.at(idxLevel).dArrPointerEmbedding,hLevelPtrEmbedding.at(idxLevel).noElemEmbedding,rmp,noElemVerOnRMP,hListVer.at(idxLevel).dListVer); //tìm listVer
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		status =-1;
		CHECK(cudaStatus);
		goto Error;
	}
	//hiển thị danh sách đỉnh
	printf("\n\n ********* listVer *********\n");
	displayDeviceArr(hListVer.at(idxLevel).dListVer,noElemListVer);

#pragma endregion

#pragma region "find vertices (global id of vertext) belong to RMP of embedding to find backward Extension"

	//dArrVidOnRMP: chứa các đỉnh thuộc RMP của mỗi Embedding. Dùng để kiểm tra sự tồn tại của đỉnh trên right most path
	//khi tìm các mở rộng backward. Chỉ dùng tới khi right most path có 3 đỉnh trở lên (chỉ xét trong đơn đồ thị vô hướng).
	int *dArrVidOnRMP = nullptr; //lưu trữ các đỉnh trên RMP của Embedding, có kích thước nhỏ hơn 2 đỉnh so với RMP
	int noElemdArrVidOnRMP= hRMP.at(0).noElem - 2;
	//int *fromPosCol=nullptr; //lưu trữ các cột của Embedding mà tại đó thuộc right most path. Thật ra mình có thể suy luận được từ rmp

	if (noElemdArrVidOnRMP >0){
		cudaStatus = cudaMalloc((void**)&dArrVidOnRMP,hLevelPtrEmbedding.at(idxLevel).noElemEmbedding*noElemdArrVidOnRMP*sizeof(int));
		CHECK(cudaStatus);
		if(cudaStatus!=cudaSuccess){
			status =-1;
			goto Error;
		}
		//cudaStatus = cudaMalloc((void**)&fromPosCol,hLevelPtrEmbedding.at(idxLevel).noElemEmbedding*noElemdArrVidOnRMP*sizeof(int));
		//CHECK(cudaStatus);
		//if(cudaStatus!=cudaSuccess){
		//	status =-1;
		//	goto Error;
		//}
	}

	if(hRMP.at(0).noElem>2){ //Nếu số lượng đỉnh trên RMP lớn hơn 2 thì mới tồn tại backward. Vì ở đây chỉ xét đơn đồ thị vô hướng
		
		FUNCHECK(status = displaydArrPointerEmbedding(hLevelPtrEmbedding.at(idxLevel).dArrPointerEmbedding,hLevelPtrEmbedding.at(idxLevel).noElem,hLevelPtrEmbedding.at(idxLevel).noElemEmbedding));
		if(status!=0){
			goto Error;
		}
		kernelFindVidOnRMP<<<grid,block>>>(hLevelPtrEmbedding.at(idxLevel).dArrPointerEmbedding,hLevelPtrEmbedding.at(idxLevel).noElemEmbedding,rmp,noElemVerOnRMP,dArrVidOnRMP,noElemdArrVidOnRMP);
		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		CHECK(cudaStatus);
		if(cudaStatus!=cudaSuccess){
			status =-1;
			goto Error;
		}

		printf("\n ******** dArrVidOnRMP *******\n");
		FUNCHECK(status = displayDeviceArr(dArrVidOnRMP,noElemdArrVidOnRMP*hLevelPtrEmbedding.at(idxLevel).noElemEmbedding));
		if(status!=0){
			goto Error;
		}

		//printf("\n ******** fromPosCol *******\n");
		//displayDeviceArr(fromPosCol,noElemdArrVidOnRMP);


	}

#pragma endregion

	int noElemEXTk =noElemVerOnRMP; //Số lượng phần tử EXTk bằng số lượng đỉnh trên right most path
	//hEXTk.resize(noElemEXTk); //Các mở rộng hợp lệ từ đỉnh k sẽ được lưu trữ vào EXTk tương ứng.

	//Quản lý theo Level phục vụ cho khai thác đệ quy
	hLevelEXT.resize(Level); //Khởi tạo vector quản lý bộ nhớ cho level
	hLevelEXT.at(idxLevel).noElem = noElemVerOnRMP; //Cập nhật số lượng phần tử vector vE bằng số lượng đỉnh trên RMP
	hLevelEXT.at(idxLevel).vE.resize(noElemVerOnRMP); //Cấp phát bộ nhớ cho vector vE.

	hLevelUniEdge.resize(Level); //Số lượng phần tử UniEdge cũng giống với EXT
	hLevelUniEdge.at(idxLevel).noElem=noElemVerOnRMP;
	hLevelUniEdge.at(idxLevel).vUE.resize(noElemVerOnRMP);

	hLevelUniEdgeSatisfyMinsup.resize(Level);
	hLevelUniEdgeSatisfyMinsup.at(idxLevel).noElem= noElemVerOnRMP;
	hLevelUniEdgeSatisfyMinsup.at(idxLevel).vecUES.resize(noElemVerOnRMP);

	int *tempListVerCol = nullptr; //chứa danh sách các đỉnh cần mở rộng thuộc một embedding column cụ thể.
	CHECK(cudaStatus = cudaMalloc((void**)&tempListVerCol,hLevelPtrEmbedding.at(idxLevel).noElemEmbedding * sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	//Nếu số lượng phần tử đỉnh thuộc RMP bằng 2 thì không tồn tại mở rộng backward, nên chúng ta
	if(noElemVerOnRMP == 2){//chỉ tìm các ở rộng forward từ tập đỉnh và lưu kết quả của các mở rộng vào EXTk tương ứng.
		for (int i = 0; i < noElemVerOnRMP ; i++)
		{
			int colEmbedding = hRMP.at(0).hArrRMP.at(i); //Tìm mở rộng cho các đỉnh tại vị trí colEmbedding trong vector hEmbedding
			currentColEmbedding=colEmbedding; //Đang mở rộng từ cột nào của Embedding. Được dùng để cập nhật prevCol, phục vụ cho việc xây dựng Right Most Path
			int k = i; //lưu vào Extk với k = i; K=0 đại diện cho EXT0: là EXT cuối
			kernelExtractFromListVer<<<grid,block>>>(hListVer.at(idxLevel).dListVer,i*hLevelPtrEmbedding.at(idxLevel).noElemEmbedding,hLevelPtrEmbedding.at(idxLevel).noElemEmbedding,tempListVerCol);//trích các đỉnh từ listVer, từ vị trí i*noElemEmbedding,trích noElemEmbedding phần tử, bỏ vào tempListVerCol
			cudaDeviceSynchronize();
			cudaStatus = cudaGetLastError();
			CHECK(cudaStatus);
			printf("\n ****** tempListVerCol ***********\n");
			displayDeviceArr(tempListVerCol,hLevelPtrEmbedding.at(idxLevel).noElemEmbedding);


			//gọi hàm forwardExtension để tìm các mở rộng forward từ cột colEmbedding, lưu kết quả vào hEXTk tại vị trí k, với các đỉnh
			// cần mở rộng là tempListVerCol, thuộc righ most path
			////Hàm này cũng đồng thời trích các mở rộng duy nhất từ các EXT và lưu vào UniEdge
			//Hàm này cũng gọi đệ quy FSMining bên trong
			FUNCHECK(status = forwardExtension(k,tempListVerCol,hLevelPtrEmbedding.at(idxLevel).noElemEmbedding,hRMP.at(0).hArrRMP.at(i)));
			if(status ==-1){
				goto Error;
			}


			//FSMining();
			if(hLevelEXT.at(idxLevel).vE.at(i).noElem>0){ //Nếu số lượng phần tử của mảng dArrExt = 0 thì chúng ta không giải phóng bộ nhớ dArrExt vì nó chưa được cấp phát.
				cudaFree(hLevelEXT.at(idxLevel).vE.at(i).dArrExt);
				cudaFree(hLevelUniEdge.at(idxLevel).vUE.at(i).dArrUniEdge);
			}			
		}
		hLevelEXT.at(idxLevel).vE.clear();
		hLevelUniEdge.at(idxLevel).vUE.clear();
		Level--;
	}
	//Nếu số lượng đỉnh thuộc RMP nhiều hơn 2 thì sẽ tồn tại mở rộng backward
	if (noElemVerOnRMP > 2){
		//1. khai thác backward và forward của đỉnh cuối cùng trước
		for (int i = 1; i < noElemVerOnRMP-1; i++)
		{
			//2. sau đó khai thác forward cho các đỉnh còn lại
			//kernelFindValidFBExtension(dArrPointerEmbedding,hEmbedding.size(),noElemEmbedding,hdb.at(0).dO,hdb.at(0).dLO,hdb.at(0).dN,hdb.at(0).dLN,dArrDegreeOfVid,maxDegreeOfVer,dArrV,dArrExtension,listOfVer,minLabel,maxId,fromRMP,dArrVidOnRMP,noElemdArrVidOnRMP,fromPosCol);
		}

	}

	cudaFree(tempListVerCol);
	cudaFree(hListVer.at(idxLevel).dListVer);
Error:
	return status;
}
int PMS::FSMiningv2() //đã có Embedding mới và RMP tương ứng với nó. Khai thác các mở rộng
{
	int status = 0;
	cudaError_t cudaStatus;

	Level++;
	idxLevel=Level-1;

#pragma region "cudaMalloc for listVer to find listVer On All EmbeddingColumn that belong to RMP"

	//Tìm danh sách các đỉnh thuộc right most path của các embedding
	//Kết quả lưu vào các vector tương ứng
	int lastCol = hEmbedding.size() - 1; //cột cuối của embedding
	hLevelPtrEmbedding.resize(Level);
	hLevelPtrEmbedding.at(idxLevel).noElem=hEmbedding.size();
	hLevelPtrEmbedding.at(idxLevel).noElemEmbedding= hEmbedding.at(lastCol).noElem; //số lượng embedding
	cudaStatus = cudaMalloc((void**)&hLevelPtrEmbedding.at(idxLevel).dArrPointerEmbedding,hEmbedding.size()*sizeof(Embedding**)); //Cấp phát bộ nhớ cho mảng dArrPointerEmbedding tại Level tương ứng.
	CHECK(cudaStatus);
	if(cudaStatus!=cudaSuccess){
		status = -1;
		std::printf("\n cudaMalloc dArrPointerEmbedding failed()");
		goto Error;
	}

	for (int i = 0; i < hEmbedding.size(); i++)
	{		
		kernelGetPointerdArrEmbedding<<<1,1>>>(hEmbedding.at(i).dArrEmbedding,hLevelPtrEmbedding.at(idxLevel).dArrPointerEmbedding,i); //Mỗi phần tử của mảng dArrPointerEmbedding chứa địa chỉ của dArrEmbedding
	}
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	CHECK(cudaStatus);
	if(cudaStatus!=cudaSuccess){
		status = -1;
		printf("\n kernelGetPointerdArrEmbedding failed");
		goto Error;
	}


	int noElemListVer= hRMPv2.at(idxLevel-1).noElem * hLevelPtrEmbedding.at(idxLevel).noElemEmbedding; //số lượng phần tử của listVer bằng số lượng đỉnh trên right most path nhân với số lượng embedding
	hListVer.resize(Level);
	hListVer.at(idxLevel).noElem=noElemListVer;
	
	CHECK(cudaStatus = cudaMalloc((void**)&hListVer.at(idxLevel).dListVer,sizeof(int)*noElemListVer)); //cấp phát bộ nhớ cho listVer
	if(cudaStatus!=cudaSuccess){
		printf("\n CudaMalloc listVer failed");
		status =-1;
		goto Error;
	}

#pragma endregion


#pragma region "Copy hRMPv2 to device rmp"

	int noElemVerOnRMP = hRMPv2.at(idxLevel-1).noElem; //right most path chứa bao nhiêu đỉnh
	int *rmp = nullptr; //rigt most path trên bộ nhớ device
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
		temp[i] = hRMPv2.at(idxLevel-1).hArrRMP.at(i);
	}
	//Chép dữ liệu từ temp trên host sang rmp trên device
	CHECK(cudaStatus =cudaMemcpy(rmp,temp,sizeof(int)*noElemVerOnRMP,cudaMemcpyHostToDevice)); //chép dữ liệu từ temp ở host sang rmp trên device
	if(cudaStatus!=cudaSuccess){
		status =-1;
		goto Error;
	}

	std::free(temp);

	printf("\n\n ******* rmp *********\n");
	displayDeviceArr(rmp,noElemVerOnRMP);

#pragma endregion

#pragma region "find listVer from All EmbeddingColumn"

	//Tìm danh sách các đỉnh thuộc right most path ở các cột embedding để thực hiện mở rộng
	dim3 block(blocksize);
	dim3 grid((hLevelPtrEmbedding.at(idxLevel).noElemEmbedding + block.x -1)/block.x);
	kernelFindListVer<<<block,grid>>>(hLevelPtrEmbedding.at(idxLevel).dArrPointerEmbedding,hLevelPtrEmbedding.at(idxLevel).noElemEmbedding,rmp,noElemVerOnRMP,hListVer.at(idxLevel).dListVer); //tìm listVer
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		status =-1;
		CHECK(cudaStatus);
		goto Error;
	}
	//hiển thị danh sách đỉnh
	printf("\n\n ********* listVer *********\n");
	displayDeviceArr(hListVer.at(idxLevel).dListVer,noElemListVer);

#pragma endregion

#pragma region "find vertices (global id of vertext) belong to RMP of embedding to find backward Extension"

	//dArrVidOnRMP: chứa các đỉnh thuộc RMP của mỗi Embedding. Dùng để kiểm tra sự tồn tại của đỉnh trên right most path
	//khi tìm các mở rộng backward. Chỉ dùng tới khi right most path có 3 đỉnh trở lên (chỉ xét trong đơn đồ thị vô hướng).
	int *dArrVidOnRMP = nullptr; //lưu trữ các đỉnh trên RMP của Embedding, có kích thước nhỏ hơn 2 đỉnh so với RMP
	int noElemdArrVidOnRMP= hRMPv2.at(idxLevel-1).noElem - 1;
	//int *fromPosCol=nullptr; //lưu trữ các cột của Embedding mà tại đó thuộc right most path. Thật ra mình có thể suy luận được từ rmp

	if (noElemdArrVidOnRMP >0){
		cudaStatus = cudaMalloc((void**)&dArrVidOnRMP,hLevelPtrEmbedding.at(idxLevel).noElemEmbedding*noElemdArrVidOnRMP*sizeof(int));
		CHECK(cudaStatus);
		if(cudaStatus!=cudaSuccess){
			status =-1;
			goto Error;
		}
		//cudaStatus = cudaMalloc((void**)&fromPosCol,hLevelPtrEmbedding.at(idxLevel).noElemEmbedding*noElemdArrVidOnRMP*sizeof(int));
		//CHECK(cudaStatus);
		//if(cudaStatus!=cudaSuccess){
		//	status =-1;
		//	goto Error;
		//}
	}
//here is problem. Cần kiểm tra lại hàm tính dArrVidOnRMP
	if(hRMPv2.at(idxLevel-1).noElem>2){ //Nếu số lượng đỉnh trên RMP lớn hơn 2 thì mới tồn tại backward. Vì ở đây chỉ xét đơn đồ thị vô hướng
		FUNCHECK(status = displaydArrPointerEmbedding(hLevelPtrEmbedding.at(idxLevel).dArrPointerEmbedding,hLevelPtrEmbedding.at(idxLevel).noElem,hLevelPtrEmbedding.at(idxLevel).noElemEmbedding));
		if(status!=0){
			goto Error;
		}

		//status = displayDeviceArr(rmp,hRMPv2.at(idxLevel-1).noElem);
		//FUNCHECK(status);
		//if(status!=0){
		//	goto Error;
		//}

		//Hàm này tìm các vid thuộc right most path và lưu vào mảng dArrVidOnRMP. Mảng này dùng để tìm các valid backward edge.
		kernelFindVidOnRMP<<<grid,block>>>(hLevelPtrEmbedding.at(idxLevel).dArrPointerEmbedding,hLevelPtrEmbedding.at(idxLevel).noElemEmbedding,rmp,noElemVerOnRMP,dArrVidOnRMP,noElemdArrVidOnRMP);
		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		CHECK(cudaStatus);
		if(cudaStatus!=cudaSuccess){
			status =-1;
			goto Error;
		}

		printf("\n ******** dArrVidOnRMP *******\n");
		displayDeviceArr(dArrVidOnRMP,noElemdArrVidOnRMP*hLevelPtrEmbedding.at(idxLevel).noElemEmbedding);
		//printf("\n ******** fromPosCol *******\n");
		//displayDeviceArr(fromPosCol,noElemdArrVidOnRMP);

		//Viết hàm khai thác các valid forward edge và valid backward edge
		//Nhưng lưu kết quả vào đâu?
		//==>Những mở rộng hợp lệ đều được lưu vào EXTk và backward edge chỉ tồn tại ở đỉnh mở rộng cuối cùng.

	}

#pragma endregion

	int noElemEXTk =noElemVerOnRMP; //Số lượng phần tử EXTk bằng số lượng đỉnh trên right most path
	//hEXTk.resize(noElemEXTk); //Các mở rộng hợp lệ từ đỉnh k sẽ được lưu trữ vào EXTk tương ứng.

	//Quản lý theo Level phục vụ cho khai thác đệ quy
	hLevelEXT.resize(Level); //Khởi tạo vector quản lý bộ nhớ cho level
	hLevelEXT.at(idxLevel).noElem = noElemVerOnRMP; //Cập nhật số lượng phần tử vector vE bằng số lượng đỉnh trên RMP
	hLevelEXT.at(idxLevel).vE.resize(noElemVerOnRMP); //Cấp phát bộ nhớ cho vector vE.

	hLevelUniEdge.resize(Level); //Số lượng phần tử UniEdge cũng giống với EXT
	hLevelUniEdge.at(idxLevel).noElem=noElemVerOnRMP;
	hLevelUniEdge.at(idxLevel).vUE.resize(noElemVerOnRMP);

	hLevelUniEdgeSatisfyMinsup.resize(Level);
	hLevelUniEdgeSatisfyMinsup.at(idxLevel).noElem= noElemVerOnRMP;
	hLevelUniEdgeSatisfyMinsup.at(idxLevel).vecUES.resize(noElemVerOnRMP);

	int *tempListVerCol = nullptr; //chứa danh sách các đỉnh cần mở rộng thuộc một embedding column cụ thể.
	CHECK(cudaStatus = cudaMalloc((void**)&tempListVerCol,hLevelPtrEmbedding.at(idxLevel).noElemEmbedding * sizeof(int)));
	if(cudaStatus!=cudaSuccess){
		status = -1;
		goto Error;
	}
	//Nếu số lượng phần tử đỉnh thuộc RMP bằng 2 thì không tồn tại mở rộng backward, nên chúng ta
	if(noElemVerOnRMP == 2){//chỉ tìm các ở rộng forward từ tập đỉnh và lưu kết quả của các mở rộng vào EXTk tương ứng.
		for (int i = 0; i < noElemVerOnRMP ; i++)
		{
			int colEmbedding = hRMPv2.at(idxLevel-1).hArrRMP.at(i); //Tìm mở rộng cho các đỉnh tại vị trí colEmbedding trong vector hEmbedding
			currentColEmbedding=colEmbedding; //Đang mở rộng từ cột nào của Embedding. Được dùng để cập nhật prevCol, phục vụ cho việc xây dựng Right Most Path
			int k = i; //lưu vào Extk với k = i; K=0 đại diện cho EXT0: là EXT cuối
			kernelExtractFromListVer<<<grid,block>>>(hListVer.at(idxLevel).dListVer,i*hLevelPtrEmbedding.at(idxLevel).noElemEmbedding,hLevelPtrEmbedding.at(idxLevel).noElemEmbedding,tempListVerCol);//trích các đỉnh từ listVer, từ vị trí i*noElemEmbedding,trích noElemEmbedding phần tử, bỏ vào tempListVerCol
			cudaDeviceSynchronize();
			cudaStatus = cudaGetLastError();
			CHECK(cudaStatus);
			printf("\n ****** tempListVerCol ***********\n");
			displayDeviceArr(tempListVerCol,hLevelPtrEmbedding.at(idxLevel).noElemEmbedding);


			//gọi hàm forwardExtension để tìm các mở rộng forward từ cột colEmbedding, lưu kết quả vào hEXTk tại vị trí k, với các đỉnh
			// cần mở rộng là tempListVerCol, thuộc righ most path
			////Hàm này cũng đồng thời trích các mở rộng duy nhất từ các EXT và lưu vào UniEdge
			//Hàm này cũng gọi đệ quy FSMining bên trong
			FUNCHECK(status = forwardExtension(k,tempListVerCol,hLevelPtrEmbedding.at(idxLevel).noElemEmbedding,hRMPv2.at(idxLevel-1).hArrRMP.at(i)));
			if(status ==-1){
				goto Error;
			}

			//FSMining();
			if(hLevelEXT.at(idxLevel).vE.at(i).noElem>0){ //Nếu số lượng phần tử của mảng dArrExt = 0 thì chúng ta không giải phóng bộ nhớ dArrExt vì nó chưa được cấp phát.
				cudaFree(hLevelEXT.at(idxLevel).vE.at(i).dArrExt);
				cudaFree(hLevelUniEdge.at(idxLevel).vUE.at(i).dArrUniEdge);
			}			
		}
		hLevelEXT.at(idxLevel).vE.clear();
		hLevelUniEdge.at(idxLevel).vUE.clear();
		
	}
	//Nếu số lượng đỉnh thuộc RMP nhiều hơn 2 thì sẽ tồn tại mở rộng backward
	if (noElemVerOnRMP > 2){
		//1. khai thác backward và forward của đỉnh cuối cùng trước
		for (int i = 1; i < noElemVerOnRMP-1; i++)
		{
			//2. sau đó khai thác forward cho các đỉnh còn lại
			//Do somthings
		}

	}

	cudaFree(tempListVerCol);
	cudaFree(hListVer.at(idxLevel).dListVer);
	--Level;
	idxLevel=Level-1;	
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
			//printf("\nThread %d, j: %d has ToVidLabel:%d",i,j,labelToVid);
			//1. Trước tiên kiểm tra nhãn của labelToVid có nhỏ hơn minLabel hay không. Nếu nhỏ hơn thì return
			if(labelToVid<minLabel) continue;
			//2. kiểm tra xem đỉnh toVid có tồn tại trong embedding hay không nếu tồn tại thì return
			//Duyệt qua embedding column từ Q cuối đến Q đầu, lần lượt lấy vid so sánh với toVid

			//printf("\n Q[%d] Row[%d] (idx:%d vid:%d)",posColumn,posRow,Q[posRow].idx,Q[posRow].vid);//Q[1][0]
			if(toVid==Q[posRow].vid) continue;
			//printf("\nj:%d toVid:%d Q.vid:%d",j,toVid,Q[posRow].vid);

			while (true)
			{
				posRow = Q[posRow].idx;//0
				posColumn=posColumn-1;		//0
				Q=dArrPointerEmbedding[posColumn];
				//printf("\n posColumn[%d] Row[%d] (idx:%d vid:%d)",posColumn,posRow,Q[posRow].idx,Q[posRow].vid);//Q[0][0]
				//printf("\nj:%d toVid:%d Q.vid:%d",j,toVid,Q[posRow].vid);
				if(toVid==Q[posRow].vid) {
					b=false; break;
				}
				posRow=Q[posRow].idx;//-1
				//printf("\nposRow:%d",posRow);
				if(posRow==-1) break;
			}
			if (b==false){b=true; continue;}
			int indexOfd_arr_V=i*maxDegreeOfVer+j;
			//printf("\nThread %d: m:%d",i,maxDegreeOfVer);
			int indexOfd_LN=indexToVidIndN;
			//dArrV[indexOfd_arr_V].valid=1;
			dArrV_valid[indexOfd_arr_V]=1;
			dArrV_backward[indexOfd_arr_V]=0;
			//printf("\ndArrV[%d].valid:%d",indexOfd_arr_V,dArrV[indexOfd_arr_V].valid);
			//cập nhật dữ liệu cho mảng dArrExtension
			dArrExtension[indexOfd_arr_V].vgi=vid;
			dArrExtension[indexOfd_arr_V].vgj=toVid;
			dArrExtension[indexOfd_arr_V].lij=d_LN[indexOfd_LN];
			//printf("\n");
			//printf("d_LN[%d]:%d ",indexOfd_LN,d_LN[indexOfd_LN]);
			dArrExtension[indexOfd_arr_V].li=labelFromVid;
			dArrExtension[indexOfd_arr_V].lj=labelToVid;
			dArrExtension[indexOfd_arr_V].vi=fromRMP;
			dArrExtension[indexOfd_arr_V].vj=maxId+1;
			//dArrExtension[indexOfd_arr_V].posColumn=idxQ;
			dArrExtension[indexOfd_arr_V].posRow=row;
		}
	}
}


//kernel tìm các mở rộng hợp lệ và ghi nhận vào mảng dArrV và dArrExtension tương ứng.
//__global__ void kernelFindValidFBExtension(Embedding **dArrPointerEmbedding,int noElem_dArrPointerEmbedding,int noElem_Embedding,int *d_O,int *d_LO,int *d_N,int *d_LN,float *dArrDegreeOfVid,int maxDegreeOfVer,V *dArrV,EXT *dArrExtension,int *listOfVer,int minLabel,int maxId,int fromRMP, int *dArrVidOnRMP,int segdArrVidOnRMP,int *fromPosCol){
//	int i = blockDim.x * blockIdx.x + threadIdx.x;
//	//Duyệt qua các Embedding và xét các mở rộng cho đỉnh tại vị trí idxQ
//	if(i<noElem_Embedding){
//		int posColumn =noElem_dArrPointerEmbedding-1;
//		int posRow=i;
//		int col = posColumn;
//		int row = posRow;
//		//Embedding *Q=dArrPointerEmbedding[idxQ];
//		int vid = listOfVer[i];
//		int degreeVid=__float2int_rn(dArrDegreeOfVid[i]);
//		//Duyệt qua các đỉnh kề với đỉnh vid dựa vào số lần duyệt là bậc
//		int indexToVidIndN=d_O[vid];
//		int labelFromVid = d_LO[vid];
//		int toVid;
//		int labelToVid;
//		bool b=true;
//		for (int j = 0; j < degreeVid; j++,indexToVidIndN++) //Duyệt qua tất cả các đỉnh kề với đỉnh vid, nếu đỉnh không thuộc embedding thì --> cạnh cũng không thuộc embedding vì đây là Q cuối
//		{			
//			//1.Kiểm tra forward
//			toVid=d_N[indexToVidIndN]; //Lấy vid của đỉnh cần kiểm tra
//			labelToVid = d_LO[toVid]; //lấy label của đỉnh cần kiểm tra
//			posColumn=col;
//			posRow=row;
//			Embedding *Q=dArrPointerEmbedding[posColumn];
//			printf("\nThread %d, j: %d has ToVidLabel:%d",i,j,labelToVid);
//			//1. Trước tiên kiểm tra nhãn của labelToVid có nhỏ hơn minLabel hay không. Nếu nhỏ hơn thì return
//			if(labelToVid<minLabel)
//					goto backward;
//			//2. kiểm tra xem đỉnh toVid có tồn tại trong embedding hay không, nếu tồn tại thì bỏ qua và tiếp tục xét đỉnh khác
//			//Duyệt qua embedding column từ Q cuối đến Q đầu, lần lượt lấy vid so sánh với toVid
//
//			//printf("\n Q[%d] Row[%d] (idx:%d vid:%d)",posColumn,posRow,Q[posRow].idx,Q[posRow].vid);//Q[1][0]
//			if(toVid==Q[posRow].vid) 
//					goto backward;
//
//			//printf("\nj:%d toVid:%d Q.vid:%d",j,toVid,Q[posRow].vid);
//
//			while (true)
//			{
//				posRow = Q[posRow].idx;//0
//				posColumn=posColumn-1;		//0
//				Q=dArrPointerEmbedding[posColumn];
//				//printf("\n posColumn[%d] Row[%d] (idx:%d vid:%d)",posColumn,posRow,Q[posRow].idx,Q[posRow].vid);//Q[0][0]
//				//printf("\nj:%d toVid:%d Q.vid:%d",j,toVid,Q[posRow].vid);
//				if(toVid==Q[posRow].vid) {
//					b=false; break;
//				}
//				posRow=Q[posRow].idx;//-1
//				//printf("\nposRow:%d",posRow);
//				if(posRow==-1) break;
//			}
//			if (b==false){
//				b=true; 
//				goto backward;
//			}
//			int indexOfd_arr_V=i*maxDegreeOfVer+j;
//			//printf("\nThread %d: m:%d",i,maxDegreeOfVer);
//			int indexOfd_LN=indexToVidIndN;
//			dArrV[indexOfd_arr_V].valid=1;
//			printf("\ndArrV[%d].valid:%d",indexOfd_arr_V,dArrV[indexOfd_arr_V].valid);
//			//cập nhật dữ liệu cho mảng dArrExtension
//			dArrExtension[indexOfd_arr_V].vgi=vid;
//			dArrExtension[indexOfd_arr_V].vgj=toVid;
//			dArrExtension[indexOfd_arr_V].lij=d_LN[indexOfd_LN];
//			printf("\n");
//			printf("d_LN[%d]:%d ",indexOfd_LN,d_LN[indexOfd_LN]);
//			dArrExtension[indexOfd_arr_V].li=labelFromVid;
//			dArrExtension[indexOfd_arr_V].lj=labelToVid;
//			dArrExtension[indexOfd_arr_V].vi=fromRMP;
//			dArrExtension[indexOfd_arr_V].vj=maxId+1;
//			//dArrExtension[indexOfd_arr_V].posColumn=idxQ;
//			dArrExtension[indexOfd_arr_V].posRow=row;
//backward:
//			//2. Kiểm tra backward
//			for (int k = 0; k < segdArrVidOnRMP; k++)
//			{
//				if(toVid == dArrVidOnRMP[i*segdArrVidOnRMP+k]){
//
//					int indexOfd_arr_V=i*maxDegreeOfVer+j;
//					//printf("\nThread %d: m:%d",i,maxDegreeOfVer);
//					int indexOfd_LN=indexToVidIndN;
//					dArrV[indexOfd_arr_V].valid=1;
//					dArrV[indexOfd_arr_V].backward=1;
//					printf("\ndArrV[%d].valid:%d backward:%d",indexOfd_arr_V,dArrV[indexOfd_arr_V].valid,dArrV[indexOfd_arr_V].backward);
//					//cập nhật dữ liệu cho mảng dArrExtension
//					dArrExtension[indexOfd_arr_V].vgi=vid;
//					dArrExtension[indexOfd_arr_V].vgj=toVid;
//					dArrExtension[indexOfd_arr_V].lij=d_LN[indexOfd_LN];
//					printf("\n");
//					printf("d_LN[%d]:%d ",indexOfd_LN,d_LN[indexOfd_LN]);
//					dArrExtension[indexOfd_arr_V].li=labelFromVid;
//					dArrExtension[indexOfd_arr_V].lj=labelToVid;
//					dArrExtension[indexOfd_arr_V].vi=maxId;
//					dArrExtension[indexOfd_arr_V].vj=fromPosCol[i*segdArrVidOnRMP+k];
//					//dArrExtension[indexOfd_arr_V].posColumn=idxQ;
//					dArrExtension[indexOfd_arr_V].posRow=row;
//
//					break; //thoát khỏi vòng lặp hiện tại
//				}
//			}
//		}
//	}
//}


__global__ void	kernelGetPointerdArrEmbedding(Embedding *dArrEmbedding,Embedding **dArrPointerEmbedding,int idx){
	dArrPointerEmbedding[idx]=dArrEmbedding;
	//printf("\n PointerdArrEmbedding:%p, PointerdArrPointerEmbedding:%p",dArrEmbedding,dArrPointerEmbedding[idx]);
}

__global__ void kernelPrintdArrPointerEmbedding(Embedding **dArrPointerEmbedding,int noElem,int sizeArr){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i<noElem){
		Embedding *E = dArrPointerEmbedding[i];
		for (int j = 0; j < sizeArr; j++)
		{
			printf("\n Thread %d pointer:%p (idx vid):(%d %d)",i,E,E[j].idx,E[j].vid);
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
		printf("\n dArrV[%d].backward:%d ,dArrV[%d].valid:%d Extension:(vgi:%d,vgj:%d) (vi:%d vj:%d li:%d lij:%d lj:%d)",i,dArrV[i].backward,i,dArrV[i].valid,dArrExtension[i].vgi,dArrExtension[i].vgj,vi,vj,li,lij,lj);
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
		printf("\n Thread %d (vi:%d vj:%d li:%d lij:%d lj:%d) (vgi:%d vgj:%d) ( posRow:%d)",i,vi,vj,li,lij,lj,vgi,vgj,posRow);
	}

}

//Hàm in dExt
inline cudaError_t printdExt(EXT *dExt,int noElem_dExt){
	cudaError_t cudaStatus;

	dim3 block(blocksize);
	dim3 grid((noElem_dExt+block.x -1)/block.x);
	kernelPrintdExt<<<grid,block>>>(dExt,noElem_dExt);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() kernelPrintdExt in printdExt() failed", cudaStatus);
		goto Error;
	}

Error:
	return cudaStatus;
}

//kernel trích các mở rộng hợp lệ từ mảng dArrExtension sang mảng dExt
__global__ void kernelExtractValidExtensionTodExt(EXT *dArrExtension,int *dArrValid,int *dArrValidScanResult,int noElem_dArrV,EXT *dExt,int noElem_dExt){
	int i =blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElem_dArrV){
		if(dArrValid[i]==1){
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

int displaydArrUniEdge(UniEdge *dArrUniEdge,int noElem_dArrUniEdge){
	cudaError_t cudaStatus;
	int status =0;
	UniEdge *hArrUniEdge = (UniEdge*)malloc(sizeof(UniEdge) * noElem_dArrUniEdge);
	if(hArrUniEdge == NULL){
		printf("\n malloc hArrUniEde failed");
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
		printf("\n U[%d] (li lij lj):(%d %d %d)",i,hArrUniEdge[i].li,hArrUniEdge[i].lij,hArrUniEdge[i].lj);
	}

	std::free(hArrUniEdge);
Error:
	return status;
}



//Hàm trích các mở rộng hợp lệ từ mảng dArrExtension sang mảng dExt
int PMS::extractValidExtensionTodExt(EXT *dArrExtension,V *dArrV,int noElem_dArrV,int idxEXT){
	cudaError_t cudaStatus;

	int status =0;
	//2. Scan mảng dArrValid để lấy kích thước của mảng cần tạo
	int *dArrValidScanResult = nullptr;

	cudaStatus = cudaMalloc((void**)&dArrValidScanResult,sizeof(int)*noElem_dArrV);
	if (cudaStatus!=cudaSuccess){
		status = -1;
		fprintf(stderr,"\n CudaMalloc dArrValidScanResult in extractValidExtensionToExt() failed");
		goto Error;
	}
	else
	{
		cudaMemset(dArrValidScanResult,0,sizeof(int)*noElem_dArrV);
	}


	//cudaStatus = scanV(dArrV->valid,noElem_dArrV,dArrValidScanResult); //hàm scan này có vấn đề. Nó làm thay đổi giá trị đầu vào.
	//if (cudaStatus!=cudaSuccess){
	//	status = -1;
	//	fprintf(stderr,"\n scanV dArrValid in extractValidExtensionToExt() failed");
	//	goto Error;
	//}
	myScanV(dArrV->valid,noElem_dArrV,dArrValidScanResult);

	////In nội dung kết quả dArrValidScanResult
	printf("\n********dArrValid******\n");
	displayDeviceArr(dArrV->valid,noElem_dArrV);

	printf("\n********dArrValidScanResult******\n");
	displayDeviceArr(dArrValidScanResult,noElem_dArrV);

	//3. Lấy kích thước của mảng dArrExt;
	int noElem_dExt=0;
	cudaStatus=getSizeBaseOnScanResult(dArrV->valid,dArrValidScanResult,noElem_dArrV,noElem_dExt);
	if (cudaStatus!=cudaSuccess){
		status = -1;
		fprintf(stderr,"\n getSizeBaseOnScanResult in extractValidExtensionToExt() failed");
		goto Error;
	}

	//In nội dung noElem_dExt
	printf("\n******** noElem In dArrExt ******\n");
	printf("\n noElem_dExt:%d",noElem_dExt);
	hLevelEXT.at(idxLevel).vE.at(idxEXT).noElem = noElem_dExt;
	/**************** Nếu không tìm được mở rộng nào thì return *************/
	if (noElem_dExt == 0) 
	{
		cudaFree(dArrValidScanResult);
		return status;
	}
	//Nếu tìm được mở rộng thì xây dựng EXTk, rồi trích các mở rộng duy nhất và tính độ hỗ trợ của chúng. Đồng thời
	//lọc ra các độ hỗ trợ thoả minsup
	//Quản lý theo Level
	//4. Khởi tạo mảng dArrExt có kích thước noElem_dExt rồi trích dữ liệu từ dArrExtension sang dựa vào dArrValid.
	//hLevelEXT.at(idxLevel).vE.at(idxEXT).noElem = noElem_dExt;
	cudaStatus = cudaMalloc((void**)&hLevelEXT.at(idxLevel).vE.at(idxEXT).dArrExt,noElem_dExt*sizeof(EXT));
	CHECK(cudaStatus);
	if(cudaStatus!=cudaSuccess){
		status =-1;
		fprintf(stderr,"\n cudaMalloc dExt in extractValidExtensionTodExt() failed", cudaStatus);
		goto Error;
	}
	else
	{
		cudaMemset(hLevelEXT.at(idxLevel).vE.at(idxEXT).dArrExt,0,sizeof(EXT)*noElem_dExt);
	}
	dim3 blockb(blocksize);
	dim3 gridb((noElem_dArrV+blockb.x -1)/blockb.x);
	kernelExtractValidExtensionTodExt<<<gridb,blockb>>>(dArrExtension,dArrV->valid,dArrValidScanResult,noElem_dArrV,hLevelEXT.at(idxLevel).vE.at(idxEXT).dArrExt,noElem_dExt);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	CHECK(cudaStatus);
	if(cudaStatus!=cudaSuccess){
		status = -1;
		fprintf(stderr,"\n cudaDeviceSynchronize() kernelExtractValidExtensionTodExt in extractValidExtensionTodExt() failed", cudaStatus);
		goto Error;
	}
	//In mảng dExt;
	printf("\n********** dArrExt **********\n");
	cudaStatus =printdExt(hLevelEXT.at(idxLevel).vE.at(idxEXT).dArrExt,noElem_dExt);
	CHECK(cudaStatus);
	if(cudaStatus!=cudaSuccess){
		status =-1;
		fprintf(stderr,"\n printdExt() in extractValidExtensionTodExt() failed", cudaStatus);
		goto Error;
	}

	//kernelGetvivj<<<1,100>>>(hLevelEXT.at(idxLevel).vE.at(idxEXT).dArrExt,noElem_dExt);
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
	cudaStatus=cudaMalloc((void**)&dArrAllPossibleExtension,noElem_dArrAllPossibleExtension*sizeof(int));
	CHECK(cudaStatus);
	if(cudaStatus!=cudaSuccess){
		status =-1;
		fprintf(stderr,"\n cudaMalloc((void**)&dArrAllPossibleExtension in extractUniExtension() failed",cudaStatus);
		goto Error;
	}
	else
	{
		cudaMemset(dArrAllPossibleExtension,0,noElem_dArrAllPossibleExtension*sizeof(int));
	}

	dim3 blockc(blocksize);
	dim3 gridc((hLevelEXT.at(idxLevel).vE.at(idxEXT).noElem + blockc.x -1)/blockc.x);
	kernelForwardPossibleExtension_NonLast<<<gridc,blockc>>>(hLevelEXT.at(idxLevel).vE.at(idxEXT).dArrExt,hLevelEXT.at(idxLevel).vE.at(idxEXT).noElem,Lv,dArrAllPossibleExtension);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	CHECK(cudaStatus);
	if(cudaStatus !=cudaSuccess){
		status =-1;
		fprintf(stderr,"\n kernelForwardPossibleExtension_NonLast  failed",cudaStatus);
		goto Error;
	}

	//Scan mảng dArrAllPossibleExtension để biết kích thước của mảng dArrUniEdge và ánh xạ từ vị trí trong dArrAllPossibleExtension thành nhãn để lưu vào dArrUniEdge được quản lý bởi hLevelUniEdge
	int *dArrAllPossibleExtensionScanResult =nullptr;
	cudaStatus = cudaMalloc((void**)&dArrAllPossibleExtensionScanResult,noElem_dArrAllPossibleExtension*sizeof(int));
	CHECK(cudaStatus);
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
	myScanV(dArrAllPossibleExtension,noElem_dArrAllPossibleExtension,dArrAllPossibleExtensionScanResult);
	//Tính kích thước của dArrUniEdge và lưu vào noElem_dArrUniEdge
	cudaStatus =getSizeBaseOnScanResult(dArrAllPossibleExtension,dArrAllPossibleExtensionScanResult,noElem_dArrAllPossibleExtension,noElem_dArrUniEdge);
	CHECK(cudaStatus);
	if(cudaStatus!=cudaSuccess){
		status = -1;
		fprintf(stderr,"\n scanV dArrAllPossibleExtension in extractUniExtension() failed",cudaStatus);
		goto Error;
	}

	//Hiển thị giá trị của noElem_dArrUniEdge
	printf("\n******noElem_dArrUniEdge************\n");
	printf("\n noElem_dArrUniEdge:%d",noElem_dArrUniEdge);

	hLevelUniEdge.at(idxLevel).vUE.at(idxEXT).noElem=noElem_dArrUniEdge;

	//Cấp phát bộ nhớ cho dArrUniEdge
	cudaStatus = cudaMalloc((void**)&hLevelUniEdge.at(idxLevel).vUE.at(idxEXT).dArrUniEdge,noElem_dArrUniEdge*sizeof(UniEdge));
	CHECK(cudaStatus);
	if(cudaStatus!=cudaSuccess){
		status =-1;
		fprintf(stderr,"\n cudaMalloc dArrUniEdge  failed",cudaStatus);
		goto Error;
	}

	//lấy nhãn Li lưu vào biến dFromLi	
	int *dFromLi=nullptr;
	cudaStatus = cudaMalloc((void**)&dFromLi,sizeof(int));
	CHECK(cudaStatus);
	if(cudaStatus!=cudaSuccess){
		status =-1;
		printf("\n cudaMalloc dFromLi failed");
		goto Error;
	}

	kernelGetFromLabel<<<1,1>>>(hLevelEXT.at(idxLevel).vE.at(idxEXT).dArrExt,dFromLi);
	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	CHECK(cudaStatus);
	if(cudaStatus!=cudaSuccess){
		status =-1;
		fprintf(stderr,"\n  kernelGetFromLabel  failed");
		goto Error;
	}



	//Gọi hàm để ánh xạ dữ liệu từ dArrAllPossibleExtension sang mảng dArrUniEdge
	/* Input Data:	dArrAllPossibleExtension, dArrAllPossibleExtensionScanResult,  */
	dim3 blockd(blocksize);
	dim3 gridd((noElem_dArrAllPossibleExtension + blockd.x -1)/blockd.x);
	kernelFilldArrUniEdge<<<gridd,blockd>>>(dArrAllPossibleExtension,dArrAllPossibleExtensionScanResult,noElem_dArrAllPossibleExtension,hLevelUniEdge.at(idxLevel).vUE.at(idxEXT).dArrUniEdge,Lv,dFromLi);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	CHECK(cudaStatus);
	if(cudaStatus!=cudaSuccess){
		status =-1;
		fprintf(stderr,"\n kernelFilldArrUniEdge failed",cudaStatus);
		goto Error;
	}

	//In nội dung mảng dArrUniEdge
	printf("\n**********dArrUniEdge************");
	displaydArrUniEdge(hLevelUniEdge.at(idxLevel).vUE.at(idxEXT).dArrUniEdge,noElem_dArrUniEdge);

	//Duyệt qua các cạnh duy nhất tính và lưu trữ độ hỗ trợ của chúng vào một mảng tạm nào đó
	//Sau đó trích những độ hỗ trợ thoả minsup vào lưu vào hLevelUniEdgeSatisfyMinsup
	//Chỉ cần quan tâm kết quả trả về gồm số lượng cạnh thoả minsup, cạnh đó là gì và độ hỗ trợ là bao nhiêu.
	status = computeSupportv2(hLevelEXT.at(idxLevel).vE.at(idxEXT).dArrExt,hLevelEXT.at(idxLevel).vE.at(idxEXT).noElem,hLevelUniEdge.at(idxLevel).vUE.at(idxEXT).dArrUniEdge,hLevelUniEdge.at(idxLevel).vUE.at(idxEXT).noElem,hLevelUniEdgeSatisfyMinsup.at(idxLevel).vecUES.at(idxEXT).noElem,hLevelUniEdgeSatisfyMinsup.at(idxLevel).vecUES.at(idxEXT).dArrUniEdge,hLevelUniEdgeSatisfyMinsup.at(idxLevel).vecUES.at(idxEXT).hArrSupport);
	if(status!=0){
		goto Error;
	}



	//printf("\n************ dArrUniEdgeSatisfyMinSup*********\n");
	//printf("\n noElem:%d",hLevelUniEdgeSatisfyMinsup.at(idxLevel).vecUES.at(idxEXT).noElem);
	//displaydArrUniEdge(hLevelUniEdgeSatisfyMinsup.at(idxLevel).vecUES.at(idxEXT).dArrUniEdge,hLevelUniEdgeSatisfyMinsup.at(idxLevel).vecUES.at(idxEXT).noElem);
	//for (int j = 0; j < hLevelUniEdgeSatisfyMinsup.at(idxLevel).vecUES.at(idxEXT).noElem; j++)
	//{
	//	printf("\n Support: %d ",hLevelUniEdgeSatisfyMinsup.at(idxLevel).vecUES.at(idxEXT).hArrSupport[j]);
	//}

	status=Miningv2(hLevelUniEdgeSatisfyMinsup.at(idxLevel).vecUES.at(idxEXT).noElem,hLevelUniEdgeSatisfyMinsup.at(idxLevel).vecUES.at(idxEXT).dArrUniEdge,hLevelUniEdgeSatisfyMinsup.at(idxLevel).vecUES.at(idxEXT).hArrSupport,hLevelEXT.at(idxLevel).vE.at(idxEXT).dArrExt,hLevelEXT.at(idxLevel).vE.at(idxEXT).noElem,idxEXT);
	FUNCHECK(status);
	if(status!=0){
		goto Error;
	}


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

Error:
	return cudaStatus;
}

int PMS::computeSupportv2(EXT *dArrExt,int noElemdArrExt,UniEdge *dArrUniEdge,int noElemdArrUniEdge,int &noElem,UniEdge *&dArrUniEdgeSup,int *&hArrSupport){
	int status=0;
	cudaError_t cudaStatus;

#pragma region "find Boundary and scan Boundary"
	int *dArrBoundary=nullptr;
	int noElemdArrBoundary = noElemdArrExt;
	cudaStatus=cudaMalloc((void**)&dArrBoundary,sizeof(int)*noElemdArrBoundary);
	CHECK(cudaStatus);
	if(cudaStatus!=cudaSuccess){
		status =-1;
		fprintf(stderr,"\n cudaMalloc dArrBoundary in computeSupportv2() failed");
		goto Error;
	}
	else
	{
		cudaMemset(dArrBoundary,0,sizeof(int)*noElemdArrBoundary);
	}

	int *dArrBoundaryScanResult=nullptr;
	cudaStatus=cudaMalloc((void**)&dArrBoundaryScanResult,sizeof(int)*noElemdArrBoundary);
	CHECK(cudaStatus);
	if(cudaStatus!=cudaSuccess){
		status=-1;
		fprintf(stderr,"\n cudaMalloc dArrBoundary in computeSupportv2() failed");
		goto Error;
	}
	else
	{
		cudaMemset(dArrBoundaryScanResult,0,sizeof(int)*noElemdArrBoundary);
	}

	//Tìm boundary của EXTk và lưu kết quả vào mảng dArrBoundary
	status = findBoundary(dArrExt,noElemdArrExt,dArrBoundary);
	FUNCHECK(status);
	if(status!=0){
		printf("\n findBoundary() in computeSupportv2() failed");
		goto Error;
	}

	printf("\n ************* dArrBoundary ************\n");
	displayDeviceArr(dArrBoundary,noElemdArrExt);

	//Scan dArrBoundary lưu kết quả vào dArrBoundaryScanResult
	//cudaStatus=scanV(dArrBoundary,noElemdArrBoundary,dArrBoundaryScanResult);
	//CHECK(cudaStatus);
	//if(cudaStatus!=cudaSuccess){
	//	status = -1;
	//	fprintf(stderr,"\n Exclusive scan dArrBoundary in computeSupportv2() failed",cudaStatus);
	//	goto Error;
	//}
	myScanV(dArrBoundary,noElemdArrBoundary,dArrBoundaryScanResult);

	printf("\n**************dArrBoundaryScanResult****************\n");
	displayDeviceArr(dArrBoundaryScanResult,noElemdArrBoundary);

	//Tính support của cạnh duy nhất.
	float *dF=nullptr; //khai báo mảng dF
	int noElemdF = 0; //Số lượng phần tử của mảng dF

	cudaStatus = cudaMemcpy(&noElemdF,&dArrBoundaryScanResult[noElemdArrBoundary-1],sizeof(int),cudaMemcpyDeviceToHost);
	CHECK(cudaStatus);
	if(cudaStatus !=cudaSuccess){
		status =-1;
		fprintf(stderr,"\n cudamemcpy noElemdF failed",cudaStatus);
		goto Error;
	}
	noElemdF++; //Phải tăng lên 1 vì giá trị hiện tại chỉ là chỉ số của mảng
	printf("\n*****noElemdF******\n");
	printf("noElemdF:%d",noElemdF);

	//Cấp phát bộ nhớ trên device cho mảng dF
	cudaStatus = cudaMalloc((void**)&dF,sizeof(float)*noElemdF);
	CHECK(cudaStatus)
		if(cudaStatus!=cudaSuccess){
			status =-1;
			fprintf(stderr,"\ncudaMalloc dF failed",cudaStatus);
			goto Error;
		}
		else
		{
			cudaMemset(dF,0,sizeof(float)*noElemdF);
		}
#pragma endregion "end of finding Boundary"

		//Tạm thời chứa độ hỗ trợ của tất cả các cạnh duy nhất.
		//Sau đó, trích những cạnh và độ hỗ trợ thoả minsup vào hLevelUniEdgeSatisfyMinsup tại level tương ứng
		int *hArrSupportTemp = (int*)malloc(sizeof(int)*noElemdArrUniEdge);
		if(hArrSupportTemp==NULL){
			status =-1;
			printf("\n Malloc hArrSupportTemp in computeSupportv2() failed");
			goto Error;
		}
		else
		{
			memset(hArrSupportTemp,0,sizeof(unsigned int)*noElemdArrUniEdge);
		}
		//		//Duyệt và tính độ hỗ trợ của các cạnh
		dim3 blocke(blocksize);
		dim3 gride((noElemdArrExt+blocke.x-1)/blocke.x);

		//printf("\n**********dArrUniEdge************");				
		//displaydArrUniEdge(dArrUniEdge,noElemdArrUniEdge);

		for (int i = 0; i < noElemdArrUniEdge; i++)
		{					
			float support=0;
			kernelFilldF<<<gride,blocke>>>(dArrUniEdge,i,dArrExt,noElemdArrExt,dArrBoundaryScanResult,dF);

			cudaDeviceSynchronize();
			cudaStatus = cudaGetLastError();
			CHECK(cudaStatus);
			if(cudaStatus !=cudaSuccess){
				status =-1;
				fprintf(stderr,"\n calcSupport failed",cudaStatus);
				goto Error;
			}				

			printf("\n**********dF****************\n");
			displayDeviceArr(dF,noElemdF);

			CHECK(cudaStatus = reduction(dF,noElemdF,support));
			if(cudaStatus!=cudaSuccess){
				status=-1;
				printf("\nreduction failed");
				goto Error;
			}

			printf("\n******support********");
			printf("\n Support:%f",support);

			CHECK(cudaStatus = cudaMemset(dF,0,noElemdF*sizeof(float)));
			if(cudaStatus!=cudaSuccess){
				status=-1;
				printf("\n cudaMemset failed");
				goto Error;
			}

			hArrSupportTemp[i]=support;
		}
		printf("\n************hArrSupportTemp**********\n");
		for (int j = 0; j < noElemdArrUniEdge; j++)
		{
			printf("j[%d]:%d ",j,hArrSupportTemp[j]);
		}

		//Tiếp theo là lọc giữ lại cạnh và độ hỗ trợ thoả minsup
		status = extractUniEdgeSatisfyMinsupV2(hArrSupportTemp,dArrUniEdge,noElemdArrUniEdge,minsup,noElem,dArrUniEdgeSup,hArrSupport);
		FUNCHECK(status)
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

	printf("\n *******dResultSup********\n");
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
	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n kernelMarkUniEdgeSatisfyMinsup in extractUniEdgeSatisfyMinsup() failed",cudaStatus);
		status = -1;
		goto Error;
	}

	printf("\n ***********dV**********\n");
	FUNCHECK(status = displayDeviceArr(dV,noElemUniEdge));
	if(status!=0){
		goto Error;
	}


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

	myScanV(dV,noElemUniEdge,dVScanResult);

	printf("\n ***********dVScanResult**********\n");
	displayDeviceArr(dVScanResult,noElemUniEdge);

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
		printf("\n malloc hArrSup of hUniEdgeSatisfyMinsup failed()");
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
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	CHECK(cudaStatus);
	if(cudaStatus!=cudaSuccess){
		status = -1;
		fprintf(stderr,"\n kernelExtractUniEdgeSatisfyMinsup() in extractUniEdgeSatisfyMinsup() failed",cudaStatus);
		goto Error;
	}
	printf("\n ********hUniEdgeSatisfyMinsup.dUniEdge****************\n");
	FUNCHECK(status=displayArrUniEdge(dArrUniEdgeSup,noElem));
	if(status!=0){
		goto Error;
	}

	printf("\n ********hUniEdgeSatisfyMinsup.dSup****************\n");
	displayDeviceArr(dSup,noElem);

	CHECK(cudaStatus = cudaMemcpy(hArrSupport,dSup,sizeof(int)*noElem,cudaMemcpyDeviceToHost));
	CHECK(cudaStatus);
	if(cudaStatus!=cudaSuccess){
		status = -1;
		fprintf(stderr,"\n kernelExtractUniEdgeSatisfyMinsup() in extractUniEdgeSatisfyMinsup() failed",cudaStatus);
		goto Error;
	}

	for (int i = 0; i < noElem; i++)
	{
		printf("\n hArrSupport:%d ",hArrSupport[i]);
	}

	cudaFree(dResultSup);
	cudaFree(dV);
	cudaFree(dVScanResult);
	cudaFree(dSup);
Error:
	return status;
}


__global__ void printdArrUniEdge(UniEdge *dArrUniEdge,int pos){
	printf("\n d[%d]: (li,lij,lj):(%d %d %d)",pos,dArrUniEdge[pos].li,dArrUniEdge[pos].lij,dArrUniEdge[pos].lj);
}


__global__ void kernelFilldF(UniEdge *dArrUniEdge,int pos,EXT *dArrExt,int noElemdArrExt,int *dArrBoundaryScanResult,float *dF){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i<noElemdArrExt){
		int li = dArrUniEdge[pos].li;
		int lij = dArrUniEdge[pos].lij;
		int lj = dArrUniEdge[pos].lj;
		int Li = dArrExt[i].li;
		int Lij = dArrExt[i].lij;
		int Lj = dArrExt[i].lj;
		if(li==Li && lij==Lij && lj==Lj){
			dF[dArrBoundaryScanResult[i]]=1;
		}
		printf("\nThread %d: UniEdge(li:%d lij:%d lj:%d) (Li:%d Lij:%d Lj:%d idxdF:%d dF:%d)",i,li,lij,lj,Li,Lij,Lj,dArrBoundaryScanResult[i],dF[dArrBoundaryScanResult[i]]);
	}
}


int PMS::findBoundary(EXT *dArrExt,int noElemdArrExt,int *&dArrBoundary){
	int status =0;
	cudaError_t cudaStatus;
	dim3 block(blocksize);
	dim3 grid((noElemdArrExt+block.x-1)/block.x);

	kernelfindBoundary<<<grid,block>>>(dArrExt,noElemdArrExt,dArrBoundary,maxOfVer);

	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	CHECK(cudaStatus);
	if(cudaStatus!=cudaSuccess){
		status = -1;
		fprintf(stderr,"\n kernelfindBoundary in findBoundary() failed",cudaStatus);
		goto Error;
	}
Error:	
	return status;
}

__global__ void kernelfindBoundary(EXT *dArrExt,int noElemdArrExt,int *dArrBoundary,unsigned int maxOfVer){
	int i = blockDim.x*blockIdx.x + threadIdx.x;	
	if(i<noElemdArrExt-1){		
		unsigned int graphIdAfter=dArrExt[i+1].vgi/maxOfVer;
		unsigned int graphIdCurrent=dArrExt[i].vgi/maxOfVer;
		if(graphIdAfter!=graphIdCurrent){
			dArrBoundary[i]=1;
		}
	}
}


int PMS::forwardExtension(int idxhEXTk,int *listOfVer,int noElemEmbedding,int fromRMP){
	int status = 0;
	int lastCol = hEmbedding.size() - 1;

	dim3 block(blocksize);
	dim3 grid((noElemEmbedding + block.x -1)/block.x);

	//Tìm bậc lớn nhất của các đỉnh cần mở rộng trong listOfVer
	int maxDegreeOfVer=0;
	float *dArrDegreeOfVid=nullptr; //chứa cậc của các đỉnh trong listOfVer, dùng để duyệt qua các đỉnh lân cận
	//trong csdl
	status=findMaxDegreeOfVer(listOfVer,maxDegreeOfVer,dArrDegreeOfVid,noElemEmbedding); //tìm bậc lớn nhất
	FUNCHECK(status);
	if(status==-1){
		printf("\n findMaxDegreeOfVer() in forwardExtension() failed");
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

	dArrV->noElem =maxDegreeOfVer*noElemEmbedding;
	cudaError_t cudaStatus=cudaMalloc((void**)&(dArrV->valid),(dArrV->noElem)*sizeof(int));
	CHECK(cudaStatus);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMalloc dArrV in  failed");
		status =-1;
		goto Error;
	}
	else
	{
		cudaStatus = cudaMemset(dArrV->valid,0,(dArrV->noElem)*sizeof(int));
		CHECK(cudaStatus);
		if(cudaStatus !=cudaSuccess){
			status =-1;
			goto Error;
		}
	}
	cudaStatus=cudaMalloc((void**)&(dArrV->backward),(dArrV->noElem)*sizeof(int));
	CHECK(cudaStatus);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMalloc dArrV in  failed");
		status =-1;
		goto Error;
	}
	else
	{
		cudaStatus=cudaMemset(dArrV->backward,0,(dArrV->noElem)*sizeof(int));
		CHECK(cudaStatus);
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
	cudaStatus = cudaMalloc((void**)&dArrExtensionTemp,(dArrV->noElem)*sizeof(EXT));
	CHECK(cudaStatus);
	if(cudaStatus!=cudaSuccess){
		status=-1;
		fprintf(stderr,"\n cudaMalloc dArrExtensionTemp forwardExtensionQ() failed",cudaStatus);
		goto Error;
	}
	else
	{
		cudaStatus=cudaMemset(dArrExtensionTemp,0,dArrV->noElem*sizeof(EXT));
		CHECK(cudaStatus);
		if(cudaStatus !=cudaSuccess){
			status =-1;
			goto Error;
		}
	}

	printf("\nnoElem_dArrV:%d",dArrV->noElem );


	////Gọi kernel với các đối số: CSDL, bậc của các đỉnh, dArrV, dArrExtension,noElem_Embedding,maxDegreeOfVer,idxQ,dArrPointerEmbedding,minLabel,maxid
	dim3 blocka(blocksize);
	dim3 grida((noElemEmbedding+block.x - 1)/blocka.x);
	//hdb.at(0).dN;
	//int noElemdArrPointerEmbedding = lastCol+1;
	//kernel tìm các mở rộng forward hợp lệ	

	kernelFindValidForwardExtension<<<grida,blocka>>>(hLevelPtrEmbedding.at(idxLevel).dArrPointerEmbedding,hLevelPtrEmbedding.at(idxLevel).noElem,hLevelPtrEmbedding.at(idxLevel).noElemEmbedding,hdb.at(0).dO,hdb.at(0).dLO,hdb.at(0).dN,hdb.at(0).dLN,dArrDegreeOfVid,maxDegreeOfVer,dArrExtensionTemp,listOfVer,minLabel,maxId,fromRMP,dArrV->valid,dArrV->backward);
	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() kernelFindValidForwardExtension in forwardExtensionQ() failed",cudaStatus);
		goto Error;
	}
	//In mảng dArrV để kiểm tra thử
	printf("\n****************dArrV_valid*******************\n");
	displayDeviceArr(dArrV->valid,dArrV->noElem);
	////Chép kết quả từ dArrExtension sang dExt
	//chúng ta cần có mảng dArrV để trích các mở rộng duy nhất
	//Hàm này cũng gọi hàm trích các mở rộng duy nhất và tính độ hỗ trợ của chúng
	displayDeviceEXT(dArrExtensionTemp,dArrV->noElem);
	status = extractValidExtensionTodExt(dArrExtensionTemp,dArrV,dArrV->noElem,idxhEXTk);
	FUNCHECK(status);
	if(status!=0){
		fprintf(stderr,"\n extractValidExtensionTodExt() in forwardExtensionQ() failed");
		goto Error;
	}


	cudaFree(dArrV->valid);
	cudaFree(dArrV->backward);
	free(dArrV);
	cudaFree(dArrDegreeOfVid);
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


int PMS::findMaxDegreeOfVer(int *listOfVer,int &maxDegreeOfVer,float *&dArrDegreeOfVid,int noElem){
	int status = 0;
	FUNCHECK(status = findDegreeOfVer(listOfVer,dArrDegreeOfVid,noElem));
	if(status == -1){
		printf("\n findDegreeOfVer() in findMaxDegreeOfVer() faild");
		goto Error;
	}

	printf("\n*******dArrDegreeOfVid*************\n");
	displayDeviceArr(dArrDegreeOfVid,noElem);

	//Tìm bậc lớn nhất và lưu kết quả vào biến maxDegreeOfVer
	float *h_max;
	h_max = (float*)malloc(sizeof(float));
	if(h_max==NULL){
		printf("\nMalloc h_max failed");
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
	printf("\nMax degree of vid in Q column is: %d",maxDegreeOfVer);

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

__global__ void kernelCalDegreeOfVid(int *listOfVer,int *d_O, int numberOfElementd_O,int noElem_Embedding,int numberOfElementd_N,unsigned int maxOfVer,float *dArrDegreeOfVid){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElem_Embedding){
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


int PMS::findDegreeOfVer(int *listOfVer,float *&dArrDegreeOfVid,int noElem_Embedding){
	int status = 0;
	cudaError_t cudaStatus;
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

	dim3 block(blocksize);
	dim3 grid((noElem_Embedding + block.x -1)/block.x);
	//Đầu vào của kernelCalDegreeOfVid là một tập đỉnh trên RMP kèm theo Embedding Header của nó.
	kernelCalDegreeOfVid<<<grid,block>>>(listOfVer,hdb.at(0).dO, hdb.at(0).noElemdO,noElem_Embedding,hdb.at(0).noElemdN, maxOfVer,dArrDegreeOfVid);	
	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();	
	CHECK(cudaStatus);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n kernelCalDegreeOfVid in findDegreeOfVer() failed",cudaStatus);
		status =-1;
		goto Error;
	}

Error:
	return status;

}

