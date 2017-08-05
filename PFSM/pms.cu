#pragma once
#include "pms.cuh"


float hTime=0.0;
float dTime=0.0;

PMS::PMS(){
	Lv=0;
	Le=0;
	maxOfVer=0;
	numberOfGraph=0;

	//std::cout<<" PMS initialized " << std::endl;
	//char* outfile;
	//outfile = "/result.graph";
	//fos.open(outfile);	
}
PMS::~PMS(){
	//std::cout<<" PMS terminated " << std::endl;
	//fos.close();

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

void PMS::displayArrExtension(Extension *dExtension,int noElem){

	//dim3 block(blocksize);
	//dim3 grid((noElem + block.x - 1)/block.x);

	//kernelPrintExtention<<<grid,block>>>(dExtension,noElem);
	//cudaDeviceSynchronize();
	Extension *hExtension = (Extension*)malloc(sizeof(Extension)*noElem);
	if(hExtension==NULL){
		printf("\n Malloc hExtension in displayArrExtension() failed");
		exit(1);
	}
	CHECK(cudaMemcpy(hExtension,dExtension,sizeof(Extension)*noElem,cudaMemcpyDeviceToHost));
	for (int i = 0; i < noElem; i++)
	{
		printf("\n[%d]: DFS code:(%d,%d,%d,%d,%d)  (vgi,vgj):(%d,%d)\n",i,hExtension[i].vi,hExtension[i].vj,hExtension[i].li,hExtension[i].lij,hExtension[i].lj,hExtension[i].vgi,hExtension[i].vgj);
	}
	
	return;
}

void PMS::displayArrUniEdge(UniEdge* dUniEdge,int noElem){
	UniEdge *hUniEdge = (UniEdge*)malloc(sizeof(UniEdge)*noElem);
	if(hUniEdge==NULL){
		printf("\n malloc hUniEdge in displayArrUniEdge() failed");
		exit(1);
	}
	CHECK(cudaMemcpy(hUniEdge,dUniEdge,sizeof(UniEdge)*noElem,cudaMemcpyDeviceToHost));
	for (int i = 0; i < noElem; i++)
	{
		printf("\n U[%d]: (li lij lj) = (%d %d %d)",i,hUniEdge[i].li,hUniEdge[i].lij,hUniEdge[i].lj);
	}
	free(hUniEdge);
	return;
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
	int *size;
	CHECK(cudaMalloc((void**)&size,sizeof(int)));
	CHECK(cudaMemset(size,0,sizeof(int)));
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
	 scanV(dV,numberElementd_Extension,dVScanResult);
	
		
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
		CHECK(cudaMalloc((void**)&hValidExtension.at(0).dExtension,sizeof(Extension)*hValidExtension.at(0).noElem));
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
	cudaStatus = scanV(d_allPossibleExtension,noElem_dallPossibleExtension,d_allPossibleExtensionScanResult);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n ScanV() in computeSupport() failed");
		status = -1;
		goto Error;
	}
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
	CHECK(scanV(dV,noElemUniEdge,dVScanResult));
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

	cudaStatus=scanV(dB,noElement_dB,dBScanResult);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\nscanB function failed",cudaStatus);
		status =-1;
		goto Error;
	}

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
	*value = (inputArray[noEleInputArray-1].vgi/maxOfVer); /*Lấy global vertex id chia cho tổng số đỉnh của đồ thị (maxOfVer). Ở đây các đồ thị luôn có số lượng đỉnh bằng nhau (maxOfVer) */
}

cudaError_t getLastElementExtension(Extension* inputArray,unsigned int numberElementOfInputArray,int &outputValue,unsigned int maxOfVer){
	cudaError_t cudaStatus;

	int *temp=nullptr;
	CHECK(cudaMalloc((int**)&temp,sizeof(int)));
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


	scanV(dV,noElemdV,dVScanResult);

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


int PMS::Mining(){
	int status = 0;

	int noElemtemp = hUniEdgeSatisfyMinsup.at(0).noElem;
	UniEdge *temp=(UniEdge*)malloc(sizeof(UniEdge)*noElemtemp);
	if(temp==NULL){
		printf("\n malloc temp failed");
		status =-1;
		goto Error;
	}

	CHECK(cudaMemcpy(temp,hUniEdgeSatisfyMinsup.at(0).dUniEdge,noElemtemp*sizeof(UniEdge),cudaMemcpyDeviceToHost));
	
	for (int i = 0; i < noElemtemp; i++) //Duyệt qua các UniEdge thoả minSup
	{
			int li,lij,lj;
			li = temp[i].li;
			lij= temp[i].lij;
			lj=temp[i].lj;

			DFS_CODE.push(0,1,temp[i].li,temp[i].lij,temp[i].lj);//xây dựng DFS_CODE
			int minLabel = temp[i].li;
			int maxid = 1;
			
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

				//Giải phóng bộ nhớ hArrGraphId
				free(hArrGraphId);	
				DFS_CODE.pop();
			}
	}
	
	free(temp);
Error:
	return status;
}



