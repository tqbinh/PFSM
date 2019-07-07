/*
 *  dfs.cpp
 *  GSPAN
 *
 *  Created by Jinseung KIM on 09. 07. 19.
 *  Copyright 2009 KyungHee. All rights reserved.
 *
 */
#pragma once
#include "gspan.cuh"
#include "pms.cuh"
#include <cstring>
#include <string>
#include <iterator>
#include <set>
using namespace std;

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
void DFSCode::add(int vi,int vj,int li,int lij,int lj)
{
	if (nodeCount()==0)
	{
		push(vi,vj,li,lij,lj); //Push 1st edge to empty DFS_CODE
		minLabel = vi;
		maxId = vj;
		return;
	}
	if(vi<vj)
	{
		push(vi,vj,-1,lij,lj);//build DFS_CODE forward
		maxId=vj;
	}
	else
	{
		push(vi,vj,-1,lij,-1);//xây dựng DFS_CODE backward
	}
}
//use
void DFSCode::remove(int vi,int vj)
{
	pop();
	if (vi<vj)
	{
		--maxId;
	}
}

void DFSCode::fromGraph(Graph& g){
	clear();
	EdgeList edges;
	for(unsigned int from=0;from<g.size();++from)
	{
		if(get_forward_root(g,g[from],edges)==false)
			continue;
		for(EdgeList::iterator it = edges.begin();it!=edges.end();++it)
			push(from,(*it)->to,g[(*it)->from].label,(*it)->elabel,g[(*it)->to].label);
	}
}

bool DFSCode::toGraph(Graph& g) //Convert DFSCode sang đồ thị.
{
	g.clear(); //g là một graph hay là một vector<vertex>, mỗi một phần tử của vector là một vertex và kèm theo các cạnh gắn liền với đỉnh đó.
	
	for(DFSCode::iterator it = begin();it != end(); ++it){ //Duyệt qua DFSCODE
		g.resize(std::max (it->from,it->to) +1); //khởi tạo kích thước cho đồ thị g chính bằng số lượng đỉnh của DFSCode
		
		if(it->fromlabel != -1) //nếu như nhãn của đỉnh là hợp lệ
			g[it->from].label = it->fromlabel; //
		if(it->tolabel != -1)
			g[it->to].label = it->tolabel;
		g[it->from].push (it->from,it->to,it->elabel);
		if(g.directed == false)
			g[it->to].push (it->to,it->from,it->elabel);
	}
	
	g.buildEdge();
	
	return (true);
}

void importDataToArray(int*& _arrayO,int*& _arrayLO,int*& _arrayN,int*& _arrayLN, \
					   const unsigned int _sizeOfarrayO,const unsigned int _noDeg,Graph& g) //return -1 if error
{
	int i=0;
	int numberOfEdges=0;
	int j=0;
	_arrayO[i]=0;
	for(Graph::vertex_iterator v = g.begin(); v !=g.end(); ++v)
	{	//Duyệt qua các cạnh của đỉnh
		for(Vertex::edge_iterator it = v->edge.begin();it!=v->edge.end();++it)
		{	//Gán nhãn cho đỉnh From trong mảng LO, bị gán nhiều lần trong mỗi lần lặp cạnh không tốt
			_arrayLO[i]=g[it->from].label; 
			_arrayN[j]=it->to; //gán id cho đỉnh to trong mảng N
			_arrayLN[j]=it->elabel; //gán nhãn cho cạnh
			j=j+1;	//tăng chỉ số trong mảng N và mảng LN
			++numberOfEdges; //số cạnh đã duyệt
		}
		if (i>=(_sizeOfarrayO-1)) return;
		_arrayO[i+1]=numberOfEdges;
		++i;
	}
}


//use
//Build DFS_Code on Device for checking minDFSCODE
//This action convert DFS_CODE to graph and store graph db on device.
void DFSCode::buildDBOnDevice()
{
	Graph	tempGraph;
	toGraph(tempGraph);
	//Get total of vertex in graph
	int noOfVer = tempGraph.vertex_size();
	int* hArrO = new int[noOfVer];
	if (hArrO==NULL){exit(-1);}
	else {memset(hArrO, -1, noOfVer*sizeof(int));}
	//Get total of degree of all vertex in graph.
	unsigned int noDeg =0;
	Graph& g = tempGraph; 
	for(Graph::vertex_iterator v = g.begin(); v !=g.end(); ++v)
	{	noDeg +=v->edge.size();}
	unsigned int sizeOfArrayN=noDeg;
	//Mảng arrayN lưu trữ id của các đỉnh kề với đỉnh tương ứng trong mảng arrayO.
	int* hArrN = new int[sizeOfArrayN];

	if(hArrN==NULL){exit(-1);}
	else {memset(hArrN, -1, noDeg*sizeof(int));}

	//Prepare dataset on host
	//Mảng arrayLO lưu trữ label cho tất cả các đỉnh trong TRANS.
	int* arrayLO = new int[noOfVer];
	if(arrayLO==NULL)
	{
		exit(-1);
	}else
	{
		memset(arrayLO, -1, noOfVer*sizeof(int));
	}


	//Mảng arrayLN lưu trữ label của tất cả các cạnh trong TRANS
	int* arrayLN = new int[noDeg];
	if(arrayLN==NULL){
		exit(0);
	}else
	{
		memset(arrayLN, -1, noDeg*sizeof(int));
	}

	importDataToArray(hArrO,arrayLO,hArrN,arrayLN,noOfVer,noDeg,g);
	for(int i = 0; i<noOfVer;i++)
	{
		cout<<hArrO[i]<<":"<<arrayLO[i]<<" ";
	}
	cout<<endl;
	for(int i = 0; i<noDeg;i++)
	{
		cout<<hArrN[i]<<":"<<arrayLN[i]<<" ";
	}
	cout<<endl;

	//Copy data from host to device
	DB graphdfscode;
	graphdfscode.noElemdO = noOfVer;
	graphdfscode.noElemdN = noDeg;
	size_t  nBytesO = noOfVer * sizeof(int);
	size_t nBytesN = noDeg * sizeof(int);
	CUCHECK(cudaMalloc((void**)&graphdfscode.dO,nBytesO));
	CUCHECK(cudaMalloc((void**)&graphdfscode.dLO,nBytesO));
	CUCHECK(cudaMalloc((void**)&graphdfscode.dN,nBytesN));
	CUCHECK(cudaMalloc((void**)&graphdfscode.dLN,nBytesN));

	//Chép dữ liệu từ mảng arrayO trên CPU sang GPU được quản lý bởi pointer dO
	CUCHECK(cudaMemcpy(graphdfscode.dO,hArrO,nBytesO,cudaMemcpyHostToDevice));
	CUCHECK(cudaMemcpy(graphdfscode.dLO,arrayLO,nBytesO,cudaMemcpyHostToDevice));
	CUCHECK(cudaMemcpy(graphdfscode.dN,hArrN,nBytesN,cudaMemcpyHostToDevice));
	CUCHECK(cudaMemcpy(graphdfscode.dLN,arrayLN,nBytesN,cudaMemcpyHostToDevice));

	//Release host memory
	delete[] hArrN;
	delete[] hArrO;
	delete[] arrayLO;
	delete[] arrayLN;
	return;
}
bool DFSCode::check_min()
{
	if (this->size() == 1) return true;
	//1. Xây dựng database của đồ thị trên GPU: dfscode_LO, dfscode_O, dfscode_N, dfscode_LN
	//Convert DFS_CODE sang Graph
	this->buildDBOnDevice();
	//Tìm số lượng đỉnh (dfscode_dO,dfscode_dLO) và tổng bậc của các đỉnh (dsfcode_dN, dfscode_dLN).
	//2. Tìm tất cả mở rộng 1 cạnh ban đầu hợp lệ (GPU step)
	//3. So sánh chúng với cạnh đầu tiên của DFS_CODE (GPU step)
		//Nếu có cạnh nhỏ hơn DFS_CODE thì return False
		//Xây dựng embeddings Colum cho cạnh bằng với cạnh đầu tiên của DFS_CODE
	//4. Duyệt qua các cạnh còn lại của DFS_CODE theo thứ tự (tạm gọi là cạnh i). Từ cạnh i ta biết được mở rộng tiếp theo
			//là từ đỉnh nào của RMP.
				//Nếu cạnh i là cho biết mở rộng backward từ đỉnh cuối của RMP. Thì phải tìm đúng mở rộng backward)
				//Nếu cạnh i là mở rộng forward từ đỉnh cuối của RMP thì phải xét cả backward và forward
				//Nếu cạnh i không là mở rộng từ đỉnh không thuộc đỉnh cuối của RMP thì phải xét backward, forward của đỉnh cuối
					//và xét các forward của các đỉnh kế cuối đến i. 
	//5. Tìm RMP của DFS_CODE hiện tại
	//6. Duyệt qua RMP từ đỉnh phải cùng.
		//6.1 Nếu cạnh i là backward và Nếu là các đỉnh phải cùng của RMP thì tìm các mở rộng backward trước.
				//6.1.1 Nếu có mở rộng nào nhỏ hơn cạnh i thì return false.
				//6.1.2 Ngược lại, xây dựng embeddings columns cho các mở rộng bằng với cạnh i,
						//rồi quay lên bước 6. (Nếu không có cạnh nào bằng với cạnh i thì sao?
												//Điều này có thể xảy ra hay không?)
		//6.2 Tìm các mở rộng forward
			//6.2.1 Nếu có mở rộng nào nhỏ hơn cạnh i thì return false (làm sao dọn dẹp bộ nhớ trên device trước khi return false)?
			//6.2.2 Ngược lại, xây dựng embedding columns cho các mở rộng bằng với cạnh i, rồi quay lên bước 6.
	//7. return true (thoả min)
}

unsigned int DFSCode::nodeCount(void) //giải thuật đếm node trên cây
{
	unsigned int nodecount = 0;
	for(DFSCode::iterator it = begin();it != end(); ++it)
		nodecount = std::max(nodecount,(unsigned int) (std::max(it->from,it->to) + 1)); 
	return (nodecount);
}

std::ostream& DFSCode::write(std::ostream& os)
{
	if(size()==0) return os;
	
	os<<"("<<(*this)[0].fromlabel<<") "<<(*this)[0].elabel<<" (of"<<(*this)[0].tolabel<<")";
	
	for(unsigned int i=1;i<size();++i){
		if((*this)[i].from < (*this)[i].to){
			os<<" "<<(*this)[i].elabel<<" ("<<(*this)[i].from<<"f"<<(*this)[i].tolabel<<")";
		}else{
			os<<" "<<(*this)[i].elabel<<" (b"<<(*this)[i].to<<")";
		}
	}
	return os;
}
