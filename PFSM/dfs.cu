/*
 *  dfs.cpp
 *  GSPAN
 *
 *  Created by Jinseung KIM on 09. 07. 19.
 *  Copyright 2009 KyungHee. All rights reserved.
 *
 */

#include "gspan.cuh"
#include <cstring>
#include <string>
#include <iterator>
#include <set>
using namespace std;

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
