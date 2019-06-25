/*
 *  misc.cpp
 *  GSPAN
 *
 *  Created by Jinseung KIM on 09. 07. 19.
 *  Copyright 2009 KyungHee. All rights reserved.
 *
 */

#include "gspan.cuh"

using namespace std;
const RMPath& DFSCode::buildRMPath() //buildRMPath là một phương thức của DFSCODE //use
{
	rmpath.clear();
	int old_from = -1;
	for(int i = size() -1;i>=0;--i) //Duyệt qua tất cả các cạnh của DFS_CODE theo thứ tự từ cạnh cuối đến cạnh đầu (Duyệt ngược)
	{
		if ((*this)[i].from < (*this)[i].to && (rmpath.empty() || old_from == (*this)[i].to)) //nếu from < to và rmpath rỗng hoặc 
		{
			rmpath.push_back(i);
			old_from = (*this)[i].from;
		}
	}
	return rmpath;
}
//use
void History::build(Graph& g,PDFS* e)
{
	clear(); //cái này là đối tượng history kế thừa từ vector<Edge*>, nên lệnh clear này là dọn dẹp vector<Edge*> của history.
	edge.clear();
	edge.resize(g.edge_size());
	vertex.clear();
	vertex.resize(g.size());
	
	if(e){ //e ở đây chính là tham số thứ 2 A_1, e là một con trỏ lưu trử địa chỉ của A_1
		push_back(e->edge);		//Đưa embedding đang xét vào vector<Edge*> của g
		edge[e->edge->id] = vertex[e->edge->from] = vertex[e->edge->to] = 1; //đồng thời đánh dấu các cạnh và đỉnh của g liên quan đến Embedding đó
		//Duyệt qua các cạnh của embedding để đánh dấu chúng đã thuộc embedding.
		for(PDFS* p = e->prev; p; p=p->prev){
			push_back(p->edge);
			edge[p->edge->id] = vertex[p->edge->from] = vertex[p->edge->to] = 1; //đánh dấu các cạnh trước đó đã kết nối với embedding đang xét.
		}
		
		std::reverse(begin(),end());
	}
}
//Tìm các mở rộng cho các đỉnh còn lại trên Right Most Path, ngoại trừ đỉnh cuối cùng của DFS_CODE
bool get_forward_rmpath (Graph &graph, Edge *e, int minlabel, History& history, EdgeList &result) //Phát triển right most path, từ mỗi đỉnh của rightmostpath sẽ thêm vào các cạnh nào? Danh dách các cạnh sẽ được lưu vào result
{
	result.clear ();


	
	int tolabel = graph[e->to].label;
	
	for (Vertex::edge_iterator it = graph[e->from].edge.begin() ; //Duyệt qua các cạnh kề với đỉnh 0 trong đồ thị có id=0
		 it != graph[e->from].edge.end() ; ++it)
	{
		int tolabel2 = graph[it->to].label;
		if (e->to == it->to || minlabel > tolabel2 || history.hasVertex (it->to))//Cạnh không hợp lệ khi: có id của đỉnh to bằng với id của cạnh thuộc rmpath chính là e->to.
			continue; //hoặc là những cạnh có nhãn đỉnh to nhỏ hơn minlabel hoặc là những cạnh đã thuộc DFS_CODE rồi.
		
		if (e->elabel < it->elabel || (e->elabel == it->elabel && tolabel <= tolabel2)) //nhãn cạnh của cạnh đang xét nhỏ hơn nhãn cạnh mở rộng, hoặc nhãn cạnh đang xét phải bằng với nhãn cạnh mở rộng và nhãn đỉnh của đỉnh đang xét nhỏ hơn nhãn đỉnh mở rộng
			result.push_back (&(*it));
	}
	
	return (! result.empty());
}

bool get_forward_pure (Graph &graph, Edge *e, int minlabel, History& history, EdgeList &result)
{
	result.clear (); //Tìm các forward edge và lưu kết quả trong result.
	for (Vertex::edge_iterator it = graph[e->to].edge.begin() ; //bắt đầu mở rộng cạnh từ đỉnh to, lưu ý đỉnh này phải thuộc right most path
		 it != graph[e->to].edge.end() ; ++it) //ở đây là duyệt qua tất cả các cạnh kề với đỉnh to trên right most path
	{
		if (minlabel > graph[it->to].label || history.hasVertex (it->to)) //Nếu đỉnh của cạnh muốn thêm đã thuộc DFS_CODE hoặc nhãn đỉnh mới < minlabel của DFS_CODE thì bỏ qua cạnh đó.
			continue; //bỏ qua cạnh đó.
		
		result.push_back (&(*it));
	}
	
	return (! result.empty());
}
//use
bool get_forward_root (Graph &g, Vertex &v, EdgeList &result)
{
	result.clear (); //xóa tất cả các phần tử trong result, tức là làm rỗng result.
	for (Vertex::edge_iterator it = v.edge.begin(); it != v.edge.end(); ++it) { //duyệt qua tất cả các cạnh của đỉnh v đang xét
		if (v.label <= g[it->to].label) //nếu như nhãn của đỉnh v nhỏ hơn nhãn đỉnh 'to' kề với nó thì lưu cạnh đó vào EdgeList &result.
			result.push_back (&(*it)); //it là một edge_iterator, về mặt kỹ thuật nó là một con trỏ, nó trỏ đến tất cả các cạnh kết nối với đỉnh v.
	}
	
	return (! result.empty()); //nếu result không rỗng tức là tồn tại forward_edge
}

Edge *get_backward (Graph &graph, Edge* e1, Edge* e2, History& history)
{
	if (e1 == e2)
		return 0;
	
	for (Vertex::edge_iterator it = graph[e2->to].edge.begin() ;
		 it != graph[e2->to].edge.end() ; ++it)
	{
		if (history.hasEdge (it->id))
			continue;
		
		if ( (it->to == e1->from) &&
			( (e1->elabel < it->elabel) ||
			 (e1->elabel == it->elabel) &&
			 (graph[e1->to].label <= graph[e2->to].label)
			 ) )
		{
			return &(*it);
		}
	}
	
	return 0;
}
