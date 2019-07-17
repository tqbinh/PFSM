/*
 *  ismin.cpp
 *  GSPAN
 *
 *  Created by Jinseung KIM on 09. 07. 19.
 *  Copyright 2009 KyungHee. All rights reserved.
 *
 */

#pragma once
#include "gspan.cuh"
#include "pms.cuh"
using namespace std;
bool gSpan::is_min ()
{
	if (DFS_CODE.size() == 1) //nếu như trong vector<DFS> chỉ có duy nhất 1 DFS thì nó là nhỏ nhất.
		return (true);
	if (DFS_CODE.size() == 6) //nếu như trong vector<DFS> chỉ có duy nhất 1 DFS thì nó là nhỏ nhất.
		check_min(DFS_CODE);
	DFS_CODE.toGraph (GRAPH_IS_MIN);  //xây dựng đồ thị cho DFS_CODE và gán đồ thị đó cho GRAPH_IS_MIN
	DFS_CODE_IS_MIN.clear ();
	
	Projected_map3 root;
	EdgeList           edges;
	
	for (unsigned int from = 0; from < GRAPH_IS_MIN.size() ; ++from) //Duyệt qua số cạnh của DFS_CODE thông qua GRAPH_IS_MIN
		if (get_forward_root (GRAPH_IS_MIN, GRAPH_IS_MIN[from], edges)) //Tại mỗi đỉnh của GRAPH_IS_MIN tìm các forward edge
			for (EdgeList::iterator it = edges.begin(); it != edges.end();  ++it)
				root[GRAPH_IS_MIN[from].label][(*it)->elabel][GRAPH_IS_MIN[(*it)->to].label].push (0, *it, 0); //đưa tất cả các forward edge vào ánh xạ map3 gọi là root
	//Lấy ra cạnh nhỏ nhất trong root (là cạnh đầu tiên).
	Projected_iterator3 fromlabel = root.begin(); //From vertex label
	Projected_iterator2 elabel    = fromlabel->second.begin();//edge label
	Projected_iterator1 tolabel   = elabel->second.begin();//To vertex label
	
	DFS_CODE_IS_MIN.push (0, 1, fromlabel->first, elabel->first, tolabel->first); //Cạnh nhỏ nhất luôn là cạnh đầu tiên trong ánh xạ map3. Đưa cạnh đó vào DFS_CODE_MIN
	
	return (project_is_min (tolabel->second)); //từ tolabel->second tức các mở rộng tính từ đỉnh to của cạnh nhỏ nhất, ta có thể tiếp tục kiểm tra xem toàn bộ DFS_CODE có phải là nhỏ nhất hay không.
}
//How to paralizing this function?
bool gSpan::project_is_min (Projected &projected) //với tham số là các mở rộng của tolabel.second()
{
	const RMPath& rmpath = DFS_CODE_IS_MIN.buildRMPath (); //xây dựng right most path cho DFS_CODE_IS_MIN
	int minlabel         = DFS_CODE_IS_MIN[0].fromlabel;
	int maxtoc           = DFS_CODE_IS_MIN[rmpath[0]].to; //là id của đỉnh cuối cùng trên rmpath của DFS_CODE_MIN
	
	{
		Projected_map1 root;
		bool flg = false; //Dùng để cho biết có mở rộng backward hay không?
		int newto = 0;
		//Trích tất cả các mở rộng backward có thể có của DFSCODE.
		for (int i = rmpath.size()-1; ! flg  && i >= 1; --i) { //i >= 1 :==> DFS_CODE có nhiều hơn 2 cạnh thì mới có mở rộng backward.
			for (unsigned int n = 0; n < projected.size(); ++n) { //Duyệt qua các embeddings.
				PDFS *cur = &projected[n];
				History history (GRAPH_IS_MIN, cur);
				Edge *e = get_backward (GRAPH_IS_MIN, history[rmpath[i]], history[rmpath[0]], history);
				if (e) {
					root[e->elabel].push (0, e, cur);
					newto = DFS_CODE_IS_MIN[rmpath[i]].from;
					flg = true;
				}
			}
		}
		//Nếu có mở rộng backward thì ưu tiên xử lý backward trước.
		if (flg) {
			Projected_iterator1 elabel = root.begin();
			DFS_CODE_IS_MIN.push (maxtoc, newto, -1, elabel->first, -1);
			if (DFS_CODE[DFS_CODE_IS_MIN.size()-1] != DFS_CODE_IS_MIN [DFS_CODE_IS_MIN.size()-1]) return false;
			return project_is_min (elabel->second);
		}
	}
	
	{
		bool flg = false; //cho biết có tìm được mở rộng forward nào từ đỉnh cuối hay ko. Nếu có thì ưu tiên Project_Is_Min em nó.
		int newfrom = 0;
		Projected_map2 root; //Lưu các forward extension edges.
		EdgeList edges;
		//Duyệt qua các embeddings để tìm các mở rộng forward từ đỉnh cuối của mỗi embedding. Đưa các mở rộng forward tìm được vào root projected2
		for (unsigned int n = 0; n < projected.size(); ++n) {
			PDFS *cur = &projected[n];
			History history (GRAPH_IS_MIN, cur); //Đánh dấu những đỉnh và cạnh của đồ thị đã thuộc embedding(đã mở rộng).
			if (get_forward_pure (GRAPH_IS_MIN, history[rmpath[0]], minlabel, history, edges)) {
				flg = true;
				newfrom = maxtoc;
				for (EdgeList::iterator it = edges.begin(); it != edges.end();  ++it)
					root[(*it)->elabel][GRAPH_IS_MIN[(*it)->to].label].push (0, *it, cur);
			}
		}
		//nếu chưa có mở rộng fw nào được tìm thấy từ đỉnh cuối của embedding thì tìm các mở rộng forward từ các đỉnh
		for (int i = 0; ! flg && i < (int)rmpath.size(); ++i) { //thuộc RMP ngoại trừ đỉnh cuối.
			for (unsigned int n = 0; n < projected.size(); ++n) {
				PDFS *cur = &projected[n];
				History history (GRAPH_IS_MIN, cur);
				if (get_forward_rmpath (GRAPH_IS_MIN, history[rmpath[i]], minlabel, history, edges)) {
					flg = true;
					newfrom = DFS_CODE_IS_MIN[rmpath[i]].from;
					for (EdgeList::iterator it = edges.begin(); it != edges.end();  ++it)
						root[(*it)->elabel][GRAPH_IS_MIN[(*it)->to].label].push (0, *it, cur);
				}
			}
		}
		//Nếu có mở rộng forward thì đối sánh nhỏ nhất với DFS_CODE_IS_MIN.
		if (flg) {
			Projected_iterator2 elabel  = root.begin();
			Projected_iterator1 tolabel = elabel->second.begin();
			DFS_CODE_IS_MIN.push (newfrom, maxtoc + 1, -1, elabel->first, tolabel->first);
			if (DFS_CODE[DFS_CODE_IS_MIN.size()-1] != DFS_CODE_IS_MIN [DFS_CODE_IS_MIN.size()-1]) return false;
			return project_is_min (tolabel->second);
		}
	}
	
	return true;
}

