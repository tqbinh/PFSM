/*
*  gspan.cpp
*  GSPAN
*
*  Created by Jinseung KIM on 09. 07. 18.
*  Copyright 2009 KyungHee. All rights reserved.
*
*/

#include "gspan.cuh"
#include <iterator>
#include <stdlib.h>
//#include <unistd.h>
using namespace std;
gSpan::gSpan(void){
	std::cout<<" gSpan initialized " << std::endl;
	char* outfile;
	outfile = "/result.graph";
	fos.open(outfile);	
}
gSpan::~gSpan(){
	std::cout<<" gSpan terminated " << std::endl;
	fos.close();
}
void gSpan::read(char* _fname) //Phương thức read của gspan dùng để đọc nội dung của file chemical_340, hàm read này sẽ gọi hàm read(_fname, lineoffset) để đọc từ lineoffset
{
	int lineoffset = 0; //bắt đầu từ vị trí đọc là 0
	Graph g(directed); //tạo đối tượng g với tham số directed=false==> g đơn đồ thị là một đồ thị vô hướng
	while(true){
		lineoffset = g.read(_fname,lineoffset); //gọi hàm read khác để đọc từ lineoffset khởi tạo, kết quả của hàm sẽ trả về lineoffset đang đọc 
		if(g.empty()) break; //đã đọc tất cả các đồ thị trong chemical_340 và không còn đồ thị nào nữa thì thoát
		TRANS.push_back(g); //đưa đồ thị vừa xây dựng được vào tập đồ thị TRANS. TRANS chính là cơ sở dữ liệu đồ thị.
	}
	return;
}

std::map<unsigned int,unsigned int> gSpan::support_counts(Projected& projected)
{
	std::map<unsigned int,unsigned int> counts;
	for(Projected::iterator cur = projected.begin();cur!=projected.end();++cur)
		counts[cur->id]+=1;
	return (counts);
}

unsigned int gSpan::support (Projected& projected) //Tính độ hỗ trợ của cạnh. Ở đây, Projected là một vector<PDFS> lưu trữ cạnh mà bạn muốn tính độ hỗ trợ và các đồ thị của cạnh đó
{
	unsigned int oid = 0xffffffff;
	unsigned int size = 0;

	for(Projected::iterator cur=projected.begin();cur!=projected.end();++cur)
	{
		if(oid!=cur->id)
			++size;
		oid = cur->id;
	}

	return size;
}

void gSpan::report(Projected& projected,unsigned int sup) //Phương thức report trả về kết quả
{
	/* Filter to small/too large graphs	 */
	if(maxpat_max > maxpat_min && DFS_CODE.nodeCount() > maxpat_max) //Đồ thị kết quả phải có số node nhiều hơn số node quy định maxpat_max bởi người dùng
		return;
	//*os << maxpat_min << ":" << DFS_CODE.nodeCount() << ":" << maxpat_min << std::endl;
	if(maxpat_min > 0 && DFS_CODE.nodeCount() < maxpat_min) //đồ thị kết quả phải có số node ít hơn số node maxpat_min quy định bởi người dùng
		return;
	if(where){
		/*
		*os<<"<pattern>\n";
		*os<<"<id>"<<ID<<"</id>\n";
		*os<<"<support>"<<sup<<"</support>\n";
		*os<<"<what>";
		fos<<"<pattern>\n";
		fos<<"<id>"<<ID<<"</id>\n";
		fos<<"<support>"<<sup<<"</support>\n";
		fos<<"<what>";		
		*/
		*os << "t # " << ID << " " << sup << endl; //ghi đồ thị đó vào tập tin result.txt
	}

	if(!enc){ //enc default value is false
		Graph g(directed); //khởi tạo g là một đồ thị vô hướng
		DFS_CODE.toGraph(g); //Convert DFS code sang đồ thị.

		if(!where) //default value of where is true
			*os << "t # " << ID << " * " << sup; //fos << "t # " << ID << " * " << sup;
		//*os << "\n";
		//fos << "\n";

		g.write(*os);

		//g.write(fos);
	}else{
		if(!where)
			*os << "<" << ID << "> " << sup << " [";
		DFS_CODE.write (*os);

		if(!where) *os << "]";
	}

	if(where) {
		/*
		*os << "</what>\n<where>";
		fos << "</what>\n<where>";
		*/
		*os << "x ";
		unsigned int oid = 0xffffffff;
		for(Projected::iterator cur = projected.begin();cur != projected.end(); ++cur){
			if(oid != cur->id){
				if(cur!=projected.begin()) *os<< " ";fos<< " ";
				*os<<cur->id;
				fos<<cur->id;
			}
			oid = cur->id;
		}
		/*
		*os << "</where>\n</pattern>";
		fos << "</where>\n</pattern>";
		*/
	}
	*os<<"\n";
	//fos<<"\n";
	++ID;
}


void gSpan::project(Projected& projected) //Đây là phương thức của lớp gSpan, đối số của nó là một vector<PDFS> hay là Projected. Các phần tử trong vector<PDFS> là những cạnh giống nhau nhưng ở các đồ thị khác nhau.
{
	unsigned int sup = support(projected); //Tính độ hỗ trợ của cạnh đó rồi lưu kết quả vào biến sup.
	if(sup<minsup) //nếu như độ hỗ trợ của cạnh đang xét nhỏ hơn minsup được truyền vào bởi người dùng thì return
		return;

	if(is_min() == false){ //kiểm tra xem DFSCode có phải là nhỏ nhất hay không? Nếu không, tức là == false thì return
		//*os << "NOT MIN ["; 
		//DFS_CODE.write (*os);
		//*os << "]" << std::endl;
		return;	//return nếu DFSCODE của cạnh không phải là nhỏ nhất.
	}

	//std::cout << "project report" << std::endl;
	report(projected,sup); //Hàm report trả về kết quả các đồ thị phổ biến thỏa minsup đã cho.
	
	if(maxpat_max>maxpat_min && DFS_CODE.nodeCount() > maxpat_max)
		return;

	const RMPath& rmpath = DFS_CODE.buildRMPath(); //tìm Right Most Path của DFS code
	int minlabel = DFS_CODE[0].fromlabel; //nhãn đỉnh bé nhất
	int maxtoc = DFS_CODE[rmpath[0]].to; //Đây có thể là đỉnh phải nhất trên right most path của DFS_CODE; được sử dụng để tìm backward edge extensions

	Projected_map3 new_fwd_root;
	Projected_map2 new_bck_root;
	EdgeList edges;

	for(unsigned int n = 0; n < projected.size(); ++n) //duyệt qua tất cả các cạnh
	{
		unsigned int id = projected[n].id; //lấy graph_id
		PDFS* cur = & projected[n]; //cur: xét cạnh hiện tại ở một đồ thị nào đó
		History history(TRANS[id],cur); //!*o*lịch sử của cạnh đó trong đồ thị đó là gì?

		for(int i = (int) rmpath.size()-1; i>=1 ; --i){ //nếu right most path chỉ có 1 cạnh thì chúng ta không thể mở rộng backward edge cho các đỉnh trên right most path đó.
			Edge* e = get_backward(TRANS[id], history[rmpath[i]],history[rmpath[0]],history); // Ngược lại chúng ta mở rộng backward edge cho các đỉnh trên right most path

			if(e)
				new_bck_root[DFS_CODE[rmpath[i]].from][e->elabel].push(id,e,cur);
		}

		if(get_forward_pure(TRANS[id], history[rmpath[0]], minlabel, history, edges)) //tìm forward edge từ đỉnh phải cùng của DFSCODE, kết quả lưu vào edges
			for(EdgeList::iterator it = edges.begin();it != edges.end(); ++it) //duyệt qua tất cả các cạnh vừa tìm được. 
				new_fwd_root[maxtoc][(*it)->elabel][TRANS[id][(*it)->to].label].push(id,*it,cur); //xây dựng nhánh mới trên root

		for(int i=0;i<(int)rmpath.size();++i) //mở rộng cạnh cho các đỉnh thuộc right most path lần lượt từ dưới lên
			if(get_forward_rmpath(TRANS[id],history[rmpath[i]],minlabel,history,edges))
				for(EdgeList::iterator it = edges.begin();it!=edges.end(); ++it)
					new_fwd_root[DFS_CODE[rmpath[i]].from][(*it)->elabel][TRANS[id][(*it)->to].label].push(id,*it,cur);

	}

	for(Projected_iterator2 to = new_bck_root.begin();to!=new_bck_root.end();++to){
		for(Projected_iterator1 elabel= to->second.begin();elabel!=to->second.end();++elabel){
			DFS_CODE.push(maxtoc, to->first,-1,elabel->first,-1);
			project(elabel->second);
			DFS_CODE.pop();
		}
	}

	for(Projected_riterator3 from = new_fwd_root.rbegin();from!=new_fwd_root.rend(); ++from)
	{
		for(Projected_iterator2 elabel = from->second.begin();elabel != from->second.end(); ++elabel)
		{
			for(Projected_iterator1 tolabel = elabel->second.begin();tolabel!=elabel->second.end();++tolabel)
			{
				DFS_CODE.push(from->first,maxtoc+1, -1, elabel->first, tolabel->first);
				project (tolabel->second);
				DFS_CODE.pop();				
			}
		}
	}

	return;																	  
}

void gSpan::run(char* fname, std::ostream& _os, unsigned int _minsup, unsigned int _maxpat_min, unsigned int _maxpat_max, bool _enc, bool _where, bool _directed)
{
	//os = & _os; //result.txt
	ID = 0; //ID của đồ thị trong CSDL đồ thị
	minsup = _minsup; //2
	maxpat_min = _maxpat_min; //2
	maxpat_max = _maxpat_max; //0

	enc = _enc; //false
	where = _where; //true
	directed = _directed; //false
	read(fname); //fname: chemical_340. Phương thức read() này dùng để xây dựng cơ sở dữ liệu đồ thị TRANS. TRANS là một vector<graph> với mỗi phần tử là một đồ thị. 
	//run_intern();
}

void gSpan::run_intern(void)
{
	if(maxpat_min <= 1){ //!maxpat_min là gì? Ý nghĩa của nó thật sự là gì?
		for(unsigned int id = 0; id<TRANS.size(); ++id) //duyệt qua tất cả các đồ thị trong cơ sở dữ liệu đồ thị TRANS
		{

			for(unsigned int nid = 0;nid<TRANS[id].size();++nid)
			{
				if(singleVertex[id][TRANS[id][nid].label]==0)
					singleVertexLabel[TRANS[id][nid].label] += 1;
				singleVertex[id][TRANS[id][nid].label] += 1;
			}
		}

		for(std::map<unsigned int,unsigned int>::iterator it = singleVertexLabel.begin();it!=singleVertexLabel.end();++it)
		{
			if((*it).second < minsup)
				continue;
			unsigned int frequent_label = (*it).first;
			Graph g(directed);
			g.resize(1);
			g[0].label = frequent_label;

			std::vector<unsigned int> counts(TRANS.size());
			for(std::map<unsigned int, std::map<unsigned int, unsigned int> >::iterator it2 = singleVertex.begin(); it2 != singleVertex.end(); ++it2)
				counts[(*it2).first] = (*it2).second[frequent_label];

		}
	}

	EdgeList edges; //edges là một vector<Edge*> mà mỗi phần tử của nó là một con trỏ Edge dùng để lưu địa chỉ của một edge.

	Projected_map3 root; //root là một ánh xạ với các ánh xạ lồng bên trong nó tạo thành một mắc xích (<int,(<int,(<int,projected>)>)>)

	for(unsigned int id=0; id< TRANS.size(); ++id) //duyệt qua các đồ thị trong cơ sở dữ liệu đồ thị TRANS
	{
		Graph& g = TRANS[id]; //khai báo g làm một alias của TRANS[id], tức g là một đồ thị đang được xem xét. Mỗi vòng lặp thì g tương ứng với một đồ thị khác nhau trong cơ sở dữ liệu đồ thị TRANS.
		for(unsigned int from = 0; from < g.size(); ++from) //g.size() là số đỉnh trong đồ thị, ở đây có nghĩa là vòng lặp để duyệt qua tất cả các đỉnh trong đồ thị.
		{
			if(get_forward_root(g,g[from],edges)) //Tìm các forward_edge với 3 tham số (g: đồ thị đang xét; g[from]: đỉnh đang xét; edges: kết quả danh sách các forward_edge được tìm thấy)
			{
				for(EdgeList::iterator it = edges.begin(); it!= edges.end(); ++it) //duyệt qua danh sách kết quả các forward_edge đã tìm thấy trong đồ thị g
					root[g[from].label][(*it)->elabel][g[(*it)->to].label].push(id,*it,0); //tương ứng với mỗi forward_edge, chúng ta xây dựng được ánh xạ root. Tương ứng với mỗi đỉnh nó sẽ lưu các cạnh liên quan đến đỉnh đó; tương ứng với mỗi cạnh nó sẽ lưu danh sách các cạnh đó và graph_id mà cạnh đó thuộc. Thông tin mà mỗi cạnh lưu giữ là một danh sách liên kết.
			}
		}
	}

	for(Projected_iterator3 fromlabel = root.begin();fromlabel != root.end(); ++fromlabel)	//duyệt qua từng phần tử của root, tương ứng với mỗi nhãn đỉnh 'from'
	{
		for(Projected_iterator2 elabel = fromlabel ->second.begin();elabel != fromlabel -> second.end(); ++elabel) //Duyệt qua các second của root, chính là các edge_label, duyệt qua các nhãn cạnh gắn với nhãn đỉnh from.
			for(Projected_iterator1 tolabel = elabel->second.begin();tolabel != elabel -> second.end();++tolabel) //Duyệt qua các seconde của second của root, chính là các To-vertex-label, duyệt qua các nhãn đỉnh 'to' tương ứng với nhãn đỉnh from và nhãn cạnh
			{
				DFS_CODE.push(0,1,fromlabel->first,elabel->first,tolabel->first); //Xây dựng các DFSCode  cho cạnh. Mỗi cạnh khác nhau sẽ có một DFS_CODE tương ứng cho cạnh đó.
				project (tolabel->second); //hàm này làm các bước sau: 1. Kiểm tra độ hỗ trợ của DFS_CODE có thỏa mãn minsup? 2. Kiểm tra DFS_CODE có phải là nhỏ nhất? 3. Chuyển DFS_CODE thành đồ thị vào ghi đồ thị vào tập tin result.txt | 4. Tìm rightmostpath của DFS_CODE
				DFS_CODE.pop();				
			}
	}
}

unsigned int gSpan::findMaxVertices()
{
	int noOfVer=0;
	for(int i=0;i<TRANS.size();i++){
		if (TRANS[i].vertex_size()>noOfVer) noOfVer = TRANS[i].vertex_size();
	}	
	return noOfVer;
}

unsigned int gSpan::noGraphs()
{
	int noOfGraph=0;
	for(int i=0;i<TRANS.size();i++){
		noOfGraph++;
	}	
	return noOfGraph;
}

unsigned int gSpan::sumOfDeg()
{
	unsigned int resultTotalDeg=0;
	for(unsigned int id=0; id< TRANS.size(); ++id) //duyệt qua các đồ thị trong cơ sở dữ liệu đồ thị TRANS
	{
		Graph& g = TRANS[id]; //khai báo g làm một alias của TRANS[id], tức g là một đồ thị đang được xem xét. Mỗi vòng lặp thì g tương ứng với một đồ thị khác nhau trong cơ sở dữ liệu đồ thị TRANS.
		for(Graph::vertex_iterator v = g.begin(); v !=g.end(); ++v) //g.size() là số đỉnh trong đồ thị, ở đây có nghĩa là vòng lặp để duyệt qua tất cả các đỉnh trong đồ thị.
		{		
			for(Vertex::edge_iterator it = v->edge.begin();it!=v->edge.end();++it)
			{
				resultTotalDeg++;
			}
		}
	}
	return resultTotalDeg;
}

void gSpan::importDataToArray(int*& _arrayO,int*& _arrayLO,int*& _arrayN,int*& _arrayLN,const unsigned int _sizeOfarrayO,const unsigned int _noDeg,const unsigned int _maxOfVer) //return -1 if error
{
	int i=0;
	int numberOfEdges=0;//số cạnh
	int j=0;
	_arrayO[i]=0;
	for(unsigned int id=0; id< TRANS.size(); ++id) //duyệt qua các đồ thị trong cơ sở dữ liệu đồ thị TRANS
	{
		//i=id*_maxOfVer;
		Graph& g = TRANS[id]; //khai báo g làm một alias của TRANS[id], tức g là một đồ thị đang được xem xét. 
		int numberOfVer=0;	//Mỗi vòng lặp thì g tương ứng với một đồ thị khác nhau trong cơ sở dữ liệu đồ thị TRANS.
		numberOfVer=g.vertex_size();
		for(Graph::vertex_iterator v = g.begin(); v !=g.end(); ++v) //g.size() là số đỉnh trong đồ thị, ở đây có nghĩa là vòng lặp để duyệt qua tất cả các đỉnh trong đồ thị.
		{		
			for(Vertex::edge_iterator it = v->edge.begin();it!=v->edge.end();++it)//Duyệt qua các cạnh của đỉnh
			{
				_arrayLO[i]=g[it->from].label; //gán nhãn cho đỉnh From trong mảng LO, bị gán nhiều lần trong mỗi lần lặp cạnh không tốt
				_arrayN[j]=it->to+id*_maxOfVer; //gán id cho đỉnh to trong mảng N
				_arrayLN[j]=it->elabel; //gán nhãn cho cạnh
				j=j+1;				//tăng chỉ số trong mảng N và mảng LN
				++numberOfEdges; //số cạnh đã duyệt
			}
			if (i>=(_sizeOfarrayO-1)){
				return;
			}else if(i==((numberOfVer-1)+(id*_maxOfVer))){				
				_arrayO[((id+1)*_maxOfVer)]=numberOfEdges;
			}else _arrayO[i+1]=numberOfEdges;			
			++i;
		}
		i=(id+1)*_maxOfVer;
	}
}

void gSpan::graphMining(int* &O,int* &LO,int* &N,int* &LN,unsigned int minsup)
{
	cout<<"Start to mining";
	cout<<"........";


}


void gSpan::report(int *graph,int noEle_graph,unsigned int sup){
	//ofstream fout("result.txt");
	
	if(where){				
		*os << "t # " << ID << " " << sup << endl; //ghi đồ thị đó vào tập tin result.txt
	}

	if(!enc){ //enc default value is false
		Graph g(directed); //khởi tạo g là một đồ thị vô hướng
		DFS_CODE.toGraph(g); //Convert DFS code sang đồ thị.

		if(!where) //default value of where is true
			*os << "t # " << ID << " * " << sup;
		g.write(*os);
		
	}else{
		if(!where)
			*os << "<" << ID << "> " << sup << " [";
		DFS_CODE.write (*os);

		if(!where) *os << "]";
	}

	if(where) {		
		*os << "x ";
		unsigned int oid = 0xffffffff;
		/*for(Projected::iterator cur = projected.begin();cur != projected.end(); ++cur){
			if(oid != cur->id){
				if(cur!=projected.begin()) *os<< " ";fos<< " ";
				*os<<cur->id;
				fos<<cur->id;
			}
			oid = cur->id;
		}	*/	
		for (int i = 0; i < noEle_graph; i++)
		{
			if(i!=0)	*os<< " ";fos<<" ";
			*os<<graph[i];
			fos<<graph[i];
		}
	}
	*os<<"\n";	
	++ID;
}
