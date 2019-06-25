/*
 *  gspan.h
 *  GSPAN
 *
 *  Created by Jinseung KIM on 09. 07. 18.
 *  Copyright 2009 KyungHee. All rights reserved.
 *
 */

#include<iostream>
#include<map>
#include<vector>
#include<set>
#include<algorithm>
#include<fstream>



template <typename T> inline void _swap(T &x,T &y){T z=x;x=y;y=z;} //Template để swap 2 phần tử có cùng kiểu dữ liệu T bất kỳ

struct Edge{ // Cấu trúc của một cạnh 
	int from; //từ đỉnh nào
	int to;	//đến đỉnh nào
	int elabel;	//nhãn cạnh 
	unsigned int id;	//mỗi cạnh đều có id duy nhất
	Edge(): from(0),to(0),elabel(0),id(0) {}; //khởi tạo cho cạnh 
};

//Đoạn code khởi tạo cạnh ở trên tương ứng với đoạn code sau đây
//Edge:Edge(){
//from = 0;
//to = 0;
//elabel = 0;
//id = 0;
//}

class Vertex
{
public:
	typedef std::vector<Edge>::iterator edge_iterator;	//định nghĩa tên kiểu dữ liệu mới ngắn gọn hơn www.stdio.vn/articles/read/165/typedef-va-enum
	
	int label; // nhãn đỉnh	
	std::vector<Edge> edge;  //mỗi phần tử của vector là một edge, ở đây cấu trúc edge là một thành phần của class Vertex
	
	void push(int from,int to, int elabel) // thêm một cạnh vào vector edge
	{
		edge.resize(edge.size()+1);		//phải tăng kích thước của edge lên 1
		edge[edge.size()-1].from = from;	//thông tin cạnh từ đỉnh nào
		edge[edge.size()-1].to = to;		//đến đỉnh nào
		edge[edge.size()-1].elabel = elabel;	//nhãn cạnh là gì
		return;		
	}
};

class Graph:public std::vector<Vertex>{ //Lớp Graph kế thừa lớp vector<Vertex>, tức khi bạn tạo một thể hiện Graph thì hệ thống cũng tạo ra một vector Graph
public:
	unsigned int edge_size_; //đồ thị có bao nhiêu cạnh	
public:
	
	typedef std::vector<Vertex>::iterator vertex_iterator; //định nghĩa kiểu dữ liệu iterator cho Vertex
	
	Graph(bool _directed)	//Đồ thị có hướng hay vô hướng.
	{
		directed = _directed;
	};
	bool directed;	//chỉ hướng của đồ thị
	
	unsigned int edge_size(){ return edge_size_;} //trả về số cạnh của đồ thị
	unsigned int vertex_size() { return (unsigned int)size(); } //wrapper function, trả về số đỉnh của đồ thị
	
	
	void buildEdge(); //khai báo hàm xây dựng các cạnh cho đồ thị
	int read (char*,int); //khai báo hàm đọc 
	std::ostream& write (std::ostream&); //khai báo hàm chỉ ghi
	std::ofstream& write(std::ofstream&); //khai báo hàm đọc và ghi file
	void check(void); //khai báo hàm kiểm tra
	
	Graph():edge_size_(0),directed(false){}; //khởi tạo đồ thị với cạnh và mặc định là vô hướng
};
//use
class DFS{	//Lớp DFS, biểu diễn DFS code của một cạnh. Gồm các cả overloading operator dùng để so sánh DFS code của 2 cạnh.
public:
	int from; //từ đỉnh nào
	int to; //đến đỉnh nào
	int fromlabel; //từ nhãn đỉnh nào
	int elabel;	//nhãn cạnh
	int tolabel; //đến nhãn đỉnh nào
	//overloading operator == để so sánh giữa 2 DFS code d1 và d2
	friend bool operator == (const DFS& d1,const DFS& d2){
		return (d1.from == d2.from && d1.to == d2.to && d1.fromlabel == d2.fromlabel && d1.elabel == d2.elabel && d1.tolabel == d2.tolabel);
	}
	//overloading operator != để so sánh giữa 2 DFS code d1 và d2
	friend bool operator != (const DFS& d1,const DFS& d2){
		return (!(d1==d2));
	}
	//Khởi tạo DFS code
	DFS():from(0),to(0),fromlabel(0),elabel(0),tolabel(0){};
};
//use
typedef std::vector<int> RMPath; //định nghĩa một kiểu vector<int> có tên là RMPath dùng để chứa Right Most Path

//cấu trúc của DFS code
struct DFSCode:public std::vector<DFS>{ //kế thừa từ vector<DFS> //use
public:
	int minLabel;
	int maxId;
	int noElemOnRMP;
	int *hRMP;
	int *dRMP;
	RMPath rmpath;  //right most path là một vector đã định nghĩa ở trên
public:
	void add(int,int,int,int,int);
	//use
	void remove(int,int);
	const RMPath& buildRMPath(); //xây dựng right most path tức là tìm nhánh phải nhất của DFSCode
	
	bool toGraph(Graph&); //*0* không biết để làm gì; có thể là để biết GID của graph
	void fromGraph(Graph& g); //*0* không biết để làm gì
	
	unsigned int nodeCount(void); //đếm số node của DFSCode
	void push(int from,int to,int fromlabel,int elabel,int tolabel){ //thêm DFS vào DFSCode
		resize(size()+1); //vì DFSCode kế thừa từ vector<DFS> nên DFSCode có thể dùng phương thức size() của vector<DFS>
		DFS& d = (*this)[size()-1]; //DFSCode chẳng qua là vector<DFS>: đây là ý nghĩa của kế thừa
		
		d.from = from;
		d.to = to;
		d.fromlabel = fromlabel;
		d.elabel = elabel;
		d.tolabel = tolabel;
	}
	
	void pop(){ resize (size()-1);} //lấy phần từ cuối ra
	std::ostream& write(std::ostream &); //*0* Hàm ghi nhưng không biết là để ghi cái gì; chắc là ghi cập nhật DFS code mới xuống tập tin.
};

//*0* cấu trúc PDFS dùng để làm gì?; đây là một cấu trúc tự trỏ hay gọi là danh sách liên kết, mỗi phần tử của nó là |(id,*edge);(*prev)|
//Có thể cấu trúc PDFS dùng để cho biết cạnh phía trước của một cạnh trên DFSCode là gì, biết mỗi cạnh có một id duy nhất
struct PDFS{	//PDFS là một cấu trúc tự trỏ(hay nói cách khác nó là một danh sách liên kết) gồm có 3 phần tử
	unsigned int id;	//Mỗi một node của danh sách liên kết có dữ liệu là id và *edge.
	Edge *edge;	//con trỏ edge, cùng để lưu trữ địa chỉ của một cạnh
	PDFS *prev;	//Trỏ đến node trước đó. Nếu tưởng tượng nó là tree, thì khi thêm một node mới, node mới sẽ trỏ đến node cũ tức là node phía trên của node vừa thêm.
	PDFS():id(0),edge(0),prev(0){};
};
//History chính là bản sao của đồ thị (Graph& g). Bản sao này ghi nhận những cạnh đã được mở rộng hay nói cách khác là chúng đã thuộc vào embedding (PDFS* P). Do đó, khi tìm các mở rộng fw/bw thì ta tránh các cạnh đã thuộc embeddings.
class History:public std::vector<Edge*>{
public:
	std::vector<int> edge;
	std::vector<int> vertex;
public:
	bool hasEdge(unsigned int id){return (bool)edge[id];}
	bool hasVertex(unsigned int id){return (bool)vertex[id];}
	void build(Graph&,PDFS*);
	History(){};
	History(Graph& g,PDFS* p){build(g,p);} //Constructor có tham số thứ 1 tham chiếu đến đồ thị TRANS[i], tham số thứ 2 là trỏ đến projected[i], hàm build sẽ được kích hoạt với 2 tham số này.
};

class Projected:public std::vector<PDFS>{ //Projected là một đối tượng kế thừa từ lớp vector<PDFS> hay nói cách khác Projected là một vector mà mỗi phần tử của nó là PDFS
public:
	void push(int id,Edge* edge,PDFS* prev){ //phương thức push dùng để thêm một phần tử vào trong Projected, tức là vector<PDFS>, vì Projected kế thừa từ Vector<PDFS>, mỗi phần tử của vector là một PDFS, tức là một danh sách tự trỏ hay còn gọi là danh sách liên kết.
		resize(size()+1); //Mở rộng 1 phần tử cho vector<PDFS>
		PDFS& d = (*this)[size()-1]; //d là một alias của phần tử  trong vector<PDFS> vừa mới mở rộng.
		d.id = id; //từ d ta truy xuất và gán giá trị cho nó. Gồm id của cạnh, edge, và liên kết thông tin của phần tử trước đó là gì
		d.edge = edge;
		d.prev = prev;
	}
};

typedef std::vector<Edge*> EdgeList; //EdgeList là một kiểu dữ liệu vector<Edge*> với mỗi phần tử của nó là một con trỏ kiểu Edge dùng để lưu trữ địa chỉ của một edge (edge là một structure).

bool get_forward_pure(Graph&,Edge*,int,History&,EdgeList&); //tìm các mở rộng forward từ đỉnh cuối thuộc right most path của các embeddings
bool get_forward_rmpath(Graph&,Edge*,int,History&,EdgeList&);//tìm các mở rộng forward từ các đỉnh còn lại (trừ đỉnh cuối) thuộc rmp của các embeddings
bool get_forward_root(Graph&,Vertex&,EdgeList&); //tìm các mở rộng forward ban đầu
Edge* get_backward(Graph&,Edge*,Edge*,History&);//tìm các mở rộng backward

class gSpan{
public:
	typedef std::map<int,std::map<int,std::map<int,Projected> > >					Projected_map3; //Định nghĩa một kiểu dữ liệu mới tên là Projected_map3. Nó gồm có 3 ánh xạ lồng nhau (f1<first1,second1>) trong đó second1=(f2<first2,second2>) và second2=(f3<first3,second3>). Trong đó second3 = Projected
	typedef std::map<int,std::map<int,Projected> >									Projected_map2; //Định nghĩa một kiểu dữ liệu mới tên là Projected_map2. Nó gồm có 2 ánh xạ lồng nhau
	typedef std::map<int,Projected>													Projected_map1; //Định nghĩa một kiểu dữ liệu mới
	typedef std::map<int,std::map<int,std::map<int,Projected> > >::iterator			Projected_iterator3; //Định nghĩa các kiểu iterator để duyệt các phần tử trong ánh xạ
	typedef std::map<int,std::map<int,Projected> >::iterator						Projected_iterator2;
	typedef std::map<int,Projected>::iterator										Projected_iterator1;
	typedef std::map<int,std::map<int,std::map<int,Projected> > >::reverse_iterator	Projected_riterator3;
	
	std::vector<Graph>	TRANS;	//TRANS là transaction. Nó là một cơ sở dữ liệu đồ thị. Nó là một vector mà mỗi phần tử của nó là một đồ thị.
	DFSCode				DFS_CODE;	//đối tượng DFS_CODE là vector<DFS> và rightMostPath
	DFSCode				DFS_CODE_IS_MIN; //đối tượng DFS_CODE_IS_MIN
	Graph				GRAPH_IS_MIN; //đối tượng đồ GRAPH_IS_MIN, là vector<vertex> + directed: cho biết hướng của đồ thị + edge_size_: là số cạnh của đồ thị.
	
	unsigned int ID; //Mỗi đồ thị đều có một id duy nhất, ID được tính từ 0, Đây là ID của đồ thị kết quả.
	unsigned int minsup;	//Đây là minsup do người dùng truyền vào
	unsigned int maxpat_min;
	unsigned int maxpat_max;
	
	
	bool where;
	bool enc;
	bool directed;
	std::ostream* os;
	std::ofstream fos; //fos là pointer trỏ tới tập tin /result.graph
	
	std::map<unsigned int,std::map<unsigned int,unsigned int> > singleVertex;
	std::map<unsigned int,unsigned int> singleVertexLabel;
	
	bool is_min();
	bool project_is_min(Projected&);
	
	std::map<unsigned int,unsigned int> support_counts(Projected& projected);
	unsigned int support(Projected&);
	void project (Projected&);
	void report (Projected&,unsigned int);
	void report(int*,int ,unsigned int);
	
	void read(char*);
	void run_intern(void);
public:
	gSpan(void);
	~gSpan(void);
	void run(char* fname,std::ostream& _os,unsigned int _minsup,unsigned int _maxpat_min,unsigned int _maxpat_max,bool _enc,bool _where,bool _directed);
	unsigned int findMaxVertices(); //Trả về số lượng đỉnh nhiều nhất của đồ thị có đỉnh nhiều nhất.
	unsigned int noGraphs(); //Trả về số lượng đồ thị trong CSDL TRANS
	unsigned int sumOfDeg(); //Trả về tổng bậc của tất cả các đỉnh trong CDSL TRANS
	void importDataToArray(int*& _arrayO,int*& _arrayLO,int*& _arrayN,int*& arrayLN,const unsigned int _sizeOfarrayO,const unsigned int _noDeg,const unsigned int _maxOfVer);
	void graphMining(int* &O,int* &LO,int* &N,int* &LN,unsigned int minsup);
};


