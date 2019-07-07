/*
 *  graph.cpp
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
#include <strstream>
#include <set>
#include <fstream>
using namespace std;
template <typename T,typename I>
void tokenize(const char* str,I iterator)
{
	std::istrstream is (str,std::strlen(str));
	std::copy (std::istream_iterator <T> (is), std::istream_iterator <T> (), iterator);
}

void Graph::buildEdge() //Hàm này vừa xây dựng ánh xạ tmp<string,int> (string:from-to-elabel, int:id_edge) và cập nhật id của cạnh và số lượng cạnh của một đồ thị
{
	char buf[512];	//khai báo vùng đệm có kích thước 512 Bytes
	std::map <std::string,unsigned int> tmp;	//tmp là một ánh xạ map <string, unsigned int>. Ở đây mượn ánh xạ tmp để cập nhật edge_id của vector<edge> cho nhanh.
	
	unsigned int id=0;
	for(int from=0;from<(int) size();++from) { //duyệt qua các đỉnh trong đồ thị, tương ứng với mỗi đỉnh, sẽ duyệt qua các cạnh của nó, và set id cho cạnh. Id của cạnh tính từ 0.
		for(Vertex::edge_iterator it = (*this)[from].edge.begin();it!=(*this)[from].edge.end();++it) //it là một edge_iterator dùng để duyệt qua các phần tử trong vector<edge>
		{
			if(directed||from<=it->to) //nếu là đồ thị vô hướng hoặc id từ đỉnh đến 'from' nhỏ hơn id của 'to'
				std::sprintf(buf,"%d %d %d",from,it->to,it->elabel); //thì bỏ nó vào bộ nhớ đệm buf theo thứ tự from - to - elabel
			else
				std::sprintf(buf,"%d %d %d",it->to,from,it->elabel); //ngược lại, nếu là đồ thị có hướng thì bỏ vào buf theo thứ tự 'to'-'from'-'elabel'
			
			if(tmp.find(buf) == tmp.end()){ //kiểm tra xem cạnh của đỉnh đang xét đã tồn tại trong ánh xạ tmp hay chưa? Nếu chưa tồn tại thì thêm vào.
				it->id = id; //cập nhật id cho cạnh
				tmp[buf]=id; //thêm cạnh đó vào ánh xạ.
				++id; //tăng id lên 1
			}else{
				it->id = tmp[buf]; //ngược lại, nếu cạnh đó đã được đưa vào ánh xạ rồi thì cập nhật lại id cho cạnh là đủ. Ở đây buf là key của ánh xạ tmp, nên tmp[buf] sẽ trả về id của tmp.
			}
			
		}
	}
	edge_size_=id; //Số lượng cạnh của đồ thị
}

int Graph::read (char* _fname,int line_offset)
{
	std::vector <std::string> result; //result là một vector mà mỗi phần tử của nó có kiểu string
	char line[1024]; //line là một mảng chứa tối đã 1024 kí tự
	int linecnt=0;	//đếm số lượng dòng đọc được trong tập tin chemical_340
	
	std::ifstream is; //dùng để đọc tập tin chemical_340
	
	is.open(_fname,std::ios::in); //mở file chemical_340 để đọc
	
	if(!is) //kiểm tra nếu không đọc được file thì báo lỗi và trả về giá trị -1
	{
		std::cerr<<"file open fail!!"<<std::endl;
		return -1; 
	}else{
		linecnt = line_offset; //nếu đọc được file thì ghi nhận lại vị trí muốn đọc từ đâu
	}
	
	clear(); //phương thức của vector, dùng để xóa các phần tử của vector<vertex>. Ở đây mình ngầm định là có con trỏ this đang trỏ tới lệnh clear(). This ở đây chính là đối tượng g của class Graph::vector<vertex>
	
	if(line_offset>0)//nếu line_offset > 0, có nghĩa là bạn đã đọc file tới vị trí line_offset đó rồi, và trong lần đọc tiếp theo bạn cần phải đi đến vị trí đó để đọc tiếp.
		for(int i=0;i<line_offset;++i){
			is.getline(line,1024); //!Tại sao nó không di chuyển thẳng tới vị trí cần đọc file để đọc và làm tiếp mà nó cứ đọc từng dòng từ trên xuống? Có thể cải tiến lại hành động này được không? Vì để như thế này sẽ rất chậm.
		}
	
	while(true){
		//std::cout<<"read in : ";
		unsigned int pos = is.tellg(); //tellg() trả về vị trí hiện tại đang đọc
		if(! is.getline(line,1024)) //Nếu không lấy được dòng dữ liệu bỏ vào line thì hiển thị thông báo lỗi và thoát
		{
			cerr << "getline error" << endl;
			break;
		}
		linecnt ++; //ngược lại thì tăng linecnt lên 1, ý muốn nói đã đọc được 1 dòng
		
		result.clear(); //xóa các phần tử string trong vector result
		tokenize<std::string>(line,std::back_inserter(result)); //phân tích dòng vừa đọc thành các từ 'word' để lưu vào vector result
		
		//std::cout<<"size:"<<size()<<std::endl;
		
		if(result.empty()){ //nếu result rỗng thì không làm gì cả
			//no action
		}else if (result[0] == "t"){ //nếu phần tử đầu tiên của result là t thì làm, ý muốn nói đây là bắt đầu 1 đồ thị ngược lại kiểm tra xem có bằng v hay không
			//cout << result[2] << endl;
			//std::cout<< "t in : ";
			if(!empty()){ //chắc là g đã khác rỗng, vì nó đã có một đồ thị rồi.
				is.seekg(pos,std::ios_base::beg); //dời con trỏ đọc file pos lên trên đầu file.
				//std::cout<<"break";
				break; //sau đó thoát khỏi vòng lặp while để xây dựng đồ thị g vừa mới đọc từ file chemical_340, tức là thực thi lên ở dòng 128 trong file này.
			}else{
				//no action
				//continue;
			}
		}else if (result[0] == "v" && result.size()>=3) { //nếu biến result[0] là v, ý muốn nói đây là đỉnh
			unsigned int id = atoi(result[1].c_str()); //chuyển phần tử thứ 2 của vector result thành số nguyên, sau đó gán giá trị vào biến id, vertex_id
			this->resize(id+1); //con trỏ this ở đây chính là graph g, g chính là một vector<vertex>. Do đó, câu lệnh này là tăng kích thước vector<vertex> lên 1.
			(*this)[id].label = atoi(result[2].c_str()); //Cập nhật giá trị nhãn đồ thị tại phần tử thứ i của vector<vertex>. Lưu ý, ở đây chỉ số của vector<vertex> cũng chính là id của đỉnh đó trong đồ thị. Vì thế nên ta chỉ cần lưu label của đỉnh là đủ.
			//std::cout<<"v "<< atoi(result[1].c_str()) << " " << atoi(result[2].c_str()) << " " << (*this)[id].label<<" added"<<std::endl;
		}else if (result[0] == "e" && result.size()>=4) {
			int from = atoi(result[1].c_str());
			int to = atoi(result[2].c_str());
			int elabel = atoi(result[3].c_str());
			//std::cout<<"size:"<<(int)size();
			if((int)size()<=from || (int)size() <= to){ //nếu như 'from' hoặc 'to' mà lớn hơn hoặc bằng kích thước của vector<vertex> thì xem như là không hợp lệ
				std::cerr<<"Input Format Error: define vertex lists before edges" << std::endl;
				exit(-1);
			}
			
			(*this)[from].push(from,to,elabel);
			if(directed==false){
				(*this)[to].push(to,from,elabel);
			}
			
			//std::cout<<"edge added"<<std::endl;
		}
	} //kết thúc 
	buildEdge(); //Đây là một phương thức của class Graph
	
	return linecnt; //sau khi buildEdge() xong, nó sẽ trả lại vị trí mà nó đang đọc đồ thị và trả quyền điều kiển về cho hàm gspan::read()
}

std::ostream& Graph::write(std::ostream& os) //ghi kết quả xuống file result.txt
{
	char buf[512];
	std::set <std::string> tmp;
	
	for(int from = 0; from < (int)size();++from){ //duyệt qua các đỉnh của đồ thị g
		os<<"v "<< from << " " << (*this)[from].label << std::endl; //ghi đỉnh đó vào file result.txt
		
		for(Vertex::edge_iterator it = (*this)[from].edge.begin();it!=(*this)[from].edge.end();++it)
		{
			if(directed||from <= it->to){ //nếu là đồ thị vô hướng hoặc from<=to
				std::sprintf(buf,"%d %d %d",from,it->to,it->elabel);
			}else{
				std::sprintf(buf,"%d %d %d",it->to,from,it->elabel);
			}
			tmp.insert(buf); //chèn cạnh đó vào tập hợp (set) tmp
		}
	}
	
	for(std::set<std::string>::iterator it = tmp.begin();it!=tmp.end();++it){ //duyệt qua tất cả các cạnh trong set tmp
		os << "e " << *it << std::endl; //ghi cạnh đó vào tập tin result.txt
	}
	
	return os;
}

std::ofstream& Graph::write(std::ofstream& os)
{
	char buf[512];
	std::set <std::string> tmp;
	
	for(int from = 0; from < (int)size();++from){
		os<<"v "<< from << " " << (*this)[from].label << std::endl;
		
		for(Vertex::edge_iterator it = (*this)[from].edge.begin();it!=(*this)[from].edge.end();++it)
		{
			if(directed||from <= it->to){
				std::sprintf(buf,"%d %d %d",from,it->to,it->elabel);
			}else{
				std::sprintf(buf,"%d %d %d",it->to,from,it->elabel);
			}
			tmp.insert(buf);
		}
	}
	
	for(std::set<std::string>::iterator it = tmp.begin();it!=tmp.end();++it){
		os << "e " << *it << std::endl;
	}
	
	return os;
}

void Graph::check(void)
{
	for(int from = 0; from < (int)size();++from){
		std::cout << "check vertex " << from << ",label " << (*this)[from].label << std::endl;
		for(Vertex::edge_iterator it = (*this)[from].edge.begin();it!=(*this)[from].edge.end();++it)
		{
			std::cout << "check edge from " << it->from << " to " << it->to << ", label " << it->elabel << std::endl; 
		}
	}
}
