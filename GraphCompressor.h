#include <iostream>
#include <vector>

#ifndef GRAPHCOMPRESSOR_H
#define GRAPHCOMPRESSOR_H

using namespace std;

struct NV {
  
  int *offsets;
  int *indices;
  
  int v;
  int e;
  
};

typedef pair<int, int> edge; 

class GraphCompressor {
 
public:
  
  vector<vector<int> >vertices; //basically an adjacency list
				//but, the vertex itself will be the index of its list
  int v, e;
  bool directional;
  bool weighted;
  
  GraphCompressor(){
    v = e = 0;
    directional = false;
  }
  
  GraphCompressor(bool dir){
    v = e = 0;
    directional = dir;
  }
  
  int addVertex();
  void addEdge(int start, int end);
  void printList();
  
  NV getCSR();
  NV getCSC();
  
};

#endif