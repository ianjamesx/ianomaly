#include "GraphCompressor.h"
#include <iostream>
#include <vector>
#include <set>
using namespace std;

int GraphCompressor::addVertex(){
  
  vector<int> temp;
  
  vertices.push_back(temp);
  return ++v;

}

void GraphCompressor::addEdge(int start, int end){
  
  vertices[start].push_back(end);
  
  if(!directional) vertices[end].push_back(start); //add backwards one if non directional
  
  e++;
  
}

void GraphCompressor::printList(){
  
  int i, j;
  for(i = 0; i < v; i++){
    
    for(j = 0; j < vertices[i].size(); j++){
      
      cout << vertices[i][j] << ", ";
      
    }
    
    cout << "\n";
    
  }
  
}

NV GraphCompressor::getCSR(){
  
  NV CSR;
  
  int *source_offsets = new int[v + 1], 
      *destination_indices = new int[e];
      
  set<edge> allEdges; //set to hold all the edges
  
  int i, j;
  int si = 0, di = 0; //source_offsets and destination_indices indexes
  
  source_offsets[0] = 0; //initialize first offset to 0
 
  for(i = 0; i < vertices.size(); i++){
    
    int totalEdges = 0;
    
    for(j = 0; j < vertices[i].size(); j++){
      
      //get the current edge based on the vertex (i) and the list index (vertices[i][j])
      
      edge temp(i, vertices[i][j]);
      
      //if edge not yet in set
      if(!(allEdges.find(temp) != allEdges.end())){
	
	//insert first edge
	allEdges.insert(temp);
	
	//also insert reversed edge (if non-directional)
	if(!directional){
	  edge reversed(vertices[i][j], i);
	  allEdges.insert(reversed);
	}
	
	totalEdges++;
	
	destination_indices[di] = temp.second;
	di++;
	
      }
      
    }
    
    si = i + 1;
    source_offsets[si] = di;
    
  }
  
  source_offsets[v] = e; //put number of edges at end
  
  CSR.offsets = source_offsets;
  CSR.indices = destination_indices;
  CSR.v = v;
  CSR.e = e;
  
  return CSR;
  
}

NV GraphCompressor::getCSC(){
  
  NV CSC;
  
  //similar to last approach, except reverse the source vertex
  //and the elements in the list for each vertex
  
  int *destination_offsets = new int[v + 1], 
      *source_indices = new int[e];
      
  set<edge> allEdges;
  
  int i, j;
  int si = 0, di = 0; 
  
  destination_offsets[0] = 0;
 
  for(i = 0; i < vertices.size(); i++){
    
    int totalEdges = 0;
    
    for(j = 0; j < vertices[i].size(); j++){
      
      edge temp(vertices[i][j], i);
      
      if(!(allEdges.find(temp) != allEdges.end())){
	
	allEdges.insert(temp);
	
	//if non-directional
	if(!directional){
	  edge reversed(i, vertices[i][j]);
	  allEdges.insert(reversed);
	}
	
	totalEdges++;
	
	source_indices[di] = temp.second;
	di++;
	
      }
      
    }
    
    si = i + 1;
    destination_offsets[si] = di;
    
  }
  
  destination_offsets[v] = e; //put number of edges at end
  
  CSC.offsets = destination_offsets;
  CSC.indices = source_indices;
  CSC.v = v;
  CSC.e = e;
  
  return CSC;
  
}
