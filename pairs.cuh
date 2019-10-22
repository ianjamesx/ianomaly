#include <iostream>

__global__
void add(float *all, float allPairs[][2], int recordcount, int recordsize, int pairsize){
  
  int i, j;
  for(i = 0; i < recordcount; i++){
    
    for(j = i; j < recordcount; j++){
      
      
      
    }
    
  }
  
  /*
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int i = index; i < n; i += stride){
    y[i] = x[i] + y[i];
  }
  */

}

__global__
void counter(int &threads, int &blocks){
  
  threads++;
  blocks++;
  
}

int main(){
  
  int t = 0, b = 0;
  
  counter(t, b);
  
  cout << b << " - " << t << endl;
  
  return 0;
  
}



void pairstimulant(double *all, double allPairs[][2], int allsize, int recordsize, int nk, int pairsize){
  
  int i, j, l, index = 0;
  
  int recordcount = allsize / recordsize;
  
  for(i = 0; i < recordcount; i++){

    int start = recordsize * i; //lower bound
    int limit = (recordsize * i) + recordsize; //upper bound
    
    index = i * nk;
  
    for(j = start; j < limit; j++){
    
      for(l = (j + 1); l < limit; l++){ //j + 1 so we don't do pairs of itself
	
	allPairs[index][0] = all[j];
	allPairs[index][1] = all[l];
	
	//cout << "pair " << all[j] << ", " << all[l] << " copied to index " << index << "\n";
	//cout << '(' << all[j] << ", " << all[l] << ")\n";
	
	index++; //move up one index
      
      }
      
    }
    
    //cout << "\n----------------------------------------------------------\n";
    
  }
  
  for(i = 0; i < pairsize; i++){
    
    cout << '(' << allPairs[i][0] << ", " << allPairs[i][1] << ")\n";
    
  }
  
}
