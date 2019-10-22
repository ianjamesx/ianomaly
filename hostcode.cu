#include <iostream>
#include <string>
#include <vector>
#include "Reader.h"
#include "GraphCompressor.h"

#include "pairs.cuh"
using namespace std;

//set up
int nchoosek(int n, int k);
void getAllRecordArr(double *all, vector<Record> recs, int lsize);

int main(){

  vector<pair<string, string> > labels;
  getLabels(labels);
  record_data sd = getRecords(labels, "symbolic");

  int lsize = labels.size() - 1; //-1 to get rid of the attack type at the end
  int pairsize = 2;
  int rsize = sd.records.size();
  int nk = nchoosek(lsize, pairsize);
  int pairArrSize = nk * rsize;

  double pairArr[pairArrSize][pairsize];
  double *allrecords = new double[lsize * rsize];

  getAllRecordArr(allrecords, sd.records, lsize);

  return 0;

}

void getAllRecordArr(double *all, vector<Record> recs, int lsize){

   int i, j, index = 0;
   for(i = 0; i < recs.size(); i++){

     double *arr = recs[i].convertToArr();

     for(j = 0; j < lsize; j++){

       all[index] = arr[j];
       index++;

     }

   }

}

int nchoosek(int n, int k){

  if (k > n) return 0;
  if (k * 2 > n) k = n-k;
  if (k == 0) return 1;

  int result = n;
  for( int i = 2; i <= k; ++i ) {
      result *= (n-i+1);
      result /= i;
  }
  return result;

}

