#include <iostream>
#include <string>
#include <vector>
#include "Reader.h"
#include "GraphCompressor.h"
using namespace std;

int nchoosek(int n, int k);
int getAllRecordArr(double *all, vector<Record> recs, int lsize);
void pairstimulant(double *all, double *allPairs, int allsize, int recordsize, int nk, int pairsize);
void printall(double *all, int lsize, int rsize);

int main(){

  vector<pair<string, string> > labels;
  getLabels(labels);
  record_data sd = getRecords(labels, "symbolic");

  int lsize = labels.size() - 1; //-1 to get rid of the attack type at the end
  int pairsize = 2;

  int rsize = sd.records.size();

  int nk = nchoosek(lsize, pairsize);

  int pairArrSize = nk * rsize;

  double pairArr[pairArrSize][2];
  double *allrecords = new double[lsize * rsize];

  int allsize = getAllRecordArr(allrecords, sd.records, lsize);

  printall(allrecords, lsize, rsize);

/*
  //pairstimulant(allrecords, pairArr, allsize, lsize, 2);

  double *sim = new double[8];
  sim[0] = 1;
  sim[1] = 3;
  sim[2] = 5;
  sim[3] = 9;

  sim[4] = 11;
  sim[5] = 13;
  sim[6] = 15;
  sim[7] = 19;

  int nk2 = nchoosek(4, 2);
  int psize = nk2 * 2;

  double *pairArr2 = new double[psize * 2];

  pairstimulant(sim, pairArr2, 8, 4, nk2, (psize*2));
*/
  return 0;

}

int getAllRecordArr(double *all, vector<Record> recs, int lsize){

   int i, j, index = 0;
   for(i = 0; i < recs.size(); i++){

     double *arr = recs[i].convertToArr();

     for(j = 0; j < lsize; j++){

       all[index] = arr[j];
       index++;

     }

   }

   return index + 1;

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

void pairstimulant(double *all, double *allPairs, int allsize, int recordsize, int nk, int pairsize){

  int i, j, l, index = 0;

  int recordcount = allsize / recordsize;

  for(i = 0; i < recordcount; i++){

    int start = recordsize * i; //lower bound
    int limit = (recordsize * i) + recordsize; //upper bound

    index = i * (nk * 2);

    for(j = start; j < limit; j++){

      for(l = (j + 1); l < limit; l++){ //j + 1 so we don't do pairs of itself

	allPairs[index] = all[j];
	allPairs[index + 1] = all[l];

	cout << "pair " << all[j] << ", " << all[l] << " copied to indices [" << index << ", " << index + 1 << "]\n";
	//cout << '(' << all[j] << ", " << all[l] << ")\n";

	index += 2; //move up two indices

      }

    }

    //cout << "\n----------------------------------------------------------\n";

  }

  for(i = 0; i < pairsize; i += 2){

    cout << '(' << allPairs[i] << ", " << allPairs[i + 1] << ")\n";

  }

}

void printall(double *all, int lsize, int rsize){

  int i;
  for(i = 0; i < (lsize * rsize); i++){

    cout << all[i] << ',';

    if(i % lsize == 0){

      cout << '\n';

    }

  }

  cout << '\n';

}
