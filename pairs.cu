#include <iostream>
#include <cmath>
#include <vector>
#include <ctime>
#include <utility>
#include <fstream>
#include "Record.cuh"
#include "Reader.cuh"
using namespace std;

const int BLOCKS = 256, THREADS = 768;

int nchoosek(int n, int k);
int getAllRecordArr(double *all, vector<Record> recs, int lsize);
int getAllRecordArr_round(double *all, vector<Record> recs, int lsize, int maxrecs, int round);
void pairstimulant(double *all, double *allPairs, int allsize, int recordsize, int nk, int pairsize);
void printall(double *all, int lsize, int rsize);
void printtofile(double *p, double *a, int psize, int asize);
void printMemoryStats();
pair<double, double> getMemData();
double getMemoryAllocation(int size);


static void HandleError(cudaError_t err, const char *file, int line){

    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );

    }
}
#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))

__global__
void initMap(double *pairs, int allpairsize, int nk, int ops, int threadsperpair){

  int nkspace = (nk * 2);
  int thread = threadIdx.x, block = blockIdx.x, dim = blockDim.x;

  int id = (dim * block) + thread; //id of this thread

  int range = id / threadsperpair; //range this squad will cover
  int localId = id - (threadsperpair * range); //id for this squad

  if(thread % 2 != 0) return; //<-- highly innefficient, fix this immediately

  int i, j, init = (nkspace * threadsperpair),
      start = (init * localId) + range,
      end = (nkspace * ops) + start;

  //printf("id: %d, will cover %d through %d, Range %d, localId %d\n", id, start, end, range, localId);

  for(i = start; i <= end; i += nkspace){

    for(j = (i + nkspace); j <= end; j += nkspace){

      if(i >= allpairsize || j >= allpairsize) continue;

      if(pairs[i] == pairs[j] && pairs[i + 1] == pairs[j + 1]){

        //printf("DUPLICATE LOCATED (%f, %f) == (%f, %f), %d, %d & %d, %d...%d, %d, %d\n", pairs[i], pairs[i + 1], pairs[j], pairs[j + 1], i, i + 1, j, j + 1, id, start, end);

      }

      //printf("comparing index %d, index %d, (%f, %f) & (%f, %f), on thread %d\n", i, j, pairs[i], pairs[i + 1], pairs[j], pairs[j + 1], id);

    }

  }

}

__global__
void pairNK(double *all, double *allPairs, int allsize, int recordsize, int nk, int pairsize, int ops){

  int thread = threadIdx.x, block = blockIdx.x, dim = blockDim.x;
  int id = (dim * block) + thread; //id of this thread

  int low = (id * recordsize) * ops;
  int nkSpace = (nk * 2); //space needed for one set of pairs
  int nkLow = (id * nkSpace) * ops; //lowest index in allPairs that we need to write in

  int i, j, l;

  for(i = 0; i < ops; i++){

    int lowcurr = low + (recordsize * i),
        highcurr = lowcurr + recordsize;
    int currNkLow = nkLow + (nkSpace * i);

    if(highcurr > allsize) return; //break if we're out of bounds

    int index = currNkLow;

    for(j = lowcurr; j < highcurr; j++){

      for(l = (j + 1); l < highcurr; l++){ //j + 1 so we don't do pairs of itself

        if(index > pairsize) continue; //skip if we're out of bounds

        //printf("Moving pair: (%f, %f) to indices (%d, %d), THREAD %d\n", all[j], all[l], index, index + 1, id);

        allPairs[index] = all[j];
        allPairs[index + 1] = all[l];

        index += 2; //move up two indices

      }

    }

  }

}

int main(){

  float elapsed = 0;
  cudaEvent_t start, stop;

  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));

  HANDLE_ERROR( cudaEventRecord(start, 0));


  cout << "Reading in Labels...\n";
  vector<pair<string, string> > labels;
  getLabels(labels);
  cout << "Reading in all Network Records...\n";
  record_data sd = getRecords(labels, "symbolic");

  int lsize = labels.size() - 1; //-1 to get rid of the attack type at the end

  int rsize = sd.records.size();

  int nk = nchoosek(lsize, 2); //choose 2 elements per pairing

  double maxRecords = getMemoryAllocation(lsize);

  int rounds = ceil(rsize / maxRecords);
  cout << "Recommended records: " << maxRecords << endl;
  cout << "Rounds: " << rounds << endl;
  cout << "\n\n\n";


  int i;
  for(i = 0; i < rounds; i++){

    int recordcount = ceil(maxRecords);

    cout << "RECORDS TO PROCESS " << recordcount << "\n\n";

    int pairArrSize = nk * 2 * recordcount;

    double *allrecords = new double[lsize * recordcount];
    double *pairs = new double[pairArrSize];

    cout << "round: " << i << endl;
    cout << "pair array size: " << pairArrSize << endl;
    cout << "allrec size: " << (lsize * recordcount) << endl;
    cout << "\n\n\n";

    int allsize = getAllRecordArr_round(allrecords, sd.records, lsize, recordcount, i);

    //printall(allrecords, lsize, rsize);

    //determine how much work is to be done on each block/thread

    int totalWorkers = BLOCKS * THREADS;
    int totalOps = recordcount; //total operations will be number of records

    float opsPerWorker = totalOps / static_cast<float>(totalWorkers);
    if(opsPerWorker < 1) opsPerWorker = 1;

    int ops = ceil(opsPerWorker);

    //memory that will be shared (between host and device)

    double *a, *p; //a = all data, p = pairs of data
    int doublesize = sizeof(double);

    cudaMallocManaged(&a, (lsize * recordcount)*doublesize);
    cudaMallocManaged(&p, pairArrSize*doublesize);

    //populate a, then pairNK will populate p

    cout << "Copying records to GPU memory...\n";
    for(i = 0; i < (lsize * recordcount); i++){
      a[i] = allrecords[i];
    }

    cout << "Generating Pairs\nrecords: " << rsize << ",\npairsperrec: " << nk << ",\ntotal pairs: " << pairArrSize;
    cout << ",\nops per worker: " << ops << ",\ntotalworkers: " << totalWorkers << "\n\n";


    pairNK<<<BLOCKS,THREADS>>>(a, p, (lsize * rsize), lsize, nk, pairArrSize, ops);
    cudaDeviceSynchronize();

    //first map

    int threadsperpair = ceil(totalWorkers / nk); //ps
    int blackops = ceil(rsize / threadsperpair); //rs

    //we also need an array to hold repeated pairings
  /*
    int repeatedsize = ((nk * 3) * rsize);
    double *repeated;
    cudaMallocManaged(&repeated, repeatedsize*doublesize);
  */
    //initMap<<<3, 2>>>(parr, (rs * ps), ps, blackops, threadsperpair);

    cout << "NKSPACE " << (2 * nk) << ", threadsperpair " << threadsperpair << ", REDOPS " << blackops << endl;
    initMap<<<BLOCKS, THREADS>>>(p, pairArrSize, nk, blackops, threadsperpair);
    cudaDeviceSynchronize();

    printMemoryStats();

    cudaFree(a); //we no longer need a (???)
    cudaFree(p);

    delete [] allrecords;
    delete [] pairs;

  }


  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));

  HANDLE_ERROR(cudaEventElapsedTime(&elapsed, start, stop));

  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));

  cout << "Took " << elapsed << "ms\n\n";









/*

  int pairArrSize = nk * 2 * rsize;

  double *allrecords = new double[lsize * rsize];
  double *pairs = new double[pairArrSize];

  double maxrecords = getMemoryAllocation(lsize);

  int allsize = getAllRecordArr(allrecords, sd.records, lsize);

  //printall(allrecords, lsize, rsize);

  //determine how much work is to be done on each block/thread

  int totalWorkers = BLOCKS * THREADS;
  int totalOps = rsize; //total operations will be number of records

  float opsPerWorker = totalOps / static_cast<float>(totalWorkers);
  if(opsPerWorker < 1) opsPerWorker = 1;

  int ops = ceil(opsPerWorker);

  //memory that will be shared (between host and device)

  double *a, *p; //a = all data, p = pairs of data
  int doublesize = sizeof(double);

  cudaMallocManaged(&a, (lsize * rsize)*doublesize);
  cudaMallocManaged(&p, pairArrSize*doublesize);

  //populate a, then pairNK will populate p

  cout << "Copying records to GPU memory...\n";
  for(i = 0; i < (lsize * rsize); i++){
    a[i] = allrecords[i];
  }

  cout << "Generating Pairs\nrecords: " << rsize << ",\npairsperrec: " << nk << ",\ntotal pairs: " << pairArrSize;
  cout << ",\nops per worker: " << ops << ",\ntotalworkers: " << totalWorkers << "\n\n";

  float elapsed = 0;
  cudaEvent_t start, stop;

  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));

  HANDLE_ERROR( cudaEventRecord(start, 0));

  pairNK<<<BLOCKS,THREADS>>>(a, p, (lsize * rsize), lsize, nk, pairArrSize, ops);
  cudaDeviceSynchronize();

  cudaFree(a); //we no longer need a (???)

  //first map

  int threadsperpair = ceil(totalWorkers / nk); //ps
  int blackops = ceil(rsize / threadsperpair); //rs

  //initMap<<<3, 2>>>(parr, (rs * ps), ps, blackops, threadsperpair);

  cout << "NKSPACE " << (2 * nk) << ", threadsperpair " << threadsperpair << ", REDOPS " << blackops << endl;
  initMap<<<BLOCKS, THREADS>>>(p, pairArrSize, nk, blackops, threadsperpair);
  cudaDeviceSynchronize();

  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));

  HANDLE_ERROR(cudaEventElapsedTime(&elapsed, start, stop));

  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));

  cout << "ELAPSED TIME: " << elapsed << "ms\n";

  printMemoryStats();

  */



  /*for(i = 0; i < pairArrSize; i += 2){

    cout << i << ") " << p[i] << ',' << p[i + 1] << endl;

  }*/

  //first reduce

/*

  //experimental, for testing mapreducer

  rsize = 7;
  nk = 2;
  pairArrSize = (rsize * nk);

  int i;

  for(i = 0; i < nk * rsize; i++){

    p[i] = i;

  }

*/

  //cout << elapsed << "ms to process\n";


  //cout << "Pairing Complete, enter a number to reduce pairings...\n";

/*
  //do sequentially, to compare results
  //printtofile(a, p, (lsize * rsize), pairArrSize);

  cout << "sequentially\n";
  pairstimulant(allrecords, pairs, allsize, lsize, nk, pairArrSize);

  for(i = 0; i < pairArrSize; i++){

    if(p[i] != pairs[i]) cout << "WARNING: ERR DETECTED IN GPU) " << p[i] << " != " << pairs[i] << "\n";

  }
*/
  return 0;

}

void timingop(){


  /*

    float elapsed = 0;
    cudaEvent_t start, stop;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    HANDLE_ERROR( cudaEventRecord(start, 0));

    //pairNK<<<BLOCKS,THREADS>>>(a, p, (lsize * rsize), lsize, nk, pairArrSize, ops);
    //cudaDeviceSynchronize();

    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));

    HANDLE_ERROR(cudaEventElapsedTime(&elapsed, start, stop));

    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
  */

}

void printMemoryStats(){

  size_t free_byte;
  size_t total_byte;
  HANDLE_ERROR(cudaMemGetInfo(&free_byte, &total_byte));
  double free_db = (double)free_byte;
  double total_db = (double)total_byte;
  double used_db = total_db - free_db;
  printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n", used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);

}

pair<double, double> getMemData(){

  size_t free_byte;
  size_t total_byte;
  HANDLE_ERROR(cudaMemGetInfo(&free_byte, &total_byte));

  double free_db = (double)free_byte;
  double total_db = (double)total_byte;

  pair<double, double> memdata(total_db, free_db);

  return memdata;

}

double getMemoryAllocation(int recordsize){

  int doublesize = sizeof(double);
  int nk = nchoosek(recordsize, 2);

  int sizeperrecord = ((nk * 2 * doublesize)) + (recordsize * doublesize); //also memory must be stored for phase1 op

  cout << "Each record will take approx: " << sizeperrecord << " bytes\n";

  pair<double, double> memdata;
  memdata = getMemData();
  double maxmemory = memdata.first, freememory = memdata.second;

  cout << "Free memory(in MB): " << (freememory/1024/1024) << endl;

  int maxrecords = freememory / sizeperrecord;

  cout << "Theoretical record max: " << maxrecords << endl;
  cout << "Recommended record max: " << (maxrecords * .25) << endl;

  return (maxrecords * .3);

}

int getAllRecordArr(double *all, vector<Record> recs, int lsize){

   int i, j, index = 0;
   for(i = 0; i < recs.size(); i++){

     double *arr = recs[i].convertToArr();

     for(j = 0; j < lsize; j++){

       all[index] = arr[j]; //put all elements from each record into this cluster fuck of an array
       index++;

     }

   }

   return index + 1; //return size

}

int getAllRecordArr_round(double *all, vector<Record> recs, int lsize, int maxrecords, int round){

   int i, j, index = 0;
   int low = round * maxrecords, high = (round * maxrecords) * maxrecords;
   for(i = low; i < high; i++){

     double *arr = recs[i].convertToArr();

     for(j = 0; j < lsize; j++){

       all[index] = arr[j]; //put all elements from each record into this cluster fuck of an array
       index++;

     }

   }

   return index + 1; //return size

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

void printtofile(double *a, double *p, int asize, int psize){

  ofstream recs, pairs;
  pairs.open("pairs.txt");
  recs.open("recs.txt");

  int i;
  for(i = 0; i < asize; i++){
    recs << a[i] << "\n";
  }

  for(i = 0; i < psize; i += 2){
    pairs << "(" << p[i] << ", " << p[i + 1] << ")\n";
  }

}
