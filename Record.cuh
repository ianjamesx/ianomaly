#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include "Record.h"
using namespace std;

Record::Record(){

}

Record::Record(vector<pair<string, string> > labels, string record){

   fragment(record);

   int i;
   for(i = 0; i < labels.size(); i++){

     string key = labels[i].first;
     properties[key] = fragments[i];

   }

}

void Record::fragment(string record){

   stringstream ss(record);
   string element; //actual element to push into fragments

   while(getline(ss, element, ',')){

     fragments.push_back(element);

   }

}

void Record::print(){

  int i;
  cout << "\n";

  for(i = 0; i < fragments.size(); i++){

    cout << fragments[i] << ",";

  }

  cout << "\n";

}

string Record::at(int index){

  return fragments[index];

}

void Record::convertSymbols(vector<int> sym_indices, vector<vector<string> > sym_vect){

  int i, j;
  for(i = 0; i < sym_indices.size(); i++){ //loop through all indexes containing symbolic values

    int index = sym_indices[i]; //(thats the index)

    for(j = 0; j < sym_vect[i].size(); j++){ //loop through corresponding vector containing all potential values

      if(fragments[index] == sym_vect[i][j]){ //if we get a match

        //convert j (a number) to a string, that is now the symbolic value

      	ostringstream oss;
      	oss << j;
        fragments[index] = oss.str();

      }

    }

  }

}

double* Record:: convertToArr(){

  double *arr = new double[fragments.size()];

  int i;
  for(i = 0; i < fragments.size() - 1; i++){ //exclude attack type

    double temp = 0.0;
    stringstream ss;
    ss << fragments[i];
    ss >> temp;

    arr[i] = temp;

  }

  return arr;

}

int Record:: size(){

  return fragments.size();

}
