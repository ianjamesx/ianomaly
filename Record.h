#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <sstream>
using namespace std;

class Record {

  map<string, string> properties;
  vector<string> fragments;

public:

  Record();
  Record(vector<pair<string, string> > labels, string record);
  void fragment(string record); //put record into fragments (unlabeled pieces of data)
  string at(int index);
  void convertSymbols(vector<int> sym_indices, vector<vector<string> > sym_vect);
  double *convertToArr();
  int size();
  void print();

};
