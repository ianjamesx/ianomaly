#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include <map>
#include <sstream>
#include <vector>
using namespace std;

struct record_data {

  vector<Record> records;
  vector<int> symbolic_indexes;
  vector<vector<string> > symbol_vector;

};

void getLabels(vector<pair<string, string> > &labels);
record_data getRecords(vector<pair<string, string> > labels, string symbol);
pair<string, string> getPair(string name, string type);
vector<vector<string> > getSymbolicVector(vector<set<string> > unique);


//get all labels from a data file
void getLabels(vector<pair<string, string> > &labels){

  ifstream infile;
  infile.open("dataset_names.txt");
  //infile.open("shortnames.txt");

  string name, type; //each individual record
  pair<string, string> labelpair;

  while(infile >> name >> type){

    labelpair = getPair(name, type);
    labels.push_back(labelpair);

  }

}

//get all records from a data file
record_data getRecords(vector<pair<string, string> > labels, string symbol){

  vector<int> symbolicIndexes; //index in fragmented vector of all symbolic values
  vector<set<string> > uniqueProps; //vector of sets holding all unique values for symbolic data
  vector<Record> records; //all records

  int i;
  for(i = 0; i < labels.size(); i++){

    if(labels[i].second == symbol){ //if property is symbolic (needs converting)

      set<string> curr;

      uniqueProps.push_back(curr); //put set of all symbolic values in vector
      symbolicIndexes.push_back(i); //indexes in symbolic indexes

    }

  }

  ifstream infile;
  //infile.open("kddcup.data_10_percent");
  //infile.open("kddcup.data");
  infile.open("kddcup.data_20_percent");
  //infile.open("recordbackup.txt");
  //infile.open("2550recs.txt");
  //infile.open("10percent.txt");
  //infile.open("perfect10.txt");

  string record; //each individual record
  while(infile >> record){

    Record rec(labels, record);

    int i;
    for(i = 0; i < symbolicIndexes.size(); i++){

      int sym = symbolicIndexes[i]; //index of symbolic val in record
      string prop = rec.at(sym); //value for current record corresponding to set

      //look to see if that value is in set, if not, insert it
      if(!(uniqueProps[i].find(prop) != uniqueProps[i].end())){
	      uniqueProps[i].insert(prop);
      }

    }

    records.push_back(rec);

  }

  vector<vector<string> > symbolVect = getSymbolicVector(uniqueProps);

  //convert strings to symbolic numbers
  for(i = 0; i < records.size(); i++){
    records[i].convertSymbols(symbolicIndexes, symbolVect);
  }

  //return data
  record_data sd = {
    records,
    symbolicIndexes,
    symbolVect
  };

  return sd;

}

pair<string, string> getPair(string name, string type){

  name = name.substr(0, name.size()-1); //remove last character from name and type (: for name, . for type)
  type = type.substr(0, type.size()-1);

  pair<string, string> p(name, type);

  return p;

}

vector<vector<string> > getSymbolicVector(vector<set<string> > unique){

  //vector holding all maps for symbolic values
  vector<vector<string> > symbolVect;

  int i;
  for(i = 0; i < unique.size(); i++){

    vector<string> curr;
    symbolVect.push_back(curr);

    set<string>::iterator it = unique[i].begin();

    while (it != unique[i].end()){

      //cout << *it << ", ";

      symbolVect[i].push_back(*it); //put the current key (string, unique value) into the vector corresponding to its property
      //index will be used as its new representative value

      it++;

    }

  //  cout << "\n";

  }

  return symbolVect;

}
