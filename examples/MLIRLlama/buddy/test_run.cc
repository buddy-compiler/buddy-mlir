#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
using namespace std;

int main(){
    vector<string> data_shape;
    ifstream infile;
    infile.open("../params_shape.txt", ios::in);
    if (!infile.is_open()){
		cout << "读取文件失败" << endl;
		return 0;
	}
    string element_shape;
    while (getline(infile, element_shape)){
		data_shape.push_back(string(element_shape));
	}
    vector<float*> params;
    for(auto iter : data_shape){
        auto pos = iter.find_first_of(" ");
        string arg_name = iter.substr(0, pos);
        string arg_shape = iter.substr(pos+1, iter.size()-pos-1);
        stringstream ss;
        ss << arg_shape;
        int arg_size;
        ss >> arg_size;
        cout<<"size:"<<arg_size<<endl;
        //float arr[arg_size];
        float* arr = (float*)malloc(sizeof(float)*arg_size);
        ifstream in("../params_data/"+arg_name+".data", ios::in | ios::binary);
        in.read((char*) arr, sizeof(float)*arg_size);
        cout << in.gcount()<<"bytes have been read" << endl;
        cout <<endl;
        params.push_back((float*)arr);
    }
    for(int i=0;i<2048;i++){
        for(int j=0;j<128;j++){
            cout<<params[params.size()-1][i*128+j]<<", ";
        }
        cout<<endl;
    }
    return 0;
}