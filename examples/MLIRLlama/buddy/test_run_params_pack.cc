#include "TextContainer.h"
#include <buddy/Core/Container.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <omp.h>
using namespace std;
using namespace chrono;
using namespace buddy;

extern "C" void _mlir_ciface_forward(MemRef<float, 3> *, MemRef<float, 1> *, MemRef<long long, 2> *);

int main() {
  // The map of string-to-id used in the test cases:
  // buddy: 8937, compiler: 21624, is: 2003, a: 1037, domain: 5884
  // specific: 3563, ":": 1024, "!": 999
  //
  // The test running directory is in <build dir>/tests/Interface/core, so the
  // vocabulary directory uses the following relative path.
  std::string vocabDir =
      "/buddy-mlir-for-transformer/examples/MLIRLlama/vocab.txt";
  //===--------------------------------------------------------------------===//
  // Test text constructor for pure string.
  //===--------------------------------------------------------------------===//
  std::string pureStr = "Hey, are you conscious? Can you talk to me?";
  Text<long long, 2> pureStrContainer(pureStr);
  pureStrContainer.tokenize(vocabDir, 80);
  // CHECK: 1
  fprintf(stderr, "%lld\n", pureStrContainer.getData()[0]);
  // CHECK: 18637
  fprintf(stderr, "%lld\n", pureStrContainer.getData()[1]);
  // CHECK: 1919
  fprintf(stderr, "%lld\n", pureStrContainer.getData()[2]);
  // CHECK: 526
  fprintf(stderr, "%lld\n", pureStrContainer.getData()[3]);
  // CHECK: 366
  fprintf(stderr, "%lld\n", pureStrContainer.getData()[4]);
  // CHECK: 19861
  fprintf(stderr, "%lld\n", pureStrContainer.getData()[5]);
  // CHECK: 1577
  fprintf(stderr, "%lld\n", pureStrContainer.getData()[6]);
  // CHECK: 1815
  fprintf(stderr, "%lld\n", pureStrContainer.getData()[7]);
  // CHECK: 366
  fprintf(stderr, "%lld\n", pureStrContainer.getData()[8]);
  // CHECK: 5193
  fprintf(stderr, "%lld\n", pureStrContainer.getData()[9]);
  // CHECK: 304
  fprintf(stderr, "%lld\n", pureStrContainer.getData()[10]);
  // CHECK: 592
  fprintf(stderr, "%lld\n", pureStrContainer.getData()[11]);
  // CHECK: 592
  fprintf(stderr, "%lld\n", pureStrContainer.getData()[12]);
  // CHECK: 2
  fprintf(stderr, "%lld\n", pureStrContainer.getData()[13]);
  auto start = system_clock::now();
  MemRef<float, 1> arg0({intptr_t(6755192832)});
  ifstream in0(
      "/buddy-mlir-for-transformer/examples/MLIRLlama/params_data/arg0.data",
      ios::in | ios::binary);
  in0.read((char *)(arg0.getData()), sizeof(float) * (arg0.getSize()));
  in0.close();
  auto end = system_clock::now();
  auto duration = duration_cast<milliseconds>(end - start);
  cout << "read params finish" << endl;
  cout << "read params use time: " << duration.count() << "ms" << endl;
  //   MemRef<long long, 2> arg355({1, 13});
  //   long long example[13] = {1,    18637, 29892, 526, 366, 19861, 29973,
  //                            1815, 366,   5193,  304, 592, 29973};
  //   for (int i = 0; i < 13; i++) {
  //     arg355.getData()[i] = example[i];
  //   }
  MemRef<float, 3> result({1, 80, 4096});
  start = system_clock::now();
  int generate_len = 80 - pureStrContainer.getTokenCnt();
  cout << "---------------------------start generate---------------------------"
       << endl;
  for (int i = 0; i < generate_len; i++) {
    _mlir_ciface_forward(&result, &arg0, &pureStrContainer);
    for(int i=0;i<80;i++){
      for(int j=0;j<4096;j++){
        cout<<result.getData()[i*4096+j]<<" ";
      }
      cout<<endl;
    }
    return 0;
    int token_index = pureStrContainer.getTokenCnt() - 1;
    int index = 0;
    int max_elem = result.getData()[token_index * 32000];
    for (int j = index + 1; j < 32000; j++) {
      if (result.getData()[token_index * 32000 + j] > max_elem) {
        max_elem = result.getData()[token_index * 32000 + j];
        index = j;
      }
    }
    pureStrContainer.getData()[pureStrContainer.getTokenCnt()] = index;
    cout << "===================" << endl;
    cout << index << endl;
    if (index == 2) {
      break;
    }
    end = system_clock::now();
    duration = duration_cast<milliseconds>(end - start);
    cout << "llama iteration use time: " << duration.count() << "ms" << endl;
    pureStrContainer.setTokenCnt(pureStrContainer.getTokenCnt() + 1);
  }
  end = system_clock::now();
  duration = duration_cast<milliseconds>(end - start);
  cout << "llama exection use time: " << duration.count() << "ms" << endl;
  for (int i = 0; i < 80; i++) {
    cout << pureStrContainer.getData()[i] << " ";
  }
  cout << endl;
  return 0;
}