#include "TextContainer.h"
#include <buddy/Core/Container.h>
#include <chrono>
#include <fstream>
#include <iostream>
using namespace std;
using namespace chrono;
using namespace buddy;

extern "C" void
_mlir_ciface_forward(MemRef<float, 3> *, MemRef<float, 1> *, MemRef<float, 1> *,
                     MemRef<float, 1> *, MemRef<float, 1> *, MemRef<float, 1> *,
                     MemRef<float, 1> *, MemRef<float, 1> *, MemRef<float, 1> *,
                     MemRef<float, 1> *, MemRef<float, 1> *, MemRef<float, 1> *,
                     MemRef<float, 1> *, MemRef<float, 1> *, MemRef<float, 1> *,
                     MemRef<float, 1> *, MemRef<float, 1> *, MemRef<float, 1> *,
                     MemRef<float, 1> *, MemRef<float, 1> *, MemRef<float, 1> *,
                     MemRef<float, 1> *, MemRef<float, 1> *, MemRef<float, 1> *,
                     MemRef<float, 1> *, MemRef<float, 1> *, MemRef<float, 1> *,
                     MemRef<float, 1> *, MemRef<float, 1> *, MemRef<float, 1> *,
                     MemRef<float, 1> *, MemRef<float, 1> *, MemRef<float, 1> *,
                     MemRef<float, 1> *, MemRef<float, 1> *, MemRef<float, 1> *,
                     MemRef<float, 1> *, MemRef<float, 1> *, MemRef<float, 1> *,
                     MemRef<float, 1> *, MemRef<float, 1> *, MemRef<float, 1> *,
                     MemRef<float, 1> *, MemRef<float, 1> *, MemRef<float, 1> *,
                     MemRef<float, 1> *, MemRef<float, 1> *, MemRef<float, 1> *,
                     MemRef<float, 1> *, MemRef<float, 1> *, MemRef<float, 1> *,
                     MemRef<float, 1> *, MemRef<float, 1> *, MemRef<float, 1> *,
                     MemRef<float, 1> *, MemRef<float, 1> *, MemRef<float, 1> *,
                     MemRef<float, 1> *, MemRef<float, 1> *, MemRef<float, 1> *,
                     MemRef<float, 1> *, MemRef<float, 1> *, MemRef<float, 1> *,
                     MemRef<float, 1> *, MemRef<float, 1> *, MemRef<float, 1> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 2> *, MemRef<float, 2> *,
                     MemRef<float, 2> *, MemRef<float, 4> *, MemRef<float, 4> *,
                     MemRef<float, 4> *, MemRef<float, 4> *, MemRef<float, 4> *,
                     MemRef<float, 4> *, MemRef<float, 4> *, MemRef<float, 4> *,
                     MemRef<float, 4> *, MemRef<float, 4> *, MemRef<float, 4> *,
                     MemRef<float, 4> *, MemRef<float, 4> *, MemRef<float, 4> *,
                     MemRef<float, 4> *, MemRef<float, 4> *, MemRef<float, 4> *,
                     MemRef<float, 4> *, MemRef<float, 4> *, MemRef<float, 4> *,
                     MemRef<float, 4> *, MemRef<float, 4> *, MemRef<float, 4> *,
                     MemRef<float, 4> *, MemRef<float, 4> *, MemRef<float, 4> *,
                     MemRef<float, 4> *, MemRef<float, 4> *, MemRef<float, 4> *,
                     MemRef<float, 4> *, MemRef<float, 4> *, MemRef<float, 4> *,
                     MemRef<float, 4> *, MemRef<float, 4> *, MemRef<float, 4> *,
                     MemRef<float, 4> *, MemRef<float, 4> *, MemRef<float, 4> *,
                     MemRef<float, 4> *, MemRef<float, 4> *, MemRef<float, 4> *,
                     MemRef<float, 4> *, MemRef<float, 4> *, MemRef<float, 4> *,
                     MemRef<float, 4> *, MemRef<float, 4> *, MemRef<float, 4> *,
                     MemRef<float, 4> *, MemRef<float, 4> *, MemRef<float, 4> *,
                     MemRef<float, 4> *, MemRef<float, 4> *, MemRef<float, 4> *,
                     MemRef<float, 4> *, MemRef<float, 4> *, MemRef<float, 4> *,
                     MemRef<float, 4> *, MemRef<float, 4> *, MemRef<float, 4> *,
                     MemRef<float, 4> *, MemRef<float, 4> *, MemRef<float, 4> *,
                     MemRef<float, 4> *, MemRef<float, 4> *,
                     MemRef<long long, 2> *);

int main() {
  // The map of string-to-id used in the test cases:
  // buddy: 8937, compiler: 21624, is: 2003, a: 1037, domain: 5884
  // specific: 3563, ":": 1024, "!": 999
  //
  // The test running directory is in <build dir>/tests/Interface/core, so the
  // vocabulary directory uses the following relative path.
  std::string vocabDir;
  std::cout<<"please input vocab file path"<<std::endl;
  getline(std::cin, vocabDir);
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

  std::string params_path;
  getline(std::cin, params_path);
  auto start = system_clock::now();
  MemRef<float, 1> arg0({4096});
  MemRef<float, 1> arg1({4096});
  MemRef<float, 1> arg2({4096});
  MemRef<float, 1> arg3({4096});
  MemRef<float, 1> arg4({4096});
  MemRef<float, 1> arg5({4096});
  MemRef<float, 1> arg6({4096});
  MemRef<float, 1> arg7({4096});
  MemRef<float, 1> arg8({4096});
  MemRef<float, 1> arg9({4096});
  MemRef<float, 1> arg10({4096});
  MemRef<float, 1> arg11({4096});
  MemRef<float, 1> arg12({4096});
  MemRef<float, 1> arg13({4096});
  MemRef<float, 1> arg14({4096});
  MemRef<float, 1> arg15({4096});
  MemRef<float, 1> arg16({4096});
  MemRef<float, 1> arg17({4096});
  MemRef<float, 1> arg18({4096});
  MemRef<float, 1> arg19({4096});
  MemRef<float, 1> arg20({4096});
  MemRef<float, 1> arg21({4096});
  MemRef<float, 1> arg22({4096});
  MemRef<float, 1> arg23({4096});
  MemRef<float, 1> arg24({4096});
  MemRef<float, 1> arg25({4096});
  MemRef<float, 1> arg26({4096});
  MemRef<float, 1> arg27({4096});
  MemRef<float, 1> arg28({4096});
  MemRef<float, 1> arg29({4096});
  MemRef<float, 1> arg30({4096});
  MemRef<float, 1> arg31({4096});
  MemRef<float, 1> arg32({4096});
  MemRef<float, 1> arg33({4096});
  MemRef<float, 1> arg34({4096});
  MemRef<float, 1> arg35({4096});
  MemRef<float, 1> arg36({4096});
  MemRef<float, 1> arg37({4096});
  MemRef<float, 1> arg38({4096});
  MemRef<float, 1> arg39({4096});
  MemRef<float, 1> arg40({4096});
  MemRef<float, 1> arg41({4096});
  MemRef<float, 1> arg42({4096});
  MemRef<float, 1> arg43({4096});
  MemRef<float, 1> arg44({4096});
  MemRef<float, 1> arg45({4096});
  MemRef<float, 1> arg46({4096});
  MemRef<float, 1> arg47({4096});
  MemRef<float, 1> arg48({4096});
  MemRef<float, 1> arg49({4096});
  MemRef<float, 1> arg50({4096});
  MemRef<float, 1> arg51({4096});
  MemRef<float, 1> arg52({4096});
  MemRef<float, 1> arg53({4096});
  MemRef<float, 1> arg54({4096});
  MemRef<float, 1> arg55({4096});
  MemRef<float, 1> arg56({4096});
  MemRef<float, 1> arg57({4096});
  MemRef<float, 1> arg58({4096});
  MemRef<float, 1> arg59({4096});
  MemRef<float, 1> arg60({4096});
  MemRef<float, 1> arg61({4096});
  MemRef<float, 1> arg62({4096});
  MemRef<float, 1> arg63({4096});
  MemRef<float, 1> arg64({4096});
  MemRef<float, 2> arg65({32000, 4096});
  MemRef<float, 2> arg66({4096, 4096});
  MemRef<float, 2> arg67({4096, 4096});
  MemRef<float, 2> arg68({4096, 4096});
  MemRef<float, 2> arg69({4096, 4096});
  MemRef<float, 2> arg70({11008, 4096});
  MemRef<float, 2> arg71({11008, 4096});
  MemRef<float, 2> arg72({4096, 11008});
  MemRef<float, 2> arg73({4096, 4096});
  MemRef<float, 2> arg74({4096, 4096});
  MemRef<float, 2> arg75({4096, 4096});
  MemRef<float, 2> arg76({4096, 4096});
  MemRef<float, 2> arg77({11008, 4096});
  MemRef<float, 2> arg78({11008, 4096});
  MemRef<float, 2> arg79({4096, 11008});
  MemRef<float, 2> arg80({4096, 4096});
  MemRef<float, 2> arg81({4096, 4096});
  MemRef<float, 2> arg82({4096, 4096});
  MemRef<float, 2> arg83({4096, 4096});
  MemRef<float, 2> arg84({11008, 4096});
  MemRef<float, 2> arg85({11008, 4096});
  MemRef<float, 2> arg86({4096, 11008});
  MemRef<float, 2> arg87({4096, 4096});
  MemRef<float, 2> arg88({4096, 4096});
  MemRef<float, 2> arg89({4096, 4096});
  MemRef<float, 2> arg90({4096, 4096});
  MemRef<float, 2> arg91({11008, 4096});
  MemRef<float, 2> arg92({11008, 4096});
  MemRef<float, 2> arg93({4096, 11008});
  MemRef<float, 2> arg94({4096, 4096});
  MemRef<float, 2> arg95({4096, 4096});
  MemRef<float, 2> arg96({4096, 4096});
  MemRef<float, 2> arg97({4096, 4096});
  MemRef<float, 2> arg98({11008, 4096});
  MemRef<float, 2> arg99({11008, 4096});
  MemRef<float, 2> arg100({4096, 11008});
  MemRef<float, 2> arg101({4096, 4096});
  MemRef<float, 2> arg102({4096, 4096});
  MemRef<float, 2> arg103({4096, 4096});
  MemRef<float, 2> arg104({4096, 4096});
  MemRef<float, 2> arg105({11008, 4096});
  MemRef<float, 2> arg106({11008, 4096});
  MemRef<float, 2> arg107({4096, 11008});
  MemRef<float, 2> arg108({4096, 4096});
  MemRef<float, 2> arg109({4096, 4096});
  MemRef<float, 2> arg110({4096, 4096});
  MemRef<float, 2> arg111({4096, 4096});
  MemRef<float, 2> arg112({11008, 4096});
  MemRef<float, 2> arg113({11008, 4096});
  MemRef<float, 2> arg114({4096, 11008});
  MemRef<float, 2> arg115({4096, 4096});
  MemRef<float, 2> arg116({4096, 4096});
  MemRef<float, 2> arg117({4096, 4096});
  MemRef<float, 2> arg118({4096, 4096});
  MemRef<float, 2> arg119({11008, 4096});
  MemRef<float, 2> arg120({11008, 4096});
  MemRef<float, 2> arg121({4096, 11008});
  MemRef<float, 2> arg122({4096, 4096});
  MemRef<float, 2> arg123({4096, 4096});
  MemRef<float, 2> arg124({4096, 4096});
  MemRef<float, 2> arg125({4096, 4096});
  MemRef<float, 2> arg126({11008, 4096});
  MemRef<float, 2> arg127({11008, 4096});
  MemRef<float, 2> arg128({4096, 11008});
  MemRef<float, 2> arg129({4096, 4096});
  MemRef<float, 2> arg130({4096, 4096});
  MemRef<float, 2> arg131({4096, 4096});
  MemRef<float, 2> arg132({4096, 4096});
  MemRef<float, 2> arg133({11008, 4096});
  MemRef<float, 2> arg134({11008, 4096});
  MemRef<float, 2> arg135({4096, 11008});
  MemRef<float, 2> arg136({4096, 4096});
  MemRef<float, 2> arg137({4096, 4096});
  MemRef<float, 2> arg138({4096, 4096});
  MemRef<float, 2> arg139({4096, 4096});
  MemRef<float, 2> arg140({11008, 4096});
  MemRef<float, 2> arg141({11008, 4096});
  MemRef<float, 2> arg142({4096, 11008});
  MemRef<float, 2> arg143({4096, 4096});
  MemRef<float, 2> arg144({4096, 4096});
  MemRef<float, 2> arg145({4096, 4096});
  MemRef<float, 2> arg146({4096, 4096});
  MemRef<float, 2> arg147({11008, 4096});
  MemRef<float, 2> arg148({11008, 4096});
  MemRef<float, 2> arg149({4096, 11008});
  MemRef<float, 2> arg150({4096, 4096});
  MemRef<float, 2> arg151({4096, 4096});
  MemRef<float, 2> arg152({4096, 4096});
  MemRef<float, 2> arg153({4096, 4096});
  MemRef<float, 2> arg154({11008, 4096});
  MemRef<float, 2> arg155({11008, 4096});
  MemRef<float, 2> arg156({4096, 11008});
  MemRef<float, 2> arg157({4096, 4096});
  MemRef<float, 2> arg158({4096, 4096});
  MemRef<float, 2> arg159({4096, 4096});
  MemRef<float, 2> arg160({4096, 4096});
  MemRef<float, 2> arg161({11008, 4096});
  MemRef<float, 2> arg162({11008, 4096});
  MemRef<float, 2> arg163({4096, 11008});
  MemRef<float, 2> arg164({4096, 4096});
  MemRef<float, 2> arg165({4096, 4096});
  MemRef<float, 2> arg166({4096, 4096});
  MemRef<float, 2> arg167({4096, 4096});
  MemRef<float, 2> arg168({11008, 4096});
  MemRef<float, 2> arg169({11008, 4096});
  MemRef<float, 2> arg170({4096, 11008});
  MemRef<float, 2> arg171({4096, 4096});
  MemRef<float, 2> arg172({4096, 4096});
  MemRef<float, 2> arg173({4096, 4096});
  MemRef<float, 2> arg174({4096, 4096});
  MemRef<float, 2> arg175({11008, 4096});
  MemRef<float, 2> arg176({11008, 4096});
  MemRef<float, 2> arg177({4096, 11008});
  MemRef<float, 2> arg178({4096, 4096});
  MemRef<float, 2> arg179({4096, 4096});
  MemRef<float, 2> arg180({4096, 4096});
  MemRef<float, 2> arg181({4096, 4096});
  MemRef<float, 2> arg182({11008, 4096});
  MemRef<float, 2> arg183({11008, 4096});
  MemRef<float, 2> arg184({4096, 11008});
  MemRef<float, 2> arg185({4096, 4096});
  MemRef<float, 2> arg186({4096, 4096});
  MemRef<float, 2> arg187({4096, 4096});
  MemRef<float, 2> arg188({4096, 4096});
  MemRef<float, 2> arg189({11008, 4096});
  MemRef<float, 2> arg190({11008, 4096});
  MemRef<float, 2> arg191({4096, 11008});
  MemRef<float, 2> arg192({4096, 4096});
  MemRef<float, 2> arg193({4096, 4096});
  MemRef<float, 2> arg194({4096, 4096});
  MemRef<float, 2> arg195({4096, 4096});
  MemRef<float, 2> arg196({11008, 4096});
  MemRef<float, 2> arg197({11008, 4096});
  MemRef<float, 2> arg198({4096, 11008});
  MemRef<float, 2> arg199({4096, 4096});
  MemRef<float, 2> arg200({4096, 4096});
  MemRef<float, 2> arg201({4096, 4096});
  MemRef<float, 2> arg202({4096, 4096});
  MemRef<float, 2> arg203({11008, 4096});
  MemRef<float, 2> arg204({11008, 4096});
  MemRef<float, 2> arg205({4096, 11008});
  MemRef<float, 2> arg206({4096, 4096});
  MemRef<float, 2> arg207({4096, 4096});
  MemRef<float, 2> arg208({4096, 4096});
  MemRef<float, 2> arg209({4096, 4096});
  MemRef<float, 2> arg210({11008, 4096});
  MemRef<float, 2> arg211({11008, 4096});
  MemRef<float, 2> arg212({4096, 11008});
  MemRef<float, 2> arg213({4096, 4096});
  MemRef<float, 2> arg214({4096, 4096});
  MemRef<float, 2> arg215({4096, 4096});
  MemRef<float, 2> arg216({4096, 4096});
  MemRef<float, 2> arg217({11008, 4096});
  MemRef<float, 2> arg218({11008, 4096});
  MemRef<float, 2> arg219({4096, 11008});
  MemRef<float, 2> arg220({4096, 4096});
  MemRef<float, 2> arg221({4096, 4096});
  MemRef<float, 2> arg222({4096, 4096});
  MemRef<float, 2> arg223({4096, 4096});
  MemRef<float, 2> arg224({11008, 4096});
  MemRef<float, 2> arg225({11008, 4096});
  MemRef<float, 2> arg226({4096, 11008});
  MemRef<float, 2> arg227({4096, 4096});
  MemRef<float, 2> arg228({4096, 4096});
  MemRef<float, 2> arg229({4096, 4096});
  MemRef<float, 2> arg230({4096, 4096});
  MemRef<float, 2> arg231({11008, 4096});
  MemRef<float, 2> arg232({11008, 4096});
  MemRef<float, 2> arg233({4096, 11008});
  MemRef<float, 2> arg234({4096, 4096});
  MemRef<float, 2> arg235({4096, 4096});
  MemRef<float, 2> arg236({4096, 4096});
  MemRef<float, 2> arg237({4096, 4096});
  MemRef<float, 2> arg238({11008, 4096});
  MemRef<float, 2> arg239({11008, 4096});
  MemRef<float, 2> arg240({4096, 11008});
  MemRef<float, 2> arg241({4096, 4096});
  MemRef<float, 2> arg242({4096, 4096});
  MemRef<float, 2> arg243({4096, 4096});
  MemRef<float, 2> arg244({4096, 4096});
  MemRef<float, 2> arg245({11008, 4096});
  MemRef<float, 2> arg246({11008, 4096});
  MemRef<float, 2> arg247({4096, 11008});
  MemRef<float, 2> arg248({4096, 4096});
  MemRef<float, 2> arg249({4096, 4096});
  MemRef<float, 2> arg250({4096, 4096});
  MemRef<float, 2> arg251({4096, 4096});
  MemRef<float, 2> arg252({11008, 4096});
  MemRef<float, 2> arg253({11008, 4096});
  MemRef<float, 2> arg254({4096, 11008});
  MemRef<float, 2> arg255({4096, 4096});
  MemRef<float, 2> arg256({4096, 4096});
  MemRef<float, 2> arg257({4096, 4096});
  MemRef<float, 2> arg258({4096, 4096});
  MemRef<float, 2> arg259({11008, 4096});
  MemRef<float, 2> arg260({11008, 4096});
  MemRef<float, 2> arg261({4096, 11008});
  MemRef<float, 2> arg262({4096, 4096});
  MemRef<float, 2> arg263({4096, 4096});
  MemRef<float, 2> arg264({4096, 4096});
  MemRef<float, 2> arg265({4096, 4096});
  MemRef<float, 2> arg266({11008, 4096});
  MemRef<float, 2> arg267({11008, 4096});
  MemRef<float, 2> arg268({4096, 11008});
  MemRef<float, 2> arg269({4096, 4096});
  MemRef<float, 2> arg270({4096, 4096});
  MemRef<float, 2> arg271({4096, 4096});
  MemRef<float, 2> arg272({4096, 4096});
  MemRef<float, 2> arg273({11008, 4096});
  MemRef<float, 2> arg274({11008, 4096});
  MemRef<float, 2> arg275({4096, 11008});
  MemRef<float, 2> arg276({4096, 4096});
  MemRef<float, 2> arg277({4096, 4096});
  MemRef<float, 2> arg278({4096, 4096});
  MemRef<float, 2> arg279({4096, 4096});
  MemRef<float, 2> arg280({11008, 4096});
  MemRef<float, 2> arg281({11008, 4096});
  MemRef<float, 2> arg282({4096, 11008});
  MemRef<float, 2> arg283({4096, 4096});
  MemRef<float, 2> arg284({4096, 4096});
  MemRef<float, 2> arg285({4096, 4096});
  MemRef<float, 2> arg286({4096, 4096});
  MemRef<float, 2> arg287({11008, 4096});
  MemRef<float, 2> arg288({11008, 4096});
  MemRef<float, 2> arg289({4096, 11008});
  MemRef<float, 2> arg290({32000, 4096});
  MemRef<float, 4> arg291({1, 1, 2048, 128});
  MemRef<float, 4> arg292({1, 1, 2048, 128});
  MemRef<float, 4> arg293({1, 1, 2048, 128});
  MemRef<float, 4> arg294({1, 1, 2048, 128});
  MemRef<float, 4> arg295({1, 1, 2048, 128});
  MemRef<float, 4> arg296({1, 1, 2048, 128});
  MemRef<float, 4> arg297({1, 1, 2048, 128});
  MemRef<float, 4> arg298({1, 1, 2048, 128});
  MemRef<float, 4> arg299({1, 1, 2048, 128});
  MemRef<float, 4> arg300({1, 1, 2048, 128});
  MemRef<float, 4> arg301({1, 1, 2048, 128});
  MemRef<float, 4> arg302({1, 1, 2048, 128});
  MemRef<float, 4> arg303({1, 1, 2048, 128});
  MemRef<float, 4> arg304({1, 1, 2048, 128});
  MemRef<float, 4> arg305({1, 1, 2048, 128});
  MemRef<float, 4> arg306({1, 1, 2048, 128});
  MemRef<float, 4> arg307({1, 1, 2048, 128});
  MemRef<float, 4> arg308({1, 1, 2048, 128});
  MemRef<float, 4> arg309({1, 1, 2048, 128});
  MemRef<float, 4> arg310({1, 1, 2048, 128});
  MemRef<float, 4> arg311({1, 1, 2048, 128});
  MemRef<float, 4> arg312({1, 1, 2048, 128});
  MemRef<float, 4> arg313({1, 1, 2048, 128});
  MemRef<float, 4> arg314({1, 1, 2048, 128});
  MemRef<float, 4> arg315({1, 1, 2048, 128});
  MemRef<float, 4> arg316({1, 1, 2048, 128});
  MemRef<float, 4> arg317({1, 1, 2048, 128});
  MemRef<float, 4> arg318({1, 1, 2048, 128});
  MemRef<float, 4> arg319({1, 1, 2048, 128});
  MemRef<float, 4> arg320({1, 1, 2048, 128});
  MemRef<float, 4> arg321({1, 1, 2048, 128});
  MemRef<float, 4> arg322({1, 1, 2048, 128});
  MemRef<float, 4> arg323({1, 1, 2048, 128});
  MemRef<float, 4> arg324({1, 1, 2048, 128});
  MemRef<float, 4> arg325({1, 1, 2048, 128});
  MemRef<float, 4> arg326({1, 1, 2048, 128});
  MemRef<float, 4> arg327({1, 1, 2048, 128});
  MemRef<float, 4> arg328({1, 1, 2048, 128});
  MemRef<float, 4> arg329({1, 1, 2048, 128});
  MemRef<float, 4> arg330({1, 1, 2048, 128});
  MemRef<float, 4> arg331({1, 1, 2048, 128});
  MemRef<float, 4> arg332({1, 1, 2048, 128});
  MemRef<float, 4> arg333({1, 1, 2048, 128});
  MemRef<float, 4> arg334({1, 1, 2048, 128});
  MemRef<float, 4> arg335({1, 1, 2048, 128});
  MemRef<float, 4> arg336({1, 1, 2048, 128});
  MemRef<float, 4> arg337({1, 1, 2048, 128});
  MemRef<float, 4> arg338({1, 1, 2048, 128});
  MemRef<float, 4> arg339({1, 1, 2048, 128});
  MemRef<float, 4> arg340({1, 1, 2048, 128});
  MemRef<float, 4> arg341({1, 1, 2048, 128});
  MemRef<float, 4> arg342({1, 1, 2048, 128});
  MemRef<float, 4> arg343({1, 1, 2048, 128});
  MemRef<float, 4> arg344({1, 1, 2048, 128});
  MemRef<float, 4> arg345({1, 1, 2048, 128});
  MemRef<float, 4> arg346({1, 1, 2048, 128});
  MemRef<float, 4> arg347({1, 1, 2048, 128});
  MemRef<float, 4> arg348({1, 1, 2048, 128});
  MemRef<float, 4> arg349({1, 1, 2048, 128});
  MemRef<float, 4> arg350({1, 1, 2048, 128});
  MemRef<float, 4> arg351({1, 1, 2048, 128});
  MemRef<float, 4> arg352({1, 1, 2048, 128});
  MemRef<float, 4> arg353({1, 1, 2048, 128});
  MemRef<float, 4> arg354({1, 1, 2048, 128});
  ifstream in0(params_path+"/arg0.data",
               ios::in | ios::binary);
  in0.read((char *)(arg0.getData()), sizeof(float) * (arg0.getSize()));
  in0.close();
  ifstream in1(params_path+"/arg1.data",
               ios::in | ios::binary);
  in1.read((char *)(arg1.getData()), sizeof(float) * (arg1.getSize()));
  in1.close();
  ifstream in2(params_path+"/arg2.data",
               ios::in | ios::binary);
  in2.read((char *)(arg2.getData()), sizeof(float) * (arg2.getSize()));
  in2.close();
  ifstream in3(params_path+"/arg3.data",
               ios::in | ios::binary);
  in3.read((char *)(arg3.getData()), sizeof(float) * (arg3.getSize()));
  in3.close();
  ifstream in4(params_path+"/arg4.data",
               ios::in | ios::binary);
  in4.read((char *)(arg4.getData()), sizeof(float) * (arg4.getSize()));
  in4.close();
  ifstream in5(params_path+"/arg5.data",
               ios::in | ios::binary);
  in5.read((char *)(arg5.getData()), sizeof(float) * (arg5.getSize()));
  in5.close();
  ifstream in6(params_path+"/arg6.data",
               ios::in | ios::binary);
  in6.read((char *)(arg6.getData()), sizeof(float) * (arg6.getSize()));
  in6.close();
  ifstream in7(params_path+"/arg7.data",
               ios::in | ios::binary);
  in7.read((char *)(arg7.getData()), sizeof(float) * (arg7.getSize()));
  in7.close();
  ifstream in8(params_path+"/arg8.data",
               ios::in | ios::binary);
  in8.read((char *)(arg8.getData()), sizeof(float) * (arg8.getSize()));
  in8.close();
  ifstream in9(params_path+"/arg9.data",
               ios::in | ios::binary);
  in9.read((char *)(arg9.getData()), sizeof(float) * (arg9.getSize()));
  in9.close();
  ifstream in10(
      params_path+"/arg10.data",
      ios::in | ios::binary);
  in10.read((char *)(arg10.getData()), sizeof(float) * (arg10.getSize()));
  in10.close();
  ifstream in11(
      params_path+"/arg11.data",
      ios::in | ios::binary);
  in11.read((char *)(arg11.getData()), sizeof(float) * (arg11.getSize()));
  in11.close();
  ifstream in12(
      params_path+"/arg12.data",
      ios::in | ios::binary);
  in12.read((char *)(arg12.getData()), sizeof(float) * (arg12.getSize()));
  in12.close();
  ifstream in13(
      params_path+"/arg13.data",
      ios::in | ios::binary);
  in13.read((char *)(arg13.getData()), sizeof(float) * (arg13.getSize()));
  in13.close();
  ifstream in14(
      params_path+"/arg14.data",
      ios::in | ios::binary);
  in14.read((char *)(arg14.getData()), sizeof(float) * (arg14.getSize()));
  in14.close();
  ifstream in15(
      params_path+"/arg15.data",
      ios::in | ios::binary);
  in15.read((char *)(arg15.getData()), sizeof(float) * (arg15.getSize()));
  in15.close();
  ifstream in16(
      params_path+"/arg16.data",
      ios::in | ios::binary);
  in16.read((char *)(arg16.getData()), sizeof(float) * (arg16.getSize()));
  in16.close();
  ifstream in17(
      params_path+"/arg17.data",
      ios::in | ios::binary);
  in17.read((char *)(arg17.getData()), sizeof(float) * (arg17.getSize()));
  in17.close();
  ifstream in18(
      params_path+"/arg18.data",
      ios::in | ios::binary);
  in18.read((char *)(arg18.getData()), sizeof(float) * (arg18.getSize()));
  in18.close();
  ifstream in19(
      params_path+"/arg19.data",
      ios::in | ios::binary);
  in19.read((char *)(arg19.getData()), sizeof(float) * (arg19.getSize()));
  in19.close();
  ifstream in20(
      params_path+"/arg20.data",
      ios::in | ios::binary);
  in20.read((char *)(arg20.getData()), sizeof(float) * (arg20.getSize()));
  in20.close();
  ifstream in21(
      params_path+"/arg21.data",
      ios::in | ios::binary);
  in21.read((char *)(arg21.getData()), sizeof(float) * (arg21.getSize()));
  in21.close();
  ifstream in22(
      params_path+"/arg22.data",
      ios::in | ios::binary);
  in22.read((char *)(arg22.getData()), sizeof(float) * (arg22.getSize()));
  in22.close();
  ifstream in23(
      params_path+"/arg23.data",
      ios::in | ios::binary);
  in23.read((char *)(arg23.getData()), sizeof(float) * (arg23.getSize()));
  in23.close();
  ifstream in24(
      params_path+"/arg24.data",
      ios::in | ios::binary);
  in24.read((char *)(arg24.getData()), sizeof(float) * (arg24.getSize()));
  in24.close();
  ifstream in25(
      params_path+"/arg25.data",
      ios::in | ios::binary);
  in25.read((char *)(arg25.getData()), sizeof(float) * (arg25.getSize()));
  in25.close();
  ifstream in26(
      params_path+"/arg26.data",
      ios::in | ios::binary);
  in26.read((char *)(arg26.getData()), sizeof(float) * (arg26.getSize()));
  in26.close();
  ifstream in27(
      params_path+"/arg27.data",
      ios::in | ios::binary);
  in27.read((char *)(arg27.getData()), sizeof(float) * (arg27.getSize()));
  in27.close();
  ifstream in28(
      params_path+"/arg28.data",
      ios::in | ios::binary);
  in28.read((char *)(arg28.getData()), sizeof(float) * (arg28.getSize()));
  in28.close();
  ifstream in29(
      params_path+"/arg29.data",
      ios::in | ios::binary);
  in29.read((char *)(arg29.getData()), sizeof(float) * (arg29.getSize()));
  in29.close();
  ifstream in30(
      params_path+"/arg30.data",
      ios::in | ios::binary);
  in30.read((char *)(arg30.getData()), sizeof(float) * (arg30.getSize()));
  in30.close();
  ifstream in31(
      params_path+"/arg31.data",
      ios::in | ios::binary);
  in31.read((char *)(arg31.getData()), sizeof(float) * (arg31.getSize()));
  in31.close();
  ifstream in32(
      params_path+"/arg32.data",
      ios::in | ios::binary);
  in32.read((char *)(arg32.getData()), sizeof(float) * (arg32.getSize()));
  in32.close();
  ifstream in33(
      params_path+"/arg33.data",
      ios::in | ios::binary);
  in33.read((char *)(arg33.getData()), sizeof(float) * (arg33.getSize()));
  in33.close();
  ifstream in34(
      params_path+"/arg34.data",
      ios::in | ios::binary);
  in34.read((char *)(arg34.getData()), sizeof(float) * (arg34.getSize()));
  in34.close();
  ifstream in35(
      params_path+"/arg35.data",
      ios::in | ios::binary);
  in35.read((char *)(arg35.getData()), sizeof(float) * (arg35.getSize()));
  in35.close();
  ifstream in36(
      params_path+"/arg36.data",
      ios::in | ios::binary);
  in36.read((char *)(arg36.getData()), sizeof(float) * (arg36.getSize()));
  in36.close();
  ifstream in37(
      params_path+"/arg37.data",
      ios::in | ios::binary);
  in37.read((char *)(arg37.getData()), sizeof(float) * (arg37.getSize()));
  in37.close();
  ifstream in38(
      params_path+"/arg38.data",
      ios::in | ios::binary);
  in38.read((char *)(arg38.getData()), sizeof(float) * (arg38.getSize()));
  in38.close();
  ifstream in39(
      params_path+"/arg39.data",
      ios::in | ios::binary);
  in39.read((char *)(arg39.getData()), sizeof(float) * (arg39.getSize()));
  in39.close();
  ifstream in40(
      params_path+"/arg40.data",
      ios::in | ios::binary);
  in40.read((char *)(arg40.getData()), sizeof(float) * (arg40.getSize()));
  in40.close();
  ifstream in41(
      params_path+"/arg41.data",
      ios::in | ios::binary);
  in41.read((char *)(arg41.getData()), sizeof(float) * (arg41.getSize()));
  in41.close();
  ifstream in42(
      params_path+"/arg42.data",
      ios::in | ios::binary);
  in42.read((char *)(arg42.getData()), sizeof(float) * (arg42.getSize()));
  in42.close();
  ifstream in43(
      params_path+"/arg43.data",
      ios::in | ios::binary);
  in43.read((char *)(arg43.getData()), sizeof(float) * (arg43.getSize()));
  in43.close();
  ifstream in44(
      params_path+"/arg44.data",
      ios::in | ios::binary);
  in44.read((char *)(arg44.getData()), sizeof(float) * (arg44.getSize()));
  in44.close();
  ifstream in45(
      params_path+"/arg45.data",
      ios::in | ios::binary);
  in45.read((char *)(arg45.getData()), sizeof(float) * (arg45.getSize()));
  in45.close();
  ifstream in46(
      params_path+"/arg46.data",
      ios::in | ios::binary);
  in46.read((char *)(arg46.getData()), sizeof(float) * (arg46.getSize()));
  in46.close();
  ifstream in47(
      params_path+"/arg47.data",
      ios::in | ios::binary);
  in47.read((char *)(arg47.getData()), sizeof(float) * (arg47.getSize()));
  in47.close();
  ifstream in48(
      params_path+"/arg48.data",
      ios::in | ios::binary);
  in48.read((char *)(arg48.getData()), sizeof(float) * (arg48.getSize()));
  in48.close();
  ifstream in49(
      params_path+"/arg49.data",
      ios::in | ios::binary);
  in49.read((char *)(arg49.getData()), sizeof(float) * (arg49.getSize()));
  in49.close();
  ifstream in50(
      params_path+"/arg50.data",
      ios::in | ios::binary);
  in50.read((char *)(arg50.getData()), sizeof(float) * (arg50.getSize()));
  in50.close();
  ifstream in51(
      params_path+"/arg51.data",
      ios::in | ios::binary);
  in51.read((char *)(arg51.getData()), sizeof(float) * (arg51.getSize()));
  in51.close();
  ifstream in52(
      params_path+"/arg52.data",
      ios::in | ios::binary);
  in52.read((char *)(arg52.getData()), sizeof(float) * (arg52.getSize()));
  in52.close();
  ifstream in53(
      params_path+"/arg53.data",
      ios::in | ios::binary);
  in53.read((char *)(arg53.getData()), sizeof(float) * (arg53.getSize()));
  in53.close();
  ifstream in54(
      params_path+"/arg54.data",
      ios::in | ios::binary);
  in54.read((char *)(arg54.getData()), sizeof(float) * (arg54.getSize()));
  in54.close();
  ifstream in55(
      params_path+"/arg55.data",
      ios::in | ios::binary);
  in55.read((char *)(arg55.getData()), sizeof(float) * (arg55.getSize()));
  in55.close();
  ifstream in56(
      params_path+"/arg56.data",
      ios::in | ios::binary);
  in56.read((char *)(arg56.getData()), sizeof(float) * (arg56.getSize()));
  in56.close();
  ifstream in57(
      params_path+"/arg57.data",
      ios::in | ios::binary);
  in57.read((char *)(arg57.getData()), sizeof(float) * (arg57.getSize()));
  in57.close();
  ifstream in58(
      params_path+"/arg58.data",
      ios::in | ios::binary);
  in58.read((char *)(arg58.getData()), sizeof(float) * (arg58.getSize()));
  in58.close();
  ifstream in59(
      params_path+"/arg59.data",
      ios::in | ios::binary);
  in59.read((char *)(arg59.getData()), sizeof(float) * (arg59.getSize()));
  in59.close();
  ifstream in60(
      params_path+"/arg60.data",
      ios::in | ios::binary);
  in60.read((char *)(arg60.getData()), sizeof(float) * (arg60.getSize()));
  in60.close();
  ifstream in61(
      params_path+"/arg61.data",
      ios::in | ios::binary);
  in61.read((char *)(arg61.getData()), sizeof(float) * (arg61.getSize()));
  in61.close();
  ifstream in62(
      params_path+"/arg62.data",
      ios::in | ios::binary);
  in62.read((char *)(arg62.getData()), sizeof(float) * (arg62.getSize()));
  in62.close();
  ifstream in63(
      params_path+"/arg63.data",
      ios::in | ios::binary);
  in63.read((char *)(arg63.getData()), sizeof(float) * (arg63.getSize()));
  in63.close();
  ifstream in64(
      params_path+"/arg64.data",
      ios::in | ios::binary);
  in64.read((char *)(arg64.getData()), sizeof(float) * (arg64.getSize()));
  in64.close();
  ifstream in65(
      params_path+"/arg65.data",
      ios::in | ios::binary);
  in65.read((char *)(arg65.getData()), sizeof(float) * (arg65.getSize()));
  in65.close();
  ifstream in66(
      params_path+"/arg66.data",
      ios::in | ios::binary);
  in66.read((char *)(arg66.getData()), sizeof(float) * (arg66.getSize()));
  in66.close();
  ifstream in67(
      params_path+"/arg67.data",
      ios::in | ios::binary);
  in67.read((char *)(arg67.getData()), sizeof(float) * (arg67.getSize()));
  in67.close();
  ifstream in68(
      params_path+"/arg68.data",
      ios::in | ios::binary);
  in68.read((char *)(arg68.getData()), sizeof(float) * (arg68.getSize()));
  in68.close();
  ifstream in69(
      params_path+"/arg69.data",
      ios::in | ios::binary);
  in69.read((char *)(arg69.getData()), sizeof(float) * (arg69.getSize()));
  in69.close();
  ifstream in70(
      params_path+"/arg70.data",
      ios::in | ios::binary);
  in70.read((char *)(arg70.getData()), sizeof(float) * (arg70.getSize()));
  in70.close();
  ifstream in71(
      params_path+"/arg71.data",
      ios::in | ios::binary);
  in71.read((char *)(arg71.getData()), sizeof(float) * (arg71.getSize()));
  in71.close();
  ifstream in72(
      params_path+"/arg72.data",
      ios::in | ios::binary);
  in72.read((char *)(arg72.getData()), sizeof(float) * (arg72.getSize()));
  in72.close();
  ifstream in73(
      params_path+"/arg73.data",
      ios::in | ios::binary);
  in73.read((char *)(arg73.getData()), sizeof(float) * (arg73.getSize()));
  in73.close();
  ifstream in74(
      params_path+"/arg74.data",
      ios::in | ios::binary);
  in74.read((char *)(arg74.getData()), sizeof(float) * (arg74.getSize()));
  in74.close();
  ifstream in75(
      params_path+"/arg75.data",
      ios::in | ios::binary);
  in75.read((char *)(arg75.getData()), sizeof(float) * (arg75.getSize()));
  in75.close();
  ifstream in76(
      params_path+"/arg76.data",
      ios::in | ios::binary);
  in76.read((char *)(arg76.getData()), sizeof(float) * (arg76.getSize()));
  in76.close();
  ifstream in77(
      params_path+"/arg77.data",
      ios::in | ios::binary);
  in77.read((char *)(arg77.getData()), sizeof(float) * (arg77.getSize()));
  in77.close();
  ifstream in78(
      params_path+"/arg78.data",
      ios::in | ios::binary);
  in78.read((char *)(arg78.getData()), sizeof(float) * (arg78.getSize()));
  in78.close();
  ifstream in79(
      params_path+"/arg79.data",
      ios::in | ios::binary);
  in79.read((char *)(arg79.getData()), sizeof(float) * (arg79.getSize()));
  in79.close();
  ifstream in80(
      params_path+"/arg80.data",
      ios::in | ios::binary);
  in80.read((char *)(arg80.getData()), sizeof(float) * (arg80.getSize()));
  in80.close();
  ifstream in81(
      params_path+"/arg81.data",
      ios::in | ios::binary);
  in81.read((char *)(arg81.getData()), sizeof(float) * (arg81.getSize()));
  in81.close();
  ifstream in82(
      params_path+"/arg82.data",
      ios::in | ios::binary);
  in82.read((char *)(arg82.getData()), sizeof(float) * (arg82.getSize()));
  in82.close();
  ifstream in83(
      params_path+"/arg83.data",
      ios::in | ios::binary);
  in83.read((char *)(arg83.getData()), sizeof(float) * (arg83.getSize()));
  in83.close();
  ifstream in84(
      params_path+"/arg84.data",
      ios::in | ios::binary);
  in84.read((char *)(arg84.getData()), sizeof(float) * (arg84.getSize()));
  in84.close();
  ifstream in85(
      params_path+"/arg85.data",
      ios::in | ios::binary);
  in85.read((char *)(arg85.getData()), sizeof(float) * (arg85.getSize()));
  in85.close();
  ifstream in86(
      params_path+"/arg86.data",
      ios::in | ios::binary);
  in86.read((char *)(arg86.getData()), sizeof(float) * (arg86.getSize()));
  in86.close();
  ifstream in87(
      params_path+"/arg87.data",
      ios::in | ios::binary);
  in87.read((char *)(arg87.getData()), sizeof(float) * (arg87.getSize()));
  in87.close();
  ifstream in88(
      params_path+"/arg88.data",
      ios::in | ios::binary);
  in88.read((char *)(arg88.getData()), sizeof(float) * (arg88.getSize()));
  in88.close();
  ifstream in89(
      params_path+"/arg89.data",
      ios::in | ios::binary);
  in89.read((char *)(arg89.getData()), sizeof(float) * (arg89.getSize()));
  in89.close();
  ifstream in90(
      params_path+"/arg90.data",
      ios::in | ios::binary);
  in90.read((char *)(arg90.getData()), sizeof(float) * (arg90.getSize()));
  in90.close();
  ifstream in91(
      params_path+"/arg91.data",
      ios::in | ios::binary);
  in91.read((char *)(arg91.getData()), sizeof(float) * (arg91.getSize()));
  in91.close();
  ifstream in92(
      params_path+"/arg92.data",
      ios::in | ios::binary);
  in92.read((char *)(arg92.getData()), sizeof(float) * (arg92.getSize()));
  in92.close();
  ifstream in93(
      params_path+"/arg93.data",
      ios::in | ios::binary);
  in93.read((char *)(arg93.getData()), sizeof(float) * (arg93.getSize()));
  in93.close();
  ifstream in94(
      params_path+"/arg94.data",
      ios::in | ios::binary);
  in94.read((char *)(arg94.getData()), sizeof(float) * (arg94.getSize()));
  in94.close();
  ifstream in95(
      params_path+"/arg95.data",
      ios::in | ios::binary);
  in95.read((char *)(arg95.getData()), sizeof(float) * (arg95.getSize()));
  in95.close();
  ifstream in96(
      params_path+"/arg96.data",
      ios::in | ios::binary);
  in96.read((char *)(arg96.getData()), sizeof(float) * (arg96.getSize()));
  in96.close();
  ifstream in97(
      params_path+"/arg97.data",
      ios::in | ios::binary);
  in97.read((char *)(arg97.getData()), sizeof(float) * (arg97.getSize()));
  in97.close();
  ifstream in98(
      params_path+"/arg98.data",
      ios::in | ios::binary);
  in98.read((char *)(arg98.getData()), sizeof(float) * (arg98.getSize()));
  in98.close();
  ifstream in99(
      params_path+"/arg99.data",
      ios::in | ios::binary);
  in99.read((char *)(arg99.getData()), sizeof(float) * (arg99.getSize()));
  in99.close();
  ifstream in100(
      params_path+"/arg100.data",
      ios::in | ios::binary);
  in100.read((char *)(arg100.getData()), sizeof(float) * (arg100.getSize()));
  in100.close();
  ifstream in101(
      params_path+"/arg101.data",
      ios::in | ios::binary);
  in101.read((char *)(arg101.getData()), sizeof(float) * (arg101.getSize()));
  in101.close();
  ifstream in102(
      params_path+"/arg102.data",
      ios::in | ios::binary);
  in102.read((char *)(arg102.getData()), sizeof(float) * (arg102.getSize()));
  in102.close();
  ifstream in103(
      params_path+"/arg103.data",
      ios::in | ios::binary);
  in103.read((char *)(arg103.getData()), sizeof(float) * (arg103.getSize()));
  in103.close();
  ifstream in104(
      params_path+"/arg104.data",
      ios::in | ios::binary);
  in104.read((char *)(arg104.getData()), sizeof(float) * (arg104.getSize()));
  in104.close();
  ifstream in105(
      params_path+"/arg105.data",
      ios::in | ios::binary);
  in105.read((char *)(arg105.getData()), sizeof(float) * (arg105.getSize()));
  in105.close();
  ifstream in106(
      params_path+"/arg106.data",
      ios::in | ios::binary);
  in106.read((char *)(arg106.getData()), sizeof(float) * (arg106.getSize()));
  in106.close();
  ifstream in107(
      params_path+"/arg107.data",
      ios::in | ios::binary);
  in107.read((char *)(arg107.getData()), sizeof(float) * (arg107.getSize()));
  in107.close();
  ifstream in108(
      params_path+"/arg108.data",
      ios::in | ios::binary);
  in108.read((char *)(arg108.getData()), sizeof(float) * (arg108.getSize()));
  in108.close();
  ifstream in109(
      params_path+"/arg109.data",
      ios::in | ios::binary);
  in109.read((char *)(arg109.getData()), sizeof(float) * (arg109.getSize()));
  in109.close();
  ifstream in110(
      params_path+"/arg110.data",
      ios::in | ios::binary);
  in110.read((char *)(arg110.getData()), sizeof(float) * (arg110.getSize()));
  in110.close();
  ifstream in111(
      params_path+"/arg111.data",
      ios::in | ios::binary);
  in111.read((char *)(arg111.getData()), sizeof(float) * (arg111.getSize()));
  in111.close();
  ifstream in112(
      params_path+"/arg112.data",
      ios::in | ios::binary);
  in112.read((char *)(arg112.getData()), sizeof(float) * (arg112.getSize()));
  in112.close();
  ifstream in113(
      params_path+"/arg113.data",
      ios::in | ios::binary);
  in113.read((char *)(arg113.getData()), sizeof(float) * (arg113.getSize()));
  in113.close();
  ifstream in114(
      params_path+"/arg114.data",
      ios::in | ios::binary);
  in114.read((char *)(arg114.getData()), sizeof(float) * (arg114.getSize()));
  in114.close();
  ifstream in115(
      params_path+"/arg115.data",
      ios::in | ios::binary);
  in115.read((char *)(arg115.getData()), sizeof(float) * (arg115.getSize()));
  in115.close();
  ifstream in116(
      params_path+"/arg116.data",
      ios::in | ios::binary);
  in116.read((char *)(arg116.getData()), sizeof(float) * (arg116.getSize()));
  in116.close();
  ifstream in117(
      params_path+"/arg117.data",
      ios::in | ios::binary);
  in117.read((char *)(arg117.getData()), sizeof(float) * (arg117.getSize()));
  in117.close();
  ifstream in118(
      params_path+"/arg118.data",
      ios::in | ios::binary);
  in118.read((char *)(arg118.getData()), sizeof(float) * (arg118.getSize()));
  in118.close();
  ifstream in119(
      params_path+"/arg119.data",
      ios::in | ios::binary);
  in119.read((char *)(arg119.getData()), sizeof(float) * (arg119.getSize()));
  in119.close();
  ifstream in120(
      params_path+"/arg120.data",
      ios::in | ios::binary);
  in120.read((char *)(arg120.getData()), sizeof(float) * (arg120.getSize()));
  in120.close();
  ifstream in121(
      params_path+"/arg121.data",
      ios::in | ios::binary);
  in121.read((char *)(arg121.getData()), sizeof(float) * (arg121.getSize()));
  in121.close();
  ifstream in122(
      params_path+"/arg122.data",
      ios::in | ios::binary);
  in122.read((char *)(arg122.getData()), sizeof(float) * (arg122.getSize()));
  in122.close();
  ifstream in123(
      params_path+"/arg123.data",
      ios::in | ios::binary);
  in123.read((char *)(arg123.getData()), sizeof(float) * (arg123.getSize()));
  in123.close();
  ifstream in124(
      params_path+"/arg124.data",
      ios::in | ios::binary);
  in124.read((char *)(arg124.getData()), sizeof(float) * (arg124.getSize()));
  in124.close();
  ifstream in125(
      params_path+"/arg125.data",
      ios::in | ios::binary);
  in125.read((char *)(arg125.getData()), sizeof(float) * (arg125.getSize()));
  in125.close();
  ifstream in126(
      params_path+"/arg126.data",
      ios::in | ios::binary);
  in126.read((char *)(arg126.getData()), sizeof(float) * (arg126.getSize()));
  in126.close();
  ifstream in127(
      params_path+"/arg127.data",
      ios::in | ios::binary);
  in127.read((char *)(arg127.getData()), sizeof(float) * (arg127.getSize()));
  in127.close();
  ifstream in128(
      params_path+"/arg128.data",
      ios::in | ios::binary);
  in128.read((char *)(arg128.getData()), sizeof(float) * (arg128.getSize()));
  in128.close();
  ifstream in129(
      params_path+"/arg129.data",
      ios::in | ios::binary);
  in129.read((char *)(arg129.getData()), sizeof(float) * (arg129.getSize()));
  in129.close();
  ifstream in130(
      params_path+"/arg130.data",
      ios::in | ios::binary);
  in130.read((char *)(arg130.getData()), sizeof(float) * (arg130.getSize()));
  in130.close();
  ifstream in131(
      params_path+"/arg131.data",
      ios::in | ios::binary);
  in131.read((char *)(arg131.getData()), sizeof(float) * (arg131.getSize()));
  in131.close();
  ifstream in132(
      params_path+"/arg132.data",
      ios::in | ios::binary);
  in132.read((char *)(arg132.getData()), sizeof(float) * (arg132.getSize()));
  in132.close();
  ifstream in133(
      params_path+"/arg133.data",
      ios::in | ios::binary);
  in133.read((char *)(arg133.getData()), sizeof(float) * (arg133.getSize()));
  in133.close();
  ifstream in134(
      params_path+"/arg134.data",
      ios::in | ios::binary);
  in134.read((char *)(arg134.getData()), sizeof(float) * (arg134.getSize()));
  in134.close();
  ifstream in135(
      params_path+"/arg135.data",
      ios::in | ios::binary);
  in135.read((char *)(arg135.getData()), sizeof(float) * (arg135.getSize()));
  in135.close();
  ifstream in136(
      params_path+"/arg136.data",
      ios::in | ios::binary);
  in136.read((char *)(arg136.getData()), sizeof(float) * (arg136.getSize()));
  in136.close();
  ifstream in137(
      params_path+"/arg137.data",
      ios::in | ios::binary);
  in137.read((char *)(arg137.getData()), sizeof(float) * (arg137.getSize()));
  in137.close();
  ifstream in138(
      params_path+"/arg138.data",
      ios::in | ios::binary);
  in138.read((char *)(arg138.getData()), sizeof(float) * (arg138.getSize()));
  in138.close();
  ifstream in139(
      params_path+"/arg139.data",
      ios::in | ios::binary);
  in139.read((char *)(arg139.getData()), sizeof(float) * (arg139.getSize()));
  in139.close();
  ifstream in140(
      params_path+"/arg140.data",
      ios::in | ios::binary);
  in140.read((char *)(arg140.getData()), sizeof(float) * (arg140.getSize()));
  in140.close();
  ifstream in141(
      params_path+"/arg141.data",
      ios::in | ios::binary);
  in141.read((char *)(arg141.getData()), sizeof(float) * (arg141.getSize()));
  in141.close();
  ifstream in142(
      params_path+"/arg142.data",
      ios::in | ios::binary);
  in142.read((char *)(arg142.getData()), sizeof(float) * (arg142.getSize()));
  in142.close();
  ifstream in143(
      params_path+"/arg143.data",
      ios::in | ios::binary);
  in143.read((char *)(arg143.getData()), sizeof(float) * (arg143.getSize()));
  in143.close();
  ifstream in144(
      params_path+"/arg144.data",
      ios::in | ios::binary);
  in144.read((char *)(arg144.getData()), sizeof(float) * (arg144.getSize()));
  in144.close();
  ifstream in145(
      params_path+"/arg145.data",
      ios::in | ios::binary);
  in145.read((char *)(arg145.getData()), sizeof(float) * (arg145.getSize()));
  in145.close();
  ifstream in146(
      params_path+"/arg146.data",
      ios::in | ios::binary);
  in146.read((char *)(arg146.getData()), sizeof(float) * (arg146.getSize()));
  in146.close();
  ifstream in147(
      params_path+"/arg147.data",
      ios::in | ios::binary);
  in147.read((char *)(arg147.getData()), sizeof(float) * (arg147.getSize()));
  in147.close();
  ifstream in148(
      params_path+"/arg148.data",
      ios::in | ios::binary);
  in148.read((char *)(arg148.getData()), sizeof(float) * (arg148.getSize()));
  in148.close();
  ifstream in149(
      params_path+"/arg149.data",
      ios::in | ios::binary);
  in149.read((char *)(arg149.getData()), sizeof(float) * (arg149.getSize()));
  in149.close();
  ifstream in150(
      params_path+"/arg150.data",
      ios::in | ios::binary);
  in150.read((char *)(arg150.getData()), sizeof(float) * (arg150.getSize()));
  in150.close();
  ifstream in151(
      params_path+"/arg151.data",
      ios::in | ios::binary);
  in151.read((char *)(arg151.getData()), sizeof(float) * (arg151.getSize()));
  in151.close();
  ifstream in152(
      params_path+"/arg152.data",
      ios::in | ios::binary);
  in152.read((char *)(arg152.getData()), sizeof(float) * (arg152.getSize()));
  in152.close();
  ifstream in153(
      params_path+"/arg153.data",
      ios::in | ios::binary);
  in153.read((char *)(arg153.getData()), sizeof(float) * (arg153.getSize()));
  in153.close();
  ifstream in154(
      params_path+"/arg154.data",
      ios::in | ios::binary);
  in154.read((char *)(arg154.getData()), sizeof(float) * (arg154.getSize()));
  in154.close();
  ifstream in155(
      params_path+"/arg155.data",
      ios::in | ios::binary);
  in155.read((char *)(arg155.getData()), sizeof(float) * (arg155.getSize()));
  in155.close();
  ifstream in156(
      params_path+"/arg156.data",
      ios::in | ios::binary);
  in156.read((char *)(arg156.getData()), sizeof(float) * (arg156.getSize()));
  in156.close();
  ifstream in157(
      params_path+"/arg157.data",
      ios::in | ios::binary);
  in157.read((char *)(arg157.getData()), sizeof(float) * (arg157.getSize()));
  in157.close();
  ifstream in158(
      params_path+"/arg158.data",
      ios::in | ios::binary);
  in158.read((char *)(arg158.getData()), sizeof(float) * (arg158.getSize()));
  in158.close();
  ifstream in159(
      params_path+"/arg159.data",
      ios::in | ios::binary);
  in159.read((char *)(arg159.getData()), sizeof(float) * (arg159.getSize()));
  in159.close();
  ifstream in160(
      params_path+"/arg160.data",
      ios::in | ios::binary);
  in160.read((char *)(arg160.getData()), sizeof(float) * (arg160.getSize()));
  in160.close();
  ifstream in161(
      params_path+"/arg161.data",
      ios::in | ios::binary);
  in161.read((char *)(arg161.getData()), sizeof(float) * (arg161.getSize()));
  in161.close();
  ifstream in162(
      params_path+"/arg162.data",
      ios::in | ios::binary);
  in162.read((char *)(arg162.getData()), sizeof(float) * (arg162.getSize()));
  in162.close();
  ifstream in163(
      params_path+"/arg163.data",
      ios::in | ios::binary);
  in163.read((char *)(arg163.getData()), sizeof(float) * (arg163.getSize()));
  in163.close();
  ifstream in164(
      params_path+"/arg164.data",
      ios::in | ios::binary);
  in164.read((char *)(arg164.getData()), sizeof(float) * (arg164.getSize()));
  in164.close();
  ifstream in165(
      params_path+"/arg165.data",
      ios::in | ios::binary);
  in165.read((char *)(arg165.getData()), sizeof(float) * (arg165.getSize()));
  in165.close();
  ifstream in166(
      params_path+"/arg166.data",
      ios::in | ios::binary);
  in166.read((char *)(arg166.getData()), sizeof(float) * (arg166.getSize()));
  in166.close();
  ifstream in167(
      params_path+"/arg167.data",
      ios::in | ios::binary);
  in167.read((char *)(arg167.getData()), sizeof(float) * (arg167.getSize()));
  in167.close();
  ifstream in168(
      params_path+"/arg168.data",
      ios::in | ios::binary);
  in168.read((char *)(arg168.getData()), sizeof(float) * (arg168.getSize()));
  in168.close();
  ifstream in169(
      params_path+"/arg169.data",
      ios::in | ios::binary);
  in169.read((char *)(arg169.getData()), sizeof(float) * (arg169.getSize()));
  in169.close();
  ifstream in170(
      params_path+"/arg170.data",
      ios::in | ios::binary);
  in170.read((char *)(arg170.getData()), sizeof(float) * (arg170.getSize()));
  in170.close();
  ifstream in171(
      params_path+"/arg171.data",
      ios::in | ios::binary);
  in171.read((char *)(arg171.getData()), sizeof(float) * (arg171.getSize()));
  in171.close();
  ifstream in172(
      params_path+"/arg172.data",
      ios::in | ios::binary);
  in172.read((char *)(arg172.getData()), sizeof(float) * (arg172.getSize()));
  in172.close();
  ifstream in173(
      params_path+"/arg173.data",
      ios::in | ios::binary);
  in173.read((char *)(arg173.getData()), sizeof(float) * (arg173.getSize()));
  in173.close();
  ifstream in174(
      params_path+"/arg174.data",
      ios::in | ios::binary);
  in174.read((char *)(arg174.getData()), sizeof(float) * (arg174.getSize()));
  in174.close();
  ifstream in175(
      params_path+"/arg175.data",
      ios::in | ios::binary);
  in175.read((char *)(arg175.getData()), sizeof(float) * (arg175.getSize()));
  in175.close();
  ifstream in176(
      params_path+"/arg176.data",
      ios::in | ios::binary);
  in176.read((char *)(arg176.getData()), sizeof(float) * (arg176.getSize()));
  in176.close();
  ifstream in177(
      params_path+"/arg177.data",
      ios::in | ios::binary);
  in177.read((char *)(arg177.getData()), sizeof(float) * (arg177.getSize()));
  in177.close();
  ifstream in178(
      params_path+"/arg178.data",
      ios::in | ios::binary);
  in178.read((char *)(arg178.getData()), sizeof(float) * (arg178.getSize()));
  in178.close();
  ifstream in179(
      params_path+"/arg179.data",
      ios::in | ios::binary);
  in179.read((char *)(arg179.getData()), sizeof(float) * (arg179.getSize()));
  in179.close();
  ifstream in180(
      params_path+"/arg180.data",
      ios::in | ios::binary);
  in180.read((char *)(arg180.getData()), sizeof(float) * (arg180.getSize()));
  in180.close();
  ifstream in181(
      params_path+"/arg181.data",
      ios::in | ios::binary);
  in181.read((char *)(arg181.getData()), sizeof(float) * (arg181.getSize()));
  in181.close();
  ifstream in182(
      params_path+"/arg182.data",
      ios::in | ios::binary);
  in182.read((char *)(arg182.getData()), sizeof(float) * (arg182.getSize()));
  in182.close();
  ifstream in183(
      params_path+"/arg183.data",
      ios::in | ios::binary);
  in183.read((char *)(arg183.getData()), sizeof(float) * (arg183.getSize()));
  in183.close();
  ifstream in184(
      params_path+"/arg184.data",
      ios::in | ios::binary);
  in184.read((char *)(arg184.getData()), sizeof(float) * (arg184.getSize()));
  in184.close();
  ifstream in185(
      params_path+"/arg185.data",
      ios::in | ios::binary);
  in185.read((char *)(arg185.getData()), sizeof(float) * (arg185.getSize()));
  in185.close();
  ifstream in186(
      params_path+"/arg186.data",
      ios::in | ios::binary);
  in186.read((char *)(arg186.getData()), sizeof(float) * (arg186.getSize()));
  in186.close();
  ifstream in187(
      params_path+"/arg187.data",
      ios::in | ios::binary);
  in187.read((char *)(arg187.getData()), sizeof(float) * (arg187.getSize()));
  in187.close();
  ifstream in188(
      params_path+"/arg188.data",
      ios::in | ios::binary);
  in188.read((char *)(arg188.getData()), sizeof(float) * (arg188.getSize()));
  in188.close();
  ifstream in189(
      params_path+"/arg189.data",
      ios::in | ios::binary);
  in189.read((char *)(arg189.getData()), sizeof(float) * (arg189.getSize()));
  in189.close();
  ifstream in190(
      params_path+"/arg190.data",
      ios::in | ios::binary);
  in190.read((char *)(arg190.getData()), sizeof(float) * (arg190.getSize()));
  in190.close();
  ifstream in191(
      params_path+"/arg191.data",
      ios::in | ios::binary);
  in191.read((char *)(arg191.getData()), sizeof(float) * (arg191.getSize()));
  in191.close();
  ifstream in192(
      params_path+"/arg192.data",
      ios::in | ios::binary);
  in192.read((char *)(arg192.getData()), sizeof(float) * (arg192.getSize()));
  in192.close();
  ifstream in193(
      params_path+"/arg193.data",
      ios::in | ios::binary);
  in193.read((char *)(arg193.getData()), sizeof(float) * (arg193.getSize()));
  in193.close();
  ifstream in194(
      params_path+"/arg194.data",
      ios::in | ios::binary);
  in194.read((char *)(arg194.getData()), sizeof(float) * (arg194.getSize()));
  in194.close();
  ifstream in195(
      params_path+"/arg195.data",
      ios::in | ios::binary);
  in195.read((char *)(arg195.getData()), sizeof(float) * (arg195.getSize()));
  in195.close();
  ifstream in196(
      params_path+"/arg196.data",
      ios::in | ios::binary);
  in196.read((char *)(arg196.getData()), sizeof(float) * (arg196.getSize()));
  in196.close();
  ifstream in197(
      params_path+"/arg197.data",
      ios::in | ios::binary);
  in197.read((char *)(arg197.getData()), sizeof(float) * (arg197.getSize()));
  in197.close();
  ifstream in198(
      params_path+"/arg198.data",
      ios::in | ios::binary);
  in198.read((char *)(arg198.getData()), sizeof(float) * (arg198.getSize()));
  in198.close();
  ifstream in199(
      params_path+"/arg199.data",
      ios::in | ios::binary);
  in199.read((char *)(arg199.getData()), sizeof(float) * (arg199.getSize()));
  in199.close();
  ifstream in200(
      params_path+"/arg200.data",
      ios::in | ios::binary);
  in200.read((char *)(arg200.getData()), sizeof(float) * (arg200.getSize()));
  in200.close();
  ifstream in201(
      params_path+"/arg201.data",
      ios::in | ios::binary);
  in201.read((char *)(arg201.getData()), sizeof(float) * (arg201.getSize()));
  in201.close();
  ifstream in202(
      params_path+"/arg202.data",
      ios::in | ios::binary);
  in202.read((char *)(arg202.getData()), sizeof(float) * (arg202.getSize()));
  in202.close();
  ifstream in203(
      params_path+"/arg203.data",
      ios::in | ios::binary);
  in203.read((char *)(arg203.getData()), sizeof(float) * (arg203.getSize()));
  in203.close();
  ifstream in204(
      params_path+"/arg204.data",
      ios::in | ios::binary);
  in204.read((char *)(arg204.getData()), sizeof(float) * (arg204.getSize()));
  in204.close();
  ifstream in205(
      params_path+"/arg205.data",
      ios::in | ios::binary);
  in205.read((char *)(arg205.getData()), sizeof(float) * (arg205.getSize()));
  in205.close();
  ifstream in206(
      params_path+"/arg206.data",
      ios::in | ios::binary);
  in206.read((char *)(arg206.getData()), sizeof(float) * (arg206.getSize()));
  in206.close();
  ifstream in207(
      params_path+"/arg207.data",
      ios::in | ios::binary);
  in207.read((char *)(arg207.getData()), sizeof(float) * (arg207.getSize()));
  in207.close();
  ifstream in208(
      params_path+"/arg208.data",
      ios::in | ios::binary);
  in208.read((char *)(arg208.getData()), sizeof(float) * (arg208.getSize()));
  in208.close();
  ifstream in209(
      params_path+"/arg209.data",
      ios::in | ios::binary);
  in209.read((char *)(arg209.getData()), sizeof(float) * (arg209.getSize()));
  in209.close();
  ifstream in210(
      params_path+"/arg210.data",
      ios::in | ios::binary);
  in210.read((char *)(arg210.getData()), sizeof(float) * (arg210.getSize()));
  in210.close();
  ifstream in211(
      params_path+"/arg211.data",
      ios::in | ios::binary);
  in211.read((char *)(arg211.getData()), sizeof(float) * (arg211.getSize()));
  in211.close();
  ifstream in212(
      params_path+"/arg212.data",
      ios::in | ios::binary);
  in212.read((char *)(arg212.getData()), sizeof(float) * (arg212.getSize()));
  in212.close();
  ifstream in213(
      params_path+"/arg213.data",
      ios::in | ios::binary);
  in213.read((char *)(arg213.getData()), sizeof(float) * (arg213.getSize()));
  in213.close();
  ifstream in214(
      params_path+"/arg214.data",
      ios::in | ios::binary);
  in214.read((char *)(arg214.getData()), sizeof(float) * (arg214.getSize()));
  in214.close();
  ifstream in215(
      params_path+"/arg215.data",
      ios::in | ios::binary);
  in215.read((char *)(arg215.getData()), sizeof(float) * (arg215.getSize()));
  in215.close();
  ifstream in216(
      params_path+"/arg216.data",
      ios::in | ios::binary);
  in216.read((char *)(arg216.getData()), sizeof(float) * (arg216.getSize()));
  in216.close();
  ifstream in217(
      params_path+"/arg217.data",
      ios::in | ios::binary);
  in217.read((char *)(arg217.getData()), sizeof(float) * (arg217.getSize()));
  in217.close();
  ifstream in218(
      params_path+"/arg218.data",
      ios::in | ios::binary);
  in218.read((char *)(arg218.getData()), sizeof(float) * (arg218.getSize()));
  in218.close();
  ifstream in219(
      params_path+"/arg219.data",
      ios::in | ios::binary);
  in219.read((char *)(arg219.getData()), sizeof(float) * (arg219.getSize()));
  in219.close();
  ifstream in220(
      params_path+"/arg220.data",
      ios::in | ios::binary);
  in220.read((char *)(arg220.getData()), sizeof(float) * (arg220.getSize()));
  in220.close();
  ifstream in221(
      params_path+"/arg221.data",
      ios::in | ios::binary);
  in221.read((char *)(arg221.getData()), sizeof(float) * (arg221.getSize()));
  in221.close();
  ifstream in222(
      params_path+"/arg222.data",
      ios::in | ios::binary);
  in222.read((char *)(arg222.getData()), sizeof(float) * (arg222.getSize()));
  in222.close();
  ifstream in223(
      params_path+"/arg223.data",
      ios::in | ios::binary);
  in223.read((char *)(arg223.getData()), sizeof(float) * (arg223.getSize()));
  in223.close();
  ifstream in224(
      params_path+"/arg224.data",
      ios::in | ios::binary);
  in224.read((char *)(arg224.getData()), sizeof(float) * (arg224.getSize()));
  in224.close();
  ifstream in225(
      params_path+"/arg225.data",
      ios::in | ios::binary);
  in225.read((char *)(arg225.getData()), sizeof(float) * (arg225.getSize()));
  in225.close();
  ifstream in226(
      params_path+"/arg226.data",
      ios::in | ios::binary);
  in226.read((char *)(arg226.getData()), sizeof(float) * (arg226.getSize()));
  in226.close();
  ifstream in227(
      params_path+"/arg227.data",
      ios::in | ios::binary);
  in227.read((char *)(arg227.getData()), sizeof(float) * (arg227.getSize()));
  in227.close();
  ifstream in228(
      params_path+"/arg228.data",
      ios::in | ios::binary);
  in228.read((char *)(arg228.getData()), sizeof(float) * (arg228.getSize()));
  in228.close();
  ifstream in229(
      params_path+"/arg229.data",
      ios::in | ios::binary);
  in229.read((char *)(arg229.getData()), sizeof(float) * (arg229.getSize()));
  in229.close();
  ifstream in230(
      params_path+"/arg230.data",
      ios::in | ios::binary);
  in230.read((char *)(arg230.getData()), sizeof(float) * (arg230.getSize()));
  in230.close();
  ifstream in231(
      params_path+"/arg231.data",
      ios::in | ios::binary);
  in231.read((char *)(arg231.getData()), sizeof(float) * (arg231.getSize()));
  in231.close();
  ifstream in232(
      params_path+"/arg232.data",
      ios::in | ios::binary);
  in232.read((char *)(arg232.getData()), sizeof(float) * (arg232.getSize()));
  in232.close();
  ifstream in233(
      params_path+"/arg233.data",
      ios::in | ios::binary);
  in233.read((char *)(arg233.getData()), sizeof(float) * (arg233.getSize()));
  in233.close();
  ifstream in234(
      params_path+"/arg234.data",
      ios::in | ios::binary);
  in234.read((char *)(arg234.getData()), sizeof(float) * (arg234.getSize()));
  in234.close();
  ifstream in235(
      params_path+"/arg235.data",
      ios::in | ios::binary);
  in235.read((char *)(arg235.getData()), sizeof(float) * (arg235.getSize()));
  in235.close();
  ifstream in236(
      params_path+"/arg236.data",
      ios::in | ios::binary);
  in236.read((char *)(arg236.getData()), sizeof(float) * (arg236.getSize()));
  in236.close();
  ifstream in237(
      params_path+"/arg237.data",
      ios::in | ios::binary);
  in237.read((char *)(arg237.getData()), sizeof(float) * (arg237.getSize()));
  in237.close();
  ifstream in238(
      params_path+"/arg238.data",
      ios::in | ios::binary);
  in238.read((char *)(arg238.getData()), sizeof(float) * (arg238.getSize()));
  in238.close();
  ifstream in239(
      params_path+"/arg239.data",
      ios::in | ios::binary);
  in239.read((char *)(arg239.getData()), sizeof(float) * (arg239.getSize()));
  in239.close();
  ifstream in240(
      params_path+"/arg240.data",
      ios::in | ios::binary);
  in240.read((char *)(arg240.getData()), sizeof(float) * (arg240.getSize()));
  in240.close();
  ifstream in241(
      params_path+"/arg241.data",
      ios::in | ios::binary);
  in241.read((char *)(arg241.getData()), sizeof(float) * (arg241.getSize()));
  in241.close();
  ifstream in242(
      params_path+"/arg242.data",
      ios::in | ios::binary);
  in242.read((char *)(arg242.getData()), sizeof(float) * (arg242.getSize()));
  in242.close();
  ifstream in243(
      params_path+"/arg243.data",
      ios::in | ios::binary);
  in243.read((char *)(arg243.getData()), sizeof(float) * (arg243.getSize()));
  in243.close();
  ifstream in244(
      params_path+"/arg244.data",
      ios::in | ios::binary);
  in244.read((char *)(arg244.getData()), sizeof(float) * (arg244.getSize()));
  in244.close();
  ifstream in245(
      params_path+"/arg245.data",
      ios::in | ios::binary);
  in245.read((char *)(arg245.getData()), sizeof(float) * (arg245.getSize()));
  in245.close();
  ifstream in246(
      params_path+"/arg246.data",
      ios::in | ios::binary);
  in246.read((char *)(arg246.getData()), sizeof(float) * (arg246.getSize()));
  in246.close();
  ifstream in247(
      params_path+"/arg247.data",
      ios::in | ios::binary);
  in247.read((char *)(arg247.getData()), sizeof(float) * (arg247.getSize()));
  in247.close();
  ifstream in248(
      params_path+"/arg248.data",
      ios::in | ios::binary);
  in248.read((char *)(arg248.getData()), sizeof(float) * (arg248.getSize()));
  in248.close();
  ifstream in249(
      params_path+"/arg249.data",
      ios::in | ios::binary);
  in249.read((char *)(arg249.getData()), sizeof(float) * (arg249.getSize()));
  in249.close();
  ifstream in250(
      params_path+"/arg250.data",
      ios::in | ios::binary);
  in250.read((char *)(arg250.getData()), sizeof(float) * (arg250.getSize()));
  in250.close();
  ifstream in251(
      params_path+"/arg251.data",
      ios::in | ios::binary);
  in251.read((char *)(arg251.getData()), sizeof(float) * (arg251.getSize()));
  in251.close();
  ifstream in252(
      params_path+"/arg252.data",
      ios::in | ios::binary);
  in252.read((char *)(arg252.getData()), sizeof(float) * (arg252.getSize()));
  in252.close();
  ifstream in253(
      params_path+"/arg253.data",
      ios::in | ios::binary);
  in253.read((char *)(arg253.getData()), sizeof(float) * (arg253.getSize()));
  in253.close();
  ifstream in254(
      params_path+"/arg254.data",
      ios::in | ios::binary);
  in254.read((char *)(arg254.getData()), sizeof(float) * (arg254.getSize()));
  in254.close();
  ifstream in255(
      params_path+"/arg255.data",
      ios::in | ios::binary);
  in255.read((char *)(arg255.getData()), sizeof(float) * (arg255.getSize()));
  in255.close();
  ifstream in256(
      params_path+"/arg256.data",
      ios::in | ios::binary);
  in256.read((char *)(arg256.getData()), sizeof(float) * (arg256.getSize()));
  in256.close();
  ifstream in257(
      params_path+"/arg257.data",
      ios::in | ios::binary);
  in257.read((char *)(arg257.getData()), sizeof(float) * (arg257.getSize()));
  in257.close();
  ifstream in258(
      params_path+"/arg258.data",
      ios::in | ios::binary);
  in258.read((char *)(arg258.getData()), sizeof(float) * (arg258.getSize()));
  in258.close();
  ifstream in259(
      params_path+"/arg259.data",
      ios::in | ios::binary);
  in259.read((char *)(arg259.getData()), sizeof(float) * (arg259.getSize()));
  in259.close();
  ifstream in260(
      params_path+"/arg260.data",
      ios::in | ios::binary);
  in260.read((char *)(arg260.getData()), sizeof(float) * (arg260.getSize()));
  in260.close();
  ifstream in261(
      params_path+"/arg261.data",
      ios::in | ios::binary);
  in261.read((char *)(arg261.getData()), sizeof(float) * (arg261.getSize()));
  in261.close();
  ifstream in262(
      params_path+"/arg262.data",
      ios::in | ios::binary);
  in262.read((char *)(arg262.getData()), sizeof(float) * (arg262.getSize()));
  in262.close();
  ifstream in263(
      params_path+"/arg263.data",
      ios::in | ios::binary);
  in263.read((char *)(arg263.getData()), sizeof(float) * (arg263.getSize()));
  in263.close();
  ifstream in264(
      params_path+"/arg264.data",
      ios::in | ios::binary);
  in264.read((char *)(arg264.getData()), sizeof(float) * (arg264.getSize()));
  in264.close();
  ifstream in265(
      params_path+"/arg265.data",
      ios::in | ios::binary);
  in265.read((char *)(arg265.getData()), sizeof(float) * (arg265.getSize()));
  in265.close();
  ifstream in266(
      params_path+"/arg266.data",
      ios::in | ios::binary);
  in266.read((char *)(arg266.getData()), sizeof(float) * (arg266.getSize()));
  in266.close();
  ifstream in267(
      params_path+"/arg267.data",
      ios::in | ios::binary);
  in267.read((char *)(arg267.getData()), sizeof(float) * (arg267.getSize()));
  in267.close();
  ifstream in268(
      params_path+"/arg268.data",
      ios::in | ios::binary);
  in268.read((char *)(arg268.getData()), sizeof(float) * (arg268.getSize()));
  in268.close();
  ifstream in269(
      params_path+"/arg269.data",
      ios::in | ios::binary);
  in269.read((char *)(arg269.getData()), sizeof(float) * (arg269.getSize()));
  in269.close();
  ifstream in270(
      params_path+"/arg270.data",
      ios::in | ios::binary);
  in270.read((char *)(arg270.getData()), sizeof(float) * (arg270.getSize()));
  in270.close();
  ifstream in271(
      params_path+"/arg271.data",
      ios::in | ios::binary);
  in271.read((char *)(arg271.getData()), sizeof(float) * (arg271.getSize()));
  in271.close();
  ifstream in272(
      params_path+"/arg272.data",
      ios::in | ios::binary);
  in272.read((char *)(arg272.getData()), sizeof(float) * (arg272.getSize()));
  in272.close();
  ifstream in273(
      params_path+"/arg273.data",
      ios::in | ios::binary);
  in273.read((char *)(arg273.getData()), sizeof(float) * (arg273.getSize()));
  in273.close();
  ifstream in274(
      params_path+"/arg274.data",
      ios::in | ios::binary);
  in274.read((char *)(arg274.getData()), sizeof(float) * (arg274.getSize()));
  in274.close();
  ifstream in275(
      params_path+"/arg275.data",
      ios::in | ios::binary);
  in275.read((char *)(arg275.getData()), sizeof(float) * (arg275.getSize()));
  in275.close();
  ifstream in276(
      params_path+"/arg276.data",
      ios::in | ios::binary);
  in276.read((char *)(arg276.getData()), sizeof(float) * (arg276.getSize()));
  in276.close();
  ifstream in277(
      params_path+"/arg277.data",
      ios::in | ios::binary);
  in277.read((char *)(arg277.getData()), sizeof(float) * (arg277.getSize()));
  in277.close();
  ifstream in278(
      params_path+"/arg278.data",
      ios::in | ios::binary);
  in278.read((char *)(arg278.getData()), sizeof(float) * (arg278.getSize()));
  in278.close();
  ifstream in279(
      params_path+"/arg279.data",
      ios::in | ios::binary);
  in279.read((char *)(arg279.getData()), sizeof(float) * (arg279.getSize()));
  in279.close();
  ifstream in280(
      params_path+"/arg280.data",
      ios::in | ios::binary);
  in280.read((char *)(arg280.getData()), sizeof(float) * (arg280.getSize()));
  in280.close();
  ifstream in281(
      params_path+"/arg281.data",
      ios::in | ios::binary);
  in281.read((char *)(arg281.getData()), sizeof(float) * (arg281.getSize()));
  in281.close();
  ifstream in282(
      params_path+"/arg282.data",
      ios::in | ios::binary);
  in282.read((char *)(arg282.getData()), sizeof(float) * (arg282.getSize()));
  in282.close();
  ifstream in283(
      params_path+"/arg283.data",
      ios::in | ios::binary);
  in283.read((char *)(arg283.getData()), sizeof(float) * (arg283.getSize()));
  in283.close();
  ifstream in284(
      params_path+"/arg284.data",
      ios::in | ios::binary);
  in284.read((char *)(arg284.getData()), sizeof(float) * (arg284.getSize()));
  in284.close();
  ifstream in285(
      params_path+"/arg285.data",
      ios::in | ios::binary);
  in285.read((char *)(arg285.getData()), sizeof(float) * (arg285.getSize()));
  in285.close();
  ifstream in286(
      params_path+"/arg286.data",
      ios::in | ios::binary);
  in286.read((char *)(arg286.getData()), sizeof(float) * (arg286.getSize()));
  in286.close();
  ifstream in287(
      params_path+"/arg287.data",
      ios::in | ios::binary);
  in287.read((char *)(arg287.getData()), sizeof(float) * (arg287.getSize()));
  in287.close();
  ifstream in288(
      params_path+"/arg288.data",
      ios::in | ios::binary);
  in288.read((char *)(arg288.getData()), sizeof(float) * (arg288.getSize()));
  in288.close();
  ifstream in289(
      params_path+"/arg289.data",
      ios::in | ios::binary);
  in289.read((char *)(arg289.getData()), sizeof(float) * (arg289.getSize()));
  in289.close();
  ifstream in290(
      params_path+"/arg290.data",
      ios::in | ios::binary);
  in290.read((char *)(arg290.getData()), sizeof(float) * (arg290.getSize()));
  in290.close();
  ifstream in291(
      params_path+"/arg291.data",
      ios::in | ios::binary);
  in291.read((char *)(arg291.getData()), sizeof(float) * (arg291.getSize()));
  in291.close();
  ifstream in292(
      params_path+"/arg292.data",
      ios::in | ios::binary);
  in292.read((char *)(arg292.getData()), sizeof(float) * (arg292.getSize()));
  in292.close();
  ifstream in293(
      params_path+"/arg293.data",
      ios::in | ios::binary);
  in293.read((char *)(arg293.getData()), sizeof(float) * (arg293.getSize()));
  in293.close();
  ifstream in294(
      params_path+"/arg294.data",
      ios::in | ios::binary);
  in294.read((char *)(arg294.getData()), sizeof(float) * (arg294.getSize()));
  in294.close();
  ifstream in295(
      params_path+"/arg295.data",
      ios::in | ios::binary);
  in295.read((char *)(arg295.getData()), sizeof(float) * (arg295.getSize()));
  in295.close();
  ifstream in296(
      params_path+"/arg296.data",
      ios::in | ios::binary);
  in296.read((char *)(arg296.getData()), sizeof(float) * (arg296.getSize()));
  in296.close();
  ifstream in297(
      params_path+"/arg297.data",
      ios::in | ios::binary);
  in297.read((char *)(arg297.getData()), sizeof(float) * (arg297.getSize()));
  in297.close();
  ifstream in298(
      params_path+"/arg298.data",
      ios::in | ios::binary);
  in298.read((char *)(arg298.getData()), sizeof(float) * (arg298.getSize()));
  in298.close();
  ifstream in299(
      params_path+"/arg299.data",
      ios::in | ios::binary);
  in299.read((char *)(arg299.getData()), sizeof(float) * (arg299.getSize()));
  in299.close();
  ifstream in300(
      params_path+"/arg300.data",
      ios::in | ios::binary);
  in300.read((char *)(arg300.getData()), sizeof(float) * (arg300.getSize()));
  in300.close();
  ifstream in301(
      params_path+"/arg301.data",
      ios::in | ios::binary);
  in301.read((char *)(arg301.getData()), sizeof(float) * (arg301.getSize()));
  in301.close();
  ifstream in302(
      params_path+"/arg302.data",
      ios::in | ios::binary);
  in302.read((char *)(arg302.getData()), sizeof(float) * (arg302.getSize()));
  in302.close();
  ifstream in303(
      params_path+"/arg303.data",
      ios::in | ios::binary);
  in303.read((char *)(arg303.getData()), sizeof(float) * (arg303.getSize()));
  in303.close();
  ifstream in304(
      params_path+"/arg304.data",
      ios::in | ios::binary);
  in304.read((char *)(arg304.getData()), sizeof(float) * (arg304.getSize()));
  in304.close();
  ifstream in305(
      params_path+"/arg305.data",
      ios::in | ios::binary);
  in305.read((char *)(arg305.getData()), sizeof(float) * (arg305.getSize()));
  in305.close();
  ifstream in306(
      params_path+"/arg306.data",
      ios::in | ios::binary);
  in306.read((char *)(arg306.getData()), sizeof(float) * (arg306.getSize()));
  in306.close();
  ifstream in307(
      params_path+"/arg307.data",
      ios::in | ios::binary);
  in307.read((char *)(arg307.getData()), sizeof(float) * (arg307.getSize()));
  in307.close();
  ifstream in308(
      params_path+"/arg308.data",
      ios::in | ios::binary);
  in308.read((char *)(arg308.getData()), sizeof(float) * (arg308.getSize()));
  in308.close();
  ifstream in309(
      params_path+"/arg309.data",
      ios::in | ios::binary);
  in309.read((char *)(arg309.getData()), sizeof(float) * (arg309.getSize()));
  in309.close();
  ifstream in310(
      params_path+"/arg310.data",
      ios::in | ios::binary);
  in310.read((char *)(arg310.getData()), sizeof(float) * (arg310.getSize()));
  in310.close();
  ifstream in311(
      params_path+"/arg311.data",
      ios::in | ios::binary);
  in311.read((char *)(arg311.getData()), sizeof(float) * (arg311.getSize()));
  in311.close();
  ifstream in312(
      params_path+"/arg312.data",
      ios::in | ios::binary);
  in312.read((char *)(arg312.getData()), sizeof(float) * (arg312.getSize()));
  in312.close();
  ifstream in313(
      params_path+"/arg313.data",
      ios::in | ios::binary);
  in313.read((char *)(arg313.getData()), sizeof(float) * (arg313.getSize()));
  in313.close();
  ifstream in314(
      params_path+"/arg314.data",
      ios::in | ios::binary);
  in314.read((char *)(arg314.getData()), sizeof(float) * (arg314.getSize()));
  in314.close();
  ifstream in315(
      params_path+"/arg315.data",
      ios::in | ios::binary);
  in315.read((char *)(arg315.getData()), sizeof(float) * (arg315.getSize()));
  in315.close();
  ifstream in316(
      params_path+"/arg316.data",
      ios::in | ios::binary);
  in316.read((char *)(arg316.getData()), sizeof(float) * (arg316.getSize()));
  in316.close();
  ifstream in317(
      params_path+"/arg317.data",
      ios::in | ios::binary);
  in317.read((char *)(arg317.getData()), sizeof(float) * (arg317.getSize()));
  in317.close();
  ifstream in318(
      params_path+"/arg318.data",
      ios::in | ios::binary);
  in318.read((char *)(arg318.getData()), sizeof(float) * (arg318.getSize()));
  in318.close();
  ifstream in319(
      params_path+"/arg319.data",
      ios::in | ios::binary);
  in319.read((char *)(arg319.getData()), sizeof(float) * (arg319.getSize()));
  in319.close();
  ifstream in320(
      params_path+"/arg320.data",
      ios::in | ios::binary);
  in320.read((char *)(arg320.getData()), sizeof(float) * (arg320.getSize()));
  in320.close();
  ifstream in321(
      params_path+"/arg321.data",
      ios::in | ios::binary);
  in321.read((char *)(arg321.getData()), sizeof(float) * (arg321.getSize()));
  in321.close();
  ifstream in322(
      params_path+"/arg322.data",
      ios::in | ios::binary);
  in322.read((char *)(arg322.getData()), sizeof(float) * (arg322.getSize()));
  in322.close();
  ifstream in323(
      params_path+"/arg323.data",
      ios::in | ios::binary);
  in323.read((char *)(arg323.getData()), sizeof(float) * (arg323.getSize()));
  in323.close();
  ifstream in324(
      params_path+"/arg324.data",
      ios::in | ios::binary);
  in324.read((char *)(arg324.getData()), sizeof(float) * (arg324.getSize()));
  in324.close();
  ifstream in325(
      params_path+"/arg325.data",
      ios::in | ios::binary);
  in325.read((char *)(arg325.getData()), sizeof(float) * (arg325.getSize()));
  in325.close();
  ifstream in326(
      params_path+"/arg326.data",
      ios::in | ios::binary);
  in326.read((char *)(arg326.getData()), sizeof(float) * (arg326.getSize()));
  in326.close();
  ifstream in327(
      params_path+"/arg327.data",
      ios::in | ios::binary);
  in327.read((char *)(arg327.getData()), sizeof(float) * (arg327.getSize()));
  in327.close();
  ifstream in328(
      params_path+"/arg328.data",
      ios::in | ios::binary);
  in328.read((char *)(arg328.getData()), sizeof(float) * (arg328.getSize()));
  in328.close();
  ifstream in329(
      params_path+"/arg329.data",
      ios::in | ios::binary);
  in329.read((char *)(arg329.getData()), sizeof(float) * (arg329.getSize()));
  in329.close();
  ifstream in330(
      params_path+"/arg330.data",
      ios::in | ios::binary);
  in330.read((char *)(arg330.getData()), sizeof(float) * (arg330.getSize()));
  in330.close();
  ifstream in331(
      params_path+"/arg331.data",
      ios::in | ios::binary);
  in331.read((char *)(arg331.getData()), sizeof(float) * (arg331.getSize()));
  in331.close();
  ifstream in332(
      params_path+"/arg332.data",
      ios::in | ios::binary);
  in332.read((char *)(arg332.getData()), sizeof(float) * (arg332.getSize()));
  in332.close();
  ifstream in333(
      params_path+"/arg333.data",
      ios::in | ios::binary);
  in333.read((char *)(arg333.getData()), sizeof(float) * (arg333.getSize()));
  in333.close();
  ifstream in334(
      params_path+"/arg334.data",
      ios::in | ios::binary);
  in334.read((char *)(arg334.getData()), sizeof(float) * (arg334.getSize()));
  in334.close();
  ifstream in335(
      params_path+"/arg335.data",
      ios::in | ios::binary);
  in335.read((char *)(arg335.getData()), sizeof(float) * (arg335.getSize()));
  in335.close();
  ifstream in336(
      params_path+"/arg336.data",
      ios::in | ios::binary);
  in336.read((char *)(arg336.getData()), sizeof(float) * (arg336.getSize()));
  in336.close();
  ifstream in337(
      params_path+"/arg337.data",
      ios::in | ios::binary);
  in337.read((char *)(arg337.getData()), sizeof(float) * (arg337.getSize()));
  in337.close();
  ifstream in338(
      params_path+"/arg338.data",
      ios::in | ios::binary);
  in338.read((char *)(arg338.getData()), sizeof(float) * (arg338.getSize()));
  in338.close();
  ifstream in339(
      params_path+"/arg339.data",
      ios::in | ios::binary);
  in339.read((char *)(arg339.getData()), sizeof(float) * (arg339.getSize()));
  in339.close();
  ifstream in340(
      params_path+"/arg340.data",
      ios::in | ios::binary);
  in340.read((char *)(arg340.getData()), sizeof(float) * (arg340.getSize()));
  in340.close();
  ifstream in341(
      params_path+"/arg341.data",
      ios::in | ios::binary);
  in341.read((char *)(arg341.getData()), sizeof(float) * (arg341.getSize()));
  in341.close();
  ifstream in342(
      params_path+"/arg342.data",
      ios::in | ios::binary);
  in342.read((char *)(arg342.getData()), sizeof(float) * (arg342.getSize()));
  in342.close();
  ifstream in343(
      params_path+"/arg343.data",
      ios::in | ios::binary);
  in343.read((char *)(arg343.getData()), sizeof(float) * (arg343.getSize()));
  in343.close();
  ifstream in344(
      params_path+"/arg344.data",
      ios::in | ios::binary);
  in344.read((char *)(arg344.getData()), sizeof(float) * (arg344.getSize()));
  in344.close();
  ifstream in345(
      params_path+"/arg345.data",
      ios::in | ios::binary);
  in345.read((char *)(arg345.getData()), sizeof(float) * (arg345.getSize()));
  in345.close();
  ifstream in346(
      params_path+"/arg346.data",
      ios::in | ios::binary);
  in346.read((char *)(arg346.getData()), sizeof(float) * (arg346.getSize()));
  in346.close();
  ifstream in347(
      params_path+"/arg347.data",
      ios::in | ios::binary);
  in347.read((char *)(arg347.getData()), sizeof(float) * (arg347.getSize()));
  in347.close();
  ifstream in348(
      params_path+"/arg348.data",
      ios::in | ios::binary);
  in348.read((char *)(arg348.getData()), sizeof(float) * (arg348.getSize()));
  in348.close();
  ifstream in349(
      params_path+"/arg349.data",
      ios::in | ios::binary);
  in349.read((char *)(arg349.getData()), sizeof(float) * (arg349.getSize()));
  in349.close();
  ifstream in350(
      params_path+"/arg350.data",
      ios::in | ios::binary);
  in350.read((char *)(arg350.getData()), sizeof(float) * (arg350.getSize()));
  in350.close();
  ifstream in351(
      params_path+"/arg351.data",
      ios::in | ios::binary);
  in351.read((char *)(arg351.getData()), sizeof(float) * (arg351.getSize()));
  in351.close();
  ifstream in352(
      params_path+"/arg352.data",
      ios::in | ios::binary);
  in352.read((char *)(arg352.getData()), sizeof(float) * (arg352.getSize()));
  in352.close();
  ifstream in353(
      params_path+"/arg353.data",
      ios::in | ios::binary);
  in353.read((char *)(arg353.getData()), sizeof(float) * (arg353.getSize()));
  in353.close();
  ifstream in354(
      params_path+"/arg354.data",
      ios::in | ios::binary);
  in354.read((char *)(arg354.getData()), sizeof(float) * (arg354.getSize()));
  in354.close();
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
  MemRef<float, 3> result({1, 80, 32000});
  start = system_clock::now();
  int generate_len = 80-pureStrContainer.getTokenCnt();
  cout<<"---------------------------start generate---------------------------"<<endl;
  for(int i=0;i<generate_len;i++){
    _mlir_ciface_forward(
      &result, &arg0, &arg1, &arg2, &arg3, &arg4, &arg5, &arg6, &arg7, &arg8,
      &arg9, &arg10, &arg11, &arg12, &arg13, &arg14, &arg15, &arg16, &arg17,
      &arg18, &arg19, &arg20, &arg21, &arg22, &arg23, &arg24, &arg25, &arg26,
      &arg27, &arg28, &arg29, &arg30, &arg31, &arg32, &arg33, &arg34, &arg35,
      &arg36, &arg37, &arg38, &arg39, &arg40, &arg41, &arg42, &arg43, &arg44,
      &arg45, &arg46, &arg47, &arg48, &arg49, &arg50, &arg51, &arg52, &arg53,
      &arg54, &arg55, &arg56, &arg57, &arg58, &arg59, &arg60, &arg61, &arg62,
      &arg63, &arg64, &arg65, &arg66, &arg67, &arg68, &arg69, &arg70, &arg71,
      &arg72, &arg73, &arg74, &arg75, &arg76, &arg77, &arg78, &arg79, &arg80,
      &arg81, &arg82, &arg83, &arg84, &arg85, &arg86, &arg87, &arg88, &arg89,
      &arg90, &arg91, &arg92, &arg93, &arg94, &arg95, &arg96, &arg97, &arg98,
      &arg99, &arg100, &arg101, &arg102, &arg103, &arg104, &arg105, &arg106,
      &arg107, &arg108, &arg109, &arg110, &arg111, &arg112, &arg113, &arg114,
      &arg115, &arg116, &arg117, &arg118, &arg119, &arg120, &arg121, &arg122,
      &arg123, &arg124, &arg125, &arg126, &arg127, &arg128, &arg129, &arg130,
      &arg131, &arg132, &arg133, &arg134, &arg135, &arg136, &arg137, &arg138,
      &arg139, &arg140, &arg141, &arg142, &arg143, &arg144, &arg145, &arg146,
      &arg147, &arg148, &arg149, &arg150, &arg151, &arg152, &arg153, &arg154,
      &arg155, &arg156, &arg157, &arg158, &arg159, &arg160, &arg161, &arg162,
      &arg163, &arg164, &arg165, &arg166, &arg167, &arg168, &arg169, &arg170,
      &arg171, &arg172, &arg173, &arg174, &arg175, &arg176, &arg177, &arg178,
      &arg179, &arg180, &arg181, &arg182, &arg183, &arg184, &arg185, &arg186,
      &arg187, &arg188, &arg189, &arg190, &arg191, &arg192, &arg193, &arg194,
      &arg195, &arg196, &arg197, &arg198, &arg199, &arg200, &arg201, &arg202,
      &arg203, &arg204, &arg205, &arg206, &arg207, &arg208, &arg209, &arg210,
      &arg211, &arg212, &arg213, &arg214, &arg215, &arg216, &arg217, &arg218,
      &arg219, &arg220, &arg221, &arg222, &arg223, &arg224, &arg225, &arg226,
      &arg227, &arg228, &arg229, &arg230, &arg231, &arg232, &arg233, &arg234,
      &arg235, &arg236, &arg237, &arg238, &arg239, &arg240, &arg241, &arg242,
      &arg243, &arg244, &arg245, &arg246, &arg247, &arg248, &arg249, &arg250,
      &arg251, &arg252, &arg253, &arg254, &arg255, &arg256, &arg257, &arg258,
      &arg259, &arg260, &arg261, &arg262, &arg263, &arg264, &arg265, &arg266,
      &arg267, &arg268, &arg269, &arg270, &arg271, &arg272, &arg273, &arg274,
      &arg275, &arg276, &arg277, &arg278, &arg279, &arg280, &arg281, &arg282,
      &arg283, &arg284, &arg285, &arg286, &arg287, &arg288, &arg289, &arg290,
      &arg291, &arg292, &arg293, &arg294, &arg295, &arg296, &arg297, &arg298,
      &arg299, &arg300, &arg301, &arg302, &arg303, &arg304, &arg305, &arg306,
      &arg307, &arg308, &arg309, &arg310, &arg311, &arg312, &arg313, &arg314,
      &arg315, &arg316, &arg317, &arg318, &arg319, &arg320, &arg321, &arg322,
      &arg323, &arg324, &arg325, &arg326, &arg327, &arg328, &arg329, &arg330,
      &arg331, &arg332, &arg333, &arg334, &arg335, &arg336, &arg337, &arg338,
      &arg339, &arg340, &arg341, &arg342, &arg343, &arg344, &arg345, &arg346,
      &arg347, &arg348, &arg349, &arg350, &arg351, &arg352, &arg353, &arg354,
      &pureStrContainer);
    int token_index = pureStrContainer.getTokenCnt()-1;
    int index = 0;
    int max_elem = result.getData()[token_index*32000];
    for(int j=index+1;j<32000;j++){
        cout<<result.getData()[token_index*32000+j]<<" ";
        if(result.getData()[token_index*32000+j]>max_elem){
            max_elem = result.getData()[token_index*32000+j];
            index = j;
        }
    }
    cout<<endl;
    pureStrContainer.getData()[pureStrContainer.getTokenCnt()] = index;
    cout<<"==================="<<endl;
    cout<<index<<endl;
    if(index == 2){
        break;
    }
    pureStrContainer.setTokenCnt(pureStrContainer.getTokenCnt()+1);
  }
  end = system_clock::now();
  duration = duration_cast<milliseconds>(end - start);
  cout << "llama exection use time: " << duration.count() << "ms" << endl;
  for (int i = 0; i < 80; i++) {
    cout << pureStrContainer.getData()[i] << " ";
  }
  cout<<endl;
  return 0;
}