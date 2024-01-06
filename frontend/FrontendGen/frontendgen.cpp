//====- frontendgen.cpp -------------------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file is the driver of frontendgen project.
//
//===----------------------------------------------------------------------===//

#include "FegenLexer.h"
#include "FegenParser.h"
#include "FegenVisitor.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace antlr4;

llvm::cl::opt<std::string> inputFileName("f", llvm::cl::desc("<input file>"));
// llvm::cl::opt<std::string> grammarName("g", llvm::cl::desc("<grammar name>"));

namespace {
enum Action { none, dumpAst, dumpTablegen, dumpAntlr, dumpAll, dumpVisitor };
}

llvm::cl::opt<Action> emitAction(
    "emit", llvm::cl::desc("Select the kind of output desired"),
    llvm::cl::values(clEnumValN(dumpAst, "ast", "Out put the ast")),
    llvm::cl::values(clEnumValN(dumpTablegen, "tablegen", "Out put the tablegen file")),
    llvm::cl::values(clEnumValN(dumpAntlr, "antlr", "Out put the antlr file")),
    llvm::cl::values(clEnumValN(dumpVisitor, "visitor",
                                "Out put the visitor file")),
    llvm::cl::values(clEnumValN(dumpAll, "all", "put out all file")));


int main(int argc, char *argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  // parser input file with ANTLR
  std::fstream in(inputFileName);
  ANTLRInputStream input(in);
  FegenLexer lexer(&input);
  CommonTokenStream tokens(&lexer);
  FegenParser parser(&tokens);
  auto fegenModuleCst = parser.fegenModule();

  // dump fegen ast
  if (emitAction == Action::dumpAst){
    llvm::errs() << fegenModuleCst->toStringTree() << "\n";
  }
  return 0;
}