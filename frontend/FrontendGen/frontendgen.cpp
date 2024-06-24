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

#include <iostream>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include "FegenLexer.h"
#include "FegenParser.h"
#include "FegenVisitor.h"
#include "antlr4-common.h"

llvm::cl::opt<std::string> inputFileName("f", llvm::cl::desc("<input file>"));

namespace {
enum Action { none, dumpAst, dumpAntlr, dumpAll, dumpVisitor };
}

llvm::cl::opt<Action> emitAction(
    "emit", llvm::cl::desc("Select the kind of output desired"),
    llvm::cl::values(clEnumValN(dumpAst, "ast", "Out put the ast")),
    llvm::cl::values(clEnumValN(dumpAntlr, "g4", "Out put the g4 file")),
    llvm::cl::values(clEnumValN(dumpVisitor, "visitor",
                                "Out put the visitor file")),
    llvm::cl::values(clEnumValN(dumpAll, "all", "put out all file")));

int dumpAST(fegen::FegenParser::FegenSpecContext *moduleAST) {
  llvm::errs() << moduleAST->toStringTree(1 /* prety format*/) << "\n";
  return 0;
}

int main(int argc, char *argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  // Parse the input file with ANTLR.
  std::fstream in(inputFileName);
  antlr4::ANTLRInputStream input(in);
  fegen::FegenLexer lexer(&input);
  antlr4::CommonTokenStream tokens(&lexer);
  fegen::FegenParser parser(&tokens);
  auto moduleAST = parser.fegenSpec();

  fegen::FegenVisitor visitor;
  visitor.visit(moduleAST);
  // visitor.emitG4();
  visitor.emitTypeDefination();
  visitor.emitDialectDefination();
  visitor.emitOpDefination();
  // if (emitAction == dumpAst) {
  //   return dumpAST(moduleAST);
  // }
  // if (emitAction == dumpAntlr) {

  // }
  return 0;
}
