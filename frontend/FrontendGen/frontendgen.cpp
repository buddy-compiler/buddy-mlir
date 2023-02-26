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

#include "CGModule.h"
#include "Diagnostics.h"
#include "Lexer.h"
#include "Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

llvm::cl::opt<std::string> inputFileName("f", llvm::cl::desc("<input file>"));
llvm::cl::opt<std::string> grammarName("g", llvm::cl::desc("<grammar name>"));

namespace {
enum Action { none, dumpAst, dumpAntlr, dumpAll, dumpVisitor };
}

llvm::cl::opt<Action> emitAction(
    "emit", llvm::cl::desc("Select the kind of output desired"),
    llvm::cl::values(clEnumValN(dumpAst, "ast", "Out put the ast")),
    llvm::cl::values(clEnumValN(dumpAntlr, "antlr", "Out put the antlr file")),
    llvm::cl::values(clEnumValN(dumpVisitor, "visitor",
                                "Out put the visitor file")),
    llvm::cl::values(clEnumValN(dumpAll, "all", "put out all file")));

/// Control generation of ast, tablegen files and antlr files.
void emit(frontendgen::Module *module, frontendgen::Terminators &terminators) {
  bool emitAst = emitAction == Action::dumpAst;
  bool emitAntlr =
      emitAction == Action::dumpAntlr || emitAction == Action::dumpAll;
  bool emitVisitor =
      emitAction == Action::dumpVisitor || emitAction == Action::dumpAll;
  // Emit antlr file.
  if (emitAntlr) {
    if (grammarName.empty()) {
      llvm::errs() << "if you want to emit g4 file you have to point out the "
                      "name of grammar.\n";
      return;
    }
    std::error_code EC;
    llvm::sys::fs::OpenFlags openFlags = llvm::sys::fs::OpenFlags::OF_None;
    std::string outputFileName = grammarName.c_str();
    outputFileName += ".g4";
    auto Out = llvm::ToolOutputFile(outputFileName, EC, openFlags);
    frontendgen::CGModule CGmodule(module, Out.os(), terminators);
    CGmodule.emitAntlr(grammarName);
    Out.keep();
  }
  // Emit antlr's AST.
  if (emitAst && !module->getRules().empty()) {
    llvm::raw_fd_ostream os(-1, true);
    frontendgen::CGModule CGmodule(module, os, terminators);
    CGmodule.emitAST();
  }
  // Emit visitor file.
  if (emitVisitor && !module->getRules().empty()) {
    std::error_code EC;
    llvm::sys::fs::OpenFlags openFlags = llvm::sys::fs::OpenFlags::OF_None;
    std::string outputFileName("MLIR");
    outputFileName = outputFileName + grammarName + "Visitor.h";
    auto Out = llvm::ToolOutputFile(outputFileName, EC, openFlags);
    frontendgen::CGModule CGmodule(module, Out.os(), terminators);
    CGmodule.emitMLIRVisitor(grammarName);
    Out.keep();
  }
  // Free memory.
  for (auto rule : module->getRules()) {
    for (auto generatorsAndOthers : rule->getGeneratorsAndOthers()) {
      for (auto element : generatorsAndOthers->getGenerator()) {
        delete element;
      }
      delete generatorsAndOthers;
    }
    delete rule;
  }

  delete module->getDialect();
  for (auto op : module->getOps()) {
    delete op->getArguments();
    delete op->getResults();
    for (auto builder : op->getBuilders()) {
      delete builder->getDag();
      delete builder;
    }
    delete op;
  }
  delete module;
}

int main(int argc, char *argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> file =
      llvm::MemoryBuffer::getFile(inputFileName.c_str());
  if (std::error_code bufferError = file.getError()) {
    llvm::errs() << "error read: " << bufferError.message() << '\n';
    exit(1);
  }
  llvm::SourceMgr srcMgr;
  srcMgr.AddNewSourceBuffer(std::move(*file), llvm::SMLoc());
  frontendgen::DiagnosticEngine diagnostic(srcMgr);
  frontendgen::Lexer lexer(srcMgr, diagnostic);
  frontendgen::Sema action;
  frontendgen::Terminators terminators;
  frontendgen::Parser parser(lexer, action, terminators);
  frontendgen::Module *module = parser.parser();
  emit(module, terminators);
  return 0;
}
