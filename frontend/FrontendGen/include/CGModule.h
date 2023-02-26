//====- CGModule.h -------------------------------------------------------===//
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

#ifndef INCLUDE_CGMODULE_H
#define INCLUDE_CGMODULE_H
#include "AST.h"
#include "Terminator.h"
#include "llvm/Support/raw_ostream.h"
namespace frontendgen {

/// TypeMap is used to store type maps.The cppMap is used to map c++ types,
/// argumentsMap and resultsMap are used to map TableGen types.
class TypeMap {
  llvm::StringMap<llvm::StringRef> cppMap;
  llvm::StringMap<llvm::StringRef> argumentsMap;
  llvm::StringMap<llvm::StringRef> resultsMap;

public:
  TypeMap() {
#define CPPMAP(key, value) cppMap.insert(std::pair(key, value));
#define RESULTSMAP(key, value) resultsMap.insert(std::pair(key, value));
#define ARGUMENTSMAP(key, value) argumentsMap.insert(std::pair(key, value));
#include "TypeMap.def"
  }
  llvm::StringRef findCppMap(llvm::StringRef value);
  llvm::StringRef findArgumentMap(llvm::StringRef value);
  llvm::StringRef findResultsMap(llvm::StringRef value);
};

/// The class for code generation.
class CGModule {
  Terminators &terminators;
  Module *module;
  llvm::raw_fd_ostream &os;
  TypeMap typeMap;

public:
  CGModule(Module *module, llvm::raw_fd_ostream &os, Terminators &terminators)
      : terminators(terminators), module(module), os(os) {}
  void emitAST();
  void emitAntlr(llvm::StringRef grammarName);
  void emit(const std::vector<Rule *> &rules);
  void emit(const std::vector<GeneratorAndOthers *> &generators);
  void emit(const std::vector<AntlrBase *> &generator);
  void emitGrammar(llvm::StringRef grammarName);
  void emitTerminators();
  void emitCustomTerminators();
  void emitWSAndComment();
  void emitIncludes(llvm::StringRef grammarName);
  void emitMLIRVisitor(llvm::StringRef grammarName);
  void emitClass(llvm::StringRef grammarName);
  void emitRuleVisitor(llvm::StringRef grammarName, Rule *rule);
  void emitBuilders(Rule *rule);
  void emitBuilder(llvm::StringRef builderOp, int index);
  Op *findOp(llvm::StringRef opName);
  void emitOp(Op *op, int index);
};
} // namespace frontendgen

#endif
