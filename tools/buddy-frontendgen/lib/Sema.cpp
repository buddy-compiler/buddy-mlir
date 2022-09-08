//====- buddy-opt.cpp - The driver of buddy-mlir --------------------------===//
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

#include "Sema.h"
#include "llvm/Support/raw_ostream.h"
using namespace frontendgen;

/// Set Module's nodes.
void Sema::actOnModule(Module *module, std::vector<Rule *> &rules,
                       Dialect *&dialect, std::vector<Op *> &ops,
                       std::vector<Opinterface *> &opInterfaces) {
  module->setRules(rules);
  module->seDialect(dialect);
  module->setOps(ops);
  module->setOpInterfaces(opInterfaces);
}
/// Set Rule's node.
void Sema::actOnRule(Rule *rule,
                     std::vector<GeneratorAndOthers *> &generators) {
  rule->setGenerators(generators);
}

/// Set Dialect's nodes.
void Sema::actOnDialect(Dialect *dialect, llvm::StringRef defName,
                        llvm::StringRef name,
                        llvm::StringRef emitAccessorPrefix,
                        llvm::StringRef cppNamespace) {
  dialect->setDefName(defName);
  dialect->setEmitAccessorPrefix(emitAccessorPrefix);
  dialect->setName(name);
  dialect->setCppNamespace(cppNamespace);
}

/// Make a op and make it in the ops.
void Sema::actOnOps(std::vector<Op *> &ops, llvm::StringRef opName,
                    llvm::StringRef mnemonic, llvm::StringRef traits,
                    llvm::StringRef summary, llvm::StringRef description,
                    DAG *arguments, DAG *results, bool hasCustomAssemblyFormat,
                    std::vector<Builder *> &builders, bool hasVerifier,
                    llvm::StringRef assemblyFormat, llvm::StringRef regions,
                    llvm::StringRef extraClassDeclaration,
                    bool skipDefaultBuilders, bool hasCanonicalizer) {
  Op *op = new Op();
  op->setOpName(opName);
  op->setMnemonic(mnemonic);
  op->setTraits(traits);
  op->setSummary(summary);
  op->setDescription(description);
  op->setArguments(arguments);
  op->setResults(results);
  op->setHasCustomAssemblyFormat(hasCustomAssemblyFormat);
  op->setBuilders(builders);
  op->setHasVerifier(hasVerifier);
  op->setAssemblyFormat(assemblyFormat);
  op->setRegions(regions);
  op->setExtraClassDeclaration(extraClassDeclaration);
  op->setSkipDefaultBuilders(skipDefaultBuilders);
  op->setHasCanonicalizer(hasCanonicalizer);
  ops.push_back(op);
}

void Sema::actOnOpInterfaces(std::vector<Opinterface *> &opInterfaces,
                             llvm::StringRef defName, llvm::StringRef name,
                             llvm::StringRef methods,
                             llvm::StringRef description) {
  Opinterface *opInterface = new Opinterface();
  opInterface->setDefName(defName);
  opInterface->setName(name);
  opInterface->setDescription(description);
  opInterface->setMethods(methods);
  opInterfaces.push_back(opInterface);
}

void Sema::actOnDag(DAG *&arguments, DAG &dag) { arguments = new DAG(dag); }