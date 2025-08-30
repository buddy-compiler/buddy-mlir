//====- Sema.cpp ---------------------------------------------------------===//
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
                       Dialect *&dialect, std::vector<Op *> &ops) {
  module->setRules(rules);
  module->seDialect(dialect);
  module->setOps(ops);
}
/// Set Rule's node.
void Sema::actOnRule(Rule *rule,
                     std::vector<GeneratorAndOthers *> &generators) {
  rule->setGenerators(generators);
}

/// Set Dialect's nodes.
void Sema::actOnDialect(Dialect *dialect, llvm::StringRef defName,
                        llvm::StringRef name, llvm::StringRef cppNamespace) {
  dialect->setDefName(defName);
  dialect->setName(name);
  dialect->setCppNamespace(cppNamespace);
}

/// Make a op and make it in the ops.
void Sema::actOnOps(std::vector<Op *> &ops, llvm::StringRef opName,
                    DAG *arguments, DAG *results,
                    std::vector<Builder *> &builders) {
  Op *op = new Op();
  op->setOpName(opName);
  op->setArguments(arguments);
  op->setResults(results);
  op->setBuilders(builders);
  ops.push_back(op);
}

void Sema::actOnDag(DAG *&arguments, DAG &dag) { arguments = new DAG(dag); }
