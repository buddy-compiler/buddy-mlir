//====- Sema.h  ----------------------------------------------------------===//
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

#ifndef INCLUDE_SEMA_H
#define INCLUDE_SEMA_H
#include "AST.h"

namespace frontendgen {

class Sema {
public:
  void actOnModule(Module *module, std::vector<Rule *> &rules,
                   Dialect *&dialect, std::vector<Op *> &ops);
  void actOnRule(Rule *rule, std::vector<GeneratorAndOthers *> &generators);
  void actOnDialect(Dialect *dialect, llvm::StringRef defName,
                    llvm::StringRef name, llvm::StringRef cppNamespace);
  void actOnOps(std::vector<Op *> &ops, llvm::StringRef opName, DAG *arguments,
                DAG *results, std::vector<Builder *> &builder);
  void actOnDag(DAG *&arguments, DAG &dag);
};
} // namespace frontendgen
#endif
