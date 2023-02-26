//====- AST.h -------------------------------------------------------------===//
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

#ifndef INCLUDE_AST_H
#define INCLUDE_AST_H
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"
#include <vector>
namespace frontendgen {

/// Base class for all generator nodes.
class AntlrBase {
public:
  enum baseKind { rule, terminator, pbexpression };

private:
  baseKind kind;

protected:
  llvm::StringRef name;
  llvm::SMLoc loc;

public:
  virtual ~AntlrBase(){};
  AntlrBase(llvm::StringRef name, llvm::SMLoc loc, baseKind kind)
      : kind(kind), name(name), loc(loc) {}
  llvm::StringRef getName() { return name; }
  llvm::SMLoc getLoc() { return loc; }
  baseKind getKind() const { return kind; }
};

class GeneratorAndOthers {
  std::vector<AntlrBase *> generator;
  llvm::SmallVector<llvm::StringRef, 4> builderNames;
  llvm::SmallVector<int> builderIdxs;

public:
  void setbuilderNames(llvm::SmallVector<llvm::StringRef, 4> &builderNames) {
    this->builderNames = builderNames;
  }
  void setbuilderIdxs(llvm::SmallVector<int> &builderIdxs) {
    this->builderIdxs = builderIdxs;
  }
  std::vector<AntlrBase *> &getGenerator() { return generator; }
  llvm::SmallVector<llvm::StringRef, 4> getBuilderNames() {
    return this->builderNames;
  }
  llvm::SmallVector<int> getBuilderIndices() { return this->builderIdxs; }
};

/// This class is used to mark the node in the generator as a rule, and can also
/// store the generators of a rule.
class Rule : public AntlrBase {
  std::vector<GeneratorAndOthers *> generatorsAndOthers;

public:
  Rule(llvm::StringRef name, llvm::SMLoc loc, baseKind kind)
      : AntlrBase(name, loc, kind) {}
  static bool classof(const AntlrBase *base) {
    return base->getKind() == baseKind::rule;
  }
  void setGenerators(std::vector<GeneratorAndOthers *> &generatorsAndOthers) {
    this->generatorsAndOthers = generatorsAndOthers;
  }
  std::vector<GeneratorAndOthers *> getGeneratorsAndOthers() {
    return generatorsAndOthers;
  }
};
/// The class is used to mark the node in the generator as a terminator.
class Terminator : public AntlrBase {
public:
  Terminator(llvm::StringRef name, llvm::SMLoc loc, baseKind kind)
      : AntlrBase(name, loc, kind) {}
  static bool classof(const AntlrBase *base) {
    return base->getKind() == baseKind::terminator;
  }
};
/// The class is used to mark the node in the generator as regular expressions.
class PBExpression : public AntlrBase {
public:
  PBExpression(llvm::StringRef name, llvm::SMLoc loc, baseKind kind)
      : AntlrBase(name, loc, kind) {}
  static bool classof(const AntlrBase *base) {
    return base->getKind() == baseKind::terminator;
  }
};

/// The class is used to store the information about Dialect class in the
/// TableGen.
class Dialect {
  llvm::StringRef defName;
  llvm::StringRef name;
  llvm::StringRef cppNamespace;

public:
  Dialect() {}
  llvm::StringRef getName() { return name; }
  llvm::StringRef getCppNamespace() { return cppNamespace; }
  llvm::StringRef getDefName() { return defName; }
  void setName(llvm::StringRef name) { this->name = name; }
  void setDefName(llvm::StringRef defName) { this->defName = defName; }
  void setCppNamespace(llvm::StringRef cppNamespace) {
    this->cppNamespace = cppNamespace;
  }
};

class DAG {
  llvm::StringRef dagOperator;
  llvm::SmallVector<llvm::StringRef, 4> operands;
  llvm::SmallVector<llvm::StringRef, 4> operandNames;
  llvm::StringMap<llvm::StringRef> values;

public:
  DAG(){};
  DAG(const DAG &dag) {
    this->dagOperator = dag.dagOperator;
    this->operands = dag.operands;
    this->operandNames = dag.operandNames;
    this->values = dag.values;
  }

  void addOperand(llvm::StringRef operand, llvm::StringRef operandName) {
    operands.push_back(operand);
    operandNames.push_back(operandName);
  }
  void setValue(llvm::StringRef operand, llvm::StringRef value) {
    values[operand] = value;
  }
  llvm::StringRef findValue(llvm::StringRef operand) {
    if (values.find(operand) == values.end())
      return llvm::StringRef();
    return values[operand];
  }
  llvm::StringRef getDagOperater() { return dagOperator; }
  void setDagOperatpr(llvm::StringRef dagOperator) {
    this->dagOperator = dagOperator;
  }
  llvm::SmallVector<llvm::StringRef, 4> getOperands() { return operands; }
  llvm::SmallVector<llvm::StringRef, 4> getOperandNames() {
    return operandNames;
  }
};
/// The class is used to store builder in Op class.
class Builder {
  DAG *dag = nullptr;
  llvm::StringRef code;

public:
  Builder(DAG *dag, llvm::StringRef code) {
    this->dag = dag;
    this->code = code;
  }
  DAG *getDag() { return dag; }
  llvm::StringRef getCode() { return code; }
};

/// The class is used to store information about Op class in the TableGen.
class Op {
  llvm::StringRef opName;
  DAG *arguments;
  DAG *results;
  std::vector<Builder *> builders;

public:
  llvm::StringRef getOpName() { return opName; }
  DAG *getArguments() { return arguments; }
  DAG *getResults() { return results; }
  std::vector<Builder *> getBuilders() { return builders; }

  void setOpName(llvm::StringRef opName) { this->opName = opName; }

  void setArguments(DAG *arguments) { this->arguments = arguments; }
  void setResults(DAG *results) { this->results = results; }
  void setBuilders(std::vector<Builder *> &builders) {
    this->builders = builders;
  }
};

/// This class will become the root of a tree which contains all information we
/// need to generate code.
class Module {
  std::vector<Rule *> rules;
  Dialect *dialect;
  std::vector<Op *> ops;

public:
  std::vector<Rule *> &getRules() { return rules; }
  Dialect *getDialect() { return dialect; }
  std::vector<Op *> &getOps() { return ops; }
  void setRules(std::vector<Rule *> &rules) { this->rules = rules; }
  void seDialect(Dialect *&dialect) { this->dialect = dialect; }
  void setOps(std::vector<Op *> &ops) { this->ops = ops; }
};

} // namespace frontendgen
#endif
