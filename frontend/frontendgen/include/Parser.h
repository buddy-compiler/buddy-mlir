//====- Parser.h --------------------------------------------------------===//
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

#ifndef INCLUDE_PARSER_H
#define INCLUDE_PARSER_H
#include "AST.h"
#include "Lexer.h"
#include "Sema.h"
#include "Terminator.h"
#include "Token.h"
namespace frontendgen {

/// A class for parsing tokens.
class Parser {
  Lexer &lexer;
  Token token;
  Sema &action;
  Terminators &terminators;

public:
  Parser(Lexer &lexer, Sema &action, Terminators &terminators)
      : lexer(lexer), action(action), terminators(terminators) {
    advance();
  }
  bool consume(tokenKinds kind);
  bool consumeNoAdvance(tokenKinds kind);
  void advance();
  Module *parser();
  void compilEngine(Module *module);
  void parserRules(Rule *rule);
  void parserGenerator(GeneratorAndOthers *generatorAndOthers);
  void lookToken();
  AntlrBase::baseKind getAntlrBaseKind(llvm::StringRef name);
  void parserIdentifier(GeneratorAndOthers *generatorAndOthers);
  void parserTerminator(GeneratorAndOthers *generatorAndOthers);
  void parserPBExpression(GeneratorAndOthers *generatorAndOthers);
  void parserDialect(Dialect *&dialect, llvm::StringRef defName);
  bool parserOp(std::vector<Op *> &ops, llvm::StringRef opName);
  void parserCurlyBracketOpen(GeneratorAndOthers *generatorAndOthers);
  void parserDAG(DAG *&dag);
  void parserBuilders(std::vector<Builder *> &builders);
  void parserCode(llvm::StringRef &code);
  void parserCArg(llvm::StringRef &operand, llvm::StringRef &value);
};
} // namespace frontendgen

#endif
