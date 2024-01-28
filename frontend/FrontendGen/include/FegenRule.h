#ifndef FEGEN_ANTLR_RULE_H
#define FEGEN_ANTLR_RULE_H

#include "FegenIR.h"
#include "FegenParser.h"
#include "FegenParserBaseVisitor.h"
#include "FegenValue.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include <map>
#include <string>

namespace fegen {

enum class RuleType { LEX_RULE, PARSE_RULE };

class FegenRule {
  friend class RuleMap;

public:
  FegenRule(RuleType r, llvm::StringRef name);

  RuleType getRuleType();
  llvm::StringRef getName();

  antlr4::ParserRuleContext * getGrammarContext();
  void setGrammarContext(antlr4::ParserRuleContext * ctx);

  FegenIR getIRContent();
  void setIRContent(FegenIR irContent);

  // return true if existed
  bool addInput(FegenValue *value);

  // return true if existed
  bool addReturn(FegenValue *value);

private:
  RuleType ruleType;
  std::string name;
  // grammar parser tree, lexerAntlrRule or parserAntlrRule
  antlr4::ParserRuleContext *grammarContext = nullptr;
  // inputs section
  std::map<llvm::StringRef, FegenValue *> inputList;
  // returns section
  std::map<llvm::StringRef, FegenValue *> returnList;
  // IR section
  FegenIR irContent;
};

class RuleMap {
private:
  RuleMap() {}
  RuleMap(const RuleMap &) = delete;
  RuleMap(const RuleMap &&) = delete;
  RuleMap &operator=(const RuleMap &) = delete;
  std::map<llvm::StringRef, FegenRule *> name2RuleMap;
  std::string grammarName;

public:
  ~RuleMap();

  static RuleMap &getRuleMap() {
    static RuleMap rm;
    return rm;
  }
  FegenRule *find(llvm::StringRef name);

  void insert(FegenRule *rule);

  void setGrammarName(llvm::StringRef name);

  llvm::StringRef getGrammarName();

  void emitG4File(llvm::raw_fd_ostream &os);

  void emitVisitorFile(llvm::raw_fd_ostream &headfile, llvm::raw_fd_ostream &cppfile);

  static FegenRule *createRule(RuleType r, llvm::StringRef name);
};

class GrammarContentGenerator : public FegenParserBaseVisitor {
public:
  GrammarContentGenerator() = default;

  std::any
  visitLexerAntlrRule(FegenParser::LexerAntlrRuleContext *ctx) override;

  std::any
  visitLexerAlternatives(FegenParser::LexerAlternativesContext *ctx) override;

  std::any
  visitLexerAlternative(FegenParser::LexerAlternativeContext *ctx) override;

  std::any
  visitLexerSuffixedRule(FegenParser::LexerSuffixedRuleContext *ctx) override;

  std::any
  visitParserAntlrRule(FegenParser::ParserAntlrRuleContext *ctx) override;

  std::any visitAlternatives(FegenParser::AlternativesContext *ctx) override;

  std::any visitAlternative(FegenParser::AlternativeContext *ctx) override;

  std::any visitSuffixedRule(FegenParser::SuffixedRuleContext *ctx) override;

  std::any visitParenSurroundedElem(
      FegenParser::ParenSurroundedElemContext *ctx) override;
};
} // namespace fegen
#endif