#ifndef FEGEN_ANTLR_RULE_H
#define FEGEN_ANTLR_RULE_H

#include "FegenIR.h"
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
  FegenRule(RuleType r, llvm::StringRef name, llvm::StringRef content);
  FegenRule() = delete;

  RuleType getRuleType();
  llvm::StringRef getName();

  llvm::StringRef getContent();
  void setContent(llvm::StringRef content);

  FegenIR getIRContent();
  void setIRContent(FegenIR irContent);

  // return true if existed
  bool addInput(FegenValue *value);

  // return true if existed
  bool addReturn(FegenValue *value);

private:
  RuleType ruleType;
  std::string name;
  // grammar section, for parse rule, its production rules; for lex rule, its
  // lex defination help to generate g4 file
  std::string content;
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

public:
  ~RuleMap();

  static RuleMap &getRuleMap() {
    static RuleMap rm;
    return rm;
  }
  FegenRule *find(llvm::StringRef name);

  void insert(FegenRule *rule);

  void emitG4File(llvm::raw_fd_ostream &os);

  static FegenRule *createRule(RuleType r, llvm::StringRef name);

  static FegenRule *createRule(RuleType r, llvm::StringRef name,
                               llvm::StringRef content);
};
} // namespace fegen

#endif