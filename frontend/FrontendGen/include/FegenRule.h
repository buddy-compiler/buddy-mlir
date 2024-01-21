#ifndef FEGEN_ANTLR_RULE_H
#define FEGEN_ANTLR_RULE_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include <string>
#include <map>

namespace fegen {

enum class RuleType { LEX_RULE, PARSE_RULE };

class FegenRule {
public:
  FegenRule(RuleType r, llvm::StringRef name);
  FegenRule(RuleType r, llvm::StringRef name, llvm::StringRef content);
  FegenRule() = delete;

  RuleType getRuleType();
  llvm::StringRef getName();
  llvm::StringRef getContent();
  void setContent(llvm::StringRef content);

private:
  RuleType ruleType;
  std::string name;
  std::string content;
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
  FegenRule *at(llvm::StringRef name);

  void insert(FegenRule *rule);

  void emitG4File(llvm::raw_fd_ostream &os);

  static FegenRule *createRule(RuleType r, llvm::StringRef name);

  static FegenRule *createRule(RuleType r, llvm::StringRef name,
                               llvm::StringRef content);
};
} // namespace fegen

#endif