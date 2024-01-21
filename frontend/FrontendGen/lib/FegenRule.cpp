#include "FegenRule.h"

fegen::FegenRule::FegenRule(RuleType r, llvm::StringRef name)
    : ruleType(r), name(name) {}

fegen::FegenRule::FegenRule(RuleType r, llvm::StringRef name,
                            llvm::StringRef content)
    : ruleType(r), name(name), content(content) {}

fegen::RuleType fegen::FegenRule::getRuleType() { return this->ruleType; }

llvm::StringRef fegen::FegenRule::getName() { return this->name; }

llvm::StringRef fegen::FegenRule::getContent() { return this->content; }

void fegen::FegenRule::setContent(llvm::StringRef content) {
  this->content = content;
}

fegen::RuleMap::~RuleMap() {
  for (auto p : this->name2RuleMap) {
    auto rule = p.second;
    delete rule;
  }
}

fegen::FegenRule *fegen::RuleMap::at(llvm::StringRef name) {
  auto rule = this->name2RuleMap.find(name);
  if (rule == this->name2RuleMap.end()) {
    // TODO: output error
    std::cerr << "cannot find rule: " << name.str() << '\n';
  }
  return rule->second;
}

void fegen::RuleMap::insert(fegen::FegenRule *rule) {
  auto name = rule->getName();
  auto flag = this->name2RuleMap.insert({name, rule});
  if (!flag.second) {
    // TODO: output error
    std::cerr << "rule " << name.str() << " is already in the map." << '\n';
  }
}

void fegen::RuleMap::emitG4File(llvm::raw_fd_ostream &os) {
  for (auto pair : this->name2RuleMap) {
    auto rule = pair.second;
    os << rule->getName() << ':' << '\n';
    os << '\t' << rule->getContent() << '\n' << ';' << '\n' << '\n';
  }
}

fegen::FegenRule *fegen::RuleMap::createRule(fegen::RuleType r,
                                             llvm::StringRef name) {
  return new fegen::FegenRule(r, name);
}

fegen::FegenRule *fegen::RuleMap::createRule(fegen::RuleType r,
                                             llvm::StringRef name,
                                             llvm::StringRef content) {
  return new fegen::FegenRule(r, name, content);
}
