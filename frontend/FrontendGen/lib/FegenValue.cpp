#include "FegenValue.h"
#include <iostream>

fegen::FegenType::FegenType(std::string name) : name(name) {}

bool fegen::FegenType::operator==(fegen::FegenType *another) {
  return this == another;
}

llvm::StringRef fegen::FegenType::getName() { return this->name; }

fegen::FegenValue::LiteralValue::LiteralValue(
    std::variant<int, float, std::string> value)
    : value(value) {}

fegen::FegenValue::RuleInOutputValue::RuleInOutputValue(SourceSectionType ty,
                                                        FegenValue *value)
    : srcSectype(ty), value(value), sourceRule(value->getSource()) {}

fegen::FegenValue::RuleAttributeValue::RuleAttributeValue(AttributeKind kind,
                                                          FegenRule *src)
    : attrType(kind), sourceRule(src) {}

fegen::FegenValue::FegenValue(FegenRule *source, FegenType *type,
                              ValueKind valueKind, std::string name,
                              bool isList)
    : source(source), type(type), valueKind(valueKind), name(name),
      isList(isList) {}

fegen::FegenRule *fegen::FegenValue::getSource() { return this->source; }
fegen::FegenType *fegen::FegenValue::getType() { return this->type; }
llvm::StringRef fegen::FegenValue::getName() { return this->name; }

fegen::ValueKind fegen::FegenValue::getValueKind() { return this->valueKind; }

std::variant<std::monostate, fegen::FegenValue::LiteralValue,
             fegen::FegenValue::RuleInOutputValue,
             fegen::FegenValue::RuleAttributeValue>&
fegen::FegenValue::getBindingValue() {
  return this->bindingInfo;
}
void fegen::FegenValue::setBindingValue(
    std::variant<std::monostate, fegen::FegenValue::LiteralValue,
                 fegen::FegenValue::RuleInOutputValue,
                 fegen::FegenValue::RuleAttributeValue>
        info) {
  this->bindingInfo = info;
}

bool fegen::FegenValue::ifList() {
  return this->isList;
}

void fegen::FegenValue::setRuleIndex(int index) {
  this->ruleIndex = index;
}

int fegen::FegenValue::getRuleIndex() {
  return this->ruleIndex;
}

fegen::ValueMap::~ValueMap() {
  // delete types
  auto iter = this->typeMap.begin();
  while (iter != this->typeMap.end()) {
    delete iter->second;
    iter++;
  }
  // delete values
  auto iter_1 = this->valueMap.begin();
  while (iter_1 != this->valueMap.end()) {
    delete iter_1->second;
    iter_1++;
  }
}

fegen::ValueMap &fegen::ValueMap::getMap() {
  static ValueMap vm;
  return vm;
}

fegen::FegenType *fegen::ValueMap::createType(std::string name) {
  return new fegen::FegenType(name);
}

fegen::FegenValue *fegen::ValueMap::createValue(fegen::FegenRule *source,
                                                std::string name,
                                                fegen::FegenType *type,
                                                ValueKind valueKind,
                                                bool isList) {
  return new fegen::FegenValue(source, type, valueKind, name, isList);
}

fegen::FegenType *fegen::ValueMap::findType(std::string name) {
  auto it = this->typeMap.find(name);
  if (it == this->typeMap.end()) {
    return nullptr;
  } else {
    return it->second;
  }
}

bool fegen::ValueMap::insertType(fegen::FegenType *type) {
  auto p = this->typeMap.insert({type->name, type});
  return p.second;
}

fegen::FegenValue *fegen::ValueMap::findValue(std::string name) {
  auto it = this->valueMap.find(name);
  if (it == this->valueMap.end()) {
    return nullptr;
  } else {
    return it->second;
  }
}

bool fegen::ValueMap::insertValue(fegen::FegenValue *value) {
  auto p = this->valueMap.insert({value->name, value});
  return p.second;
}
