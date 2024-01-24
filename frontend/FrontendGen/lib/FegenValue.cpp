#include "FegenValue.h"
#include <iostream>

fegen::FegenType::FegenType(std::string name) : name(name) {}

bool fegen::FegenType::operator==(fegen::FegenType *another) {
  return this == another;
}

llvm::StringRef fegen::FegenType::getName() { return this->name; }

fegen::FegenValue::FegenValue(FegenType *type, ValueKind valueKind,
                              std::string name, bool isList)
    : type(type), valueKind(valueKind), name(name), isList(isList) {}

fegen::FegenType *fegen::FegenValue::getType() { return this->type; }
llvm::StringRef fegen::FegenValue::getName() { return this->name; }

fegen::ValueKind fegen::FegenValue::getValueKind() { return this->valueKind; }

fegen::FegenValue *fegen::FegenValue::getBindingValue() {
  return this->bindingValue;
}
void fegen::FegenValue::setBindingValue(fegen::FegenValue *value) {
  this->bindingValue = value;
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

fegen::FegenValue *fegen::ValueMap::createValue(std::string name,
                                                FegenType *type,
                                                ValueKind valueKind,
                                                bool isList) {
  return new fegen::FegenValue(type, valueKind, name, isList);
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
