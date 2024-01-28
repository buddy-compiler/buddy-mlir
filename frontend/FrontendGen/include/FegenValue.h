#ifndef FEGEN_VALUE_H
#define FEGEN_VALUE_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <string>
#include <variant>

namespace fegen {
class FegenRule;
class ValueMap;

class FegenType {
  friend class ValueMap;

private:
  std::string name;
  FegenType() = delete;
  FegenType(std::string name);

public:
  bool operator==(FegenType *another);
  llvm::StringRef getName();
};

enum class ValueKind { OPERAND, ATTRIBUTE, CPP };

class FegenValue {
  friend class ValueMap;
public:
class LiteralValue {
public:
  LiteralValue(std::variant<int, float, std::string> value);
  std::variant<int, float, std::string> value;
};

class RuleInOutputValue {
public:
  enum class SourceSectionType{
    INPUTS, RETURNS
  };
  RuleInOutputValue(SourceSectionType ty, FegenValue* value);
  SourceSectionType srcSectype;
  FegenValue* value;
  FegenRule* sourceRule;
};

class RuleAttributeValue {
public:
  enum class AttributeKind{
    TEXT
  };
  RuleAttributeValue(AttributeKind kind, FegenRule* src);
  AttributeKind attrType;
  FegenRule* sourceRule;
};

private:
  // value source
  FegenRule* source;
  // type of value 
  FegenType *type;
  // value type, attribute/operand/cpp value
  ValueKind valueKind;
  // name of value
  std::string name;
  bool isList;
  // bind info 
  std::variant<std::monostate, LiteralValue, RuleInOutputValue, RuleAttributeValue> bindingInfo;
  // index of source rule
  // -1: only one rule, and this index do not act
  // other: use specfical rule
  // ex: grammar defination is 'rule1*' and input x is return of the second 'rule1', so x = $rule1(1).ret
  // and for x, 'source' is rule1, and 'ruleIndex' = 1  
  int ruleIndex = -1;

  FegenValue() = delete;
  FegenValue(FegenRule* source, FegenType *type, ValueKind valueKind, std::string name,
             bool isList);

public:
  FegenRule* getSource();
  FegenType *getType();

  llvm::StringRef getName();

  ValueKind getValueKind();
  std::variant<std::monostate, LiteralValue, RuleInOutputValue, RuleAttributeValue>& getBindingValue();
  void setBindingValue(std::variant<std::monostate, LiteralValue, RuleInOutputValue, RuleAttributeValue> value);
  bool ifList();
  void setRuleIndex(int index);
  int getRuleIndex();
};

class ValueMap {
private:
  ValueMap() {}
  ValueMap(const ValueMap &) = delete;
  ValueMap(const ValueMap &&) = delete;
  ValueMap &operator=(const ValueMap &) = delete;

  llvm::StringMap<FegenType *> typeMap;
  llvm::StringMap<FegenValue *> valueMap;

public:
  ~ValueMap();

  static ValueMap &getMap();
  static FegenType *createType(std::string name);
  static FegenValue *createValue(FegenRule* source, std::string name, FegenType *type,
                                 ValueKind valueKind, bool isList = false);
  // return nullptr if not found
  FegenType *findType(std::string name);
  // return true if type existed
  bool insertType(FegenType *type);
  //  return nullptr if not found
  FegenValue *findValue(std::string name);
  // return true if value existed
  bool insertValue(FegenValue *value);
};
} // namespace fegen

#endif