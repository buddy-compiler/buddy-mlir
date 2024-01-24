#ifndef FEGEN_VALUE_H
#define FEGEN_VALUE_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <string>

namespace fegen {
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

private:
  FegenType *type;
  ValueKind valueKind;
  std::string name;
  bool isList;
  FegenValue *bindingValue = nullptr;
  FegenValue() = delete;
  FegenValue(FegenType *type, ValueKind valueKind, std::string name,
             bool isList);

public:
  FegenType *getType();

  llvm::StringRef getName();

  ValueKind getValueKind();
  FegenValue *getBindingValue();
  void setBindingValue(FegenValue *value);
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
  static FegenValue *createValue(std::string name, FegenType *type,
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