#ifndef FEGEN_SCOPE_H
#define FEGEN_SCOPE_H

#include "FegenManager.h"
#include <stack>

namespace fegen {

template <typename T> class SymbolTable {
private:
  std::map<std::string, T *> table;

public:
  SymbolTable() = default;
  void add(std::string, T *e);
  T *get(std::string name);
  /// @brief return true if name exist in map.
  bool exist(std::string name);
  ~SymbolTable();
};

class FegenScope {
  using TypeDefTable = SymbolTable<TypeDefination>;
  using VariableTable = SymbolTable<Value>;
  friend class ScopeStack;

private:
  unsigned int scopeId;
  FegenScope *parentScope;
  TypeDefTable typeTable;
  VariableTable varTable;

public:
  explicit FegenScope(unsigned int scopeId, FegenScope *parentScope);
  ~FegenScope() = default;

  /// @brief this will not check.
  TypeDefination *findTypeDef(std::string name);
  /// @brief this will not check whether tyDef is already existed or not.
  void addTypeDef(TypeDefination *tyDef);
  /// @brief return true if exist.
  bool isExistTypeDef(std::string name);
  /// @brief this will not check.
  Value *findVar(std::string name);
  /// @brief this will not check whether var is already existed or not.
  void addVar(Value *var);
  /// @brief return true if exist.
  bool isExistVar(std::string name);
};

class ScopeStack {
private:
  std::vector<FegenScope *> scopes;
  std::stack<FegenScope *> scopeStack;

  FegenScope *currentScope;
  FegenScope *globalScope;
  // scope total count
  size_t count;

  ScopeStack();
  ~ScopeStack();
  ScopeStack(const ScopeStack &) = delete;
  const ScopeStack &operator=(const ScopeStack &) = delete;

public:
  static ScopeStack &getScopeStack();
  void pushScope();
  void popScope();
  /// @brief check and add var to current scope, return false if failed.
  bool attemptAddVar(Value *var);
  /// @brief check add find var from current scope, return nullptr if failed.
  Value *attemptFindVar(std::string name);
  /// @brief check and add tyDef to current scope, return false if failed.
  bool attemptAddTypeDef(TypeDefination *tyDef);
  /// @brief check and find tyDef from current scope, return nullptr if failed.
  TypeDefination *attemptFindTypeDef(std::string name);
};
} // namespace fegen

#endif