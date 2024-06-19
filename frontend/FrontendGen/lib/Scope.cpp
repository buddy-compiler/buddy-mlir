#include "Scope.h"

// SymbolTable
template <typename T> void fegen::SymbolTable<T>::add(std::string name, T *e) {
  this->table.insert({name, e});
}

template <typename T> T *fegen::SymbolTable<T>::get(std::string name) {
  return this->table[name];
}

template <typename T> bool fegen::SymbolTable<T>::exist(std::string name) {
  return (this->table.count(name) > 0);
}

template <typename T> fegen::SymbolTable<T>::~SymbolTable() {
    for(auto pair : this->table){
        delete pair.second;
    }
}

// FegenScope
fegen::FegenScope::FegenScope(
                              unsigned int scopeId,
                              fegen::FegenScope *parentScope)
    :scopeId(scopeId), parentScope(parentScope) {}


fegen::FegenTypeDefination* fegen::FegenScope::findTypeDef(std::string name) {
    return this->typeTable.get(name);
}

void fegen::FegenScope::addTypeDef(FegenTypeDefination* tyDef) {
    this->typeTable.add(tyDef->getName(), tyDef);
}

bool fegen::FegenScope::isExistTypeDef(std::string name) {
    return this->typeTable.exist(name);
}

fegen::FegenValue* fegen::FegenScope::findVar(std::string name) {
    return this->varTable.get(name);
}

void fegen::FegenScope::addVar(fegen::FegenValue* var) {
    this->varTable.add(var->getName(), var);
}

bool fegen::FegenScope::isExistVar(std::string name) {
    return this->varTable.exist(name);
}

fegen::ScopeStack::ScopeStack() : count(1) {
    this->globalScope = new fegen::FegenScope(0, nullptr);
    this->currentScope = this->globalScope;
    this->scopeStack.push(this->globalScope);
    this->scopes.push_back(this->globalScope);
}

fegen::ScopeStack::~ScopeStack() {
    for(auto scope : this->scopes){
        delete scope;
    }
}

fegen::ScopeStack& fegen::ScopeStack::getScopeStack() {
    static fegen::ScopeStack sstack;
    return sstack;
}

void fegen::ScopeStack::pushScope() {
    auto newScope = new fegen::FegenScope(this->count++, this->currentScope);
    this->scopeStack.push(newScope);
    this->scopes.push_back(newScope);
    this->currentScope = newScope;
}

void fegen::ScopeStack::popScope() {
    this->scopeStack.pop();
    this->currentScope = this->scopeStack.top();
}
bool fegen::ScopeStack::attemptAddVar(fegen::FegenValue* var) {
    if(this->currentScope->isExistVar(var->getName())){
        return false;
    }
    this->currentScope->addVar(var);
    return true;
}

fegen::FegenValue* fegen::ScopeStack::attemptFindVar(std::string name) {
    auto p = this->currentScope;
    while(p != nullptr){
        if(p->isExistVar(name)){
            return p->findVar(name);
        }
        p = p->parentScope;
    }
    return nullptr;
}

bool fegen::ScopeStack::attemptAddTypeDef(fegen::FegenTypeDefination* tyDef) {
    if(this->currentScope->isExistTypeDef(tyDef->getName())){
        return false;
    }
    this->currentScope->addTypeDef(tyDef);
    return true;
}

fegen::FegenTypeDefination* fegen::ScopeStack::attemptFindTypeDef(std::string name) {
    auto p = this->currentScope;
    while(p != nullptr){
        if(p->isExistTypeDef(name)){
            return p->findTypeDef(name);
        }
        p = p->parentScope;
    }
    return nullptr;
}