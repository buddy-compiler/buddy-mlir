#ifndef FEGEN_MANAGER_H
#define FEGEN_MANAGER_H

#include "FegenParser.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <map>
#include <string>
#include <vector>

namespace fegen {

class FegenType;

// user defined function
class FegenFunction {
private:
  // cpp function name
  std::string name;
  // input types
  std::vector<FegenType *> inputTypeList;
  // return type
  FegenType *returnType;
  explicit FegenFunction(llvm::StringRef name,
                         std::vector<FegenType *> &&inputTypeList,
                         FegenType *returnType);

public:
  static FegenFunction *get(llvm::StringRef name,
                            std::vector<FegenType *> inputTypeList,
                            FegenType *returnType = nullptr);
  ~FegenFunction() = default;
};

class FegenValue;

// user defined operation
class FegenOperation {
private:
  std::string dialectName;
  std::string operationName;
  // arguments of operation
  std::vector<FegenValue *> arguments;
  // results of operation
  std::vector<FegenValue *> results;
  explicit FegenOperation(llvm::StringRef dialectName,
                          llvm::StringRef operationName,
                          std::vector<FegenValue *> &&arguments,
                          std::vector<FegenValue *> &&results);

public:
  static FegenOperation *get(llvm::StringRef dialectName,
                             llvm::StringRef operationName,
                             std::vector<FegenValue *> arguments,
                             std::vector<FegenValue *> results);
  ~FegenOperation() = default;
};

class FegenType {
public:
  enum class TypeKind { ATTRIBUTE_VALUE, OPERAND_VALUE, CPP_VALUE };

private:
  TypeKind kind;
  std::string dialectName;
  std::string typeName;
  std::string assemblyFormat;
  std::vector<FegenValue *> parameters;
  // context of type in parser tree
  FegenParser::TypeDefinationDeclContext *ctx;
  explicit FegenType(TypeKind kind, llvm::StringRef dialectName,
                     llvm::StringRef typeName, llvm::StringRef assemblyFormat,
                     std::vector<FegenValue *> &&parameters,
                     FegenParser::TypeDefinationDeclContext *ctx);

public:
  // convert from fegen type to cpp type
  std::string convertToCppType();
  static FegenType *get(TypeKind kind, llvm::StringRef dialectName,
                        llvm::StringRef typeName,
                        llvm::StringRef assemblyFormat,
                        std::vector<FegenValue *> parameters,
                        FegenParser::TypeDefinationDeclContext *ctx);
  ~FegenType() = default;
};

class FegenValue {

private:
  bool isList;
  FegenType *type;
  std::string name;
  antlr4::ParserRuleContext *ctx;
  explicit FegenValue(bool isList, FegenType *type, llvm::StringRef name,
                      antlr4::ParserRuleContext *ctx);

public:
  static FegenValue *get(bool isList, FegenType *type, llvm::StringRef name,
                         antlr4::ParserRuleContext *ctx);
  llvm::StringRef getName();
  ~FegenValue() = default;
};

class FegenNode;

class FegenRule {
private:
  std::string content;
  // from which node
  FegenNode *src;
  std::map<llvm::StringRef, FegenValue *> inputs;
  std::map<llvm::StringRef, FegenValue *> returns;
  // context in parser tree
  FegenParser::AlternativeContext *ctx;
  explicit FegenRule(std::string content, FegenNode *src,
                     FegenParser::AlternativeContext *ctx);

public:
  static FegenRule *get(std::string content, FegenNode *src,
                        FegenParser::AlternativeContext *ctx);
  // check and add input value
  bool addInput(FegenValue *input);
  // check and add return value
  bool addReturn(FegenValue *output);
  // set source node
  void setSrc(FegenNode *src);
};

class FegenNode {
public:
  enum class NodeType { PARSER_RULE, LEXER_RULE };

private:
  NodeType ntype;
  std::vector<FegenRule *> rules;
  antlr4::ParserRuleContext *ctx;
  explicit FegenNode(std::vector<FegenRule *> &&rules,
                     antlr4::ParserRuleContext *ctx, NodeType ntype);

public:
  static FegenNode *get(std::vector<FegenRule *> rules,
                        antlr4::ParserRuleContext *ctx, NodeType ntype);
  static FegenNode *get(antlr4::ParserRuleContext *ctx, NodeType ntype);
  void addFegenRule(FegenRule *rule);
  // release rules first
  ~FegenNode();
};

class FegenVisitor;

class FegenManager {
  friend class FegenVisitor;

private:
  std::vector<std::string> headFiles;
  std::map<llvm::StringRef, FegenNode *> nodeMap;
  llvm::StringMap<FegenType *> typeMap;
  llvm::StringMap<FegenOperation *> operationMap;
  llvm::StringMap<FegenFunction *> functionMap;

public:
  // release nodes, type, operation, function
  ~FegenManager();
};

} // namespace fegen

#endif