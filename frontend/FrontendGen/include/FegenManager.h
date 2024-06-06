#ifndef FEGEN_MANAGER_H
#define FEGEN_MANAGER_H

#include <any>
#include <map>
#include <string>
#include <variant>
#include <vector>

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include "FegenParser.h"

namespace fegen {

class FegenType;
class FegenManager;

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
  friend class FegenValue;

public:
  enum class TypeKind { ATTRIBUTE, OPERAND, CPP };

private:
  TypeKind kind;
  std::string dialectName;
  std::string typeName;
  std::vector<FegenValue> parameters;

public:
  FegenType(TypeKind kind, std::string dialectName, std::string typeName,
            std::vector<FegenValue> parameters);
  FegenType(const FegenType &);
  FegenType(FegenType &&);

  // Type
  static FegenType getMetaType();

  // TypeTemplate
  static FegenType getMetaTemplateType();

  // int
  static FegenType getInt32Type();

  // float
  static FegenType getFloatType();

  // float
  static FegenType getDoubleType();

  // bool
  static FegenType getBoolType();

  // Integer<size>
  static FegenType getIntegerType(FegenValue size);

  // FloatPoint<size>
  static FegenType getFloatPointType(FegenValue size);

  // char
  static FegenType getCharType();

  // string
  static FegenType getStringType();

  // Vector<size, elementType>
  static FegenType getVectorType(FegenValue size, FegenValue elementType);
  static FegenType getVectorType(FegenValue size, FegenType elementType);

  // Tensor<shape, elementType>
  static FegenType getTensorType(FegenValue shape, FegenValue elementType);
  static FegenType getTensorType(FegenValue shape, FegenType elementType);

  // static FegenType get(TypeKind kind, std::string dialectName,
  //                       std::string typeName,
  //                       std::vector<FegenValue> parameters);

  ~FegenType() = default;
};

class FegenLiteral {
  friend class FegenType;
  friend class FegenValue;
  using literalType = std::variant<int, float, std::string, FegenType,
                                   std::vector<FegenLiteral>>;

private:
  literalType content;

public:
  enum class LiteralKind { INT, FLOAT, STRING, TYPE, VECTOR };
  FegenLiteral(literalType content);
  FegenLiteral(const FegenLiteral &);
  FegenLiteral(FegenLiteral &&);
  static FegenLiteral get(int content);
  static FegenLiteral get(float content);
  static FegenLiteral get(std::string content);
  static FegenLiteral get(FegenType content);

  /// @brief receive vector of number string, FegenType or vector and build it
  /// to FegenLiteral
  /// @tparam T element type, should be one of int, float, std::string,
  /// FegenType or std::vector
  template <typename T> static FegenLiteral get(std::vector<T> content);

  template <typename T> T getContent() { return std::get<T>(this->content); }

private:
  LiteralKind kind;
};

class FegenValue {
  friend class FegenType;

private:
  FegenType type;
  std::string name;
  FegenLiteral content;

public:
  FegenValue(FegenType type, std::string name, FegenLiteral content);
  FegenValue(const FegenValue &rhs);
  FegenValue(FegenValue &&rhs);

  static FegenValue *get(FegenType type, std::string name,
                         FegenLiteral constant);

  llvm::StringRef getName();

  template <typename T> T getContent() { return this->content.getContent<T>(); }

  ~FegenValue() = default;
};

class FegenNode;

class FegenRule {
  friend class FegenManager;

private:
  std::string content;
  // from which node
  FegenNode *src;
  std::map<llvm::StringRef, FegenValue *> inputs;
  std::map<llvm::StringRef, FegenValue *> returns;
  // context in parser tree
  antlr4::ParserRuleContext *ctx;
  explicit FegenRule(std::string content, FegenNode *src,
                     antlr4::ParserRuleContext *ctx);

public:
  static FegenRule *get(std::string content, FegenNode *src,
                        antlr4::ParserRuleContext *ctx);
  llvm::StringRef getContent();
  // check and add input value
  bool addInput(FegenValue input);
  // check and add return value
  bool addReturn(FegenValue output);
  // set source node
  void setSrc(FegenNode *src);
};

class FegenNode {
  friend class FegenManager;

public:
  enum class NodeType { PARSER_RULE, LEXER_RULE };

private:
  std::vector<FegenRule *> rules;
  antlr4::ParserRuleContext *ctx;
  NodeType ntype;
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
  std::string moduleName;
  std::vector<std::string> headFiles;
  std::map<std::string, FegenNode *> nodeMap;
  llvm::StringMap<FegenType *> typeMap;
  llvm::StringMap<FegenOperation *> operationMap;
  llvm::StringMap<FegenFunction *> functionMap;

public:
  void setModuleName(std::string name);
  std::string emitG4();
  // release nodes, type, operation, function
  ~FegenManager();
};

} // namespace fegen

#endif