#ifndef FEGEN_MANAGER_H
#define FEGEN_MANAGER_H

#include <any>
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include "FegenParser.h"
#include "ParserRuleContext.h"

#define FEGEN_PLACEHOLDER "Placeholder"
#define FEGEN_TYPE "Type"
#define FEGEN_TYPETEMPLATE "TypeTemplate"
#define FEGEN_INTEGER "Integer"
#define FEGEN_FLOATPOINT "FloatPoint"
#define FEGEN_CHAR "Char"
#define FEGEN_STRING "String"
#define FEGEN_VECTOR "Vector"
#define FEGEN_TENSOR "Tensor"
#define FEGEN_LIST "List"
#define FEGEN_OPTINAL "Optional"
#define FEGEN_ANY "Any"
#define FEGEN_DIALECT_NAME "fegen_builtin"
#define FEGEN_NOT_IMPLEMENTED_ERROR false


namespace fegen {
class Type;
class Manager;
class Value;
class RightValue;
class Expression;

using TypePtr = std::shared_ptr<Type>;
using largestInt = long long int;
// binary operation

enum class FegenOperator {
  OR,
  AND,
  EQUAL,
  NOT_EQUAL,
  LESS,
  LESS_EQUAL,
  GREATER,
  GREATER_EQUAL,
  ADD,
  SUB,
  MUL,
  DIV,
  MOD,
  POWER,
  NEG,
  NOT
};

// user defined function
class Function {
private:
  // cpp function name
  std::string name;
  // input object
  std::vector<Value *> inputTypeList;
  // return type
  TypePtr returnType;
  explicit Function(std::string name,
                         std::vector<Value *> &&inputTypeList,
                         TypePtr returnType);

public:
  static Function *get(std::string name,
                            std::vector<Value *> inputTypeList,
                            TypePtr returnType = nullptr);
  ~Function() = default;
  std::string getName();
  std::vector<Value *> &getInputTypeList();
  Value *getInputTypeList(size_t i);
  TypePtr getReturnType();
};

class Value;

// user defined operation
class Operation {
private:
  std::string dialectName;
  std::string operationName;
  // arguments of operation
  std::vector<Value *> arguments;
  // results of operation
  std::vector<Value *> results;
  // operation body context
  FegenParser::BodySpecContext *ctx;
  explicit Operation(std::string dialectName, std::string operationName,
                          std::vector<Value *> &&arguments,
                          std::vector<Value *> &&results,
                          FegenParser::BodySpecContext *ctx);

public:
  void setOpName(std::string);
  std::string getOpName();
  std::vector<Value *> &getArguments();
  Value *getArguments(size_t i);
  std::vector<Value *> &getResults();
  Value *getResults(size_t i);
  static Operation *get(std::string operationName,
                             std::vector<Value *> arguments,
                             std::vector<Value *> results,
                             FegenParser::BodySpecContext *ctx);
  ~Operation() = default;
};

class TypeDefination;
class RightValue;
class Type {
  friend class Value;

public:
  enum class TypeKind { ATTRIBUTE, OPERAND, CPP };

private:
  TypeKind kind;
  std::string typeName;
  // std::vector<Value *> parameters;
  TypeDefination *typeDefine;
  int typeLevel;
  bool isConstType;

public:
  Type(TypeKind kind, std::string name, TypeDefination *tyDef, int typeLevel, bool isConstType);

  Type(const Type &) = default;
  Type(Type &&) = default;
  TypeKind getTypeKind();
  void setTypeKind(TypeKind kind);
  TypeDefination *getTypeDefination();
  void setTypeDefination(TypeDefination *tyDef);
  std::string getTypeName();
  int getTypeLevel();
  bool isConstant();
  // for generating typedef td file.
  virtual std::string toStringForTypedef();
  // for generating op def td file.
  virtual std::string toStringForOpdef();
  // for generating cpp type kind.
  virtual std::string toStringForCppKind();
  static bool isSameType(Type *type1, Type *type2);
  virtual ~Type() = default;

  // placeholder
  static TypePtr getPlaceHolder();

  // Type
  static TypePtr getMetaType();

  // TypeTemplate
  static TypePtr getMetaTemplateType();

  // int
  static TypePtr getInt32Type();

  // float
  static TypePtr getFloatType();

  // float
  static TypePtr getDoubleType();

  // bool
  static TypePtr getBoolType();

  // Integer<size>
  static TypePtr getIntegerType(RightValue size);

  // FloatPoint<size>
  static TypePtr getFloatPointType(RightValue size);

  // string
  static TypePtr getStringType();

  // List<elementType>
  static TypePtr getListType(TypePtr elementType);
  static TypePtr getListType(RightValue elementType);

  // Vector<size, elementType>
  static TypePtr getVectorType(TypePtr elementType, RightValue size);
  static TypePtr getVectorType(RightValue elementType, RightValue size);

  // Tensor<shape, elementType>
  static TypePtr getTensorType(TypePtr elementType, RightValue shape);
  static TypePtr getTensorType(RightValue elementType, RightValue shape);

  // Optional<elementType>
  static TypePtr getOptionalType(TypePtr elementType);
  static TypePtr getOptionalType(RightValue elementType);

  // Any<[elementType1, elementType2, ...]>
  static TypePtr getAnyType(RightValue elementTypes);

  static TypePtr getCustomeType(std::vector<RightValue> params, TypeDefination* tydef);

  static TypePtr getTemplateType(TypeDefination *typeDefination);
};

class TypeDefination {
  friend class Manager;

private:
  std::string dialectName;
  std::string name;
  std::vector<fegen::Value *> parameters;
  FegenParser::TypeDefinationDeclContext *ctx;
  bool ifCustome;
  std::string mnemonic;

public:
  TypeDefination(std::string dialectName, std::string name,
                      std::vector<fegen::Value *> parameters,
                      FegenParser::TypeDefinationDeclContext *ctx,
                      bool ifCustome);
  static TypeDefination *get(std::string dialectName, std::string name,
                                  std::vector<fegen::Value *> parameters,
                                  FegenParser::TypeDefinationDeclContext *ctx,
                                  bool ifCustome = true);
  std::string getDialectName();
  void setDialectName(std::string);
  std::string getName();
  std::string getMnemonic();
  void setName(std::string);
  const std::vector<fegen::Value *> &getParameters();
  FegenParser::TypeDefinationDeclContext *getCtx();
  void setCtx(FegenParser::TypeDefinationDeclContext *);
  bool isCustome();
};

/// @brief Represent right value, and pass by value.
class RightValue {
  friend class Type;
  friend class Value;
  
public:
  enum class LiteralKind {
    MONOSTATE,
    INT,
    FLOAT,
    STRING,
    TYPE,
    VECTOR,
    LEFT_VAR,
    FUNC_CALL,
    OPERATION_CALL,
    OPERATOR_CALL
  };
  struct ExpressionNode;
  struct FunctionCall;
  struct OperationCall;
  struct OperatorCall;
  struct ExpressionTerminal;
  struct Expression {
    bool ifTerminal;
    LiteralKind kind;
    bool isLiteral;
    bool ifConstexpr;
    Expression(bool, LiteralKind, bool);
    virtual ~Expression() = default;
    virtual bool isTerminal();
    virtual std::string toString() = 0;
    virtual std::string toStringForTypedef() = 0;
    virtual std::string toStringForOpdef() = 0;
    virtual std::string toStringForCppKind() = 0;
    LiteralKind getKind();
    virtual TypePtr getType() = 0;
    virtual std::any getContent() = 0;
    virtual bool isConstexpr();

    /// @brief operate lhs and rhs using binary operator.
    static std::shared_ptr<OperatorCall>
    binaryOperation(std::shared_ptr<Expression> lhs,
                    std::shared_ptr<Expression> rhs, FegenOperator op);
    /// @brief operate expr using unary operator
    static std::shared_ptr<OperatorCall>
        unaryOperation(std::shared_ptr<Expression>, FegenOperator);

    // TODO: callFunction
    static std::shared_ptr<FunctionCall>
    callFunction(std::vector<std::shared_ptr<Expression>>, Function *);

    // TODO: callOperation
    static std::shared_ptr<OperationCall>
    callOperation(std::vector<std::shared_ptr<Expression>>, Operation *);

    static std::shared_ptr<ExpressionTerminal> getPlaceHolder();
    static std::shared_ptr<ExpressionTerminal> getInteger(largestInt,
                                                          size_t size = 32);
    static std::shared_ptr<ExpressionTerminal> getFloatPoint(long double,
                                                             size_t size = 32);
    static std::shared_ptr<ExpressionTerminal> getString(std::string);
    static std::shared_ptr<ExpressionTerminal> getTypeRightValue(TypePtr);
    static std::shared_ptr<ExpressionTerminal>
    getList(std::vector<std::shared_ptr<Expression>> &);
    static std::shared_ptr<ExpressionTerminal>
    getLeftValue(fegen::Value *);
  };

  struct ExpressionNode : public Expression {
    ExpressionNode(LiteralKind, bool);
    virtual std::string toString() override;
    virtual std::string toStringForTypedef() override;
    virtual std::string toStringForOpdef() override;
    virtual std::string toStringForCppKind() override;
    virtual std::any getContent() override = 0;
    virtual TypePtr getType() override;
  };

  struct FunctionCall : public ExpressionNode {
    Function *func;
    std::vector<std::shared_ptr<Expression>> params;
    FunctionCall(Function *, std::vector<std::shared_ptr<Expression>>);
    virtual std::string toString() override;
    virtual std::string toStringForTypedef() override;
    virtual std::string toStringForOpdef() override;
    virtual std::string toStringForCppKind() override;
    virtual std::any getContent() override;
    virtual TypePtr getType() override;
  };

  struct OperationCall : public ExpressionNode {
    Operation *op;
    std::vector<std::shared_ptr<Expression>> params;
    OperationCall(Operation *, std::vector<std::shared_ptr<Expression>>);
    virtual std::string toString() override;
    virtual std::string toStringForTypedef() override;
    virtual std::string toStringForOpdef() override;
    virtual std::string toStringForCppKind() override;
    virtual std::any getContent() override;
    virtual TypePtr getType() override;
  };

  struct OperatorCall : public ExpressionNode {
    FegenOperator op;
    std::vector<std::shared_ptr<Expression>> params;
    OperatorCall(FegenOperator, std::vector<std::shared_ptr<Expression>>);
    virtual std::string toString() override;
    virtual std::string toStringForTypedef() override;
    virtual std::string toStringForOpdef() override;
    virtual std::string toStringForCppKind() override;
    virtual std::any getContent() override;
    virtual TypePtr getType() override;
  };

  struct ExpressionTerminal : public Expression {
    ExpressionTerminal(LiteralKind, bool);
    virtual std::string toString() override;
    virtual std::string toStringForTypedef() override;
    virtual std::string toStringForOpdef() override;
    virtual std::string toStringForCppKind() override;
    virtual std::any getContent() override = 0;
    virtual TypePtr getType() override;
  };

  struct PlaceHolder : public ExpressionTerminal {
    PlaceHolder();
    virtual std::any getContent() override;
    virtual std::string toString() override;
  };

  struct IntegerLiteral : public ExpressionTerminal {
    size_t size;
    largestInt content;
    IntegerLiteral(largestInt content, size_t size);
    virtual std::any getContent() override;
    virtual std::string toString() override;
    virtual TypePtr getType() override;
  };

  struct FloatPointLiteral : public ExpressionTerminal {
    size_t size;
    long double content;
    FloatPointLiteral(long double content, size_t size);
    virtual std::any getContent() override;
    virtual std::string toString() override;
    virtual TypePtr getType() override;
  };

  struct StringLiteral : public ExpressionTerminal {
    std::string content;
    StringLiteral(std::string content);
    virtual std::any getContent() override;
    virtual std::string toString() override;
    virtual TypePtr getType() override;
  };

  struct TypeLiteral : public ExpressionTerminal {
    TypePtr content;
    TypeLiteral(TypePtr content);
    virtual std::any getContent() override;
    virtual std::string toString() override;
    virtual std::string toStringForTypedef() override;
    virtual std::string toStringForOpdef() override;
    virtual std::string toStringForCppKind() override;
    virtual TypePtr getType() override;
  };

  struct ListLiteral : public ExpressionTerminal {
    std::vector<std::shared_ptr<Expression>> content;
    ListLiteral(std::vector<std::shared_ptr<Expression>> &content);
    virtual std::any getContent() override;
    virtual std::string toString() override;
    virtual std::string toStringForTypedef() override;
    virtual std::string toStringForOpdef() override;
    virtual TypePtr getType() override;
  };

  struct LeftValue : public ExpressionTerminal {
    Value *content;
    LeftValue(Value *content);
    virtual std::any getContent() override;
    virtual std::string toString() override;
    virtual TypePtr getType() override;
  };

public:
  using ExprPtr = std::shared_ptr<Expression>;
  RightValue(std::shared_ptr<Expression>);
  RightValue(const RightValue &) = default;
  RightValue(RightValue &&) = default;
  RightValue &operator=(const RightValue &another) = default;
  RightValue::LiteralKind getLiteralKind();
  std::string toString();
  std::string toStringForTypedef();
  std::string toStringForOpdef();
  std::string toStringForCppKind();
  std::any getContent();
  TypePtr getType();
  std::shared_ptr<Expression> getExpr();
  bool isConstant();

  static RightValue getPlaceHolder();
  static RightValue getInteger(largestInt content, size_t size = 32);
  static RightValue getFloatPoint(long double content, size_t size = 32);
  static RightValue getString(std::string content);
  static RightValue getTypeRightValue(TypePtr content);
  static RightValue
  getList(std::vector<std::shared_ptr<Expression>> &content);
  static RightValue getLeftValue(fegen::Value *content);
  static RightValue getByExpr(std::shared_ptr<Expression> expr);
  ~RightValue() = default;

private:
  std::shared_ptr<Expression> content;
};

// PlaceHolder
class PlaceHolderType : public Type {
  public:
  PlaceHolderType();
};

// Type
class MetaType : public Type {
  public:
  MetaType();
  // for generating typedef td file.
  virtual std::string toStringForTypedef() override;

};
// Template
class MetaTemplate : public Type {
  public:
  MetaTemplate();
};
// Integer<size>
class IntegerType : public Type {
  RightValue size;
  public:
  IntegerType(RightValue size, TypeDefination* tyDef);
  IntegerType(RightValue size);
  // for generating typedef td file.
  virtual std::string toStringForTypedef() override;
  // for generating op def td file.
  virtual std::string toStringForOpdef() override;
  // for generating cpp type kind.
  virtual std::string toStringForCppKind() override;
};
// FloatPoint<size>
class FloatPointType : public Type {
  RightValue size;
  public:
  FloatPointType(RightValue size);
  // for generating typedef td file.
  virtual std::string toStringForTypedef() override;
  // for generating op def td file.
  virtual std::string toStringForOpdef() override;
  // for generating cpp type kind.
  virtual std::string toStringForCppKind() override;
};
// String
class StringType : public Type {
  public:
  StringType();
};
// List<ty>
class ListType : public Type {
  RightValue elementType;
  public:
  ListType(RightValue elementType);
  // for generating typedef td file.
  virtual std::string toStringForTypedef() override;
  // for generating op def td file.
  virtual std::string toStringForOpdef() override;
  // for generating cpp type kind.
  virtual std::string toStringForCppKind() override;
};
// Vector<ty, size>
class VectorType : public Type {
  RightValue elementType;
  RightValue size;
  public:
  VectorType(RightValue elementType, RightValue size);
};
// Tensor<ty, shape>
class TensorType : public Type {
  RightValue elementType;
  RightValue shape;
  public:
  TensorType(RightValue elementType, RightValue shape);
};
// Optional<ty>
class OptionalType : public Type {
  RightValue elementType;
  public:
  OptionalType(RightValue elementType);
};
// Any<[ty1, ty2, ...]>
class AnyType : public Type {
  RightValue elementTypes;
  public:
  AnyType(RightValue elementTypes);
};
// custome type
class CustomeType : public Type {
  std::vector<RightValue> params;
  public:
  CustomeType(std::vector<RightValue> params, TypeDefination* tydef);
};

class TemplateType : public Type {
  public:
  TemplateType(TypeDefination* tydef);
  TypePtr instantiate(std::vector<RightValue> params);
  // for generating typedef td file.
  virtual std::string toStringForTypedef() override;
  // for generating op def td file.
  virtual std::string toStringForOpdef() override;
};


class Value {
  friend class Type;

private:
  TypePtr type;
  std::string name;
  RightValue content;

public:
  Value(TypePtr type, std::string name, RightValue content);
  Value(const Value &rhs);
  Value(Value &&rhs);

  static Value *get(TypePtr type, std::string name,
                         RightValue constant);

  std::string getName();
  TypePtr getType();
  /// @brief return content of right value, get ExprssionNode* if kind is
  /// EXPRESSION.
  template <typename T> T getContent() {
    return std::any_cast<T>(this->content.getContent());
  }
  void setContent(fegen::RightValue content);
  RightValue::LiteralKind getContentKind();
  std::string getContentString();
  std::string getContentStringForTypedef();
  std::string getContentStringForOpdef();
  std::string getContentStringForCppKind();
  std::shared_ptr<RightValue::Expression> getExpr();
  ~Value() = default;
};

class ParserNode;

class ParserRule {
  friend class Manager;

private:
  std::string content;
  // from which node
  ParserNode *src;
  std::map<llvm::StringRef, Value *> inputs;
  std::map<llvm::StringRef, Value *> returns;
  // context in parser tree
  antlr4::ParserRuleContext *ctx;
  explicit ParserRule(std::string content, ParserNode *src,
                     antlr4::ParserRuleContext *ctx);

public:
  static ParserRule *get(std::string content, ParserNode *src,
                        antlr4::ParserRuleContext *ctx);
  llvm::StringRef getContent();
  // check and add input value
  bool addInput(Value input);
  // check and add return value
  bool addReturn(Value output);
  // set source node
  void setSrc(ParserNode *src);
};

class ParserNode {
  friend class Manager;

public:
  enum class NodeType { PARSER_RULE, LEXER_RULE };

private:
  std::vector<ParserRule *> rules;
  antlr4::ParserRuleContext *ctx;
  NodeType ntype;
  explicit ParserNode(std::vector<ParserRule *> &&rules,
                     antlr4::ParserRuleContext *ctx, NodeType ntype);

public:
  static ParserNode *get(std::vector<ParserRule *> rules,
                        antlr4::ParserRuleContext *ctx, NodeType ntype);
  static ParserNode *get(antlr4::ParserRuleContext *ctx, NodeType ntype);
  void addFegenRule(ParserRule *rule);
  // release rules first
  ~ParserNode();
};

class FegenVisitor;

class Manager {
  friend class FegenVisitor;

private:
  Manager();
  Manager(const Manager &) = delete;
  const Manager &operator=(const Manager &) = delete;
  // release nodes, type, operation, function
  ~Manager();
  void initbuiltinTypes();

public:
  std::string moduleName;
  std::vector<std::string> headFiles;
  std::map<std::string, ParserNode *> nodeMap;
  llvm::StringMap<Type *> typeMap;
  std::map<std::string, TypeDefination *> typeDefMap;
  std::map<std::string, Operation *> operationMap;
  std::map<std::string, Function *> functionMap;
  // stmt contents
  std::unordered_map<antlr4::ParserRuleContext *, std::any> stmtContentMap;
  void addStmtContent(antlr4::ParserRuleContext *ctx, std::any content);
  template <typename T> T getStmtContent(antlr4::ParserRuleContext *ctx) {
    assert(this->stmtContentMap.count(ctx));
    return std::any_cast<T>(this->stmtContentMap[ctx]);
  }

  static Manager &getManager();
  void setModuleName(std::string name);

  TypeDefination *getTypeDefination(std::string name);
  bool addTypeDefination(TypeDefination *tyDef);

  Operation *getOperationDefination(std::string name);
  bool addOperationDefination(Operation *opDef);
  void emitG4();
  void emitTypeDefination();
  void emitOpDefination();
  void emitDialectDefination();
  void emitTdFiles();
  void emitBuiltinFunction();
};

TypePtr
    inferenceType(std::vector<std::shared_ptr<RightValue::Expression>>,
                  FegenOperator);

} // namespace fegen

#endif