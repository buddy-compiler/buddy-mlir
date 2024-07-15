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

#define FEGEN_NOT_IMPLEMENTED_ERROR false

namespace fegen {

class Type;
class Manager;
class Value;

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
  Type *returnType;
  explicit Function(std::string name,
                         std::vector<Value *> &&inputTypeList,
                         Type *returnType);

public:
  static Function *get(std::string name,
                            std::vector<Value *> inputTypeList,
                            Type *returnType = nullptr);
  ~Function() = default;
  std::string getName();
  std::vector<Value *> &getInputTypeList();
  Value *getInputTypeList(size_t i);
  Type *getReturnType();
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

class Type {
  friend class Value;

public:
  enum class TypeKind { ATTRIBUTE, OPERAND, CPP };

private:
  TypeKind kind;
  std::string typeName;
  std::vector<Value *> parameters;
  TypeDefination *typeDefine;
  int typeLevel;

public:
  Type(TypeKind kind, std::string name,
            std::vector<Value *> parameters, TypeDefination *tyDef,
            int typeLevel);
  Type(TypeKind kind, std::vector<Value *> parameters,
            TypeDefination *tyDef, int typeLevel);
  Type(const Type &);
  Type(Type &&);
  TypeKind getTypeKind();
  void setTypeKind(TypeKind kind);
  std::vector<Value *> &getParameters();
  Value *getParameters(size_t i);
  void setParameters(std::vector<Value *> &params);
  TypeDefination *getTypeDefination();
  void setTypeDefination(TypeDefination *tyDef);
  std::string getTypeName();
  int getTypeLevel();
  // for generating typedef td file.
  std::string toStringForTypedef();
  // for generating op def td file.
  std::string toStringForOpdef();
  // for generating cpp type kind.
  std::string toStringForCppKind();
  static bool isSameType(Type *type1, Type *type2);
  ~Type();
  // placeholder
  static Type getPlaceHolder();
  // Type
  static Type getMetaType();

  // TypeTemplate
  static Type getMetaTemplateType();

  // int
  static Type getInt32Type();

  // float
  static Type getFloatType();

  // float
  static Type getDoubleType();

  // bool
  static Type getBoolType();

  // Integer<size>
  static Type getIntegerType(Value *size);

  // FloatPoint<size>
  static Type getFloatPointType(Value *size);

  // char
  static Type getCharType();

  // string
  static Type getStringType();

  // Vector<size, elementType>
  static Type getVectorType(Value *size, Type elementType);

  // Tensor<shape, elementType>
  static Type getTensorType(Value *shape, Type elementType);

  // List<elementType>
  static Type getListType(Type elementType);

  // Optional<elementType>
  static Type getOptionalType(Type elementType);

  // Any<elementType1, elementType2, ...>
  static Type getAnyType(std::vector<Type> elementTypes);

  static Type getIntegerTemplate();
  static Type getFloatPointTemplate();

  static Type getInstanceType(TypeDefination *typeDefination,
                                   std::vector<Value *> parameters);

  static Type getTemplateType(TypeDefination *typeDefination);
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
    Type exprType;
    bool isLiteral;
    bool ifConstexpr;
    Expression(bool, LiteralKind, Type &, bool);
    virtual ~Expression() = default;
    virtual bool isTerminal();
    virtual std::string toString() = 0;
    virtual std::string toStringForTypedef() = 0;
    virtual std::string toStringForOpdef() = 0;
    virtual std::string toStringForCppKind() = 0;
    LiteralKind getKind();
    Type &getType();
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
    static std::shared_ptr<ExpressionTerminal> getInteger(long long int,
                                                          size_t size = 32);
    static std::shared_ptr<ExpressionTerminal> getFloatPoint(long double,
                                                             size_t size = 32);
    static std::shared_ptr<ExpressionTerminal> getString(std::string);
    static std::shared_ptr<ExpressionTerminal> getType(Type &);
    static std::shared_ptr<ExpressionTerminal>
    getList(std::vector<std::shared_ptr<Expression>> &);
    static std::shared_ptr<ExpressionTerminal>
    getLeftValue(fegen::Value *);
  };

  struct ExpressionNode : public Expression {
    ExpressionNode(LiteralKind, Type, bool);
    virtual std::string toString() override;
    virtual std::string toStringForTypedef() override;
    virtual std::string toStringForOpdef() override;
    virtual std::string toStringForCppKind() override;
    virtual std::any getContent() override = 0;
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
  };

  struct ExpressionTerminal : public Expression {
    ExpressionTerminal(LiteralKind, Type, bool);
    virtual std::string toString() override;
    virtual std::string toStringForTypedef() override;
    virtual std::string toStringForOpdef() override;
    virtual std::string toStringForCppKind() override;
    virtual std::any getContent() override = 0;
  };

  struct PlaceHolder : public ExpressionTerminal {
    PlaceHolder();
    virtual std::any getContent() override;
    virtual std::string toString() override;
  };

  struct IntegerLiteral : public ExpressionTerminal {
    size_t size;
    long long int content;
    // size = 32
    IntegerLiteral(int content);
    IntegerLiteral(long long int content, size_t size);
    virtual std::any getContent() override;
    virtual std::string toString() override;
  };

  struct FloatPointLiteral : public ExpressionTerminal {
    size_t size;
    long double content;
    FloatPointLiteral(long double content, size_t size);
    virtual std::any getContent() override;
    virtual std::string toString() override;
  };

  struct StringLiteral : public ExpressionTerminal {
    std::string content;
    StringLiteral(std::string content);
    virtual std::any getContent() override;
    virtual std::string toString() override;
  };

  struct TypeLiteral : public ExpressionTerminal {
    Type content;
    TypeLiteral(Type &content);
    virtual std::any getContent() override;
    virtual std::string toString() override;
    virtual std::string toStringForTypedef() override;
    virtual std::string toStringForOpdef() override;
    virtual std::string toStringForCppKind() override;
  };

  struct ListLiteral : public ExpressionTerminal {
    std::vector<std::shared_ptr<Expression>> content;
    ListLiteral(std::vector<std::shared_ptr<Expression>> &content);
    virtual std::any getContent() override;
    virtual std::string toString() override;
    virtual std::string toStringForTypedef() override;
    virtual std::string toStringForOpdef() override;
  };

  struct LeftValue : public ExpressionTerminal {
    Value *content;
    LeftValue(Value *content);
    virtual std::any getContent() override;
    virtual std::string toString() override;
  };

public:
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
  Type &getType();
  std::shared_ptr<Expression> getExpr();

  static RightValue getPlaceHolder();
  static RightValue getInteger(long long int content, size_t size = 32);
  static RightValue getFloatPoint(long double content, size_t size = 32);
  static RightValue getString(std::string content);
  static RightValue getType(Type &content);
  static RightValue
  getList(std::vector<std::shared_ptr<Expression>> &content);
  static RightValue getLeftValue(fegen::Value *content);
  static RightValue getByExpr(std::shared_ptr<Expression> expr);
  ~RightValue() = default;

private:
  std::shared_ptr<Expression> content;
};

class Value {
  friend class Type;

private:
  Type type;
  std::string name;
  RightValue content;

public:
  Value(Type type, std::string name, RightValue content);
  Value(const Value &rhs);
  Value(Value &&rhs);

  static Value *get(Type type, std::string name,
                         RightValue constant);

  std::string getName();
  Type &getType();
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

Type
    inferenceType(std::vector<std::shared_ptr<RightValue::Expression>>,
                  FegenOperator);

} // namespace fegen

#endif