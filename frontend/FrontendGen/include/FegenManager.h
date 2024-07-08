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

class FegenType;
class FegenManager;
class FegenValue;

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
class FegenFunction {
private:
  // cpp function name
  std::string name;
  // input object
  std::vector<FegenValue *> inputTypeList;
  // return type
  FegenType *returnType;
  explicit FegenFunction(std::string name,
                         std::vector<FegenValue *> &&inputTypeList,
                         FegenType *returnType);

public:
  static FegenFunction *get(std::string name,
                            std::vector<FegenValue *> inputTypeList,
                            FegenType *returnType = nullptr);
  ~FegenFunction() = default;
  std::string getName();
  std::vector<FegenValue *> &getInputTypeList();
  FegenValue *getInputTypeList(size_t i);
  FegenType *getReturnType();
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
  // operation body context
  FegenParser::BodySpecContext *ctx;
  explicit FegenOperation(std::string dialectName, std::string operationName,
                          std::vector<FegenValue *> &&arguments,
                          std::vector<FegenValue *> &&results,
                          FegenParser::BodySpecContext *ctx);

public:
  void setOpName(std::string);
  std::string getOpName();
  std::vector<FegenValue *> &getArguments();
  FegenValue *getArguments(size_t i);
  std::vector<FegenValue *> &getResults();
  FegenValue *getResults(size_t i);
  static FegenOperation *get(std::string operationName,
                             std::vector<FegenValue *> arguments,
                             std::vector<FegenValue *> results,
                             FegenParser::BodySpecContext *ctx);
  ~FegenOperation() = default;
};

class FegenTypeDefination;

class FegenType {
  friend class FegenValue;

public:
  enum class TypeKind { ATTRIBUTE, OPERAND, CPP };

private:
  TypeKind kind;
  std::string typeName;
  std::vector<FegenValue *> parameters;
  FegenTypeDefination *typeDefine;
  int typeLevel;

public:
  FegenType(TypeKind kind, std::string name,
            std::vector<FegenValue *> parameters, FegenTypeDefination *tyDef,
            int typeLevel);
  FegenType(TypeKind kind, std::vector<FegenValue *> parameters,
            FegenTypeDefination *tyDef, int typeLevel);
  FegenType(const FegenType &);
  FegenType(FegenType &&);
  TypeKind getTypeKind();
  void setTypeKind(TypeKind kind);
  std::vector<FegenValue *> &getParameters();
  FegenValue *getParameters(size_t i);
  void setParameters(std::vector<FegenValue *> &params);
  FegenTypeDefination *getTypeDefination();
  void setTypeDefination(FegenTypeDefination *tyDef);
  std::string getTypeName();
  int getTypeLevel();
  // for generating typedef td file.
  std::string toStringForTypedef();
  // for generating op def td file.
  std::string toStringForOpdef();
  // for generating cpp type kind.
  std::string toStringForCppKind();
  static bool isSameType(FegenType *type1, FegenType *type2);
  ~FegenType();
  // placeholder
  static FegenType getPlaceHolder();
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
  static FegenType getIntegerType(FegenValue *size);

  // FloatPoint<size>
  static FegenType getFloatPointType(FegenValue *size);

  // char
  static FegenType getCharType();

  // string
  static FegenType getStringType();

  // Vector<size, elementType>
  static FegenType getVectorType(FegenValue *size, FegenType elementType);

  // Tensor<shape, elementType>
  static FegenType getTensorType(FegenValue *shape, FegenType elementType);

  // List<elementType>
  static FegenType getListType(FegenType elementType);

  // Optional<elementType>
  static FegenType getOptionalType(FegenType elementType);

  // Any<elementType1, elementType2, ...>
  static FegenType getAnyType(std::vector<FegenType> elementTypes);

  static FegenType getIntegerTemplate();
  static FegenType getFloatPointTemplate();

  static FegenType getInstanceType(FegenTypeDefination *typeDefination,
                                   std::vector<FegenValue *> parameters);

  static FegenType getTemplateType(FegenTypeDefination *typeDefination);
};

class FegenTypeDefination {
  friend class FegenManager;

private:
  std::string dialectName;
  std::string name;
  std::vector<fegen::FegenValue *> parameters;
  FegenParser::TypeDefinationDeclContext *ctx;
  bool ifCustome;
  std::string mnemonic;

public:
  FegenTypeDefination(std::string dialectName, std::string name,
                      std::vector<fegen::FegenValue *> parameters,
                      FegenParser::TypeDefinationDeclContext *ctx,
                      bool ifCustome);
  static FegenTypeDefination *get(std::string dialectName, std::string name,
                                  std::vector<fegen::FegenValue *> parameters,
                                  FegenParser::TypeDefinationDeclContext *ctx,
                                  bool ifCustome = true);
  std::string getDialectName();
  void setDialectName(std::string);
  std::string getName();
  std::string getMnemonic();
  void setName(std::string);
  const std::vector<fegen::FegenValue *> &getParameters();
  FegenParser::TypeDefinationDeclContext *getCtx();
  void setCtx(FegenParser::TypeDefinationDeclContext *);
  bool isCustome();
};

/// @brief Represent right value, and pass by value.
class FegenRightValue {
  friend class FegenType;
  friend class FegenValue;

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
    FegenType exprType;
    bool isLiteral;
    bool ifConstexpr;
    Expression(bool, LiteralKind, FegenType &, bool);
    virtual ~Expression() = default;
    virtual bool isTerminal();
    virtual std::string toString() = 0;
    virtual std::string toStringForTypedef() = 0;
    virtual std::string toStringForOpdef() = 0;
    virtual std::string toStringForCppKind() = 0;
    LiteralKind getKind();
    FegenType &getType();
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
    callFunction(std::vector<std::shared_ptr<Expression>>, FegenFunction *);

    // TODO: callOperation
    static std::shared_ptr<OperationCall>
    callOperation(std::vector<std::shared_ptr<Expression>>, FegenOperation *);

    static std::shared_ptr<ExpressionTerminal> getPlaceHolder();
    static std::shared_ptr<ExpressionTerminal> getInteger(long long int,
                                                          size_t size = 32);
    static std::shared_ptr<ExpressionTerminal> getFloatPoint(long double,
                                                             size_t size = 32);
    static std::shared_ptr<ExpressionTerminal> getString(std::string);
    static std::shared_ptr<ExpressionTerminal> getType(FegenType &);
    static std::shared_ptr<ExpressionTerminal>
    getList(std::vector<std::shared_ptr<Expression>> &);
    static std::shared_ptr<ExpressionTerminal>
    getLeftValue(fegen::FegenValue *);
  };

  struct ExpressionNode : public Expression {
    ExpressionNode(LiteralKind, FegenType, bool);
    virtual std::string toString() override;
    virtual std::string toStringForTypedef() override;
    virtual std::string toStringForOpdef() override;
    virtual std::string toStringForCppKind() override;
    virtual std::any getContent() override = 0;
  };

  struct FunctionCall : public ExpressionNode {
    FegenFunction *func;
    std::vector<std::shared_ptr<Expression>> params;
    FunctionCall(FegenFunction *, std::vector<std::shared_ptr<Expression>>);
    virtual std::string toString() override;
    virtual std::string toStringForTypedef() override;
    virtual std::string toStringForOpdef() override;
    virtual std::string toStringForCppKind() override;
    virtual std::any getContent() override;
  };

  struct OperationCall : public ExpressionNode {
    FegenOperation *op;
    std::vector<std::shared_ptr<Expression>> params;
    OperationCall(FegenOperation *, std::vector<std::shared_ptr<Expression>>);
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
    ExpressionTerminal(LiteralKind, FegenType, bool);
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
    FegenType content;
    TypeLiteral(FegenType &content);
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
    FegenValue *content;
    LeftValue(FegenValue *content);
    virtual std::any getContent() override;
    virtual std::string toString() override;
  };

public:
  FegenRightValue(std::shared_ptr<Expression>);
  FegenRightValue(const FegenRightValue &) = default;
  FegenRightValue(FegenRightValue &&) = default;
  FegenRightValue::LiteralKind getLiteralKind();
  std::string toString();
  std::string toStringForTypedef();
  std::string toStringForOpdef();
  std::string toStringForCppKind();
  std::any getContent();
  FegenType &getType();
  std::shared_ptr<Expression> getExpr();

  static FegenRightValue getPlaceHolder();
  static FegenRightValue getInteger(long long int content, size_t size = 32);
  static FegenRightValue getFloatPoint(long double content, size_t size = 32);
  static FegenRightValue getString(std::string content);
  static FegenRightValue getType(FegenType &content);
  static FegenRightValue
  getList(std::vector<std::shared_ptr<Expression>> &content);
  static FegenRightValue getLeftValue(fegen::FegenValue *content);
  static FegenRightValue getByExpr(std::shared_ptr<Expression> expr);
  ~FegenRightValue() = default;

private:
  std::shared_ptr<Expression> content;
};

class FegenValue {
  friend class FegenType;

private:
  FegenType type;
  std::string name;
  FegenRightValue content;

public:
  FegenValue(FegenType type, std::string name, FegenRightValue content);
  FegenValue(const FegenValue &rhs);
  FegenValue(FegenValue &&rhs);

  static FegenValue *get(FegenType type, std::string name,
                         FegenRightValue constant);

  std::string getName();
  FegenType &getType();
  /// @brief return content of right value, get ExprssionNode* if kind is
  /// EXPRESSION.
  template <typename T> T getContent() {
    return std::any_cast<T>(this->content.getContent());
  }
  FegenRightValue::LiteralKind getContentKind();
  std::string getContentString();
  std::string getContentStringForTypedef();
  std::string getContentStringForOpdef();
  std::string getContentStringForCppKind();
  std::shared_ptr<FegenRightValue::Expression> getExpr();
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
  FegenManager();
  FegenManager(const FegenManager &) = delete;
  const FegenManager &operator=(const FegenManager &) = delete;
  // release nodes, type, operation, function
  ~FegenManager();
  void initbuiltinTypes();

public:
  std::string moduleName;
  std::vector<std::string> headFiles;
  std::map<std::string, FegenNode *> nodeMap;
  llvm::StringMap<FegenType *> typeMap;
  std::map<std::string, FegenTypeDefination *> typeDefMap;
  std::map<std::string, FegenOperation *> operationMap;
  std::map<std::string, FegenFunction *> functionMap;
  // stmt contents
  std::unordered_map<antlr4::ParserRuleContext *, std::any> stmtContentMap;
  void addStmtContent(antlr4::ParserRuleContext *ctx, std::any content);
  template <typename T> T getStmtContent(antlr4::ParserRuleContext *ctx) {
    assert(this->stmtContentMap.count(ctx));
    return std::any_cast<T>(this->stmtContentMap[ctx]);
  }

  static FegenManager &getManager();
  void setModuleName(std::string name);

  FegenTypeDefination *getTypeDefination(std::string name);
  bool addTypeDefination(FegenTypeDefination *tyDef);

  FegenOperation *getOperationDefination(std::string name);
  bool addOperationDefination(FegenOperation *opDef);
  void emitG4();
  void emitTypeDefination();
  void emitOpDefination();
  void emitDialectDefination();
  void emitTdFiles();
  void emitBuiltinFunction();
};

FegenType
    inferenceType(std::vector<std::shared_ptr<FegenRightValue::Expression>>,
                  FegenOperator);

} // namespace fegen

#endif