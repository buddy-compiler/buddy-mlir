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

namespace fegen {

class FegenType;
class FegenManager;

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
  std::map<FegenType *, std::string> inputTypeMap;
  // return type
  FegenType *returnType;
  explicit FegenFunction(llvm::StringRef name,
                         std::map<FegenType *, std::string> &&inputTypeMap,
                         FegenType *returnType);

public:
  static FegenFunction *get(llvm::StringRef name,
                            std::map<FegenType *, std::string> inputTypeMap,
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

class FegenTypeDefination;

class FegenType {
  friend class FegenValue;

public:
  enum class TypeKind { ATTRIBUTE, OPERAND, CPP };

private:
  TypeKind kind;
  std::string typeName;
  std::vector<FegenValue*> parameters;
  FegenTypeDefination *typeDefine;
  bool ifTemplate;

public:
  FegenType(TypeKind kind, std::vector<FegenValue*> parameters,
            FegenTypeDefination *tyDef, bool isTemplate);
  FegenType(const FegenType &);
  FegenType(FegenType &&);
  TypeKind getTypeKind();
  void setTypeKind(TypeKind kind);
  std::vector<FegenValue*> &getParameters();
  void setParameters(std::vector<FegenValue*> &params);
  FegenTypeDefination *getTypeDefination();
  void setTypeDefination(FegenTypeDefination *tyDef);
  std::string getTypeName();
  bool isTemplate();
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
  static FegenType getIntegerType(FegenValue* size);

  // FloatPoint<size>
  static FegenType getFloatPointType(FegenValue* size);

  // char
  static FegenType getCharType();

  // string
  static FegenType getStringType();

  // Vector<size, elementType>
  static FegenType getVectorType(FegenValue* size, FegenValue* elementType);
  static FegenType getVectorType(FegenValue* size, FegenType elementType);

  // Tensor<shape, elementType>
  static FegenType getTensorType(FegenValue* shape, FegenValue* elementType);
  static FegenType getTensorType(FegenValue* shape, FegenType elementType);

  // List<elementType>
  static FegenType getListType(FegenValue* elementType);
  static FegenType getListType(FegenType elementType);

  // Optional<elementType>
  static FegenType getOptionalType(FegenValue* elementType);
  static FegenType getOptionalType(FegenType elementType);

  // Any<elementType1, elementType2, ...>
  static FegenType getAnyType(std::vector<FegenValue*> elementTypes);
  static FegenType getAnyType(std::vector<FegenType> elementTypes);

  static FegenType getIntegerTemplate();
  static FegenType getFloatPointTemplate();
  static FegenType getVectorTemplate();
  static FegenType getTensorTemplate();
  static FegenType getListTemplate();
  static FegenType getOptionalTemplate();
  static FegenType getAnyTemplate();

  static FegenType getInstanceType(FegenTypeDefination *typeDefination,
                                   std::vector<FegenValue*> parameters);

  static FegenType getTemplateType(FegenTypeDefination *typeDefination);
};

class FegenTypeDefination {
  friend class FegenManager;

private:
  std::string dialectName;
  std::string name;
  std::vector<fegen::FegenValue*> parameters;
  FegenParser::TypeDefinationDeclContext *ctx;
  bool ifCustome;

public:
  FegenTypeDefination(std::string dialectName, std::string name,
                      std::vector<fegen::FegenValue*> parameters,
                      FegenParser::TypeDefinationDeclContext *ctx,
                      bool ifCustome);
  static FegenTypeDefination *get(std::string dialectName, std::string name,
                                  std::vector<fegen::FegenValue*> parameters,
                                  FegenParser::TypeDefinationDeclContext *ctx,
                                  bool ifCustome = true);
  std::string getDialectName();
  std::string getName();
  const std::vector<fegen::FegenValue*> &getParameters();
  FegenParser::TypeDefinationDeclContext *getCtx();
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
    EXPRESSION,
    LEFT_VAR
  };

  struct Expression {
    bool ifTerminal;
    LiteralKind kind;
    FegenType exprType;
    Expression(bool, LiteralKind, FegenType&);
    virtual ~Expression() = default;
    virtual bool isTerminal();
    virtual std::string toString() = 0;
    LiteralKind getKind();
    virtual std::any getContent() = 0;
  };

  struct ExpressionNode : public Expression {
    using opType =
        std::variant<FegenFunction *, FegenOperation *, FegenOperator>;
    opType op;
    std::vector<Expression*> params;
    ExpressionNode(std::vector<Expression*>, opType, FegenType&);
    ExpressionNode(ExpressionNode&)=default;
    ~ExpressionNode();
    virtual std::string toString() override;
    virtual std::any getContent() override;

    /// @brief operate lhs and rhs using binary operator.
    static ExpressionNode *binaryOperation(Expression *lhs, Expression *rhs,
                                           FegenOperator op);
    /// @brief operate expr using unary operator
    static ExpressionNode *unaryOperation(Expression *, FegenOperator);

    // TODO: callFunction
    static ExpressionNode* callFunction(std::vector<Expression*>, FegenFunction*);

    // TODO: callOperation
    static ExpressionNode* callOperation(std::vector<Expression*>, FegenOperation*);
  };

  struct ExpressionTerminal : public Expression {
    // monostate, int literal, float literal, string literal, type literal, list
    // literal, reference of variable
    using primLiteralType =
        std::variant<std::monostate, int, float, std::string, FegenType,
                     std::vector<Expression*>, FegenValue *>;
    primLiteralType content;
    ExpressionTerminal(primLiteralType, LiteralKind, FegenType);
    ExpressionTerminal(ExpressionTerminal&)=default;
    ~ExpressionTerminal();
    virtual std::string toString() override;
    virtual std::any getContent() override;
    static ExpressionTerminal *get(std::monostate);
    static ExpressionTerminal *get(int);
    static ExpressionTerminal *get(float);
    static ExpressionTerminal *get(std::string);
    static ExpressionTerminal *get(FegenType &);
    static ExpressionTerminal *get(std::vector<Expression*> &);
    static ExpressionTerminal *get(fegen::FegenValue *);
  };

public:
  FegenRightValue(Expression *content);
  FegenRightValue(const FegenRightValue&);
  FegenRightValue(FegenRightValue&&);
  FegenRightValue::LiteralKind getKind();
  std::string toString();
  std::any getContent();

  static FegenRightValue get();
  static FegenRightValue get(int content);
  static FegenRightValue get(float content);
  static FegenRightValue get(std::string content);
  static FegenRightValue get(FegenType& content);
  static FegenRightValue get(std::vector<Expression*> & content);
  static FegenRightValue get(fegen::FegenValue * content);
  static FegenRightValue get(Expression* expr);
  ~FegenRightValue();

private:
  Expression *content;
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
  /// @brief return content of right value, get ExprssionNode* if kind is EXPRESSION.
  template <typename T> T getContent() { return std::any_cast<T>(this->content.getContent()); }
  FegenRightValue::LiteralKind getContentKind();
  std::string getContentString();
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
  std::string moduleName;
  std::vector<std::string> headFiles;
  std::map<std::string, FegenNode *> nodeMap;
  llvm::StringMap<FegenType *> typeMap;
  std::map<std::string, FegenTypeDefination *> typeDefMap;
  llvm::StringMap<FegenOperation *> operationMap;
  std::vector<FegenFunction *> functions;
  // llvm::StringMap<FegenFunction *> functionMap;
  void initbuiltinTypes();

public:
  static FegenManager &getManager();
  void setModuleName(std::string name);

  FegenTypeDefination *getTypeDefination(std::string name);
  bool addTypeDefination(FegenTypeDefination *tyDef);
  std::string emitG4();
};

FegenType inferenceType(std::vector<FegenRightValue::Expression*>, FegenOperator);

} // namespace fegen

#endif