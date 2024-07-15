#include "FegenManager.h"
#include "FegenParser.h"
#include "FegenParserBaseVisitor.h"
#include "Scope.h"
#include <algorithm>
#include <any>
#include <cassert>
#include <cstddef>
#include <fstream>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <variant>

fegen::Function::Function(std::string name,
                                    std::vector<Value *> &&inputTypeList,
                                    Type *returnType)
    : name(name), inputTypeList(inputTypeList), returnType(returnType) {}

fegen::Function *
fegen::Function::get(std::string name,
                          std::vector<Value *> inputTypeList,
                          Type *returnType) {
  return new fegen::Function(name, std::move(inputTypeList), returnType);
}
std::string fegen::Function::getName() { return this->name; }

std::vector<fegen::Value *> &fegen::Function::getInputTypeList() {
  return this->inputTypeList;
}

fegen::Value *fegen::Function::getInputTypeList(size_t i) {
  return this->inputTypeList[i];
}

fegen::Type *fegen::Function::getReturnType() {
  return this->returnType;
}

fegen::Operation::Operation(std::string dialectName,
                                      std::string operationName,
                                      std::vector<Value *> &&arguments,
                                      std::vector<Value *> &&results,
                                      fegen::FegenParser::BodySpecContext *ctx)
    : dialectName(dialectName), arguments(arguments), results(results),
      ctx(ctx) {}

void fegen::Operation::setOpName(std::string name) {
  this->operationName = name;
}
std::string fegen::Operation::getOpName() { return this->operationName; }

std::vector<fegen::Value *> &fegen::Operation::getArguments() {
  return this->arguments;
}

fegen::Value *fegen::Operation::getArguments(size_t i) {
  return this->arguments[i];
}

std::vector<fegen::Value *> &fegen::Operation::getResults() {
  return this->results;
}

fegen::Value *fegen::Operation::getResults(size_t i) {
  return this->results[i];
}

fegen::Operation *fegen::Operation::get(
    std::string operationName, std::vector<Value *> arguments,
    std::vector<Value *> results, FegenParser::BodySpecContext *ctx) {
  return new fegen::Operation(fegen::Manager::getManager().moduleName,
                                   operationName, std::move(arguments),
                                   std::move(results), ctx);
}

// class FegenType

/// @brief get name of Type Instance by jointsing template name and parameters,
/// for example: Integer + 32 --> Integer<32>
/// @return joint name
std::string jointTypeName(std::string templateName,
                          const std::vector<fegen::Value *> &parameters) {
  if (parameters.empty()) {
    return templateName;
  }
  std::string res = templateName;
  res.append("<");
  size_t count = parameters.size();
  auto firstParamStr = parameters[0]->getContentString();
  res.append(firstParamStr);
  for (size_t i = 1; i <= count - 1; i++) {
    auto paramStr = parameters[i]->getContentString();
    res.append(", ");
    res.append(paramStr);
  }
  res.append(">");
  return res;
}

fegen::Type::Type(TypeKind kind, std::string name,
                            std::vector<Value *> parameters,
                            TypeDefination *tyDef, int typeLevel)
    : kind(kind), typeName(name), parameters(std::move(parameters)),
      typeDefine(tyDef), typeLevel(typeLevel) {}

fegen::Type::Type(fegen::Type::TypeKind kind,
                            std::vector<Value *> parameters,
                            TypeDefination *tyDef, int typeLevel)
    : kind(kind), typeName(jointTypeName(tyDef->getName(), parameters)),
      parameters(std::move(parameters)), typeDefine(tyDef),
      typeLevel((typeLevel)) {}

fegen::Type::Type(const fegen::Type &fty)
    : kind(fty.kind), typeName(fty.typeName), typeDefine(fty.typeDefine),
      typeLevel(fty.typeLevel) {
  // deep copy parameters
  for (auto paramPtr : fty.parameters) {
    this->parameters.push_back(new fegen::Value(*paramPtr));
  }
}

fegen::Type::Type(fegen::Type &&fty)
    : kind(fty.kind), typeName(std::move(fty.typeName)),
      parameters(std::move(fty.parameters)), typeDefine(fty.typeDefine),
      typeLevel(fty.typeLevel) {}

fegen::Type::TypeKind fegen::Type::getTypeKind() {
  return this->kind;
}

void fegen::Type::setTypeKind(fegen::Type::TypeKind kind) {
  this->kind = kind;
}

std::vector<fegen::Value *> &fegen::Type::getParameters() {
  return this->parameters;
}

fegen::Value *fegen::Type::getParameters(size_t i) {
  return this->parameters[i];
}

void fegen::Type::setParameters(std::vector<fegen::Value *> &params) {
  this->parameters = params;
  // set parameters and level up!
  this->typeLevel++;
}

fegen::TypeDefination *fegen::Type::getTypeDefination() {
  return this->typeDefine;
}

void fegen::Type::setTypeDefination(fegen::TypeDefination *tyDef) {
  this->typeDefine = tyDef;
}

std::string fegen::Type::getTypeName() { return this->typeName; }

int fegen::Type::getTypeLevel() { return this->typeLevel; }

bool fegen::Type::isSameType(fegen::Type *type1,
                                  fegen::Type *type2) {
  if (type1->getTypeName() == type2->getTypeName())
    return true;
  else
    return false;
}

std::string fegen::Type::toStringForTypedef() {
  // handle builtin type instance
  auto typeName = this->typeName;
  auto typedefName = this->typeDefine->getName();
  if (this->typeDefine->isCustome()) {
    return this->typeDefine->getName();
  } else if (typedefName == FEGEN_TYPE) {
    return "\"Type\"";
  } else if (typedefName == FEGEN_LIST) {
    std::string res = "ArrayRefParameter<";
    for (size_t i = 0; i <= this->parameters.size() - 1; i++) {
      res.append(this->parameters[i]->getContentStringForTypedef());
      if (i != this->parameters.size() - 1) {
        res.append(", ");
      }
    }
    res.append(">");
    return res;
  } else if (typedefName == FEGEN_INTEGER) {
    if (this->parameters.size() == 0) {
      return "Builtin_IntegerAttr";
    } else {
      if (typeName == "int") {
        return "\"int\"";
      } else if (typeName == "bool") {
        return "\"bool\"";
      }
      int size = this->getParameters(0)->getContent<int>();
      if (size == 64) {
        return "\"long\"";
      } else if (size == 16) {
        return "\"short\"";
      } else {
        std::cerr << "unsupport type: " << typeName << std::endl;
        exit(0);
      }
    }
  } else if (typedefName == FEGEN_FLOATPOINT) {
    if (this->parameters.size() == 0) {
      return "Builtin_FloatAttr";
    } else {
      if (typeName == "float") {
        return "\"float\"";
      } else if (typeName == "double") {
        return "\"double\"";
      } else {
        std::cerr << "unsupport type: " << typeName << std::endl;
        exit(0);
      }
    }
  } else {
    std::cerr << "unsupport type: " << typeName << std::endl;
    exit(0);
  }
}

std::string fegen::Type::toStringForOpdef() {
  // handle builtin type instance
  auto typeName = this->typeName;
  auto typedefName = this->typeDefine->getName();
  if (this->typeDefine->isCustome()) {
    return this->typeDefine->getName();
  } else if (typedefName == FEGEN_LIST) {
    std::string res = "Variadic<";
    assert(this->parameters.size() == 1);
    res.append(this->parameters[0]->getContentStringForTypedef());
    res.append(">");
    return res;
  } else if (typedefName == FEGEN_INTEGER) {
    if (this->parameters.size() == 0) {
      return "Builtin_Integer";
    } else {
      if (typeName == "int") {
        return "I32";
      }
      int size = this->getParameters(0)->getContent<int>();
      if (size == 64) {
        return "I64";
      } else if (size == 16) {
        return "I16";
      }
    }
  }

  std::cerr << "unsupport type: " << typeName << std::endl;
  exit(0);
}

std::string fegen::Type::toStringForCppKind() {
  // handle builtin type instance
  auto typeName = this->typeName;
  auto typedefName = this->typeDefine->getName();
  if (typedefName == FEGEN_LIST) {
    assert(this->parameters.size() == 1);
    std::string res = "std::vector<";
    res.append(this->parameters[0]->getContentStringForTypedef());
    res.append(">");
    return res;
  } else if (typedefName == FEGEN_INTEGER) {
    assert(this->parameters.size() == 1);
    if (typeName == "int") {
      return "int";
    }
    int size = this->getParameters(0)->getContent<int>();
    if (size == 64) {
      return "long";
    } else if (size == 16) {
      return "short";
    }
  } else if (typedefName == FEGEN_FLOATPOINT) {
    assert(this->parameters.size() == 1);
    if (typeName == "float") {
      return "float";
    } else if (typeName == "double") {
      return "double";
    }
  }
  std::cerr << "Unsupported type: " << typeName << "in generating cpp type."
            << std::endl;
  exit(0);
}

fegen::Type::~Type() {
  for (auto p : this->parameters) {
    delete p;
  }
}

fegen::Type fegen::Type::getPlaceHolder() {
  return fegen::Type(
      fegen::Type::TypeKind::CPP, {},
      fegen::Manager::getManager().getTypeDefination(FEGEN_PLACEHOLDER),
      0);
}

fegen::Type fegen::Type::getMetaType() {
  return fegen::Type(
      fegen::Type::TypeKind::CPP, {},
      fegen::Manager::getManager().getTypeDefination(FEGEN_TYPE), 2);
}

fegen::Type fegen::Type::getMetaTemplateType() {
  return fegen::Type(
      fegen::Type::TypeKind::CPP, {},
      fegen::Manager::getManager().getTypeDefination(FEGEN_TYPETEMPLATE),
      1);
}

fegen::Type fegen::Type::getInt32Type() {
  return fegen::Type(
      fegen::Type::TypeKind::CPP, "int",
      {fegen::Value::get(fegen::Type::getPlaceHolder(), "size",
                              fegen::RightValue::getPlaceHolder())},
      fegen::Manager::getManager().getTypeDefination(FEGEN_INTEGER), 3);
}

fegen::Type fegen::Type::getFloatType() {
  return fegen::Type(
      fegen::Type::TypeKind::CPP, "float",
      {fegen::Value::get(fegen::Type::getInt32Type(), "size",
                              fegen::RightValue::getInteger(32))},
      fegen::Manager::getManager().getTypeDefination(FEGEN_FLOATPOINT), 3);
}

fegen::Type fegen::Type::getDoubleType() {
  return fegen::Type(
      fegen::Type::TypeKind::CPP, "double",
      {fegen::Value::get(fegen::Type::getInt32Type(), "size",
                              fegen::RightValue::getInteger(64))},
      fegen::Manager::getManager().getTypeDefination(FEGEN_FLOATPOINT), 3);
}

fegen::Type fegen::Type::getBoolType() {
  return fegen::Type(
      fegen::Type::TypeKind::CPP, "bool",
      {fegen::Value::get(fegen::Type::getInt32Type(), "size",
                              fegen::RightValue::getInteger(1))},
      fegen::Manager::getManager().getTypeDefination(FEGEN_INTEGER), 3);
}

fegen::Type fegen::Type::getIntegerType(fegen::Value *size) {
  if (size->getContent<int>() == 32)
    return fegen::Type::getInt32Type();
  return fegen::Type(
      fegen::Type::TypeKind::CPP, {size},
      fegen::Manager::getManager().getTypeDefination(FEGEN_INTEGER), 3);
}

fegen::Type fegen::Type::getFloatPointType(fegen::Value *size) {
  if (size->getContent<int>() == 32) {
    return fegen::Type::getFloatType();
  } else if (size->getContent<int>() == 64) {
    return fegen::Type::getDoubleType();
  }
  return fegen::Type(
      fegen::Type::TypeKind::CPP, {size},
      fegen::Manager::getManager().getTypeDefination(FEGEN_FLOATPOINT), 3);
}

fegen::Type fegen::Type::getCharType() {
  return fegen::Type(
      fegen::Type::TypeKind::CPP, {},
      fegen::Manager::getManager().getTypeDefination(FEGEN_CHAR), 3);
}

fegen::Type fegen::Type::getStringType() {
  return fegen::Type(
      fegen::Type::TypeKind::CPP, {},
      fegen::Manager::getManager().getTypeDefination(FEGEN_STRING), 3);
}

fegen::Type fegen::Type::getVectorType(fegen::Value *size,
                                                 fegen::Type elementType) {
  assert(elementType.typeLevel == 3);
  return fegen::Type(
      fegen::Type::TypeKind::CPP,
      {size,
       fegen::Value::get(fegen::Type::getMetaType(), "elementType",
                              fegen::RightValue::getType(elementType))},
      fegen::Manager::getManager().getTypeDefination(FEGEN_VECTOR),
      elementType.typeLevel);
}

fegen::Type fegen::Type::getTensorType(fegen::Value *shape,
                                                 fegen::Type elementType) {
  assert(elementType.typeLevel == 3);
  return fegen::Type(
      fegen::Type::TypeKind::CPP,
      {shape,
       fegen::Value::get(fegen::Type::getMetaType(), "elementType",
                              fegen::RightValue::getType(elementType))},
      fegen::Manager::getManager().getTypeDefination(FEGEN_TENSOR),
      elementType.typeLevel);
}

// List<elementType>
fegen::Type fegen::Type::getListType(fegen::Type elementType) {
  assert(elementType.typeLevel == 2 || elementType.typeLevel == 3);
  return fegen::Type(
      fegen::Type::TypeKind::CPP,
      {fegen::Value::get(
          elementType.typeLevel == 2 ? fegen::Type::getMetaTemplateType()
                                     : fegen::Type::getMetaType(),
          "elementType", fegen::RightValue::getType(elementType))},
      fegen::Manager::getManager().getTypeDefination(FEGEN_LIST),
      elementType.typeLevel);
}

// Optional<elementType>
fegen::Type
fegen::Type::getOptionalType(fegen::Type elementType) {
  assert(elementType.typeLevel == 2 || elementType.typeLevel == 3);
  return fegen::Type(
      fegen::Type::TypeKind::CPP,
      {fegen::Value::get(
          elementType.typeLevel == 2 ? fegen::Type::getMetaTemplateType()
                                     : fegen::Type::getMetaType(),
          "elementType", fegen::RightValue::getType(elementType))},
      fegen::Manager::getManager().getTypeDefination(FEGEN_OPTINAL),
      elementType.typeLevel);
}

// Any<elementType1, elementType2, ...>
fegen::Type
fegen::Type::getAnyType(std::vector<fegen::Type> elementTypes) {
  std::vector<fegen::Value *> p_elemTy;
  int i = 0;
  std::string name("elementType_");
  auto tyLevel = elementTypes[0].typeLevel;
  assert(tyLevel == 2 || tyLevel == 3);
  auto tyty = tyLevel == 2 ? fegen::Type::getMetaTemplateType()
                           : fegen::Type::getMetaType();
  for (auto &ty : elementTypes) {
    assert(ty.typeLevel == tyLevel);
    p_elemTy.push_back(fegen::Value::get(
        tyty, name + std::to_string(i), fegen::RightValue::getType(ty)));
    i++;
  }
  return fegen::Type(
      fegen::Type::TypeKind::CPP, p_elemTy,
      fegen::Manager::getManager().getTypeDefination(FEGEN_ANY), tyLevel);
}

fegen::Type fegen::Type::getIntegerTemplate() {
  return fegen::Type(
      fegen::Type::TypeKind::CPP, {},
      fegen::Manager::getManager().getTypeDefination(FEGEN_INTEGER), 2);
}

fegen::Type fegen::Type::getFloatPointTemplate() {
  return fegen::Type(
      fegen::Type::TypeKind::CPP, {},
      fegen::Manager::getManager().getTypeDefination(FEGEN_FLOATPOINT), 2);
}

fegen::Type
fegen::Type::getInstanceType(fegen::TypeDefination *typeDefination,
                                  std::vector<fegen::Value *> parameters) {
  return fegen::Type(fegen::Type::TypeKind::CPP, parameters,
                          typeDefination, 3);
}
fegen::Type
fegen::Type::getTemplateType(fegen::TypeDefination *typeDefination) {
  return fegen::Type(fegen::Type::TypeKind::CPP, {}, typeDefination,
                          2);
}

// class FegenTypeDefination
fegen::TypeDefination::TypeDefination(
    std::string dialectName, std::string name,
    std::vector<fegen::Value *> parameters,
    FegenParser::TypeDefinationDeclContext *ctx, bool ifCustome)
    : dialectName(std::move(dialectName)), name(std::move(name)),
      parameters(std::move(parameters)), ctx(ctx), ifCustome(ifCustome) {}

fegen::TypeDefination *
fegen::TypeDefination::get(std::string dialectName, std::string name,
                                std::vector<fegen::Value *> parameters,
                                FegenParser::TypeDefinationDeclContext *ctx,
                                bool ifCustome) {
  return new fegen::TypeDefination(std::move(dialectName), std::move(name),
                                        std::move(parameters), ctx, ifCustome);
}

std::string fegen::TypeDefination::getDialectName() {
  return this->dialectName;
}

void fegen::TypeDefination::setDialectName(std::string name) {
  this->dialectName = name;
}

std::string fegen::TypeDefination::getName() { return this->name; }

std::string fegen::TypeDefination::getMnemonic() {
  if (this->mnemonic.empty()) {
    this->mnemonic = this->name;
    std::transform(this->mnemonic.begin(), this->mnemonic.end(),
                   this->mnemonic.begin(), ::tolower);
  }
  return this->mnemonic;
}

void fegen::TypeDefination::setName(std::string name) {
  this->name = name;
}

const std::vector<fegen::Value *> &
fegen::TypeDefination::getParameters() {
  return this->parameters;
}

fegen::FegenParser::TypeDefinationDeclContext *
fegen::TypeDefination::getCtx() {
  return this->ctx;
}

void fegen::TypeDefination::setCtx(
    FegenParser::TypeDefinationDeclContext *ctx) {
  this->ctx = ctx;
}

bool fegen::TypeDefination::isCustome() { return this->ifCustome; }

// class Expression

fegen::RightValue::Expression::Expression(bool ifTerminal,
                                               LiteralKind kind,
                                               Type &exprTy,
                                               bool isConstexpr)
    : ifTerminal(ifTerminal), kind(kind), exprType(exprTy),
      ifConstexpr(isConstexpr) {}

bool fegen::RightValue::Expression::isTerminal() {
  return this->ifTerminal;
}

fegen::RightValue::LiteralKind
fegen::RightValue::Expression::getKind() {
  return this->kind;
}

fegen::Type &fegen::RightValue::Expression::getType() {
  return this->exprType;
}

bool fegen::RightValue::Expression::isConstexpr() {
  return this->ifConstexpr;
}

std::shared_ptr<fegen::RightValue::ExpressionTerminal>
fegen::RightValue::Expression::getPlaceHolder() {
  return std::make_shared<fegen::RightValue::PlaceHolder>();
}

std::shared_ptr<fegen::RightValue::ExpressionTerminal>
fegen::RightValue::Expression::getInteger(long long int content,
                                               size_t size) {
  return std::make_shared<fegen::RightValue::IntegerLiteral>(content,
                                                                  size);
}

std::shared_ptr<fegen::RightValue::ExpressionTerminal>
fegen::RightValue::Expression::getFloatPoint(long double content,
                                                  size_t size) {
  return std::make_shared<fegen::RightValue::FloatPointLiteral>(content,
                                                                     size);
}

std::shared_ptr<fegen::RightValue::ExpressionTerminal>
fegen::RightValue::Expression::getString(std::string content) {
  return std::make_shared<fegen::RightValue::StringLiteral>(content);
}

std::shared_ptr<fegen::RightValue::ExpressionTerminal>
fegen::RightValue::Expression::getType(fegen::Type &content) {
  return std::make_shared<fegen::RightValue::TypeLiteral>(content);
}

std::shared_ptr<fegen::RightValue::ExpressionTerminal>
fegen::RightValue::Expression::getList(
    std::vector<std::shared_ptr<fegen::RightValue::Expression>> &content) {
  return std::make_shared<fegen::RightValue::ListLiteral>(content);
}

std::shared_ptr<fegen::RightValue::ExpressionTerminal>
fegen::RightValue::Expression::getLeftValue(fegen::Value *content) {
  return std::make_shared<fegen::RightValue::LeftValue>(content);
}

std::shared_ptr<fegen::RightValue::OperatorCall>
fegen::RightValue::Expression::binaryOperation(
    std::shared_ptr<fegen::RightValue::Expression> lhs,
    std::shared_ptr<fegen::RightValue::Expression> rhs, FegenOperator op) {
  Type resTy = fegen::inferenceType({lhs, rhs}, op);
  return std::make_shared<fegen::RightValue::OperatorCall>(
      op, std::vector<std::shared_ptr<fegen::RightValue::Expression>>{
              lhs, rhs});
}

std::shared_ptr<fegen::RightValue::OperatorCall>
fegen::RightValue::Expression::unaryOperation(
    std::shared_ptr<fegen::RightValue::Expression> v, FegenOperator op) {
  Type resTy = fegen::inferenceType({v}, op);
  return std::make_shared<fegen::RightValue::OperatorCall>(
      op, std::vector<std::shared_ptr<fegen::RightValue::Expression>>{v});
}

// class ExpressionNode

fegen::RightValue::ExpressionNode::ExpressionNode(LiteralKind kind,
                                                       Type exprTy,
                                                       bool ifConstexpr)
    : Expression(false, kind, exprTy, ifConstexpr) {}

std::string fegen::RightValue::ExpressionNode::toString() {
  assert(FEGEN_NOT_IMPLEMENTED_ERROR);
}

std::string fegen::RightValue::ExpressionNode::toStringForTypedef() {
  assert(FEGEN_NOT_IMPLEMENTED_ERROR);
}

std::string fegen::RightValue::ExpressionNode::toStringForOpdef() {
  assert(FEGEN_NOT_IMPLEMENTED_ERROR);
}

std::string fegen::RightValue::ExpressionNode::toStringForCppKind() {
  assert(FEGEN_NOT_IMPLEMENTED_ERROR);
}

inline bool isBinaryOperator(fegen::FegenOperator &op) {
  switch (op) {
  case fegen::FegenOperator::NEG:
  case fegen::FegenOperator::NOT:
    return false;
  default:
    return true;
  }
}

std::string getCppOperator(fegen::FegenOperator op) {
  // switch(op){
  // OR,
  // AND,
  // EQUAL,
  // NOT_EQUAL,
  // LESS,
  // LESS_EQUAL,
  // GREATER,
  // GREATER_EQUAL,
  // ADD,
  // SUB,
  // MUL,
  // DIV,
  // MOD,
  // POWER,
  // NEG,
  // NOT
  // }
}

// std::string res;
// auto opKind = this->op.index();
// if(opKind == 0){ // function
//   auto func = std::get<0>(this->op);
//   // res.append(func.)
//   // TODO: add FegenFunction methods.
// }else if(opKind == 1) { // operation
//   assert(false);
//   return res;
// }else{ // operator
//   auto op = std::get<2>(this->op);
//   if(isBinaryOperator(op)){
//     assert(this->params.size() == 2);
//     res.append(this->params[0]->toStringForCppKind());
//     switch(op){
//       case fegen::FegenOperator::ADD:{
//         res.append()
//       }
//     }
//     res.append(this->params[1]->toStringForCppKind());
//   }else{

//   }
//   switch(op) {
//     case fegen::FegenOperator::ADD: {

//     }
//   }
// }

// class FunctionCall
inline bool isFuncParamsAllConstant(
    std::vector<std::shared_ptr<fegen::RightValue::Expression>> &params) {
  for (auto param : params) {
    if (!param->isConstexpr()) {
      return false;
    }
  }
  return true;
}

// TODO: invoke methods of FegenFunction
fegen::RightValue::FunctionCall::FunctionCall(
    fegen::Function *func,
    std::vector<std::shared_ptr<fegen::RightValue::Expression>> params)
    : ExpressionNode(fegen::RightValue::LiteralKind::FUNC_CALL,
                     fegen::Type::getInt32Type(),
                     isFuncParamsAllConstant(params)),
      func(func), params(std::move(params)) {}

std::string fegen::RightValue::FunctionCall::toString() {
  return "FunctionCall::toString";
}

std::string fegen::RightValue::FunctionCall::toStringForTypedef() {
  return "FunctionCall::toStringForTypedef";
}

std::string fegen::RightValue::FunctionCall::toStringForOpdef() {
  return "FunctionCall::toStringForOpdef";
}

std::string fegen::RightValue::FunctionCall::toStringForCppKind() {
  return "FunctionCall::toStringForCppKind";
}

std::any fegen::RightValue::FunctionCall::getContent() { return this; }

// class OperationCall
fegen::RightValue::OperationCall::OperationCall(
    fegen::Operation *op,
    std::vector<std::shared_ptr<fegen::RightValue::Expression>> params)
    : ExpressionNode(fegen::RightValue::LiteralKind::OPERATION_CALL,
                     fegen::Type::getInt32Type(),
                     isFuncParamsAllConstant(params)),
      op(op), params(std::move(params)) {}

std::string fegen::RightValue::OperationCall::toString() {
  return "OperationCall::toString";
}

std::string fegen::RightValue::OperationCall::toStringForTypedef() {
  return "OperationCall::toStringForTypedef";
}

std::string fegen::RightValue::OperationCall::toStringForOpdef() {
  return "OperationCall::toStringForOpdef";
}

std::string fegen::RightValue::OperationCall::toStringForCppKind() {
  return "OperationCall::toStringForCppKind";
}

std::any fegen::RightValue::OperationCall::getContent() { return this; }

// class OperatorCall
fegen::RightValue::OperatorCall::OperatorCall(
    fegen::FegenOperator op,
    std::vector<std::shared_ptr<fegen::RightValue::Expression>> params)
    : ExpressionNode(fegen::RightValue::LiteralKind::OPERATION_CALL,
                     fegen::inferenceType(params, op),
                     isFuncParamsAllConstant(params)),
      op(op), params(std::move(params)) {}

std::string fegen::RightValue::OperatorCall::toString() {
  return "OperatorCall::toString";
}

std::string fegen::RightValue::OperatorCall::toStringForTypedef() {
  return "OperatorCall::toStringForTypedef";
}

std::string fegen::RightValue::OperatorCall::toStringForOpdef() {
  return "OperatorCall::toStringForOpdef";
}

std::string fegen::RightValue::OperatorCall::toStringForCppKind() {
  return "OperatorCall::toStringForCppKind";
}

std::any fegen::RightValue::OperatorCall::getContent() { return this; }

// class ExpressionTerminal
fegen::RightValue::ExpressionTerminal::ExpressionTerminal(
    fegen::RightValue::LiteralKind kind, Type exprTy,
    bool ifConstexpr)
    : Expression(true, kind, exprTy, ifConstexpr) {}

std::string fegen::RightValue::ExpressionTerminal::toString() {
  assert(FEGEN_NOT_IMPLEMENTED_ERROR);
}

std::string fegen::RightValue::ExpressionTerminal::toStringForTypedef() {
  assert(FEGEN_NOT_IMPLEMENTED_ERROR);
}

std::string fegen::RightValue::ExpressionTerminal::toStringForOpdef() {
  assert(FEGEN_NOT_IMPLEMENTED_ERROR);
}

std::string fegen::RightValue::ExpressionTerminal::toStringForCppKind() {
  assert(FEGEN_NOT_IMPLEMENTED_ERROR);
}

// class PlaceHolder
fegen::RightValue::PlaceHolder::PlaceHolder()
    : ExpressionTerminal(fegen::RightValue::LiteralKind::MONOSTATE,
                         fegen::Type::getPlaceHolder(), true) {}

std::any fegen::RightValue::PlaceHolder::getContent() {
  return std::monostate();
}

std::string fegen::RightValue::PlaceHolder::toString() { return ""; }

// class IntegerLiteral
fegen::RightValue::IntegerLiteral::IntegerLiteral(int content)
    : ExpressionTerminal(fegen::RightValue::LiteralKind::INT,
                         fegen::Type::getInt32Type(), true),
      content(content) {}

fegen::RightValue::IntegerLiteral::IntegerLiteral(long long int content,
                                                       size_t size)
    : ExpressionTerminal(
          fegen::RightValue::LiteralKind::INT,
          fegen::Type::getIntegerType(fegen::Value::get(
              fegen::Type::getInt32Type(), "size",
              fegen::RightValue::getByExpr(
                  std::make_shared<fegen::RightValue::IntegerLiteral>(
                      size)))),
          true),
      content(content) {}

std::any fegen::RightValue::IntegerLiteral::getContent() {
  return this->content;
}

std::string fegen::RightValue::IntegerLiteral::toString() {
  return std::to_string(this->content);
}

// class FloatPointLiteral
fegen::RightValue::FloatPointLiteral::FloatPointLiteral(
    long double content, size_t size)
    : ExpressionTerminal(
          fegen::RightValue::LiteralKind::FLOAT,
          fegen::Type::getFloatPointType(
              fegen::Value::get(fegen::Type::getInt32Type(), "size",
                                     fegen::RightValue::getInteger(size))),
          true),
      content(content) {}

std::any fegen::RightValue::FloatPointLiteral::getContent() {
  return this->content;
}

std::string fegen::RightValue::FloatPointLiteral::toString() {
  return std::to_string(this->content);
}

// class StringLiteral
fegen::RightValue::StringLiteral::StringLiteral(std::string content)
    : ExpressionTerminal(fegen::RightValue::LiteralKind::STRING,
                         fegen::Type::getStringType(), true),
      content(content) {}

std::any fegen::RightValue::StringLiteral::getContent() {
  return this->content;
}

std::string fegen::RightValue::StringLiteral::toString() {
  std::string res;
  res.append("\"");
  res.append(this->content);
  res.append("\"");
  return res;
}

// class TypeLiteral

// Check params of content and return ture if params are all const expr.
inline bool isParamsConstant(fegen::Type &content) {
  for (auto param : content.getParameters()) {
    if (!param->getExpr()->isConstexpr()) {
      return false;
    }
  }
  return true;
}

// Get type of type literal.
fegen::Type getTypeLiteralType(fegen::Type &content) {
  if (content.getTypeLevel() == 2) {
    return fegen::Type::getMetaTemplateType();
  } else if (content.getTypeLevel() == 3) {
    return fegen::Type::getMetaType();
  } else {
    return fegen::Type::getPlaceHolder();
  }
}

fegen::RightValue::TypeLiteral::TypeLiteral(fegen::Type &content)
    : ExpressionTerminal(fegen::RightValue::LiteralKind::TYPE,
                         getTypeLiteralType(content),
                         isParamsConstant(content)),
      content(content) {}

std::any fegen::RightValue::TypeLiteral::getContent() {
  return this->content;
}

std::string fegen::RightValue::TypeLiteral::toString() {
  return this->content.getTypeName();
}

std::string fegen::RightValue::TypeLiteral::toStringForTypedef() {
  return this->content.toStringForTypedef();
}

std::string fegen::RightValue::TypeLiteral::toStringForOpdef() {
  return this->content.toStringForOpdef();
}

std::string fegen::RightValue::TypeLiteral::toStringForCppKind() {
  return this->content.toStringForCppKind();
}

// class ExpressionTerminal

// Return ture if all Expressions in content are all true.
bool isExpressionListConst(
    std::vector<std::shared_ptr<fegen::RightValue::Expression>> &content) {
  for (auto p : content) {
    if (!p->isConstexpr()) {
      return false;
      break;
    }
  }
  return true;
}

fegen::RightValue::ListLiteral::ListLiteral(
    std::vector<std::shared_ptr<Expression>> &content)
    : ExpressionTerminal(fegen::RightValue::LiteralKind::VECTOR,
                         content[0]->exprType, isExpressionListConst(content)),
      content(content) {}

std::any fegen::RightValue::ListLiteral::getContent() {
  return this->content;
}

std::string fegen::RightValue::ListLiteral::toString() {
  std::string res;
  res.append("[");
  for (size_t i = 0; i <= this->content.size() - 1; i++) {
    res.append(this->content[i]->toString());
    if (i != this->content.size() - 1) {
      res.append(", ");
    }
  }
  res.append("]");
  return res;
}

std::string fegen::RightValue::ListLiteral::toStringForTypedef() {
  std::string res;
  res.append("[");
  for (size_t i = 0; i <= this->content.size() - 1; i++) {
    res.append(this->content[i]->toStringForTypedef());
    if (i != this->content.size() - 1) {
      res.append(", ");
    }
  }
  res.append("]");
  return res;
}

std::string fegen::RightValue::ListLiteral::toStringForOpdef() {
  std::string res;
  res.append("[");
  for (size_t i = 0; i <= this->content.size() - 1; i++) {
    res.append(this->content[i]->toStringForOpdef());
    if (i != this->content.size() - 1) {
      res.append(", ");
    }
  }
  res.append("]");
  return res;
}

// class LeftValue
fegen::RightValue::LeftValue::LeftValue(fegen::Value *content)
    : ExpressionTerminal(fegen::RightValue::LiteralKind::LEFT_VAR,
                         content->getType(), content->getExpr()->isConstexpr()),
      content(content) {}

std::any fegen::RightValue::LeftValue::getContent() {
  return this->content;
}

std::string fegen::RightValue::LeftValue::toString() {
  return this->content->getName();
}

// class FegenRightValue
fegen::RightValue::RightValue(
    std::shared_ptr<fegen::RightValue::Expression> content)
    : content(content) {}

fegen::RightValue::LiteralKind fegen::RightValue::getLiteralKind() {
  return this->content->getKind();
}

std::string fegen::RightValue::toString() {
  return this->content->toString();
}

std::string fegen::RightValue::toStringForTypedef() {
  return this->content->toStringForTypedef();
}

std::string fegen::RightValue::toStringForOpdef() {
  return this->content->toStringForOpdef();
}

std::string fegen::RightValue::toStringForCppKind() {
  return this->content->toStringForCppKind();
}

std::any fegen::RightValue::getContent() {
  return this->content->getContent();
}

fegen::Type &fegen::RightValue::getType() {
  return this->content->getType();
}

std::shared_ptr<fegen::RightValue::Expression>
fegen::RightValue::getExpr() {
  return this->content;
}

fegen::RightValue fegen::RightValue::getPlaceHolder() {
  return fegen::RightValue(
      fegen::RightValue::Expression::getPlaceHolder());
}

fegen::RightValue fegen::RightValue::getInteger(long long int content,
                                                          size_t size) {
  return fegen::RightValue(
      fegen::RightValue::Expression::getInteger(content, size));
}

fegen::RightValue
fegen::RightValue::getFloatPoint(long double content, size_t size) {
  return fegen::RightValue(
      fegen::RightValue::Expression::getFloatPoint(content, size));
}
fegen::RightValue fegen::RightValue::getString(std::string content) {
  return fegen::RightValue(
      fegen::RightValue::Expression::getString(content));
}
fegen::RightValue
fegen::RightValue::getType(fegen::Type &content) {
  return fegen::RightValue(
      fegen::RightValue::Expression::getType(content));
}

fegen::RightValue fegen::RightValue::getList(
    std::vector<std::shared_ptr<fegen::RightValue::Expression>> &content) {
  return fegen::RightValue(
      fegen::RightValue::Expression::getList(content));
}
fegen::RightValue
fegen::RightValue::getLeftValue(fegen::Value *content) {
  return fegen::RightValue(
      fegen::RightValue::Expression::getLeftValue(content));
}

fegen::RightValue fegen::RightValue::getByExpr(
    std::shared_ptr<fegen::RightValue::Expression> expr) {
  assert(expr != nullptr);
  return fegen::RightValue(expr);
}

// class FegenValue
fegen::Value::Value(fegen::Type type, std::string name,
                              fegen::RightValue content)
    : type(std::move(type)), name(std::move(name)),
      content(std::move(content)) {}

fegen::Value::Value(const fegen::Value &rhs)
    : type(rhs.type), name(rhs.name), content(rhs.content) {}
fegen::Value::Value(fegen::Value &&rhs)
    : type(std::move(rhs.type)), name(std::move(rhs.name)),
      content(std::move(rhs.content)) {}

fegen::Value *fegen::Value::get(fegen::Type type,
                                          std::string name,
                                          RightValue content) {
  return new fegen::Value(std::move(type), std::move(name),
                               std::move(content));
}

fegen::Type &fegen::Value::getType() { return this->type; }

std::string fegen::Value::getName() { return this->name; }

void fegen::Value::setContent(fegen::RightValue content) {
  this->content = content;
}

fegen::RightValue::LiteralKind fegen::Value::getContentKind() {
  return this->content.getLiteralKind();
}

std::string fegen::Value::getContentString() {
  return this->content.toString();
}

std::string fegen::Value::getContentStringForTypedef() {
  return this->content.toStringForTypedef();
}

std::string fegen::Value::getContentStringForOpdef() {
  return this->content.toStringForOpdef();
}

std::string fegen::Value::getContentStringForCppKind() {
  return this->content.toStringForCppKind();
}

std::shared_ptr<fegen::RightValue::Expression>
fegen::Value::getExpr() {
  return this->content.getExpr();
}

fegen::ParserRule::ParserRule(std::string content, fegen::ParserNode *src,
                            antlr4::ParserRuleContext *ctx)
    : content(content), src(src), ctx(ctx) {}

fegen::ParserRule *fegen::ParserRule::get(std::string content,
                                        fegen::ParserNode *src,
                                        antlr4::ParserRuleContext *ctx) {
  return new fegen::ParserRule(content, src, ctx);
}

llvm::StringRef fegen::ParserRule::getContent() { return this->content; }

bool fegen::ParserRule::addInput(fegen::Value input) {
  auto name = input.getName();
  if (this->inputs.count(name) == 0) {
    return false;
  }
  this->inputs.insert({name, new fegen::Value(input)});
  return true;
}

bool fegen::ParserRule::addReturn(fegen::Value output) {
  auto name = output.getName();
  if (this->returns.count(name) == 0) {
    return false;
  }
  this->returns.insert({name, new fegen::Value(output)});
  return true;
}

void fegen::ParserRule::setSrc(ParserNode *src) { this->src = src; }

fegen::ParserNode::ParserNode(std::vector<fegen::ParserRule *> &&rules,
                            antlr4::ParserRuleContext *ctx,
                            fegen::ParserNode::NodeType ntype)
    : rules(rules), ctx(ctx), ntype(ntype) {}

fegen::ParserNode *fegen::ParserNode::get(std::vector<fegen::ParserRule *> rules,
                                        antlr4::ParserRuleContext *ctx,
                                        fegen::ParserNode::NodeType ntype) {
  return new fegen::ParserNode(std::move(rules), ctx, ntype);
}
fegen::ParserNode *fegen::ParserNode::get(antlr4::ParserRuleContext *ctx,
                                        fegen::ParserNode::NodeType ntype) {
  std::vector<fegen::ParserRule *> rules;
  return new fegen::ParserNode(std::move(rules), ctx, ntype);
}

void fegen::ParserNode::addFegenRule(fegen::ParserRule *rule) {
  this->rules.push_back(rule);
}

fegen::ParserNode::~ParserNode() {
  for (auto rule : this->rules) {
    delete rule;
  }
}

void fegen::Manager::setModuleName(std::string name) {
  this->moduleName = name;
}

std::string getChildrenText(antlr4::tree::ParseTree *ctx) {
  std::string ruleText;
  for (auto child : ctx->children) {
    if (antlr4::tree::TerminalNode::is(child)) {
      ruleText.append(child->getText()).append(" ");
    } else {
      ruleText.append(getChildrenText(child)).append(" ");
    }
  }
  return ruleText;
}

fegen::Manager::Manager() {}

namespace fegen {

class Emitter {
private:
  std::ostream &stream;
  int tabCount;
  bool isNewLine;

public:
  Emitter() = delete;
  Emitter(Emitter &) = delete;
  Emitter(Emitter &&) = delete;
  Emitter(std::ostream &stream)
      : stream(stream), tabCount(0), isNewLine(true) {}
  void tab() { tabCount++; }

  void shiftTab() {
    tabCount--;
    if (tabCount < 0) {
      tabCount = 0;
    }
  }

  void newLine() {
    this->stream << std::endl;
    isNewLine = true;
  }

  std::ostream &operator<<(std::string s) {
    if (this->isNewLine) {
      for (int i = 0; i <= (this->tabCount - 1); i++) {
        this->stream << '\t';
      }
      this->isNewLine = false;
    }
    this->stream << s;
    return this->stream;
  }
};

class StmtGenerator : FegenParserBaseVisitor {
private:
  Manager &manager;
  Emitter &emitter;

public:
  StmtGenerator(Emitter &emitter)
      : manager(Manager::getManager()), emitter(emitter) {}
  std::any visitVarDeclStmt(FegenParser::VarDeclStmtContext *ctx) override {
    auto var = manager.getStmtContent<Value *>(ctx->identifier());
    switch (var->getType().getTypeKind()) {
    case fegen::Type::TypeKind::CPP: {
      this->emitter << var->getType().toStringForCppKind() << " "
                    << var->getName();
      if (ctx->expression()) {
        auto expr = this->manager.getStmtContent<RightValue::Expression *>(
            ctx->expression());
        this->emitter << " = " << expr->toStringForCppKind();
      }
      this->emitter << ";";
      this->emitter.newLine();
      break;
    }
    case fegen::Type::TypeKind::ATTRIBUTE: {
      break;
    }
    case fegen::Type::TypeKind::OPERAND: {
      break;
    }
    }
    return nullptr;
  }

  std::any visitAssignStmt(FegenParser::AssignStmtContext *ctx) override {}

  std::any visitFunctionCall(FegenParser::FunctionCallContext *ctx) override {}

  std::any visitOpInvokeStmt(FegenParser::OpInvokeStmtContext *ctx) override {}

  std::any visitIfStmt(FegenParser::IfStmtContext *ctx) override {}

  std::any visitForStmt(FegenParser::ForStmtContext *ctx) override {}
};

} // namespace fegen

void fegen::Manager::emitG4() {
  std::ofstream fileStream;
  fileStream.open(this->moduleName + ".g4");
  fegen::Emitter emitter(fileStream);
  emitter << "grammar " << this->moduleName << ";";
  emitter.newLine();
  for (auto node_pair : this->nodeMap) {
    auto nodeName = node_pair.first;
    auto node = node_pair.second;
    emitter << nodeName;
    emitter.newLine();
    emitter.tab();
    auto ruleCount = node->rules.size();
    if (ruleCount > 0) {
      emitter << ": " << getChildrenText(node->rules[0]->ctx);
      emitter.newLine();
      for (size_t i = 1; i <= ruleCount - 1; i++) {
        emitter << "| " << getChildrenText(node->rules[i]->ctx);
        emitter.newLine();
      }
      emitter << ";" << std::endl;
    }
    emitter.shiftTab();
    emitter.newLine();
  }
}

// TODO: emit to file
void fegen::Manager::emitTypeDefination() {
  std::ofstream fileStream;
  fileStream.open(this->moduleName + "Types.td");
  fegen::Emitter emitter(fileStream);
  // file head
  std::string mn(this->moduleName);
  std::transform(mn.begin(), mn.end(), mn.begin(), ::toupper);
  emitter << "#ifndef " << mn << "_TYPE_TD";
  emitter.newLine();
  emitter << "#define " << mn << "_TYPE_TD";
  emitter << "\n";
  emitter.newLine();

  // include files
  emitter << "include \"mlir/IR/AttrTypeBase.td\"";
  emitter.newLine();
  emitter << "include \"" << this->moduleName << "Dialect.td\"";
  emitter << "\n";
  emitter.newLine();
  // Type class defination
  std::string typeClassName = this->moduleName + "Type";
  emitter << "class " << typeClassName
          << "<string typename, string typeMnemonic, list<Trait> traits = []>";
  emitter.tab();
  emitter << ": TypeDef<Toy_Dialect, typename, traits> {";
  emitter.newLine();
  emitter << "let mnemonic = typeMnemonic;";
  emitter.newLine();
  emitter.shiftTab();
  emitter << "}" << std::endl;
  emitter.newLine();

  for (auto pair : this->typeDefMap) {
    auto tyDef = pair.second;
    if (!tyDef->isCustome()) {
      continue;
    }
    auto typeName = pair.first;
    // head of typedef
    emitter << "def " << typeName << " : " << typeClassName << "<\"" << typeName
            << "\", \"" << tyDef->getMnemonic() << "\"> {";
    emitter.newLine();
    emitter.tab();
    // summary
    emitter << "let summary = \"This is generated by buddy fegen.\";";
    emitter.newLine();
    // description
    emitter << "let description = [{ This is generated by buddy fegen. }];";
    emitter.newLine();
    // parameters
    emitter << "let parameters = ( ins";
    emitter.newLine();
    emitter.tab();
    for (size_t i = 0; i <= tyDef->getParameters().size() - 1; i++) {
      auto param = tyDef->getParameters()[i];
      auto &paramTy = param->getType();
      auto paramName = param->getName();
      auto paramTyStr = paramTy.toStringForTypedef();
      emitter << paramTyStr << ":" << "$" << paramName;
      if (i != tyDef->getParameters().size() - 1) {
        emitter << ", ";
      }
      emitter.newLine();
    }
    emitter.shiftTab();
    emitter << ");";
    emitter.newLine();
    // assemblyFormat
    // TODO: handle list, Type ...
    emitter << "let assemblyFormat = [{";
    emitter.newLine();
    emitter.tab();
    emitter << "`<` ";
    for (size_t i = 0; i <= tyDef->getParameters().size() - 1; i++) {
      auto param = tyDef->getParameters()[i];
      auto paramName = param->getName();
      emitter << "$" << paramName << " ";
      if (i != tyDef->getParameters().size() - 1) {
        emitter << "`x` ";
      }
    }
    emitter << "`>`";
    emitter.shiftTab();
    emitter.newLine();
    emitter << "}];";
    emitter.newLine();
    emitter.shiftTab();
    emitter << "}";
    emitter.newLine();
  }
  emitter.shiftTab();
  emitter << "\n";
  emitter << "#endif // " << mn << "_TYPE_TD";
  fileStream.close();
}

void fegen::Manager::emitOpDefination() {
  std::ofstream fileStream;
  fileStream.open(this->moduleName + "Ops.td");
  fegen::Emitter emitter(fileStream);

  // file head
  std::string mn(this->moduleName);
  std::transform(mn.begin(), mn.end(), mn.begin(), ::toupper);
  emitter << "#ifndef " << mn << "_OPS_TD";
  emitter.newLine();
  emitter << "#define " << mn << "_OPS_TD";
  emitter << "\n";
  emitter.newLine();

  // TODO: custome include files
  // include
  emitter << "include \"mlir/IR/BuiltinAttributes.td\"";
  emitter.newLine();
  emitter << "include \"mlir/IR/BuiltinTypes.td\"";
  emitter.newLine();
  emitter << "include \"mlir/IR/CommonAttrConstraints.td\"";
  emitter.newLine();
  emitter << "include \"" << this->moduleName << "Dialect.td\"";
  emitter.newLine();
  emitter << "include \"" << this->moduleName << "Types.td\"";
  emitter.newLine();
  emitter << "\n";

  // op class defination
  std::string classname = this->moduleName + "Op";
  emitter << "class " << classname
          << "<string mnemonic, list<Trait> traits = []>:";
  emitter.newLine();
  emitter.tab();
  emitter << "Op<Toy_Dialect, mnemonic, traits>;";
  emitter << "\n";
  emitter.shiftTab();
  emitter.newLine();

  // op definations
  for (auto pair : this->operationMap) {
    auto opName = pair.first;
    auto opDef = pair.second;
    // head of def
    emitter << "def " << opName << " : " << classname << "<\"" << opName
            << "\", [Pure]> {";
    emitter.newLine();
    emitter.tab();
    // summary and description
    emitter << "let summary = \"This is generated by buddy fegen.\";";
    emitter.newLine();
    emitter << "let description = [{This is generated by buddy fegen.}];";
    emitter.newLine();
    // arguments
    emitter << "let arguments = ( ins ";
    emitter.newLine();
    emitter.tab();
    for (auto param : opDef->getArguments()) {
      auto paramTyStr = param->getType().toStringForOpdef();
      auto paramName = param->getName();
      emitter << paramTyStr << " : $" << paramName;
      emitter.newLine();
    }
    emitter.shiftTab();
    emitter << ");";
    emitter.newLine();
    // results
    emitter << "let results = (outs ";
    emitter.newLine();
    emitter.tab();
    for (auto param : opDef->getArguments()) {
      auto paramTyStr = param->getType().toStringForOpdef();
      auto paramName = param->getName();
      emitter << paramTyStr << " : $" << paramName;
      emitter.newLine();
    }
    emitter.shiftTab();
    emitter << ");";
    emitter.newLine();
    // end of def
    emitter.shiftTab();
    emitter << "}";
    emitter.newLine();
  }

  // end of file
  emitter << "\n";
  emitter << "#endif // " << mn << "_DIALECT_TD";
  fileStream.close();
}

void fegen::Manager::emitDialectDefination() {
  std::ofstream fileStream;
  fileStream.open(this->moduleName + "Dialect.td");
  fegen::Emitter emitter(fileStream);

  // file head
  std::string mn(this->moduleName);
  std::transform(mn.begin(), mn.end(), mn.begin(), ::toupper);
  emitter << "#ifndef " << mn << "_DIALECT_TD";
  emitter.newLine();
  emitter << "#define " << mn << "_DIALECT_TD";
  emitter << "\n";
  emitter.newLine();

  // include
  emitter << "include \"mlir/IR/OpBase.td\"";
  emitter << "\n";
  emitter.newLine();

  // dialect defination
  emitter << "def " << this->moduleName << "_Dialect : Dialect {";
  emitter.newLine();
  emitter.tab();
  emitter << "let name = \"" << this->moduleName << "\";";
  emitter.newLine();
  emitter << "let summary = \"This is generated by buddy fegen.\";";
  emitter.newLine();
  emitter << "let description = [{This is generated by buddy fegen.}];";
  emitter.newLine();
  emitter << "let cppNamespace = \"::mlir::" << this->moduleName << "\";";
  emitter.newLine();
  emitter << "let extraClassDeclaration = [{";
  emitter.newLine();
  emitter.tab();
  emitter << "/// Register all types.";
  emitter.newLine();
  emitter << "void registerTypes();";
  emitter.newLine();
  emitter.shiftTab();
  emitter << "}];";
  emitter.newLine();
  emitter.shiftTab();
  emitter << "}";
  emitter.newLine();

  // end of file
  emitter << "#endif // " << mn << "_DIALECT_TD";
  fileStream.close();
}

void fegen::Manager::emitTdFiles() {
  this->emitDialectDefination();
  this->emitTypeDefination();
  this->emitOpDefination();
}

void fegen::Manager::initbuiltinTypes() {
  // placeholder type
  auto placeholderTypeDefination = fegen::TypeDefination::get(
      "fegen_builtin", FEGEN_PLACEHOLDER, {}, nullptr, false);
  this->typeDefMap.insert({FEGEN_PLACEHOLDER, placeholderTypeDefination});

  // Type
  this->typeDefMap.insert(
      {FEGEN_TYPE, fegen::TypeDefination::get("fegen_builtin", FEGEN_TYPE,
                                                   {}, nullptr, false)});

  // TypeTemplate
  this->typeDefMap.insert(
      {FEGEN_TYPETEMPLATE,
       fegen::TypeDefination::get("fegen_builtin", FEGEN_TYPETEMPLATE, {},
                                       nullptr, false)});

  // recursive define Integer Type
  // Integer<Integer<Integer<...>>>
  auto intTypeDefination = fegen::TypeDefination::get(
      "fegen_builtin", FEGEN_INTEGER, {}, nullptr, false);
  auto intType = fegen::Type(
      fegen::Type::TypeKind::CPP,
      {fegen::Value::get(fegen::Type::getPlaceHolder(), "size",
                              fegen::RightValue::getPlaceHolder())},
      intTypeDefination, false);
  // parameters of Integer is int32(Integer<32>)
  intTypeDefination->parameters.push_back(fegen::Value::get(
      intType, "size", fegen::RightValue::getPlaceHolder()));
  this->typeDefMap.insert({FEGEN_INTEGER, intTypeDefination});

  // FloatPoint
  this->typeDefMap.insert(
      {FEGEN_FLOATPOINT,
       fegen::TypeDefination::get(
           "fegen_builtin", FEGEN_FLOATPOINT,
           {fegen::Value::get(fegen::Type::getInt32Type(), "size",
                                   fegen::RightValue::getPlaceHolder())},
           nullptr, false)});

  // Char
  this->typeDefMap.insert(
      {FEGEN_CHAR, fegen::TypeDefination::get("fegen_builtin", FEGEN_CHAR,
                                                   {}, nullptr, false)});

  // String
  this->typeDefMap.insert(
      {FEGEN_STRING, fegen::TypeDefination::get(
                         "fegen_builtin", FEGEN_STRING, {}, nullptr, false)});

  // Vector
  this->typeDefMap.insert(
      {FEGEN_VECTOR,
       fegen::TypeDefination::get(
           "fegen_builtin", FEGEN_VECTOR,
           {fegen::Value::get(fegen::Type::getInt32Type(), "size",
                                   fegen::RightValue::getPlaceHolder()),
            fegen::Value::get(fegen::Type::getMetaType(),
                                   "elementType",
                                   fegen::RightValue::getPlaceHolder())},
           nullptr, false)});

  // List (this should be ahead of Tensor and Any Type defination)
  this->typeDefMap.insert(
      {FEGEN_LIST, fegen::TypeDefination::get(
                       "fegen_builtin", FEGEN_LIST,
                       {fegen::Value::get(
                           fegen::Type::getMetaType(), "elementType",
                           fegen::RightValue::getPlaceHolder())},
                       nullptr, false)});

  // Tensor
  this->typeDefMap.insert(
      {FEGEN_TENSOR,
       fegen::TypeDefination::get(
           "fegen_builtin", FEGEN_TENSOR,
           {fegen::Value::get(
                fegen::Type::getListType(fegen::Type::getInt32Type()),
                "shape", fegen::RightValue::getPlaceHolder()),
            fegen::Value::get(fegen::Type::getMetaType(),
                                   "elementType",
                                   fegen::RightValue::getPlaceHolder())},
           nullptr, false)});

  // Optional
  this->typeDefMap.insert(
      {FEGEN_OPTINAL, fegen::TypeDefination::get(
                          "fegen_builtin", FEGEN_OPTINAL,
                          {fegen::Value::get(
                              fegen::Type::getMetaType(), "elementType",
                              fegen::RightValue::getPlaceHolder())},
                          nullptr, false)});

  // Any
  this->typeDefMap.insert(
      {FEGEN_ANY,
       fegen::TypeDefination::get(
           "fegen_builtin", FEGEN_ANY,
           {fegen::Value::get(
               fegen::Type::getListType(fegen::Type::getMetaType()),
               "elementType", fegen::RightValue::getPlaceHolder())},
           nullptr, false)});
}

fegen::TypeDefination *
fegen::Manager::getTypeDefination(std::string name) {
  return this->typeDefMap[name];
}

bool fegen::Manager::addTypeDefination(fegen::TypeDefination *tyDef) {
  if (this->typeDefMap.count(tyDef->name) != 0) {
    return false;
  }
  this->typeDefMap[tyDef->name] = tyDef;
  return true;
}

fegen::Operation *
fegen::Manager::getOperationDefination(std::string name) {
  return this->operationMap[name];
}

bool fegen::Manager::addOperationDefination(fegen::Operation *opDef) {
  if (this->operationMap.count(opDef->getOpName()) != 0) {
    return false;
  }
  this->operationMap[opDef->getOpName()] = opDef;
  return true;
}

void fegen::Manager::addStmtContent(antlr4::ParserRuleContext *ctx,
                                         std::any content) {
  this->stmtContentMap.insert({ctx, content});
}

fegen::Manager &fegen::Manager::getManager() {
  static fegen::Manager fmg;
  return fmg;
}

fegen::Manager::~Manager() {
  // release nodes
  for (auto node_pair : this->nodeMap) {
    delete node_pair.second;
  }
}

fegen::Type fegen::inferenceType(
    std::vector<std::shared_ptr<fegen::RightValue::Expression>> operands,
    fegen::FegenOperator op) {
  // TODO: infer type
  return fegen::Type::getInt32Type();
}

namespace fegen {

// class StmtVisitor : public FegenParserBaseVisitor{
// public:
// };

}
void fegen::Manager::emitBuiltinFunction() {
  Emitter emitter(std::cout);
  for (auto function_pair : this->functionMap) {
    auto functionName = function_pair.first;
    auto function = function_pair.second;
    auto paraList = function->getInputTypeList();
    emitter << function->getReturnType()->toStringForTypedef() << " "
            << functionName << "(";
    for (auto para : paraList) {
      emitter << para->getContentStringForTypedef() << " " << para->getName();
      if (para != paraList.back())
        emitter << ", ";
    }
    emitter << "){";
    emitter.newLine();
    emitter.tab();
    // TODO::function body

    emitter.shiftTab();
    emitter.newLine();
    emitter << "}";
  }
}