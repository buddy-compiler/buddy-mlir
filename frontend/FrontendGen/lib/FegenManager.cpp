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

fegen::FegenFunction::FegenFunction(std::string name,
                                    std::vector<FegenValue *> &&inputTypeList,
                                    FegenType *returnType)
    : name(name), inputTypeList(inputTypeList), returnType(returnType) {}

fegen::FegenFunction *
fegen::FegenFunction::get(std::string name,
                          std::vector<FegenValue *> inputTypeList,
                          FegenType *returnType) {
  return new fegen::FegenFunction(name, std::move(inputTypeList), returnType);
}
std::string fegen::FegenFunction::getName() { return this->name; }

std::vector<fegen::FegenValue *> &fegen::FegenFunction::getInputTypeList() {
  return this->inputTypeList;
}

fegen::FegenValue *fegen::FegenFunction::getInputTypeList(size_t i) {
  return this->inputTypeList[i];
}

fegen::FegenType *fegen::FegenFunction::getReturnType() {
  return this->returnType;
}

fegen::FegenOperation::FegenOperation(std::string dialectName,
                                      std::string operationName,
                                      std::vector<FegenValue *> &&arguments,
                                      std::vector<FegenValue *> &&results,
                                      fegen::FegenParser::BodySpecContext *ctx)
    : dialectName(dialectName), arguments(arguments), results(results),
      ctx(ctx) {}

void fegen::FegenOperation::setOpName(std::string name) {
  this->operationName = name;
}
std::string fegen::FegenOperation::getOpName() { return this->operationName; }

std::vector<fegen::FegenValue *> &fegen::FegenOperation::getArguments() {
  return this->arguments;
}

fegen::FegenValue *fegen::FegenOperation::getArguments(size_t i) {
  return this->arguments[i];
}

std::vector<fegen::FegenValue *> &fegen::FegenOperation::getResults() {
  return this->results;
}

fegen::FegenValue *fegen::FegenOperation::getResults(size_t i) {
  return this->results[i];
}

fegen::FegenOperation *fegen::FegenOperation::get(
    std::string operationName, std::vector<FegenValue *> arguments,
    std::vector<FegenValue *> results, FegenParser::BodySpecContext *ctx) {
  return new fegen::FegenOperation(fegen::FegenManager::getManager().moduleName,
                                   operationName, std::move(arguments),
                                   std::move(results), ctx);
}

// class FegenType

/// @brief get name of Type Instance by jointsing template name and parameters,
/// for example: Integer + 32 --> Integer<32>
/// @return joint name
std::string jointTypeName(std::string templateName,
                          const std::vector<fegen::FegenValue *> &parameters) {
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

fegen::FegenType::FegenType(TypeKind kind, std::string name,
                            std::vector<FegenValue *> parameters,
                            FegenTypeDefination *tyDef, int typeLevel)
    : kind(kind), typeName(name), parameters(std::move(parameters)),
      typeDefine(tyDef), typeLevel(typeLevel) {}

fegen::FegenType::FegenType(fegen::FegenType::TypeKind kind,
                            std::vector<FegenValue *> parameters,
                            FegenTypeDefination *tyDef, int typeLevel)
    : kind(kind), typeName(jointTypeName(tyDef->getName(), parameters)),
      parameters(std::move(parameters)), typeDefine(tyDef),
      typeLevel((typeLevel)) {}

fegen::FegenType::FegenType(const fegen::FegenType &fty)
    : kind(fty.kind), typeName(fty.typeName), typeDefine(fty.typeDefine),
      typeLevel(fty.typeLevel) {
  // deep copy parameters
  for (auto paramPtr : fty.parameters) {
    this->parameters.push_back(new fegen::FegenValue(*paramPtr));
  }
}

fegen::FegenType::FegenType(fegen::FegenType &&fty)
    : kind(fty.kind), typeName(std::move(fty.typeName)),
      parameters(std::move(fty.parameters)), typeDefine(fty.typeDefine),
      typeLevel(fty.typeLevel) {}

fegen::FegenType::TypeKind fegen::FegenType::getTypeKind() {
  return this->kind;
}

void fegen::FegenType::setTypeKind(fegen::FegenType::TypeKind kind) {
  this->kind = kind;
}

std::vector<fegen::FegenValue *> &fegen::FegenType::getParameters() {
  return this->parameters;
}

fegen::FegenValue *fegen::FegenType::getParameters(size_t i) {
  return this->parameters[i];
}

void fegen::FegenType::setParameters(std::vector<fegen::FegenValue *> &params) {
  this->parameters = params;
  // set parameters and level up!
  this->typeLevel++;
}

fegen::FegenTypeDefination *fegen::FegenType::getTypeDefination() {
  return this->typeDefine;
}

void fegen::FegenType::setTypeDefination(fegen::FegenTypeDefination *tyDef) {
  this->typeDefine = tyDef;
}

std::string fegen::FegenType::getTypeName() { return this->typeName; }

int fegen::FegenType::getTypeLevel() { return this->typeLevel; }

bool fegen::FegenType::isSameType(fegen::FegenType *type1,
                                  fegen::FegenType *type2) {
  if (type1->getTypeName() == type2->getTypeName())
    return true;
  else
    return false;
}

std::string fegen::FegenType::toStringForTypedef() {
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

std::string fegen::FegenType::toStringForOpdef() {
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

std::string fegen::FegenType::toStringForCppKind() {
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

fegen::FegenType::~FegenType() {
  for (auto p : this->parameters) {
    delete p;
  }
}

fegen::FegenType fegen::FegenType::getPlaceHolder() {
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, {},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_PLACEHOLDER),
      0);
}

fegen::FegenType fegen::FegenType::getMetaType() {
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, {},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_TYPE), 2);
}

fegen::FegenType fegen::FegenType::getMetaTemplateType() {
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, {},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_TYPETEMPLATE),
      1);
}

fegen::FegenType fegen::FegenType::getInt32Type() {
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, "int",
      {fegen::FegenValue::get(fegen::FegenType::getPlaceHolder(), "size",
                              fegen::FegenRightValue::getPlaceHolder())},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_INTEGER), 3);
}

fegen::FegenType fegen::FegenType::getFloatType() {
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, "float",
      {fegen::FegenValue::get(fegen::FegenType::getInt32Type(), "size",
                              fegen::FegenRightValue::getInteger(32))},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_FLOATPOINT), 3);
}

fegen::FegenType fegen::FegenType::getDoubleType() {
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, "double",
      {fegen::FegenValue::get(fegen::FegenType::getInt32Type(), "size",
                              fegen::FegenRightValue::getInteger(64))},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_FLOATPOINT), 3);
}

fegen::FegenType fegen::FegenType::getBoolType() {
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, "bool",
      {fegen::FegenValue::get(fegen::FegenType::getInt32Type(), "size",
                              fegen::FegenRightValue::getInteger(1))},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_INTEGER), 3);
}

fegen::FegenType fegen::FegenType::getIntegerType(fegen::FegenValue *size) {
  if (size->getContent<int>() == 32)
    return fegen::FegenType::getInt32Type();
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, {size},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_INTEGER), 3);
}

fegen::FegenType fegen::FegenType::getFloatPointType(fegen::FegenValue *size) {
  if (size->getContent<int>() == 32) {
    return fegen::FegenType::getFloatType();
  } else if (size->getContent<int>() == 64) {
    return fegen::FegenType::getDoubleType();
  }
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, {size},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_FLOATPOINT), 3);
}

fegen::FegenType fegen::FegenType::getCharType() {
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, {},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_CHAR), 3);
}

fegen::FegenType fegen::FegenType::getStringType() {
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, {},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_STRING), 3);
}

fegen::FegenType fegen::FegenType::getVectorType(fegen::FegenValue *size,
                                                 fegen::FegenType elementType) {
  assert(elementType.typeLevel == 3);
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP,
      {size,
       fegen::FegenValue::get(fegen::FegenType::getMetaType(), "elementType",
                              fegen::FegenRightValue::getType(elementType))},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_VECTOR),
      elementType.typeLevel);
}

fegen::FegenType fegen::FegenType::getTensorType(fegen::FegenValue *shape,
                                                 fegen::FegenType elementType) {
  assert(elementType.typeLevel == 3);
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP,
      {shape,
       fegen::FegenValue::get(fegen::FegenType::getMetaType(), "elementType",
                              fegen::FegenRightValue::getType(elementType))},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_TENSOR),
      elementType.typeLevel);
}

// List<elementType>
fegen::FegenType fegen::FegenType::getListType(fegen::FegenType elementType) {
  assert(elementType.typeLevel == 2 || elementType.typeLevel == 3);
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP,
      {fegen::FegenValue::get(
          elementType.typeLevel == 2 ? fegen::FegenType::getMetaTemplateType()
                                     : fegen::FegenType::getMetaType(),
          "elementType", fegen::FegenRightValue::getType(elementType))},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_LIST),
      elementType.typeLevel);
}

// Optional<elementType>
fegen::FegenType
fegen::FegenType::getOptionalType(fegen::FegenType elementType) {
  assert(elementType.typeLevel == 2 || elementType.typeLevel == 3);
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP,
      {fegen::FegenValue::get(
          elementType.typeLevel == 2 ? fegen::FegenType::getMetaTemplateType()
                                     : fegen::FegenType::getMetaType(),
          "elementType", fegen::FegenRightValue::getType(elementType))},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_OPTINAL),
      elementType.typeLevel);
}

// Any<elementType1, elementType2, ...>
fegen::FegenType
fegen::FegenType::getAnyType(std::vector<fegen::FegenType> elementTypes) {
  std::vector<fegen::FegenValue *> p_elemTy;
  int i = 0;
  std::string name("elementType_");
  auto tyLevel = elementTypes[0].typeLevel;
  assert(tyLevel == 2 || tyLevel == 3);
  auto tyty = tyLevel == 2 ? fegen::FegenType::getMetaTemplateType()
                           : fegen::FegenType::getMetaType();
  for (auto &ty : elementTypes) {
    assert(ty.typeLevel == tyLevel);
    p_elemTy.push_back(fegen::FegenValue::get(
        tyty, name + std::to_string(i), fegen::FegenRightValue::getType(ty)));
    i++;
  }
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, p_elemTy,
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_ANY), tyLevel);
}

fegen::FegenType fegen::FegenType::getIntegerTemplate() {
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, {},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_INTEGER), 2);
}

fegen::FegenType fegen::FegenType::getFloatPointTemplate() {
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, {},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_FLOATPOINT), 2);
}

fegen::FegenType
fegen::FegenType::getInstanceType(fegen::FegenTypeDefination *typeDefination,
                                  std::vector<fegen::FegenValue *> parameters) {
  return fegen::FegenType(fegen::FegenType::TypeKind::CPP, parameters,
                          typeDefination, 3);
}
fegen::FegenType
fegen::FegenType::getTemplateType(fegen::FegenTypeDefination *typeDefination) {
  return fegen::FegenType(fegen::FegenType::TypeKind::CPP, {}, typeDefination,
                          2);
}

// class FegenTypeDefination
fegen::FegenTypeDefination::FegenTypeDefination(
    std::string dialectName, std::string name,
    std::vector<fegen::FegenValue *> parameters,
    FegenParser::TypeDefinationDeclContext *ctx, bool ifCustome)
    : dialectName(std::move(dialectName)), name(std::move(name)),
      parameters(std::move(parameters)), ctx(ctx), ifCustome(ifCustome) {}

fegen::FegenTypeDefination *
fegen::FegenTypeDefination::get(std::string dialectName, std::string name,
                                std::vector<fegen::FegenValue *> parameters,
                                FegenParser::TypeDefinationDeclContext *ctx,
                                bool ifCustome) {
  return new fegen::FegenTypeDefination(std::move(dialectName), std::move(name),
                                        std::move(parameters), ctx, ifCustome);
}

std::string fegen::FegenTypeDefination::getDialectName() {
  return this->dialectName;
}

void fegen::FegenTypeDefination::setDialectName(std::string name) {
  this->dialectName = name;
}

std::string fegen::FegenTypeDefination::getName() { return this->name; }

std::string fegen::FegenTypeDefination::getMnemonic() {
  if (this->mnemonic.empty()) {
    this->mnemonic = this->name;
    std::transform(this->mnemonic.begin(), this->mnemonic.end(),
                   this->mnemonic.begin(), ::tolower);
  }
  return this->mnemonic;
}

void fegen::FegenTypeDefination::setName(std::string name) {
  this->name = name;
}

const std::vector<fegen::FegenValue *> &
fegen::FegenTypeDefination::getParameters() {
  return this->parameters;
}

fegen::FegenParser::TypeDefinationDeclContext *
fegen::FegenTypeDefination::getCtx() {
  return this->ctx;
}

void fegen::FegenTypeDefination::setCtx(
    FegenParser::TypeDefinationDeclContext *ctx) {
  this->ctx = ctx;
}

bool fegen::FegenTypeDefination::isCustome() { return this->ifCustome; }

// class Expression

fegen::FegenRightValue::Expression::Expression(bool ifTerminal,
                                               LiteralKind kind,
                                               FegenType &exprTy,
                                               bool isConstexpr)
    : ifTerminal(ifTerminal), kind(kind), exprType(exprTy),
      ifConstexpr(isConstexpr) {}

bool fegen::FegenRightValue::Expression::isTerminal() {
  return this->ifTerminal;
}

fegen::FegenRightValue::LiteralKind
fegen::FegenRightValue::Expression::getKind() {
  return this->kind;
}

fegen::FegenType &fegen::FegenRightValue::Expression::getType() {
  return this->exprType;
}

bool fegen::FegenRightValue::Expression::isConstexpr() {
  return this->ifConstexpr;
}

std::shared_ptr<fegen::FegenRightValue::ExpressionTerminal>
fegen::FegenRightValue::Expression::getPlaceHolder() {
  return std::make_shared<fegen::FegenRightValue::PlaceHolder>();
}

std::shared_ptr<fegen::FegenRightValue::ExpressionTerminal>
fegen::FegenRightValue::Expression::getInteger(long long int content,
                                               size_t size) {
  return std::make_shared<fegen::FegenRightValue::IntegerLiteral>(content,
                                                                  size);
}

std::shared_ptr<fegen::FegenRightValue::ExpressionTerminal>
fegen::FegenRightValue::Expression::getFloatPoint(long double content,
                                                  size_t size) {
  return std::make_shared<fegen::FegenRightValue::FloatPointLiteral>(content,
                                                                     size);
}

std::shared_ptr<fegen::FegenRightValue::ExpressionTerminal>
fegen::FegenRightValue::Expression::getString(std::string content) {
  return std::make_shared<fegen::FegenRightValue::StringLiteral>(content);
}

std::shared_ptr<fegen::FegenRightValue::ExpressionTerminal>
fegen::FegenRightValue::Expression::getType(fegen::FegenType &content) {
  return std::make_shared<fegen::FegenRightValue::TypeLiteral>(content);
}

std::shared_ptr<fegen::FegenRightValue::ExpressionTerminal>
fegen::FegenRightValue::Expression::getList(
    std::vector<std::shared_ptr<fegen::FegenRightValue::Expression>> &content) {
  return std::make_shared<fegen::FegenRightValue::ListLiteral>(content);
}

std::shared_ptr<fegen::FegenRightValue::ExpressionTerminal>
fegen::FegenRightValue::Expression::getLeftValue(fegen::FegenValue *content) {
  return std::make_shared<fegen::FegenRightValue::LeftValue>(content);
}

std::shared_ptr<fegen::FegenRightValue::OperatorCall>
fegen::FegenRightValue::Expression::binaryOperation(
    std::shared_ptr<fegen::FegenRightValue::Expression> lhs,
    std::shared_ptr<fegen::FegenRightValue::Expression> rhs, FegenOperator op) {
  FegenType resTy = fegen::inferenceType({lhs, rhs}, op);
  return std::make_shared<fegen::FegenRightValue::OperatorCall>(
      op, std::vector<std::shared_ptr<fegen::FegenRightValue::Expression>>{
              lhs, rhs});
}

std::shared_ptr<fegen::FegenRightValue::OperatorCall>
fegen::FegenRightValue::Expression::unaryOperation(
    std::shared_ptr<fegen::FegenRightValue::Expression> v, FegenOperator op) {
  FegenType resTy = fegen::inferenceType({v}, op);
  return std::make_shared<fegen::FegenRightValue::OperatorCall>(
      op, std::vector<std::shared_ptr<fegen::FegenRightValue::Expression>>{v});
}

// class ExpressionNode

fegen::FegenRightValue::ExpressionNode::ExpressionNode(LiteralKind kind,
                                                       FegenType exprTy,
                                                       bool ifConstexpr)
    : Expression(false, kind, exprTy, ifConstexpr) {}

std::string fegen::FegenRightValue::ExpressionNode::toString() {
  assert(FEGEN_NOT_IMPLEMENTED_ERROR);
}

std::string fegen::FegenRightValue::ExpressionNode::toStringForTypedef() {
  assert(FEGEN_NOT_IMPLEMENTED_ERROR);
}

std::string fegen::FegenRightValue::ExpressionNode::toStringForOpdef() {
  assert(FEGEN_NOT_IMPLEMENTED_ERROR);
}

std::string fegen::FegenRightValue::ExpressionNode::toStringForCppKind() {
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
    std::vector<std::shared_ptr<fegen::FegenRightValue::Expression>> &params) {
  for (auto param : params) {
    if (!param->isConstexpr()) {
      return false;
    }
  }
  return true;
}

// TODO: invoke methods of FegenFunction
fegen::FegenRightValue::FunctionCall::FunctionCall(
    fegen::FegenFunction *func,
    std::vector<std::shared_ptr<fegen::FegenRightValue::Expression>> params)
    : ExpressionNode(fegen::FegenRightValue::LiteralKind::FUNC_CALL,
                     fegen::FegenType::getInt32Type(),
                     isFuncParamsAllConstant(params)),
      func(func), params(std::move(params)) {}

std::string fegen::FegenRightValue::FunctionCall::toString() {
  return "FunctionCall::toString";
}

std::string fegen::FegenRightValue::FunctionCall::toStringForTypedef() {
  return "FunctionCall::toStringForTypedef";
}

std::string fegen::FegenRightValue::FunctionCall::toStringForOpdef() {
  return "FunctionCall::toStringForOpdef";
}

std::string fegen::FegenRightValue::FunctionCall::toStringForCppKind() {
  return "FunctionCall::toStringForCppKind";
}

std::any fegen::FegenRightValue::FunctionCall::getContent() { return this; }

// class OperationCall
fegen::FegenRightValue::OperationCall::OperationCall(
    fegen::FegenOperation *op,
    std::vector<std::shared_ptr<fegen::FegenRightValue::Expression>> params)
    : ExpressionNode(fegen::FegenRightValue::LiteralKind::OPERATION_CALL,
                     fegen::FegenType::getInt32Type(),
                     isFuncParamsAllConstant(params)),
      op(op), params(std::move(params)) {}

std::string fegen::FegenRightValue::OperationCall::toString() {
  return "OperationCall::toString";
}

std::string fegen::FegenRightValue::OperationCall::toStringForTypedef() {
  return "OperationCall::toStringForTypedef";
}

std::string fegen::FegenRightValue::OperationCall::toStringForOpdef() {
  return "OperationCall::toStringForOpdef";
}

std::string fegen::FegenRightValue::OperationCall::toStringForCppKind() {
  return "OperationCall::toStringForCppKind";
}

std::any fegen::FegenRightValue::OperationCall::getContent() { return this; }

// class OperatorCall
fegen::FegenRightValue::OperatorCall::OperatorCall(
    fegen::FegenOperator op,
    std::vector<std::shared_ptr<fegen::FegenRightValue::Expression>> params)
    : ExpressionNode(fegen::FegenRightValue::LiteralKind::OPERATION_CALL,
                     fegen::inferenceType(params, op),
                     isFuncParamsAllConstant(params)),
      op(op), params(std::move(params)) {}

std::string fegen::FegenRightValue::OperatorCall::toString() {
  return "OperatorCall::toString";
}

std::string fegen::FegenRightValue::OperatorCall::toStringForTypedef() {
  return "OperatorCall::toStringForTypedef";
}

std::string fegen::FegenRightValue::OperatorCall::toStringForOpdef() {
  return "OperatorCall::toStringForOpdef";
}

std::string fegen::FegenRightValue::OperatorCall::toStringForCppKind() {
  return "OperatorCall::toStringForCppKind";
}

std::any fegen::FegenRightValue::OperatorCall::getContent() { return this; }

// class ExpressionTerminal
fegen::FegenRightValue::ExpressionTerminal::ExpressionTerminal(
    fegen::FegenRightValue::LiteralKind kind, FegenType exprTy,
    bool ifConstexpr)
    : Expression(true, kind, exprTy, ifConstexpr) {}

std::string fegen::FegenRightValue::ExpressionTerminal::toString() {
  assert(FEGEN_NOT_IMPLEMENTED_ERROR);
}

std::string fegen::FegenRightValue::ExpressionTerminal::toStringForTypedef() {
  assert(FEGEN_NOT_IMPLEMENTED_ERROR);
}

std::string fegen::FegenRightValue::ExpressionTerminal::toStringForOpdef() {
  assert(FEGEN_NOT_IMPLEMENTED_ERROR);
}

std::string fegen::FegenRightValue::ExpressionTerminal::toStringForCppKind() {
  assert(FEGEN_NOT_IMPLEMENTED_ERROR);
}

// class PlaceHolder
fegen::FegenRightValue::PlaceHolder::PlaceHolder()
    : ExpressionTerminal(fegen::FegenRightValue::LiteralKind::MONOSTATE,
                         fegen::FegenType::getPlaceHolder(), true) {}

std::any fegen::FegenRightValue::PlaceHolder::getContent() {
  return std::monostate();
}

std::string fegen::FegenRightValue::PlaceHolder::toString() { return ""; }

// class IntegerLiteral
fegen::FegenRightValue::IntegerLiteral::IntegerLiteral(int content)
    : ExpressionTerminal(fegen::FegenRightValue::LiteralKind::INT,
                         fegen::FegenType::getInt32Type(), true),
      content(content) {}

fegen::FegenRightValue::IntegerLiteral::IntegerLiteral(long long int content,
                                                       size_t size)
    : ExpressionTerminal(
          fegen::FegenRightValue::LiteralKind::INT,
          fegen::FegenType::getIntegerType(fegen::FegenValue::get(
              fegen::FegenType::getInt32Type(), "size",
              fegen::FegenRightValue::getByExpr(
                  std::make_shared<fegen::FegenRightValue::IntegerLiteral>(
                      size)))),
          true),
      content(content) {}

std::any fegen::FegenRightValue::IntegerLiteral::getContent() {
  return this->content;
}

std::string fegen::FegenRightValue::IntegerLiteral::toString() {
  return std::to_string(this->content);
}

// class FloatPointLiteral
fegen::FegenRightValue::FloatPointLiteral::FloatPointLiteral(
    long double content, size_t size)
    : ExpressionTerminal(
          fegen::FegenRightValue::LiteralKind::FLOAT,
          fegen::FegenType::getFloatPointType(
              fegen::FegenValue::get(fegen::FegenType::getInt32Type(), "size",
                                     fegen::FegenRightValue::getInteger(size))),
          true),
      content(content) {}

std::any fegen::FegenRightValue::FloatPointLiteral::getContent() {
  return this->content;
}

std::string fegen::FegenRightValue::FloatPointLiteral::toString() {
  return std::to_string(this->content);
}

// class StringLiteral
fegen::FegenRightValue::StringLiteral::StringLiteral(std::string content)
    : ExpressionTerminal(fegen::FegenRightValue::LiteralKind::STRING,
                         fegen::FegenType::getStringType(), true),
      content(content) {}

std::any fegen::FegenRightValue::StringLiteral::getContent() {
  return this->content;
}

std::string fegen::FegenRightValue::StringLiteral::toString() {
  std::string res;
  res.append("\"");
  res.append(this->content);
  res.append("\"");
  return res;
}

// class TypeLiteral

// Check params of content and return ture if params are all const expr.
inline bool isParamsConstant(fegen::FegenType &content) {
  for (auto param : content.getParameters()) {
    if (!param->getExpr()->isConstexpr()) {
      return false;
    }
  }
  return true;
}

// Get type of type literal.
fegen::FegenType getTypeLiteralType(fegen::FegenType &content) {
  if (content.getTypeLevel() == 2) {
    return fegen::FegenType::getMetaTemplateType();
  } else if (content.getTypeLevel() == 3) {
    return fegen::FegenType::getMetaType();
  } else {
    return fegen::FegenType::getPlaceHolder();
  }
}

fegen::FegenRightValue::TypeLiteral::TypeLiteral(fegen::FegenType &content)
    : ExpressionTerminal(fegen::FegenRightValue::LiteralKind::TYPE,
                         getTypeLiteralType(content),
                         isParamsConstant(content)),
      content(content) {}

std::any fegen::FegenRightValue::TypeLiteral::getContent() {
  return this->content;
}

std::string fegen::FegenRightValue::TypeLiteral::toString() {
  return this->content.getTypeName();
}

std::string fegen::FegenRightValue::TypeLiteral::toStringForTypedef() {
  return this->content.toStringForTypedef();
}

std::string fegen::FegenRightValue::TypeLiteral::toStringForOpdef() {
  return this->content.toStringForOpdef();
}

std::string fegen::FegenRightValue::TypeLiteral::toStringForCppKind() {
  return this->content.toStringForCppKind();
}

// class ExpressionTerminal

// Return ture if all Expressions in content are all true.
bool isExpressionListConst(
    std::vector<std::shared_ptr<fegen::FegenRightValue::Expression>> &content) {
  for (auto p : content) {
    if (!p->isConstexpr()) {
      return false;
      break;
    }
  }
  return true;
}

fegen::FegenRightValue::ListLiteral::ListLiteral(
    std::vector<std::shared_ptr<Expression>> &content)
    : ExpressionTerminal(fegen::FegenRightValue::LiteralKind::VECTOR,
                         content[0]->exprType, isExpressionListConst(content)),
      content(content) {}

std::any fegen::FegenRightValue::ListLiteral::getContent() {
  return this->content;
}

std::string fegen::FegenRightValue::ListLiteral::toString() {
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

std::string fegen::FegenRightValue::ListLiteral::toStringForTypedef() {
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

std::string fegen::FegenRightValue::ListLiteral::toStringForOpdef() {
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
fegen::FegenRightValue::LeftValue::LeftValue(fegen::FegenValue *content)
    : ExpressionTerminal(fegen::FegenRightValue::LiteralKind::LEFT_VAR,
                         content->getType(), content->getExpr()->isConstexpr()),
      content(content) {}

std::any fegen::FegenRightValue::LeftValue::getContent() {
  return this->content;
}

std::string fegen::FegenRightValue::LeftValue::toString() {
  return this->content->getName();
}

// class FegenRightValue
fegen::FegenRightValue::FegenRightValue(
    std::shared_ptr<fegen::FegenRightValue::Expression> content)
    : content(content) {}

fegen::FegenRightValue::LiteralKind fegen::FegenRightValue::getLiteralKind() {
  return this->content->getKind();
}

std::string fegen::FegenRightValue::toString() {
  return this->content->toString();
}

std::string fegen::FegenRightValue::toStringForTypedef() {
  return this->content->toStringForTypedef();
}

std::string fegen::FegenRightValue::toStringForOpdef() {
  return this->content->toStringForOpdef();
}

std::string fegen::FegenRightValue::toStringForCppKind() {
  return this->content->toStringForCppKind();
}

std::any fegen::FegenRightValue::getContent() {
  return this->content->getContent();
}

fegen::FegenType &fegen::FegenRightValue::getType() {
  return this->content->getType();
}

std::shared_ptr<fegen::FegenRightValue::Expression>
fegen::FegenRightValue::getExpr() {
  return this->content;
}

fegen::FegenRightValue fegen::FegenRightValue::getPlaceHolder() {
  return fegen::FegenRightValue(
      fegen::FegenRightValue::Expression::getPlaceHolder());
}

fegen::FegenRightValue fegen::FegenRightValue::getInteger(long long int content,
                                                          size_t size) {
  return fegen::FegenRightValue(
      fegen::FegenRightValue::Expression::getInteger(content, size));
}

fegen::FegenRightValue
fegen::FegenRightValue::getFloatPoint(long double content, size_t size) {
  return fegen::FegenRightValue(
      fegen::FegenRightValue::Expression::getFloatPoint(content, size));
}
fegen::FegenRightValue fegen::FegenRightValue::getString(std::string content) {
  return fegen::FegenRightValue(
      fegen::FegenRightValue::Expression::getString(content));
}
fegen::FegenRightValue
fegen::FegenRightValue::getType(fegen::FegenType &content) {
  return fegen::FegenRightValue(
      fegen::FegenRightValue::Expression::getType(content));
}

fegen::FegenRightValue fegen::FegenRightValue::getList(
    std::vector<std::shared_ptr<fegen::FegenRightValue::Expression>> &content) {
  return fegen::FegenRightValue(
      fegen::FegenRightValue::Expression::getList(content));
}
fegen::FegenRightValue
fegen::FegenRightValue::getLeftValue(fegen::FegenValue *content) {
  return fegen::FegenRightValue(
      fegen::FegenRightValue::Expression::getLeftValue(content));
}

fegen::FegenRightValue fegen::FegenRightValue::getByExpr(
    std::shared_ptr<fegen::FegenRightValue::Expression> expr) {
  assert(expr != nullptr);
  return fegen::FegenRightValue(expr);
}

// class FegenValue
fegen::FegenValue::FegenValue(fegen::FegenType type, std::string name,
                              fegen::FegenRightValue content)
    : type(std::move(type)), name(std::move(name)),
      content(std::move(content)) {}

fegen::FegenValue::FegenValue(const fegen::FegenValue &rhs)
    : type(rhs.type), name(rhs.name), content(rhs.content) {}
fegen::FegenValue::FegenValue(fegen::FegenValue &&rhs)
    : type(std::move(rhs.type)), name(std::move(rhs.name)),
      content(std::move(rhs.content)) {}

fegen::FegenValue *fegen::FegenValue::get(fegen::FegenType type,
                                          std::string name,
                                          FegenRightValue content) {
  return new fegen::FegenValue(std::move(type), std::move(name),
                               std::move(content));
}

fegen::FegenType &fegen::FegenValue::getType() { return this->type; }

std::string fegen::FegenValue::getName() { return this->name; }

fegen::FegenRightValue::LiteralKind fegen::FegenValue::getContentKind() {
  return this->content.getLiteralKind();
}

std::string fegen::FegenValue::getContentString() {
  return this->content.toString();
}

std::string fegen::FegenValue::getContentStringForTypedef() {
  return this->content.toStringForTypedef();
}

std::string fegen::FegenValue::getContentStringForOpdef() {
  return this->content.toStringForOpdef();
}

std::string fegen::FegenValue::getContentStringForCppKind() {
  return this->content.toStringForCppKind();
}

std::shared_ptr<fegen::FegenRightValue::Expression>
fegen::FegenValue::getExpr() {
  return this->content.getExpr();
}

fegen::FegenRule::FegenRule(std::string content, fegen::FegenNode *src,
                            antlr4::ParserRuleContext *ctx)
    : content(content), src(src), ctx(ctx) {}

fegen::FegenRule *fegen::FegenRule::get(std::string content,
                                        fegen::FegenNode *src,
                                        antlr4::ParserRuleContext *ctx) {
  return new fegen::FegenRule(content, src, ctx);
}

llvm::StringRef fegen::FegenRule::getContent() { return this->content; }

bool fegen::FegenRule::addInput(fegen::FegenValue input) {
  auto name = input.getName();
  if (this->inputs.count(name) == 0) {
    return false;
  }
  this->inputs.insert({name, new fegen::FegenValue(input)});
  return true;
}

bool fegen::FegenRule::addReturn(fegen::FegenValue output) {
  auto name = output.getName();
  if (this->returns.count(name) == 0) {
    return false;
  }
  this->returns.insert({name, new fegen::FegenValue(output)});
  return true;
}

void fegen::FegenRule::setSrc(FegenNode *src) { this->src = src; }

fegen::FegenNode::FegenNode(std::vector<fegen::FegenRule *> &&rules,
                            antlr4::ParserRuleContext *ctx,
                            fegen::FegenNode::NodeType ntype)
    : rules(rules), ctx(ctx), ntype(ntype) {}

fegen::FegenNode *fegen::FegenNode::get(std::vector<fegen::FegenRule *> rules,
                                        antlr4::ParserRuleContext *ctx,
                                        fegen::FegenNode::NodeType ntype) {
  return new fegen::FegenNode(std::move(rules), ctx, ntype);
}
fegen::FegenNode *fegen::FegenNode::get(antlr4::ParserRuleContext *ctx,
                                        fegen::FegenNode::NodeType ntype) {
  std::vector<fegen::FegenRule *> rules;
  return new fegen::FegenNode(std::move(rules), ctx, ntype);
}

void fegen::FegenNode::addFegenRule(fegen::FegenRule *rule) {
  this->rules.push_back(rule);
}

fegen::FegenNode::~FegenNode() {
  for (auto rule : this->rules) {
    delete rule;
  }
}

void fegen::FegenManager::setModuleName(std::string name) {
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

fegen::FegenManager::FegenManager() {}

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
  FegenManager &manager;
  Emitter &emitter;

public:
  StmtGenerator(Emitter &emitter)
      : manager(FegenManager::getManager()), emitter(emitter) {}
  std::any visitVarDeclStmt(FegenParser::VarDeclStmtContext *ctx) override {
    auto var = manager.getStmtContent<FegenValue *>(ctx->identifier());
    switch (var->getType().getTypeKind()) {
    case fegen::FegenType::TypeKind::CPP: {
      this->emitter << var->getType().toStringForCppKind() << " "
                    << var->getName();
      if (ctx->expression()) {
        auto expr = this->manager.getStmtContent<FegenRightValue::Expression *>(
            ctx->expression());
        this->emitter << " = " << expr->toStringForCppKind();
      }
      this->emitter << ";";
      this->emitter.newLine();
      break;
    }
    case fegen::FegenType::TypeKind::ATTRIBUTE: {
      break;
    }
    case fegen::FegenType::TypeKind::OPERAND: {
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

void fegen::FegenManager::emitG4() {
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
void fegen::FegenManager::emitTypeDefination() {
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

void fegen::FegenManager::emitOpDefination() {
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

void fegen::FegenManager::emitDialectDefination() {
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

void fegen::FegenManager::emitTdFiles() {
  this->emitDialectDefination();
  this->emitTypeDefination();
  this->emitOpDefination();
}

void fegen::FegenManager::initbuiltinTypes() {
  // placeholder type
  auto placeholderTypeDefination = fegen::FegenTypeDefination::get(
      "fegen_builtin", FEGEN_PLACEHOLDER, {}, nullptr, false);
  this->typeDefMap.insert({FEGEN_PLACEHOLDER, placeholderTypeDefination});

  // Type
  this->typeDefMap.insert(
      {FEGEN_TYPE, fegen::FegenTypeDefination::get("fegen_builtin", FEGEN_TYPE,
                                                   {}, nullptr, false)});

  // TypeTemplate
  this->typeDefMap.insert(
      {FEGEN_TYPETEMPLATE,
       fegen::FegenTypeDefination::get("fegen_builtin", FEGEN_TYPETEMPLATE, {},
                                       nullptr, false)});

  // recursive define Integer Type
  // Integer<Integer<Integer<...>>>
  auto intTypeDefination = fegen::FegenTypeDefination::get(
      "fegen_builtin", FEGEN_INTEGER, {}, nullptr, false);
  auto intType = fegen::FegenType(
      fegen::FegenType::TypeKind::CPP,
      {fegen::FegenValue::get(fegen::FegenType::getPlaceHolder(), "size",
                              fegen::FegenRightValue::getPlaceHolder())},
      intTypeDefination, false);
  // parameters of Integer is int32(Integer<32>)
  intTypeDefination->parameters.push_back(fegen::FegenValue::get(
      intType, "size", fegen::FegenRightValue::getPlaceHolder()));
  this->typeDefMap.insert({FEGEN_INTEGER, intTypeDefination});

  // FloatPoint
  this->typeDefMap.insert(
      {FEGEN_FLOATPOINT,
       fegen::FegenTypeDefination::get(
           "fegen_builtin", FEGEN_FLOATPOINT,
           {fegen::FegenValue::get(fegen::FegenType::getInt32Type(), "size",
                                   fegen::FegenRightValue::getPlaceHolder())},
           nullptr, false)});

  // Char
  this->typeDefMap.insert(
      {FEGEN_CHAR, fegen::FegenTypeDefination::get("fegen_builtin", FEGEN_CHAR,
                                                   {}, nullptr, false)});

  // String
  this->typeDefMap.insert(
      {FEGEN_STRING, fegen::FegenTypeDefination::get(
                         "fegen_builtin", FEGEN_STRING, {}, nullptr, false)});

  // Vector
  this->typeDefMap.insert(
      {FEGEN_VECTOR,
       fegen::FegenTypeDefination::get(
           "fegen_builtin", FEGEN_VECTOR,
           {fegen::FegenValue::get(fegen::FegenType::getInt32Type(), "size",
                                   fegen::FegenRightValue::getPlaceHolder()),
            fegen::FegenValue::get(fegen::FegenType::getMetaType(),
                                   "elementType",
                                   fegen::FegenRightValue::getPlaceHolder())},
           nullptr, false)});

  // List (this should be ahead of Tensor and Any Type defination)
  this->typeDefMap.insert(
      {FEGEN_LIST, fegen::FegenTypeDefination::get(
                       "fegen_builtin", FEGEN_LIST,
                       {fegen::FegenValue::get(
                           fegen::FegenType::getMetaType(), "elementType",
                           fegen::FegenRightValue::getPlaceHolder())},
                       nullptr, false)});

  // Tensor
  this->typeDefMap.insert(
      {FEGEN_TENSOR,
       fegen::FegenTypeDefination::get(
           "fegen_builtin", FEGEN_TENSOR,
           {fegen::FegenValue::get(
                fegen::FegenType::getListType(fegen::FegenType::getInt32Type()),
                "shape", fegen::FegenRightValue::getPlaceHolder()),
            fegen::FegenValue::get(fegen::FegenType::getMetaType(),
                                   "elementType",
                                   fegen::FegenRightValue::getPlaceHolder())},
           nullptr, false)});

  // Optional
  this->typeDefMap.insert(
      {FEGEN_OPTINAL, fegen::FegenTypeDefination::get(
                          "fegen_builtin", FEGEN_OPTINAL,
                          {fegen::FegenValue::get(
                              fegen::FegenType::getMetaType(), "elementType",
                              fegen::FegenRightValue::getPlaceHolder())},
                          nullptr, false)});

  // Any
  this->typeDefMap.insert(
      {FEGEN_ANY,
       fegen::FegenTypeDefination::get(
           "fegen_builtin", FEGEN_ANY,
           {fegen::FegenValue::get(
               fegen::FegenType::getListType(fegen::FegenType::getMetaType()),
               "elementType", fegen::FegenRightValue::getPlaceHolder())},
           nullptr, false)});
}

fegen::FegenTypeDefination *
fegen::FegenManager::getTypeDefination(std::string name) {
  return this->typeDefMap[name];
}

bool fegen::FegenManager::addTypeDefination(fegen::FegenTypeDefination *tyDef) {
  if (this->typeDefMap.count(tyDef->name) != 0) {
    return false;
  }
  this->typeDefMap[tyDef->name] = tyDef;
  return true;
}

fegen::FegenOperation *
fegen::FegenManager::getOperationDefination(std::string name) {
  return this->operationMap[name];
}

bool fegen::FegenManager::addOperationDefination(fegen::FegenOperation *opDef) {
  if (this->operationMap.count(opDef->getOpName()) != 0) {
    return false;
  }
  this->operationMap[opDef->getOpName()] = opDef;
  return true;
}

void fegen::FegenManager::addStmtContent(antlr4::ParserRuleContext *ctx,
                                         std::any content) {
  this->stmtContentMap.insert({ctx, content});
}

fegen::FegenManager &fegen::FegenManager::getManager() {
  static fegen::FegenManager fmg;
  return fmg;
}

fegen::FegenManager::~FegenManager() {
  // release nodes
  for (auto node_pair : this->nodeMap) {
    delete node_pair.second;
  }
}

fegen::FegenType fegen::inferenceType(
    std::vector<std::shared_ptr<fegen::FegenRightValue::Expression>> operands,
    fegen::FegenOperator op) {
  // TODO: infer type
  return fegen::FegenType::getInt32Type();
}

namespace fegen {

// class StmtVisitor : public FegenParserBaseVisitor{
// public:
// };

}
void fegen::FegenManager::emitBuiltinFunction() {
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