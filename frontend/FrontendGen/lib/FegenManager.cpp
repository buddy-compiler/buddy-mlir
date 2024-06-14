#include "FegenManager.h"
#include <algorithm>
#include <type_traits>

fegen::FegenFunction::FegenFunction(llvm::StringRef name,
                                    std::vector<FegenType *> &&inputTypeList,
                                    FegenType *returnType)
    : name(name), inputTypeList(inputTypeList), returnType(returnType) {}

fegen::FegenFunction *
fegen::FegenFunction::get(llvm::StringRef name,
                          std::vector<FegenType *> inputTypeList,
                          FegenType *returnType) {
  return new fegen::FegenFunction(name, std::move(inputTypeList), returnType);
}

fegen::FegenOperation::FegenOperation(llvm::StringRef dialectName,
                                      llvm::StringRef operationName,
                                      std::vector<FegenValue *> &&arguments,
                                      std::vector<FegenValue *> &&results)
    : dialectName(dialectName), arguments(arguments), results(results) {}

fegen::FegenOperation *fegen::FegenOperation::get(
    llvm::StringRef dialectName, llvm::StringRef operationName,
    std::vector<FegenValue *> arguments, std::vector<FegenValue *> results) {
  return new fegen::FegenOperation(dialectName, operationName,
                                   std::move(arguments), std::move(results));
}

// class FegenType

/// @brief get name of Type Instance by jointsing template name and parameters,
/// for example: Integer + 32 --> Integer<32>
/// @return joint name
std::string jointTypeName(std::string templateName,
                          const std::vector<fegen::FegenValue*> &parameters) {
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

fegen::FegenType::FegenType(fegen::FegenType::TypeKind kind,
                            std::vector<FegenValue*> parameters,
                            FegenTypeDefination *tyDef, bool isTemplate)
    : kind(kind), typeName(jointTypeName(tyDef->getName(), parameters)),
      parameters(std::move(parameters)), typeDefine(tyDef),
      ifTemplate(isTemplate) {}

fegen::FegenType::FegenType(const fegen::FegenType &fty)
    : kind(fty.kind),
      typeName(jointTypeName(fty.typeDefine->getName(), fty.parameters)),
      typeDefine(fty.typeDefine),
      ifTemplate(fty.ifTemplate) {
        // deep copy parameters
        for(auto paramPtr : fty.parameters){
          this->parameters.push_back(new fegen::FegenValue(*paramPtr));
        }
      }

fegen::FegenType::FegenType(fegen::FegenType &&fty)
    : kind(fty.kind),
      typeName(jointTypeName(fty.typeDefine->getName(), fty.parameters)),
      parameters(std::move(fty.parameters)), typeDefine(fty.typeDefine),
      ifTemplate(fty.ifTemplate) {}

fegen::FegenType::TypeKind fegen::FegenType::getTypeKind() {
  return this->kind;
}

void fegen::FegenType::setTypeKind(fegen::FegenType::TypeKind kind) {
  this->kind = kind;
}

std::vector<fegen::FegenValue*> &fegen::FegenType::getParameters() {
  return this->parameters;
}

void fegen::FegenType::setParameters(std::vector<fegen::FegenValue*> &params) {
  this->parameters = params;
  this->ifTemplate = false;
}

fegen::FegenTypeDefination *fegen::FegenType::getTypeDefination() {
  return this->typeDefine;
}

void fegen::FegenType::setTypeDefination(fegen::FegenTypeDefination *tyDef) {
  this->typeDefine = tyDef;
}

std::string fegen::FegenType::getTypeName() { return this->typeName; }

bool fegen::FegenType::isTemplate() {
  return this->ifTemplate;
}

fegen::FegenType::~FegenType() {
  for(auto p : this->parameters){
    delete p;
  }
}

fegen::FegenType fegen::FegenType::getPlaceHolder() {
    return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, {},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_PLACEHOLDER), true);
}

fegen::FegenType fegen::FegenType::getMetaType() {
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, {},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_TYPE), true);
}

fegen::FegenType fegen::FegenType::getMetaTemplateType() {
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, {},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_TYPETEMPLATE),
      true);
}

fegen::FegenType fegen::FegenType::getInt32Type() {
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, {},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_INTEGER),
      false);
}

fegen::FegenType fegen::FegenType::getFloatType() {
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, {},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_FLOATPOINT),
      false);
}

fegen::FegenType fegen::FegenType::getDoubleType() {
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, {},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_FLOATPOINT),
      false);
}

fegen::FegenType fegen::FegenType::getBoolType() {
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, {},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_INTEGER),
      false);
}

fegen::FegenType fegen::FegenType::getIntegerType(fegen::FegenValue* size) {
  if (size->getContent<int>() == 32)
    return fegen::FegenType::getInt32Type();
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, {size},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_INTEGER),
      false);
}

fegen::FegenType fegen::FegenType::getFloatPointType(fegen::FegenValue* size) {
  if (size->getContent<int>() == 32) {
    return fegen::FegenType::getFloatType();
  } else if (size->getContent<int>() == 64) {
    return fegen::FegenType::getDoubleType();
  }
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, {size},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_FLOATPOINT),
      false);
}

fegen::FegenType fegen::FegenType::getCharType() {
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, {},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_CHAR), false);
}

fegen::FegenType fegen::FegenType::getStringType() {
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, {},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_STRING), false);
}

fegen::FegenType
fegen::FegenType::getVectorType(fegen::FegenValue* size,
                                fegen::FegenValue* elementType) {
  std::vector<fegen::FegenValue> params;
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, {size, elementType},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_VECTOR), false);
}

fegen::FegenType fegen::FegenType::getVectorType(fegen::FegenValue* size,
                                                 fegen::FegenType elementType) {
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP,
      {size, fegen::FegenValue::get(fegen::FegenType::getMetaType(), "elementType",
                               fegen::FegenRightValue::get(elementType))},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_VECTOR), false);
}

fegen::FegenType
fegen::FegenType::getTensorType(fegen::FegenValue* shape,
                                fegen::FegenValue* elementType) {
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, {shape, elementType},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_TENSOR), false);
}
fegen::FegenType fegen::FegenType::getTensorType(fegen::FegenValue* shape,
                                                 fegen::FegenType elementType) {
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP,
      {shape, fegen::FegenValue::get(fegen::FegenType::getMetaType(), "elementType",
                                fegen::FegenRightValue::get(elementType))},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_TENSOR), false);
}

// List<elementType>
fegen::FegenType fegen::FegenType::getListType(fegen::FegenValue* elementType) {
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, {elementType},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_LIST), false);
}

fegen::FegenType fegen::FegenType::getListType(fegen::FegenType elementType) {
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP,
      {fegen::FegenValue::get(fegen::FegenType::getMetaType(), "elementType",
                         fegen::FegenRightValue::get(elementType))},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_LIST), false);
}

// Optional<elementType>
fegen::FegenType
fegen::FegenType::getOptionalType(fegen::FegenValue* elementType) {
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, {elementType},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_OPTINAL),
      false);
}

fegen::FegenType
fegen::FegenType::getOptionalType(fegen::FegenType elementType) {
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP,
      {fegen::FegenValue::get(fegen::FegenType::getMetaType(), "elementType",
                         fegen::FegenRightValue::get(elementType))},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_OPTINAL),
      false);
}

// Any<elementType1, elementType2, ...>
fegen::FegenType
fegen::FegenType::getAnyType(std::vector<fegen::FegenValue*> elementTypes) {
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, elementTypes,
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_ANY), false);
}

fegen::FegenType
fegen::FegenType::getAnyType(std::vector<fegen::FegenType> elementTypes) {
  std::vector<fegen::FegenValue*> p_elemTy;
  int i = 0;
  std::string name("elementType_");
  for (auto ty : elementTypes) {
    p_elemTy.push_back(fegen::FegenValue::get(fegen::FegenType::getMetaType(),
                                         name + std::to_string(i),
                                         fegen::FegenRightValue::get(ty)));
    i++;
  }
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, p_elemTy,
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_ANY), false);
}

fegen::FegenType fegen::FegenType::getIntegerTemplate() {
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, {},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_INTEGER), true);
}

fegen::FegenType fegen::FegenType::getFloatPointTemplate() {
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, {},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_FLOATPOINT),
      true);
}

fegen::FegenType fegen::FegenType::getVectorTemplate() {
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, {},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_VECTOR), true);
}

fegen::FegenType fegen::FegenType::getTensorTemplate() {
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, {},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_TENSOR), true);
}

fegen::FegenType fegen::FegenType::getListTemplate() {
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, {},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_LIST), true);
}

fegen::FegenType fegen::FegenType::getOptionalTemplate() {
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, {},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_OPTINAL), true);
}

fegen::FegenType fegen::FegenType::getAnyTemplate() {
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, {},
      fegen::FegenManager::getManager().getTypeDefination(FEGEN_ANY), true);
}

fegen::FegenType
fegen::FegenType::getInstanceType(fegen::FegenTypeDefination *typeDefination,
                                  std::vector<fegen::FegenValue*> parameters) {
  return fegen::FegenType(fegen::FegenType::TypeKind::CPP, parameters,
                          typeDefination, false);
}
fegen::FegenType
fegen::FegenType::getTemplateType(fegen::FegenTypeDefination *typeDefination) {
  return fegen::FegenType(fegen::FegenType::TypeKind::CPP, {}, typeDefination,
                          true);
}

// class FegenTypeDefination
fegen::FegenTypeDefination::FegenTypeDefination(
    std::string dialectName, std::string name,
    std::vector<fegen::FegenValue*> parameters,
    FegenParser::TypeDefinationDeclContext *ctx, bool ifCustome)
    : dialectName(std::move(dialectName)), name(std::move(name)),
      parameters(std::move(parameters)), ctx(ctx), ifCustome(ifCustome) {}

fegen::FegenTypeDefination *
fegen::FegenTypeDefination::get(std::string dialectName, std::string name,
                                std::vector<fegen::FegenValue*> parameters,
                                FegenParser::TypeDefinationDeclContext *ctx,
                                bool ifCustome) {
  return new fegen::FegenTypeDefination(std::move(dialectName), std::move(name),
                                        std::move(parameters), ctx, ifCustome);
}

std::string fegen::FegenTypeDefination::getDialectName() {
  return this->dialectName;
}

std::string fegen::FegenTypeDefination::getName() { return this->name; }

const std::vector<fegen::FegenValue*> &
fegen::FegenTypeDefination::getParameters() {
  return this->parameters;
}

fegen::FegenParser::TypeDefinationDeclContext *
fegen::FegenTypeDefination::getCtx() {
  return this->ctx;
}

bool fegen::FegenTypeDefination::isCustome() { return this->ifCustome; }

// class Expression

fegen::FegenRightValue::Expression::Expression(bool ifTerminal, LiteralKind kind, FegenType& exprTy) : 
  ifTerminal(ifTerminal), kind(kind), exprType(exprTy)
{

} 

bool fegen::FegenRightValue::Expression::isTerminal() {
  return this->ifTerminal;
}

fegen::FegenRightValue::LiteralKind
fegen::FegenRightValue::Expression::getKind() {
  return this->kind;
}

std::any fegen::FegenRightValue::Expression::getContent() {
  if(this->ifTerminal){
    auto tPtr = dynamic_cast<fegen::FegenRightValue::ExpressionTerminal*>(this);
    return tPtr->content;
  }else{
    return dynamic_cast<fegen::FegenRightValue::ExpressionNode*>(this);;
  }
}

// class ExpressionNode

fegen::FegenRightValue::ExpressionNode::ExpressionNode(
    std::vector<fegen::FegenRightValue::Expression*> params,
    std::variant<fegen::FegenFunction *, fegen::FegenOperation *,
                 fegen::FegenOperator>
        op, FegenType& exprTy)
    :Expression(false, fegen::FegenRightValue::LiteralKind::EXPRESSION, exprTy), op(op), params(params) {
}

fegen::FegenRightValue::ExpressionNode::~ExpressionNode() {
  for(auto p : this->params){
    delete p;
  }
}


std::string fegen::FegenRightValue::ExpressionNode::toString() {
  // TODO: toString
  return "todo: fegen::FegenRightValue::ExpressionNode::toString";
}

std::any fegen::FegenRightValue::ExpressionNode::getContent() {
  return this;
}

fegen::FegenRightValue::ExpressionNode *
fegen::FegenRightValue::ExpressionNode::binaryOperation(
    fegen::FegenRightValue::Expression *lhs,
    fegen::FegenRightValue::Expression *rhs, FegenOperator op) {
    // TODO: infer type kind: cpp, attribute, or operand
    FegenType resTy = fegen::inferenceType({lhs, rhs}, op);
    return new fegen::FegenRightValue::ExpressionNode({lhs, rhs}, op, resTy);
}

fegen::FegenRightValue::ExpressionNode *
fegen::FegenRightValue::ExpressionNode::unaryOperation(
    fegen::FegenRightValue::Expression *v, FegenOperator op) {
  // TODO: infer type kind: cpp, attribute, or operand
  FegenType resTy = fegen::inferenceType({v}, op);
  return new fegen::FegenRightValue::ExpressionNode({v}, op, resTy);
}

// class ExpressionTerminal
fegen::FegenRightValue::ExpressionTerminal::ExpressionTerminal(
    primLiteralType c, fegen::FegenRightValue::LiteralKind kind, FegenType exprTy)
    : Expression(true, kind, exprTy), content(c) {
}

fegen::FegenRightValue::ExpressionTerminal::~ExpressionTerminal() {
  if(this->kind == fegen::FegenRightValue::LiteralKind::VECTOR){
    auto v = std::get<std::vector<Expression*>>(this->content);
    for(auto p : v){
      delete p;
    }
  }
}

std::string fegen::FegenRightValue::ExpressionTerminal::toString() {
  // TODO: toString
  return "todo: fegen::FegenRightValue::ExpressionTerminal::toString";
}

std::any fegen::FegenRightValue::ExpressionTerminal::getContent() {
  switch(this->kind){
    case fegen::FegenRightValue::LiteralKind::INT:
      return std::get<int>(this->content);
    case fegen::FegenRightValue::LiteralKind::FLOAT:
      return std::get<float>(this->content);
    case fegen::FegenRightValue::LiteralKind::STRING:
      return std::get<std::string>(this->content);
    case fegen::FegenRightValue::LiteralKind::TYPE:
      return std::get<FegenType>(this->content);
    case fegen::FegenRightValue::LiteralKind::VECTOR:
      return std::get<std::vector<Expression*>>(this->content);
    case fegen::FegenRightValue::LiteralKind::LEFT_VAR:
      return std::get<FegenValue *>(this->content);
    default:
      return std::monostate();
  }
}

fegen::FegenRightValue::ExpressionTerminal *
fegen::FegenRightValue::ExpressionTerminal::get(std::monostate content) {
  return new fegen::FegenRightValue::ExpressionTerminal(
      content, fegen::FegenRightValue::LiteralKind::MONOSTATE, fegen::FegenType::getPlaceHolder());
}

fegen::FegenRightValue::ExpressionTerminal *
fegen::FegenRightValue::ExpressionTerminal::get(int content) {
  return new fegen::FegenRightValue::ExpressionTerminal(
      content, fegen::FegenRightValue::LiteralKind::INT, fegen::FegenType::getInt32Type());
}

fegen::FegenRightValue::ExpressionTerminal *
fegen::FegenRightValue::ExpressionTerminal::get(float content) {
  return new fegen::FegenRightValue::ExpressionTerminal(
      content, fegen::FegenRightValue::LiteralKind::FLOAT, fegen::FegenType::getFloatType());
}

fegen::FegenRightValue::ExpressionTerminal *
fegen::FegenRightValue::ExpressionTerminal::get(std::string content) {
  return new fegen::FegenRightValue::ExpressionTerminal(
      content, fegen::FegenRightValue::LiteralKind::STRING, fegen::FegenType::getStringType());
}

fegen::FegenRightValue::ExpressionTerminal *
fegen::FegenRightValue::ExpressionTerminal::get(fegen::FegenType &content) {
  if(content.isTemplate()){
      return new fegen::FegenRightValue::ExpressionTerminal(
      content, fegen::FegenRightValue::LiteralKind::TYPE, fegen::FegenType::getMetaTemplateType());
  }else{
        return new fegen::FegenRightValue::ExpressionTerminal(
      content, fegen::FegenRightValue::LiteralKind::TYPE, fegen::FegenType::getMetaType());
  }

}

fegen::FegenRightValue::ExpressionTerminal *
fegen::FegenRightValue::ExpressionTerminal::get(
    std::vector<fegen::FegenRightValue::Expression *> &content) {
  return new fegen::FegenRightValue::ExpressionTerminal(
      content, fegen::FegenRightValue::LiteralKind::VECTOR, fegen::FegenType::getListType(content[0]->exprType));
}

fegen::FegenRightValue::ExpressionTerminal *
fegen::FegenRightValue::ExpressionTerminal::get(fegen::FegenValue *content) {
  return new fegen::FegenRightValue::ExpressionTerminal(
      content, fegen::FegenRightValue::LiteralKind::LEFT_VAR, content->getType());
}

// class FegenRightValue
fegen::FegenRightValue::FegenRightValue(
    fegen::FegenRightValue::Expression *content)
    : content(content) {}

fegen::FegenRightValue::FegenRightValue(const fegen::FegenRightValue& rhs) {
  if(rhs.content->isTerminal()){
    auto expr = dynamic_cast<fegen::FegenRightValue::ExpressionTerminal*>(rhs.content);
    this->content = new fegen::FegenRightValue::ExpressionTerminal(*expr);
  }else{
    auto expr = dynamic_cast<fegen::FegenRightValue::ExpressionNode*>(rhs.content);
    this->content = new fegen::FegenRightValue::ExpressionNode(*expr);
  }
}

fegen::FegenRightValue::FegenRightValue(fegen::FegenRightValue&& rhs) {
  this->content = rhs.content;
  rhs.content = nullptr;
}

fegen::FegenRightValue::LiteralKind fegen::FegenRightValue::getKind() {
  return this->content->getKind();
}

std::string fegen::FegenRightValue::toString() {
  // TODO: toString
  return "TODO: fegen::FegenRightValue::toString.";
}

std::any fegen::FegenRightValue::getContent() {
  return this->content->getContent();
}

fegen::FegenRightValue fegen::FegenRightValue::get() {
  return fegen::FegenRightValue(
    fegen::FegenRightValue::ExpressionTerminal::get(std::monostate())
  );
}

fegen::FegenRightValue fegen::FegenRightValue::get(int content) {
  return fegen::FegenRightValue(
    fegen::FegenRightValue::ExpressionTerminal::get(content)
  );
}
fegen::FegenRightValue fegen::FegenRightValue::get(float content) {
  return fegen::FegenRightValue(
    fegen::FegenRightValue::ExpressionTerminal::get(content)
  );
}
fegen::FegenRightValue fegen::FegenRightValue::get(std::string content) {
  return fegen::FegenRightValue(
    fegen::FegenRightValue::ExpressionTerminal::get(content)
  );
}
fegen::FegenRightValue fegen::FegenRightValue::get(fegen::FegenType& content) {
  return fegen::FegenRightValue(
    fegen::FegenRightValue::ExpressionTerminal::get(content)
  );
}

fegen::FegenRightValue fegen::FegenRightValue::get(std::vector<fegen::FegenRightValue::Expression*> & content) {
  return fegen::FegenRightValue(
    fegen::FegenRightValue::ExpressionTerminal::get(content)
  );
}
fegen::FegenRightValue fegen::FegenRightValue::get(fegen::FegenValue * content) {
  return fegen::FegenRightValue(
    fegen::FegenRightValue::ExpressionTerminal::get(content)
  );
}

fegen::FegenRightValue fegen::FegenRightValue::get(fegen::FegenRightValue::Expression* expr) {
  return fegen::FegenRightValue(expr);
}

fegen::FegenRightValue::~FegenRightValue() {
  delete this->content;
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
  return this->content.getKind();
}

std::string fegen::FegenValue::getContentString() {
  return this->content.toString();
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

fegen::FegenManager::FegenManager() { 
  }

// TODO: emit to file
std::string fegen::FegenManager::emitG4() {
#define OUT std::cout
#define OUT_TAB1 std::cout << "\t"
#define OUT_TAB2 std::cout << "\t\t"

  OUT << "grammar " << this->moduleName << ";" << std::endl;
  for (auto node_pair : this->nodeMap) {
    auto nodeName = node_pair.first;
    auto node = node_pair.second;
    OUT << nodeName << std::endl;
    auto ruleCount = node->rules.size();
    if (ruleCount > 0) {
      OUT_TAB1 << ": " << getChildrenText(node->rules[0]->ctx) << std::endl;
      for (size_t i = 1; i <= ruleCount - 1; i++) {
        OUT_TAB1 << "| " << getChildrenText(node->rules[i]->ctx) << std::endl;
      }
      OUT_TAB1 << ";" << std::endl;
    }
    OUT << std::endl;
  }

#undef OUT
#undef OUT_TAB1
#undef OUT_TAB2
  return std::string();
}

void fegen::FegenManager::initbuiltinTypes() {
  // placeholder type
  auto placeholderTypeDefination = fegen::FegenTypeDefination::get(
    "fegen_builtin", FEGEN_PLACEHOLDER, {}, nullptr, false
  );
  this->typeDefMap.insert({FEGEN_PLACEHOLDER, placeholderTypeDefination});

  // Type
  this->typeDefMap.insert({FEGEN_TYPE, placeholderTypeDefination});

  // TypeTemplate
  this->typeDefMap.insert({FEGEN_TYPETEMPLATE, placeholderTypeDefination});

  // recursive define Integer Type
  // Integer<Integer<Integer<...>>>
  auto intTypeDefination = fegen::FegenTypeDefination::get(
      "fegen_builtin", FEGEN_INTEGER, {}, nullptr, false);
  auto intType = fegen::FegenType(fegen::FegenType::TypeKind::CPP, {},
                                  intTypeDefination, false);
  // parameters of Integer is int32(Integer<32>)
  intTypeDefination->parameters.push_back(
      fegen::FegenValue::get(intType, "size", fegen::FegenRightValue::get()));
  this->typeDefMap.insert({FEGEN_INTEGER, intTypeDefination});

  // FloatPoint
  this->typeDefMap.insert(
      {FEGEN_FLOATPOINT,
       fegen::FegenTypeDefination::get(
           "fegen_builtin", FEGEN_FLOATPOINT,
           {fegen::FegenValue::get(fegen::FegenType::getInt32Type(), "size",
                              fegen::FegenRightValue::get())},
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
                              fegen::FegenRightValue::get()),
            fegen::FegenValue::get(fegen::FegenType::getMetaType(), "elementType",
                              fegen::FegenRightValue::get())},
           nullptr, false)});

  // List (this should be ahead of Tensor and Any Type defination)
  this->typeDefMap.insert(
      {FEGEN_LIST,
       fegen::FegenTypeDefination::get(
           "fegen_builtin", FEGEN_LIST,
           {fegen::FegenValue::get(fegen::FegenType::getMetaType(), "elementType",
                              fegen::FegenRightValue::get())},
           nullptr, false)});

  // Tensor
  this->typeDefMap.insert(
      {FEGEN_TENSOR,
       fegen::FegenTypeDefination::get(
           "fegen_builtin", FEGEN_TENSOR,
           {fegen::FegenValue::get(
                fegen::FegenType::getListType(fegen::FegenType::getInt32Type()),
                "shape", fegen::FegenRightValue::get()),
            fegen::FegenValue::get(fegen::FegenType::getMetaType(), "elementType",
                              fegen::FegenRightValue::get())},
           nullptr, false)});

  // Optional
  this->typeDefMap.insert(
      {FEGEN_OPTINAL,
       fegen::FegenTypeDefination::get(
           "fegen_builtin", FEGEN_OPTINAL,
           {fegen::FegenValue::get(fegen::FegenType::getMetaType(), "elementType",
                              fegen::FegenRightValue::get())},
           nullptr, false)});

  // Any
  this->typeDefMap.insert(
      {FEGEN_ANY,
       fegen::FegenTypeDefination::get(
           "fegen_builtin", FEGEN_ANY,
           {fegen::FegenValue::get(
               fegen::FegenType::getListType(fegen::FegenType::getMetaType()),
               "elementType", fegen::FegenRightValue::get())},
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

fegen::FegenType fegen::inferenceType(std::vector<fegen::FegenRightValue::Expression*> operands, fegen::FegenOperator op) {
  // TODO: infer type
  return fegen::FegenType::getInt32Type();
}