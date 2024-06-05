#include <type_traits>
#include <algorithm>
#include "FegenManager.h"

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
fegen::FegenType::FegenType(fegen::FegenType::TypeKind kind,
                            std::string dialectName, std::string typeName,
                            std::vector<FegenValue> parameters)
    : kind(kind), dialectName(std::move(dialectName)), typeName(std::move(typeName)), parameters(std::move(parameters)) {
      }

fegen::FegenType::FegenType(const fegen::FegenType &fty)
    : kind(fty.kind), dialectName(fty.dialectName), typeName(fty.typeName), parameters(fty.parameters) {
}

fegen::FegenType::FegenType(fegen::FegenType &&fty)
    : kind(fty.kind), dialectName(std::move(fty.dialectName)),
      typeName(std::move(fty.typeName)), parameters(std::move(fty.parameters)) {
}


fegen::FegenType fegen::FegenType::getMetaType() {
  return fegen::FegenType(fegen::FegenType::TypeKind::CPP, "builtin", "Type",
                          {});
}

fegen::FegenType fegen::FegenType::getMetaTemplateType() {
  return fegen::FegenType(fegen::FegenType::TypeKind::CPP, "builtin",
                          "TypeTemplate", {});
}

fegen::FegenType fegen::FegenType::getInt32Type() {
  return fegen::FegenType(fegen::FegenType::TypeKind::CPP, "builtin", "int32",
                          {});
}

fegen::FegenType fegen::FegenType::getFloatType() {
  return fegen::FegenType(fegen::FegenType::TypeKind::CPP, "builtin", "float",
                          {});
}

fegen::FegenType fegen::FegenType::getDoubleType() {
  return fegen::FegenType(fegen::FegenType::TypeKind::CPP, "builtin", "double",
                          {});
}

fegen::FegenType fegen::FegenType::getBoolType() {
  return fegen::FegenType(fegen::FegenType::TypeKind::CPP, "builtin", "Bool",
                          {});
}

fegen::FegenType fegen::FegenType::getIntegerType(fegen::FegenValue size) {
  if (size.getContent<int>() == 32)
    return fegen::FegenType::getInt32Type();
  return fegen::FegenType(fegen::FegenType::TypeKind::CPP, "builtin", "Integer",
                          {size});
}

fegen::FegenType fegen::FegenType::getFloatPointType(fegen::FegenValue size) {
  if (size.getContent<int>() == 32) {
    return fegen::FegenType::getFloatType();
  } else if (size.getContent<int>() == 64) {
    return fegen::FegenType::getDoubleType();
  }
  return fegen::FegenType(fegen::FegenType::TypeKind::CPP, "builtin",
                          "FloatPoint", {size});
}

fegen::FegenType fegen::FegenType::getCharType() {
  return fegen::FegenType(fegen::FegenType::TypeKind::CPP, "builtin", "Char",
                          {});
}

fegen::FegenType fegen::FegenType::getStringType() {
  return fegen::FegenType(fegen::FegenType::TypeKind::CPP, "builtin", "String",
                          {});
}

fegen::FegenType
fegen::FegenType::getVectorType(fegen::FegenValue size,
                                fegen::FegenValue elementType) {
  std::vector<fegen::FegenValue> params;
  return fegen::FegenType(fegen::FegenType::TypeKind::CPP, "builtin", "Vector",
                          {size, elementType});
}

fegen::FegenType fegen::FegenType::getVectorType(fegen::FegenValue size,
                                                 fegen::FegenType elementType) {
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, "builtin", "Vector",
      {size,
       fegen::FegenValue(fegen::FegenType::getMetaTemplateType(), "elementType",
                         fegen::FegenLiteral::get(elementType))});
}

fegen::FegenType
fegen::FegenType::getTensorType(fegen::FegenValue shape,
                                fegen::FegenValue elementType) {
  return fegen::FegenType(fegen::FegenType::TypeKind::CPP, "builtin", "Tensor",
                          {shape, elementType});
}
fegen::FegenType fegen::FegenType::getTensorType(fegen::FegenValue shape,
                                                 fegen::FegenType elementType) {
  return fegen::FegenType(
      fegen::FegenType::TypeKind::CPP, "builtin", "Tensor",
      {shape,
       fegen::FegenValue(fegen::FegenType::getMetaTemplateType(), "elementType",
                         fegen::FegenLiteral::get(elementType))});
}


// class FegenLiteral
fegen::FegenLiteral::FegenLiteral(
    literalType content)
    : content(std::move(content)) {
      auto index = this->content.index();
      switch(index){
        case 0:
          this->kind = fegen::FegenLiteral::LiteralKind::INT;
          break;
        case 1:
          this->kind = fegen::FegenLiteral::LiteralKind::FLOAT;
          break;
        case 2:
          this->kind = fegen::FegenLiteral::LiteralKind::STRING;
          break;
        case 3: 
          this->kind = fegen::FegenLiteral::LiteralKind::TYPE;
          break;
        case 4:
          this->kind = fegen::FegenLiteral::LiteralKind::VECTOR;
          break;
        default:
          break;
      }
    }

fegen::FegenLiteral::FegenLiteral(const FegenLiteral& flt): kind(flt.kind) {
  switch(this->kind) {
    case fegen::FegenLiteral::LiteralKind::INT:
      this->content = std::get<0>(flt.content);
      break;
    case fegen::FegenLiteral::LiteralKind::FLOAT:
      this->content = std::get<1>(flt.content);
      break;
    case fegen::FegenLiteral::LiteralKind::STRING:
      this->content = std::get<1>(flt.content);
      break;
    case fegen::FegenLiteral::LiteralKind::TYPE:
      this->content = std::get<1>(flt.content);
      break;
    case fegen::FegenLiteral::LiteralKind::VECTOR:
      this->content = std::get<1>(flt.content);
      break;
    default:
      break;
  }

}
fegen::FegenLiteral::FegenLiteral(FegenLiteral&& flt) : content(std::move(flt.content)){}

fegen::FegenLiteral fegen::FegenLiteral::get(int content) {
  return fegen::FegenLiteral(content);
}

fegen::FegenLiteral fegen::FegenLiteral::get(float content) {
  return fegen::FegenLiteral(content);
}

fegen::FegenLiteral fegen::FegenLiteral::get(std::string content) {
  return fegen::FegenLiteral(content);
}

fegen::FegenLiteral fegen::FegenLiteral::get(FegenType content) {
  return fegen::FegenLiteral(content);
}

template<typename T>
fegen::FegenLiteral fegen::FegenLiteral::get(std::vector<T> content) {
  std::vector<fegen::FegenLiteral> processed;
  for(T elem : content){
    auto elemLiteral = fegen::FegenLiteral::get(elem);
    processed.push_back(elemLiteral);
  }
  return fegen::FegenLiteral(processed);
}

// class FegenValue
fegen::FegenValue::FegenValue(fegen::FegenType type, std::string name,
                              fegen::FegenLiteral content)
    : type(std::move(type)), name(std::move(name)), content(std::move(content)) {}

fegen::FegenValue::FegenValue(const fegen::FegenValue& rhs): type(rhs.type), name(rhs.name), content(rhs.content){

}
fegen::FegenValue::FegenValue(fegen::FegenValue&& rhs): type(std::move(rhs.type)), name(std::move(rhs.name)), content(std::move(rhs.content)) {

}

fegen::FegenValue *fegen::FegenValue::get(fegen::FegenType type,
                                          std::string name, FegenLiteral content)
                                          {
  return new fegen::FegenValue(std::move(type), std::move(name), std::move(content));
}

llvm::StringRef fegen::FegenValue::getName() { return this->name; }

fegen::FegenRule::FegenRule(std::string content, fegen::FegenNode *src,
                            antlr4::ParserRuleContext *ctx)
    : content(content), src(src), ctx(ctx) {}

fegen::FegenRule *fegen::FegenRule::get(std::string content,
                                        fegen::FegenNode *src,
                                        antlr4::ParserRuleContext *ctx) {
  return new fegen::FegenRule(content, src, ctx);
}

llvm::StringRef fegen::FegenRule::getContent() {
  return this->content;
}

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
  for(auto rule : this->rules){
    delete rule;
  }
}

void fegen::FegenManager::setModuleName(std::string name) {
  this->moduleName = name;
}

std::string getChildrenText(antlr4::tree::ParseTree* ctx){
  std::string ruleText;
  for(auto child : ctx->children){
    if(antlr4::tree::TerminalNode::is(child)){
      ruleText.append(child->getText()).append(" ");
    }else{
      ruleText.append(getChildrenText(child)).append(" ");
    }
  }
  return ruleText;
}

// TODO: emit to file
std::string fegen::FegenManager::emitG4() {
#define OUT std::cout
#define OUT_TAB1 std::cout << "\t"
#define OUT_TAB2 std::cout << "\t\t"

  OUT << "grammar " << this->moduleName << ";" << std::endl;
  for(auto node_pair : this->nodeMap){
    auto nodeName = node_pair.first;
    auto node = node_pair.second;
    OUT << nodeName << std::endl;
    auto ruleCount = node->rules.size();
    if(ruleCount > 0){
      OUT_TAB1 << ": " << getChildrenText(node->rules[0]->ctx) << std::endl;
      for(size_t i = 1; i <= ruleCount - 1; i++){
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

fegen::FegenManager::~FegenManager() {
  // release nodes
  for(auto node_pair : this->nodeMap){
    delete node_pair.second;
  }
}