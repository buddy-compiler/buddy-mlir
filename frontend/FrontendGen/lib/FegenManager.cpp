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

fegen::FegenType::FegenType(fegen::FegenType::TypeKind kind,
                            llvm::StringRef dialectName,
                            llvm::StringRef typeName,
                            llvm::StringRef assemblyFormat,
                            std::vector<FegenValue *> &&parameters,
                            FegenParser::TypeDefinationDeclContext *ctx)
    : kind(kind), dialectName(dialectName), typeName(typeName),
      assemblyFormat(assemblyFormat), parameters(parameters), ctx(ctx) {}

std::string fegen::FegenType::convertToCppType() {
  // TODO
  return std::string();
}

fegen::FegenType *
fegen::FegenType::get(fegen::FegenType::TypeKind kind,
                      llvm::StringRef dialectName, llvm::StringRef typeName,
                      llvm::StringRef assemblyFormat,
                      std::vector<FegenValue *> parameters,
                      FegenParser::TypeDefinationDeclContext *ctx) {
  return new fegen::FegenType(kind, dialectName, typeName, assemblyFormat,
                              std::move(parameters), ctx);
}

fegen::FegenValue::FegenValue(bool isList, fegen::FegenType *type,
                              llvm::StringRef name,
                              antlr4::ParserRuleContext *ctx)
    : isList(isList), type(type), name(name), ctx(ctx) {}

fegen::FegenValue *fegen::FegenValue::get(bool isList, fegen::FegenType *type,
                                          llvm::StringRef name,
                                          antlr4::ParserRuleContext *ctx) {
  return new fegen::FegenValue(isList, type, name, ctx);
}

llvm::StringRef fegen::FegenValue::getName() { return this->name; }

fegen::FegenRule::FegenRule(std::string content, fegen::FegenNode *src,
                            FegenParser::AlternativeContext *ctx)
    : content(content), src(src), ctx(ctx) {}

fegen::FegenRule *fegen::FegenRule::get(std::string content,
                                        fegen::FegenNode *src,
                                        FegenParser::AlternativeContext *ctx) {
  return new fegen::FegenRule(content, src, ctx);
}

bool fegen::FegenRule::addInput(fegen::FegenValue *input) {
  auto name = input->getName();
  if (this->inputs.count(name) == 0) {
    return false;
  }
  this->inputs.insert({name, input});
  return true;
}

bool fegen::FegenRule::addReturn(fegen::FegenValue *output) {
  auto name = output->getName();
  if (this->returns.count(name) == 0) {
    return false;
  }
  this->returns.insert({name, output});
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

fegen::FegenManager::~FegenManager() {
  // TODO
}