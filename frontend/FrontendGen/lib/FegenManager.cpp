#include "FegenManager.h"

fegen::FegenFunction::FegenFunction(llvm::StringRef name, std::vector<FegenType*>&& inputTypeList, FegenType* returnType) : 
    name(name), inputTypeList(inputTypeList), returnType(returnType){ }

fegen::FegenFunction* fegen::FegenFunction::get(llvm::StringRef name, std::vector<FegenType*> inputTypeList, FegenType* returnType) {
    return new fegen::FegenFunction(name, std::move(inputTypeList), returnType);
}

fegen::FegenOperation::FegenOperation(llvm::StringRef dialectName, llvm::StringRef operationName, std::vector<FegenValue*>&& arguments, std::vector<FegenValue*>&& results) :
    dialectName(dialectName), arguments(arguments), results(results){}

fegen::FegenOperation* fegen::FegenOperation::get(llvm::StringRef dialectName, llvm::StringRef operationName, std::vector<FegenValue*> arguments, std::vector<FegenValue*> results){
    return new fegen::FegenOperation(dialectName, operationName, std::move(arguments), std::move(results));
}

fegen::FegenType::FegenType(fegen::FegenType::TypeKind kind, llvm::StringRef dialectName, llvm::StringRef typeName, llvm::StringRef assemblyFormat, std::vector<FegenValue*>&& parameters, FegenParser::TypeDefinationDeclContext* ctx) :
    kind(kind), dialectName(dialectName), typeName(typeName), assemblyFormat(assemblyFormat), parameters(parameters), ctx(ctx){}

std::string fegen::FegenType::convertToCppType() {
    // TODO
    return std::string();
}

fegen::FegenType* fegen::FegenType::get(fegen::FegenType::TypeKind kind, llvm::StringRef dialectName, llvm::StringRef typeName, llvm::StringRef assemblyFormat, std::vector<FegenValue*> parameters, FegenParser::TypeDefinationDeclContext* ctx) {
    return new fegen::FegenType(kind, dialectName, typeName, assemblyFormat, std::move(parameters), ctx);
}

fegen::FegenValue::FegenValue(bool isList, fegen::FegenType* type, llvm::StringRef name, antlr4::ParserRuleContext* ctx) :
    isList(isList), type(type), name(name), ctx(ctx){}

fegen::FegenValue* fegen::FegenValue::get(bool isList, fegen::FegenType* type, llvm::StringRef name, antlr4::ParserRuleContext* ctx){
    return new fegen::FegenValue(isList, type, name, ctx);
}

fegen::FegenRule::FegenRule(std::string content, fegen::FegenNode* src, std::vector<fegen::FegenValue*>&& inputs, std::vector<fegen::FegenValue*>&& returns, FegenParser::ActionAltContext* ctx) :
    content(content), src(src), inputs(inputs), returns(returns), ctx(ctx) {}

fegen::FegenRule* fegen::FegenRule::get(std::string content, fegen::FegenNode* src, std::vector<fegen::FegenValue*> inputs, std::vector<fegen::FegenValue*> returns, FegenParser::ActionAltContext* ctx) {
    return new fegen::FegenRule(content, src, std::move(inputs), std::move(returns), ctx);
}

fegen::FegenNode::FegenNode(std::vector<fegen::FegenRule*>&& rules, FegenParser::RuleSpecContext* ctx) :
    rules(rules), ctx(ctx){}

fegen::FegenNode* fegen::FegenNode::get(std::vector<fegen::FegenRule*> rules, FegenParser::RuleSpecContext* ctx) {
    return new fegen::FegenNode(std::move(rules), ctx);
}
fegen::FegenNode* fegen::FegenNode::get(FegenParser::RuleSpecContext* ctx) {
    std::vector<fegen::FegenRule*> rules;
    return new fegen::FegenNode(std::move(rules), ctx);
}

void fegen::FegenNode::addFegenRule(fegen::FegenRule* rule) {
    this->rules.push_back(rule);
}

fegen::FegenManager::~FegenManager(){
    // TODO
}