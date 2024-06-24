#ifndef FEGEN_FEGENVISITOR_H
#define FEGEN_FEGENVISITOR_H

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include "FegenManager.h"
#include "Scope.h"
#include "FegenParserBaseVisitor.h"


using namespace antlr4;

namespace fegen {

/// @brief  check if params are right.
/// @param expected expected params.
/// @param actual actual params.
/// @return true if correct.
bool checkParams(std::vector<FegenValue*> &expected,
                 std::vector<FegenValue*> &actual);

/// @brief check if the type of elements in list are correct.
bool checkListLiteral(std::vector<FegenRightValue::Expression*> listLiteral);

class FegenVisitor : public FegenParserBaseVisitor {
private:
  FegenManager& manager;
  ScopeStack& sstack;

public:
  void emitG4() { this->manager.emitG4(); }

  FegenVisitor():manager(FegenManager::getManager()), sstack(ScopeStack::getScopeStack()) {
    this->manager.initbuiltinTypes();
  }

  std::any visitFegenDecl(FegenParser::FegenDeclContext *ctx) override {
    this->manager.setModuleName(ctx->identifier()->getText());
    return nullptr;
  }

  std::any
  visitParserRuleSpec(FegenParser::ParserRuleSpecContext *ctx) override {
    auto ruleList =
        std::any_cast<std::vector<FegenRule *>>(this->visit(ctx->ruleBlock()));
    auto ruleNode =
        FegenNode::get(ruleList, ctx, FegenNode::NodeType::PARSER_RULE);
    // set source node for rules
    for (auto rule : ruleList) {
      rule->setSrc(ruleNode);
    }
    this->manager.nodeMap.insert({ctx->ParserRuleName()->getText(), ruleNode});
    return nullptr;
  }

  std::any visitRuleAltList(FegenParser::RuleAltListContext *ctx) override {
    std::vector<FegenRule *> ruleList;
    for (auto alt : ctx->actionAlt()) {
      auto fegenRule = std::any_cast<FegenRule *>(this->visit(alt));
      ruleList.push_back(fegenRule);
    }
    return ruleList;
  }

  std::any visitActionAlt(FegenParser::ActionAltContext *ctx) override {
    auto rawRule = this->visit(ctx->alternative());
    if (ctx->actionBlock()) {
      auto blockValues = std::any_cast<
          std::tuple<std::vector<FegenValue *>, std::vector<FegenValue *>>>(
          this->visit(ctx->actionBlock()));
      auto inputs = std::get<0>(blockValues);
      auto returns = std::get<1>(blockValues);
      auto rule = std::any_cast<FegenRule *>(rawRule);
      for (auto in : inputs) {
        auto flag = rule->addInput(*in);
        if (!flag) { // TODO: error report
          std::cerr << "input of " << rule->getContent().str() << " \""
                    << in->getName() << "\" existed." << std::endl;
        }
      }
      for (auto out : returns) {
        auto flag = rule->addReturn(*out);
        if (!flag) { // TODO: error report
          std::cerr << "return of " << rule->getContent().str() << " \""
                    << out->getName() << "\" existed." << std::endl;
        }
      }
    }
    return rawRule;
  }

  // return tuple<vector<FegenValue*>, vector<FegenValue*>>
  std::any visitActionBlock(FegenParser::ActionBlockContext *ctx) override {
    std::vector<FegenValue *> inputs;
    std::vector<FegenValue *> returns;
    if (ctx->inputsSpec()) {
      inputs = std::move(std::any_cast<std::vector<FegenValue *>>(
          this->visit(ctx->inputsSpec())));
    }

    if (ctx->returnsSpec()) {
      returns = std::move(std::any_cast<std::vector<FegenValue *>>(
          this->visit(ctx->returnsSpec())));
    }

    if (ctx->actionSpec()) {
      this->visit(ctx->actionSpec());
    }
    return std::tuple(inputs, returns);
  }

  // return FegenRule Object
  // TODO: do more check
  std::any visitAlternative(FegenParser::AlternativeContext *ctx) override {
    auto content = ctx->getText();
    auto rule = FegenRule::get(content, nullptr, ctx);
    return rule;
  }

  std::any visitLexerRuleSpec(FegenParser::LexerRuleSpecContext *ctx) override {
    // create node, get rules from child, and insert to node map
    auto ruleList = std::any_cast<std::vector<FegenRule *>>(
        this->visit(ctx->lexerRuleBlock()));
    auto ruleNode =
        FegenNode::get(ruleList, ctx, FegenNode::NodeType::LEXER_RULE);
    // set source node for rules
    for (auto rule : ruleList) {
      rule->setSrc(ruleNode);
    }
    this->manager.nodeMap.insert({ctx->LexerRuleName()->getText(), ruleNode});
    return nullptr;
  }

  std::any visitLexerAltList(FegenParser::LexerAltListContext *ctx) override {
    std::vector<fegen::FegenRule *> ruleList;
    for (auto alt : ctx->lexerAlt()) {
      auto rule = fegen::FegenRule::get(alt->getText(), nullptr, alt);
      ruleList.push_back(rule);
    }
    return ruleList;
  }

  // return vector<FegenValue*>
  std::any visitVarDecls(FegenParser::VarDeclsContext *ctx) override {
    size_t varCount = ctx->typeSpec().size();
    std::vector<FegenValue *> valueList(varCount);
    for (size_t i = 0; i <= varCount - 1; i++) {
      auto ty = std::any_cast<fegen::FegenType>(this->visit(ctx->typeSpec(i)));
      auto varName = ctx->identifier(i)->getText();
      auto var = fegen::FegenValue::get(ty, varName,
                                        fegen::FegenRightValue::get());
      valueList.push_back(var);
    }

    return valueList;
  }

  // return fegen::FegenType
  std::any
  visitTypeInstanceSpec(FegenParser::TypeInstanceSpecContext *ctx) override {
    auto valueKind = ctx->valueKind()
                         ? std::any_cast<fegen::FegenType::TypeKind>(
                               this->visit(ctx->valueKind()))
                         : fegen::FegenType::TypeKind::CPP;
    auto typeInst =
        std::any_cast<fegen::FegenType>(this->visit(ctx->typeInstance()));
    typeInst.setTypeKind(valueKind);
    return typeInst;
  }

  // return fegen::FegenType::TypeKind
  std::any visitValueKind(FegenParser::ValueKindContext *ctx) override {
    auto kind = fegen::FegenType::TypeKind::ATTRIBUTE;
    if (ctx->CPP()) {
      kind = fegen::FegenType::TypeKind::CPP;
    } else if (ctx->OPERAND()) {
      kind = fegen::FegenType::TypeKind::OPERAND;
    }
    //otherwise: ATTRIBUTE
    return kind; 
  }

  // return fegen::FegenType
  std::any visitTypeInstance(FegenParser::TypeInstanceContext *ctx) override {
    if (ctx->typeTemplate()) { // typeTemplate (Less typeTemplateParam (Comma typeTemplateParam)* Greater)?
      auto typeTeplt =
          std::any_cast<fegen::FegenType>(this->visit(ctx->typeTemplate()));
      // get parameters
      std::vector<fegen::FegenValue*> paramList;
      for (auto paramCtx : ctx->typeTemplateParam()) {
        auto tepltParams =
            std::any_cast<fegen::FegenValue*>(this->visit(paramCtx));
        paramList.push_back(tepltParams);
      }

      // check parameters
      auto expectedParams = typeTeplt.getTypeDefination()->getParameters();
      if (!checkParams(expectedParams, paramList)) {
        std::cerr << "parameters error in context: " << ctx->getText()
                  << std::endl;
        exit(0);
      }
      // get FegenType of instance
      auto typeInst = FegenType::getInstanceType(typeTeplt.getTypeDefination(), paramList);
      return typeInst;
    }else if(ctx->identifier()) { // identifier
      auto varName = ctx->identifier()->getText();
      auto var = this->sstack.attemptFindVar(varName);
      if(var){
        if(var->getContentKind() == fegen::FegenRightValue::LiteralKind::TYPE){
          return var->getContent<fegen::FegenType>();
        }else{
          std::cerr << "variable " << varName << " is not a Type or TypeTemplate." << std::endl;
          exit(0);
          return nullptr;
        }
      }else{ // variable does not exist.
          std::cerr << "undefined variable: " << varName << std::endl;
          exit(0);
          return nullptr;
      }
    }else { // builtinTypeInstances
      return visitChildren(ctx);
    }
  }

  // return FegenValue*
  std::any visitTypeTemplateParam(FegenParser::TypeTemplateParamContext* ctx) override {
    if(ctx->builtinTypeInstances()){
      auto ty = std::any_cast<fegen::FegenType>(this->visit(ctx->builtinTypeInstances()));
      return fegen::FegenValue::get(ty, "param", fegen::FegenRightValue::get());
    }else{
      auto expr = std::any_cast<fegen::FegenRightValue::Expression*>(this->visit(ctx->expression()));
      return fegen::FegenValue::get(expr->exprType, "expression_tmp", 
      fegen::FegenRightValue(expr));
    }
  }

  // return fegen::FegenType
  std::any visitBuiltinTypeInstances(FegenParser::BuiltinTypeInstancesContext* ctx) override {
    if(ctx->BOOL()){
      return FegenType::getBoolType();
    }else if(ctx->INT()){
      return FegenType::getInt32Type();
    }else if(ctx->FLOAT()){
      return FegenType::getFloatType();
    }else if(ctx->DOUBLE()){
      return FegenType::getDoubleType();
    }else if(ctx->CHAR()){
      return FegenType::getCharType();
    }else if(ctx->STRING()){
      return FegenType::getStringType();
    }else{
      std::cerr << "error builtin type." << std::endl;
      return nullptr;
    }
  } 

  // return FegenType
  std::any visitTypeTemplate(FegenParser::TypeTemplateContext* ctx) override {
    if(ctx->prefixedName()){ // prefixedName
      if(ctx->prefixedName()->identifier().size() == 2){ // dialect.type
        auto dialectName = ctx->prefixedName()->identifier(0);
        auto typeDefName = ctx->prefixedName()->identifier(1);
        // TODO: return type from other dialect
        return nullptr;
      }else{ // type
        auto tyDef = this->sstack.attemptFindTypeDef(ctx->prefixedName()->identifier(0)->getText());
        return fegen::FegenType::getTemplateType(tyDef);
      }
    }else if(ctx->builtinTypeTemplate()){ // builtinTypeTemplate
      return this->visit(ctx->builtinTypeTemplate());
    }else{ // TYPE
      return fegen::FegenType::getMetaType();
    }
  }

  // return FegenType
  std::any visitBuiltinTypeTemplate(FegenParser::BuiltinTypeTemplateContext* ctx) override {
    if(ctx->INTEGER()){
      return fegen::FegenType::getIntegerTemplate();
    }else if(ctx->FLOATPOINT()){
      return fegen::FegenType::getFloatPointTemplate();
    }else if(ctx->TENSOR()){
      return fegen::FegenType::getTensorTemplate();
    }else if(ctx->VECTOR()){
      return fegen::FegenType::getVectorTemplate();
    }else{
      return nullptr;
    }
  }

  // return FegenType
  std::any visitCollectTypeSpec(FegenParser::CollectTypeSpecContext* ctx) override {
    auto kind = fegen::FegenType::TypeKind::CPP;
    if(ctx->valueKind()){
      kind = std::any_cast<fegen::FegenType::TypeKind>(this->visit(ctx->valueKind()));
    }
    auto ty = std::any_cast<fegen::FegenType>(this->visit(ctx->collectType()));
    ty.setTypeKind(kind);
    return ty;
  }

  // return FegenType
  std::any visitCollectType(FegenParser::CollectTypeContext* ctx) override {
    auto tyTmpt = std::any_cast<fegen::FegenTypeDefination*>(this->visit(ctx->collectProtoType()));
    if(ctx->expression()){ // return a type instance
      auto expr = std::any_cast<fegen::FegenRightValue::Expression*>(this->visit(ctx->expression()));
      return fegen::FegenType::getInstanceType(tyTmpt, {
        fegen::FegenValue::get(expr->exprType, "expr", fegen::FegenRightValue::get(expr))
      });
    }else{ // return a type template
      return fegen::FegenType::getTemplateType(tyTmpt);
    }
  }

  // return FegenTypeDefination*
  std::any visitCollectProtoType(FegenParser::CollectProtoTypeContext* ctx) override {
    if(ctx->ANY()){
      return this->manager.getTypeDefination(FEGEN_ANY);
    }else if(ctx->LIST()){
      return this->manager.getTypeDefination(FEGEN_LIST);
    }else{
      return this->manager.getTypeDefination(FEGEN_OPTINAL);
    }
  }

  // return FegenRightValue::Expression*
  std::any visitExpression(FegenParser::ExpressionContext* ctx) override {
    auto expr = std::any_cast<FegenRightValue::Expression*>(this->visit(ctx->andExpr(0)));
    for(size_t i = 1; i <= ctx->andExpr().size() - 1; i++){
      auto rhs = std::any_cast<FegenRightValue::Expression*>(this->visit(ctx->andExpr(i)));
      expr = FegenRightValue::ExpressionNode::binaryOperation(expr, rhs, FegenOperator::OR);
    }
    return expr;
  }

  // return FegenRightValue::Expression*
  std::any visitAndExpr(FegenParser::AndExprContext* ctx) override {
    auto expr = std::any_cast<FegenRightValue::Expression*>(this->visit(ctx->equExpr(0)));
    for(size_t i = 1; i <= ctx->equExpr().size() - 1; i++){
      auto rhs = std::any_cast<FegenRightValue::Expression*>(this->visit(ctx->equExpr(i)));
      expr = FegenRightValue::ExpressionNode::binaryOperation(expr, rhs, FegenOperator::AND);
    }
    return expr;
  }

  // return FegenRightValue::Expression*
  std::any visitEquExpr(FegenParser::EquExprContext* ctx) override {
    auto expr = std::any_cast<FegenRightValue::Expression*>(this->visit(ctx->compareExpr(0)));
    for(size_t i = 1; i <= ctx->compareExpr().size() - 1; i++){
      FegenOperator op;
      if(ctx->children[2 * i - 1]->getText() == "=="){
        op = FegenOperator::EQUAL;
      }else{
        op = FegenOperator::NOT_EQUAL;
      }
      auto rhs = std::any_cast<FegenRightValue::Expression*>(this->visit(ctx->compareExpr(i)));
      expr = FegenRightValue::ExpressionNode::binaryOperation(expr, rhs, op);
    }
    return expr;
  }

  // return FegenRightValue::Expression*
  std::any visitCompareExpr(FegenParser::CompareExprContext* ctx) override {
    auto expr = std::any_cast<FegenRightValue::Expression*>(this->visit(ctx->addExpr(0)));
    for(size_t i = 1; i <= ctx->addExpr().size() - 1; i++){
      FegenOperator op;
      auto opStr = ctx->children[2 * i - 1]->getText(); 
      if(opStr == "<"){
        op = FegenOperator::LESS;
      }else if(opStr == "<="){
        op = FegenOperator::LESS_EQUAL;
      }else if(opStr == "<="){
        op = FegenOperator::LESS_EQUAL;
      }else if(opStr == ">"){
        op = FegenOperator::GREATER;
      }else{
        op = FegenOperator::GREATER_EQUAL;
      }
      auto rhs = std::any_cast<FegenRightValue::Expression*>(this->visit(ctx->addExpr(i)));
      expr = FegenRightValue::ExpressionNode::binaryOperation(expr, rhs, op);
    }
    return expr;
  }

  // return FegenRightValue::Expression*
  std::any visitAddExpr(FegenParser::AddExprContext* ctx) override {
    auto expr = std::any_cast<FegenRightValue::Expression*>(this->visit(ctx->term(0)));
    for(size_t i = 1; i <= ctx->term().size() - 1; i++){
      FegenOperator op;
      auto opStr = ctx->children[2 * i - 1]->getText(); 
      if(opStr == "+"){
        op = FegenOperator::ADD;
      }else{
        op = FegenOperator::SUB;
      }
      auto rhs = std::any_cast<FegenRightValue::Expression*>(this->visit(ctx->term(i)));
      expr = FegenRightValue::ExpressionNode::binaryOperation(expr, rhs, op);
    }
    return expr;
  }

  // return FegenRightValue::Expression*
  std::any visitTerm(FegenParser::TermContext* ctx) override {
    auto expr = std::any_cast<FegenRightValue::Expression*>(this->visit(ctx->powerExpr(0)));
    for(size_t i = 1; i <= ctx->powerExpr().size() - 1; i++){
      FegenOperator op;
      auto opStr = ctx->children[2 * i - 1]->getText(); 
      if(opStr == "*"){
        op = FegenOperator::MUL;
      }else if(opStr == "/"){
        op = FegenOperator::DIV;
      }else{
        op = FegenOperator::MOD;
      }
      auto rhs = std::any_cast<FegenRightValue::Expression*>(this->visit(ctx->powerExpr(i)));
      expr = FegenRightValue::ExpressionNode::binaryOperation(expr, rhs, op);
    }
    return expr;
  }

  // return FegenRightValue::Expression*
  std::any visitPowerExpr(FegenParser::PowerExprContext* ctx) override {
    auto expr = std::any_cast<FegenRightValue::Expression*>(this->visit(ctx->unaryExpr(0)));
    for(size_t i = 1; i <= ctx->unaryExpr().size() - 1; i++){
      auto rhs = std::any_cast<FegenRightValue::Expression*>(this->visit(ctx->unaryExpr(i)));
      expr = FegenRightValue::ExpressionNode::binaryOperation(expr, rhs, FegenOperator::POWER);
    }
    return expr;
  }

  // return FegenRightValue::Expression*
  std::any visitUnaryExpr(FegenParser::UnaryExprContext* ctx) override {
    if(ctx->children.size() == 1 || ctx->Plus()){
      return this->visit(ctx->primaryExpr());
    }
    auto expr = std::any_cast<FegenRightValue::Expression*>(this->visit(ctx->primaryExpr()));
    FegenOperator op;
    if(ctx->Minus()){
      op = FegenOperator::NEG;
    }else{
      op = FegenOperator::NOT;
    }
    expr = FegenRightValue::ExpressionNode::unaryOperation(expr, op);
    return expr;
  }

  // return FegenRightValue::Expression*
  std::any visitParenSurroundedExpr(FegenParser::ParenSurroundedExprContext* ctx) override {
    return this->visit(ctx->expression());
  }

  // return FegenRightValue::Expression*
  std::any visitPrimaryExpr(FegenParser::PrimaryExprContext* ctx) override {
    if(ctx->identifier()){
      auto var = this->sstack.attemptFindVar(ctx->identifier()->getText());
      if(var){
        return (FegenRightValue::Expression*)fegen::FegenRightValue::ExpressionTerminal::get(var);
      }else{
        // TODO: error report
        std::cerr << "can not find variable: " << ctx->identifier()->getText() << "." << std::endl;
        exit(0);
        return nullptr;
      }
    }else if(ctx->typeSpec()){
      auto ty = std::any_cast<fegen::FegenType>(this->visit(ctx->typeSpec()));
      return (FegenRightValue::Expression*)FegenRightValue::ExpressionTerminal::get(ty);
    }else{ // constant, functionCall, parenSurroundedExpr,contextMethodInvoke, and variableAccess
      return this->visit(ctx->children[0]);
    }
  }

  // return ExpressionTerminal*
  std::any visitIntLiteral(FegenParser::IntLiteralContext* ctx) override {
    int number = std::stoi(ctx->getText());
    return (FegenRightValue::Expression*)fegen::FegenRightValue::ExpressionTerminal::get(number);
  }

  // return ExpressionTerminal*
  std::any visitRealLiteral(FegenParser::RealLiteralContext* ctx) override {
    double number = std::stod(ctx->getText());
    return (FegenRightValue::Expression*)fegen::FegenRightValue::ExpressionTerminal::get(float(number));
  }

  // return ExpressionTerminal*
  std::any visitCharLiteral(FegenParser::CharLiteralContext* ctx) override {
    std::string s = ctx->getText();
    // remove quotation marks
    std::string strWithoutQuotation = s.substr(1, s.size() - 2);
    return (FegenRightValue::Expression*)fegen::FegenRightValue::ExpressionTerminal::get(strWithoutQuotation);
  }

  // return ExpressionTerminal*
  std::any visitBoolLiteral(FegenParser::BoolLiteralContext* ctx) override {
    int content = 0;
    if(ctx->getText() == "true"){
      content = 1;
    }
    return (FegenRightValue::Expression*)fegen::FegenRightValue::ExpressionTerminal::get(content);
  }

  // return ExpressionTerminal*
  std::any visitListLiteral(FegenParser::ListLiteralContext* ctx) override {
    std::vector<fegen::FegenRightValue::Expression*> elements;
    for(auto exprCtx : ctx->expression()){
      auto expr = std::any_cast<fegen::FegenRightValue::Expression*>(this->visit(exprCtx));
      elements.push_back(expr);
    }
    return (FegenRightValue::Expression*)fegen::FegenRightValue::ExpressionTerminal::get(elements);
  }

  std::any visitActionSpec(FegenParser::ActionSpecContext *ctx) override {
    return nullptr;
  }

    std::any visitFunctionDecl(FegenParser::FunctionDeclContext *ctx) override{
        auto returnType = std::any_cast<fegen::FegenType*>(this->visit(ctx->typeSpec()));
        auto functionName = std::any_cast<llvm::StringRef>(this->visit(ctx->funcName()));
        auto functionParams = std::any_cast<std::map<fegen::FegenType*, std::string>>(this->visit(ctx->funcParams()));
        this->visit(ctx->statementBlock());

        fegen::FegenFunction* function = fegen::FegenFunction::get(functionName, functionParams, returnType);
        manager.functions.push_back(function);
    }

    // TODO: 类型存在性检查 
    std::any visitTypeInstanceSpec(FegenParser::TypeInstanceSpecContext *ctx) override{
        if(ctx->typeInstance())
            return this->visit(ctx->typeInstance());
        else {
            // TODO: visit typeTemplate
        }
    }

    std::any visitFuncName(FegenParser::FuncNameContext *ctx) override{
        return ctx->identifier()->getText();
    }

    std::any visitFuncParams(FegenParser::FuncParamsContext *ctx) override{
        std::map<fegen::FegenType *, std::string> params;
        if(ctx->children.size() == 2){
            auto typespec = std::any_cast<fegen::FegenType*>(this->visit(ctx->typeSpec(0)));
            auto paramname = ctx->identifier(0)->getText();
            params.insert(std::pair(typespec, paramname));
            return params;
        }
        for(int i = 0; i < ctx->typeSpec().size(); i++){
            auto typespec = std::any_cast<fegen::FegenType*>(this->visit(ctx->typeSpec(i)));
            auto paramname = ctx->identifier(i)->getText();
            params.insert(std::pair(typespec, paramname));
        }
        return params;
    }

    std::any visitStatementBlock(FegenParser::StatementBlockContext *ctx) override{
        
    }

};
} // namespace fegen
#endif