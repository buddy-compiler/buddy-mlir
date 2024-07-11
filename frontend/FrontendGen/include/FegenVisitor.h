#ifndef FEGEN_FEGENVISITOR_H
#define FEGEN_FEGENVISITOR_H

#include <any>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include "FegenManager.h"
#include "FegenParser.h"
#include "FegenParserBaseVisitor.h"
#include "Scope.h"

using namespace antlr4;

namespace fegen {

/// @brief  check if params are right.
/// @param expected expected params.
/// @param actual actual params.
/// @return true if correct.
bool checkParams(std::vector<FegenValue *> &expected,
                 std::vector<FegenValue *> &actual);

/// @brief check if the type of elements in list are correct.
bool checkListLiteral(std::vector<FegenRightValue::Expression *> listLiteral);

class FegenVisitor : public FegenParserBaseVisitor {
private:
  FegenManager &manager;
  ScopeStack &sstack;

public:
  void emitG4() { this->manager.emitG4(); }
  void emitTypeDefination() { this->manager.emitTypeDefination(); }
  void emitDialectDefination() { this->manager.emitDialectDefination(); }
  void emitOpDefination() { this->manager.emitOpDefination(); }

  FegenVisitor()
      : manager(FegenManager::getManager()),
        sstack(ScopeStack::getScopeStack()) {
    this->manager.initbuiltinTypes();
  }

  std::any visitTypeDefinationDecl(
      FegenParser::TypeDefinationDeclContext *ctx) override {
    auto typeName = ctx->typeDefinationName()->getText();
    auto tyDef = std::any_cast<FegenTypeDefination *>(
        this->visit(ctx->typeDefinationBlock()));
    // set name and ctx for type defination
    tyDef->setName(typeName);
    tyDef->setCtx(ctx);
    // add defination to manager map
    this->manager.typeDefMap.insert({typeName, tyDef});
    return nullptr;
  }

  // return FegenTypeDefination*
  std::any visitTypeDefinationBlock(
      FegenParser::TypeDefinationBlockContext *ctx) override {
    auto params = std::any_cast<std::vector<FegenValue *>>(
        this->visit(ctx->parametersSpec()));
    auto tyDef =
        FegenTypeDefination::get(this->manager.moduleName, "", params, nullptr);
    return tyDef;
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
      inputs = std::any_cast<std::vector<FegenValue *>>(
          this->visit(ctx->inputsSpec()));
    }

    if (ctx->returnsSpec()) {
      returns = std::any_cast<std::vector<FegenValue *>>(
          this->visit(ctx->returnsSpec()));
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
    std::vector<FegenValue *> valueList;
    for (size_t i = 0; i <= varCount - 1; i++) {
      auto ty = std::any_cast<fegen::FegenType>(this->visit(ctx->typeSpec(i)));
      auto varName = ctx->identifier(i)->getText();
      auto var =
          fegen::FegenValue::get(ty, varName, fegen::FegenRightValue::get());
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
    // otherwise: ATTRIBUTE
    return kind;
  }

  // return fegen::FegenType
  std::any visitTypeInstance(FegenParser::TypeInstanceContext *ctx) override {
    if (ctx->typeTemplate()) { // typeTemplate (Less typeTemplateParam (Comma
                               // typeTemplateParam)* Greater)?
      auto typeTeplt =
          std::any_cast<fegen::FegenType>(this->visit(ctx->typeTemplate()));
      // get parameters
      std::vector<fegen::FegenValue *> paramList;
      for (auto paramCtx : ctx->typeTemplateParam()) {
        auto tepltParams =
            std::any_cast<fegen::FegenValue *>(this->visit(paramCtx));
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
      auto typeInst =
          FegenType::getInstanceType(typeTeplt.getTypeDefination(), paramList);
      return typeInst;
    } else if (ctx->identifier()) { // identifier
      auto varName = ctx->identifier()->getText();
      auto var = this->sstack.attemptFindVar(varName);
      if (var) {
        if (var->getContentKind() ==
            fegen::FegenRightValue::LiteralKind::TYPE) {
          return var->getContent<fegen::FegenType>();
        } else {
          std::cerr << "variable " << varName
                    << " is not a Type or TypeTemplate." << std::endl;
          exit(0);
          return nullptr;
        }
      } else { // variable does not exist.
        std::cerr << "undefined variable: " << varName << std::endl;
        exit(0);
        return nullptr;
      }
    } else { // builtinTypeInstances
      return visitChildren(ctx);
    }
  }

  // return FegenValue*
  std::any
  visitTypeTemplateParam(FegenParser::TypeTemplateParamContext *ctx) override {
    if (ctx->builtinTypeInstances()) {
      auto ty = std::any_cast<fegen::FegenType>(
          this->visit(ctx->builtinTypeInstances()));
      return fegen::FegenValue::get(ty, "param", fegen::FegenRightValue::get());
    } else {
      auto expr = std::any_cast<fegen::FegenRightValue::Expression *>(
          this->visit(ctx->expression()));
      return fegen::FegenValue::get(expr->exprType, "expression_tmp",
                                    fegen::FegenRightValue(expr));
    }
  }

  // return fegen::FegenType
  std::any visitBuiltinTypeInstances(
      FegenParser::BuiltinTypeInstancesContext *ctx) override {
    if (ctx->BOOL()) {
      return FegenType::getBoolType();
    } else if (ctx->INT()) {
      return FegenType::getInt32Type();
    } else if (ctx->FLOAT()) {
      return FegenType::getFloatType();
    } else if (ctx->DOUBLE()) {
      return FegenType::getDoubleType();
    } else if (ctx->CHAR()) {
      return FegenType::getCharType();
    } else if (ctx->STRING()) {
      return FegenType::getStringType();
    } else {
      std::cerr << "error builtin type." << std::endl;
      return nullptr;
    }
  }

  // return FegenType
  std::any visitTypeTemplate(FegenParser::TypeTemplateContext *ctx) override {
    if (ctx->prefixedName()) {                             // prefixedName
      if (ctx->prefixedName()->identifier().size() == 2) { // dialect.type
        // TODO: return type from other dialect
        return nullptr;
      } else { // type
        auto tyDef = this->sstack.attemptFindTypeDef(
            ctx->prefixedName()->identifier(0)->getText());
        return fegen::FegenType::getTemplateType(tyDef);
      }
    } else if (ctx->builtinTypeTemplate()) { // builtinTypeTemplate
      return this->visit(ctx->builtinTypeTemplate());
    } else { // TYPE
      return fegen::FegenType::getMetaType();
    }
  }

  // return FegenType
  std::any visitBuiltinTypeTemplate(
      FegenParser::BuiltinTypeTemplateContext *ctx) override {
    if (ctx->INTEGER()) {
      return fegen::FegenType::getIntegerTemplate();
    } else if (ctx->FLOATPOINT()) {
      return fegen::FegenType::getFloatPointTemplate();
    } else if (ctx->TENSOR()) {
      // return fegen::FegenType::getTensorTemplate();
      return fegen::FegenType::getPlaceHolder();
    } else if (ctx->VECTOR()) {
      // return fegen::FegenType::getVectorTemplate();
      return fegen::FegenType::getPlaceHolder();
    } else {
      return nullptr;
    }
  }

  // return FegenType
  std::any
  visitCollectTypeSpec(FegenParser::CollectTypeSpecContext *ctx) override {
    auto kind = fegen::FegenType::TypeKind::CPP;
    if (ctx->valueKind()) {
      kind = std::any_cast<fegen::FegenType::TypeKind>(
          this->visit(ctx->valueKind()));
    }
    auto ty = std::any_cast<fegen::FegenType>(this->visit(ctx->collectType()));
    ty.setTypeKind(kind);
    return ty;
  }

  // return FegenType
  std::any visitCollectType(FegenParser::CollectTypeContext *ctx) override {
    auto expr = std::any_cast<fegen::FegenRightValue::Expression *>(
        this->visit(ctx->expression()));
    if (ctx->collectProtoType()->ANY()) {
      std::vector<fegen::FegenType> tys;
      // TODO: reprot error
      assert(expr->getKind() == fegen::FegenRightValue::LiteralKind::VECTOR);
      auto exprs =
          std::any_cast<std::vector<fegen::FegenRightValue::Expression *>>(
              expr->getContent());
      for (auto expr : exprs) {
        auto ty = std::any_cast<fegen::FegenType>(expr->getContent());
        tys.push_back(ty);
      }
      return fegen::FegenType::getAnyType(tys);
    } else if (ctx->collectProtoType()->LIST()) {
      auto ty = std::any_cast<fegen::FegenType>(expr->getContent());
      return fegen::FegenType::getListType(ty);
    } else { // optional
      auto ty = std::any_cast<fegen::FegenType>(expr->getContent());
      return fegen::FegenType::getOptionalType(ty);
    }
  }

  // return FegenRightValue::Expression*
  std::any visitExpression(FegenParser::ExpressionContext *ctx) override {
    auto expr = std::any_cast<FegenRightValue::Expression *>(
        this->visit(ctx->andExpr(0)));
    for (size_t i = 1; i <= ctx->andExpr().size() - 1; i++) {
      auto rhs = std::any_cast<FegenRightValue::Expression *>(
          this->visit(ctx->andExpr(i)));
      expr = FegenRightValue::ExpressionNode::binaryOperation(
          expr, rhs, FegenOperator::OR);
    }
    return expr;
  }

  // return FegenRightValue::Expression*
  std::any visitAndExpr(FegenParser::AndExprContext *ctx) override {
    auto expr = std::any_cast<FegenRightValue::Expression *>(
        this->visit(ctx->equExpr(0)));
    for (size_t i = 1; i <= ctx->equExpr().size() - 1; i++) {
      auto rhs = std::any_cast<FegenRightValue::Expression *>(
          this->visit(ctx->equExpr(i)));
      expr = FegenRightValue::ExpressionNode::binaryOperation(
          expr, rhs, FegenOperator::AND);
    }
    return expr;
  }

  // return FegenRightValue::Expression*
  std::any visitEquExpr(FegenParser::EquExprContext *ctx) override {
    auto expr = std::any_cast<FegenRightValue::Expression *>(
        this->visit(ctx->compareExpr(0)));
    for (size_t i = 1; i <= ctx->compareExpr().size() - 1; i++) {
      FegenOperator op;
      if (ctx->children[2 * i - 1]->getText() == "==") {
        op = FegenOperator::EQUAL;
      } else {
        op = FegenOperator::NOT_EQUAL;
      }
      auto rhs = std::any_cast<FegenRightValue::Expression *>(
          this->visit(ctx->compareExpr(i)));
      expr = FegenRightValue::ExpressionNode::binaryOperation(expr, rhs, op);
    }
    return expr;
  }

  // return FegenRightValue::Expression*
  std::any visitCompareExpr(FegenParser::CompareExprContext *ctx) override {
    auto expr = std::any_cast<FegenRightValue::Expression *>(
        this->visit(ctx->addExpr(0)));
    for (size_t i = 1; i <= ctx->addExpr().size() - 1; i++) {
      FegenOperator op;
      auto opStr = ctx->children[2 * i - 1]->getText();
      if (opStr == "<") {
        op = FegenOperator::LESS;
      } else if (opStr == "<=") {
        op = FegenOperator::LESS_EQUAL;
      } else if (opStr == "<=") {
        op = FegenOperator::LESS_EQUAL;
      } else if (opStr == ">") {
        op = FegenOperator::GREATER;
      } else {
        op = FegenOperator::GREATER_EQUAL;
      }
      auto rhs = std::any_cast<FegenRightValue::Expression *>(
          this->visit(ctx->addExpr(i)));
      expr = FegenRightValue::ExpressionNode::binaryOperation(expr, rhs, op);
    }
    return expr;
  }

  // return FegenRightValue::Expression*
  std::any visitAddExpr(FegenParser::AddExprContext *ctx) override {
    auto expr =
        std::any_cast<FegenRightValue::Expression *>(this->visit(ctx->term(0)));
    for (size_t i = 1; i <= ctx->term().size() - 1; i++) {
      FegenOperator op;
      auto opStr = ctx->children[2 * i - 1]->getText();
      if (opStr == "+") {
        op = FegenOperator::ADD;
      } else {
        op = FegenOperator::SUB;
      }
      auto rhs = std::any_cast<FegenRightValue::Expression *>(
          this->visit(ctx->term(i)));
      expr = FegenRightValue::ExpressionNode::binaryOperation(expr, rhs, op);
    }
    return expr;
  }

  // return FegenRightValue::Expression*
  std::any visitTerm(FegenParser::TermContext *ctx) override {
    auto expr = std::any_cast<FegenRightValue::Expression *>(
        this->visit(ctx->powerExpr(0)));
    for (size_t i = 1; i <= ctx->powerExpr().size() - 1; i++) {
      FegenOperator op;
      auto opStr = ctx->children[2 * i - 1]->getText();
      if (opStr == "*") {
        op = FegenOperator::MUL;
      } else if (opStr == "/") {
        op = FegenOperator::DIV;
      } else {
        op = FegenOperator::MOD;
      }
      auto rhs = std::any_cast<FegenRightValue::Expression *>(
          this->visit(ctx->powerExpr(i)));
      expr = FegenRightValue::ExpressionNode::binaryOperation(expr, rhs, op);
    }
    return expr;
  }

  // return FegenRightValue::Expression*
  std::any visitPowerExpr(FegenParser::PowerExprContext *ctx) override {
    auto expr = std::any_cast<FegenRightValue::Expression *>(
        this->visit(ctx->unaryExpr(0)));
    for (size_t i = 1; i <= ctx->unaryExpr().size() - 1; i++) {
      auto rhs = std::any_cast<FegenRightValue::Expression *>(
          this->visit(ctx->unaryExpr(i)));
      expr = FegenRightValue::ExpressionNode::binaryOperation(
          expr, rhs, FegenOperator::POWER);
    }
    return expr;
  }

  // return FegenRightValue::Expression*
  std::any visitUnaryExpr(FegenParser::UnaryExprContext *ctx) override {
    if (ctx->children.size() == 1 || ctx->Plus()) {
      return this->visit(ctx->primaryExpr());
    }
    auto expr = std::any_cast<FegenRightValue::Expression *>(
        this->visit(ctx->primaryExpr()));
    FegenOperator op;
    if (ctx->Minus()) {
      op = FegenOperator::NEG;
    } else {
      op = FegenOperator::NOT;
    }
    expr = FegenRightValue::ExpressionNode::unaryOperation(expr, op);
    return expr;
  }

  // return FegenRightValue::Expression*
  std::any visitParenSurroundedExpr(
      FegenParser::ParenSurroundedExprContext *ctx) override {
    return this->visit(ctx->expression());
  }

  // return FegenRightValue::Expression*
  std::any visitPrimaryExpr(FegenParser::PrimaryExprContext *ctx) override {
    if (ctx->identifier()) {
      auto name = ctx->identifier()->getText();
      auto var = this->sstack.attemptFindVar(name);
      if (var) {
        return (FegenRightValue::Expression *)
            fegen::FegenRightValue::ExpressionTerminal::get(var);
      } else {
        auto tyDef = this->sstack.attemptFindTypeDef(name);
        if (tyDef) {
          auto tyVar = fegen::FegenType::getTemplateType(tyDef);
          return fegen::FegenValue::get(fegen::FegenType::getMetaTemplateType(),
                                        "", fegen::FegenRightValue::get(tyVar));
        } else {
          // TODO: error report
          std::cerr << "can not find variable: " << ctx->identifier()->getText()
                    << "." << std::endl;
          exit(0);
          return nullptr;
        }
      }
    } else if (ctx->typeSpec()) {
      auto ty = std::any_cast<fegen::FegenType>(this->visit(ctx->typeSpec()));
      return (FegenRightValue::Expression *)
          FegenRightValue::ExpressionTerminal::get(ty);
    } else { // constant, functionCall, parenSurroundedExpr,contextMethodInvoke,
             // and variableAccess
      return this->visit(ctx->children[0]);
    }
  }

  // return ExpressionTerminal*
  std::any visitIntLiteral(FegenParser::IntLiteralContext *ctx) override {
    int number = std::stoi(ctx->getText());
    return (FegenRightValue::Expression *)
        fegen::FegenRightValue::ExpressionTerminal::get(number);
  }

  // return ExpressionTerminal*
  std::any visitRealLiteral(FegenParser::RealLiteralContext *ctx) override {
    double number = std::stod(ctx->getText());
    return (FegenRightValue::Expression *)
        fegen::FegenRightValue::ExpressionTerminal::get(float(number));
  }

  // return ExpressionTerminal*
  std::any visitCharLiteral(FegenParser::CharLiteralContext *ctx) override {
    std::string s = ctx->getText();
    // remove quotation marks
    std::string strWithoutQuotation = s.substr(1, s.size() - 2);
    return (FegenRightValue::Expression *)
        fegen::FegenRightValue::ExpressionTerminal::get(strWithoutQuotation);
  }

  // return ExpressionTerminal*
  std::any visitBoolLiteral(FegenParser::BoolLiteralContext *ctx) override {
    int content = 0;
    if (ctx->getText() == "true") {
      content = 1;
    }
    return (FegenRightValue::Expression *)
        fegen::FegenRightValue::ExpressionTerminal::get(content);
  }

  // return ExpressionTerminal*
  std::any visitListLiteral(FegenParser::ListLiteralContext *ctx) override {
    std::vector<fegen::FegenRightValue::Expression *> elements;
    for (auto exprCtx : ctx->expression()) {
      auto expr = std::any_cast<fegen::FegenRightValue::Expression *>(
          this->visit(exprCtx));
      elements.push_back(expr);
    }
    return (FegenRightValue::Expression *)
        fegen::FegenRightValue::ExpressionTerminal::get(elements);
  }

  std::any visitActionSpec(FegenParser::ActionSpecContext *ctx) override {
    return nullptr;
  }

  std::any visitFunctionDecl(FegenParser::FunctionDeclContext *ctx) override {
    sstack.pushScope();
    auto returnType =
        std::any_cast<fegen::FegenType>(this->visit(ctx->typeSpec()));
    auto functionName =
        std::any_cast<std::string>(this->visit(ctx->funcName()));
    auto hasfunc = manager.functionMap.find(functionName);
    if (hasfunc != manager.functionMap.end()) {
      std::cerr << "The function name \" " << functionName
                << "\" has already been used. Please use another name."
                << std::endl;
      exit(0);
      return nullptr;
    }
    auto functionParams = std::any_cast<std::vector<fegen::FegenValue *>>(
        this->visit(ctx->funcParams()));
    this->visit(ctx->statementBlock());

    fegen::FegenFunction *function =
        fegen::FegenFunction::get(functionName, functionParams, &returnType);
    manager.functionMap.insert(std::pair{functionName, function});
    sstack.popScope();
    return nullptr;
  }

  std::any visitFuncName(FegenParser::FuncNameContext *ctx) override {
    auto functionName = ctx->identifier()->getText();
    return functionName;
  }

  std::any visitFuncParams(FegenParser::FuncParamsContext *ctx) override {
    std::vector<fegen::FegenValue *> paramsList = {};

    for (size_t i = 0; i < ctx->typeSpec().size(); i++) {
      auto paramType =
          std::any_cast<fegen::FegenType>(this->visit(ctx->typeSpec(i)));
      auto paramName = ctx->identifier(i)->getText();
      auto param = fegen::FegenValue::get(paramType, paramName, nullptr);
      paramsList.push_back(param);
      sstack.attemptAddVar(param);
    }
    return paramsList;
  }

  std::any visitVarDeclStmt(FegenParser::VarDeclStmtContext *ctx) override {
    auto varType =
        std::any_cast<fegen::FegenType>(this->visit(ctx->typeSpec()));
    auto varName = ctx->identifier()->getText();
    fegen::FegenValue *var;
    if (ctx->expression()) {
      auto varContent = std::any_cast<fegen::FegenRightValue::Expression *>(
          this->visit(ctx->expression()));
      if (!fegen::FegenType::isSameType(&varType, &varContent->exprType)) {
        std::cerr << "The variabel \"" << varName << "\" need \""
                  << varType.getTypeName() << " \" type rightvalue. But now is " << varContent->exprType.getTypeName()
                  << std::endl;
        exit(0);
        return nullptr;
      }
      var = fegen::FegenValue::get(varType, varName, varContent);
    } else {
      var = fegen::FegenValue::get(varType, varName, nullptr);
    }
    sstack.attemptAddVar(var);
    manager.stmtContentMap.insert(std::pair{ctx, var});
    return var;
  }

  std::any visitAssignStmt(FegenParser::AssignStmtContext *ctx) override {
    auto varName = ctx->identifier()->getText();
    auto varcontent = std::any_cast<fegen::FegenRightValue::Expression *>(
        this->visit(ctx->expression()));
    auto var = sstack.attemptFindVar(varName);
    if (!fegen::FegenType::isSameType(&var->getType(), &varcontent->exprType)) {
      std::cerr << "The variabel \" " << varName << "\" need \""
                << var->getType().getTypeName() << " \" type rightvalue."
                << std::endl;
      exit(0);
      return nullptr;
    }
    fegen::FegenValue *stmt =
        fegen::FegenValue::get(var->getType(), varName, varcontent);
    manager.stmtContentMap.insert(std::pair{ctx, stmt});

    return stmt;
  }

  std::any visitFunctionCall(FegenParser::FunctionCallContext *ctx) override {
    std::vector<fegen::FegenRightValue::Expression *> parasList = {};
    fegen::FegenFunction *function;
    auto functionName =
        std::any_cast<std::string>(this->visit(ctx->funcName()));
    auto hasFunc = manager.functionMap.find(functionName);
    if(hasFunc == manager.functionMap.end()){
        std::cerr << "The called function \"" << functionName
                  << "\" is not exist." << std::endl;
        exit(0);
        return nullptr;
    }
    function = hasFunc->second;
    auto paramsNum = ctx->expression().size();
    auto paraList = function->getInputTypeList();
    if (paramsNum > 0) {
      for (size_t i = 0; i < paramsNum; i++) {
        auto oprand = std::any_cast<fegen::FegenRightValue::Expression *>(
            this->visit(ctx->expression(i)));
        parasList.push_back(oprand);
      }
      size_t len1 = paraList.size();
      size_t len2 = parasList.size();
      if (len1 != len2) {
        std::cerr << "The function \"" << functionName
                  << "\" parameter count mismatch." << std::endl;
        exit(0);
        return nullptr;
      }
      for (size_t i = 0; i < len1; i++) {
        if (!fegen::FegenType::isSameType(&paraList[i]->getType(),
                                          &parasList[i]->exprType)) {
          std::cerr << "The function \"" << functionName << "\" parameter" << i
                    << " type mismatch." << std::endl;
          exit(0);
          return nullptr;
        }
      }
    }
    auto returnType = function->getReturnType();
    fegen::FegenFunction *funcCall =
        fegen::FegenFunction::get(functionName, paraList, returnType);
    manager.stmtContentMap.insert(std::pair{ctx, funcCall});
    return returnType;
  }

  std::any visitOpInvokeStmt(FegenParser::OpInvokeStmtContext *ctx) override {
    return nullptr;
  }

  std::any visitIfStmt(FegenParser::IfStmtContext *ctx) override {
    sstack.pushScope();
    this->visit(ctx->expression(0));
    this->visit(ctx->statementBlock(0));
    for (size_t i = 1; i <= ctx->expression().size() - 1; i++) {
      this->visit(ctx->expression(i));
      this->visit(ctx->statementBlock(i));
    }
    if (ctx->statementBlock(ctx->expression().size() + 1))
      this->visit(ctx->statementBlock(ctx->expression().size() + 1));
    sstack.popScope();

    return nullptr;
  }

  std::any visitForStmt(FegenParser::ForStmtContext *ctx) override {
    sstack.pushScope();
    if (ctx->varDeclStmt()) {
      this->visit(ctx->varDeclStmt());
      this->visit(ctx->expression());
      this->visit(ctx->assignStmt(0));
    } else {
      this->visit(ctx->assignStmt(0));
      this->visit(ctx->expression());
      this->visit(ctx->assignStmt(1));
    }
    this->visit(ctx->statementBlock());
    sstack.popScope();

    return nullptr;
  }

  std::any visitOpDecl(FegenParser::OpDeclContext *ctx) override {
    auto opName = ctx->opName()->getText();
    auto opDef =
        std::any_cast<fegen::FegenOperation *>(this->visit(ctx->opBlock()));
    opDef->setOpName(opName);
    bool success = this->manager.addOperationDefination(opDef);
    if (!success) {
      // TODO: error report
      std::cerr << "operation " << opName << " already exist." << std::endl;
    }
    return nullptr;
  }

  // return FegenOperation*
  std::any visitOpBlock(FegenParser::OpBlockContext *ctx) override {
    std::vector<fegen::FegenValue *> args;
    std::vector<fegen::FegenValue *> res;
    if (ctx->argumentSpec()) {
      args = std::any_cast<std::vector<fegen::FegenValue *>>(
          this->visit(ctx->argumentSpec()));
    }
    if (ctx->resultSpec()) {
      res = std::any_cast<std::vector<fegen::FegenValue *>>(
          this->visit(ctx->resultSpec()));
    }
    return fegen::FegenOperation::get("", args, res, ctx->bodySpec());
  }
};
} // namespace fegen
#endif