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
bool checkParams(std::vector<Value *> &expected,
                 std::vector<Value *> &actual);

/// @brief check if the type of elements in list are correct.
bool checkListLiteral(
    std::vector<std::shared_ptr<fegen::RightValue::Expression>>
        &listLiteral);

class FegenVisitor : public FegenParserBaseVisitor {
private:
  Manager &manager;
  ScopeStack &sstack;

public:
  void emitG4() { this->manager.emitG4(); }
  void emitTypeDefination() { this->manager.emitTypeDefination(); }
  void emitDialectDefination() { this->manager.emitDialectDefination(); }
  void emitOpDefination() { this->manager.emitOpDefination(); }

  FegenVisitor()
      : manager(Manager::getManager()),
        sstack(ScopeStack::getScopeStack()) {
    this->manager.initbuiltinTypes();
  }

  std::any visitTypeDefinationDecl(
      FegenParser::TypeDefinationDeclContext *ctx) override {
    auto typeName = ctx->typeDefinationName()->getText();
    auto tyDef = std::any_cast<TypeDefination *>(
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
    auto params = std::any_cast<std::vector<Value *>>(
        this->visit(ctx->parametersSpec()));
    auto tyDef =
        TypeDefination::get(this->manager.moduleName, "", params, nullptr);
    return tyDef;
  }

  std::any visitFegenDecl(FegenParser::FegenDeclContext *ctx) override {
    this->manager.setModuleName(ctx->identifier()->getText());
    return nullptr;
  }

  std::any
  visitParserRuleSpec(FegenParser::ParserRuleSpecContext *ctx) override {
    auto ruleList =
        std::any_cast<std::vector<ParserRule *>>(this->visit(ctx->ruleBlock()));
    auto ruleNode =
        ParserNode::get(ruleList, ctx, ParserNode::NodeType::PARSER_RULE);
    // set source node for rules
    for (auto rule : ruleList) {
      rule->setSrc(ruleNode);
    }
    this->manager.nodeMap.insert({ctx->ParserRuleName()->getText(), ruleNode});
    return nullptr;
  }

  std::any visitRuleAltList(FegenParser::RuleAltListContext *ctx) override {
    std::vector<ParserRule *> ruleList;
    for (auto alt : ctx->actionAlt()) {
      auto fegenRule = std::any_cast<ParserRule *>(this->visit(alt));
      ruleList.push_back(fegenRule);
    }
    return ruleList;
  }

  std::any visitActionAlt(FegenParser::ActionAltContext *ctx) override {
    auto rawRule = this->visit(ctx->alternative());
    if (ctx->actionBlock()) {
      auto blockValues = std::any_cast<
          std::tuple<std::vector<Value *>, std::vector<Value *>>>(
          this->visit(ctx->actionBlock()));
      auto inputs = std::get<0>(blockValues);
      auto returns = std::get<1>(blockValues);
      auto rule = std::any_cast<ParserRule *>(rawRule);
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
    std::vector<Value *> inputs;
    std::vector<Value *> returns;
    if (ctx->inputsSpec()) {
      inputs = std::any_cast<std::vector<Value *>>(
          this->visit(ctx->inputsSpec()));
    }

    if (ctx->returnsSpec()) {
      returns = std::any_cast<std::vector<Value *>>(
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
    auto rule = ParserRule::get(content, nullptr, ctx);
    return rule;
  }

  std::any visitLexerRuleSpec(FegenParser::LexerRuleSpecContext *ctx) override {
    // create node, get rules from child, and insert to node map
    auto ruleList = std::any_cast<std::vector<ParserRule *>>(
        this->visit(ctx->lexerRuleBlock()));
    auto ruleNode =
        ParserNode::get(ruleList, ctx, ParserNode::NodeType::LEXER_RULE);
    // set source node for rules
    for (auto rule : ruleList) {
      rule->setSrc(ruleNode);
    }
    this->manager.nodeMap.insert({ctx->LexerRuleName()->getText(), ruleNode});
    return nullptr;
  }

  std::any visitLexerAltList(FegenParser::LexerAltListContext *ctx) override {
    std::vector<fegen::ParserRule *> ruleList;
    for (auto alt : ctx->lexerAlt()) {
      auto rule = fegen::ParserRule::get(alt->getText(), nullptr, alt);
      ruleList.push_back(rule);
    }
    return ruleList;
  }

  // return vector<FegenValue*>
  std::any visitVarDecls(FegenParser::VarDeclsContext *ctx) override {
    size_t varCount = ctx->typeSpec().size();
    std::vector<Value *> valueList;
    for (size_t i = 0; i <= varCount - 1; i++) {
      auto ty = std::any_cast<fegen::Type>(this->visit(ctx->typeSpec(i)));
      auto varName = ctx->identifier(i)->getText();
      auto var = fegen::Value::get(
          ty, varName, fegen::RightValue::getPlaceHolder());
      valueList.push_back(var);
    }

    return valueList;
  }

  // return fegen::FegenType
  std::any
  visitTypeInstanceSpec(FegenParser::TypeInstanceSpecContext *ctx) override {
    auto valueKind = ctx->valueKind()
                         ? std::any_cast<fegen::Type::TypeKind>(
                               this->visit(ctx->valueKind()))
                         : fegen::Type::TypeKind::CPP;
    auto typeInst =
        std::any_cast<fegen::Type>(this->visit(ctx->typeInstance()));
    typeInst.setTypeKind(valueKind);
    return typeInst;
  }

  // return fegen::FegenType::TypeKind
  std::any visitValueKind(FegenParser::ValueKindContext *ctx) override {
    auto kind = fegen::Type::TypeKind::ATTRIBUTE;
    if (ctx->CPP()) {
      kind = fegen::Type::TypeKind::CPP;
    } else if (ctx->OPERAND()) {
      kind = fegen::Type::TypeKind::OPERAND;
    }
    // otherwise: ATTRIBUTE
    return kind;
  }

  // return fegen::FegenType
  std::any visitTypeInstance(FegenParser::TypeInstanceContext *ctx) override {
    if (ctx->typeTemplate()) { // typeTemplate (Less typeTemplateParam (Comma
                               // typeTemplateParam)* Greater)?
      auto typeTeplt =
          std::any_cast<fegen::Type>(this->visit(ctx->typeTemplate()));
      // get parameters
      std::vector<fegen::Value *> paramList;
      for (auto paramCtx : ctx->typeTemplateParam()) {
        auto tepltParams =
            std::any_cast<fegen::Value *>(this->visit(paramCtx));
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
          Type::getInstanceType(typeTeplt.getTypeDefination(), paramList);
      return typeInst;
    } else if (ctx->identifier()) { // identifier
      auto varName = ctx->identifier()->getText();
      auto var = this->sstack.attemptFindVar(varName);
      if (var) {
        if (var->getContentKind() ==
            fegen::RightValue::LiteralKind::TYPE) {
          return var->getContent<fegen::Type>();
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
      auto ty = std::any_cast<fegen::Type>(
          this->visit(ctx->builtinTypeInstances()));
      return fegen::Value::get(ty, "param",
                                    fegen::RightValue::getPlaceHolder());
    } else {
      auto expr =
          std::any_cast<std::shared_ptr<fegen::RightValue::Expression>>(
              this->visit(ctx->expression()));
      return fegen::Value::get(expr->exprType, "expression_tmp",
                                    fegen::RightValue::getByExpr(expr));
    }
  }

  // return fegen::FegenType
  std::any visitBuiltinTypeInstances(
      FegenParser::BuiltinTypeInstancesContext *ctx) override {
    if (ctx->BOOL()) {
      return Type::getBoolType();
    } else if (ctx->INT()) {
      return Type::getInt32Type();
    } else if (ctx->FLOAT()) {
      return Type::getFloatType();
    } else if (ctx->DOUBLE()) {
      return Type::getDoubleType();
    } else if (ctx->CHAR()) {
      return Type::getCharType();
    } else if (ctx->STRING()) {
      return Type::getStringType();
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
        return fegen::Type::getTemplateType(tyDef);
      }
    } else if (ctx->builtinTypeTemplate()) { // builtinTypeTemplate
      return this->visit(ctx->builtinTypeTemplate());
    } else { // TYPE
      return fegen::Type::getMetaType();
    }
  }

  // return FegenType
  std::any visitBuiltinTypeTemplate(
      FegenParser::BuiltinTypeTemplateContext *ctx) override {
    if (ctx->INTEGER()) {
      return fegen::Type::getIntegerTemplate();
    } else if (ctx->FLOATPOINT()) {
      return fegen::Type::getFloatPointTemplate();
    } else if (ctx->TENSOR()) {
      // return fegen::FegenType::getTensorTemplate();
      return fegen::Type::getPlaceHolder();
    } else if (ctx->VECTOR()) {
      // return fegen::FegenType::getVectorTemplate();
      return fegen::Type::getPlaceHolder();
    } else {
      return nullptr;
    }
  }

  // return FegenType
  std::any
  visitCollectTypeSpec(FegenParser::CollectTypeSpecContext *ctx) override {
    auto kind = fegen::Type::TypeKind::CPP;
    if (ctx->valueKind()) {
      kind = std::any_cast<fegen::Type::TypeKind>(
          this->visit(ctx->valueKind()));
    }
    auto ty = std::any_cast<fegen::Type>(this->visit(ctx->collectType()));
    ty.setTypeKind(kind);
    return ty;
  }

  // return FegenType
  std::any visitCollectType(FegenParser::CollectTypeContext *ctx) override {
    auto expr =
        std::any_cast<std::shared_ptr<fegen::RightValue::Expression>>(
            this->visit(ctx->expression()));
    if (ctx->collectProtoType()->ANY()) {
      std::vector<fegen::Type> tys;
      // TODO: reprot error
      assert(expr->getKind() == fegen::RightValue::LiteralKind::VECTOR);
      auto exprs = std::any_cast<
          std::vector<std::shared_ptr<fegen::RightValue::Expression>>>(
          expr->getContent());
      for (auto expr : exprs) {
        auto ty = std::any_cast<fegen::Type>(expr->getContent());
        tys.push_back(ty);
      }
      return fegen::Type::getAnyType(tys);
    } else if (ctx->collectProtoType()->LIST()) {
      assert(expr->getKind() == fegen::RightValue::LiteralKind::TYPE);
      auto ty = std::any_cast<fegen::Type>(expr->getContent());
      return fegen::Type::getListType(ty);
    } else { // optional
      assert(expr->getKind() == fegen::RightValue::LiteralKind::TYPE);
      auto ty = std::any_cast<fegen::Type>(expr->getContent());
      return fegen::Type::getOptionalType(ty);
    }
  }

  // return std::shared_ptr<fegen::FegenRightValue::Expression>
  std::any visitExpression(FegenParser::ExpressionContext *ctx) override {
    auto expr =
        std::any_cast<std::shared_ptr<fegen::RightValue::Expression>>(
            this->visit(ctx->andExpr(0)));
    for (size_t i = 1; i <= ctx->andExpr().size() - 1; i++) {
      auto rhs =
          std::any_cast<std::shared_ptr<fegen::RightValue::Expression>>(
              this->visit(ctx->andExpr(i)));
      expr = RightValue::ExpressionNode::binaryOperation(
          expr, rhs, FegenOperator::OR);
    }
    return expr;
  }

  // return std::shared_ptr<fegen::FegenRightValue::Expression>
  std::any visitAndExpr(FegenParser::AndExprContext *ctx) override {
    auto expr =
        std::any_cast<std::shared_ptr<fegen::RightValue::Expression>>(
            this->visit(ctx->equExpr(0)));
    for (size_t i = 1; i <= ctx->equExpr().size() - 1; i++) {
      auto rhs =
          std::any_cast<std::shared_ptr<fegen::RightValue::Expression>>(
              this->visit(ctx->equExpr(i)));
      expr = RightValue::ExpressionNode::binaryOperation(
          expr, rhs, FegenOperator::AND);
    }
    return expr;
  }

  // return std::shared_ptr<fegen::FegenRightValue::Expression>
  std::any visitEquExpr(FegenParser::EquExprContext *ctx) override {
    auto expr =
        std::any_cast<std::shared_ptr<fegen::RightValue::Expression>>(
            this->visit(ctx->compareExpr(0)));
    for (size_t i = 1; i <= ctx->compareExpr().size() - 1; i++) {
      FegenOperator op;
      if (ctx->children[2 * i - 1]->getText() == "==") {
        op = FegenOperator::EQUAL;
      } else {
        op = FegenOperator::NOT_EQUAL;
      }
      auto rhs =
          std::any_cast<std::shared_ptr<fegen::RightValue::Expression>>(
              this->visit(ctx->compareExpr(i)));
      expr = RightValue::ExpressionNode::binaryOperation(expr, rhs, op);
    }
    return expr;
  }

  // return std::shared_ptr<fegen::FegenRightValue::Expression>
  std::any visitCompareExpr(FegenParser::CompareExprContext *ctx) override {
    auto expr =
        std::any_cast<std::shared_ptr<fegen::RightValue::Expression>>(
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
      auto rhs =
          std::any_cast<std::shared_ptr<fegen::RightValue::Expression>>(
              this->visit(ctx->addExpr(i)));
      expr = RightValue::ExpressionNode::binaryOperation(expr, rhs, op);
    }
    return expr;
  }

  // return std::shared_ptr<fegen::FegenRightValue::Expression>
  std::any visitAddExpr(FegenParser::AddExprContext *ctx) override {
    auto expr =
        std::any_cast<std::shared_ptr<fegen::RightValue::Expression>>(
            this->visit(ctx->term(0)));
    for (size_t i = 1; i <= ctx->term().size() - 1; i++) {
      FegenOperator op;
      auto opStr = ctx->children[2 * i - 1]->getText();
      if (opStr == "+") {
        op = FegenOperator::ADD;
      } else {
        op = FegenOperator::SUB;
      }
      auto rhs =
          std::any_cast<std::shared_ptr<fegen::RightValue::Expression>>(
              this->visit(ctx->term(i)));
      expr = RightValue::ExpressionNode::binaryOperation(expr, rhs, op);
    }
    return expr;
  }

  // return std::shared_ptr<fegen::FegenRightValue::Expression>
  std::any visitTerm(FegenParser::TermContext *ctx) override {
    auto expr =
        std::any_cast<std::shared_ptr<fegen::RightValue::Expression>>(
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
      auto rhs =
          std::any_cast<std::shared_ptr<fegen::RightValue::Expression>>(
              this->visit(ctx->powerExpr(i)));
      expr = RightValue::ExpressionNode::binaryOperation(expr, rhs, op);
    }
    return expr;
  }

  // return std::shared_ptr<fegen::FegenRightValue::Expression>
  std::any visitPowerExpr(FegenParser::PowerExprContext *ctx) override {
    auto expr =
        std::any_cast<std::shared_ptr<fegen::RightValue::Expression>>(
            this->visit(ctx->unaryExpr(0)));
    for (size_t i = 1; i <= ctx->unaryExpr().size() - 1; i++) {
      auto rhs =
          std::any_cast<std::shared_ptr<fegen::RightValue::Expression>>(
              this->visit(ctx->unaryExpr(i)));
      expr = RightValue::ExpressionNode::binaryOperation(
          expr, rhs, FegenOperator::POWER);
    }
    return expr;
  }

  // return std::shared_ptr<fegen::FegenRightValue::Expression>
  std::any visitUnaryExpr(FegenParser::UnaryExprContext *ctx) override {
    if (ctx->children.size() == 1 || ctx->Plus()) {
      return this->visit(ctx->primaryExpr());
    }
    auto expr =
        std::any_cast<std::shared_ptr<fegen::RightValue::Expression>>(
            this->visit(ctx->primaryExpr()));
    FegenOperator op;
    if (ctx->Minus()) {
      op = FegenOperator::NEG;
    } else {
      op = FegenOperator::NOT;
    }
    expr = RightValue::ExpressionNode::unaryOperation(expr, op);
    return expr;
  }

  // return std::shared_ptr<fegen::FegenRightValue::Expression>
  std::any visitParenSurroundedExpr(
      FegenParser::ParenSurroundedExprContext *ctx) override {
    return this->visit(ctx->expression());
  }

  // return std::shared_ptr<fegen::FegenRightValue::Expression>
  std::any visitPrimaryExpr(FegenParser::PrimaryExprContext *ctx) override {
    if (ctx->identifier()) {
      auto name = ctx->identifier()->getText();
      auto var = this->sstack.attemptFindVar(name);
      if (var) {
        return (std::shared_ptr<fegen::RightValue::Expression>)
            fegen::RightValue::ExpressionTerminal::getLeftValue(var);
      } else {
        // TODO
        auto tyDef = this->manager.getTypeDefination(name);
        if (tyDef) {
          auto tyVar = fegen::Type::getTemplateType(tyDef);
          return (std::shared_ptr<fegen::RightValue::Expression>)
              fegen::RightValue::Expression::getType(tyVar);
        } else {
          // TODO: error report
          std::cerr << "can not find variable: " << ctx->identifier()->getText()
                    << "." << std::endl;
          exit(0);
          return nullptr;
        }
      }
    } else if (ctx->typeSpec()) {
      auto ty = std::any_cast<fegen::Type>(this->visit(ctx->typeSpec()));
      return (std::shared_ptr<fegen::RightValue::Expression>)
          RightValue::ExpressionTerminal::getType(ty);
    } else { // constant, functionCall, parenSurroundedExpr,contextMethodInvoke,
             // and variableAccess
      return this->visit(ctx->children[0]);
    }
  }

  // return std::shared_ptr<fegen::FegenRightValue::Expression>
  std::any visitIntLiteral(FegenParser::IntLiteralContext *ctx) override {
    long long int number = std::stoi(ctx->getText());
    size_t size = 32; // TODO: Get size of number.
    return (std::shared_ptr<fegen::RightValue::Expression>)
        fegen::RightValue::Expression::getInteger(number, size);
  }

  // return std::shared_ptr<fegen::FegenRightValue::Expression>
  std::any visitRealLiteral(FegenParser::RealLiteralContext *ctx) override {
    long double number = std::stod(ctx->getText());
    size_t size = 32; // TODO: Get size of number.
    return (std::shared_ptr<fegen::RightValue::Expression>)
        fegen::RightValue::Expression::getFloatPoint(number, size);
  }

  // return std::shared_ptr<fegen::FegenRightValue::Expression>
  std::any visitCharLiteral(FegenParser::CharLiteralContext *ctx) override {
    std::string s = ctx->getText();
    // remove quotation marks
    std::string strWithoutQuotation = s.substr(1, s.size() - 2);
    return (std::shared_ptr<fegen::RightValue::Expression>)
        fegen::RightValue::Expression::getString(strWithoutQuotation);
  }

  // return std::shared_ptr<fegen::FegenRightValue::Expression>
  std::any visitBoolLiteral(FegenParser::BoolLiteralContext *ctx) override {
    int content = 0;
    if (ctx->getText() == "true") {
      content = 1;
    }
    return (std::shared_ptr<fegen::RightValue::Expression>)
        fegen::RightValue::Expression::getInteger(content, 1);
  }

  // return std::shared_ptr<fegen::FegenRightValue::Expression>
  std::any visitListLiteral(FegenParser::ListLiteralContext *ctx) override {
    std::vector<std::shared_ptr<fegen::RightValue::Expression>> elements;
    for (auto exprCtx : ctx->expression()) {
      auto expr =
          std::any_cast<std::shared_ptr<fegen::RightValue::Expression>>(
              this->visit(exprCtx));
      elements.push_back(expr);
    }
    return (std::shared_ptr<fegen::RightValue::Expression>)
        fegen::RightValue::Expression::getList(elements);
  }

  std::any visitActionSpec(FegenParser::ActionSpecContext *ctx) override {
    return nullptr;
  }

  std::any visitFunctionDecl(FegenParser::FunctionDeclContext *ctx) override {
    sstack.pushScope();
    auto returnType =
        std::any_cast<fegen::Type>(this->visit(ctx->typeSpec()));
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
    auto functionParams = std::any_cast<std::vector<fegen::Value *>>(
        this->visit(ctx->funcParams()));
    this->visit(ctx->statementBlock());

        fegen::Function* function = fegen::Function::get(functionName, functionParams, &returnType);
        manager.functionMap.insert(std::pair{functionName, function});
        sstack.popScope();
        return nullptr;
    }

  std::any visitFuncName(FegenParser::FuncNameContext *ctx) override {
    auto functionName = ctx->identifier()->getText();
    return functionName;
  }

  std::any visitFuncParams(FegenParser::FuncParamsContext *ctx) override {
    std::vector<fegen::Value *> paramsList = {};

    for (size_t i = 0; i < ctx->typeSpec().size(); i++) {
      auto paramType =
          std::any_cast<fegen::Type>(this->visit(ctx->typeSpec(i)));
      auto paramName = ctx->identifier(i)->getText();
      auto param = fegen::Value::get(
          paramType, paramName, fegen::RightValue::getPlaceHolder());
      paramsList.push_back(param);
      sstack.attemptAddVar(param);
    }
    return paramsList;
  }

  std::any visitVarDeclStmt(FegenParser::VarDeclStmtContext *ctx) override {
    auto varType =
        std::any_cast<fegen::Type>(this->visit(ctx->typeSpec()));
    auto varName = ctx->identifier()->getText();
    fegen::Value *var;
    if (ctx->expression()) {
      auto varcontent =
          std::any_cast<std::shared_ptr<fegen::RightValue::Expression>>(
              this->visit(ctx->expression()));
      // TODO: check error
      // if(!fegen::FegenType::isSameType(&varType, &varcontent->exprType)){
      //     std::cerr << "The variabel \" " << varName
      //     << "\" need \"" << varType.getTypeName() << " \" type rightvalue."
      //     << std::endl; exit(0); return nullptr;
      // }
      var = fegen::Value::get(
          varType, varName, fegen::RightValue::getByExpr(varcontent));
    } else {
      var = fegen::Value::get(varType, varName,
                                   fegen::RightValue::getPlaceHolder());
    }
    sstack.attemptAddVar(var);
    manager.stmtContentMap.insert(std::pair{ctx, var});
    return var;
  }

  std::any visitAssignStmt(FegenParser::AssignStmtContext *ctx) override {
    auto varName = ctx->identifier()->getText();
    auto varcontent =
        std::any_cast<std::shared_ptr<fegen::RightValue::Expression>>(
            this->visit(ctx->expression()));
    auto var = sstack.attemptFindVar(varName);
    if (!fegen::Type::isSameType(&var->getType(), &varcontent->exprType)) {
      std::cerr << "The variabel \" " << varName << "\" need \""
                << var->getType().getTypeName() << " \" type rightvalue."
                << std::endl;
      exit(0);
      return nullptr;
    }
    fegen::Value *stmt = fegen::Value::get(
        var->getType(), varName, fegen::RightValue::getByExpr(varcontent));
    manager.stmtContentMap.insert(std::pair{ctx, stmt});

    return stmt;
  }

  std::any visitFunctionCall(FegenParser::FunctionCallContext *ctx) override {
    std::vector<fegen::RightValue::Expression *> parasList = {};
    auto functionName =
        std::any_cast<std::string>(this->visit(ctx->funcName()));
    auto hasFunc = manager.functionMap.at(functionName);
    auto paramsNum = ctx->expression().size();
    auto paraList = hasFunc->getInputTypeList();
    if (paramsNum > 0) {
      for (size_t i = 0; i < paramsNum; i++) {
        auto oprand = std::any_cast<fegen::RightValue::Expression *>(
            this->visit(ctx->expression(i)));
        parasList.push_back(oprand);
      }
      size_t len1 = paraList.size();
      size_t len2 = parasList.size();
      if (len1 != len2) {
        std::cerr << "The function \" " << functionName
                  << "\" parameter count mismatch." << std::endl;
        exit(0);
        return nullptr;
      }
      for (size_t i = 0; i < len1; i++) {
        if (!fegen::Type::isSameType(&paraList[i]->getType(),
                                          &parasList[i]->exprType)) {
          std::cerr << "The function \" " << functionName << "\" parameter" << i
                    << " type mismatch." << std::endl;
          exit(0);
          return nullptr;
        }
      }
    }
    auto returnType = hasFunc->getReturnType();
    fegen::Function *funcCall =
        fegen::Function::get(functionName, paraList, returnType);
    manager.stmtContentMap.insert(std::pair{ctx, funcCall});
    return returnType;
  }

  std::any visitOpInvokeStmt(FegenParser::OpInvokeStmtContext *ctx) override {
    return nullptr;
  }

  std::any visitIfStmt(FegenParser::IfStmtContext *ctx) override {
    for (size_t i = 0; i < ctx->ifBlock().size(); i++) {
      this->visit(ctx->ifBlock(i));
    }

    if (ctx->elseBlock()) {
      this->visit(ctx->elseBlock());
    }
    return nullptr;
  }

  std::any visitIfBlock(FegenParser::IfBlockContext *ctx) override {
    sstack.pushScope();
    this->visit(ctx->expression());
    this->visit(ctx->statementBlock());
    sstack.popScope();

    return nullptr;
  }

  std::any visitElseBlock(FegenParser::ElseBlockContext *ctx) override {
    sstack.pushScope();
    this->visit(ctx->statementBlock());
    sstack.popScope();
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
        std::any_cast<fegen::Operation *>(this->visit(ctx->opBlock()));
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
    std::vector<fegen::Value *> args;
    std::vector<fegen::Value *> res;
    if (ctx->argumentSpec()) {
      args = std::any_cast<std::vector<fegen::Value *>>(
          this->visit(ctx->argumentSpec()));
    }
    if (ctx->resultSpec()) {
      res = std::any_cast<std::vector<fegen::Value *>>(
          this->visit(ctx->resultSpec()));
    }
    return fegen::Operation::get("", args, res, ctx->bodySpec());
  }
};
} // namespace fegen
#endif