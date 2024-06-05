#ifndef FEGEN_FEGENVISITOR_H
#define FEGEN_FEGENVISITOR_H

#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include "FegenManager.h"
#include "FegenParserBaseVisitor.h"

using namespace antlr4;

namespace fegen {

class FegenVisitor : public FegenParserBaseVisitor {
private:
  FegenManager manager;

public:
  void emitG4() {
    this->manager.emitG4();
  }

  FegenVisitor() {}

  std::any visitFegenDecl(FegenParser::FegenDeclContext* ctx) override {
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
          std::tuple<std::vector<FegenValue *> *, std::vector<FegenValue *> *>>(
          this->visit(ctx->actionBlock()));
      auto inputs = std::get<0>(blockValues);
      auto returns = std::get<1>(blockValues);
      auto rule = std::any_cast<FegenRule *>(rawRule);
      for (auto in : *inputs) {
        auto flag = rule->addInput(*in);
        if(!flag) { // todo: error report
          std::cerr << "input of " << rule->getContent().str() << " \"" << in->getName().str() << "\" existed." << std::endl;
        }
      }
      for (auto out : *returns) {
        auto flag = rule->addReturn(*out);
        if(!flag) { // todo: error report
          std::cerr << "return of " << rule->getContent().str() << " \"" << out->getName().str() << "\" existed." << std::endl;
        }
      }
    }
    return rawRule;
  }

  std::any visitActionBlock(FegenParser::ActionBlockContext *ctx) override {
    std::vector<FegenValue *> *inputs = nullptr;
    std::vector<FegenValue *> *returns = nullptr;
    if (ctx->inputsSpec()) {
      inputs = std::any_cast<std::vector<FegenValue *> *>(
          this->visit(ctx->inputsSpec()));
    }else{
      inputs = new std::vector<FegenValue *>();
    }

    if (ctx->returnsSpec()) {
      returns = std::any_cast<std::vector<FegenValue *> *>(
          this->visit(ctx->returnsSpec()));
    }else{
      returns = new std::vector<FegenValue *>();
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

  std::any visitLexerAltList(FegenParser::LexerAltListContext* ctx) override {
    std::vector<fegen::FegenRule*> ruleList;
    for(auto alt : ctx->lexerAlt()){
      auto rule = fegen::FegenRule::get(alt->getText(), nullptr, alt);
      ruleList.push_back(rule);
    }
    return ruleList;
  }

  std::any visitVarDecls(FegenParser::VarDeclsContext* ctx) override {
    return new std::vector<FegenValue *>();
  }

  std::any visitActionSpec(FegenParser::ActionSpecContext* ctx) override {
    return nullptr;
  }
};
} // namespace fegen
#endif