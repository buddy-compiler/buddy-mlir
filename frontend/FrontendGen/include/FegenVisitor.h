#ifndef FEGEN_FEGENVISITOR_H
#define FEGEN_FEGENVISITOR_H

<<<<<<< HEAD
#include "FegenParserBaseVisitor.h"
#include "FegenRule.h"
#include "FegenValue.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <map>
#include <string>

using namespace antlr4;
using namespace fegen;

class FegenVisitor : public FegenParserBaseVisitor {
=======
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

>>>>>>> 8cb8a2b9eb8a15bf99280049e7f321d3114fb4bc
public:
  FegenVisitor() {}

  std::any
<<<<<<< HEAD
  visitLexerGrammarNode(FegenParser::LexerGrammarNodeContext *ctx) override {
    auto name = ctx->LexerRuleName()->getText();
    auto rule = RuleMap::createRule(RuleType::LEX_RULE, name);
    rule->setGrammarContext(ctx->lexerGrammarSpec()->lexerAntlrRule());
    RuleMap::getRuleMap().insert(rule);
    return nullptr;
  }

  std::any
  visitParserGrammarNode(FegenParser::ParserGrammarNodeContext *ctx) override {
    auto name = ctx->ParserRuleName()->getText();
    auto rule = RuleMap::createRule(RuleType::PARSE_RULE, name);
    rule->setGrammarContext(ctx->parserGrammarSpec()->parserAntlrRule());
    RuleMap::getRuleMap().insert(rule);
    if (ctx->inputsSpec()) {
      this->visit(ctx->inputsSpec());
    }
    if (ctx->returnsSpec()) {
      this->visit(ctx->returnsSpec());
    }
    if (ctx->irSpec()) {
      this->visit(ctx->irSpec());
    }
    return nullptr;
  }

  
};

=======
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
        auto flag = rule->addInput(in);
        // todo: throw error if flag is false
      }
      for (auto out : *returns) {
        auto flag = rule->addReturn(out);
        // todo: throw error if flag is false
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
    }
    if (ctx->returnsSpec()) {
      returns = std::any_cast<std::vector<FegenValue *> *>(
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
    auto ruleList = std::any_cast<std::vector<FegenRule *>>(
        this->visit(ctx->lexerRuleBlock()));
    auto ruleNode =
        FegenNode::get(ruleList, ctx, FegenNode::NodeType::LEXER_RULE);
    this->manager.nodeMap.insert({ctx->LexerRuleName()->getText(), ruleNode});
    return nullptr;
  }
};
} // namespace fegen
>>>>>>> 8cb8a2b9eb8a15bf99280049e7f321d3114fb4bc
#endif