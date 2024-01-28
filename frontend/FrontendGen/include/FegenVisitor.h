#ifndef FEGEN_FEGENVISITOR_H
#define FEGEN_FEGENVISITOR_H

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
public:
  FegenVisitor() {}

  std::any
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

#endif