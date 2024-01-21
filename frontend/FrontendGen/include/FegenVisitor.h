#ifndef FEGEN_FEGENVISITOR_H
#define FEGEN_FEGENVISITOR_H

#include "FegenParserBaseVisitor.h"
#include "FegenRule.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <map>
#include <string>

using namespace antlr4;
using namespace fegen;

class FegenVisitor : public FegenParserBaseVisitor {
private:
  std::string getAntlrRuleContent(antlr4::ParserRuleContext* ctx) {
    std::string content;
    auto iter = ctx->getStart();
    while(iter != ctx->getStop()){
      content.append(iter->getText());
      content.append(" ");
      iter++;
    }
    return content;
  }

public:
  FegenVisitor(){}
  std::any
  visitLexerGrammarNode(FegenParser::LexerGrammarNodeContext *ctx) override {
    auto name = ctx->LexerRuleName()->getText();
    auto content = std::any_cast<std::string>(this->visit(ctx->lexerGrammarSpec()));
    auto rule = RuleMap::createRule(RuleType::LEX_RULE, name, content);
    RuleMap::getRuleMap().insert(rule);
    return nullptr;
  }

  std::any visitLexerGrammarSpec(FegenParser::LexerGrammarSpecContext* ctx) override {
    return this->visit(ctx->lexerAntlrRule());
  }

  std::any visitLexerAntlrRule(FegenParser::LexerAntlrRuleContext* ctx) override {
    std::string content;
    for(auto child : ctx->lexerAlternatives()){
      auto s = std::any_cast<std::string>(this->visit(child));
      content.append(s);
      content.append(" ");
    }
    return content;
  }

  std::any visitLexerAlternatives(FegenParser::LexerAlternativesContext* ctx) override {
    std::string content;
    auto first = std::any_cast<std::string>(this->visit(ctx->lexerAlternative(0)));
    content.append(first);
    for(int i = 1; i < int(ctx->lexerAlternative().size())-1; i++){
      auto child = ctx->lexerAlternative(i);
      auto s = std::any_cast<std::string>(this->visit(child));
      content.append("| ");
      content.append(s);
      content.append(" ");
    }
    return content;
  }

  std::any visitLexerAlternative(FegenParser::LexerAlternativeContext* ctx) override {
    std::string content;
    auto suffixedRuleContent = std::any_cast<std::string>(this->visit(ctx->lexerSuffixedRule()));
    content.append(suffixedRuleContent);
    if(ctx->ruleSuffix()){
      content.append(ctx->ruleSuffix()->getText());
    }
    return content;
  }

  std::any visitLexerSuffixedRule(FegenParser::LexerSuffixedRuleContext* ctx) override {
    return ctx->getText();
  }

  std::any visitParserGrammarNode(FegenParser::ParserGrammarNodeContext* ctx) override {
    auto name = ctx->ParserRuleName()->getText();
    auto content = std::any_cast<std::string>(this->visit(ctx->parserGrammarSpec()));
    auto rule = RuleMap::createRule(RuleType::PARSE_RULE, name, content);
    RuleMap::getRuleMap().insert(rule);
    if(ctx->inputsSpec()){
      this->visit(ctx->inputsSpec());
    }
    if(ctx->returnsSpec()){
      this->visit(ctx->returnsSpec());
    }
    if(ctx->irSpec()){
      this->visit(ctx->irSpec());
    }
    return nullptr;
  }

  std::any visitParserGrammarSpec(FegenParser::ParserGrammarSpecContext* ctx) override {
    return this->visit(ctx->antlrRule());
  }

  std::any visitAntlrRule(FegenParser::AntlrRuleContext* ctx) override {
    std::string content;
    for(auto child : ctx->alternatives()){
      auto s = std::any_cast<std::string>(this->visit(child));
      content.append(s);
      content.append(" ");
    }
    return content;
  }

  std::any visitAlternatives(FegenParser::AlternativesContext* ctx) override {
    std::string content;
    auto first = std::any_cast<std::string>(this->visit(ctx->alternative(0)));
    content.append(first);
    for(int i = 1; i < int(ctx->alternative().size())-1; i++){
      auto child = ctx->alternative(i);
      auto s = std::any_cast<std::string>(this->visit(child));
      content.append("| ");
      content.append(s);
      content.append(" ");
    }
    return content;
  }

  std::any visitAlternative(FegenParser::AlternativeContext* ctx) override {
    std::string content;
    auto suffixedRuleContent = std::any_cast<std::string>(this->visit(ctx->suffixedRule()));
    content.append(suffixedRuleContent);
    if(ctx->ruleSuffix()){
      content.append(ctx->ruleSuffix()->getText());
    }
    return content;
  }

  std::any visitSuffixedRule(FegenParser::SuffixedRuleContext* ctx) override {
    if(ctx->parenSurroundedElem()){
      return this->visit(ctx->parenSurroundedElem());
    }else{
      return ctx->getText();
    }
  }

  std::any visitParenSurroundedElem(FegenParser::ParenSurroundedElemContext* ctx) override {
    std::string content;
    auto s = std::any_cast<std::string>(this->visit(ctx->antlrRule()));
    content.append("(");
    content.append(s);
    content.append(")");
    return content;
  }
};

#endif