#include "FegenRule.h"

fegen::FegenRule::FegenRule(RuleType r, llvm::StringRef name)
    : ruleType(r), name(name) {}

fegen::RuleType fegen::FegenRule::getRuleType() { return this->ruleType; }

llvm::StringRef fegen::FegenRule::getName() { return this->name; }

antlr4::ParserRuleContext *fegen::FegenRule::getGrammarContext() {
  return this->grammarContext;
}

void fegen::FegenRule::setGrammarContext(antlr4::ParserRuleContext *ctx) {
  this->grammarContext = ctx;
}

fegen::FegenIR fegen::FegenRule::getIRContent() { return this->irContent; }
void fegen::FegenRule::setIRContent(fegen::FegenIR irContent) {
  this->irContent = irContent;
}

bool fegen::FegenRule::addInput(fegen::FegenValue *value) {
  auto p = this->inputList.insert({value->getName(), value});
  return p.second;
}

bool fegen::FegenRule::addReturn(fegen::FegenValue *value) {
  auto p = this->returnList.insert({value->getName(), value});
  return p.second;
}

fegen::RuleMap::~RuleMap() {
  for (auto p : this->name2RuleMap) {
    auto rule = p.second;
    delete rule;
  }
}

fegen::FegenRule *fegen::RuleMap::find(llvm::StringRef name) {
  auto rule = this->name2RuleMap.find(name);
  if (rule == this->name2RuleMap.end()) {
    // TODO: output error
    std::cerr << "cannot find rule: " << name.str() << '\n';
  }
  return rule->second;
}

void fegen::RuleMap::insert(fegen::FegenRule *rule) {
  auto name = rule->getName();
  auto flag = this->name2RuleMap.insert({name, rule});
  if (!flag.second) {
    // TODO: output error
    std::cerr << "rule " << name.str() << " is already in the map." << '\n';
  }
}

void fegen::RuleMap::setGrammarName(llvm::StringRef name) {
  this->grammarName = name;
}

llvm::StringRef fegen::RuleMap::getGrammarName() { return this->grammarName; }

void fegen::RuleMap::emitG4File(llvm::raw_fd_ostream &os) {
  fegen::GrammarContentGenerator generator;
  for (auto pair : this->name2RuleMap) {
    auto rule = pair.second;
    os << rule->getName() << ':' << '\n' << '\t';
    auto ctx = rule->getGrammarContext();
    // if ctx is not setted
    if (ctx) {
      auto raw_content = generator.visit(ctx);
      auto content = std::any_cast<std::string>(raw_content);
      os << content;
    }
    os << '\n' << ';' << '\n' << '\n';
  }
}

// get type name string in cpp
std::string getTypeNameString(fegen::FegenValue *v) {
  using namespace fegen;
  std::string res;
  switch (v->getValueKind()) {
  case ValueKind::ATTRIBUTE: {
    res.append("mlir::");
    res.append(RuleMap::getRuleMap().getGrammarName());
    res.append("::");
    res.append(v->getType()->getName());
    break;
  }
  case ValueKind::OPERAND: {
    res.append("mlir::Value");
    break;
  }
  case ValueKind::CPP: {
    res.append(v->getType()->getName());
    break;
  }
  default: {
  }
  }
  return res;
}

void fegen::RuleMap::emitVisitorFile(llvm::raw_fd_ostream &headfile,
                                     llvm::raw_fd_ostream &cppfile) {
// TODO: change head files of visitor
// head file
#define HEADFILE_TAB1 headfile << '\t'
#define HEADFILE_TAB2 headfile << '\t' << '\t'
#define HEADFILE_TAB3 headfile << '\t' << '\t' << '\t'

  // include head file
  headfile << "// toyVisitor.h" << '\n';
  headfile << "#include \"toyBaseVisitor.h\"" << '\n';
  // begin of class defination
  headfile << "class toyVisitor : public toyBaseVisitor {" << '\n';
  headfile << "public:" << '\n';
  // loop to declare visit functions
  for (auto pair : this->name2RuleMap) {
    auto rule = pair.second;
    // lexer rule do not have visit function
    if (rule->getRuleType() == RuleType::LEX_RULE) {
      continue;
    }
    // get inital Uppercased rule name
    auto ruleName = pair.first.str();
    auto initial = ruleName.at(0);
    char upperInitial = char(initial - 32);
    ruleName[0] = upperInitial;
    // begin of function
    HEADFILE_TAB1 << "std::any "
                  << "visit" << ruleName << '(' << "toyParser::" << ruleName
                  << "Context *ctx"
                  << ") override {" << '\n';
    // visit function content
    for (auto input : rule->inputList) {
      auto inputValueName = input.first;
      auto inputValue = input.second;
      // init input value
      switch (inputValue->getBindingValue().index()) {
      // bind to literal
      case 1: {
        auto literalValueInfo = std::get<1>(inputValue->getBindingValue());
        switch (literalValueInfo.value.index()) {
        case 0: {
          switch (inputValue->getValueKind()) {
          case ValueKind::ATTRIBUTE: {
            HEADFILE_TAB2 << "auto " << inputValueName << " = "
                          << "IntegerAttr::get(this->builder.getI32Type(), "
                          << std::get<0>(literalValueInfo.value) << ");\n";
            break;
          }
          case ValueKind::OPERAND: {
            HEADFILE_TAB2 << "auto " << inputValueName << " = "
                          << "builder.create<arith::constant>("
                          << std::get<0>(literalValueInfo.value) << ");\n";
            break;
          }
          case ValueKind::CPP: {
            HEADFILE_TAB2 << "int " << inputValueName << " = "
                          << std::get<0>(literalValueInfo.value) << ";\n";
            break;
          }
          default: {
          }
          }
          break;
        }
        case 1: {
          switch (inputValue->getValueKind()) {
          case ValueKind::ATTRIBUTE: {
            HEADFILE_TAB2 << "auto " << inputValueName << " = "
                          << "stmt to build inputValue;\n";
            break;
          }
          case ValueKind::OPERAND: {
            HEADFILE_TAB2 << "auto " << inputValueName << " = "
                          << "stmt to build inputValue;\n";
            break;
          }
          case ValueKind::CPP: {
            HEADFILE_TAB2 << "float " << inputValueName << " = "
                          << std::get<1>(literalValueInfo.value) << ";\n";
            break;
          }
          default: {
          }
          }
          break;
        }
        case 2: {
          switch (inputValue->getValueKind()) {
          case ValueKind::ATTRIBUTE: {
            HEADFILE_TAB2 << "auto " << inputValueName << " = "
                          << "stmt to build inputValue;\n";
            break;
          }
          case ValueKind::OPERAND: {
            HEADFILE_TAB2 << "auto " << inputValueName << " = "
                          << "stmt to build inputValue;\n";
            break;
          }
          case ValueKind::CPP: {
            HEADFILE_TAB2 << "std::string " << inputValueName << " = "
                          << std::get<2>(literalValueInfo.value) << ";\n";
            break;
          }
          default: {
          }
          }
          break;
        }
        default: {
        }
        }
        break;
      }
      // bind to the input or return of a rule
      case 2: {
        auto inoutputValueInfo = std::get<2>(inputValue->getBindingValue());
        // from this rule
        if (inoutputValueInfo.sourceRule == rule) {
          HEADFILE_TAB2 << "auto " << inputValueName << " = "
                        << inoutputValueInfo.value->getName() << ";\n";
        } else { // from child rule, visit to get result
          auto typeName = getTypeNameString(inputValue);
          if (inputValue
                  ->ifList()) { // if input value is list, loop to get array
            HEADFILE_TAB2 << "llvm::ArrayRef<" << typeName << "> "
                          << inputValueName << ";\n";
            HEADFILE_TAB2 << "for(auto child : ctx->"
                          << inoutputValueInfo.sourceRule->getName() << "()) {\n";
            HEADFILE_TAB3 << "auto raw_res = this->visit(child);\n";
            HEADFILE_TAB3 << "auto res = std::any_cast<" << typeName
                          << ">(raw_res);\n";
            HEADFILE_TAB3 << inputValueName << ".insert(res);\n";
            HEADFILE_TAB2 << "}";
          } else {
            if(inputValue->getRuleIndex() == -1){ // only one rules
              HEADFILE_TAB2 << "auto raw_" << inputValueName << " = visit(ctx->"
                          << inoutputValueInfo.sourceRule->getName() << "());\n";
              HEADFILE_TAB2 << "auto " << inputValueName << " = std::any_cast<"
                            << typeName << ">("
                            << "raw_" << inputValueName << ");\n";
            }else{ // more than one rule
              HEADFILE_TAB2 << "auto raw_" << inputValueName << " = visit(ctx->"
                          << inoutputValueInfo.sourceRule->getName() << "( "<< inputValue->getRuleIndex() << "));\n";
              HEADFILE_TAB2 << "auto " << inputValueName << " = std::any_cast<"
                            << typeName << ">("
                            << "raw_" << inputValueName << ");\n";
            }
            
          }
        }
        break;
      }
      // bind to the attribute of a rule
      case 3: {
        auto attrValueInfo = std::get<3>(inputValue->getBindingValue());
        std::string nodeName("ctx");
        if(rule != attrValueInfo.sourceRule){ // from child rule
          ruleName.append("->");
          ruleName.append(attrValueInfo.sourceRule->getName());
          ruleName.append("()");
        }
        switch(attrValueInfo.attrType){
          case FegenValue::RuleAttributeValue::AttributeKind::TEXT: {
            HEADFILE_TAB2 << "auto " << inputValueName << " = " << ruleName << "->getText();\n";
            break;
          }
          default :{}
        }
        break;
      }
      // bind info of input is not setted
      case 0:
      default: {
        continue;
      }
      }
    }

    // end of function
    HEADFILE_TAB1 << '}' << '\n';
  }
  // end of class
  headfile << "};" << '\n';

#undef HEADFILE_TAB1
#undef HEADFILE_TAB2
#undef HEADFILE_TAB3

  // cpp file
  cppfile << "// toyVisitor.cpp" << '\n';
  cppfile << "#include \"toyVisitor.h\"" << '\n';
}

fegen::FegenRule *fegen::RuleMap::createRule(fegen::RuleType r,
                                             llvm::StringRef name) {
  return new fegen::FegenRule(r, name);
}

std::any fegen::GrammarContentGenerator::visitLexerAntlrRule(
    fegen::FegenParser::LexerAntlrRuleContext *ctx) {
  std::string content;
  for (auto child : ctx->lexerAlternatives()) {
    auto s = std::any_cast<std::string>(this->visit(child));
    content.append(s);
    content.append(" ");
  }
  return content;
}

std::any fegen::GrammarContentGenerator::visitLexerAlternatives(
    fegen::FegenParser::LexerAlternativesContext *ctx) {
  std::string content;
  auto first =
      std::any_cast<std::string>(this->visit(ctx->lexerAlternative(0)));
  content.append(first);
  for (int i = 1; i < int(ctx->lexerAlternative().size()) - 1; i++) {
    auto child = ctx->lexerAlternative(i);
    auto s = std::any_cast<std::string>(this->visit(child));
    content.append("| ");
    content.append(s);
    content.append(" ");
  }
  return content;
}

std::any fegen::GrammarContentGenerator::visitLexerAlternative(
    fegen::FegenParser::LexerAlternativeContext *ctx) {
  std::string content;
  auto suffixedRuleContent =
      std::any_cast<std::string>(this->visit(ctx->lexerSuffixedRule()));
  content.append(suffixedRuleContent);
  if (ctx->ruleSuffix()) {
    content.append(ctx->ruleSuffix()->getText());
  }
  return content;
}

std::any fegen::GrammarContentGenerator::visitLexerSuffixedRule(
    fegen::FegenParser::LexerSuffixedRuleContext *ctx) {
  return ctx->getText();
}

std::any fegen::GrammarContentGenerator::visitParserAntlrRule(
    fegen::FegenParser::ParserAntlrRuleContext *ctx) {
  std::string content;
  for (auto child : ctx->alternatives()) {
    auto s = std::any_cast<std::string>(this->visit(child));
    content.append(s);
    content.append(" ");
  }
  return content;
}

std::any fegen::GrammarContentGenerator::visitAlternatives(
    fegen::FegenParser::AlternativesContext *ctx) {
  std::string content;
  auto first = std::any_cast<std::string>(this->visit(ctx->alternative(0)));
  content.append(first);
  for (int i = 1; i < int(ctx->alternative().size()) - 1; i++) {
    auto child = ctx->alternative(i);
    auto s = std::any_cast<std::string>(this->visit(child));
    content.append("| ");
    content.append(s);
    content.append(" ");
  }
  return content;
}

std::any fegen::GrammarContentGenerator::visitAlternative(
    fegen::FegenParser::AlternativeContext *ctx) {
  std::string content;
  auto suffixedRuleContent =
      std::any_cast<std::string>(this->visit(ctx->suffixedRule()));
  content.append(suffixedRuleContent);
  if (ctx->ruleSuffix()) {
    content.append(ctx->ruleSuffix()->getText());
  }
  return content;
}

std::any fegen::GrammarContentGenerator::visitSuffixedRule(
    fegen::FegenParser::SuffixedRuleContext *ctx) {
  if (ctx->parenSurroundedElem()) {
    return this->visit(ctx->parenSurroundedElem());
  } else {
    return ctx->getText();
  }
}

std::any fegen::GrammarContentGenerator::visitParenSurroundedElem(
    fegen::FegenParser::ParenSurroundedElemContext *ctx) {
  std::string content;
  auto s = std::any_cast<std::string>(this->visit(ctx->parserAntlrRule()));
  content.append("(");
  content.append(s);
  content.append(")");
  return content;
}