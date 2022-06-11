//===- Main.cpp -----------------------------------------------------------===//
//
// This file is the DSL frontend to generate MLIR from input file.
//
//===----------------------------------------------------------------------===//

#include <iostream>

#include "TLexer.h"
#include "TParser.h"
#include "antlr4-runtime.h"

#include "toy/Dialect.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

using namespace antlrcpptest;
using namespace antlr4;

int main(int, const char **) {
  // TODO: Read a input file.
  ANTLRInputStream input(u8"a + (x * (y ? 0 : 1) + z);");
  TLexer lexer(&input);
  CommonTokenStream tokens(&lexer);

  tokens.fill();
  for (auto token : tokens.getTokens()) {
    std::cout << token->toString() << std::endl;
  }

  TParser parser(&tokens);
  tree::ParseTree *tree = parser.main();

  std::cout << tree->toStringTree(&parser) << std::endl << std::endl;

  // TODO: Traverse AST to generate MLIR file.
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::toy::ToyDialect>();
  std::cout << "Is Toy Constant Operation Registered: "
            << context.isOperationRegistered("toy.constant") << std::endl;

  return 0;
}
