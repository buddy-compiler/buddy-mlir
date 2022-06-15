#include <iostream>
#include <fstream>
#include "toyLexer.h"
#include "toyParser.h"
#include "Dialect.h"
#include "mlirvisitor.h"
#include "Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char* argv[]) {
  // you should input the name of the file which contains code.
  if (argc == 2) {
    std::fstream in(argv[1]);
    antlr4::ANTLRInputStream input(in);
    toyLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);
    toyParser parser(&tokens);
    antlr4::tree::ParseTree* tree = parser.module();
    // load our dialect in context.
    context.getOrLoadDialect<mlir::toy::ToyDialect>();
    mlirvisitor visitor(argv[1]); 
    visitor.visit(tree);
    mlir::PassManager pm(&context);
    // apply any generic pass manager command line options and run the pipeline.
    mlir::applyPassManagerCLOptions(pm);
    pm.addNestedPass<mlir::toy::FuncOp>(mlir::createCanonicalizerPass());
    // lower toy dialect to standard dialect.
    pm.addPass(mlir::toy::createLowerToAffinePass());
    // lower standard dialect to llvm dialect.
    pm.addPass(mlir::toy::createLowerToLLVMPass());
    pm.run(visitor.theModule);
    // dump llvm dialect.
    visitor.theModule.dump();
  }
  return 0;

}


