//====- main.cpp - The driver of buddy-toy-dsl --------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file is about oprimizing code and lowering code from toy dialect to
// llvm dialect buddy-mlir project.
//
//===----------------------------------------------------------------------===//

#include "MLIRVisitor.h"
#include "ToyLexer.h"
#include "ToyParser.h"
#include "antlr4-common.h"
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
#include "toy/Dialect.h"
#include "toy/Passes.h"
#include <fstream>
#include <iostream>

int main(int argc, char *argv[]) {
  // you should input the name of the file which contains code.
  if (argc == 2) {
    std::fstream in(argv[1]);
    antlr4::ANTLRInputStream input(in);
    ToyLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);
    ToyParser parser(&tokens);
    antlr4::tree::ParseTree *tree = parser.module();
    // load our dialect in context.
    // context.getOrLoadDialect<mlir::toy::ToyDialect>();
    MLIRVisitor visitor(argv[1]);
    visitor.visit(tree);
    mlir::PassManager pm(&visitor.context);
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
