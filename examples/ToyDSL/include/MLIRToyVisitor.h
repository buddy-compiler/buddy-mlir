//===- MLIRToyVisitor.h ---------------------------------------------------===//
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
// This file is the visitor for the MLIR Toy language AST.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOY_VISITOR_H
#define MLIR_TOY_VISITOR_H

#include "ToyBaseVisitor.h"
#include "ToyLexer.h"
#include "ToyParser.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "toy/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

class MLIRToyVisitor : public ToyBaseVisitor {
public:
  /// AST Visitor Constructor
  MLIRToyVisitor(std::string filename, mlir::MLIRContext &context)
      : builder(&context), fileName(filename) {
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
  }
  /// Get the MLIR module.
  mlir::ModuleOp getModule() { return theModule; }

private:
  /// The Top Level MLIR Module Operation
  /// The module contains all the components generated from the source toy file.
  mlir::ModuleOp theModule;
  /// The MLIR Operations Builder
  /// The builder helps create MLIR operations when traversing the AST.
  mlir::OpBuilder builder;
  /// The Symbol Table
  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;
  llvm::ScopedHashTable<llvm::StringRef, int> funSymbolTable;
  llvm::StringMap<mlir::toy::FuncOp> functionMap;
  // Register the filename for the string attribute in MLIR location object.
  std::string fileName;

  /// Declare a variable in the current scope
  /// - Check if the variable is already registered.
  /// -  Register variable in the symbol table.
  mlir::LogicalResult declare(llvm::StringRef var, mlir::Value value) {
    if (symbolTable.count(var))
      return mlir::failure();
    symbolTable.insert(var, value);
    return mlir::success();
  }
  // Declear a function in the current module
  /// - Check the parameter number of the function.
  mlir::LogicalResult funcDeclare(llvm::StringRef functionName,
                                  int argsNumber) {
    if (funSymbolTable.count(functionName))
      return mlir::failure();
    funSymbolTable.insert(functionName, argsNumber);
    return mlir::success();
  }

  /// Location
  /// Get the MLIR location object with the current line and row of the toy
  /// source file.
  mlir::Location loc(int line, int row) {
    return mlir::FileLineColLoc::get(builder.getStringAttr(fileName), line,
                                     row);
  }

  /// Get the tensor type according to the shape.
  mlir::Type getType(llvm::ArrayRef<int64_t> shape) {
    if (shape.empty())
      return mlir::UnrankedTensorType::get(builder.getF64Type());
    return mlir::RankedTensorType::get(shape, builder.getF64Type());
  }

  // Get the tensor value from the tensor literal node.
  std::any getTensor(ToyParser::TensorLiteralContext *ctx) {
    std::vector<int64_t> dims;
    // get dimensions.
    dims.push_back(ctx->Comma().size() + 1);
    if (ctx->tensorLiteral(0)->tensorLiteral(0)) {
      ToyParser::TensorLiteralContext *list = ctx->tensorLiteral(0);
      while (list) {
        dims.push_back(list->Comma().size() + 1);
        if (list->tensorLiteral(0) && list->tensorLiteral(0)->Comma().size())
          list = list->tensorLiteral(0);
        else
          break;
      }
    }
    mlir::Type elementType = builder.getF64Type();
    mlir::Type type = getType(dims);
    auto dataType = mlir::RankedTensorType::get(dims, elementType);
    mlir::DenseElementsAttr dataAttribute =
        mlir::DenseElementsAttr::get(dataType, llvm::makeArrayRef(ctx->data));
    mlir::Location loaction =
        loc(ctx->start->getLine(), ctx->start->getCharPositionInLine());
    mlir::Value value =
        builder.create<mlir::toy::ConstantOp>(loaction, type, dataAttribute);
    return value;
  }

  virtual std::any visitModule(ToyParser::ModuleContext *ctx) override {
    llvm::ScopedHashTableScope<llvm::StringRef, int> protoTypeSymbolTable(
        funSymbolTable);
    for (auto &function : ctx->funDefine()) {
      ToyParser::PrototypeContext *protoType = function->prototype();
      std::string functionName = protoType->Identifier()->toString();
      int declNumber = 0;
      if (protoType->declList()) {
        ToyParser::DeclListContext *list = protoType->declList();
        while (list) {
          declNumber++;
          if (list->declList())
            list = list->declList();
          else
            break;
        }
      }
      funcDeclare(function->prototype()->idName, declNumber);
    }
    return visitChildren(ctx);
  }

  /// Function Definition Visitor
  /// - Register the function name, argument list, and return value into the
  /// symbol table.
  /// - Visit function prototype.
  /// - Visit fucntion block.
  /// - Process the return operation.
  virtual std::any visitFunDefine(ToyParser::FunDefineContext *ctx) override {
    llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(
        symbolTable);
    builder.setInsertionPointToEnd(theModule.getBody());
    // Visit function prototype.
    mlir::toy::FuncOp function =
        std::any_cast<mlir::toy::FuncOp>(visit(ctx->prototype()));
    mlir::Block &entryBlock = function.front();

    // Set the insertion point in the builder to the beginning of the function
    // body, it will be used throughout the codegen to create operations in this
    // function.
    builder.setInsertionPointToStart(&entryBlock);

    std::vector<std::string> args;
    if (ctx->prototype()->declList()) {
      ToyParser::DeclListContext *list = ctx->prototype()->declList();
      while (list->Identifier()) {
        args.push_back(list->Identifier()->toString());
        if (list->declList())
          list = list->declList();
        else
          break;
      }
    }
    // Declare all the function arguments in the symbol table.
    llvm::ArrayRef<std::string> protoArgs = args;
    for (auto value : llvm::zip(protoArgs, entryBlock.getArguments())) {
      declare(std::get<0>(value), std::get<1>(value));
    }

    // Visit fucntion block.
    visit(ctx->block());
    // Check the return status.
    // If there is no return expression at the end of the function, it will
    // generate a return operation automatically.
    mlir::toy::ReturnOp returnOp;
    if (!entryBlock.empty())
      returnOp = llvm::dyn_cast<mlir::toy::ReturnOp>(entryBlock.back());
    if (!returnOp) {
      mlir::Location location =
          loc(ctx->start->getLine(), ctx->start->getCharPositionInLine());
      builder.create<mlir::toy::ReturnOp>(location);
    } else if (returnOp.hasOperand()) {
      // Otherwise, if this return operation has an operand then add a result to
      // the function.
      std::vector<int64_t> shape;
      function.setType(builder.getFunctionType(
          function.getFunctionType().getInputs(), getType(shape)));
    }
    // If this function isn't main, then set the visibility to private.
    if (ctx->prototype()->Identifier()->toString() != "main")
      function.setPrivate();
    functionMap.insert({function.getName(), function});
    return 0;
  }

  /// Prototype Visitor
  virtual std::any visitPrototype(ToyParser::PrototypeContext *ctx) override {
    mlir::Location location =
        loc(ctx->start->getLine(), ctx->start->getCharPositionInLine());
    int varNumber = 0;
    // Get the number of arguments.
    if (ctx->declList()) {
      ToyParser::DeclListContext *list = ctx->declList();
      while (list->Identifier()) {
        varNumber++;
        if (list->declList())
          list = list->declList();
        else
          break;
      }
    }
    llvm::SmallVector<mlir::Type, 4> argTypes(
        varNumber, mlir::UnrankedTensorType::get(builder.getF64Type()));
    mlir::FunctionType funType = builder.getFunctionType(argTypes, llvm::None);
    auto func = builder.create<mlir::toy::FuncOp>(
        location, ctx->Identifier()->toString(), funType);
    return func;
  }

  /// Expression Visitor
  /// - If the expression is tensor literal, return the tensor MLIR value.
  /// - If the expression is function call or variable, visit the identifier.
  /// - If the expression is add expression or mul expression return add or mul
  /// value.
  virtual std::any visitExpression(ToyParser::ExpressionContext *ctx) override {
    mlir::Value value;
    if (ctx->tensorLiteral()) {
      return getTensor(ctx->tensorLiteral());
    } else if (ctx->identifierExpr()) {
      return visit(ctx->identifierExpr());
    } else if (ctx->Add() || ctx->Mul()) {
      // Derive the operation name from the binary operator. At the moment we
      // only support '+' and '*'.
      mlir::Value lhs = std::any_cast<mlir::Value>(visit(ctx->expression(0)));
      mlir::Value rhs = std::any_cast<mlir::Value>(visit(ctx->expression(1)));
      mlir::Location loaction =
          loc(ctx->start->getLine(), ctx->start->getCharPositionInLine());
      if (ctx->Add())
        value = builder.create<mlir::toy::AddOp>(loaction, lhs, rhs);
      else
        value = builder.create<mlir::toy::MulOp>(loaction, lhs, rhs);
      return value;
    }
    return value;
  }

  /// Variable Declaration Visitor
  /// - If the variable has the shape attribute, create the reshape operation.
  /// - Register the variable into the symbol table.
  virtual std::any visitVarDecl(ToyParser::VarDeclContext *ctx) override {
    // Get the variable MLIR value.
    mlir::Value value = std::any_cast<mlir::Value>(visit(ctx->expression()));
    // If the variable has the shape attribute, create the reshape operation.
    if (ctx->type()) {
      // [TODO][HIGH] try to use a better way to create the shape layout.
      std::vector<int64_t> v0;
      auto v1 = ctx->type()->Number();
      for (auto i : v1) {
        int64_t j = atoi(i->toString().c_str());
        v0.push_back(j);
      }
      mlir::Location location =
          loc(ctx->Identifier()->getSymbol()->getLine(),
              ctx->Identifier()->getSymbol()->getCharPositionInLine());
      value =
          builder.create<mlir::toy::ReshapeOp>(location, getType(v0), value);
    }
    // Register the variable into the symbol table.
    mlir::failed(declare(ctx->idName, value));
    return 0;
  }

  /// Identifier Expression Visitor
  /// - Process function call.
  /// - Process variables.
  virtual std::any
  visitIdentifierExpr(ToyParser::IdentifierExprContext *ctx) override {
    mlir::Value value;
    int argsNumber = 0;
    mlir::Location location =
        loc(ctx->start->getLine(), ctx->start->getCharPositionInLine());
    // If the identifier is a function call, visit and register all the
    // arguments. [TODO][LOW] add the semantic check (look up the symbol table)
    // for the function call.
    if (ctx->ParentheseOpen()) {
      mlir::Location location =
          loc(ctx->start->getLine(), ctx->start->getCharPositionInLine());
      llvm::SmallVector<mlir::Value, 4> oprands;
      for (ToyParser::ExpressionContext *i : ctx->expression()) {
        mlir::Value arg = std::any_cast<mlir::Value>(visit(i));
        oprands.push_back(arg);
        argsNumber++;
      }
      // If function call is a built-in operation, create the corresponding
      // operation.
      if (ctx->Identifier()->toString() == "print") {
        if (argsNumber != 1) {
          mlir::emitError(location)
              << "mismatch of function parameters 'print'";
          return nullptr;
        }
        mlir::Value arg = oprands[0];
        builder.create<mlir::toy::PrintOp>(location, arg);
        return 0;
      } else if (ctx->Identifier()->toString() == "transpose") {
        if (argsNumber != 1) {
          mlir::emitError(location)
              << "mlismatch of function parameters 'transpose'";
          return nullptr;
        }
        mlir::Value arg = oprands[0];
        value = builder.create<mlir::toy::TransposeOp>(location, arg);
        return value;
      }
      // Otherwise this is a call to a user-defined function. Calls to
      // user-defined functions are mapped to a custom call that takes the
      // callee name as an attribute.
      auto callee = functionMap.find(ctx->Identifier()->toString());
      if (callee == functionMap.end()) {
        mlir::emitError(location) << "error: no defined function '"
                                  << ctx->Identifier()->toString() << "'";
        return nullptr;
      }
      int numberdecl = funSymbolTable.lookup(ctx->Identifier()->toString());
      if (numberdecl != argsNumber) {
        mlir::emitError(location) << "error: mismatch of function parameters '"
                                  << ctx->Identifier()->toString() << "'";
        return nullptr;
      }
      // If the function call cannot be mapped to the built-in operation, create
      // the GenericCallOp.
      mlir::toy::FuncOp calledFunc = callee->second;
      value = builder.create<mlir::toy::GenericCallOp>(
          location, calledFunc.getFunctionType().getResult(0),
          mlir::SymbolRefAttr::get(builder.getContext(),
                                   ctx->Identifier()->toString()),
          oprands);
      return value;
    } else {
      // If the identifier is a variable, return the MLIR value from the symbol
      // table.
      value = symbolTable.lookup(ctx->Identifier()->toString());
      return value;
    }
  }

  /// Return Expression Visitor
  virtual std::any visitReturnExpr(ToyParser::ReturnExprContext *ctx) override {
    mlir::Location location =
        loc(ctx->start->getLine(), ctx->start->getCharPositionInLine());
    mlir::Value expr = nullptr;
    if (ctx->expression()) {
      expr = std::any_cast<mlir::Value>(visit(ctx->expression()));
    }
    // Generate return operation based on whether the function has the return
    // value.
    builder.create<mlir::toy::ReturnOp>(location,
                                        expr ? llvm::makeArrayRef(expr)
                                             : llvm::ArrayRef<mlir::Value>());
    return 0;
  }
};

#endif // MLIR_TOY_VISITOR_H
