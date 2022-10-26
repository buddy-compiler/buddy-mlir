#include "ToyBaseVisitor.h"
#include "ToyLexer.h"
#include "ToyParser.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "toy/Dialect.h"

class MLIRToyVisitor : public ToyBaseVisitor {
mlir::ModuleOp theModule;
mlir::OpBuilder builder;
std::string fileName;
llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;
bool returnFlag = false;

public:
MLIRToyVisitor(std::string filename, mlir::MLIRContext &context)
: builder(&context), fileName(filename) {
 theModule = mlir::ModuleOp::create(builder.getUnknownLoc()); 
}

mlir::ModuleOp getModule() { return theModule; }

mlir::Location loc(int line, int row) {
  return mlir::FileLineColLoc::get(builder.getStringAttr(fileName), line, row);
}

mlir::LogicalResult declare(llvm::StringRef var, mlir::Value value) {
  if (symbolTable.count(var))
    return mlir::failure();
  symbolTable.insert(var, value);
  return mlir::success();
}

mlir::Type getType(llvm::ArrayRef<int64_t> shape) {
  if (shape.empty())
    return mlir::UnrankedTensorType::get(builder.getF64Type());
  return mlir::RankedTensorType::get(shape, builder.getF64Type());
}

std::any tensor(ToyParser::TensorLiteralContext *ctx) {
  bool flag = false;
  std::vector<int64_t> dims;
  std::vector<double> data;
  dims.push_back(ctx->Comma().size() + 1);
  if (ctx->tensorLiteral(0)->tensorLiteral(0)) {
    flag = true;
    dims.push_back(ctx->tensorLiteral(0)->Comma().size() + 1);
  }
  auto list = ctx;
  if (flag)
    for (auto i : ctx->tensorLiteral()) {
      for (auto j : i->tensorLiteral()) {
        data.push_back(std::atof(j->Number()->toString().c_str()));
      }
    }
  else if (!flag) {
    for (auto i : ctx->tensorLiteral()) {
      data.push_back(std::atof(i->Number()->toString().c_str()));
    }
  }
  mlir::Type elementType = builder.getF64Type();
  auto type = getType(dims);
  auto dataType = mlir::RankedTensorType::get(dims, elementType);
  auto dataAttribute =
  mlir::DenseElementsAttr::get(dataType, llvm::makeArrayRef(data));
  auto loaction =
  loc(ctx->start->getLine(), ctx->start->getCharPositionInLine());
  mlir::Value value =
  builder.create<mlir::toy::ConstantOp>(loaction, type, dataAttribute);
  return value;
}

virtual std::any visitModule(ToyParser::ModuleContext *ctx) {
  return visitChildren(ctx);
}

virtual std::any visitExpression(ToyParser::ExpressionContext *ctx) {
  if (ctx->tensorLiteral()) {
    return tensor(ctx->tensorLiteral());
  } else if (ctx->identifierExpr()) {
    return visit(ctx->identifierExpr());
  }
}

virtual std::any visitReturnExpr(ToyParser::ReturnExprContext *ctx) {
  return visitChildren(ctx);
}

virtual std::any visitIdentifierExpr(ToyParser::IdentifierExprContext *ctx) {
  if (ctx->ParentheseOpen()){ 
    llvm::SmallVector<mlir::Value, 4> operands;
    for (auto i : ctx->expression()) {
      mlir::Value arg = std::any_cast<mlir::Value>(visit(i));
      operands.push_back(arg);
    }
    if (ctx->Identifier()->toString() == "print") {
      mlir::Value input = operands[0];
      mlir::Location location = loc(ctx->start->getLine(), ctx->start->getCharPositionInLine());
      builder.create<mlir::toy::PrintOp>(location, input);
      return 0;
    } 
    mlir::Location location = loc(ctx->start->getLine(), ctx->start->getCharPositionInLine());
    mlir::Value value = builder.create<mlir::toy::GenericCallOp>(location, ctx->Identifier()->toString(), operands);
    return value;
  } else {
    mlir::Value value = symbolTable.lookup(ctx->Identifier()->toString());
    return value;
  }
}

virtual std::any visitTensorLiteral(ToyParser::TensorLiteralContext *ctx) {
  return visitChildren(ctx);
}

virtual std::any visitVarDecl(ToyParser::VarDeclContext *ctx) {
  mlir::Value value = std::any_cast<mlir::Value>(visit(ctx->expression()));
  if (ctx->type()){
    std::vector<int64_t> v0;
    auto v1 = ctx->type()->Number();
    for (auto i : v1) {
      auto j = atoi(i->toString().c_str());
      v0.push_back(j);
    }
    mlir::Type res0 = getType(v0);
    mlir::Value input = value;
    mlir::Location location = loc(ctx->start->getLine(), ctx->start->getCharPositionInLine());
    value = builder.create<mlir::toy::ReshapeOp>(location, res0, input);
  }
  mlir::failed(declare(ctx->idName, value));
  return 0;
}

virtual std::any visitType(ToyParser::TypeContext *ctx) {
  return visitChildren(ctx);
}

virtual std::any visitFunDefine(ToyParser::FunDefineContext *ctx) {
  returnFlag = false;
  llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);
  builder.setInsertionPointToEnd(theModule.getBody());
  visit(ctx->prototype());
  visit(ctx->block());
  if (!returnFlag) {
    mlir::Location location = loc(ctx->start->getLine(), ctx->start->getCharPositionInLine());
    builder.create<mlir::toy::ReturnOp>(location, llvm::ArrayRef<mlir::Value>());
  }
  return 0;
}

virtual std::any visitPrototype(ToyParser::PrototypeContext *ctx) {
  auto varNumber = 0;
    if (ctx->declList()) {
      auto list = ctx->declList();
      while (list->Identifier()) {
        varNumber++;
        if (list->declList())
          list = list->declList();
        else
          break;
    }
  }
  llvm::SmallVector<mlir::Type, 4> argTypes(varNumber, mlir::UnrankedTensorType::get(builder.getF64Type()));
  mlir::FunctionType function_type = builder.getFunctionType(argTypes, llvm::None);
  mlir::Location location = loc(ctx->start->getLine(), ctx->start->getCharPositionInLine());
  mlir::toy::FuncOp func = builder.create<mlir::toy::FuncOp>(location, ctx->Identifier()->toString(), function_type);
  mlir::Block& entryBlock = func.front();
  builder.setInsertionPointToStart(&entryBlock);
  return 0;
}

virtual std::any visitDeclList(ToyParser::DeclListContext *ctx) {
  return visitChildren(ctx);
}

virtual std::any visitBlock(ToyParser::BlockContext *ctx) {
  return visitChildren(ctx);
}

virtual std::any visitBlockExpr(ToyParser::BlockExprContext *ctx) {
  return visitChildren(ctx);
}

};

