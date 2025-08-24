#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

// Print attribute information for an operation.
void printAttrsInfo(Operation *op) {
  for (NamedAttribute attr : op->getAttrs()) {
    llvm::outs() << attr.getName() << ", value: ";
    attr.getValue().print(llvm::outs());
    llvm::outs() << "\n";
  }
}

// Print operand information for an operation.
void printOperandsInfo(Operation *op) {
  size_t cnt = 0;
  for (Value operand : op->getOperands()) {
    Type type = operand.getType();
    if (auto shapedType = type.dyn_cast<ShapedType>()) {
      llvm::ArrayRef<int64_t> shape = shapedType.getShape();
      llvm::outs() << "operand" << cnt++ << ": ";
      for (int64_t val : shape) 
        llvm::outs() <<  val << 'x';
      Type eleType = shapedType.getElementType();
      if (eleType.isInteger(32)) 
        llvm::outs() << "i32\n";
      else if (eleType.isF32())
        llvm::outs() << "f32\n";
      else if (eleType.isF64())
        llvm::outs() << "f64\n";
      else if (eleType.isBF16())
        llvm::outs() << "bf16\n";
      else 
        llvm::outs() << "not i32, f32, f64, bf16\n";
    }
  }
}

// Print result information for an operation.
void printResultsInfo(Operation *op) {
  size_t cnt = 0; op->getOperand(0).getDefiningOp();
  for (Value result : op->getResults()) {
    Type type = result.getType();
    // Type eleType = dyn_cast<RankedTensorType>(type).getElementType();
    if (auto shapedType = type.dyn_cast<ShapedType>()) {
      llvm::ArrayRef<int64_t> shape = shapedType.getShape();
      llvm::outs() << "result" << cnt++ << ": ";
      for (int64_t val : shape) 
        llvm::outs() <<  val << 'x';
      Type eleType = shapedType.getElementType();
      if (eleType.isF32())
        llvm::outs() << "f32\n";
      else if (eleType.isF64())
        llvm::outs() << "f64\n";
      else if (eleType.isBF16())
        llvm::outs() << "bf16\n";
      else 
        llvm::outs() << "not f32, f64, bf16\n";
    }
  }
}

// Print location for an operation.
void printLocInfo(Operation *op) {
  Location loc = op->getLoc();
  loc.print(llvm::outs()); 
  llvm::outs() << "\n";
}

//===----------------------------------------------------------------------===//
// PrintInfoPass
//===----------------------------------------------------------------------===//

namespace {
class PrintInfoPass
    : public PassWrapper<PrintInfoPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrintInfoPass)
  explicit PrintInfoPass() = default;

  StringRef getArgument() const final { return "print-info"; }
  StringRef getDescription() const final { return "Print info for neural network."; }

  void runOnOperation() override;

  void printDependence(Operation* op);
  // Print symmary of op info.
  void printSummary();

private:
  llvm::StringMap<int64_t> opCount;
  llvm::DenseMap<Operation*, int> opID;
};
} // end anonymous namespace.

void PrintInfoPass::runOnOperation() {
  opCount.clear();
  size_t nextId = 0;
  Operation* thisOp = getOperation();
  if (thisOp->getName().getStringRef() == "builtin.module") {
    llvm::outs() << thisOp->getName().getStringRef() << '\n';
    Operation* funcOp = &thisOp->getRegion(0).getBlocks().front().getOperations().front();
    llvm::outs() << funcOp->getName().getStringRef() << '\n';
    FunctionType funcType = dyn_cast<mlir::func::FuncOp>(funcOp).getFunctionType();
    // for (Type ty : funcType.getInputs()) {
    //   llvm::outs() << "input" << nextId++ << ": ";
    //   ty.print(llvm::outs());
    //   llvm::outs() << "\n";
    // }
  }
  // printOperandsInfo(thisOp);
  // printResultsInfo(thisOp);

  // Compute the operation statistics for the currently visited operation.
  nextId = 0;
  getOperation()->walk(
      [&](Operation *op) { 
        ++opCount[op->getName().getStringRef()]; 
        opID[op] = nextId++;
        llvm::outs() << op->getName().getStringRef() << '-'<< opID[op] <<'\n';
        // Print operands shape and element type.
        llvm::StringRef opNameString = op->getName().getStringRef();
        if (opNameString == "tosa.conv2d" || opNameString == "tosa.matmul") {
          printAttrsInfo(op);
          printOperandsInfo(op);
          printResultsInfo(op);
          printDependence(op);
        } 
      });
  
  printSummary();
}

void PrintInfoPass::printDependence(Operation* op) {
  for (Value val : op->getOperands()) {
    auto *parent = val.getDefiningOp();
    if (parent == nullptr) 
      llvm::outs() << "No defining operation, this is an input for the function.\n";
    else {
      llvm::outs() << "depend on: " << parent->getName().getStringRef() << '-' << opID[parent] << '\n';
    }
  }
}

void PrintInfoPass::printSummary() {
    llvm::outs() << "Operations information:\n";
    llvm::outs() << "-----------------------\n";
    SmallVector<StringRef, 64> sorted(opCount.keys());
    llvm::sort(sorted);
  
    // Split an operation name from its dialect prefix.
    auto splitOperationName = [](StringRef opName) {
      auto splitName = opName.split('.');
      return splitName.second.empty() ? std::make_pair("", splitName.first)
                                      : splitName;
    };
  
    // Compute the largest dialect and operation name.
    size_t maxLenOpName = 0, maxLenDialect = 0;
    for (const auto &key : sorted) {
      auto [dialectName, opName] = splitOperationName(key);
      maxLenDialect = std::max(maxLenDialect, dialectName.size());
      maxLenOpName = std::max(maxLenOpName, opName.size());
    }
  
    for (const auto &key : sorted) {
      auto [dialectName, opName] = splitOperationName(key);
  
      // Left-align the names (aligning on the dialect) and right-align the count
      // below. The alignment is for readability and does not affect CSV/FileCheck
      // parsing.
      if (dialectName.empty())
        llvm::outs() << "EMPTY.";
      else
        llvm::outs() << llvm::right_justify(dialectName, maxLenDialect + 2) << '.';
  
      // Left justify the operation name.
      llvm::outs() << llvm::left_justify(opName, maxLenOpName) << " , " << opCount[key]
         << '\n';
    }
  }


namespace mlir {
namespace buddy {
void registerPrintInfoPass() { 
  PassRegistration<PrintInfoPass>(); 
}
} // namespace buddy
} // namespace mlir
