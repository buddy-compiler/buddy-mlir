//===- LegalizeShmemOutlining.cpp -----------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
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
// This file implements the pass that legalizes shared memory operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/GPU/Utils/GPUUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>

#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
using namespace mlir;
using namespace vector;

//===---------------------------------------------------------------------===//
// From mlir/lib/Dialect/GPU/Transforms/KernelOutlining.cpp
//===---------------------------------------------------------------------===//

namespace mlir {
#define GEN_PASS_DEF_GPULAUNCHSINKINDEXCOMPUTATIONS
#define GEN_PASS_DEF_GPUKERNELOUTLINING
#include "mlir/Dialect/GPU/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

template <typename OpTy>
static void createForAllDimensions(OpBuilder &builder, Location loc,
                                   SmallVectorImpl<Value> &values) {
  for (auto dim : {gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z})
    values.push_back(builder.create<OpTy>(loc, builder.getIndexType(), dim));
}

/// Adds operations generating block/thread ids and grid/block dimensions at the
/// beginning of the `launchFuncOpBody` region. Add mapping from argument in
/// entry block of `launchOpBody`, to the corresponding result value of the
/// added operations.
static void injectGpuIndexOperations(Location loc, Region &launchFuncOpBody,
                                     Region &launchOpBody, IRMapping &map) {
  OpBuilder builder(loc->getContext());
  Block &firstBlock = launchOpBody.front();
  builder.setInsertionPointToStart(&launchFuncOpBody.front());
  SmallVector<Value, 12> indexOps;
  createForAllDimensions<gpu::BlockIdOp>(builder, loc, indexOps);
  createForAllDimensions<gpu::ThreadIdOp>(builder, loc, indexOps);
  createForAllDimensions<gpu::GridDimOp>(builder, loc, indexOps);
  createForAllDimensions<gpu::BlockDimOp>(builder, loc, indexOps);
  // Replace the leading 12 function args with the respective thread/block index
  // operations. Iterate backwards since args are erased and indices change.
  for (const auto &indexOp : enumerate(indexOps))
    map.map(firstBlock.getArgument(indexOp.index()), indexOp.value());
}

/// Return the provided KernelDim3 as an array of i32 constants if possible.
static DenseI32ArrayAttr maybeConstantDimsAttr(gpu::KernelDim3 dims) {
  SmallVector<int32_t, 3> constants;
  MLIRContext *ctx = dims.x.getContext();
  for (Value v : {dims.x, dims.y, dims.z}) {
    APInt constValue;
    if (!matchPattern(v, m_ConstantInt(&constValue)))
      return nullptr;
    // In the event someone called for a too-large block or grid dimension,
    // don't set bounds as it is likely to cause more confusing behavior.
    if (constValue.ugt(std::numeric_limits<uint32_t>::max()))
      return nullptr;
    constants.push_back(
        constValue.getLimitedValue(std::numeric_limits<uint32_t>::max()));
  }
  return DenseI32ArrayAttr::get(ctx, constants);
}

/// Outline the `gpu.launch` operation body into a kernel function. Replace
/// `gpu.terminator` operations by `gpu.return` in the generated function.
/// Set block and grid size bounds if known.
static gpu::GPUFuncOp outlineKernelFuncImpl(gpu::LaunchOp launchOp,
                                            StringRef kernelFnName,
                                            SetVector<Value> &operands) {
  Location loc = launchOp.getLoc();
  // Create a builder with no insertion point, insertion will happen separately
  // due to symbol table manipulation.
  OpBuilder builder(launchOp.getContext());
  Region &launchOpBody = launchOp.getBody();

  // Identify uses from values defined outside of the scope of the launch
  // operation.
  getUsedValuesDefinedAbove(launchOpBody, operands);

  // Create the gpu.func operation.
  SmallVector<Type, 4> kernelOperandTypes;
  kernelOperandTypes.reserve(operands.size());
  for (Value operand : operands) {
    kernelOperandTypes.push_back(operand.getType());
  }
  FunctionType type =
      FunctionType::get(launchOp.getContext(), kernelOperandTypes, {});
  auto outlinedFunc = builder.create<gpu::GPUFuncOp>(
      loc, kernelFnName, type,
      TypeRange(ValueRange(launchOp.getWorkgroupAttributions())),
      TypeRange(ValueRange(launchOp.getPrivateAttributions())));
  outlinedFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                        builder.getUnitAttr());

  // If we can infer bounds on the grid and/or block sizes from the arguments
  // to the launch op, propagate them to the generated kernel. This is safe
  // because multiple launches with the same body are not deduplicated.
  if (auto blockBounds =
          maybeConstantDimsAttr(launchOp.getBlockSizeOperandValues()))
    outlinedFunc->setAttr(outlinedFunc.getKnownBlockSizeAttrName(),
                          blockBounds);
  if (auto gridBounds =
          maybeConstantDimsAttr(launchOp.getGridSizeOperandValues()))
    outlinedFunc->setAttr(outlinedFunc.getKnownGridSizeAttrName(), gridBounds);

  IRMapping map;

  // Map the arguments corresponding to the launch parameters like blockIdx,
  // threadIdx, etc.
  Region &outlinedFuncBody = outlinedFunc.getBody();
  injectGpuIndexOperations(loc, outlinedFuncBody, launchOpBody, map);

  // Map memory attributions from the LaunOp op to the GPUFuncOp attributions.
  for (const auto &[launchArg, funcArg] :
       llvm::zip(launchOp.getWorkgroupAttributions(),
                 outlinedFunc.getWorkgroupAttributions()))
    map.map(launchArg, funcArg);
  for (const auto &[launchArg, funcArg] :
       llvm::zip(launchOp.getPrivateAttributions(),
                 outlinedFunc.getPrivateAttributions()))
    map.map(launchArg, funcArg);

  // Map arguments from gpu.launch region to the arguments of the gpu.func
  // operation.
  Block &entryBlock = outlinedFuncBody.front();
  for (const auto &operand : enumerate(operands))
    map.map(operand.value(), entryBlock.getArgument(operand.index()));

  // Clone the region of the gpu.launch operation into the gpu.func operation.
  // TODO: If cloneInto can be modified such that if a mapping for
  // a block exists, that block will be used to clone operations into (at the
  // end of the block), instead of creating a new block, this would be much
  // cleaner.
  launchOpBody.cloneInto(&outlinedFuncBody, map);

  // Branch from entry of the gpu.func operation to the block that is cloned
  // from the entry block of the gpu.launch operation.
  Block &launchOpEntry = launchOpBody.front();
  Block *clonedLaunchOpEntry = map.lookup(&launchOpEntry);
  builder.setInsertionPointToEnd(&entryBlock);
  builder.create<cf::BranchOp>(loc, clonedLaunchOpEntry);

  outlinedFunc.walk([](gpu::TerminatorOp op) {
    OpBuilder replacer(op);
    replacer.create<gpu::ReturnOp>(op.getLoc());
    op.erase();
  });
  return outlinedFunc;
}

/// Replace `gpu.launch` operations with an `gpu.launch_func` operation
/// launching `kernelFunc`. The kernel func contains the body of the
/// `gpu.launch` with constant region arguments inlined.
static void convertToLaunchFuncOp(gpu::LaunchOp launchOp,
                                  gpu::GPUFuncOp kernelFunc,
                                  ValueRange operands) {
  OpBuilder builder(launchOp);
  // The launch op has an optional dynamic shared memory size. If it doesn't
  // exist, we use zero.
  Value asyncToken = launchOp.getAsyncToken();
  auto launchFunc = builder.create<gpu::LaunchFuncOp>(
      launchOp.getLoc(), kernelFunc, launchOp.getGridSizeOperandValues(),
      launchOp.getBlockSizeOperandValues(),
      launchOp.getDynamicSharedMemorySize(), operands,
      asyncToken ? asyncToken.getType() : nullptr,
      launchOp.getAsyncDependencies());
  launchOp.replaceAllUsesWith(launchFunc);
  launchOp.erase();
}

/// Pass that moves the kernel of each LaunchOp into its separate nested module.
///
/// This pass moves the kernel code of each LaunchOp into a function created
/// inside a nested module. It also creates an external function of the same
/// name in the parent module.
///
/// The gpu.modules are intended to be compiled to a cubin blob independently in
/// a separate pass. The external functions can then be annotated with the
/// symbol of the cubin accessor function.

namespace {
class LegalizeShmemOutliningPass
    : public PassWrapper<LegalizeShmemOutliningPass, OperationPass<ModuleOp>> {
public:
  std::vector<Operation *> shmemAllocations;
  std::map<Operation *, Operation *> shmemGlobalPairs;
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeShmemOutliningPass)
  StringRef getArgument() const final { return "legalize-shmem-outlining"; }
  StringRef getDescription() const final {
    return "Convert shared memory outlining to global memref declaration.";
  }

  void runOnOperation() override {
    SymbolTable symbolTable(getOperation());

    bool modified = false;
    for (auto func : getOperation().getOps<func::FuncOp>()) {
      // Insert just after the function.
      Block::iterator insertPt(func->getNextNode());

      // Collects all allocations for shared memory outside the kernel.
      // The collection must happen before the kernel outlining.
      // It moves back all shared allocations back into their GPU body
      // Allowing the functions to create kernels without shared memory
      // as parameters.
      func.walk([&](memref::AllocOp allocOp) {
        auto result = allocOp->getResult(0);
        auto memrefType = dyn_cast<MemRefType>(result.getType());
        auto memorySpace = memrefType.getMemorySpace();
        if (!memorySpace)
          return WalkResult::advance();
        else {
          if (auto intMemorySpace = llvm::dyn_cast<IntegerAttr>(memorySpace)) {
            if (intMemorySpace.getInt() != 3) {
              return WalkResult::advance();
            }
          } else if (auto gpuMemorySpace =
                         llvm::dyn_cast<gpu::AddressSpaceAttr>(memorySpace)) {
            if (gpuMemorySpace.getValue() != gpu::AddressSpace::Workgroup) {
              return WalkResult::advance();
            }
          } else
            return WalkResult::advance();
        }
        auto users = allocOp->getUsers();
        for (auto user : users) {
          if (isa<memref::DeallocOp>(user)) {
            user->erase();
            continue;
          }
          // Locates the gpu kernel wrapper
          auto launchOp = user->getParentOfType<gpu::LaunchOp>();
          OpBuilder builder(launchOp);
          builder.setInsertionPointToStart(
              &launchOp.getBody().getBlocks().front());
          auto newAllocOp =
              builder.create<memref::AllocOp>(launchOp.getLoc(), memrefType);
          allocOp->replaceAllUsesWith(newAllocOp);
          allocOp->erase();
          break;
        }
        return WalkResult::advance();
      });

      auto funcWalkResult = func.walk([&](gpu::LaunchOp op) {
        SetVector<Value> operands;
        std::string kernelFnName =
            Twine(op->getParentOfType<func::FuncOp>().getName(), "_kernel")
                .str();

        gpu::GPUFuncOp outlinedFunc =
            outlineKernelFuncImpl(op, kernelFnName, operands);

        // Create nested module and insert outlinedFunc. The module will
        // originally get the same name as the function, but may be renamed on
        // insertion into the parent module.
        auto kernelModule = createKernelModule(outlinedFunc, symbolTable);
        symbolTable.insert(kernelModule, insertPt);

        size_t counter = 0;
        // Walk the funcop and replace all shmem allocations with global memref
        outlinedFunc->walk([&](memref::AllocOp allocOp) {
          auto result = allocOp->getResult(0);
          auto memrefType = dyn_cast<MemRefType>(result.getType());
          auto memorySpace = memrefType.getMemorySpace();
          if (!memorySpace)
            allocOp->emitOpError()
                << "Found non-shared memory inside a kernel function";
          else {
            if (auto intMemorySpace =
                    llvm::dyn_cast<IntegerAttr>(memorySpace)) {
              if (intMemorySpace.getInt() != 3) {
                return WalkResult::advance();
              }
            } else if (auto gpuMemorySpace =
                           llvm::dyn_cast<gpu::AddressSpaceAttr>(memorySpace)) {
              if (gpuMemorySpace.getValue() != gpu::AddressSpace::Workgroup) {
                return WalkResult::advance();
              }
            } else
              return WalkResult::advance();
          }

          OpBuilder builder(outlinedFunc);

          auto name = Twine("shmem_", std::to_string(counter++)).str();

          auto globalOp = builder.create<memref::GlobalOp>(
              kernelModule->getLoc(),
              /*sym_name=*/name,
              /*sym_visibility=*/builder.getStringAttr("private"),
              /*type=*/memrefType,
              /*initial_value=*/ElementsAttr(),
              /*constant=*/false,
              /*alignment=*/builder.getI64IntegerAttr(64));
          // symbolTable.insert(globalOp);
          builder.setInsertionPointAfter(allocOp);
          Value getGlobalOp = builder.create<memref::GetGlobalOp>(
              allocOp->getLoc(), globalOp.getType(), name);
          allocOp.replaceAllUsesWith(getGlobalOp);
          allocOp->erase();
          return WalkResult::advance();
        });

        // Potentially changes signature, pulling in constants.
        convertToLaunchFuncOp(op, outlinedFunc, operands.getArrayRef());
        modified = true;
        return WalkResult::advance();
      });
      if (funcWalkResult.wasInterrupted())
        return signalPassFailure();
    }

    // If any new module was inserted in this module, annotate this module as
    // a container module.
    if (modified)
      getOperation()->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                              UnitAttr::get(&getContext()));
  }

private:
  /// Returns a gpu.module containing kernelFunc and all callees (recursive).
  gpu::GPUModuleOp createKernelModule(gpu::GPUFuncOp kernelFunc,
                                      const SymbolTable &parentSymbolTable) {
    // TODO: This code cannot use an OpBuilder because it must be inserted into
    // a SymbolTable by the caller. SymbolTable needs to be refactored to
    // prevent manual building of Ops with symbols in code using SymbolTables
    // and then this needs to use the OpBuilder.
    auto *context = getOperation().getContext();
    OpBuilder builder(context);
    auto kernelModule = builder.create<gpu::GPUModuleOp>(kernelFunc.getLoc(),
                                                         kernelFunc.getName());

    SymbolTable symbolTable(kernelModule);
    symbolTable.insert(kernelFunc);

    SmallVector<Operation *, 8> symbolDefWorklist = {kernelFunc};
    while (!symbolDefWorklist.empty()) {
      if (std::optional<SymbolTable::UseRange> symbolUses =
              SymbolTable::getSymbolUses(symbolDefWorklist.pop_back_val())) {
        for (SymbolTable::SymbolUse symbolUse : *symbolUses) {
          StringRef symbolName =
              cast<FlatSymbolRefAttr>(symbolUse.getSymbolRef()).getValue();
          if (symbolTable.lookup(symbolName))
            continue;

          Operation *symbolDefClone =
              parentSymbolTable.lookup(symbolName)->clone();
          symbolDefWorklist.push_back(symbolDefClone);
          symbolTable.insert(symbolDefClone);
        }
      }
    }

    return kernelModule;
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// LegalizeShmemOutliningPass
//===----------------------------------------------------------------------===//

namespace mlir {
namespace buddy {
void registerLegalizeShmemOutliningPass() {
  PassRegistration<LegalizeShmemOutliningPass>();
}
} // namespace buddy
} // namespace mlir
