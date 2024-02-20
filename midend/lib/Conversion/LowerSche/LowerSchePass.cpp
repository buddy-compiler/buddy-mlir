//====- LowerSchePass.cpp - Sche Dialect Lowering Pass  -------------------===//
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
// This file defines sche dialect lowering pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Bud/BudDialect.h"
#include "Bud/BudOps.h"
#include "Sche/ScheDialect.h"
#include "Sche/ScheOps.h"

#include <unordered_map>

using namespace mlir;
using namespace buddy;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

class WaitOpScheLowering : public ConversionPattern {
public:
  explicit WaitOpScheLowering(TypeConverter &typeConverter,
                              MLIRContext *context)
      : ConversionPattern(typeConverter, sche::WaitOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    assert(operands.size() == 1);
    auto loc = op->getLoc();
    // auto typeConverter = getTypeConverter();
    rewriter.setInsertionPoint(op);
    rewriter.create<async::AwaitOp>(loc, operands[0]);
    rewriter.eraseOp(op);
    return success();
  }
};

// lower to GPU Dialect
class OnDeviceOpScheLowering : public ConversionPattern {
public:
  explicit OnDeviceOpScheLowering(TypeConverter &typeConverter,
                                  MLIRContext *context)
      : ConversionPattern(typeConverter, sche::OnDeviceOp::getOperationName(),
                          1, context) {}

  // convert operands with tensor or vector type into memref operands, and
  // register these operands to GPU
  OpBuilder::InsertPoint
  convertOperands(mlir::ConversionPatternRewriter &rewriter,
                  ValueRange operands, IRMapping &mp, Location &loc,
                  OpBuilder::InsertPoint insertPointBeforeOp,
                  OpBuilder::InsertPoint insertPointToBlockStart) const {
    rewriter.restoreInsertionPoint(insertPointBeforeOp);
    for (auto v : operands) {
      auto t = v.getType();
      if (isa<TensorType>(t)) {
        auto shape = dyn_cast<TensorType>(t).getShape();
        auto ele_type = dyn_cast<TensorType>(t).getElementType();
        auto to_memref_op = rewriter.create<bufferization::ToMemrefOp>(
            loc, MemRefType::get(shape, ele_type), v);
        mp.map(v, to_memref_op.getResult());

        auto memref_cast_op = rewriter.create<memref::CastOp>(
            loc, UnrankedMemRefType::get(ele_type, {}),
            to_memref_op.getResult());
        rewriter.create<gpu::HostRegisterOp>(loc, memref_cast_op.getResult());
      } else if (isa<VectorType>(t)) {
        auto shape = dyn_cast<VectorType>(t).getShape();
        auto ele_type = dyn_cast<VectorType>(t).getElementType();
        auto mem_type = MemRefType::get(shape, ele_type);
        auto alloc_op = rewriter.create<memref::AllocOp>(loc, mem_type);
        auto memref_cast_op = rewriter.create<memref::CastOp>(
            loc, UnrankedMemRefType::get(ele_type, {}), alloc_op.getResult());
        rewriter.create<gpu::HostRegisterOp>(loc, memref_cast_op.getResult());
        auto idx0 =
            rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0))
                .getResult();
        llvm::SmallVector<Value> indices(shape.size(), idx0);
        rewriter.create<vector::TransferWriteOp>(loc, v, alloc_op.getResult(),
                                                 indices);
        mp.map(v, alloc_op.getResult());
      } else if (isa<UnrankedMemRefType>(t)) {
        rewriter.create<gpu::HostRegisterOp>(loc, v);
      } else if (isa<MemRefType>(t)) {
        auto memref_type = dyn_cast<MemRefType>(t);
        auto memref_cast_op = rewriter.create<memref::CastOp>(
            loc,
            UnrankedMemRefType::get(memref_type.getElementType(),
                                    memref_type.getMemorySpace()),
            v);
        rewriter.create<gpu::HostRegisterOp>(loc, memref_cast_op.getResult());
      } else {
        continue;
      }
    }

    rewriter.restoreInsertionPoint(insertPointToBlockStart);
    for (auto v : operands) {
      auto t = v.getType();
      if (isa<TensorType>(t)) {
        auto to_tensor_op = rewriter.create<bufferization::ToTensorOp>(
            loc, t, mp.lookup<Value>(v));
        mp.map(v, to_tensor_op.getResult());
      } else if (isa<VectorType>(t)) {
        auto idx0 =
            rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0))
                .getResult();
        llvm::SmallVector<Value> indices(
            dyn_cast<VectorType>(t).getShape().size(), idx0);
        auto transfer_read_op = rewriter.create<vector::TransferReadOp>(
            loc, dyn_cast<VectorType>(t), mp.lookup<Value>(v), indices);
        mp.map(v, transfer_read_op.getResult());
      }
    }

    return rewriter.saveInsertionPoint();
  }

  // convert results with tensor or vector type into memref , and register these
  // results to GPU
  SmallVector<Value>
  convertResults(mlir::ConversionPatternRewriter &rewriter, ValueRange results,
                 IRMapping &mp, Location &loc,
                 OpBuilder::InsertPoint insertPointBeforeOp,
                 OpBuilder::InsertPoint insertPointAfterGpuLaunchOp) const {
    rewriter.restoreInsertionPoint(insertPointBeforeOp);
    SmallVector<Value> result_memrefs;
    for (auto v : results) {
      MemRefType mem_type;
      auto t = v.getType();
      // TODO: must have the rank
      if (isa<TensorType>(t)) {
        auto shape = dyn_cast<TensorType>(t).getShape();
        auto ele_type = dyn_cast<TensorType>(t).getElementType();
        mem_type = MemRefType::get(shape, ele_type);
      } else if (isa<VectorType>(t)) {
        auto shape = dyn_cast<VectorType>(t).getShape();
        auto ele_type = dyn_cast<VectorType>(t).getElementType();
        mem_type = MemRefType::get(shape, ele_type);
      } else if (isa<MemRefType>(t)) {
        mem_type = dyn_cast<MemRefType>(t);
      } else {
        mem_type = MemRefType::get({1}, t);
      }
      auto alloc_op = rewriter.create<memref::AllocOp>(loc, mem_type);
      result_memrefs.push_back(alloc_op.getResult());
      auto memref_cast_op = rewriter.create<memref::CastOp>(
          loc,
          UnrankedMemRefType::get(mem_type.getElementType(),
                                  mem_type.getMemorySpace()),
          alloc_op.getResult());
      rewriter.create<gpu::HostRegisterOp>(loc, memref_cast_op.getResult());
    }

    rewriter.restoreInsertionPoint(insertPointAfterGpuLaunchOp);
    // convert result'type into original type for returning
    int i = 0;
    for (auto v : results) {
      auto t = v.getType();
      // TODO: must have the rank
      if (isa<TensorType>(t)) {
        auto to_tensor_op = rewriter.create<bufferization::ToTensorOp>(
            loc, t, result_memrefs[i++]);
        v.replaceAllUsesWith(to_tensor_op.getResult());
      } else if (isa<VectorType>(t)) {
        auto idx0 =
            rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0))
                .getResult();
        llvm::SmallVector<Value> indices(
            dyn_cast<VectorType>(t).getShape().size(), idx0);
        auto transfer_read_op = rewriter.create<vector::TransferReadOp>(
            loc, dyn_cast<VectorType>(t), result_memrefs[i++], indices);
        v.replaceAllUsesWith(transfer_read_op.getResult());
      } else if (isa<MemRefType>(t)) {
        v.replaceAllUsesWith(result_memrefs[i++]);
      } else {
        auto idx0 =
            rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0))
                .getResult();
        auto load_op = rewriter.create<memref::LoadOp>(
            loc, v.getType(), result_memrefs[i++], ValueRange{idx0});
        v.replaceAllUsesWith(load_op.getResult());
      }
    }
    return result_memrefs;
  }

  // OnDeviceOp from ScfForOp conversion
  void lowerFromForOp(scf::ForOp forOp, gpu::LaunchOp gpuLaunchOp,
                      OpBuilder::InsertPoint insertPointBeforeOp,
                      OpBuilder::InsertPoint insertPointInGpuLaunchBody,
                      Location loc, PatternRewriter &rewriter, Value gridX,
                      Value gridY, Value gridZ, Value blockX, Value blockY,
                      Value blockZ) const {
    rewriter.restoreInsertionPoint(insertPointBeforeOp);
    Value upperBound = forOp.getUpperBound();
    Value lowerBound = forOp.getLowerBound();
    Value step = forOp.getStep();
    // Calculate the step size range required in a block
    auto range = rewriter.create<arith::SubIOp>(loc, upperBound, lowerBound);
    Value stepRange =
        rewriter.create<arith::DivSIOp>(loc, range.getResult(), step);
    Value stepRangeInBlock =
        rewriter.create<arith::DivSIOp>(loc, stepRange, gridX);
    Value remInBlock = rewriter.create<arith::RemSIOp>(loc, stepRange, gridX);
    auto idx0 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0))
            .getResult();
    auto idx1 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1))
            .getResult();

    rewriter.restoreInsertionPoint(insertPointInGpuLaunchBody);
    Value start = rewriter.create<arith::MulIOp>(loc, stepRangeInBlock,
                                                 gpuLaunchOp.getBlockIds().x);
    start = rewriter.create<arith::AddIOp>(loc, start, lowerBound);
    Value cmp_rem_blkId =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                       remInBlock, gpuLaunchOp.getBlockIds().x);
    Value cmp_rem_blkId_index = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getIndexType(), cmp_rem_blkId);
    stepRangeInBlock = rewriter.create<arith::AddIOp>(loc, cmp_rem_blkId_index,
                                                      stepRangeInBlock);
    Value min = rewriter.create<arith::MinUIOp>(
        loc, gpuLaunchOp.getBlockIds().x, remInBlock);
    start = rewriter.create<arith::AddIOp>(loc, start, min);
    // Calculate the step size range required in a thread
    Value stepRangeInThread = rewriter.create<arith::DivSIOp>(
        loc, stepRangeInBlock, gpuLaunchOp.getBlockSizeX());
    Value remInThread = rewriter.create<arith::RemSIOp>(
        loc, stepRangeInBlock, gpuLaunchOp.getBlockSizeX());
    Value cmp_rem_threadId = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, remInThread,
        gpuLaunchOp.getThreadIds().x);
    Value cmp_rem_threadId_index = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getIndexType(), cmp_rem_threadId);
    stepRangeInThread = rewriter.create<arith::AddIOp>(
        loc, cmp_rem_threadId_index, stepRangeInThread);

    rewriter.create<arith::AddIOp>(loc, start, stepRangeInThread);

    rewriter.create<scf::ForOp>(
        loc, idx0, stepRangeInThread, idx1, forOp.getInitArgs(),
        [&](OpBuilder &builder, Location loc, Value iv, ValueRange iterArgs) {
          Block &bodyBlock =
              forOp.getRegion().front(); // original forOp's bodyBlock
          IRMapping mp;
          iv = builder.create<arith::MulIOp>(loc, iv,
                                             gpuLaunchOp.getBlockSizeX());
          iv = builder.create<arith::AddIOp>(loc, iv,
                                             gpuLaunchOp.getThreadIds().x);
          iv = builder.create<arith::MulIOp>(loc, iv, step);
          iv = builder.create<arith::AddIOp>(loc, iv, start);
          mp.map(bodyBlock.getArgument(0), iv);
          for (auto &&[a, b] :
               llvm::zip(bodyBlock.getArguments().drop_front(), iterArgs)) {
            mp.map(a, b);
          }
          for (auto &&op_ : bodyBlock.getOperations()) {
            builder.insert(op_.clone(mp));
          }
        });
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto onDeviceOp = dyn_cast<sche::OnDeviceOp>(op);

    rewriter.setInsertionPoint(op);

    auto grid_x =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(3))
            .getResult();
    auto grid_y =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1))
            .getResult();
    auto grid_z =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1))
            .getResult();
    auto block_x =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(3))
            .getResult();
    auto block_y =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1))
            .getResult();
    auto block_z =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1))
            .getResult();

    OpBuilder::InsertPoint insertBeforeOp, insertAfterOp;
    gpu::LaunchOp gpu_launch_op;

    // if use async
    Value token = onDeviceOp.getAsyncToken();
    if (token) {
      auto asyncDependencies =
          operands.take_front(onDeviceOp.getODSOperandIndexAndLength(0).second -
                              onDeviceOp.getODSOperandIndexAndLength(0).first);
      auto async_exec_op = rewriter.create<async::ExecuteOp>(
          loc, TypeRange{}, asyncDependencies, ValueRange{});
      rewriter.replaceAllUsesWith(token, async_exec_op.getToken());
      auto &bodyBlock = async_exec_op.getBodyRegion().front();
      rewriter.setInsertionPointToStart(&bodyBlock);
      gpu_launch_op = rewriter.create<gpu::LaunchOp>(
          loc, grid_x, grid_y, grid_z, block_x, block_y, block_z);
      rewriter.setInsertionPoint(async_exec_op);
      insertBeforeOp = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointAfter(async_exec_op);
      insertAfterOp = rewriter.saveInsertionPoint();
    } else {
      gpu_launch_op = rewriter.create<gpu::LaunchOp>(
          loc, grid_x, grid_y, grid_z, block_x, block_y, block_z);
      rewriter.setInsertionPoint(gpu_launch_op);
      insertBeforeOp = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointAfter(gpu_launch_op);
      insertAfterOp = rewriter.saveInsertionPoint();
    }

    auto &bodyBlock = gpu_launch_op.getBody().front();

    rewriter.setInsertionPointToStart(&bodyBlock);
    auto insertToStart = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToEnd(&bodyBlock);
    auto insertToEnd = rewriter.saveInsertionPoint();

    IRMapping mp;
    auto innerOperands =
        operands.take_back(onDeviceOp.getODSOperandIndexAndLength(1).second -
                           onDeviceOp.getODSOperandIndexAndLength(1).first);
    auto insertPointInGpuLaunchBody = convertOperands(
        rewriter, innerOperands, mp, loc, insertBeforeOp, insertToStart);
    auto results = onDeviceOp.getInnerResults();
    auto result_memrefs = convertResults(rewriter, results, mp, loc,
                                         insertBeforeOp, insertAfterOp);

    assert(isa<StringAttr>(op->getAttr("sche.source")));
    auto sche_source =
        dyn_cast_or_null<StringAttr>(op->getAttr("sche.source")).strref();

    // scf::for lower
    if (sche_source == "scf.for") {
      Operation &op_ = onDeviceOp.getRegion().front().front();
      auto for_op = dyn_cast<scf::ForOp>(op_);
      lowerFromForOp(for_op, gpu_launch_op, insertBeforeOp,
                     insertPointInGpuLaunchBody, loc, rewriter, grid_x, grid_y,
                     grid_z, block_x, block_y, block_z);
    } else if (sche_source == "func") {
      rewriter.restoreInsertionPoint(insertPointInGpuLaunchBody);
      for (auto &&op_ : onDeviceOp.getRegion().front().getOperations()) {
        if (!op_.hasTrait<OpTrait::ReturnLike>()) {
          auto new_op = rewriter.clone(op_, mp);
          for (auto &&[a, b] :
               llvm::zip(op_.getResults(), new_op->getResults())) {
            mp.map(a, b);
          }
        } else {
          int i = 0;
          rewriter.restoreInsertionPoint(insertToEnd);
          for (auto res : op_.getOperands()) {
            auto t = res.getType();
            // TODO: must have the rank
            if (isa<TensorType>(t)) {
              auto shape = dyn_cast<TensorType>(t).getShape();
              auto ele_type = dyn_cast<TensorType>(t).getElementType();
              auto to_memref_op = rewriter.create<bufferization::ToMemrefOp>(
                  loc, MemRefType::get(shape, ele_type),
                  mp.lookupOrNull<Value>(res));
              rewriter.create<memref::CopyOp>(loc, to_memref_op.getResult(),
                                              result_memrefs[i++]);
            } else if (isa<VectorType>(t)) {
              auto idx0 =
                  rewriter
                      .create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0))
                      .getResult();
              llvm::SmallVector<Value> indices(
                  dyn_cast<VectorType>(t).getShape().size(), idx0);
              rewriter.create<vector::TransferWriteOp>(
                  loc, mp.lookupOrNull<Value>(res), result_memrefs[i++],
                  indices);
            } else if (isa<MemRefType>(t)) {
              rewriter.create<memref::CopyOp>(loc, mp.lookupOrNull<Value>(res),
                                              result_memrefs[i++]);
            } else {
              rewriter.create<memref::StoreOp>(loc, mp.lookupOrNull<Value>(res),
                                               result_memrefs[i++]);
            }
          }
        }
      }
    } else {
      // TODO add conversion of onDeviceOp from more op
      op->emitError("Conversion from source " + sche_source +
                    " has not been implemented");
    }

    rewriter.setInsertionPointToEnd(&bodyBlock);
    rewriter.create<gpu::TerminatorOp>(loc);

    rewriter.eraseOp(op);

    return success();
  }
};

} // end anonymous namespace

void populateLowerScheConversionPatterns(TypeConverter &typeConverter,
                                         RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<OnDeviceOpScheLowering>(typeConverter, patterns.getContext());
  patterns.add<WaitOpScheLowering>(typeConverter, patterns.getContext());
  // clang-format on
}

//===----------------------------------------------------------------------===//
// LowerSchePass
//===----------------------------------------------------------------------===//

namespace {
class LowerSchePass
    : public PassWrapper<LowerSchePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerSchePass)
  LowerSchePass() = default;
  LowerSchePass(const LowerSchePass &) {}

  StringRef getArgument() const final { return "lower-sche"; }
  StringRef getDescription() const final { return "lower sche dialect."; }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<
        buddy::bud::BudDialect,
        func::FuncDialect,
        vector::VectorDialect,
        memref::MemRefDialect,
        LLVM::LLVMDialect,
        buddy::sche::ScheDialect,
        gpu::GPUDialect,
        bufferization::BufferizationDialect,
        scf::SCFDialect,
        async::AsyncDialect>();
    // clang-format on
  }
};
} // end anonymous namespace.

void LowerSchePass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  // clang-format off
  target.addLegalDialect<
      arith::ArithDialect,
      func::FuncDialect,
      vector::VectorDialect,
      memref::MemRefDialect,
      LLVM::LLVMDialect,
      gpu::GPUDialect,
      bufferization::BufferizationDialect,
      scf::SCFDialect,
      async::AsyncDialect,
      BuiltinDialect>();
  // clang-format on
  target.addLegalOp<ModuleOp, func::ReturnOp>();

  target.addIllegalDialect<buddy::sche::ScheDialect>();
  TypeConverter typeConverter;
  typeConverter.addConversion([&](sche::AsyncTokenType type) {
    return async::TokenType::get(context);
  });
  typeConverter.addConversion([&](Type type) { return type; });

  RewritePatternSet patterns(context);
  populateLowerScheConversionPatterns(typeConverter, patterns);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerLowerSchePass() { PassRegistration<LowerSchePass>(); }
} // namespace buddy
} // namespace mlir
