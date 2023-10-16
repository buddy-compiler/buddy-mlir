//====- LowerBudPass.cpp - Bud Dialect Lowering Pass  ---------------------===//
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
// This file defines bud dialect lowering pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

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

class LaunchFuncOpScheLowering : public OpRewritePattern<sche::LaunchFuncOp>  {
public:
  using OpRewritePattern<sche::LaunchFuncOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(sche::LaunchFuncOp op, PatternRewriter &rewriter) const override {
    // printf("begin\n");
    auto loc = op.getLoc();

    // printf("finish\n");
    return success();

  }
};

//gpu 同步/异步，使用映射内存还是copy
class OnDeviceOpScheLowering : public OpRewritePattern<sche::OnDeviceOp>  {
public:
  using OpRewritePattern<sche::OnDeviceOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(sche::OnDeviceOp onDeviceOp, PatternRewriter &rewriter) const override {
    auto loc = onDeviceOp.getLoc();

    auto op = onDeviceOp.getOperation();
    rewriter.setInsertionPoint(op);

    IRMapping mp;

    auto idx0 = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0)).getResult();

    auto operands = op->getOperands();
    for(auto v : operands){
      // v.print(llvm::outs());
      auto t = v.getType();
      // t.print(llvm::outs());
      if(t.isa<TensorType>()){
        auto shape = t.dyn_cast<TensorType>().getShape();
        auto ele_type = t.dyn_cast<TensorType>().getElementType();
        auto to_memref_op = rewriter.create<bufferization::ToMemrefOp>(loc, MemRefType::get(shape, ele_type), v);
        mp.map(v, to_memref_op.getResult());
        // v.replaceAllUsesWith(to_memref_op.getResult());
        // to_memref_op.print(llvm::outs());

        auto memref_cast_op = rewriter.create<memref::CastOp>(loc, UnrankedMemRefType::get(ele_type, {}), to_memref_op.getResult());
        auto host_register_op = rewriter.create<gpu::HostRegisterOp>(loc, memref_cast_op.getResult());
      }
      else if(t.isa<VectorType>()){
        auto shape = t.dyn_cast<VectorType>().getShape();
        auto ele_type = t.dyn_cast<VectorType>().getElementType();
        auto mem_type = MemRefType::get(shape, ele_type);
        auto alloc_op = rewriter.create<memref::AllocOp>(loc, mem_type);
        auto memref_cast_op = rewriter.create<memref::CastOp>(loc, UnrankedMemRefType::get(ele_type, {}), alloc_op.getResult());
        auto host_register_op = rewriter.create<gpu::HostRegisterOp>(loc, memref_cast_op.getResult());
        llvm::SmallVector<Value> indices(shape.size(), idx0);
        auto vector_transfer_write_op = rewriter.create<vector::TransferWriteOp>(loc, v, alloc_op.getResult(), indices);
        mp.map(v, alloc_op.getResult());
      }
      else if(t.isa<UnrankedMemRefType>()){
        auto host_register_op = rewriter.create<gpu::HostRegisterOp>(loc, v);
      }
      else if(t.isa<MemRefType>()){
        auto memref_type = t.dyn_cast<MemRefType>();
        auto memref_cast_op = rewriter.create<memref::CastOp>(loc, UnrankedMemRefType::get(memref_type.getElementType(), memref_type.getMemorySpace()), v);
        auto host_register_op = rewriter.create<gpu::HostRegisterOp>(loc, memref_cast_op.getResult());
      }
    }

    SmallVector<Value> result_memrefs;
    auto results = op->getResults();
    for(auto v : results){
      MemRefType mem_type;
      auto t = v.getType();
      //TODO:必须要有rank
      if(t.isa<TensorType>()){
        auto shape = t.dyn_cast<TensorType>().getShape();
        auto ele_type = t.dyn_cast<TensorType>().getElementType();
        mem_type = MemRefType::get(shape, ele_type);
      }
      else if(t.isa<VectorType>()){
        auto shape = t.dyn_cast<VectorType>().getShape();
        auto ele_type = t.dyn_cast<VectorType>().getElementType();
        mem_type = MemRefType::get(shape, ele_type);
      }
      else if(t.isa<MemRefType>()){
        mem_type = t.dyn_cast<MemRefType>();
      }
      else{
        mem_type = MemRefType::get({1}, t);
      }
      auto alloc_op = rewriter.create<memref::AllocOp>(loc, mem_type);
      result_memrefs.push_back(alloc_op.getResult());
      auto memref_cast_op = rewriter.create<memref::CastOp>(loc, UnrankedMemRefType::get(mem_type.getElementType(), mem_type.getMemorySpace()), alloc_op.getResult());
      auto host_register_op = rewriter.create<gpu::HostRegisterOp>(loc, memref_cast_op.getResult());
    }

    auto grid_x = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(3)).getResult();
    auto grid_y = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1)).getResult();
    auto grid_z = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1)).getResult();

    auto block_x = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(3)).getResult();
    auto block_y = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1)).getResult();
    auto block_z = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1)).getResult();
    

    auto gpu_launch_op = rewriter.create<gpu::LaunchOp>(loc, grid_x, grid_y, grid_z, block_x, block_y, block_z, nullptr, gpu::AsyncTokenType::get(rewriter.getContext()));

    auto& body = gpu_launch_op.getBody();
    auto& bodyBlock = body.front();

    rewriter.setInsertionPointToStart(&bodyBlock);

    auto idx0_gpu = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0)).getResult();

    for(auto v : operands){
      auto t = v.getType();
      if(t.isa<TensorType>()){
        auto to_tensor_op = rewriter.create<bufferization::ToTensorOp>(loc, t, mp.lookup<Value>(v));
        mp.map(v, to_tensor_op.getResult());
      }
      else if(t.isa<VectorType>()){
        llvm::SmallVector<Value> indices(t.dyn_cast<VectorType>().getShape().size(), idx0_gpu);
        auto transfer_read_op = rewriter.create<vector::TransferReadOp>(loc, t.dyn_cast<VectorType>(), mp.lookup<Value>(v), indices);
        mp.map(v, transfer_read_op.getResult());
      }
    }
    //scf::for lower
    assert(op->getAttr("sche.source").isa<StringAttr>());
    auto sche_source = op->getAttr("sche.source").dyn_cast_or_null<StringAttr>().strref();
    if(sche_source == "scf.for"){
      Operation& op_ = onDeviceOp.getRegion().front().front();
      auto for_op = dyn_cast<scf::ForOp>(op_);
      // for_op.print(llvm::outs());
      Value upperBound = for_op.getUpperBound();
      Value lowerBound = for_op.getLowerBound();
      Value step = for_op.getStep();

      //计算需要在一个block中的步长范围
      auto range = rewriter.create<arith::SubIOp>(loc, upperBound, lowerBound); //ti
      Value stepRange = rewriter.create<arith::DivSIOp>(loc, range.getResult(), step);//ti
      // stepRange = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), stepRange);
      Value stepRangeInBlock = rewriter.create<arith::DivSIOp>(loc, stepRange, gpu_launch_op.getGridSizeX());//ti
      Value start = rewriter.create<arith::MulIOp>(loc, stepRangeInBlock, gpu_launch_op.getBlockIds().x);
      start = rewriter.create<arith::AddIOp>(loc, start, lowerBound);
      Value rem = rewriter.create<arith::RemSIOp>(loc, stepRange, gpu_launch_op.getGridSizeX());//ti
      Value cmp_rem_blkId = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, rem, gpu_launch_op.getBlockIds().x);
      Value cmp_rem_blkId_index = rewriter.create<arith::IndexCastUIOp>(loc, rewriter.getIndexType(), cmp_rem_blkId);
      auto idx1 = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1)).getResult();
      stepRangeInBlock = rewriter.create<arith::AddIOp>(loc, cmp_rem_blkId_index, stepRangeInBlock);
      Value min = rewriter.create<arith::MinUIOp>(loc, gpu_launch_op.getBlockIds().x, rem);
      start = rewriter.create<arith::AddIOp>(loc, start, min);
      //一个thread中的步长范围
      Value stepRangeInThread = rewriter.create<arith::DivSIOp>(loc, stepRangeInBlock, block_x);//ti
      rem = rewriter.create<arith::RemSIOp>(loc, stepRangeInBlock, block_x);//ti
      Value cmp_rem_threadId = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, rem, gpu_launch_op.getThreadIds().x);
      Value cmp_rem_threadId_index = rewriter.create<arith::IndexCastUIOp>(loc, rewriter.getIndexType(), cmp_rem_threadId);
      stepRangeInThread = rewriter.create<arith::AddIOp>(loc, cmp_rem_threadId_index, stepRangeInThread);

      Value end = rewriter.create<arith::AddIOp>(loc, start, stepRangeInThread);

      auto sub_forOp = rewriter.create<scf::ForOp>(loc, idx0, stepRangeInThread, idx1, for_op.getInitArgs(), 
                                                    [&](OpBuilder& builder, Location loc, 
                                                    Value iv, ValueRange iterArgs)
      {
        Block &bodyBlock = for_op.getLoopBody().front();//原始for的bodyBlock
        IRMapping mp;
        iv = builder.create<arith::MulIOp>(loc, iv, gpu_launch_op.getBlockSizeX());
        iv = builder.create<arith::AddIOp>(loc, iv, gpu_launch_op.getThreadIds().x);
        iv = builder.create<arith::MulIOp>(loc, iv, step);
        iv = builder.create<arith::AddIOp>(loc, iv, start);
        mp.map(bodyBlock.getArgument(0), iv);
        for(auto&& [a, b] : llvm::zip(bodyBlock.getArguments().drop_front(), iterArgs)){
          mp.map(a, b);
        }
        for(auto&& op_ : bodyBlock.getOperations()){
          builder.insert(op_.clone(mp));
        }
      });

      rewriter.create<gpu::TerminatorOp>(loc);
    }
    else{
      for(auto&& op_ : onDeviceOp.getRegion().front().getOperations()){
      // op_.print(llvm::outs());
      if(!op_.hasTrait<OpTrait::ReturnLike>()){
        auto new_op = rewriter.clone(op_, mp);
        for(auto&& [a, b] : llvm::zip(op_.getResults(), new_op->getResults())){
          mp.map(a, b);
        }
      }else{
        int i=0;
        for(auto res : op_.getOperands()){
          auto t = res.getType();
          // t.print(llvm::outs());
          //TODO:必须要有rank
          if(t.isa<TensorType>()){
            auto shape = t.dyn_cast<TensorType>().getShape();
            auto ele_type = t.dyn_cast<TensorType>().getElementType();
            auto to_memref_op = rewriter.create<bufferization::ToMemrefOp>(loc, MemRefType::get(shape, ele_type), mp.lookupOrNull<Value>(res));
            auto copy_op = rewriter.create<memref::CopyOp>(loc, to_memref_op.getResult(), result_memrefs[i++]);
          }
          else if(t.isa<VectorType>()){
            llvm::SmallVector<Value> indices(t.dyn_cast<VectorType>().getShape().size(), idx0_gpu);
            auto vector_transfer_write_op = rewriter.create<vector::TransferWriteOp>(loc, mp.lookupOrNull<Value>(res), result_memrefs[i++], indices);
          }
          else if(t.isa<MemRefType>()){
            auto copy_op = rewriter.create<memref::CopyOp>(loc, mp.lookupOrNull<Value>(res), result_memrefs[i++]);
          }
          else{
            auto store_op = rewriter.create<memref::StoreOp>(loc, mp.lookupOrNull<Value>(res), result_memrefs[i++]);
          }
        }
      }
    }
    rewriter.create<gpu::TerminatorOp>(loc);
    }


    rewriter.setInsertionPointAfter(gpu_launch_op);
    int i = 0;
    for(auto v : results){
      auto t = v.getType();
      //TODO:必须要有rank
      if(t.isa<TensorType>()){
        auto shape = t.dyn_cast_or_null<TensorType>().getShape();
        auto ele_type = t.dyn_cast_or_null<TensorType>().getElementType();
        auto to_tensor_op = rewriter.create<bufferization::ToTensorOp>(loc, t, result_memrefs[i++]);
        v.replaceAllUsesWith(to_tensor_op.getResult());
      }
      else if(t.isa<VectorType>()){
        llvm::SmallVector<Value> indices(t.dyn_cast<VectorType>().getShape().size(), idx0);
        auto transfer_read_op = rewriter.create<vector::TransferReadOp>(loc, t.dyn_cast<VectorType>(), result_memrefs[i++], indices);
        v.replaceAllUsesWith(transfer_read_op.getResult());
      }
      else if(t.isa<MemRefType>()){
        v.replaceAllUsesWith(result_memrefs[i++]);
      }
      else{
        auto load_op = rewriter.create<memref::LoadOp>(loc, v.getType(), result_memrefs[i++], ValueRange{idx0});
        v.replaceAllUsesWith(load_op.getResult());
      }
    }

    rewriter.create<gpu::WaitOp>(loc, (Type)nullptr, ValueRange{gpu_launch_op.getAsyncToken()});
    rewriter.eraseOp(op);


    // op->getParentOp()->print(llvm::outs());


    return success();

  }
};

} // end anonymous namespace

void populateLowerScheConversionPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<OnDeviceOpScheLowering>(patterns.getContext());
  // clang-format on
}

//===----------------------------------------------------------------------===//
// LowerSchePass
//===----------------------------------------------------------------------===//

namespace {
class LowerSchePass : public PassWrapper<LowerSchePass, OperationPass<ModuleOp>> {
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
        bufferization::BufferizationDialect>();
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
      scf::SCFDialect>();
  // clang-format on
  target.addLegalOp<ModuleOp, func::ReturnOp>();

  // target.addIllegalDialect<buddy::sche::ScheDialect>();

  RewritePatternSet patterns(context);
  populateLowerScheConversionPatterns(patterns);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerLowerSchePass() { PassRegistration<LowerSchePass>(); }
} // namespace buddy
} // namespace mlir
