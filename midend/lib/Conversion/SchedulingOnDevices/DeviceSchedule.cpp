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
#include "mlir/Pass/Pass.h"

#include "Bud/BudDialect.h"
#include "Bud/BudOps.h"
#include "Sche/ScheDialect.h"
#include "Sche/ScheOps.h"

#include <unordered_map>
#include <vector>
#include <algorithm>

using namespace mlir;
using namespace buddy;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

typedef struct{
  StringRef targetId;
  StringRef targetConfig;
} targetInfo;

class ScheTargetNode{
  public:
    void update(){
      updateUsedValue();
      updateDefinedValue();
      updateOperands();
      updateReturnValue();
    }

    ValueRange getReturnValues(){
      return return_;
    }

    ValueRange getOperands(){
      return  operands_;
    }

    StringRef getTargetId(){
      return target_id_;
    }

    StringRef getTargetConfig(){
      return target_config_;
    }

    std::vector<Operation*> getOpList(){
      return op_list_;
    }

    void setOpList(std::vector<Operation*> op_list){
      op_list_ = op_list;
    }

    void setTargetInfo(StringRef target_id, StringRef target_config){
      target_id_ = target_id;
      target_config_ = target_config;
    }


  private:
    std::vector<Operation*> op_list_; 
    std::vector<Value> used_;
    std::vector<Value> defined_;
    std::vector<Value> operands_;
    std::vector<Value> return_;

    StringRef target_id_;
    StringRef target_config_;


    void updateUsedValue(){
      for(auto&& op : op_list_){
        for(auto&& operand : op->getOperands()){
          used_.push_back(operand);
        }
      }
    }
    void updateDefinedValue(){
      for(auto&& op : op_list_){
        for(auto&& result : op->getResults()){
          defined_.push_back(result);
        }
      }
    }

    void updateOperands(){
      for(auto&& v : used_){
        if(std::find(defined_.begin(), defined_.end(), v) == defined_.end()){
          operands_.push_back(v);
        }
      }
    }

    void updateReturnValue(){
      for(auto&& v : defined_){
        for(auto&& user : v.getUsers()){
          if (user->hasTrait<mlir::OpTrait::ReturnLike>()) {
            return_.push_back(v);
            break;
          }
          if(std::find(op_list_.begin(), op_list_.end(), user) == op_list_.end()){
            return_.push_back(v);
            break;
          }
        }
      }
    }
};

std::vector<ScheTargetNode> splitDevice(Region& region){
  auto op_iter = region.getOps().begin();

  ScheTargetNode node_cpu, node_gpu;
  std::vector<Operation*> op_list_cpu, op_list_gpu;

  op_list_cpu.push_back(&*op_iter++);
  op_list_cpu.push_back(&*op_iter++);
  op_list_gpu.push_back(&*op_iter);

  node_cpu.setOpList(op_list_cpu);
  node_cpu.setTargetInfo("cpu", "");
  node_cpu.update();
  node_gpu.setOpList(op_list_gpu);
  node_gpu.setTargetInfo("gpu", "");
  node_gpu.update();

  std::vector<ScheTargetNode> result;
  result.push_back(node_cpu);
  result.push_back(node_gpu);
  return result;
}

class DeviceSchedulePattern : public OpRewritePattern<func::FuncOp>  {
public:
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(func::FuncOp func, PatternRewriter &rewriter) const override {
    printf("begin\n");

    auto loc = func.getLoc();
    auto ctx = rewriter.getContext();

    Region& region = func.getBody();

    assert(region.hasOneBlock());

    auto sche_target_node_list = splitDevice(region);

    rewriter.updateRootInPlace(func, [&](){
      IRMapping mp;
      rewriter.setInsertionPointToStart(&region.front());
      for (auto&& node : sche_target_node_list) {
        llvm::SmallVector<Value> operands;
        for(auto v : node.getOperands()){
          auto new_v = mp.lookupOrNull<Value>(v);
          new_v = new_v ? new_v : v;
          operands.push_back(new_v);
        }
        auto on_device_op = rewriter.create<sche::OnDeviceOp>(loc, node.getTargetId(), node.getTargetConfig(), 
                                    node.getReturnValues().getTypes(), operands, [&](OpBuilder& builder, Location loc, ValueRange valueRange){
          IRMapping mp_2;
          for(auto&& [a, b] : llvm::zip(node.getOperands(), valueRange)){
            mp_2.map(a, b);
          }
          for(auto&& op : node.getOpList()){
            auto new_op = builder.insert(op->clone(mp_2));
            for(auto&& [a, b] : llvm::zip(op->getResults(), new_op->getResults())){
              mp_2.map(a, b);
            }
            builder.setInsertionPointAfter(new_op);
          }
          std::vector<Value> return_values;

          for(auto&& v : node.getReturnValues()){
            auto new_v = mp_2.lookupOrNull<Value>(v);
            assert(new_v != nullptr);
            return_values.push_back(new_v);
          }
          
          builder.create<sche::ReturnOp>(loc, return_values);
        });

        for(auto&& [a, b] : llvm::zip(node.getReturnValues(), on_device_op.getResults())){
          a.replaceAllUsesWith(b);
          mp.map(a, b);
        }

        //TODO:更改删除方法，将原op记录下来，然后删除
        for(auto op : node.getOpList()){
          rewriter.eraseOp(op);
        }
      }

      func.getOperation()->setAttr("sche.dispatched", rewriter.getUnitAttr());
    });

    func.print(llvm::outs());

    printf("finish\n");
    return success();

  }
};

} // end anonymous namespace

void populateDeviceScheduleConversionPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<DeviceSchedulePattern>(patterns.getContext());
  // clang-format on
}

//===----------------------------------------------------------------------===//
// DeviceSchedulePass
//===----------------------------------------------------------------------===//

namespace {
class DeviceSchedulePass : public PassWrapper<DeviceSchedulePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DeviceSchedulePass)
  DeviceSchedulePass() = default;
  DeviceSchedulePass(const DeviceSchedulePass &) {}

  StringRef getArgument() const final { return "device-schedule"; }
  StringRef getDescription() const final { return "schedule on devices."; }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<
        buddy::bud::BudDialect,
        func::FuncDialect,
        vector::VectorDialect,
        memref::MemRefDialect,
        LLVM::LLVMDialect,
        buddy::sche::ScheDialect>();
    // clang-format on
  }
};
} // end anonymous namespace.

void DeviceSchedulePass::runOnOperation() {
  MLIRContext *context = &getContext();
  // ModuleOp module = getOperation();

  ConversionTarget target(*context);
  // clang-format off
  target.addLegalDialect<
      arith::ArithDialect,
      func::FuncDialect,
      vector::VectorDialect,
      memref::MemRefDialect,
      LLVM::LLVMDialect,
      buddy::sche::ScheDialect>();
  // clang-format on
  target.addLegalOp<ModuleOp, func::ReturnOp>();

  RewritePatternSet patterns(context);
  populateDeviceScheduleConversionPatterns(patterns);

  

  auto moduleOp = getOperation();

  target.addDynamicallyLegalOp<func::FuncOp>([](Operation *op) {
    return op->hasAttr("sche.dispatched");
  });
  // matchAndRewrite(getOperation(), builder);

  if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerDeviceSchedulePass() { PassRegistration<DeviceSchedulePass>(); }
} // namespace buddy
} // namespace mlir
