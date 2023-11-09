//====- DeviceSchedule.cpp - Other dialect lowering to sche dialect Pass  ---------------------===//
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
// This file defines upper level operators scheduling on different hardwares using sche dialect pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include  "mlir/Dialect/Tosa/IR/TosaOps.h"

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

    SmallVector<Operation*> getOpList(){
      return op_list_;
    }

    void setOpList(SmallVector<Operation*> op_list){
      op_list_ = op_list;
    }

    void setTargetInfo(StringRef target_id, StringRef target_config){
      target_id_ = target_id;
      target_config_ = target_config;
    }


  private:
    SmallVector<Operation*> op_list_; 
    SmallVector<Value> used_;
    SmallVector<Value> defined_;
    SmallVector<Value> operands_;
    SmallVector<Value> return_;

    StringRef target_id_;
    StringRef target_config_;


    void updateUsedValue(){
      for(auto&& op : op_list_){
        for(auto&& operand : op->getOperands()){
          if(std::find(used_.begin(), used_.end(), operand) == used_.end())
            used_.push_back(operand);
        }
      }
    }
    void updateDefinedValue(){
      for(auto&& op : op_list_){
        for(auto&& result : op->getResults()){
          if(std::find(defined_.begin(), defined_.end(), result) == defined_.end())
            defined_.push_back(result);
        }
      }
    }

    void updateOperands(){
      for(auto&& v : used_){
        if(std::find(defined_.begin(), defined_.end(), v) == defined_.end()){
          if(std::find(operands_.begin(), operands_.end(), v) == operands_.end())
            operands_.push_back(v);
        }
      }
    }

    void updateReturnValue(){
      for(auto&& v : defined_){
        for(auto&& user : v.getUsers()){
          if (user->hasTrait<mlir::OpTrait::ReturnLike>()) {
            if(std::find(return_.begin(), return_.end(), v) == return_.end())
              return_.push_back(v);
            break;
          }
          if(std::find(op_list_.begin(), op_list_.end(), user) == op_list_.end()){
            if(std::find(return_.begin(), return_.end(), v) == return_.end())
              return_.push_back(v);
            break;
          }
        }
      }
    }
};

SmallVector<ScheTargetNode*> splitDevice(Region& region){
  auto op_iter = region.getOps().begin();

  ScheTargetNode* node_cpu =new ScheTargetNode();
  ScheTargetNode* node_gpu =new ScheTargetNode();
  SmallVector<Operation*> op_list_cpu, op_list_gpu;

  op_list_cpu.push_back(&*op_iter++);
  op_list_cpu.push_back(&*op_iter++);
  op_list_gpu.push_back(&*op_iter);

  node_cpu->setOpList(op_list_cpu);
  node_cpu->setTargetInfo("gpu1", "");
  node_cpu->update();
  node_gpu->setOpList(op_list_gpu);
  node_gpu->setTargetInfo("gpu2", "");
  node_gpu->update();

  SmallVector<ScheTargetNode*> result;
  result.push_back(node_cpu);
  result.push_back(node_gpu);
  return result;
}

class FuncOpDeviceSchedulePattern : public OpRewritePattern<func::FuncOp>  {
public:
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(func::FuncOp func, PatternRewriter &rewriter) const override {

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
        for(auto v : node->getOperands()){
          auto new_v = mp.lookupOrNull<Value>(v);
          new_v = new_v ? new_v : v;
          operands.push_back(new_v);
        }
        auto on_device_op = rewriter.create<sche::OnDeviceOp>(loc, node->getTargetId(), node->getTargetConfig(), 
                                    node->getReturnValues().getTypes(), operands, [&](OpBuilder& builder, Location loc, ValueRange valueRange){
          IRMapping mp_2;
          for(auto&& [a, b] : llvm::zip(node->getOperands(), valueRange)){
            mp_2.map(a, b);
          }
          for(auto&& op : node->getOpList()){
            auto new_op = builder.insert(op->clone(mp_2));
            for(auto&& [a, b] : llvm::zip(op->getResults(), new_op->getResults())){
              mp_2.map(a, b);
            }
            builder.setInsertionPointAfter(new_op);
          }
          SmallVector<Value> return_values;

          for(auto&& v : node->getReturnValues()){
            auto new_v = mp_2.lookupOrNull<Value>(v);
            assert(new_v != nullptr);
            return_values.push_back(new_v);
          }
          builder.create<sche::ReturnOp>(loc, return_values);
        });
        on_device_op.getOperation()->setAttr("sche.source", rewriter.getStringAttr("func"));

        for(auto&& [a, b] : llvm::zip(node->getReturnValues(), on_device_op.getResults())){
          a.replaceAllUsesWith(b);
          mp.map(a, b);
        }

        for(auto op : node->getOpList()){
          rewriter.eraseOp(op);
        }
        delete(node);
      }
      func.getOperation()->removeAttr("sche.devices");
    });
    return success();

  }
};

class ForOpDeviceSchedulePattern : public OpRewritePattern<scf::ForOp>  {
public:
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(scf::ForOp forOp, PatternRewriter &rewriter) const override {
    auto loc = forOp.getLoc();
    auto ctx = rewriter.getContext();
    auto op = forOp.getOperation();
    assert(op->getNumResults() == 0);
    assert(isa<ArrayAttr>(op->getAttr("sche.devices")));
    auto devices = dyn_cast_or_null<ArrayAttr>(op->getAttr("sche.devices")).getValue();
    Value upperBound = forOp.getUpperBound();
    Value lowerBound = forOp.getLowerBound();
    Value step = forOp.getStep();
    rewriter.setInsertionPoint(op);
    auto placeHolder = rewriter.create<UnrealizedConversionCastOp>(loc, TypeRange{}, ValueRange{});
    rewriter.setInsertionPoint(placeHolder);
    auto range = rewriter.create<arith::SubIOp>(loc, upperBound, lowerBound);
    Value stepRange = rewriter.create<arith::DivSIOp>(loc, range.getResult(), step);
    stepRange = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), stepRange);
    stepRange = rewriter.create<arith::SIToFPOp>(loc, rewriter.getF32Type(), stepRange);
    auto start = lowerBound;
    auto end = lowerBound;

    //Build for loops on different hardware, requiring the last device to have the highest load
    for(auto i = 0; i < devices.size(); i++){
      auto device_info = devices[i];
      auto dict_attr = dyn_cast_or_null<DictionaryAttr>(device_info);
      assert(dict_attr != nullptr);
      auto targetId = dict_attr.get("targetId");
      auto targetConfig = dict_attr.get("targetConfig");
      assert(isa<StringAttr>(targetId) && isa<StringAttr>(targetConfig) && isa<FloatAttr>(dict_attr.get("duty_ratio")));

      rewriter.setInsertionPoint(placeHolder);
      //The upperBound of the last for loop is the original upperBound, and the previous for loop is rounded towards zero
      if(i == devices.size() - 1){
        end = upperBound;
      }
      else{     
        auto duty_ratio = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32Type(), dict_attr.get("duty_ratio").dyn_cast<FloatAttr>());
        Value duty_value = rewriter.create<arith::MulFOp>(loc, stepRange, duty_ratio.getResult());
        duty_value = rewriter.create<arith::FPToSIOp>(loc, rewriter.getI32Type(), duty_value);
        duty_value = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), duty_value);
        duty_value = rewriter.create<arith::MulIOp>(loc, duty_value, step);
        end = rewriter.create<arith::AddIOp>(loc, start, duty_value);
      }

      if(targetId.dyn_cast<StringAttr>().getValue() == "cpu"){
        rewriter.setInsertionPointAfter(placeHolder);
        auto sub_forOp = rewriter.create<scf::ForOp>(loc, start, end, step, forOp.getInitArgs(), [&](OpBuilder& builder, Location loc, Value iv, ValueRange iterArgs){
          Block &bodyBlock = forOp.getRegion().front();
          IRMapping mp;
          mp.map(bodyBlock.getArgument(0), iv);
          for(auto&& [a, b] : llvm::zip(bodyBlock.getArguments().drop_front(), iterArgs)){
            mp.map(a, b);
          }
          for(auto&& op_ : bodyBlock.getOperations()){
            builder.insert(op_.clone(mp));
          }
        });
      }
      else if(targetId.dyn_cast<StringAttr>().getValue() == "gpu"){
        rewriter.setInsertionPoint(placeHolder);
        Block &bodyBlock = forOp.getRegion().front();
        SmallVector<Operation*> op_list;
        ScheTargetNode node;
        for(auto it = bodyBlock.begin(); it != bodyBlock.end(); it++){
          op_list.push_back(&*it);
        }
        node.setOpList(op_list);
        node.update();
        SmallVector<Value> operands_;
        for(Value v : node.getOperands()){
          bool contain = 0;
          for(Value v_ : bodyBlock.getArguments()){
            if(v==v_){
              contain = 1;
              break;
            }
            if(!contain) operands_.push_back(v);
          }
        }
        
        auto on_device_op = rewriter.create<sche::OnDeviceOp>(loc, targetId.dyn_cast<StringAttr>().getValue(), targetConfig.dyn_cast<StringAttr>().getValue(), 
                                    forOp.getResults().getTypes(), operands_, [&](OpBuilder& builder, Location loc, ValueRange valueRange){
                                      auto sub_forOp = builder.create<scf::ForOp>(loc, start, end, step, forOp.getInitArgs(), [&](OpBuilder& builder, Location loc, Value iv, ValueRange iterArgs){
                                        IRMapping mp;
                                        mp.map(bodyBlock.getArgument(0), iv);
                                        for(auto&& [a, b] : llvm::zip(bodyBlock.getArguments().drop_front(), iterArgs)){
                                          mp.map(a, b);
                                        }
                                        for(auto&& op_ : bodyBlock.getOperations()){
                                          builder.insert(op_.clone(mp));
                                        }
                                      });
                                      builder.create<sche::ReturnOp>(loc, sub_forOp.getResults());
                                    }, sche::AsyncTokenType::get(ctx));
        on_device_op.getOperation()->setAttr("sche.source", rewriter.getStringAttr("scf.for"));

        rewriter.setInsertionPointAfter(op);
        rewriter.create<sche::WaitOp>(loc, (Type)nullptr, ValueRange{on_device_op.getAsyncToken()});
      }

      start = end;
    }

    rewriter.eraseOp(op);
    rewriter.eraseOp(placeHolder);
    return success();

  }
};

class ReduceSumOpDeviceSchedulePattern : public OpRewritePattern<tosa::ReduceSumOp>  {
public:
  using OpRewritePattern<tosa::ReduceSumOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(tosa::ReduceSumOp reduceSumOp, PatternRewriter &rewriter) const override {

    auto loc = reduceSumOp.getLoc();
    auto ctx = rewriter.getContext();
    auto op = reduceSumOp.getOperation();
    
    assert(isa<ArrayAttr>(op->getAttr("sche.devices")));
    auto devices = dyn_cast_or_null<ArrayAttr>(op->getAttr("sche.devices")).getValue();
    
    auto reduce_axis = reduceSumOp.getAxis();
    auto input = reduceSumOp.getInput();
    Type input_type = input.getType();
    assert(isa<TensorType>(input_type));
    TensorType input_tensor_type = input_type.dyn_cast<TensorType>();
    assert(input_tensor_type.hasRank());
    auto input_shape = input_tensor_type.getShape();
    auto rank = input_shape.size();
    auto split_axis = dyn_cast_or_null<IntegerAttr>(op->getAttr("sche.axis")).getInt();

    //use for sliceOp
    SmallVector<int64_t> size_shape;
    SmallVector<int64_t> start_shape;
    SmallVector<int64_t> result_shape;
    for(auto i=0; i<rank; i++){
      size_shape.push_back(input_shape[i]);
      result_shape.push_back(input_shape[i]);
      start_shape.push_back(0);
    }
    auto start = 0;
    auto end = 0;
    auto range = input_shape[split_axis];
    SmallVector<Value> inter_results; 

    rewriter.setInsertionPoint(op);
    
    for(auto i = 0; i < devices.size(); i++){
      auto device_info = devices[i];
      auto dict_attr = dyn_cast_or_null<DictionaryAttr>(device_info);
      assert(dict_attr != nullptr);
      auto targetId = dict_attr.get("targetId");
      auto targetConfig = dict_attr.get("targetConfig");
      assert(isa<StringAttr>(targetId) && isa<StringAttr>(targetConfig) && isa<FloatAttr>(dict_attr.get("duty_ratio")));
  
      if(i == devices.size() - 1){
        end = input_shape[split_axis];
      }
      else{     
        auto duty_ratio = dict_attr.get("duty_ratio").dyn_cast<FloatAttr>().getValueAsDouble();
        end = start + (int)(duty_ratio * range);
      }
      if(end == start) continue;
      size_shape[split_axis] = end - start;
      start_shape[split_axis] = start;
      result_shape[split_axis] = end - start;
      result_shape[reduce_axis] = 1;
      if(targetId.dyn_cast<StringAttr>().getValue() == "gpu"){
        rewriter.setInsertionPoint(op);
      }else{
        rewriter.setInsertionPointAfter(op);
      }
      Value result = rewriter.create<tosa::SliceOp>(loc, RankedTensorType::get(size_shape, input_tensor_type.getElementType()), input, start_shape, size_shape);
      if(targetId.dyn_cast<StringAttr>().getValue() == "cpu"){
        result = rewriter.create<tosa::ReduceSumOp>(loc, RankedTensorType::get(result_shape, input_tensor_type.getElementType()), result, reduce_axis);
        inter_results.push_back(result);
      }
      else if(targetId.dyn_cast<StringAttr>().getValue() == "gpu"){
        auto on_device_op = rewriter.create<sche::OnDeviceOp>
            (loc, targetId.dyn_cast<StringAttr>().getValue(), 
              targetConfig.dyn_cast<StringAttr>().getValue(), 
              TypeRange{RankedTensorType::get(result_shape, input_tensor_type.getElementType())}, 
              ValueRange{}, 
              [&](OpBuilder& builder, Location loc, ValueRange valueRange){
                result = rewriter.create<tosa::ReduceSumOp>
                        (loc, RankedTensorType::get(result_shape, input_tensor_type.getElementType()), 
                        result, reduce_axis);
                builder.create<sche::ReturnOp>(loc, ValueRange{result});
        });
        inter_results.push_back(on_device_op.getResult(0));
          
      }

      start = end;
    }

    rewriter.setInsertionPoint(*(reduceSumOp.getOutput().user_begin()));
    Value res = inter_results[0];
    if(reduce_axis == split_axis){
      for(int i=1; i<inter_results.size(); i++){
        res = rewriter.create<tosa::AddOp>(loc, res.getType(), res, inter_results[i]);
      }
    }
    else{
      for(int i=1; i<inter_results.size(); i++){
        res = rewriter.create<tosa::ConcatOp>(loc, ValueRange{res, inter_results[i]}, split_axis);
      }
    }
    
    reduceSumOp.getOutput().replaceAllUsesWith(res);
    rewriter.eraseOp(op);
    
    return success();

  }
};

} // end anonymous namespace

void populateDeviceScheduleConversionPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<FuncOpDeviceSchedulePattern>(patterns.getContext());
  patterns.add<ForOpDeviceSchedulePattern>(patterns.getContext());
  patterns.add<ReduceSumOpDeviceSchedulePattern>(patterns.getContext());
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
        buddy::sche::ScheDialect,
        scf::SCFDialect,
        linalg::LinalgDialect>();
    // clang-format on
  }
};
} // end anonymous namespace.

void DeviceSchedulePass::runOnOperation() {
  MLIRContext *context = &getContext();

  ConversionTarget target(*context);
  // clang-format off
  target.addLegalDialect<
      arith::ArithDialect,
      func::FuncDialect,
      vector::VectorDialect,
      memref::MemRefDialect,
      LLVM::LLVMDialect,
      buddy::sche::ScheDialect,
      scf::SCFDialect,
      tosa::TosaDialect>();
  // clang-format on
  target.addLegalOp<ModuleOp, linalg::ReduceOp>();

  RewritePatternSet patterns(context);
  populateDeviceScheduleConversionPatterns(patterns);

  

  auto moduleOp = getOperation();

  target.addDynamicallyLegalOp<func::FuncOp>([](Operation *op) {
    return !op->hasAttr("sche.devices");
  });
  target.addDynamicallyLegalOp<scf::ForOp, tosa::ReduceSumOp>([&](Operation *op) {
    return !op->hasAttr("sche.devices");
  });

  if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerDeviceSchedulePass() { PassRegistration<DeviceSchedulePass>(); }
} // namespace buddy
} // namespace mlir
