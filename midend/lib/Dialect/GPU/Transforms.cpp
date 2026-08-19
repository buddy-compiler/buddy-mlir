#include "GPU/Transforms.h"
#include "GPU/TransformOps.h"

#include "mlir/IR/PatternMatch.h"          
#include "mlir/IR/Operation.h"            
#include "mlir/IR/Builders.h"             
#include "mlir/IR/BuiltinOps.h"           
#include "mlir/IR/BuiltinTypes.h"         
#include "mlir/IR/Value.h"       
#include "mlir/Dialect/SCF/IR/SCF.h" 
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/MLIRContext.h"  
#include "llvm/Support/Debug.h"
using namespace mlir;


using llvm::dbgs;
//===---------------------------------------------------------------------===//
// EliminateBroadcastExtractPattern
// Eliminate the Broadcast and Extract in the begin and end of scf.for, for example:
//    ...
//    %1 = vector.broadcast %0 : vector<16xf32> to vector<1x16xf32>
//    %2 = scf.for %iv = %c0 to %c1024 step %16 iter_args(%arg = %1) -> (vector<1x16xf32>)
//         %3 = extract arg[0] :  vector<16xf32> from vector<1x16xf32>
//             ...
//         %5 = vector.broadcast %4 : vector<16xf32> to vector<1x16xf32>
//    scf.yield %5
//    %6 = extract %2[0] :  vector<16xf32> from vector<1x16xf32>
//    ...
// is translated to:
//    ...
//    %1 = scf.for %iv = %c0 to %c1024 step %16 iter_args(%arg = %0) -> (vector<16xf32>)
//        ...
//    scf.yield %5
//    ... 
//===---------------------------------------------------------------------===//
struct EliminateBroadcastExtractPattern : public mlir::RewritePattern {
  explicit EliminateBroadcastExtractPattern(mlir::MLIRContext *context)
      : RewritePattern(scf::ForOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(mlir::Operation *op, mlir::PatternRewriter &rewriter) const override {
    auto forOp = dyn_cast<scf::ForOp>(op);
    if (!forOp)
      return failure();
    if (!forOp.getBody()->mightHaveTerminator()) {
    return failure();
  }
    auto yieldOp = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());

    SmallVector<Value> newIterArgs;       
    SmallVector<Value> scalarValues;      
    SmallVector<Value> newYeildArgs;
    SmallVector<Value> deleteiter;      
    SmallVector<vector::ExtractOp> extractsToErase1;      
    bool change=false;
     for (auto tuple : llvm::zip(forOp.getInitArgs(),forOp.getRegionIterArgs(), forOp.getResults(), yieldOp.getOperands()))
    {
      auto initArg = std::get<0>(tuple);
      auto regionArg = std::get<1>(tuple);
      auto result = std::get<2>(tuple);
      auto yieldResult = std::get<3>(tuple);
      auto broadcastOp0 = initArg.getDefiningOp<vector::BroadcastOp>();
      auto broadcastOp1 = yieldResult.getDefiningOp<vector::BroadcastOp>();
      if (broadcastOp0&&broadcastOp1) {
        Value scalarValue0 = broadcastOp0.getSource();
        Value scalarValue1 = broadcastOp1.getSource();
        //Determine whether the subsequent operators are all extractops
        bool hasOnlyExtractUse0 = llvm::all_of(regionArg.getUses(), [&](OpOperand &use) {
          if (auto extractOp = dyn_cast<vector::ExtractOp>(use.getOwner())) {
            if(extractOp.getResult().getType()==scalarValue0.getType()) return true;
          }
          return false;
        });
        bool hasOnlyExtractUse1 = llvm::all_of(result.getUses(), [&](OpOperand &use) {
          if (auto extractOp = dyn_cast<vector::ExtractOp>(use.getOwner())) {
            if(extractOp.getResult().getType()==scalarValue1.getType()) return true;
          }
          return false;
        });
        //creat new initarg and delete extract2 and broadcast
        if ((!hasOnlyExtractUse0)||(!hasOnlyExtractUse1))
        {
          scalarValues.push_back(initArg);
          newYeildArgs.push_back(yieldResult);
        }
        else{
          change=true;
          for (auto &use : regionArg.getUses()) {
            auto extractOp = dyn_cast<vector::ExtractOp>(use.getOwner());
            extractsToErase1.push_back(extractOp);
          }
          for (auto &use : result.getUses()) {
            auto extractOp = dyn_cast<vector::ExtractOp>(use.getOwner());
            rewriter.replaceOp(extractOp, result);
          }
          deleteiter.push_back(regionArg);
          newIterArgs.push_back(scalarValue0);
          scalarValues.push_back(scalarValue0);
          newYeildArgs.push_back(scalarValue1);
          rewriter.replaceOp(broadcastOp0, scalarValue0);
          rewriter.replaceOp(broadcastOp1, scalarValue1);
        }
      } else {
        scalarValues.push_back(initArg);
        newYeildArgs.push_back(yieldResult);
      }
    }
    if (change==false)
      return failure();
    //creat new forop
    scf::ForOp newLoop = rewriter.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
      scalarValues);
    newLoop.getBody()->erase();
    newLoop.getRegion().getBlocks().splice(
                                    newLoop.getRegion().getBlocks().begin(), 
                                        forOp.getRegion().getBlocks());
    //creat new RegionIterArgs and delete extract1
    for (int i=0;i<scalarValues.size();i++)
      {
        auto arg=newLoop.getBody()->addArgument(scalarValues[i].getType(), scalarValues[i].getLoc());
        rewriter.replaceOp(extractsToErase1[i], arg);
      }
    for(int i=newLoop.getBody()->getNumArguments()-1;i>=0;i--)
    {
        if(newLoop.getBody()->getArgument(i).use_empty()) 
          {
            newLoop.getBody()->eraseArgument(i);
          }
    }
    rewriter.replaceOp(forOp, newLoop.getResults());
    return success();
  }
};


void mlir::buddy::EliminateBroadcastExtractPatterns(RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<EliminateBroadcastExtractPattern>(context);
}