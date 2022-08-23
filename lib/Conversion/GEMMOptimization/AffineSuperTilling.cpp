//===- AffineSuperTilling.cpp - Better Tilling, now for GEMM. ------===//
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
// This file implements Another Affine Tilling algorithm.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Value.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "llvm/Support/Endian.h"

#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <ostream>
#include <string>
#include <type_traits>

using namespace mlir;
using namespace vector;

//===----------------------------------------------------------------------===//
// AffineSuperTilingPass 
//===----------------------------------------------------------------------===//

namespace {
class AffineSuperTillingPass 
    : public PassWrapper<AffineSuperTillingPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AffineSuperTillingPass)
  StringRef getArgument() const final { return "affine-super-tilling"; }
  StringRef getDescription() const final {
    return "Another Affine tilling.";
  }
  AffineSuperTillingPass() = default;
  AffineSuperTillingPass(const AffineSuperTillingPass &) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, func::FuncDialect>();
  }

  ListOption<int64_t> tileStrategy{*this, "stratege",
                                   llvm::cl::desc("tile stratege, split by -1"),
                                   llvm::cl::ZeroOrMore};
};
} // end anonymous namespace.

/// Constructs and sets new loop bounds after tiling for the case of
/// hyper-rectangular index sets, where the bounds of one dimension do not
/// depend on other dimensions. Bounds of each dimension can thus be treated
/// independently, and deriving the new bounds is much simpler and faster
/// than for the case of tiling arbitrary polyhedral shapes.
static void
constructSuperTiledIndexSetHyperRect(MutableArrayRef<AffineForOp> origLoops,
                                MutableArrayRef<AffineForOp> newLoops,
                                SmallVector<SmallVector<unsigned>> tileSizes) {
  assert(!origLoops.empty());

  OpBuilder b(origLoops[0].getOperation());
  unsigned width = origLoops.size();

  // prepare for tile size.
  DenseMap<unsigned, unsigned> tileCount;
  for(unsigned i = 0; i < width; ++ i){
      unsigned c = 1;
      for(auto j : tileSizes[i]){
        c *= j;
      }
      tileCount[i] = c;
  }

  // Bounds for tile space loops.
  SmallVector<bool, 3> needRemainCalc;
  for (unsigned i = 0; i < width; i++) {
    auto origLoop = origLoops[i];

    // for now we think input val is null.
    newLoops[i].setLowerBound({}, b.getConstantAffineMap(0));
    // here we should get Ceil val
    if(origLoop.hasConstantUpperBound()){
        auto upBound = origLoop.getConstantUpperBound();
        needRemainCalc.push_back(upBound % tileCount[i]); // 这里我们要想办法将其构造为动态的
        newLoops[i].setUpperBound({}, b.getConstantAffineMap(llvm::divideCeil(upBound, tileCount[i])));
    } else {
        // Dyn shape
        auto iv = origLoop.getUpperBound().getOperand(0); // 这里我们要构造AffineMap来说明这件事
        auto loopUppderBoundExpr = b.getAffineDimExpr(0).ceilDiv(b.getAffineConstantExpr(tileCount[i]));
        auto ubMap = AffineMap::get(1, 0, loopUppderBoundExpr);
        newLoops[i].setUpperBound(iv, ubMap);
        needRemainCalc.push_back(true); // 这里我们要想办法将其构造为动态的
    }
    tileCount[i] /= tileSizes[i][0];
  }

  // Bounds for intra-tile loops.
  unsigned newLoopIdx = width;
  for (unsigned i = 0; i < width; i++) {
    auto origLoop = origLoops[i];
    for(unsigned j = 0; j < tileSizes[i].size(); j++, newLoopIdx++) {
        newLoops[newLoopIdx].setLowerBound(
                {}, b.getConstantAffineMap(0));
        // The step sizes of intra-tile loops is just the original loops' step size.
        // newLoops[newLoopIdx].setStep(origLoops[i].getStep());
        
        // we only left innermost 2 loop for 'pure loop', others with min or max.

        bool computeInKernel = tileSizes[i].size() - j < 2; 

        auto lastIV = newLoops[i].getInductionVar();

        if(j != 0) {
            lastIV = newLoops[newLoopIdx - 1].getInductionVar();
            // lastSeg = tileSizes[i][j - 1];
        }

        if(computeInKernel){
            AffineMap ubMap;
            unsigned dimNum = 0;
            SmallVector<AffineExpr, 2> mapExprs;
            mapExprs.push_back(b.getAffineConstantExpr(tileSizes[i][j]));
            SmallVector<Value> inputs;
            if(needRemainCalc[i] && tileSizes[i][j] != 1) {
               auto cTileSize = b.getAffineConstantExpr(tileSizes[i][j]);
               auto d0 = b.getAffineDimExpr(0);
               dimNum ++;
               inputs.push_back(lastIV);
               // 这里我们要转换一下
               AffineExpr remainExpr;
               if(origLoop.hasConstantUpperBound()){
                   remainExpr = origLoop.getConstantUpperBound() - cTileSize * d0;
               } else {
                   inputs.push_back(origLoop.getUpperBound().getOperand(0));
                   remainExpr = b.getAffineDimExpr(inputs.size() - 1) - cTileSize * d0;
                   dimNum ++;
               }
               mapExprs.push_back(remainExpr);
               needRemainCalc[i] = false;
            }
            ubMap = AffineMap::get(dimNum, 0, mapExprs, b.getContext());    
            newLoops[newLoopIdx].setUpperBound(inputs, ubMap);
        } else {
            SmallVector<Value> inputs;
            inputs.push_back(lastIV);

            auto segConstExpr = b.getAffineConstantExpr(tileSizes[i][j]);
            auto lbResult = b.getAffineDimExpr(0) * segConstExpr;
            AffineMap lbMap = AffineMap::get(1, 0, lbResult, b.getContext());

            // now for ubMap
            AffineMap ubMap;
            if(needRemainCalc[i]){
                if(origLoop.hasConstantUpperBound()){
                    ubMap = AffineMap::get(inputs.size(), 0, {lbResult + segConstExpr, b.getAffineConstantExpr(origLoop.getConstantUpperBound() - tileSizes[i][j + 1])}, b.getContext());
                } else {
                    inputs.push_back(origLoop.getUpperBound().getOperand(0));
                    ubMap = AffineMap::get(inputs.size(), 0, {lbResult + segConstExpr, b.getAffineDimExpr(inputs.size() - 1).floorDiv(tileSizes[i][j + 1])}, b.getContext());
                }
               needRemainCalc[i] = false;
            } else {
                ubMap = AffineMap::get(inputs.size(), 0, lbResult + segConstExpr, b.getContext());
            }
            
            newLoops[newLoopIdx].setLowerBound(lastIV, lbMap);
            newLoops[newLoopIdx].setUpperBound(inputs, ubMap);
        }
    }
  }
}

/// Move the loop body of AffineForOp 'src' from 'src' into the specified
/// location in destination's body, ignoring the terminator.
static void _moveLoopBodyImpl(AffineForOp src, AffineForOp dest,
                             Block::iterator loc) {
  auto &ops = src.getBody()->getOperations();
  dest.getBody()->getOperations().splice(loc, ops, ops.begin(),
                                         std::prev(ops.end()));
}

/// Move the loop body of AffineForOp 'src' from 'src' to the start of dest
/// body.
void _moveLoopBody(AffineForOp src, AffineForOp dest) {
  _moveLoopBodyImpl(src, dest, dest.getBody()->begin());
}

/// Constructs tiled loop nest, without setting the loop bounds and move the
/// body of the original loop nest to the tiled loop nest.
void _constructTiledLoopNest(MutableArrayRef<AffineForOp> origLoops,
                            AffineForOp rootAffineForOp, unsigned cnt,
                            MutableArrayRef<AffineForOp> tiledLoops) {
  Location loc = rootAffineForOp.getLoc();

  // The outermost among the loops as we add more..
  Operation *topLoop = rootAffineForOp.getOperation();
  AffineForOp innermostPointLoop;

  // Add intra-tile (or point) loops.
  for (unsigned i = 0; i < cnt - origLoops.size(); i++) {
    OpBuilder b(topLoop);
    // Loop bounds will be set later.
    AffineForOp pointLoop = b.create<AffineForOp>(loc, 0, 0);
    pointLoop.getBody()->getOperations().splice(
        pointLoop.getBody()->begin(), topLoop->getBlock()->getOperations(),
        topLoop);
    tiledLoops[cnt - i - 1] = pointLoop;
    topLoop = pointLoop.getOperation();
    if (i == 0)
      innermostPointLoop = pointLoop;
  }

  // Add tile space loops;
  for (unsigned i = 0; i < origLoops.size(); i++) {
    OpBuilder b(topLoop);
    // Loop bounds will be set later.
    AffineForOp tileSpaceLoop = b.create<AffineForOp>(loc, 0, 0);
    tileSpaceLoop.getBody()->getOperations().splice(
        tileSpaceLoop.getBody()->begin(), topLoop->getBlock()->getOperations(),
        topLoop);
    tiledLoops[origLoops.size() - i - 1] = tileSpaceLoop;
    topLoop = tileSpaceLoop.getOperation();
  }

  // Move the loop body of the original nest to the new one.
  _moveLoopBody(origLoops.back(), innermostPointLoop);
}

LogicalResult
superTilePerfectlyNested(MutableArrayRef<AffineForOp> input,
                          SmallVector<SmallVector<unsigned>> tileSizes,
                          SmallVectorImpl<AffineForOp> *tiledNest) {
    if (input.empty())
      return success();

    MutableArrayRef<AffineForOp> origLoops = input;
    AffineForOp rootAffineForOp = origLoops[0];

    // Note that width is at least one since the band isn't empty.
    unsigned width = input.size();
    unsigned cnt = 0;
    for(auto v : tileSizes){
        cnt += v.size() + 1;
    }
    SmallVector<AffineForOp, 6> tiledLoops(cnt);

    // Construct a tiled loop nest without setting their bounds. Bounds are
    // set later.
    _constructTiledLoopNest(origLoops, rootAffineForOp, cnt, tiledLoops);

    SmallVector<Value, 8> origLoopIVs;
    extractForInductionVars(input, &origLoopIVs);

    // Set loop bounds for the tiled loop nest.
    constructSuperTiledIndexSetHyperRect(origLoops, tiledLoops, tileSizes);

    // Replace original IVs with intra-tile loop IVs.
    OpBuilder b(tiledLoops.back().getRegion());
    unsigned newLoopIdx = width;
    for (unsigned i = 0; i < width; i++) {
      Value oldIV = origLoops[i].getInductionVar();
      SmallVector<Value, 2> val;
      AffineExpr newMixIVExpr = b.getAffineConstantExpr(tileSizes[i][0]) * b.getAffineDimExpr(0);
      val.push_back(tiledLoops[i].getInductionVar());
      for(unsigned j = 0; j < tileSizes[i].size(); j ++, newLoopIdx ++){
          if(tileSizes[i].size() - j - 1 < 2){
              if(tileSizes[i].size() - j - 1 == 1){
                   newMixIVExpr = b.getAffineConstantExpr(tileSizes[i][j + 1]) * b.getAffineDimExpr(0);
                   val.clear();
              } else {
                   newMixIVExpr = newMixIVExpr + b.getAffineDimExpr(1);
              }
              val.push_back(tiledLoops[newLoopIdx].getInductionVar());
          }
      }

      AffineMap newIV = AffineMap::get(2, 0, newMixIVExpr, b.getContext());
      AffineApplyOp applyOp = b.create<AffineApplyOp>(rootAffineForOp.getLoc(), newIV, ValueRange(val));

      oldIV.replaceAllUsesWith(applyOp);
    }

    // Erase the old loop nest.
    rootAffineForOp.erase();

    if (tiledNest)
      *tiledNest = std::move(tiledLoops);

    return success();
}

void AffineSuperTillingPass::runOnOperation() {
    // first rebuilt tile strategy.
    SmallVector<SmallVector<unsigned>> tileStrategies;
    bool needNewDim = true;
    for(auto size : tileStrategy){
        if(needNewDim){
            tileStrategies.push_back(SmallVector<unsigned>());
            needNewDim = false;
        }
        if(size == -1){
            needNewDim = true;
        }else {
            tileStrategies.back().push_back(size);
        }
    }

    // Collect all band.
    std::vector<SmallVector<AffineForOp, 6>> bands;
    getTileableBands(getOperation(), &bands);

    // Tile each band.
    for (int i = 0; i < bands.size(); ++ i) {
      auto band = bands[i];
      auto tileSizes = tileStrategies;

      SmallVector<AffineForOp, 6> tiledNest;
      if (failed(superTilePerfectlyNested(band, tileSizes, &tiledNest))) {
        // An empty band always succeeds.
        assert(!band.empty() && "guaranteed to succeed on empty bands");
        llvm::errs() << "Tile Fail\n";
        continue;
      }
    }

}

namespace mlir {
namespace buddy {
void registerAffineSuperTillingPass() {
  PassRegistration<AffineSuperTillingPass>();
}
} // namespace buddy
} // namespace mlir
