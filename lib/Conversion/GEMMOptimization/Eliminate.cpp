//===- Eliminate.cpp - Make loop perfect ------===//
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
// This file implements Eliminate algorithm.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Instructions.h"
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
// LoopPerfectPass
//===----------------------------------------------------------------------===//

namespace {
class EliminatePass
    : public PassWrapper<EliminatePass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EliminatePass)
  StringRef getArgument() const final { return "gemm-eliminate"; }
  StringRef getDescription() const final {
    return "Eliminate redundent access and load.";
  }
  EliminatePass() = default;
  EliminatePass(const EliminatePass &) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, memref::MemRefDialect, func::FuncDialect>();
  }
};
} // end anonymous namespace.

void EliminatePass::runOnOperation() {
  auto func = getOperation();
  std::stringstream ss;
  llvm::DenseMap<std::pair<mlir::Value, mlir::Value>, llvm::SmallVector<vector::TransferWriteOp, 32>> map;
  func.walk([&](vector::TransferWriteOp writeOp){
		  auto indi = writeOp.getIndices();
		  map[std::make_pair(indi[0], indi[1])].push_back(writeOp);
  });
  for(auto a : map){
	  auto writeOps = a.second;
	  Value lastUsed;
	  for(size_t i = 0; i < writeOps.size(); ++ i){
		  auto op = writeOps[i];
		if(i == 0){
			lastUsed = op.getVector();
			op.erase();
		} else if(i < writeOps.size() - 1){
			auto t = op.getVector();
			t.getDefiningOp()->setOperand(0, lastUsed);
			lastUsed = t;
			op.erase();
		} else {
			auto t = op.getVector();
			t.getDefiningOp()->setOperand(0, lastUsed);
			lastUsed = t;
		}

	  }
  }
  

}

namespace mlir {
namespace buddy {
void registerEliminatePass() {
  PassRegistration<EliminatePass>();
}
} // namespace buddy
} // namespace mlir
