#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "Pipelines/GPU/Utils.h"

using namespace mlir;
using namespace llvm;

namespace mlir {
namespace buddy {

bool isLinalgMatmul(Operation *op) {
    if (!llvm::isa<linalg::LinalgOp>(op)) {
        return false;
    }

    linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(op);
    if (isa<linalg::MatmulOp>(linalgOp) || isa<linalg::BatchMatmulOp>(linalgOp)) {
        return true;
    } else {
        if (!(linalg::isaContractionOpInterface(linalgOp) &&
                linalgOp.getNumParallelLoops() >= 2 &&
                linalgOp.getNumParallelLoops() <= 3)) {
            return false;
        }
        Region &body = linalgOp->getRegion(0);
        Region::OpIterator it = body.op_begin();
        while (it != body.op_end() && isa<arith::ExtFOp>(*it)) 
            it++;
        if (it == body.op_end() || !isa<arith::MulFOp>(*(it++)))
            return false;
        if (it == body.op_end() || !isa<arith::AddFOp>(*(it++)))
            return false;
        if (it == body.op_end() || !isa<linalg::YieldOp>(*(it++)))
            return false;
        AffineMap outputMap = linalgOp.getMatchingIndexingMap(linalgOp.getDpsInitOperand(0));
        if (outputMap.getNumResults() != outputMap.getNumDims() - 1)
            return false;
        OpBuilder b(linalgOp);
        for (unsigned i = 0; i < outputMap.getNumResults(); i++) {
            if (outputMap.getResult(i) != b.getAffineDimExpr(i))
                return false;
        }
        return true;
    }
}

} // namespace mlir::buddy
} // namespace mlir