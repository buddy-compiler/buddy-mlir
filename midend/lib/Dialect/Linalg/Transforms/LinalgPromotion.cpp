#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LLVM.h"

#include "Linalg/Transforms/LinalgPromotion.h"
#include "PassDetail.h"
#include "Utils/GemmCodegenUtils.h"

#include <cstdint>
#include <memory>
#include <mlir/Dialect/Linalg/IR/LinalgInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>
#include <optional>

using namespace mlir;
using namespace mlir::buddy;

namespace {

constexpr int64_t MatmulOperand_A = 0;
constexpr int64_t MatmulOperand_B = 1;
constexpr int64_t MatmulOperand_C = 2;

constexpr StringRef allocMarker[3] = {getAllocSharedMemoryAMarker(),
                                      getAllocSharedMemoryBMarker(),
                                      getAllocSharedMemoryAccMarker()};

constexpr StringRef copyMarker[3] = {getCopyToSharedMemoryAMarker(),
                                     getCopyToSharedMemoryBMarker(),
                                     getCopyFromSharedMemoryAccMarker()};

template<int OPERAND>
std::optional<Value> allocateWorkgroupMemory(OpBuilder &builder, memref::SubViewOp subview,
                        ArrayRef<Value> boundingSubViewSize, DataLayout &layout) {

    OpBuilder::InsertionGuard guard(builder);

    scf::ForallOp forallOp = subview->getParentOfType<scf::ForallOp>();
    if (!forallOp) {
        return std::nullopt;
    }

    SmallVector<int64_t, 2> shapes;
    for (Value bound : boundingSubViewSize) {
        APInt value;
        if (!matchPattern(bound, m_ConstantInt(&value))) {
            return std::nullopt;
        }
        shapes.push_back(value.getSExtValue());
    }

    builder.setInsertionPointToStart(forallOp.getBody());

    auto smemBufferType = MemRefType::get(shapes, subview.getType().getElementType(),
                                        MemRefLayoutAttrInterface{},
                                        gpu::AddressSpaceAttr::get(builder.getContext(),
                                            gpu::GPUDialect::getWorkgroupAddressSpace()));
    
    memref::AllocOp smemBuffer = builder.create<memref::AllocOp>(forallOp.getLoc(), smemBufferType);
    setMarker(smemBuffer, allocMarker[OPERAND]);

    return smemBuffer;

}

// nvgpu device mem no need to manually deallocate
LogicalResult deallocateWorkgroupMemory(OpBuilder &, Value /*buffer*/) {
    return success();
}

template <int OPERAND>
LogicalResult copyGlobalMemoryToWorkgroupMemory(OpBuilder &b, Value src,
                                                Value dst) {
    if (OPERAND == MatmulOperand_C) {
        return success();
    }
    auto copyOp = b.create<linalg::CopyOp>(src.getLoc(), src, dst);
    setMarker(copyOp, copyMarker[OPERAND]);
    return success();
}

template <int OPERAND>
LogicalResult copyWorkgroupMemoryToGlobalMemory(OpBuilder &b, Value src,
                                                Value dst) {
    OpBuilder::InsertionGuard guard(b);
    if (OPERAND == MatmulOperand_A || 
        OPERAND == MatmulOperand_B) {
        return success();        
    }
    auto op = src.getDefiningOp();

    // Because MatmulC Operand is out of scf::ForOp so we need to get ForallOp first
    scf::ForallOp forallOp = op->getParentOfType<scf::ForallOp>();
    auto forOps = llvm::to_vector(forallOp.getOps<scf::ForOp>());
    if (forOps.size() == 1) {
        // set copymem op insertion point after compute block 
        b.setInsertionPointAfter(forOps[0]);
    }
    if (forOps.size() > 1) {
        return failure();
    }
    b.create<gpu::BarrierOp>(src.getLoc());
    linalg::CopyOp copyOp = b.create<linalg::CopyOp>(src.getLoc(), src, dst);
    setMarker(copyOp, copyMarker[MatmulOperand_C]);
    return success();
}


template <int OPERAND>
static linalg::LinalgPromotionOptions getPromotionOptionsForMatmulOperand() {
  linalg::LinalgPromotionOptions promotionOptions;
  promotionOptions
      .setAllocationDeallocationFns(allocateWorkgroupMemory<OPERAND>,
                                    deallocateWorkgroupMemory)
      .setCopyInOutFns(copyGlobalMemoryToWorkgroupMemory<OPERAND>,
                       copyWorkgroupMemoryToGlobalMemory<OPERAND>)
      .setOperandsToPromote({OPERAND})
      .setUseFullTileBuffers({false, false});
  return promotionOptions;
}

template <int OPERAND>
static LogicalResult promotionImpl(OpBuilder &builder, Operation *op) {
    linalg::LinalgPromotionOptions promotionOptions = 
        getPromotionOptionsForMatmulOperand<OPERAND>();

    if (failed(linalg::promoteSubviewsPrecondition(op, promotionOptions))) {
        return failure();
    }

    std::optional<linalg::LinalgOp> promotedLinalgOp =
        linalg::promoteSubViews(builder, cast<linalg::LinalgOp>(op), promotionOptions);
    
    if (!promotedLinalgOp) {
        return op->emitError("subview promotion failed!");
    }
    return success();
}

struct LinalgPromotionPass : public LinalgPromotionBase<LinalgPromotionPass> {
public:
    LinalgPromotionPass() = default;

    void runOnOperation() override {
        // the whole promotion pipeline is 
        // split m, n, promote C, split k, promote A & B
        func::FuncOp funcOp = getOperation();
        SmallVector<linalg::LinalgOp> LinalgOpsToPromote;

        if (!funcHasGemm(funcOp)) {
            return;
        }

        if (!getForallOpMappedToBlock(funcOp)) {
            return;
        }
        scf::ForallOp forallOp = getForallOpMappedToBlock(funcOp).value();

        forallOp->walk([&](linalg::LinalgOp linalgOp) {
            if (isLinalgMatmul(linalgOp)) {
                LinalgOpsToPromote.push_back(linalgOp);
            }
        });
        if (LinalgOpsToPromote.empty()) {
            return;
        }
        assert(LinalgOpsToPromote.size() == 1);
        auto linalgContractOp = LinalgOpsToPromote[0];

        // set Builder insertion point before the linalgContractOp
        OpBuilder b(linalgContractOp);
        promotionImpl<MatmulOperand_A>(b, linalgContractOp);
        promotionImpl<MatmulOperand_B>(b, linalgContractOp);

        // set the insertion before forop to alloc MatrixC
        scf::ForOp forOp = linalgContractOp->getParentOfType<scf::ForOp>();
        if (!forOp) {
            b.setInsertionPoint(linalgContractOp);
        } else {
            b.setInsertionPoint(forOp);
        }

        promotionImpl<MatmulOperand_C>(b, linalgContractOp);
        b.setInsertionPoint(linalgContractOp);
        b.create<gpu::BarrierOp>(linalgContractOp->getLoc());
        b.setInsertionPointAfter(linalgContractOp);
        b.create<gpu::BarrierOp>(linalgContractOp->getLoc());
        
    }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLinalgPromotionPass() {
    return std::make_unique<LinalgPromotionPass>();
}
