#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"

#include "PassDetail.h"
#include "Transform/Transforms/TransformInsertion.h"

#include <optional>
#include <vector>

using namespace mlir;
using namespace llvm;

namespace {

inline std::string getAnnotationUniqueIdentifier(const std::string matchPrefix)
{
    static size_t cnt = 0;
    return matchPrefix + "_" + std::to_string(cnt++);
}

void insertTransformIR(func::FuncOp funcOp, OpBuilder &builder, 
                        const TransformInsertionConfig &config) {
    funcOp->walk([&](Operation *op) { 
        if (config.opFilter(op)) {
            ImplicitLocOpBuilder b(op->getLoc(), builder);
            MLIRContext *ctx = b.getContext();

            auto annotation = getAnnotationUniqueIdentifier(config.matchPrefix);
            op->setAttr(annotation, UnitAttr::get(ctx));

            auto pdlOperationType = pdl::OperationType::get(ctx);
            b.create<transform::SequenceOp>(
                TypeRange(), transform::FailurePropagationMode::Propagate, pdlOperationType, 
                [&](OpBuilder &b, Location loc, Value blockArg) {
                    auto annotationAttr = DictionaryAttr::get(
                        ctx, b.getNamedAttr(annotation, UnitAttr::get(ctx)));
                    auto match = b.create<transform::MatchOp>(
                        loc, blockArg.getType(), blockArg, ArrayAttr(),
                        transform::MatchInterfaceEnumAttr(), annotationAttr, TypeAttr()
                        /*ArrayAttr()*/);
                    ImplicitLocOpBuilder ib(loc, b);
                    config.transformBuilder(ib, op, match);
                    b.create<transform::YieldOp>(loc);
                });
        }
    });
}

void insertTransformIR(ModuleOp module, const TransformInsertionConfig &config) {
    OpBuilder builder = OpBuilder::atBlockEnd(module.getBody());
    for (auto funcOp : module.getOps<func::FuncOp>()) {
        if (!config.funcAnchor.empty() && !funcOp->hasAttr(config.funcAnchor)) {
            continue;
        }
        insertTransformIR(funcOp, builder, config);
    }
}

struct GenericTransformInsertionPass
    : public PassWrapper<GenericTransformInsertionPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenericTransformInsertionPass)

    GenericTransformInsertionPass(const TransformInsertionConfig &config) : config(config) {}

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<transform::TransformDialect>();
    }

    void runOnOperation() override {
        insertTransformIR(getOperation(), config);
    }

protected:
  TransformInsertionConfig config;
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createGenericTransformInsertionPass(const mlir::TransformInsertionConfig &config) {
    return std::make_unique<GenericTransformInsertionPass>(config);
}
