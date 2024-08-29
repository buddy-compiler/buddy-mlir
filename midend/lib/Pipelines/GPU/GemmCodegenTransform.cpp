#include "Transform/Transforms/TransformInsertion.h"
#include "GPU/Transforms/Utils.h"
#include "Pipelines/GPU/GemmCodegenTransform.h"
#include "Pipelines/GPU/Utils.h"

#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"
#include "mlir/Dialect/NVGPU/TransformOps/NVGPUTransformOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallSet.h"

#include <optional>

using namespace mlir;

namespace {

void createGPUAddGemmCodegenLoweringConfigTransformImpl(
    OpPassManager &pm, const std::string &anchor, const std::string &prefix,
    ArrayRef<int64_t> tileConfig, ArrayRef<int64_t> workGroup, int64_t stages) {

    SmallVector<int64_t> vecTileConfig{tileConfig};
    SmallVector<int64_t> vecWorkGroup{workGroup};

    TransformInsertionConfig config;
    config.funcAnchor = anchor;

    config.matchPrefix = prefix;

    config.opFilter = [=](Operation *op){
        if (mlir::buddy::isLinalgMatmul(op)) {
            return true;
        }
        return false;
    };

    config.transformBuilder = [=](ImplicitLocOpBuilder &b, Operation *op, Value pdlV) {
        auto tileConfigAttrs = b.getAttr<ArrayAttr>(llvm::to_vector(
            llvm::map_range(vecTileConfig, [&](int64_t i) -> Attribute {
            return b.getI64IntegerAttr(i);
            })));
        auto workgroupAttrs = b.getAttr<ArrayAttr>(llvm::to_vector(
            llvm::map_range(vecWorkGroup, [&](int64_t i) -> Attribute {
            return b.getI64IntegerAttr(i);
            })));
        auto stagesAttr = b.getI64IntegerAttr(stages);

        auto func = b.create<transform::GetParentOp>(
            pdlV.getType(), pdlV,
            /* isolated_from_above */ true,
            /* allow_empty_results */ false,
            /* op_name */ b.getStringAttr(func::FuncOp::getOperationName()),
            /* deduplicate */ false,
            /* nth_parent */ 1);
            
        Value tileConfigValue = b.create<transform::ParamConstantOp>(
            /* type */ pdl::AttributeType::get(b.getContext()),
            /* value */ tileConfigAttrs
        );

        llvm::errs() << tileConfigValue << "\n";

        Value workGroupValue = b.create<transform::ParamConstantOp>(
            /* type */ pdl::AttributeType::get(b.getContext()),
            /* value */ workgroupAttrs
        );

        Value stagesValue = b.create<transform::ParamConstantOp>(
            /* type */ pdl::AttributeType::get(b.getContext()),
            /* value */ stagesAttr
        );

        b.create<transform::AnnotateOp>(func, getGemmTileConfigAttrName(),
                                    tileConfigValue);
        b.create<transform::AnnotateOp>(func, getGemmBlockSizeAttrName(),
                                        workGroupValue);
        b.create<transform::AnnotateOp>(func, getGemmPipelineStageAttrName(),
                                        stagesValue);
    };

    pm.addPass(createGenericTransformInsertionPass(config));
}

} // namespace

void mlir::buddy::createGPUGemmTileConfigInsertTransform(
    OpPassManager &pm, const GPUGemmCodegenConfigOptions &options) {
    mlir::buddy::invokeOpPassPipelineBuilder(
        createGPUAddGemmCodegenLoweringConfigTransformImpl, pm,
        options.funcAnchor, options.annotatePrefix, options.tileConfig,
        options.workGroup, options.stages);
}