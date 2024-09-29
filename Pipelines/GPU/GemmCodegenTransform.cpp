#include "Transform/Transforms/TransformInsertion.h"
#include "Pipelines/GPU/GemmCodegenTransform.h"
#include "Utils/GemmCodegenUtils.h"
#include "Utils/PipelineUtils.h"

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
using namespace mlir::buddy;

namespace {

void createAddGemmCodegenLoweringConfigTransformImpl(
    OpPassManager &pm, const std::string &anchor, const std::string &prefix,
    ArrayRef<int64_t> tileConfig, ArrayRef<int64_t> workGroup, int64_t stages) {

    SmallVector<int64_t> vecTileConfig{tileConfig};
    SmallVector<int64_t> vecWorkGroup{workGroup};

    TransformInsertionConfig config;
    config.funcAnchor = anchor;
    config.matchPrefix = prefix;
    // transform operation takes effect needed to have this op
    config.opFilter = [=](Operation *op){
        if (isLinalgMatmul(op)) {
            return true;
        }
        return false;
    };

    // pdlV is a handle of op
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
        
        Value tileConfigValue_M = b.create<transform::ParamConstantOp>(
            transform::ParamType::get(b.getContext(), mlir::IntegerType::get(b.getContext(), 64)),
            /* value */ tileConfigAttrs.getValue()[0]
        );

        Value tileConfigValue_N = b.create<transform::ParamConstantOp>(
            transform::ParamType::get(b.getContext(), mlir::IntegerType::get(b.getContext(), 64)),
            /* value */ tileConfigAttrs.getValue()[1]
        );

        Value tileConfigValue_K = b.create<transform::ParamConstantOp>(
            // /* type */ pdl::AttributeType::get(b.getContext()),
            transform::ParamType::get(b.getContext(), mlir::IntegerType::get(b.getContext(), 64)),
            /* value */ tileConfigAttrs.getValue()[2]
        );

        Value workGroupValue_X = b.create<transform::ParamConstantOp>(
            transform::ParamType::get(b.getContext(), mlir::IntegerType::get(b.getContext(), 64)),
            /* value */ workgroupAttrs.getValue()[0]
        );

        Value workGroupValue_Y = b.create<transform::ParamConstantOp>(
            transform::ParamType::get(b.getContext(), mlir::IntegerType::get(b.getContext(), 64)),
            /* value */ workgroupAttrs.getValue()[1]
        );

        Value workGroupValue_Z = b.create<transform::ParamConstantOp>(
            transform::ParamType::get(b.getContext(), mlir::IntegerType::get(b.getContext(), 64)),
            /* value */ workgroupAttrs.getValue()[2]
        );

        Value stagesValue = b.create<transform::ParamConstantOp>(
            transform::ParamType::get(b.getContext(), mlir::IntegerType::get(b.getContext(), 64)),
            /* value */ stagesAttr
        );

        b.create<transform::AnnotateOp>(func, getGemmTileMConfigAttrName(),
                                        tileConfigValue_M);
        b.create<transform::AnnotateOp>(func, getGemmTileNConfigAttrName(),
                                        tileConfigValue_N);
        b.create<transform::AnnotateOp>(func, getGemmTileKConfigAttrName(),
                                        tileConfigValue_K);
        b.create<transform::AnnotateOp>(func, getGemmBlockXSizeAttrName(),
                                        workGroupValue_X);
        b.create<transform::AnnotateOp>(func, getGemmBlockYSizeAttrName(),
                                        workGroupValue_Y);
        b.create<transform::AnnotateOp>(func, getGemmBlockZSizeAttrName(),
                                        workGroupValue_Z);
        b.create<transform::AnnotateOp>(func, getGemmPipelineStageAttrName(),
                                        stagesValue);
    };

    pm.addPass(createGenericTransformInsertionPass(config));
}

} // namespace

void mlir::buddy::createGemmTileConfigInsertTransform(
    OpPassManager &pm, const GPUGemmCodegenConfigOptions &options) {
    invokeOpPassPipelineBuilder(
        createAddGemmCodegenLoweringConfigTransformImpl, pm,
        options.funcAnchor, options.annotatePrefix, options.tileConfig,
        options.workGroup, options.stages);
}

namespace {

// TODO: Epilogue
void createGemmTileTransformImpl(OpPassManager &pm,
                             const std::string &anchor, 
                             const std::string &prefix) {
    TransformInsertionConfig config;
    config.funcAnchor = anchor;
    config.matchPrefix = prefix;
    config.opFilter = [=](Operation *op){
        if (isLinalgMatmul(op)) {
            return true;
        }
        return false;
    };
    config.transformBuilder = [=](ImplicitLocOpBuilder &b, Operation *op, Value pdlV) {
        func::FuncOp funcOp = op->getParentOfType<func::FuncOp>();
        linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(op);

        SmallVector<int64_t, 3> tileConfig = getGemmTileSize(funcOp).value();
        SmallVector<int64_t, 3> workGroup = getGemmBlockSize(funcOp).value();
        int64_t stages = getGemmPipelineStages(funcOp).value();

        bool hasEpilogue =  false;

        auto func = b.create<transform::GetParentOp>(
            pdlV.getType(), pdlV,
            /* isolated_from_above */ false,
            /* allow_empty_results */ false,
            /* op_name */ b.getStringAttr(func::FuncOp::getOperationName()),
            /* deduplicate */ false,
            /* nth_parent */ 1);

        auto linalgFillType = transform::OperationType::get(
            b.getContext(), linalg::FillOp::getOperationName()
        );
        auto linalgFillOp = b.create<transform::MatchOp>(
            /* resultTypes */ linalgFillType, 
            /* target */ func, 
            /* opNames */ linalg::FillOp::getOperationName()
        );

        SmallVector<int64_t> mappingIdx;
        bool isBMM = linalgOp.getNumParallelLoops() == 3;
        if (isBMM) {
            // 2 -> blockIdx.z 1 -> blockIdx.y 0->blockIdx.x
            mappingIdx = {2, 1, 0};
        } else {
            // 1 -> blockIdx.y 0 -> blockIdx.x
            mappingIdx = {1, 0};
        }

        // get GPU BlockIdx mapping
        auto mapping = llvm::to_vector(llvm::map_range(
            mappingIdx, 
            [](int64_t i){return static_cast<gpu::MappingId>(i);
            }));
        auto mappingAttrs = llvm::to_vector(llvm::map_range(
            mapping,
            [&](gpu::MappingId dim) -> Attribute {
                return gpu::GPUBlockMappingAttr::get(b.getContext(), dim);
            }));

        SmallVector<int64_t> parallelTileSizes;
        if (isBMM) {
            parallelTileSizes = {1, tileConfig[0], tileConfig[1]};
        } else {
            parallelTileSizes = {tileConfig[0], tileConfig[1]};
        }
        
        // tile DimM and DimN and each tile dispathes to block 
        Value tiledMatmulOp;
        if (hasEpilogue) {
            // TODO
        } else {
            transform::TileUsingForallOp tiledResultOp = 
                b.create<transform::TileUsingForallOp>(
                /* target */ pdlV,
                /* staticTileSizes */ parallelTileSizes,
                /* ctor tag */ transform::TileSizesSpec(),
                /* mapping */ b.getArrayAttr(mappingAttrs)
                );

            if (linalgFillOp) {
                b.create<transform::FuseIntoContainingOp>(
                    /* producerOp */ linalgFillOp,
                    /* containingOp */ tiledResultOp.getForallOp()
                );
            }
            tiledMatmulOp = tiledResultOp.getTiledOp();
        }
                
        // only tile DimK of the matmul which is dispatched to each block
        SmallVector<int64_t> reduceTileSize;
        if (isBMM) {
            reduceTileSize = {0, 0, 0, tileConfig[2]};
        } else {
            reduceTileSize = {0, 0, tileConfig[2]};
        }

        auto tiledKMatmulOp = 
            b.create<transform::TileUsingForOp>(
              /* target */ tiledMatmulOp,
              /* staticTileSizes */ reduceTileSize
            );

        // for k in K steps tileConfig[2]
        auto forLoops = tiledKMatmulOp.getLoops();
        // tiledmatmul computes at (BM, BN, tileConfig[2])
        auto kMatmulOp = tiledKMatmulOp.getTiledLinalgOp();

        if (!forLoops.empty()) {
            b.create<transform::AnnotateOp>(forLoops[0], getMatmulKMainLoopMarker(),
                                            Value());
        } else {
            b.create<transform::AnnotateOp>(kMatmulOp, getMatmulKMainLoopMarker(),
                                            Value());
        }

        // Value mmaLevel = b.create<transform::ParamConstantOp>(
        //     /* type */ transform::ParamType::get(b.getContext(), b.getStringAttr()),
        //     /* value */ b.getStringAttr("Threadblock")
        // );

        // b.create<transform::AnnotateOp>(kMatmulOp, getLinalgMMALevelAttrName(),
        //                                 mmaLevel);
        b.create<transform::AnnotateOp>(kMatmulOp, getMMAPatternAttrName(),
                                        Value());
    };
    pm.addPass(createGenericTransformInsertionPass(config));
}
} // namespace

void mlir::buddy::createGemmTileTransform(OpPassManager &pm,
                             const GPUGemmGeneralOptions &options) {
    invokeOpPassPipelineBuilder(
        createGemmTileTransformImpl, pm,
        options.funcAnchor, options.annotatePrefix);
}