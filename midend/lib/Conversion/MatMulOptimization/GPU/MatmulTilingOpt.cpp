#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"

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

namespace mlir {

struct GPUGemmCodegenConfigOptions
    : public PassPipelineOptions<GPUGemmCodegenConfigOptions> {
  Option<std::string> funcAnchor{
      *this, "func-anchor",
      llvm::cl::desc(
          "An optional Unit attribute anchoring on target functions."),
      llvm::cl::init("")};
  Option<std::string> annotatePrefix{
      *this, "annotate-prefix",
      llvm::cl::desc("An optional annotate prefix attribute on target ops."),
      llvm::cl::init("__buddyir_gpu_tile_gemm")};
  ListOption<int64_t> tileSizeConfig{
      *this, "tile-size-config",
      llvm::cl::desc("An optional tile size config for tile matmul op.")};
  ListOption<int64_t> workgroupSize{
      *this, "workgroup-size",
      llvm::cl::desc("An optional workgroup size config for tile matmul op.")};
  Option<int64_t> stages{
      *this, "stages", llvm::cl::desc("An optional stages for tile matmul op."),
      llvm::cl::init(3)};
};

struct GPUGemmGeneralOptions
    : public PassPipelineOptions<GPUGemmGeneralOptions> {
  Option<std::string> funcAnchor{
      *this, "func-anchor",
      llvm::cl::desc(
          "An optional Unit attribute anchoring on target functions."),
      llvm::cl::init("")};
  Option<std::string> annotatePrefix{
      *this, "annotate-prefix",
      llvm::cl::desc("An optional annotate prefix attribute on target ops."),
      llvm::cl::init("__buddyir_gpu_tile_gemm")};
};

using namespace mlir;

namespace {

constexpr StringRef getLinalgToGPUAttrName() { return "__buddyir_to_gpu__"; }

constexpr StringRef getLinalgTargetAttrName() { return "__buddyir_target__"; }

template <typename OpClass = ModuleOp, typename Builder, typename... Args>
void invokeOpPassPipelineBuilder(Builder builder, OpPassManager &pm,
                                 Args &&...args) {
  if (pm.getOpAnchorName() != OpPassManager::getAnyOpAnchorName() &&
      pm.getOpAnchorName() != OpClass::getOperationName()) {
    if (pm.getNesting() == OpPassManager::Nesting::Implicit) {
      builder(pm.nest<OpClass>(), std::forward<Args>(args)...);
      return;
    }
    llvm::report_fatal_error(
        llvm::Twine("Can't build pass pipeline on expected op type ") +
        OpClass::getOperationName() + " but got " + pm.getOpAnchorName());
  } else {
    builder(pm, std::forward<Args>(args)...);
  }
}

void createGPUTileGemmTransformImpl(OpPassManager &pm,
                                    const std::string &anchor,
                                    const std::string &prefix) {
  TransformInsertionConfig config;
  config.funcAnchor = anchor;
  config.matchPrefix = prefix;
  config.opFilter = [=](Operation *op) {
    if (!isLinalgOpMatmul(op))
      return false;
    return true;
  };

  config.transformBuilder = [=](ImplicitLocOpBuilder &b, Operation *op,
                                Value pdlV) {
    func::FuncOp funcOp = op->getParentOfType<func::FuncOp>();
    linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(op);
    Operation *user = *linalgOp->getUsers().begin();
    bool hasEpilogue = isa<linalg::GenericOp>(user);

    if (hasEpilogue) {
      setMarker(user, getEpilogueMarker());
    }

    bool isBMM = linalgOp.getNumParallelLoops() == 3;

    SmallVector<int64_t, 3> tileSizeConfig = getGemmTileSize(funcOp).value();

    auto func = b.create<transform::GetParentOp>(
        pdlV.getType(), pdlV,
        /* isolated_from_above */ false,
        /* allow_empty_results */ false,
        /* op_name */ b.getStringAttr(func::FuncOp::getOperationName()),
        /* deduplicate */ false,
        /* nth_parent */ 1);

    auto anyType = transform::AnyOpType::get(b.getContext());
    auto linalgFillType = transform::OperationType::get(
        b.getContext(), linalg::FillOp::getOperationName());
    auto linalgFill = b.create<transform::MatchOp>(
        linalgFillType, func, linalg::FillOp::getOperationName());

    Value mmaLevel = b.create<transform::ParamConstantOp>(
        /* type */ pdl::AttributeType::get(b.getContext()),
        /* value */ b.getStringAttr("Threadblock"));
    Value target = b.create<transform::ParamConstantOp>(
        /* type */ pdl::AttributeType::get(b.getContext()),
        /* value */ b.getStringAttr("nv_sm_80"));

    SmallVector<int64_t> mappingIdx;
    if (isBMM) {
      mappingIdx = {2, 1, 0};
    } else {
      mappingIdx = {1, 0};
    }
    auto mapping = llvm::to_vector(llvm::map_range(
        mappingIdx, [](int64_t i) { return static_cast<gpu::MappingId>(i); }));
    auto mappingAttrs = llvm::to_vector(
        llvm::map_range(mapping, [&](gpu::MappingId dim) -> Attribute {
          return gpu::GPUBlockMappingAttr::get(b.getContext(), dim);
        }));

    SmallVector<int64_t> parrallelTileSizes;
    if (isBMM) {
      parrallelTileSizes = {1, tileSizeConfig[0], tileSizeConfig[1]};
    } else {
      parrallelTileSizes = {tileSizeConfig[0], tileSizeConfig[1]};
    }
    Value tiledMatmulOp;
    if (hasEpilogue) {
      auto linalgGenericType = transform::OperationType::get(
          b.getContext(), linalg::GenericOp::getOperationName());
      auto epilogue = b.create<transform::MatchOp>(
          linalgGenericType, func,
          b.getStrArrayAttr({linalg::GenericOp::getOperationName()}),
          /*matchInterfaceEnum=*/transform::MatchInterfaceEnumAttr(),
          /*opAttrs=*/
          b.getDictionaryAttr({NamedAttribute(
              b.getStringAttr(getEpilogueMarker()), b.getUnitAttr())}),
          /*filterResultType=*/TypeAttr(),
          /*filterOperandTYpes=*/ArrayAttr());

      transform::TileUsingForallOp tileOp =
          b.create<transform::TileUsingForallOp>(
              /* target */ epilogue,
              /* staticTileSizes */ parrallelTileSizes,
              /* ctor tag */ transform::TileSizesSpec(),
              /* mapping */ b.getArrayAttr(mappingAttrs));
      transform::FuseIntoContainingOp fuse =
          b.create<transform::FuseIntoContainingOp>(
              /* producerOp */ pdlV,
              /* containingOp */ tileOp.getForallOp());
      b.create<transform::FuseIntoContainingOp>(
          /* producerOp */ linalgFill,
          /* containingOp */ fuse.getNewContainingOp());
      tiledMatmulOp = fuse.getFusedOp();
    } else {
      transform::TileUsingForallOp tileOp =
          b.create<transform::TileUsingForallOp>(
              /* target */ pdlV,
              /* staticTileSizes */ parrallelTileSizes,
              /* ctor tag */ transform::TileSizesSpec(),
              /* mapping */ b.getArrayAttr(mappingAttrs));

      b.create<transform::FuseIntoContainingOp>(
          /* producerOp */ linalgFill,
          /* containingOp */ tileOp.getForallOp());
      tiledMatmulOp = tileOp.getTiledOp();
    }

    SmallVector<int64_t> reductionTileSizes;
    if (isBMM)
      reductionTileSizes = {0, 0, 0, tileSizeConfig[2]};
    else
      reductionTileSizes = {0, 0, tileSizeConfig[2]};
    auto tileKMatmulOp =
        b.create<transform::TileUsingForOp>(tiledMatmulOp, reductionTileSizes);
    auto matmulKOp = tileKMatmulOp.getTiledLinalgOp();
    auto forLoops = tileKMatmulOp.getLoops();
    if (!forLoops.empty()) {
      b.create<transform::AnnotateOp>(forLoops[0], getMatmulMainLoopMarker(),
                                      Value());
    } else {
      b.create<transform::AnnotateOp>(matmulKOp, getMatmulMainLoopMarker(),
                                      Value());
    }

    b.create<transform::AnnotateOp>(matmulKOp, getLinalgMMALevelAttrName(),
                                    mmaLevel);
    b.create<transform::AnnotateOp>(matmulKOp, getLinalgTargetAttrName(),
                                    target);
    b.create<transform::AnnotateOp>(matmulKOp, getMMAPatternAttrName(),
                                    Value());
  };

  pm.addPass(createGenericTransformInsertionPass(config));
}

} // namespace

void mlir::createGPUTileGemmTransform(OpPassManager &pm,
                                      const GPUGemmGeneralOptions &options) {
  invokeOpPassPipelineBuilder(createGPUTileGemmTransformImpl, pm,
                              options.funcAnchor, options.annotatePrefix);
}

namespace {

void createGPUAddGemmCodegenLoweringConfigTransformImpl(
    OpPassManager &pm, const std::string &anchor, const std::string &prefix,
    ArrayRef<int64_t> tileSizeConfig, ArrayRef<int64_t> workgroupSize,
    int64_t stages) {

  SmallVector<int64_t> tileSizeConfigVec{tileSizeConfig};
  SmallVector<int64_t> workgroupSizeVec{workgroupSize};

  TransformInsertionConfig config;
  config.funcAnchor = anchor;
  config.matchPrefix = prefix;

  config.opFilter = [=](Operation *op) {
    if (isLinalgOpMatmul(op)) {
      // TODO: check if the matmul op is already annotated
      // TODO: Add different lowering config for different matmul op size
      return true;
    }
    return false;
  };

  config.transformBuilder = [=](ImplicitLocOpBuilder &b, Operation *op,
                                Value pdlV) {
    // auto linalgOp = llvm::cast<linalg::LinalgOp>(op);
    auto tileSizeConfigAttrs = b.getAttr<ArrayAttr>(llvm::to_vector(
        llvm::map_range(tileSizeConfigVec, [&](int64_t i) -> Attribute {
          return b.getI64IntegerAttr(i);
        })));
    auto workgroupSizeAttrs = b.getAttr<ArrayAttr>(llvm::to_vector(
        llvm::map_range(workgroupSizeVec, [&](int64_t i) -> Attribute {
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

    Value tileSizeConfigValue = b.create<transform::ParamConstantOp>(
        /* type */ pdl::AttributeType::get(b.getContext()),
        /* value */ tileSizeConfigAttrs);
    Value workgroupSizeValue = b.create<transform::ParamConstantOp>(
        /* type */ pdl::AttributeType::get(b.getContext()),
        /* value */ workgroupSizeAttrs);
    Value stagesValue = b.create<transform::ParamConstantOp>(
        /* type */ pdl::AttributeType::get(b.getContext()),
        /* value */ stagesAttr);

    b.create<transform::AnnotateOp>(func, getGemmTileConfigAttrName(),
                                    tileSizeConfigValue);
    b.create<transform::AnnotateOp>(func, getGemmBlockSizeAttrName(),
                                    workgroupSizeValue);
    b.create<transform::AnnotateOp>(func, getGemmPipelineDepthAttrName(),
                                    stagesValue);
  };
  pm.addPass(createGenericTransformInsertionPass(config));
}
} // namespace

void mlir::createGPUAddGemmCodegenLoweringConfigTransform(
    OpPassManager &pm, const GPUGemmCodegenConfigOptions &options) {
  invokeOpPassPipelineBuilder(
      createGPUAddGemmCodegenLoweringConfigTransformImpl, pm,
      options.funcAnchor, options.annotatePrefix, options.tileSizeConfig,
      options.workgroupSize, options.stages);
}

}