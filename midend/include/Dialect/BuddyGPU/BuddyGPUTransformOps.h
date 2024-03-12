
#ifndef BUDDYGPU_TRANSFORM_OPS_H
#define BUDDYGPU_TRANSFORM_OPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"


namespace mlir {
class DialectRegistry;

namespace func {
class FuncOp;
}

namespace scf {
class ForallOp;
class IfOp;
class ForOp;
} // namespace scf

namespace vector {
class VectorDialect;
class WarpExecuteOnLane0Op;
} // namespace vector

}

namespace mlir{
namespace buddy{
void registerBuddyTransformOps(mlir::DialectRegistry &registry);

namespace buddygpu::transform_dialect {

class BuddyGPUTransformExtensions
    : public mlir::transform::TransformDialectExtension<
          BuddyGPUTransformExtensions> {
public:
  BuddyGPUTransformExtensions();
};
} // namespace buddygpu::transform_dialect
} // namespace buddy
} // namespace mlir

#define GET_OP_CLASSES
#include "BuddyGPU/BuddyGPUTransformOps.h.inc"

#endif // BUDDYGPU_TRANSFORM_OPS_H
