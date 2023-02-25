#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsRISCV.h"

#include "Gemmini/GemminiDialect.h"
#include "Gemmini/GemminiOps.h"
#include "Target/LLVMIR/Dialect/Gemmini/GemminiToLLVMIRTranslation.h"

using namespace mlir;
using namespace mlir::LLVM;
using namespace buddy;

namespace {
/// Implementation of the dialect interface that converts operations belonging
/// to the RVV dialect to LLVM IR.
class GemminiDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    Operation &opInst = *op;
#include "Gemmini/GemminiConversions.inc"

    return failure();
  }
};
} // end namespace

void buddy::registerGemminiDialectTranslation(DialectRegistry &registry) {
  registry.insert<gemmini::GemminiDialect>();
  registry.addExtension(+[](MLIRContext *ctx, gemmini::GemminiDialect *dialect) {
    dialect->addInterfaces<GemminiDialectLLVMIRTranslationInterface>();
  });
}

void buddy::registerGemminiDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerGemminiDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
