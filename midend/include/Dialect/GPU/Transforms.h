#include "mlir/IR/PatternMatch.h" 
#include "mlir/IR/MLIRContext.h"  
namespace mlir{
    namespace buddy{
    void EliminateBroadcastExtractPatterns(mlir::RewritePatternSet &patterns, mlir::MLIRContext *context);
}
}