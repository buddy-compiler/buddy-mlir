include "mlir/Pass/PassBase.td"

def LinalgPromotion : Pass<"linalg-promotion", "func::FuncOp"> {
    let summary = "promote Linalg's MatmulOp operand subview to memref.alloca and linalg.copy";
    let constructor = "mlir::createLinalgPromotionPass()";
}
