include "mlir/Pass/PassBase.td"

// TransformDialectInterpreter
def TransformDialectInterpreter : Pass<"transform-dialect-interpreter", "mlir::ModuleOp"> {
    let summary = "Apply transform dialect operations one by one";
    let constructor = "mlir::createTransformDialectInterpreter()";
    let options = [
        Option<"eraseAfter", "erase-after", "bool", "false", "erase Transform Ops after applied">
    ];
}
