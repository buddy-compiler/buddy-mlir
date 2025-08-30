# Register the VIR dialect (operations, types, attributes).
add_mlir_dialect(VIR vir)

set(LLVM_TARGET_DEFINITIONS VIRAttrs.td)
mlir_tablegen(VIRAttrs.h.inc  -gen-attrdef-decls)
mlir_tablegen(VIRAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRVIRAttrsIncGen)
