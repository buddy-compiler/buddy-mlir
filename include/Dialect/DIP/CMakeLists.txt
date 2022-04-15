set(LLVM_TARGET_DEFINITIONS DIPOps.td)
mlir_tablegen(DIPOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(DIPOpsEnums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS DIPOps.td)
mlir_tablegen(DIPOpsAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(DIPOpsAttributes.cpp.inc -gen-attrdef-defs)

add_mlir_dialect(DIPOps dip)
# add_mlir_doc(DIPDialect -gen-dialect-doc DIPDialect DIP/)
# add_mlir_doc(DIPOps -gen-op-doc DIPOps DIP/)
