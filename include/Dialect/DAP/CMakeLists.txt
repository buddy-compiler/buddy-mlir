set(LLVM_TARGET_DEFINITIONS DAPOps.td)
mlir_tablegen(DAPOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(DAPOpsEnums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS DAPOps.td)
mlir_tablegen(DAPOpsAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(DAPOpsAttributes.cpp.inc -gen-attrdef-defs)

add_mlir_dialect(DAPOps dap)
# add_mlir_doc(DAPDialect -gen-dialect-doc DAPDialect DAP/)
# add_mlir_doc(DAPOps -gen-op-doc DAPOps DAP/)
