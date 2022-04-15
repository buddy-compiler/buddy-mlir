set(LLVM_TARGET_DEFINITIONS BudOps.td)
mlir_tablegen(BudOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(BudOpsEnums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS BudOps.td)
mlir_tablegen(BudOpsAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(BudOpsAttributes.cpp.inc -gen-attrdef-defs)

add_mlir_dialect(BudOps bud)
add_mlir_doc(BudDialect -gen-dialect-doc BudDialect Bud/)
add_mlir_doc(BudOps -gen-op-doc BudOps Bud/)
