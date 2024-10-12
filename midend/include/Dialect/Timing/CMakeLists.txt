set(LLVM_TARGET_DEFINITIONS TimingOps.td)
mlir_tablegen(TimingOps.h.inc -gen-op-decls)
mlir_tablegen(TimingOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS TimingOps.td)
mlir_tablegen(TimingOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(TimingOpsEnums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS TimingOps.td)
mlir_tablegen(TimingOpsAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(TimingOpsAttributes.cpp.inc -gen-attrdef-defs)

add_mlir_dialect(TimingOps timing)
# add_mlir_doc(BudDialect -gen-dialect-doc BudDialect Bud/)
# add_mlir_doc(BudOps -gen-op-doc BudOps Bud/)
