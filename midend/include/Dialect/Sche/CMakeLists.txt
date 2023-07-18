set(LLVM_TARGET_DEFINITIONS ScheOps.td)
mlir_tablegen(ScheOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(ScheOpsEnums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS ScheOps.td)
mlir_tablegen(ScheOpsAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(ScheOpsAttributes.cpp.inc -gen-attrdef-defs)

add_mlir_dialect(ScheOps sche)
# add_mlir_doc(ScheDialect -gen-dialect-doc ScheDialect Sche/)
# add_mlir_doc(ScheOps -gen-op-doc ScheOps Sche/)
