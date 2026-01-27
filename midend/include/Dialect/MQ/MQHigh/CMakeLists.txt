set(LLVM_TARGET_DEFINITIONS MQHighOps.td)
mlir_tablegen(MQHighOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(MQHighOpsEnums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS MQHighOps.td)
mlir_tablegen(MQHighOpsAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(MQHighOpsAttributes.cpp.inc -gen-attrdef-defs)

add_mlir_dialect(MQHighOps mqhigh)
# add_mlir_doc(MQHighDialect -gen-dialect-doc MQHighDialect MQHigh/)
# add_mlir_doc(MQHighOps -gen-op-doc MQHighOps MQHigh/)
