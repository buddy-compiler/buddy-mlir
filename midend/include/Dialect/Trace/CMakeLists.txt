set(LLVM_TARGET_DEFINITIONS TraceOps.td)
mlir_tablegen(TraceOps.h.inc -gen-op-decls)
mlir_tablegen(TraceOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS TraceOps.td)
mlir_tablegen(TraceOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(TraceOpsEnums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS TraceOps.td)
mlir_tablegen(TraceOpsAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(TraceOpsAttributes.cpp.inc -gen-attrdef-defs)

add_mlir_dialect(TraceOps trace)
