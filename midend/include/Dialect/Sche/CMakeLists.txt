set(LLVM_TARGET_DEFINITIONS ScheOps.td)
mlir_tablegen(ScheOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(ScheOpsEnums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS ScheOps.td)
mlir_tablegen(ScheOpsAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(ScheOpsAttributes.cpp.inc -gen-attrdef-defs)

set(LLVM_TARGET_DEFINITIONS ScheDialect.td)
mlir_tablegen(ScheOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ScheOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRScheOpInterfacesIncGen)

add_mlir_dialect(ScheOps sche)
