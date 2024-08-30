set(LLVM_TARGET_DEFINITIONS BuddyGPUTransformOps.td)
mlir_tablegen(BuddyGPUTransformOps.h.inc -gen-op-decls)
mlir_tablegen(BuddyGPUTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(BuddyGPUTransformOpsIncGen)

# set(LLVM_TARGET_DEFINITIONS BuddyGPUOps.td)
# mlir_tablegen(BuddyGPUOpsEnums.h.inc -gen-enum-decls)
# mlir_tablegen(BuddyGPUOpsEnums.cpp.inc -gen-enum-defs)

# set(LLVM_TARGET_DEFINITIONS BuddyGPUOps.td)
# mlir_tablegen(BuddyGPUOpsAttributes.h.inc -gen-attrdef-decls)
# mlir_tablegen(BuddyGPUOpsAttributes.cpp.inc -gen-attrdef-defs)

# add_mlir_dialect(BuddyGPUOps buddygpu)
