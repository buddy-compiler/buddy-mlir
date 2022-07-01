# Use MLIR TableGen to generate Ops code.
set(LLVM_TARGET_DEFINITIONS ../../../../llvm/mlir/examples/toy/Ch5/include/toy/Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(ToyDSLOpsIncGen)

# Most dialects should use add_mlir_interfaces().
set(LLVM_TARGET_DEFINITIONS ../../../../llvm/mlir/examples/toy/Ch5/include/toy/ShapeInferenceInterface.td)
mlir_tablegen(ShapeInferenceOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ShapeInferenceOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(ToyShapeInferenceInterfaceIncGen)

# Use MLIR TableGen to generate toycombine code.
set(LLVM_TARGET_DEFINITIONS ../../../../llvm/mlir/examples/toy/Ch5/mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCombineIncGen)
