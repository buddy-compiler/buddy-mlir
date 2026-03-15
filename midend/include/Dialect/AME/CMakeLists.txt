add_mlir_dialect(AME ame)
add_mlir_doc(AME AME Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS AME.td)
mlir_tablegen(AMEConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(BuddyAMEConversionsIncGen)
