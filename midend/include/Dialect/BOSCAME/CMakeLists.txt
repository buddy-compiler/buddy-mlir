add_mlir_dialect(BOSCAME bosc_ame)
add_mlir_doc(BOSCAME BOSCAME Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS BOSCAME.td)
mlir_tablegen(BOSCAMEConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(BuddyBOSCAMEConversionsIncGen)
