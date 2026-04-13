add_mlir_dialect(XTAME xt_ame)
add_mlir_doc(XTAME XTAME Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS XTAME.td)
mlir_tablegen(XTAMEConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(BuddyXTAMEConversionsIncGen)
