add_mlir_dialect(RVV rvv)
add_mlir_doc(RVV RVV Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS RVV.td)
mlir_tablegen(RVVConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(BuddyRVVConversionsIncGen)
