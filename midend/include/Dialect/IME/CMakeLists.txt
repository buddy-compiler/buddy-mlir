add_mlir_dialect(IME ime)
add_mlir_doc(IME IME Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS IME.td)
mlir_tablegen(IMEConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(BuddyIMEConversionsIncGen)
