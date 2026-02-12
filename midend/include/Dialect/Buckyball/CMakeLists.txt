add_mlir_dialect(Buckyball buckyball)
add_mlir_doc(Buckyball Buckyball Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS Buckyball.td)
mlir_tablegen(BuckyballConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(BuddyBuckyballConversionsIncGen)
