add_mlir_dialect(BOSCAME bosc_ame)
add_mlir_doc(BOSCAME BOSCAME Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS BOSCAME.td)
mlir_tablegen(BOSCAMEConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(BuddyBOSCAMEConversionsIncGen)

# `add_mlir_dialect` above only emits the op and type declarations for this
# dialect; attribute definitions (`#bosc_ame.spm<bank>`) need their own
# generators.  They share BOSCAME.td, so they are emitted under a separate
# target rather than folded into MLIRBOSCAMEIncGen.
set(LLVM_TARGET_DEFINITIONS BOSCAME.td)
mlir_tablegen(BOSCAMEAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=bosc_ame)
mlir_tablegen(BOSCAMEAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=bosc_ame)
add_public_tablegen_target(BuddyBOSCAMEAttrsIncGen)
