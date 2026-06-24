include_directories(${LLVM_MAIN_SRC_DIR}/include/llvm/IR/)

set(LLVM_TARGET_DEFINITIONS IntrinsicsBuddyExt.td)
tablegen(LLVM IntrinsicImpl.inc -gen-intrinsic-impl)
tablegen(LLVM IntrinsicEnums.inc -gen-intrinsic-enums)
tablegen(LLVM IntrinsicsRISCV.h -gen-intrinsic-enums -intrinsic-prefix=riscv)
add_public_tablegen_target(buddy_intrinsics_gen)
