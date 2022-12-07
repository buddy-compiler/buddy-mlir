include_directories(${LLVM_PROJECT_SOURCE_DIR}/llvm/include/llvm/IR/)

set(LLVM_TARGET_DEFINITIONS IntrinsicsBuddyExt.td)
tablegen(LLVM IntrinsicsRISCV.h -gen-intrinsic-enums -intrinsic-prefix=riscv)
add_public_tablegen_target(buddy_intrinsics_gen)
