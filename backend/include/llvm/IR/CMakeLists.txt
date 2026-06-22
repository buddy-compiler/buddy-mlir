include_directories(${LLVM_MAIN_SRC_DIR}/include/llvm/IR/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Add Buckyball intrinsics include path
get_filename_component(BUCKYBALL_INTRINSICS_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../src/Backend/include"
  ABSOLUTE)
include_directories(${BUCKYBALL_INTRINSICS_DIR})

set(LLVM_TARGET_DEFINITIONS IntrinsicsBuddyExt.td)
tablegen(LLVM IntrinsicImpl.inc -gen-intrinsic-impl EXTRA_INCLUDES ${BUCKYBALL_INTRINSICS_DIR})
tablegen(LLVM IntrinsicEnums.inc -gen-intrinsic-enums EXTRA_INCLUDES ${BUCKYBALL_INTRINSICS_DIR})
tablegen(LLVM IntrinsicsRISCV.h -gen-intrinsic-enums -intrinsic-prefix=riscv EXTRA_INCLUDES ${BUCKYBALL_INTRINSICS_DIR})
add_public_tablegen_target(buddy_intrinsics_gen)
