set(BUDDY_OPT ../../build/bin/buddy-opt)
set(BUDDY_TRANSLATE ../../build/bin/buddy-translate)
set(BUDDY_LLC ../../build/bin/buddy-llc)
set(LINUX_RISCV_GCC riscv64-unknown-linux-gnu-gcc)

# MLIR + debug_helper.c (bb_test_report), for host-side TEST PASS / TEST FAILED lines.
# Always run -convert-tile-to-buckyball before -lower-buckyball: no-op if the module has no tile ops.
function(add_buckyball_linux_mlir_c STEM)
  set(_in "${CMAKE_CURRENT_SOURCE_DIR}/${STEM}.mlir")
  set(_c "${CMAKE_CURRENT_SOURCE_DIR}/debug_helper.c")
  set(_mid "${CMAKE_CURRENT_BINARY_DIR}/${STEM}.llvm.mlir")
  set(_ll "${CMAKE_CURRENT_BINARY_DIR}/${STEM}.ll")
  set(_o "${CMAKE_CURRENT_BINARY_DIR}/${STEM}.o")
  set(_co "${CMAKE_CURRENT_BINARY_DIR}/${STEM}_debug_helper.o")
  set(_exe "${CMAKE_CURRENT_BINARY_DIR}/${STEM}-linux")

  add_custom_command(
    OUTPUT ${_exe}
    BYPRODUCTS ${_mid} ${_ll} ${_o} ${_co}
    COMMAND ${BUDDY_OPT} ${_in}
      -convert-tile-to-buckyball
      -lower-buckyball-to-bank-ssa
      -assign-physical-banks
      -lower-bank-ssa-to-intrinsics
      -lower-buckyball -expand-strided-metadata -memref-expand
      -convert-to-llvm -reconcile-unrealized-casts -o ${_mid}
    COMMAND ${BUDDY_TRANSLATE} ${_mid} --buddy-to-llvmir -o ${_ll}
    COMMAND ${BUDDY_LLC} ${_ll} -filetype=obj -mtriple=riscv64
      -mattr=+buddyext,+D -float-abi=hard -o ${_o}
    COMMAND ${LINUX_RISCV_GCC} -c ${_c} -o ${_co} -O2
    COMMAND ${LINUX_RISCV_GCC} ${_o} ${_co} -O2 -static -o ${_exe}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ${_in} ${_c}
    COMMENT "BuckyballDialect: ${STEM}.mlir -> ${STEM}-linux"
  )
  add_custom_target(buckyball_${STEM}_linux DEPENDS ${_exe})
endfunction()

add_buckyball_linux_mlir_c(bb_mvin_mvout)
add_buckyball_linux_mlir_c(bb_mul_warp16)
add_buckyball_linux_mlir_c(bb_matmul)
add_buckyball_linux_mlir_c(bb_transpose)
add_buckyball_linux_mlir_c(bb_im2col)
add_buckyball_linux_mlir_c(bb_tile_matmul)

add_custom_target(buckyball-dialect-examples ALL)
add_dependencies(buckyball-dialect-examples
  buckyball_bb_mvin_mvout_linux
  buckyball_bb_mul_warp16_linux
  buckyball_bb_matmul_linux
  buckyball_bb_transpose_linux
  buckyball_bb_im2col_linux
  buckyball_bb_tile_matmul_linux
  )
