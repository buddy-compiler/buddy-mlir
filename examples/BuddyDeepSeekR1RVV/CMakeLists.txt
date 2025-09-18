add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../BuddyDeepSeekR1/import-deepseek-r1.py
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating forward.mlir, subgraph0.mlir and arg0.data..."
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward-f16.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-f16.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/arg0-f16.data
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../BuddyDeepSeekR1/import-deepseek-r1.py
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}
          --precision f16
  COMMENT "Generating forward.mlir, subgraph0.mlir and arg0.data..."
)

add_custom_command(
  OUTPUT forward-f16-rvv.o
  COMMAND
    ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
    ${CMAKE_CURRENT_BINARY_DIR}/forward-f16.mlir
    -pass-pipeline
    "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
    ${BUDDY_BINARY_DIR}/buddy-opt
      -arith-expand
      -eliminate-empty-tensors
      -empty-tensor-to-alloc-tensor
      -one-shot-bufferize="bufferize-function-boundaries"
      -ownership-based-buffer-deallocation
      -buffer-deallocation-simplification
      -bufferization-lower-deallocations
      -matmul-parallel-vectorization-optimize
      -batchmatmul-optimize
      -convert-linalg-to-affine-loops
      -affine-loop-fusion
      -affine-parallelize
      -convert-scf-to-openmp
      -convert-vector-to-scf
      -expand-strided-metadata
      -lower-affine
      -convert-vector-to-llvm
      -memref-expand
      -arith-expand
      -convert-arith-to-llvm
      -finalize-memref-to-llvm
      -convert-scf-to-cf
      -convert-cf-to-llvm
      -llvm-request-c-wrappers
      -convert-openmp-to-llvm
      -convert-arith-to-llvm
      -convert-math-to-llvm
      -convert-math-to-libm
      -convert-func-to-llvm
      -reconcile-unrealized-casts |
    ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
    ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
    ${LLVM_TOOLS_BINARY_DIR}/llc
      -mtriple=riscv64-unknown-elf
      -mattr=+m,+d,+v,+zfh,+zfhmin,+zvfh,+zvfhmin
      -filetype=obj
      -relocation-model=pic
      -O3
      -o ${CMAKE_CURRENT_BINARY_DIR}/forward-f16-rvv.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward-f16.mlir
  COMMENT "Building forward.o "
  VERBATIM
)

add_custom_command(
  OUTPUT subgraph-f16-rvv.o
  COMMAND
    ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
    ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-f16.mlir
    -pass-pipeline
    "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
    ${BUDDY_BINARY_DIR}/buddy-opt
      -convert-elementwise-to-linalg
      -arith-expand
      -eliminate-empty-tensors
      -empty-tensor-to-alloc-tensor
      -one-shot-bufferize="bufferize-function-boundaries"
      -ownership-based-buffer-deallocation
      -buffer-deallocation-simplification
      -bufferization-lower-deallocations
      -matmul-parallel-vectorization-optimize
      -batchmatmul-optimize
      -convert-linalg-to-affine-loops
      -affine-loop-fusion
      -affine-parallelize
      -lower-affine
      -convert-scf-to-openmp
      -func-bufferize-dynamic-offset
      -convert-vector-to-scf
      -expand-strided-metadata
      -lower-affine
      -cse
      -convert-vector-to-llvm
      -memref-expand
      -arith-expand
      -convert-arith-to-llvm
      -finalize-memref-to-llvm
      -convert-scf-to-cf
      -convert-cf-to-llvm
      -llvm-request-c-wrappers
      -convert-openmp-to-llvm
      -convert-arith-to-llvm
      -convert-math-to-llvm
      -convert-math-to-libm
      -convert-func-to-llvm
      -reconcile-unrealized-casts |
    ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
    ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
    ${LLVM_TOOLS_BINARY_DIR}/llc
      -mtriple=riscv64-unknown-elf
      -mattr=+m,+d,+v,+zfh,+zfhmin,+zvfh,+zvfhmin
      -filetype=obj
      -relocation-model=pic
      -O3
      -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph-f16-rvv.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-f16.mlir
  COMMENT "Building subgraph.o "
  VERBATIM
)

add_custom_command(
  OUTPUT forward-rvv.o
  COMMAND
    ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
    ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
    -pass-pipeline
    "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
    ${BUDDY_BINARY_DIR}/buddy-opt
      -arith-expand
      -eliminate-empty-tensors
      -empty-tensor-to-alloc-tensor
      -one-shot-bufferize="bufferize-function-boundaries"
      -ownership-based-buffer-deallocation
      -buffer-deallocation-simplification
      -bufferization-lower-deallocations
      -matmul-parallel-vectorization-optimize
      -batchmatmul-optimize
      -convert-linalg-to-affine-loops
      -affine-loop-fusion
      -affine-parallelize
      -convert-scf-to-openmp
      -convert-vector-to-scf
      -expand-strided-metadata
      -lower-affine
      -convert-vector-to-llvm
      -memref-expand
      -arith-expand
      -convert-arith-to-llvm
      -finalize-memref-to-llvm
      -convert-scf-to-cf
      -convert-cf-to-llvm
      -llvm-request-c-wrappers
      -convert-openmp-to-llvm
      -convert-arith-to-llvm
      -convert-math-to-llvm
      -convert-math-to-libm
      -convert-func-to-llvm
      -reconcile-unrealized-casts |
    ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
    ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
    ${LLVM_TOOLS_BINARY_DIR}/llc
      -mtriple=riscv64-unknown-elf
      -mattr=+m,+d,+v,+zfh,+zfhmin,+zvfh,+zvfhmin
      -filetype=obj
      -relocation-model=pic
      -O3
      -o ${CMAKE_CURRENT_BINARY_DIR}/forward-rvv.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
  COMMENT "Building forward.o "
  VERBATIM
)

add_custom_command(
  OUTPUT subgraph-rvv.o
  COMMAND
    ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
    ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
    -pass-pipeline
    "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
    ${BUDDY_BINARY_DIR}/buddy-opt
      -convert-elementwise-to-linalg
      -arith-expand
      -eliminate-empty-tensors
      -empty-tensor-to-alloc-tensor
      -one-shot-bufferize="bufferize-function-boundaries"
      -ownership-based-buffer-deallocation
      -buffer-deallocation-simplification
      -bufferization-lower-deallocations
      -matmul-parallel-vectorization-optimize
      -batchmatmul-optimize
      -convert-linalg-to-affine-loops
      -affine-loop-fusion
      -affine-parallelize
      -lower-affine
      -convert-scf-to-openmp
      -func-bufferize-dynamic-offset
      -convert-vector-to-scf
      -expand-strided-metadata
      -lower-affine
      -cse
      -convert-vector-to-llvm
      -memref-expand
      -arith-expand
      -convert-arith-to-llvm
      -finalize-memref-to-llvm
      -convert-scf-to-cf
      -convert-cf-to-llvm
      -llvm-request-c-wrappers
      -convert-openmp-to-llvm
      -convert-arith-to-llvm
      -convert-math-to-llvm
      -convert-math-to-libm
      -convert-func-to-llvm
      -reconcile-unrealized-casts |
    ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
    ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
    ${LLVM_TOOLS_BINARY_DIR}/llc
      -mtriple=riscv64-unknown-elf
      -mattr=+m,+d,+v,+zfh,+zfhmin,+zvfh,+zvfhmin
      -filetype=obj
      -relocation-model=pic
      -O3
      -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph-rvv.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
  COMMENT "Building subgraph.o "
  VERBATIM
)

set(BUDDY_MLIR_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../build/)
set(RISCV_GNU_TOOLCHAIN ${BUDDY_MLIR_BUILD_DIR}/thirdparty/riscv-gnu-toolchain)
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)
set(CMAKE_C_COMPILER ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)
set(MLIR_RV_LIB_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../llvm/build-cross-mlir-rv/lib)

add_executable(buddy-deepseek-r1-rvv-run ${CMAKE_CURRENT_SOURCE_DIR}/../BuddyDeepSeekR1/buddy-deepseek-r1-main.cpp forward-rvv.o subgraph-rvv.o)
add_executable(buddy-deepseek-r1-f16-rvv-run ${CMAKE_CURRENT_SOURCE_DIR}/../BuddyDeepSeekR1/buddy-deepseek-r1-f16-main.cpp forward-f16-rvv.o subgraph-f16-rvv.o)

target_compile_definitions(
  buddy-deepseek-r1-f16-rvv-run PRIVATE
  DEEPSEEKR1_EXAMPLE_PATH=""
  DEEPSEEKR1_EXAMPLE_BUILD_PATH=""
)
target_compile_definitions(
  buddy-deepseek-r1-rvv-run PRIVATE
  DEEPSEEKR1_EXAMPLE_PATH=""
  DEEPSEEKR1_EXAMPLE_BUILD_PATH=""
)

target_link_libraries(
  buddy-deepseek-r1-f16-rvv-run PRIVATE
  ${MLIR_RV_LIB_DIR}/libmlir_c_runner_utils.so
  ${MLIR_RV_LIB_DIR}/libmlir_float16_utils.so
  ${MLIR_RV_LIB_DIR}/libomp.so
)
target_link_libraries(
  buddy-deepseek-r1-rvv-run PRIVATE
  ${MLIR_RV_LIB_DIR}/libmlir_c_runner_utils.so
  ${MLIR_RV_LIB_DIR}/libomp.so
)
