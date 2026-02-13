# Define bufferize options as variables to avoid CMake quote parsing issues
set(BUFFERIZE_FULL_OPTS "unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries")
set(BUFFERIZE_SIMPLE_OPTS "bufferize-function-boundaries")
set(TOSA_PIPELINE "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))")

if(IS_RVV_CROSSCOMPILING)
set(CMAKE_C_COMPILER ${BUDDY_MLIR_BUILD_DIR}/../llvm/build/bin/clang)
set(CMAKE_CXX_COMPILER ${BUDDY_MLIR_BUILD_DIR}/../llvm/build/bin/clang++)
endif()
# Detect RISC-V Vector (RVV) platform
# Use HAVE_LOCAL_RVV from check_simd.cmake which is already called in the main CMakeLists.txt
if(HAVE_LOCAL_RVV)
  set(IS_RVV_PLATFORM ON)
  message(STATUS "RISC-V Vector (RVV) platform detected")
endif()

# Set platform-specific variables
if(IS_RVV_PLATFORM)
  set(OPENMP_NUM_THREADS "num-threads=32")
  set(LLC_RISCV_ATTRS "-mattr=+m,+d,+v")
  set(RISCV_COMPILE_OPTS "-march=rv64gcv;-mabi=lp64d;-O3;-Wall")
  set(RISCV_LINK_OPTS "-march=rv64gcv;-mabi=lp64d")
else()
  set(OPENMP_NUM_THREADS "num-threads=48")
  set(LLC_RISCV_ATTRS "")
endif()

if(IS_RVV_CROSSCOMPILING)
  message(STATUS "CROSSCOMPILING RISC-V Vector (RVV) platform")
  set(OPENMP_NUM_THREADS "num-threads=8")
  set(LLC_RISCV_ATTRS "-march=riscv64;-mattr=+m,+d,+v")
  set(RISCV_COMPILE_OPTS
      -march=rv64gcv
      --target=riscv64-unknown-linux-gnu
      --sysroot=${RISCV_GNU_TOOLCHAIN}/sysroot
      --gcc-toolchain=${RISCV_GNU_TOOLCHAIN}
      -fPIC
  )
  set(RISCV_LINK_OPTS
    --target=riscv64-unknown-linux-gnu
    --sysroot=${RISCV_GNU_TOOLCHAIN}/sysroot
    --gcc-toolchain=${RISCV_GNU_TOOLCHAIN}
  )
endif()

# Skip import commands on RVV platform
if(NOT IS_RVV_PLATFORM)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/forward_decode.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/import-deepseek-r1.py
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating forward.mlir, subgraph0.mlir and arg0.data..."
  )

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill-f16.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill-f16.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/forward_decode-f16.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode-f16.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/arg0-f16.data
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/import-deepseek-r1.py
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}
            --precision f16
    COMMENT "Generating forward.mlir, subgraph0.mlir and arg0.data..."
  )

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward-bf16.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-bf16.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/arg0-bf16.data
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/import-deepseek-r1.py
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}
            --precision bf16
    COMMENT "Generating forward.mlir, subgraph0.mlir and arg0.data..."
  )
endif()

add_custom_command(
  OUTPUT forward_prefill.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill.mlir
            -simplify-tosa-reshape |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -pass-pipeline ${TOSA_PIPELINE} |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -canonicalize
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -cse
            -canonicalize
            -optimize-allocation-liveness
            -matmul-vectorization-blis
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-parallelize
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp=${OPENMP_NUM_THREADS}
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
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
        ${LLVM_TOOLS_BINARY_DIR}/llc ${LLC_RISCV_ATTRS} -filetype=obj -relocation-model=pic -O3
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill.mlir
  COMMENT "Building forward_prefill.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph_prefill.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill.mlir
              -simplify-tosa-reshape |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
              -pass-pipeline ${TOSA_PIPELINE} |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -convert-elementwise-to-linalg
            -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -canonicalize
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -cse
            -canonicalize
            -optimize-allocation-liveness
            -matmul-vectorization-blis
            -batchmatmul-optimize
            -batchmatmul-transpose-b-vectorization
            -convert-linalg-to-affine-loops
            -affine-parallelize
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp=num-threads=48
            -cse
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
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
          ${LLVM_TOOLS_BINARY_DIR}/llc ${LLC_RISCV_ATTRS} -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph_prefill.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill.mlir
    COMMENT "Building subgraph_prefill.o "
    VERBATIM)

add_custom_command(
  OUTPUT forward_decode.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_decode.mlir
            -simplify-tosa-reshape |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -pass-pipeline ${TOSA_PIPELINE} |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -canonicalize
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -cse
            -canonicalize
            -optimize-allocation-liveness
            -matmul-vectorization-blis
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-parallelize
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp=${OPENMP_NUM_THREADS}
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
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
        ${LLVM_TOOLS_BINARY_DIR}/llc ${LLC_RISCV_ATTRS} -filetype=obj -relocation-model=pic -O3
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_decode.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_decode.mlir
  COMMENT "Building forward_decode.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph_decode.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode.mlir
              -simplify-tosa-reshape
              -simplify-tosa-matmul-scalar |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
              -pass-pipeline ${TOSA_PIPELINE} |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -convert-elementwise-to-linalg
            -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -canonicalize
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -cse
            -canonicalize
            -optimize-allocation-liveness
            -eliminate-memref-copy
            -assume-tight-memref-layout
            -staticize-memref-layout
            -matmul-vectorization-decode=vector-size=32
            -batch-matmul-vectorization-decode=vector-size=128
            -batchmatmul-transpose-b-vectorization=vector-size=16
            -convert-linalg-to-affine-loops
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp=${OPENMP_NUM_THREADS}
            -cse
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
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
          ${LLVM_TOOLS_BINARY_DIR}/llc ${LLC_RISCV_ATTRS} -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph_decode.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode.mlir
    COMMENT "Building subgraph_decode.o "
    VERBATIM)

add_custom_command(
  OUTPUT forward_prefill-f16.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill-f16.mlir
            -simplify-tosa-reshape |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -pass-pipeline ${TOSA_PIPELINE} |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -canonicalize
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -cse
            -canonicalize
            -optimize-allocation-liveness
            -matmul-vectorization-blis
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-parallelize
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp=num-threads=48
            -cse
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
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
        ${LLVM_TOOLS_BINARY_DIR}/llc ${LLC_RISCV_ATTRS} -filetype=obj -relocation-model=pic -O3
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill-f16.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill-f16.mlir
  COMMENT "Building forward_prefill-f16.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph_prefill-f16.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill-f16.mlir
              -simplify-tosa-reshape |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
              -pass-pipeline ${TOSA_PIPELINE} |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -convert-elementwise-to-linalg
            -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -canonicalize
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -cse
            -canonicalize
            -optimize-allocation-liveness
            -matmul-vectorization-blis
            -batchmatmul-optimize
            -batchmatmul-transpose-b-vectorization
            -convert-linalg-to-affine-loops
            -affine-parallelize
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp=num-threads=48
            -cse
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
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
          ${LLVM_TOOLS_BINARY_DIR}/llc ${LLC_RISCV_ATTRS} -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph_prefill-f16.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill-f16.mlir
    COMMENT "Building subgraph_prefill-f16.o "
    VERBATIM)

add_custom_command(
  OUTPUT forward_decode-f16.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_decode-f16.mlir
            -simplify-tosa-reshape |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -pass-pipeline ${TOSA_PIPELINE} |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -canonicalize
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -cse
            -canonicalize
            -optimize-allocation-liveness
            -assume-tight-memref-layout
            -matmul-vectorization-blis
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-parallelize
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp=num-threads=48
            -cse
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
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
        ${LLVM_TOOLS_BINARY_DIR}/llc ${LLC_RISCV_ATTRS} -filetype=obj -relocation-model=pic -O3
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_decode-f16.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_decode-f16.mlir
  COMMENT "Building forward_decode-f16.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph_decode-f16.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode-f16.mlir
              -simplify-tosa-reshape
              -simplify-tosa-matmul-scalar |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
              -pass-pipeline ${TOSA_PIPELINE} |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -convert-elementwise-to-linalg
            -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -canonicalize
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -cse
            -canonicalize
            -optimize-allocation-liveness
            -assume-tight-memref-layout
            -matmul-vectorization-decode=vector-size=128
            -batch-matmul-vectorization-decode=vector-size=128
            -batchmatmul-transpose-b-vectorization
            -convert-linalg-to-affine-loops
            -affine-parallelize
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp=num-threads=48
            -cse
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
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
          ${LLVM_TOOLS_BINARY_DIR}/llc ${LLC_RISCV_ATTRS} -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph_decode-f16.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode-f16.mlir
    COMMENT "Building subgraph_decode-f16.o "
    VERBATIM)

add_custom_command(
  OUTPUT forward-bf16.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/forward-bf16.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize="bufferize-function-boundaries"
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -canonicalize
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -cse
            -canonicalize
            -optimize-allocation-liveness
            -matmul-parallel-vectorization-optimize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-parallelize
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp
            -cse
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
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
        ${LLVM_TOOLS_BINARY_DIR}/llc ${LLC_RISCV_ATTRS} -filetype=obj -relocation-model=pic -O3
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward-bf16.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward-bf16.mlir
  COMMENT "Building forward.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph-bf16.o
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-bf16.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -convert-elementwise-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -canonicalize
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -cse
            -canonicalize
            -optimize-allocation-liveness
            -matmul-parallel-vectorization-optimize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-parallelize
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp
            -cse
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
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
          ${LLVM_TOOLS_BINARY_DIR}/llc ${LLC_RISCV_ATTRS} -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph-bf16.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-bf16.mlir
    COMMENT "Building subgraph.o "
    VERBATIM)

add_library(DEEPSEEKR1 STATIC forward_prefill.o subgraph_prefill.o forward_decode.o subgraph_decode.o)
add_library(DEEPSEEKR1_F16 STATIC forward_prefill-f16.o subgraph_prefill-f16.o forward_decode-f16.o subgraph_decode-f16.o)
add_library(DEEPSEEKR1_BF16 STATIC forward-bf16.o subgraph-bf16.o)

SET_SOURCE_FILES_PROPERTIES(
  template.o
  PROPERTIES
  EXTERNAL_OBJECT true
  GENERATED true)

SET_TARGET_PROPERTIES(
  DEEPSEEKR1
  PROPERTIES
  LINKER_LANGUAGE C)

SET_TARGET_PROPERTIES(
  DEEPSEEKR1_F16
  PROPERTIES
  LINKER_LANGUAGE C)

SET_TARGET_PROPERTIES(
  DEEPSEEKR1_BF16
  PROPERTIES
  LINKER_LANGUAGE C)

add_executable(buddy-deepseek-r1-run buddy-deepseek-r1-main.cpp)
add_executable(buddy-deepseek-r1-f16-run buddy-deepseek-r1-f16-main.cpp)
add_executable(buddy-deepseek-r1-bf16-run buddy-deepseek-r1-bf16-main.cpp)
add_executable(buddy-deepseek-r1-cli buddy-deepseek-r1-cli.cpp)

if(IS_RVV_CROSSCOMPILING)
  set(DEEPSEEKR1_EXAMPLE_PATH ${DEEPSEEKR1_EXAMPLE_PATH})
  set(DEEPSEEKR1_EXAMPLE_BUILD_PATH ${DEEPSEEKR1_EXAMPLE_BUILD_PATH})
else()
  set(DEEPSEEKR1_EXAMPLE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
  set(DEEPSEEKR1_EXAMPLE_BUILD_PATH ${CMAKE_CURRENT_BINARY_DIR})
endif()

target_compile_definitions(buddy-deepseek-r1-run PRIVATE
  DEEPSEEKR1_EXAMPLE_PATH="${DEEPSEEKR1_EXAMPLE_PATH}/"
  DEEPSEEKR1_EXAMPLE_BUILD_PATH="${DEEPSEEKR1_EXAMPLE_BUILD_PATH}/"
)
target_compile_definitions(buddy-deepseek-r1-f16-run PRIVATE
  DEEPSEEKR1_EXAMPLE_PATH="${DEEPSEEKR1_EXAMPLE_PATH}/"
  DEEPSEEKR1_EXAMPLE_BUILD_PATH="${DEEPSEEKR1_EXAMPLE_BUILD_PATH}/"
)
target_compile_definitions(buddy-deepseek-r1-bf16-run PRIVATE
  DEEPSEEKR1_EXAMPLE_PATH="${DEEPSEEKR1_EXAMPLE_PATH}"
  DEEPSEEKR1_EXAMPLE_BUILD_PATH="${DEEPSEEKR1_EXAMPLE_BUILD_PATH}"
)
target_compile_definitions(buddy-deepseek-r1-cli PRIVATE
  DEEPSEEKR1_EXAMPLE_PATH="${DEEPSEEKR1_EXAMPLE_PATH}/"
  DEEPSEEKR1_EXAMPLE_BUILD_PATH="${DEEPSEEKR1_EXAMPLE_BUILD_PATH}/"
)
target_compile_options(buddy-deepseek-r1-cli PRIVATE -fno-rtti)

# Add RISC-V specific compile and link options for the main executable
if(IS_RVV_PLATFORM OR IS_RVV_CROSSCOMPILING)
  target_compile_options(buddy-deepseek-r1-run PRIVATE ${RISCV_COMPILE_OPTS})
  target_compile_options(buddy-deepseek-r1-f16-run PRIVATE ${RISCV_COMPILE_OPTS})
  target_compile_options(buddy-deepseek-r1-bf16-run PRIVATE ${RISCV_COMPILE_OPTS})
  target_compile_options(buddy-deepseek-r1-cli PRIVATE ${RISCV_COMPILE_OPTS})

  target_link_options(buddy-deepseek-r1-run PRIVATE ${RISCV_LINK_OPTS})
  target_link_options(buddy-deepseek-r1-f16-run PRIVATE ${RISCV_LINK_OPTS})
  target_link_options(buddy-deepseek-r1-bf16-run PRIVATE ${RISCV_LINK_OPTS})
  target_link_options(buddy-deepseek-r1-cli PRIVATE ${RISCV_LINK_OPTS})
 endif()

target_link_directories(buddy-deepseek-r1-run PRIVATE ${LLVM_LIBRARY_DIR})
target_link_directories(buddy-deepseek-r1-f16-run PRIVATE ${LLVM_LIBRARY_DIR})
target_link_directories(buddy-deepseek-r1-bf16-run PRIVATE ${LLVM_LIBRARY_DIR})
target_link_directories(buddy-deepseek-r1-cli PRIVATE ${LLVM_LIBRARY_DIR})

if(IS_RVV_CROSSCOMPILING)
  set(BUDDY_DEEPSEEKR1_LIBS
    DEEPSEEKR1
    ${RISCV_MLIR_C_RUNNER_UTILS}
    ${RISCV_OMP_SHARED}
  )
  set(BUDDY_DEEPSEEKR1_F16_LIBS
    DEEPSEEKR1_F16
    ${RISCV_MLIR_C_RUNNER_UTILS}
    ${RISCV_OMP_SHARED}
  )
  set(BUDDY_DEEPSEEKR1_BF16_LIBS
    DEEPSEEKR1_BF16
    ${RISCV_MLIR_C_RUNNER_UTILS}
    ${RISCV_OMP_SHARED}
  )
else()
  set(BUDDY_DEEPSEEKR1_LIBS
    DEEPSEEKR1
    mlir_c_runner_utils
    omp
  )
  set(BUDDY_DEEPSEEKR1_F16_LIBS
    DEEPSEEKR1_F16
    mlir_c_runner_utils
    omp
  )
  set(BUDDY_DEEPSEEKR1_BF16_LIBS
    DEEPSEEKR1_BF16
    mlir_c_runner_utils
    omp
  )
endif()

if(BUDDY_MLIR_USE_MIMALLOC)
  list(APPEND BUDDY_DEEPSEEKR1_LIBS mimalloc)
  list(APPEND BUDDY_DEEPSEEKR1_F16_LIBS mimalloc)
  list(APPEND BUDDY_DEEPSEEKR1_BF16_LIBS mimalloc)
endif()

target_link_libraries(buddy-deepseek-r1-run ${BUDDY_DEEPSEEKR1_LIBS})
target_link_libraries(buddy-deepseek-r1-f16-run ${BUDDY_DEEPSEEKR1_F16_LIBS})
target_link_libraries(buddy-deepseek-r1-bf16-run ${BUDDY_DEEPSEEKR1_BF16_LIBS})
if(IS_RVV_CROSSCOMPILING)
  target_link_libraries(buddy-deepseek-r1-cli
                        ${BUDDY_DEEPSEEKR1_LIBS}
                        ${BUILD_CROSS_MLIR_DIR}/lib/libLLVMOption.a
                        ${BUILD_CROSS_MLIR_DIR}/lib/libLLVMSupport.a
                        ${BUILD_CROSS_MLIR_DIR}/lib/libLLVMDemangle.a)
else()
  target_link_libraries(buddy-deepseek-r1-cli ${BUDDY_DEEPSEEKR1_LIBS} LLVMOption)
endif()


# Create a distributable package
set(DIST_PACKAGE_NAME "buddy-deepseek-r1")
set(DIST_PACKAGE_DIR ${CMAKE_CURRENT_BINARY_DIR}/${DIST_PACKAGE_NAME})
set(DIST_PACKAGE_ARCHIVE ${CMAKE_CURRENT_BINARY_DIR}/${DIST_PACKAGE_NAME}.tar.zst)

add_custom_target(buddy-deepseek-r1-package
  COMMAND ${CMAKE_COMMAND} -E make_directory ${DIST_PACKAGE_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill.mlir
    ${CMAKE_CURRENT_BINARY_DIR}/forward_decode.mlir
    ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill.mlir
    ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode.mlir
    ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
    ${DIST_PACKAGE_DIR}/
  COMMAND sh -c "tar -I 'zstd -T0 -10' -cvf ${DIST_PACKAGE_ARCHIVE} ${DIST_PACKAGE_NAME}"
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${DIST_PACKAGE_DIR}
  COMMENT "Creating distributable package: ${DIST_PACKAGE_ARCHIVE}"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill.mlir
    ${CMAKE_CURRENT_BINARY_DIR}/forward_decode.mlir
    ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill.mlir
    ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode.mlir
    ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
  USES_TERMINAL
  VERBATIM
)

set(R1_RVV_PKG_NAME "buddy-deepseek-r1-rvv-package")
set(R1_RVV_PKG_DIR ${CMAKE_CURRENT_BINARY_DIR}/${R1_RVV_PKG_NAME})
set(R1_RVV_PKG_ARCHIVE ${CMAKE_CURRENT_BINARY_DIR}/${R1_RVV_PKG_NAME}.tgz)
add_custom_target(buddy-deepseek-r1-rvv-package DEPENDS ${R1_RVV_PKG_ARCHIVE})
add_custom_command(
  OUTPUT ${R1_RVV_PKG_ARCHIVE}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${R1_RVV_PKG_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${BUILD_CROSS_MLIR_DIR}/lib/libmlir_float16_utils.so.22.0git
    ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
    ${CMAKE_BINARY_DIR}/bin/buddy-deepseek-r1-run
    ${CMAKE_SOURCE_DIR}/examples/BuddyDeepSeekR1/vocab.txt
    ${RISCV_MLIR_C_RUNNER_UTILS}
    ${RISCV_OMP_SHARED}
    ${R1_RVV_PKG_DIR}/
  COMMAND sh -c "tar -czf ${R1_RVV_PKG_ARCHIVE} ${R1_RVV_PKG_NAME}"
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${R1_RVV_PKG_DIR}
  COMMENT "Creating RVV package: ${R1_RVV_PKG_ARCHIVE}"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
    ${BUILD_CROSS_MLIR_DIR}/lib/libmlir_float16_utils.so.22.0git
    ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
    ${CMAKE_BINARY_DIR}/bin/buddy-deepseek-r1-run
    ${CMAKE_SOURCE_DIR}/examples/BuddyDeepSeekR1/vocab.txt
    ${RISCV_MLIR_C_RUNNER_UTILS}
    ${RISCV_OMP_SHARED}
  USES_TERMINAL
  VERBATIM
)

set(R1F16_RVV_PKG_NAME "buddy-deepseek-r1-f16-rvv-package")
set(R1F16_RVV_PKG_DIR ${CMAKE_CURRENT_BINARY_DIR}/${R1F16_RVV_PKG_NAME})
set(R1F16_RVV_PKG_ARCHIVE ${CMAKE_CURRENT_BINARY_DIR}/${R1F16_RVV_PKG_NAME}.tgz)
add_custom_target(buddy-deepseek-r1-f16-rvv-package DEPENDS ${R1F16_RVV_PKG_ARCHIVE})
add_custom_command(
  OUTPUT ${R1F16_RVV_PKG_ARCHIVE}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${R1F16_RVV_PKG_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${BUILD_CROSS_MLIR_DIR}/lib/libmlir_float16_utils.so.22.0git
    ${CMAKE_CURRENT_BINARY_DIR}/arg0-f16.data
    ${CMAKE_BINARY_DIR}/bin/buddy-deepseek-r1-f16-run
    ${CMAKE_SOURCE_DIR}/examples/BuddyDeepSeekR1/vocab.txt
    ${RISCV_MLIR_C_RUNNER_UTILS}
    ${RISCV_OMP_SHARED}
    ${R1F16_RVV_PKG_DIR}/
  COMMAND sh -c "tar -czf ${R1F16_RVV_PKG_ARCHIVE} ${R1F16_RVV_PKG_NAME}"
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${R1F16_RVV_PKG_DIR}
  COMMENT "Creating RVV package: ${R1F16_RVV_PKG_ARCHIVE}"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
    ${BUILD_CROSS_MLIR_DIR}/lib/libmlir_float16_utils.so.22.0git
    ${CMAKE_CURRENT_BINARY_DIR}/arg0-f16.data
    ${CMAKE_BINARY_DIR}/bin/buddy-deepseek-r1-f16-run
    ${CMAKE_SOURCE_DIR}/examples/BuddyDeepSeekR1/vocab.txt
    ${RISCV_MLIR_C_RUNNER_UTILS}
    ${RISCV_OMP_SHARED}
  USES_TERMINAL
  VERBATIM
)

set(R1BF16_RVV_PKG_NAME "buddy-deepseek-r1-bf16-rvv-package")
set(R1BF16_RVV_PKG_DIR ${CMAKE_CURRENT_BINARY_DIR}/${R1BF16_RVV_PKG_NAME})
set(R1BF16_RVV_PKG_ARCHIVE ${CMAKE_CURRENT_BINARY_DIR}/${R1BF16_RVV_PKG_NAME}.tgz)
add_custom_target(buddy-deepseek-r1-bf16-rvv-package DEPENDS ${R1BF16_RVV_PKG_ARCHIVE})
add_custom_command(
  OUTPUT ${R1BF16_RVV_PKG_ARCHIVE}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${R1BF16_RVV_PKG_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${BUILD_CROSS_MLIR_DIR}/lib/libmlir_float16_utils.so.22.0git
    ${CMAKE_CURRENT_BINARY_DIR}/arg0-bf16.data
    ${CMAKE_BINARY_DIR}/bin/buddy-deepseek-r1-bf16-run
    ${CMAKE_SOURCE_DIR}/examples/BuddyDeepSeekR1/vocab.txt
    ${RISCV_MLIR_C_RUNNER_UTILS}
    ${RISCV_OMP_SHARED}
    ${R1F16_RVV_PKG_DIR}/
  COMMAND sh -c "tar -czf ${R1BF16_RVV_PKG_ARCHIVE} ${R1BF16_RVV_PKG_NAME}"
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${R1BF16_RVV_PKG_DIR}
  COMMENT "Creating RVV package: ${R1BF16_RVV_PKG_ARCHIVE}"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
    ${BUILD_CROSS_MLIR_DIR}/lib/libmlir_float16_utils.so.22.0git
    ${CMAKE_CURRENT_BINARY_DIR}/arg0-bf16.data
    ${CMAKE_BINARY_DIR}/bin/buddy-deepseek-r1-bf16-run
    ${CMAKE_SOURCE_DIR}/examples/BuddyDeepSeekR1/vocab.txt
    ${RISCV_MLIR_C_RUNNER_UTILS}
    ${RISCV_OMP_SHARED}
  USES_TERMINAL
  VERBATIM
)



set(R1CLI_RVV_PKG_NAME "buddy-deepseek-r1-cli-rvv-package")
set(R1CLI_RVV_PKG_DIR ${CMAKE_CURRENT_BINARY_DIR}/${R1CLI_RVV_PKG_NAME})
set(R1CLI_RVV_PKG_ARCHIVE ${CMAKE_CURRENT_BINARY_DIR}/${R1CLI_RVV_PKG_NAME}.tgz)
add_custom_target(buddy-deepseek-r1-cli-rvv-package DEPENDS ${R1CLI_RVV_PKG_ARCHIVE})
add_custom_command(
  OUTPUT ${R1CLI_RVV_PKG_ARCHIVE}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${R1CLI_RVV_PKG_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${BUILD_CROSS_MLIR_DIR}/lib/libmlir_float16_utils.so.22.0git
    ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
    ${CMAKE_BINARY_DIR}/bin/buddy-deepseek-r1-cli
    ${CMAKE_SOURCE_DIR}/examples/BuddyDeepSeekR1/vocab.txt
    ${RISCV_MLIR_C_RUNNER_UTILS}
    ${RISCV_OMP_SHARED}
    ${R1CLI_RVV_PKG_DIR}/
  COMMAND sh -c "tar -czf ${R1CLI_RVV_PKG_ARCHIVE} ${R1CLI_RVV_PKG_NAME}"
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${R1CLI_RVV_PKG_DIR}
  COMMENT "Creating RVV package: ${R1CLI_RVV_PKG_ARCHIVE}"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
    ${BUILD_CROSS_MLIR_DIR}/lib/libmlir_float16_utils.so.22.0git
    ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
    ${CMAKE_BINARY_DIR}/bin/buddy-deepseek-r1-cli
    ${CMAKE_SOURCE_DIR}/examples/BuddyDeepSeekR1/vocab.txt
    ${RISCV_MLIR_C_RUNNER_UTILS}
    ${RISCV_OMP_SHARED}
  USES_TERMINAL
  VERBATIM
)

# Build DeepSeekR1 with multiple KV cache sizes for dynamic cache selection
# Generates both prefill and decode subgraphs for cache sizes: 32, 64, 128, 256, 512, 1024

set(KV_CACHE_SIZES 32 64 128 256 512 1024)

if(NOT IS_RVV_PLATFORM)
  set(TIERED_KV_CACHE_OUTPUTS "")
  foreach(CACHE_SIZE ${KV_CACHE_SIZES})
    list(APPEND TIERED_KV_CACHE_OUTPUTS
      ${CMAKE_CURRENT_BINARY_DIR}/forward_decode_${CACHE_SIZE}.mlir
      ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode_${CACHE_SIZE}.mlir
    )
  endforeach()
  foreach(CACHE_SIZE ${KV_CACHE_SIZES})
    list(APPEND TIERED_KV_CACHE_OUTPUTS
      ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill_${CACHE_SIZE}.mlir
      ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill_${CACHE_SIZE}.mlir
    )
  endforeach()

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/arg0_mc.data
           ${TIERED_KV_CACHE_OUTPUTS}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/import-deepseek-r1-tiered-kv-cache.py
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}
            --cache-sizes "32,64,128,256,512,1024"
            --max-cache-len 1024
    COMMENT "Generating tiered-kv-cache MLIR files (prefill_32/64/128/256/512/1024 + decode_32/64/128/256/512/1024)..."
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )

  add_custom_target(deepseekr1-tiered-kv-cache-import
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/arg0_mc.data
            ${TIERED_KV_CACHE_OUTPUTS}
  )
endif()

foreach(CACHE_SIZE ${KV_CACHE_SIZES})
  add_custom_command(
    OUTPUT forward_decode_${CACHE_SIZE}.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_decode_${CACHE_SIZE}.mlir
              -simplify-tosa-reshape |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
              -pass-pipeline ${TOSA_PIPELINE} |
            ${BUDDY_BINARY_DIR}/buddy-opt
              -eliminate-empty-tensors
              -empty-tensor-to-alloc-tensor
              -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
              -expand-strided-metadata
              -ownership-based-buffer-deallocation
              -buffer-deallocation-simplification
              -bufferization-lower-deallocations
              -matmul-vectorization-blis
              -batchmatmul-optimize
              -convert-linalg-to-affine-loops
              -affine-parallelize
              -convert-vector-to-scf
              -lower-affine
              -convert-scf-to-openmp=${OPENMP_NUM_THREADS}
              -cse
              -memref-expand
              -arith-expand
              -convert-vector-to-llvm
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
          ${LLVM_TOOLS_BINARY_DIR}/llc ${LLC_RISCV_ATTRS} -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/forward_decode_${CACHE_SIZE}.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_decode_${CACHE_SIZE}.mlir
    COMMENT "Building forward_decode_${CACHE_SIZE}.o"
    VERBATIM)
endforeach()

foreach(CACHE_SIZE ${KV_CACHE_SIZES})
  add_custom_command(
    OUTPUT subgraph_decode_${CACHE_SIZE}.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode_${CACHE_SIZE}.mlir
              -simplify-tosa-reshape
              -simplify-tosa-matmul-scalar |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
              -pass-pipeline ${TOSA_PIPELINE} |
            ${BUDDY_BINARY_DIR}/buddy-opt
              -eliminate-empty-tensors
              -empty-tensor-to-alloc-tensor
              -convert-elementwise-to-linalg
              -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
              -expand-strided-metadata
              -ownership-based-buffer-deallocation
              -buffer-deallocation-simplification
              -bufferization-lower-deallocations
              -assume-tight-memref-layout
              -staticize-memref-layout
              -matmul-vectorization-decode=vector-size=128
              -batch-matmul-vectorization-decode=vector-size=128
              -batchmatmul-transpose-b-vectorization=vector-size=16
              -convert-linalg-to-affine-loops
              -convert-vector-to-scf
              -lower-affine
              -convert-scf-to-openmp=${OPENMP_NUM_THREADS}
              -cse
              -memref-expand
              -arith-expand
              -convert-vector-to-llvm
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
            ${LLVM_TOOLS_BINARY_DIR}/llc ${LLC_RISCV_ATTRS} -filetype=obj -relocation-model=pic -O3
              -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph_decode_${CACHE_SIZE}.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode_${CACHE_SIZE}.mlir
    COMMENT "Building subgraph_decode_${CACHE_SIZE}.o"
    VERBATIM)
endforeach()

# Build prefill subgraphs for each cache size
foreach(CACHE_SIZE ${KV_CACHE_SIZES})
  add_custom_command(
    OUTPUT forward_prefill_${CACHE_SIZE}.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill_${CACHE_SIZE}.mlir
              -simplify-tosa-reshape |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
              -pass-pipeline ${TOSA_PIPELINE} |
            ${BUDDY_BINARY_DIR}/buddy-opt
              -eliminate-empty-tensors
              -empty-tensor-to-alloc-tensor
              -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
              -expand-strided-metadata
              -ownership-based-buffer-deallocation
              -buffer-deallocation-simplification
              -bufferization-lower-deallocations
              -matmul-vectorization-blis
              -batchmatmul-optimize
              -convert-linalg-to-affine-loops
              -affine-parallelize
              -convert-vector-to-scf
              -lower-affine
              -convert-scf-to-openmp=${OPENMP_NUM_THREADS}
              -cse
              -memref-expand
              -arith-expand
              -convert-vector-to-llvm
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
          ${LLVM_TOOLS_BINARY_DIR}/llc ${LLC_RISCV_ATTRS} -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill_${CACHE_SIZE}.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill_${CACHE_SIZE}.mlir
    COMMENT "Building forward_prefill_${CACHE_SIZE}.o"
    VERBATIM)
endforeach()

foreach(CACHE_SIZE ${KV_CACHE_SIZES})
  add_custom_command(
    OUTPUT subgraph_prefill_${CACHE_SIZE}.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill_${CACHE_SIZE}.mlir
              -simplify-tosa-reshape |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
              -pass-pipeline ${TOSA_PIPELINE} |
            ${BUDDY_BINARY_DIR}/buddy-opt
              -eliminate-empty-tensors
              -empty-tensor-to-alloc-tensor
              -convert-elementwise-to-linalg
              -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
              -expand-strided-metadata
              -ownership-based-buffer-deallocation
              -buffer-deallocation-simplification
              -bufferization-lower-deallocations
              -matmul-vectorization-blis
              -batchmatmul-optimize
              -batchmatmul-transpose-b-vectorization
              -convert-linalg-to-affine-loops
              -affine-parallelize
              -convert-vector-to-scf
              -lower-affine
              -convert-scf-to-openmp=${OPENMP_NUM_THREADS}
              -cse
              -memref-expand
              -arith-expand
              -convert-vector-to-llvm
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
            ${LLVM_TOOLS_BINARY_DIR}/llc ${LLC_RISCV_ATTRS} -filetype=obj -relocation-model=pic -O3
              -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph_prefill_${CACHE_SIZE}.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill_${CACHE_SIZE}.mlir
    COMMENT "Building subgraph_prefill_${CACHE_SIZE}.o"
    VERBATIM)
endforeach()

set(TIERED_KV_CACHE_DECODE_OBJS "")
foreach(CACHE_SIZE ${KV_CACHE_SIZES})
  list(APPEND TIERED_KV_CACHE_DECODE_OBJS
    ${CMAKE_CURRENT_BINARY_DIR}/forward_decode_${CACHE_SIZE}.o
    ${CMAKE_CURRENT_BINARY_DIR}/subgraph_decode_${CACHE_SIZE}.o
  )
endforeach()

set(TIERED_KV_CACHE_PREFILL_OBJS "")
foreach(CACHE_SIZE ${KV_CACHE_SIZES})
  list(APPEND TIERED_KV_CACHE_PREFILL_OBJS
    ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill_${CACHE_SIZE}.o
    ${CMAKE_CURRENT_BINARY_DIR}/subgraph_prefill_${CACHE_SIZE}.o
  )
endforeach()

add_library(DEEPSEEKR1_TIERED_KV_CACHE STATIC
  ${TIERED_KV_CACHE_PREFILL_OBJS}
  ${TIERED_KV_CACHE_DECODE_OBJS}
)

SET_TARGET_PROPERTIES(
  DEEPSEEKR1_TIERED_KV_CACHE
  PROPERTIES
  LINKER_LANGUAGE C
)

add_executable(buddy-deepseek-r1-tiered-kv-cache-run buddy-deepseek-r1-tiered-kv-cache-main.cpp)

target_compile_definitions(buddy-deepseek-r1-tiered-kv-cache-run PRIVATE
  DEEPSEEKR1_EXAMPLE_PATH="${DEEPSEEKR1_EXAMPLE_PATH}/"
  DEEPSEEKR1_EXAMPLE_BUILD_PATH="${DEEPSEEKR1_EXAMPLE_BUILD_PATH}/"
)

if(IS_RVV_PLATFORM OR IS_RVV_CROSSCOMPILING)
  target_compile_options(buddy-deepseek-r1-tiered-kv-cache-run PRIVATE ${RISCV_COMPILE_OPTS})
  target_link_options(buddy-deepseek-r1-tiered-kv-cache-run PRIVATE ${RISCV_LINK_OPTS})
endif()

target_link_directories(buddy-deepseek-r1-tiered-kv-cache-run PRIVATE ${LLVM_LIBRARY_DIR})

# Each decode subgraph object contains a copy of dealloc_helper from MLIR
# bufferization-lower-deallocations; allow multiple definitions when linking.
target_link_options(buddy-deepseek-r1-tiered-kv-cache-run PRIVATE -Wl,--allow-multiple-definition)

if(IS_RVV_CROSSCOMPILING)
  set(BUDDY_DEEPSEEKR1_TIERED_KV_CACHE_LIBS
    DEEPSEEKR1_TIERED_KV_CACHE
    ${RISCV_MLIR_C_RUNNER_UTILS}
    ${RISCV_OMP_SHARED}
  )
else()
set(BUDDY_DEEPSEEKR1_TIERED_KV_CACHE_LIBS
  DEEPSEEKR1_TIERED_KV_CACHE
  mlir_c_runner_utils
  omp
)
endif()
if(BUDDY_MLIR_USE_MIMALLOC)
  list(APPEND BUDDY_DEEPSEEKR1_TIERED_KV_CACHE_LIBS mimalloc)
endif()

target_link_libraries(buddy-deepseek-r1-tiered-kv-cache-run ${BUDDY_DEEPSEEKR1_TIERED_KV_CACHE_LIBS})

add_custom_target(deepseekr1-tiered-kv-cache
  DEPENDS buddy-deepseek-r1-tiered-kv-cache-run
  COMMENT "Building DeepSeekR1 tiered-kv-cache inference"
)

message(STATUS "DeepSeekR1 Multi-Cache: Building prefill and decode subgraphs for cache sizes: ${KV_CACHE_SIZES}")

set(R1KV_CACHE_RVV_PKG_NAME "buddy-deepseek-r1-tiered-kv-cache-run-rvv-pkg")
set(R1KV_CACHE_RVV_PKG_DIR ${CMAKE_CURRENT_BINARY_DIR}/${R1KV_CACHE_RVV_PKG_NAME})
set(R1KV_CACHE_RVV_PKG_ARCHIVE ${CMAKE_CURRENT_BINARY_DIR}/${R1KV_CACHE_RVV_PKG_NAME}.tgz)
add_custom_target(buddy-deepseek-r1-tiered-kv-cache-run-rvv-package DEPENDS ${R1KV_CACHE_RVV_PKG_ARCHIVE})
add_custom_command(
  OUTPUT ${R1KV_CACHE_RVV_PKG_ARCHIVE}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${R1KV_CACHE_RVV_PKG_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${BUILD_CROSS_MLIR_DIR}/lib/libmlir_float16_utils.so.22.0git
    ${CMAKE_CURRENT_BINARY_DIR}/arg0_mc.data
    ${CMAKE_BINARY_DIR}/bin/buddy-deepseek-r1-tiered-kv-cache-run
    ${CMAKE_SOURCE_DIR}/examples/BuddyDeepSeekR1/vocab.txt
    ${RISCV_MLIR_C_RUNNER_UTILS}
    ${RISCV_OMP_SHARED}
    ${R1KV_CACHE_RVV_PKG_DIR}/
  COMMAND sh -c "tar -czf ${R1KV_CACHE_RVV_PKG_ARCHIVE} ${R1KV_CACHE_RVV_PKG_NAME}"
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${R1KV_CACHE_RVV_PKG_DIR}
  COMMENT "Creating RVV package: ${R1KV_CACHE_RVV_PKG_ARCHIVE}"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
    ${BUILD_CROSS_MLIR_DIR}/lib/libmlir_float16_utils.so.22.0git
    ${CMAKE_CURRENT_BINARY_DIR}/arg0_mc.data
    ${CMAKE_BINARY_DIR}/bin/buddy-deepseek-r1-tiered-kv-cache-run
    ${CMAKE_SOURCE_DIR}/examples/BuddyDeepSeekR1/vocab.txt
    ${RISCV_MLIR_C_RUNNER_UTILS}
    ${RISCV_OMP_SHARED}
  USES_TERMINAL
  VERBATIM
)
