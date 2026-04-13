# Define bufferize options as variables to avoid CMake quote parsing issues
set(BUFFERIZE_FULL_OPTS "unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries")
set(BUFFERIZE_SIMPLE_OPTS "bufferize-function-boundaries")
set(TOSA_PIPELINE "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))")

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
  set(LLC_RISCV_ATTRS "-mcpu=native")
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
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill-bf16.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill-bf16.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/forward_decode-bf16.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode-bf16.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/arg0-bf16.data
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/import-deepseek-r1.py
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}
            --precision bf16
    COMMENT "Generating forward.mlir, subgraph0.mlir and arg0.data..."
  )

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill-w8a32.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill-w8a32.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/forward_decode-w8a32.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode-w8a32.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/arg0-w8a32-f32.data
           ${CMAKE_CURRENT_BINARY_DIR}/arg0-w8a32-i8.data
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/import-deepseek-r1-w8a32.py
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating w8a32 quantized forward.mlir, subgraph0.mlir and param data..."
  )

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill-w8a8.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill-w8a8.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/forward_decode-w8a8.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode-w8a8.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/arg0-w8a8-f32.data
           ${CMAKE_CURRENT_BINARY_DIR}/arg0-w8a8-i8.data
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/import-deepseek-r1-w8a8.py
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating w8a8 quantized forward.mlir, subgraph0.mlir and param data..."
  )

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill-w4a16.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill-w4a16.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/forward_decode-w4a16.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode-w4a16.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/arg0-w4a16-f16.data
           ${CMAKE_CURRENT_BINARY_DIR}/arg0-w4a16-i4packed.data
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/import-deepseek-r1-w4a16.py
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating w4a16 quantized forward.mlir, subgraph0.mlir and param data..."
  )

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill-w8a16.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill-w8a16.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/forward_decode-w8a16.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode-w8a16.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/arg0-w8a16-f16.data
           ${CMAKE_CURRENT_BINARY_DIR}/arg0-w8a16-i8.data
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/import-deepseek-r1-w8a16.py
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating w8a16 quantized forward.mlir, subgraph0.mlir and param data..."
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
            -eliminate-memref-copy
            -assume-tight-memref-layout
            -staticize-memref-layout
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
            -eliminate-memref-copy
            -assume-tight-memref-layout
            -staticize-memref-layout
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
            -eliminate-memref-copy
            -assume-tight-memref-layout
            -staticize-memref-layout
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
            -eliminate-memref-copy
            -assume-tight-memref-layout
            -staticize-memref-layout
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
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph_decode-f16.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode-f16.mlir
    COMMENT "Building subgraph_decode-f16.o "
    VERBATIM)

add_custom_command(
  OUTPUT forward_prefill-bf16.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill-bf16.mlir
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
            -eliminate-memref-copy
            -assume-tight-memref-layout
            -staticize-memref-layout
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
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill-bf16.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill-bf16.mlir
  COMMENT "Building forward_prefill-bf16.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph_prefill-bf16.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill-bf16.mlir
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
            -eliminate-memref-copy
            -assume-tight-memref-layout
            -staticize-memref-layout
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
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph_prefill-bf16.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill-bf16.mlir
    COMMENT "Building subgraph_prefill-bf16.o "
    VERBATIM)

add_custom_command(
  OUTPUT forward_decode-bf16.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_decode-bf16.mlir
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
            -eliminate-memref-copy
            -assume-tight-memref-layout
            -staticize-memref-layout
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
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_decode-bf16.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_decode-bf16.mlir
  COMMENT "Building forward_decode-bf16.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph_decode-bf16.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode-bf16.mlir
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
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph_decode-bf16.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode-bf16.mlir
    COMMENT "Building subgraph_decode-bf16.o "
    VERBATIM)

add_custom_command(
  OUTPUT forward_prefill-w8a32.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill-w8a32.mlir
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
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill-w8a32.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill-w8a32.mlir
  COMMENT "Building forward_prefill-w8a32.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph_prefill-w8a32.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill-w8a32.mlir
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
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph_prefill-w8a32.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill-w8a32.mlir
    COMMENT "Building subgraph_prefill-w8a32.o "
    VERBATIM)

add_custom_command(
  OUTPUT forward_decode-w8a32.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_decode-w8a32.mlir
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
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_decode-w8a32.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_decode-w8a32.mlir
  COMMENT "Building forward_decode-w8a32.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph_decode-w8a32.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode-w8a32.mlir
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
            -dequant-matmul-vectorization-decode=vector-size=32
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
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph_decode-w8a32.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode-w8a32.mlir
    COMMENT "Building subgraph_decode-w8a32.o "
    VERBATIM)

add_custom_command(
  OUTPUT forward_prefill-w4a16.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill-w4a16.mlir
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
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill-w4a16.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill-w4a16.mlir
  COMMENT "Building forward_prefill-w4a16.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph_prefill-w4a16.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill-w4a16.mlir
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
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph_prefill-w4a16.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill-w4a16.mlir
    COMMENT "Building subgraph_prefill-w4a16.o "
    VERBATIM)

add_custom_command(
  OUTPUT forward_decode-w4a16.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_decode-w4a16.mlir
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
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_decode-w4a16.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_decode-w4a16.mlir
  COMMENT "Building forward_decode-w4a16.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph_decode-w4a16.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode-w4a16.mlir
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
            -int4-dequant-matmul-vectorization-decode=vector-size=32
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
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph_decode-w4a16.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode-w4a16.mlir
    COMMENT "Building subgraph_decode-w4a16.o "
    VERBATIM)

add_custom_command(
  OUTPUT forward_prefill-w8a8.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill-w8a8.mlir
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
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill-w8a8.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill-w8a8.mlir
  COMMENT "Building forward_prefill-w8a8.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph_prefill-w8a8.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill-w8a8.mlir
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
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph_prefill-w8a8.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill-w8a8.mlir
    COMMENT "Building subgraph_prefill-w8a8.o "
    VERBATIM)

add_custom_command(
  OUTPUT forward_decode-w8a8.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_decode-w8a8.mlir
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
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_decode-w8a8.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_decode-w8a8.mlir
  COMMENT "Building forward_decode-w8a8.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph_decode-w8a8.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode-w8a8.mlir
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
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph_decode-w8a8.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode-w8a8.mlir
    COMMENT "Building subgraph_decode-w8a8.o "
    VERBATIM)

add_custom_command(
  OUTPUT forward_prefill-w8a16.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill-w8a16.mlir
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
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill-w8a16.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill-w8a16.mlir
  COMMENT "Building forward_prefill-w8a16.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph_prefill-w8a16.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill-w8a16.mlir
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
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph_prefill-w8a16.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill-w8a16.mlir
    COMMENT "Building subgraph_prefill-w8a16.o "
    VERBATIM)

add_custom_command(
  OUTPUT forward_decode-w8a16.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_decode-w8a16.mlir
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
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_decode-w8a16.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_decode-w8a16.mlir
  COMMENT "Building forward_decode-w8a16.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph_decode-w8a16.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode-w8a16.mlir
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
            -dequant-matmul-vectorization-decode=vector-size=32
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
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph_decode-w8a16.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode-w8a16.mlir
    COMMENT "Building subgraph_decode-w8a16.o "
    VERBATIM)

add_library(DEEPSEEKR1 STATIC forward_prefill.o subgraph_prefill.o forward_decode.o subgraph_decode.o)
add_library(DEEPSEEKR1_F16 STATIC forward_prefill-f16.o subgraph_prefill-f16.o forward_decode-f16.o subgraph_decode-f16.o)
add_library(DEEPSEEKR1_BF16 STATIC forward_prefill-bf16.o subgraph_prefill-bf16.o forward_decode-bf16.o subgraph_decode-bf16.o)
add_library(DEEPSEEKR1_W8A32 STATIC forward_prefill-w8a32.o subgraph_prefill-w8a32.o forward_decode-w8a32.o subgraph_decode-w8a32.o)
add_library(DEEPSEEKR1_W4A16 STATIC forward_prefill-w4a16.o subgraph_prefill-w4a16.o forward_decode-w4a16.o subgraph_decode-w4a16.o)
add_library(DEEPSEEKR1_W8A16 STATIC forward_prefill-w8a16.o subgraph_prefill-w8a16.o forward_decode-w8a16.o subgraph_decode-w8a16.o)
add_library(DEEPSEEKR1_W8A8 STATIC forward_prefill-w8a8.o subgraph_prefill-w8a8.o forward_decode-w8a8.o subgraph_decode-w8a8.o)

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

SET_TARGET_PROPERTIES(
  DEEPSEEKR1_W8A32
  PROPERTIES
  LINKER_LANGUAGE C)

SET_TARGET_PROPERTIES(
  DEEPSEEKR1_W8A8
  PROPERTIES
  LINKER_LANGUAGE C)

SET_TARGET_PROPERTIES(
  DEEPSEEKR1_W4A16
  PROPERTIES
  LINKER_LANGUAGE C)

SET_TARGET_PROPERTIES(
  DEEPSEEKR1_W8A16
  PROPERTIES
  LINKER_LANGUAGE C)

add_executable(buddy-deepseek-r1-run buddy-deepseek-r1-main.cpp)
add_executable(buddy-deepseek-r1-f16-run buddy-deepseek-r1-f16-main.cpp)
add_executable(buddy-deepseek-r1-bf16-run buddy-deepseek-r1-bf16-main.cpp)
add_executable(buddy-deepseek-r1-w8a32-run buddy-deepseek-r1-w8a32-main.cpp)
add_executable(buddy-deepseek-r1-w8a8-run buddy-deepseek-r1-w8a8-main.cpp)
add_executable(buddy-deepseek-r1-w4a16-run buddy-deepseek-r1-w4a16-main.cpp)
add_executable(buddy-deepseek-r1-w8a16-run buddy-deepseek-r1-w8a16-main.cpp)
add_executable(buddy-deepseek-r1-cli buddy-deepseek-r1-cli.cpp)
# buddy/runtime/llm/Sampler.h and related headers live under runtime/include
target_include_directories(buddy-deepseek-r1-cli PRIVATE
  ${CMAKE_SOURCE_DIR}/runtime/include)

set(DEEPSEEKR1_EXAMPLE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(DEEPSEEKR1_EXAMPLE_BUILD_PATH ${CMAKE_CURRENT_BINARY_DIR})

target_compile_definitions(buddy-deepseek-r1-run PRIVATE
  DEEPSEEKR1_EXAMPLE_PATH="${DEEPSEEKR1_EXAMPLE_PATH}/"
  DEEPSEEKR1_EXAMPLE_BUILD_PATH="${DEEPSEEKR1_EXAMPLE_BUILD_PATH}/"
)
target_compile_definitions(buddy-deepseek-r1-f16-run PRIVATE
  DEEPSEEKR1_EXAMPLE_PATH="${DEEPSEEKR1_EXAMPLE_PATH}/"
  DEEPSEEKR1_EXAMPLE_BUILD_PATH="${DEEPSEEKR1_EXAMPLE_BUILD_PATH}/"
)
target_compile_definitions(buddy-deepseek-r1-bf16-run PRIVATE
  DEEPSEEKR1_EXAMPLE_PATH="${DEEPSEEKR1_EXAMPLE_PATH}/"
  DEEPSEEKR1_EXAMPLE_BUILD_PATH="${DEEPSEEKR1_EXAMPLE_BUILD_PATH}/"
)
target_compile_definitions(buddy-deepseek-r1-w8a32-run PRIVATE
  DEEPSEEKR1_EXAMPLE_PATH="${DEEPSEEKR1_EXAMPLE_PATH}/"
  DEEPSEEKR1_EXAMPLE_BUILD_PATH="${DEEPSEEKR1_EXAMPLE_BUILD_PATH}/"
)
target_compile_definitions(buddy-deepseek-r1-w8a8-run PRIVATE
  DEEPSEEKR1_EXAMPLE_PATH="${DEEPSEEKR1_EXAMPLE_PATH}/"
  DEEPSEEKR1_EXAMPLE_BUILD_PATH="${DEEPSEEKR1_EXAMPLE_BUILD_PATH}/"
)
target_compile_definitions(buddy-deepseek-r1-w4a16-run PRIVATE
  DEEPSEEKR1_EXAMPLE_PATH="${DEEPSEEKR1_EXAMPLE_PATH}/"
  DEEPSEEKR1_EXAMPLE_BUILD_PATH="${DEEPSEEKR1_EXAMPLE_BUILD_PATH}/"
)
target_compile_definitions(buddy-deepseek-r1-w8a16-run PRIVATE
  DEEPSEEKR1_EXAMPLE_PATH="${DEEPSEEKR1_EXAMPLE_PATH}/"
  DEEPSEEKR1_EXAMPLE_BUILD_PATH="${DEEPSEEKR1_EXAMPLE_BUILD_PATH}/"
)
target_compile_definitions(buddy-deepseek-r1-cli PRIVATE
  DEEPSEEKR1_EXAMPLE_PATH="${DEEPSEEKR1_EXAMPLE_PATH}/"
  DEEPSEEKR1_EXAMPLE_BUILD_PATH="${DEEPSEEKR1_EXAMPLE_BUILD_PATH}/"
)
target_compile_options(buddy-deepseek-r1-cli PRIVATE -fno-rtti)

# Add RISC-V specific compile and link options for the main executable
if(IS_RVV_PLATFORM)
  target_compile_options(buddy-deepseek-r1-run PRIVATE ${RISCV_COMPILE_OPTS})
  target_link_options(buddy-deepseek-r1-run PRIVATE ${RISCV_LINK_OPTS})
endif()

target_link_directories(buddy-deepseek-r1-run PRIVATE ${BUDDY_LLVM_WITH_OPENMP_LINK_DIRS})
target_link_directories(buddy-deepseek-r1-f16-run PRIVATE ${BUDDY_LLVM_WITH_OPENMP_LINK_DIRS})
target_link_directories(buddy-deepseek-r1-bf16-run PRIVATE ${BUDDY_LLVM_WITH_OPENMP_LINK_DIRS})
target_link_directories(buddy-deepseek-r1-w8a32-run PRIVATE ${BUDDY_LLVM_WITH_OPENMP_LINK_DIRS})
target_link_directories(buddy-deepseek-r1-w8a8-run PRIVATE ${BUDDY_LLVM_WITH_OPENMP_LINK_DIRS})
target_link_directories(buddy-deepseek-r1-w4a16-run PRIVATE ${BUDDY_LLVM_WITH_OPENMP_LINK_DIRS})
target_link_directories(buddy-deepseek-r1-w8a16-run PRIVATE ${BUDDY_LLVM_WITH_OPENMP_LINK_DIRS})
target_link_directories(buddy-deepseek-r1-cli PRIVATE ${BUDDY_LLVM_WITH_OPENMP_LINK_DIRS})

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
set(BUDDY_DEEPSEEKR1_W8A32_LIBS
  DEEPSEEKR1_W8A32
  mlir_c_runner_utils
  omp
)
set(BUDDY_DEEPSEEKR1_W8A8_LIBS
  DEEPSEEKR1_W8A8
  mlir_c_runner_utils
  omp
)
set(BUDDY_DEEPSEEKR1_W4A16_LIBS
  DEEPSEEKR1_W4A16
  mlir_c_runner_utils
  omp
)
set(BUDDY_DEEPSEEKR1_W8A16_LIBS
  DEEPSEEKR1_W8A16
  mlir_c_runner_utils
  omp
)
if(BUDDY_MLIR_USE_MIMALLOC)
  list(APPEND BUDDY_DEEPSEEKR1_LIBS mimalloc)
  list(APPEND BUDDY_DEEPSEEKR1_F16_LIBS mimalloc)
  list(APPEND BUDDY_DEEPSEEKR1_BF16_LIBS mimalloc)
  list(APPEND BUDDY_DEEPSEEKR1_W8A32_LIBS mimalloc)
  list(APPEND BUDDY_DEEPSEEKR1_W8A8_LIBS mimalloc)
  list(APPEND BUDDY_DEEPSEEKR1_W4A16_LIBS mimalloc)
  list(APPEND BUDDY_DEEPSEEKR1_W8A16_LIBS mimalloc)
endif()

target_link_libraries(buddy-deepseek-r1-run ${BUDDY_DEEPSEEKR1_LIBS})
target_link_libraries(buddy-deepseek-r1-f16-run ${BUDDY_DEEPSEEKR1_F16_LIBS})
target_link_libraries(buddy-deepseek-r1-bf16-run ${BUDDY_DEEPSEEKR1_BF16_LIBS})
target_link_libraries(buddy-deepseek-r1-w8a32-run ${BUDDY_DEEPSEEKR1_W8A32_LIBS})
target_link_libraries(buddy-deepseek-r1-w8a8-run ${BUDDY_DEEPSEEKR1_W8A8_LIBS})
target_link_libraries(buddy-deepseek-r1-w4a16-run ${BUDDY_DEEPSEEKR1_W4A16_LIBS})
target_link_libraries(buddy-deepseek-r1-w8a16-run ${BUDDY_DEEPSEEKR1_W8A16_LIBS})
target_link_libraries(buddy-deepseek-r1-cli ${BUDDY_DEEPSEEKR1_LIBS} LLVMOption)

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
            ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
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

if(IS_RVV_PLATFORM)
  target_compile_options(buddy-deepseek-r1-tiered-kv-cache-run PRIVATE ${RISCV_COMPILE_OPTS})
  target_link_options(buddy-deepseek-r1-tiered-kv-cache-run PRIVATE ${RISCV_LINK_OPTS})
endif()

target_link_directories(buddy-deepseek-r1-tiered-kv-cache-run PRIVATE ${BUDDY_LLVM_WITH_OPENMP_LINK_DIRS})

# Each decode subgraph object contains a copy of dealloc_helper from MLIR
# bufferization-lower-deallocations; allow multiple definitions when linking.
target_link_options(buddy-deepseek-r1-tiered-kv-cache-run PRIVATE -Wl,--allow-multiple-definition)

set(BUDDY_DEEPSEEKR1_TIERED_KV_CACHE_LIBS
  DEEPSEEKR1_TIERED_KV_CACHE
  mlir_c_runner_utils
  omp
)
if(BUDDY_MLIR_USE_MIMALLOC)
  list(APPEND BUDDY_DEEPSEEKR1_TIERED_KV_CACHE_LIBS mimalloc)
endif()

target_link_libraries(buddy-deepseek-r1-tiered-kv-cache-run ${BUDDY_DEEPSEEKR1_TIERED_KV_CACHE_LIBS})

add_custom_target(deepseekr1-tiered-kv-cache
  DEPENDS buddy-deepseek-r1-tiered-kv-cache-run
  COMMENT "Building DeepSeekR1 tiered-kv-cache inference"
)

message(STATUS "DeepSeekR1 Multi-Cache: Building prefill and decode subgraphs for cache sizes: ${KV_CACHE_SIZES}")
