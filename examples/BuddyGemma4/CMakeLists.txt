# Define bufferize options as variables to avoid CMake quote parsing issues
set(BUFFERIZE_FULL_OPTS "unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries")
set(BUFFERIZE_SIMPLE_OPTS "bufferize-function-boundaries")
set(TOSA_PIPELINE "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))")

# Detect RISC-V Vector (RVV) platform
if(HAVE_LOCAL_RVV)
  set(IS_RVV_PLATFORM ON)
  message(STATUS "RISC-V Vector (RVV) platform detected")
endif()

# Cross-compilation support: set SPACEMIT_TOOLCHAIN_ROOT to enable
if(SPACEMIT_TOOLCHAIN_ROOT AND NOT IS_RVV_PLATFORM)
  set(CROSS_COMPILE_RVV ON)
  set(CROSS_CXX "${SPACEMIT_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-g++")
  set(CROSS_SYSROOT "${SPACEMIT_TOOLCHAIN_ROOT}/sysroot")
  set(CROSS_LIBOMP_PATH "${CMAKE_SOURCE_DIR}/llvm/openmp/build-rv64/runtime/src/libomp.a"
    CACHE FILEPATH "Path to cross-compiled libomp.a for RISC-V")
  message(STATUS "Cross-compilation for RVV enabled: ${SPACEMIT_TOOLCHAIN_ROOT}")
endif()

# Set platform-specific variables
if(CROSS_COMPILE_RVV)
  set(OPENMP_NUM_THREADS "num-threads=8")
  set(LLC_TARGET_ATTRS "-mtriple=riscv64-unknown-linux-gnu" "-mattr=+m,+d,+v,+zfh,+zvfh")
elseif(IS_RVV_PLATFORM)
  set(OPENMP_NUM_THREADS "num-threads=8")
  set(LLC_TARGET_ATTRS "-mattr=+m,+d,+v,+zfh,+zvfh")
  set(RISCV_COMPILE_OPTS "-march=rv64gcv_zvfh;-mabi=lp64d;-O3;-Wall")
  set(RISCV_LINK_OPTS "-march=rv64gcv_zvfh;-mabi=lp64d")
else()
  set(OPENMP_NUM_THREADS "num-threads=48")
  set(LLC_TARGET_ATTRS "-mcpu=native")
endif()

# Object file suffix for cross-compilation differentiation
if(CROSS_COMPILE_RVV)
  set(OBJ_SUFFIX "-rvv")
else()
  set(OBJ_SUFFIX "")
endif()

# Object file suffix for cross-compilation differentiation
if(CROSS_COMPILE_RVV)
  set(OBJ_SUFFIX "-rvv")
else()
  set(OBJ_SUFFIX "")
endif()

# Vectorization parameters tuned per platform
if(CROSS_COMPILE_RVV OR IS_RVV_PLATFORM)
  # SpacemiT A100 VLEN=1024: optimize for wider vector registers
  # LMUL=4 is the sweet spot (good throughput without excessive register pressure)
  set(DECODE_VEC_SIZE 128)          # f32: 128 elems × 32bit = 4096bit = LMUL=4
  set(DECODE_VEC_SIZE_F16 256)      # f16: 256 elems × 16bit = 4096bit = LMUL=4
  set(TRANSPOSE_B_VEC_SIZE 32)      # f32: 32 elems × 32bit = 1024bit = LMUL=1
  set(TRANSPOSE_B_VEC_SIZE_F16 64)  # f16: 64 elems × 16bit = 1024bit = LMUL=1
  message(STATUS "RVV vectorization: f32 decode=${DECODE_VEC_SIZE}, f16 decode=${DECODE_VEC_SIZE_F16}, f32 transB=${TRANSPOSE_B_VEC_SIZE}, f16 transB=${TRANSPOSE_B_VEC_SIZE_F16}")
else()
  # x86: keep existing proven defaults
  set(DECODE_VEC_SIZE 128)
  set(DECODE_VEC_SIZE_F16 128)
  set(TRANSPOSE_B_VEC_SIZE 16)
  set(TRANSPOSE_B_VEC_SIZE_F16 16)
endif()

# Skip import commands on RVV platform
if(NOT IS_RVV_PLATFORM)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill_e2b.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill_e2b.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/forward_decode_e2b.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode_e2b.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/arg0_e2b.data
           ${CMAKE_CURRENT_BINARY_DIR}/vocab.txt
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/import-gemma4.py
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}
            --precision f32
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/patch_decode_mlir.py
            ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode_e2b.mlir
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/import-gemma4.py
            ${CMAKE_CURRENT_SOURCE_DIR}/patch_decode_mlir.py
    COMMENT "Generating Gemma4-E2B MLIR and data files..."
  )

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill_e2b-f16.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill_e2b-f16.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/forward_decode_e2b-f16.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode_e2b-f16.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/arg0_e2b-f16.data
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/import-gemma4.py
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}
            --precision f16
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/patch_decode_mlir.py
            ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode_e2b-f16.mlir
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/import-gemma4.py
            ${CMAKE_CURRENT_SOURCE_DIR}/patch_decode_mlir.py
    COMMENT "Generating Gemma4-E2B F16 MLIR and data files..."
  )
endif()

add_custom_command(
  OUTPUT forward_prefill_e2b${OBJ_SUFFIX}.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill_e2b.mlir
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
        ${LLVM_TOOLS_BINARY_DIR}/llc ${LLC_TARGET_ATTRS} -filetype=obj -relocation-model=pic -O3
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill_e2b${OBJ_SUFFIX}.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill_e2b.mlir
  COMMENT "Building forward_prefill_e2b${OBJ_SUFFIX}.o"
  VERBATIM)


add_custom_command(
    OUTPUT subgraph_prefill_e2b${OBJ_SUFFIX}.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill_e2b.mlir
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
            -batchmatmul-transpose-b-vectorization=vector-size=${TRANSPOSE_B_VEC_SIZE}
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
          ${LLVM_TOOLS_BINARY_DIR}/llc ${LLC_TARGET_ATTRS} -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph_prefill_e2b${OBJ_SUFFIX}.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill_e2b.mlir
    COMMENT "Building subgraph_prefill_e2b${OBJ_SUFFIX}.o"
    VERBATIM)

add_custom_command(
  OUTPUT forward_decode_e2b${OBJ_SUFFIX}.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_decode_e2b.mlir
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
        ${LLVM_TOOLS_BINARY_DIR}/llc ${LLC_TARGET_ATTRS} -filetype=obj -relocation-model=pic -O3
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_decode_e2b${OBJ_SUFFIX}.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_decode_e2b.mlir
  COMMENT "Building forward_decode_e2b${OBJ_SUFFIX}.o"
  VERBATIM)

add_custom_command(
    OUTPUT subgraph_decode_e2b${OBJ_SUFFIX}.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode_e2b.mlir
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
            -matmul-vectorization-decode=vector-size=${DECODE_VEC_SIZE}
            -batch-matmul-vectorization-decode=vector-size=${DECODE_VEC_SIZE}
            -batchmatmul-transpose-b-vectorization=vector-size=${TRANSPOSE_B_VEC_SIZE}
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
          ${LLVM_TOOLS_BINARY_DIR}/llc ${LLC_TARGET_ATTRS} -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph_decode_e2b${OBJ_SUFFIX}.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode_e2b.mlir
    COMMENT "Building subgraph_decode_e2b${OBJ_SUFFIX}.o"
    VERBATIM)


add_custom_command(
  OUTPUT forward_prefill_e2b-f16${OBJ_SUFFIX}.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill_e2b-f16.mlir
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
        ${LLVM_TOOLS_BINARY_DIR}/llc ${LLC_TARGET_ATTRS} -filetype=obj -relocation-model=pic -O3
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill_e2b-f16${OBJ_SUFFIX}.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill_e2b-f16.mlir
  COMMENT "Building forward_prefill_e2b-f16${OBJ_SUFFIX}.o"
  VERBATIM)

add_custom_command(
    OUTPUT subgraph_prefill_e2b-f16${OBJ_SUFFIX}.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill_e2b-f16.mlir
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
            -batchmatmul-transpose-b-vectorization=vector-size=${TRANSPOSE_B_VEC_SIZE_F16}
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
          ${LLVM_TOOLS_BINARY_DIR}/llc ${LLC_TARGET_ATTRS} -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph_prefill_e2b-f16${OBJ_SUFFIX}.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill_e2b-f16.mlir
    COMMENT "Building subgraph_prefill_e2b-f16${OBJ_SUFFIX}.o"
    VERBATIM)

add_custom_command(
  OUTPUT forward_decode_e2b-f16${OBJ_SUFFIX}.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_decode_e2b-f16.mlir
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
        ${LLVM_TOOLS_BINARY_DIR}/llc ${LLC_TARGET_ATTRS} -filetype=obj -relocation-model=pic -O3
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_decode_e2b-f16${OBJ_SUFFIX}.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_decode_e2b-f16.mlir
  COMMENT "Building forward_decode_e2b-f16${OBJ_SUFFIX}.o"
  VERBATIM)

add_custom_command(
    OUTPUT subgraph_decode_e2b-f16${OBJ_SUFFIX}.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode_e2b-f16.mlir
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
            -matmul-vectorization-decode=vector-size=${DECODE_VEC_SIZE_F16}
            -batch-matmul-vectorization-decode=vector-size=${DECODE_VEC_SIZE_F16}
            -batchmatmul-transpose-b-vectorization=vector-size=${TRANSPOSE_B_VEC_SIZE_F16}
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
          ${LLVM_TOOLS_BINARY_DIR}/llc ${LLC_TARGET_ATTRS} -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph_decode_e2b-f16${OBJ_SUFFIX}.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode_e2b-f16.mlir
    COMMENT "Building subgraph_decode_e2b-f16${OBJ_SUFFIX}.o"
    VERBATIM)


if(CROSS_COMPILE_RVV)
  # ============================================================
  # Cross-compilation: build RISC-V executables on x86 host
  # ============================================================

  set(CROSS_MARCH "-march=rv64gcv_zvfh")
  set(CROSS_MABI "-mabi=lp64d")
  set(CRUNNERUTILS_SRC "${CMAKE_SOURCE_DIR}/llvm/mlir/lib/ExecutionEngine/CRunnerUtils.cpp")
  set(MLIR_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/llvm/mlir/include")
  set(BUDDY_INTERFACES_DIR "${CMAKE_SOURCE_DIR}/frontend/Interfaces")

  # Cross-compile CRunnerUtils.cpp (provides memrefCopy)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/CRunnerUtils.o
    COMMAND ${CROSS_CXX}
      ${CROSS_MARCH} ${CROSS_MABI} -O3
      --sysroot=${CROSS_SYSROOT}
      -I${MLIR_INCLUDE_DIR}
      -DMLIR_CRUNNERUTILS_DEFINE_FUNCTIONS
      -std=c++17 -fPIC
      -c ${CRUNNERUTILS_SRC}
      -o ${CMAKE_CURRENT_BINARY_DIR}/CRunnerUtils.o
    DEPENDS ${CRUNNERUTILS_SRC}
    COMMENT "Cross-compiling CRunnerUtils.cpp for RISC-V"
    VERBATIM)

  # Cross-compile f32 main
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/buddy-gemma4-e2b-main.o
    COMMAND ${CROSS_CXX}
      ${CROSS_MARCH} ${CROSS_MABI} -O3
      --sysroot=${CROSS_SYSROOT}
      -I${BUDDY_INTERFACES_DIR}
      "-DGEMMA4_E2B_EXAMPLE_PATH=\"./\""
      "-DGEMMA4_E2B_EXAMPLE_BUILD_PATH=\"./\""
      -std=c++17 -fPIC
      -c ${CMAKE_CURRENT_SOURCE_DIR}/buddy-gemma4-e2b-main.cpp
      -o ${CMAKE_CURRENT_BINARY_DIR}/buddy-gemma4-e2b-main.o
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buddy-gemma4-e2b-main.cpp
    COMMENT "Cross-compiling buddy-gemma4-e2b-main.cpp for RISC-V"
    VERBATIM)

  # Cross-compile f16 main
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/buddy-gemma4-e2b-f16-main.o
    COMMAND ${CROSS_CXX}
      ${CROSS_MARCH} ${CROSS_MABI} -O3
      --sysroot=${CROSS_SYSROOT}
      -I${BUDDY_INTERFACES_DIR}
      "-DGEMMA4_E2B_EXAMPLE_PATH=\"./\""
      "-DGEMMA4_E2B_EXAMPLE_BUILD_PATH=\"./\""
      -std=c++17 -fPIC
      -c ${CMAKE_CURRENT_SOURCE_DIR}/buddy-gemma4-e2b-f16-main.cpp
      -o ${CMAKE_CURRENT_BINARY_DIR}/buddy-gemma4-e2b-f16-main.o
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buddy-gemma4-e2b-f16-main.cpp
    COMMENT "Cross-compiling buddy-gemma4-e2b-f16-main.cpp for RISC-V"
    VERBATIM)

  # Link f32 executable for RISC-V
  add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/bin/buddy-gemma4-e2b-run
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin
    COMMAND ${CROSS_CXX}
      ${CROSS_MARCH} ${CROSS_MABI}
      --sysroot=${CROSS_SYSROOT}
      ${CMAKE_CURRENT_BINARY_DIR}/buddy-gemma4-e2b-main.o
      ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill_e2b${OBJ_SUFFIX}.o
      ${CMAKE_CURRENT_BINARY_DIR}/subgraph_prefill_e2b${OBJ_SUFFIX}.o
      ${CMAKE_CURRENT_BINARY_DIR}/forward_decode_e2b${OBJ_SUFFIX}.o
      ${CMAKE_CURRENT_BINARY_DIR}/subgraph_decode_e2b${OBJ_SUFFIX}.o
      ${CMAKE_CURRENT_BINARY_DIR}/CRunnerUtils.o
      ${CROSS_LIBOMP_PATH}
      -lm -lpthread -ldl
      -o ${CMAKE_BINARY_DIR}/bin/buddy-gemma4-e2b-run
    DEPENDS
      ${CMAKE_CURRENT_BINARY_DIR}/buddy-gemma4-e2b-main.o
      forward_prefill_e2b${OBJ_SUFFIX}.o
      subgraph_prefill_e2b${OBJ_SUFFIX}.o
      forward_decode_e2b${OBJ_SUFFIX}.o
      subgraph_decode_e2b${OBJ_SUFFIX}.o
      ${CMAKE_CURRENT_BINARY_DIR}/CRunnerUtils.o
      ${CROSS_LIBOMP_PATH}
    COMMENT "Linking buddy-gemma4-e2b-run for RISC-V"
    VERBATIM)

  add_custom_target(buddy-gemma4-e2b-run
    DEPENDS ${CMAKE_BINARY_DIR}/bin/buddy-gemma4-e2b-run)

  # Link f16 executable for RISC-V
  add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/bin/buddy-gemma4-e2b-f16-run
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin
    COMMAND ${CROSS_CXX}
      ${CROSS_MARCH} ${CROSS_MABI}
      --sysroot=${CROSS_SYSROOT}
      ${CMAKE_CURRENT_BINARY_DIR}/buddy-gemma4-e2b-f16-main.o
      ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill_e2b-f16${OBJ_SUFFIX}.o
      ${CMAKE_CURRENT_BINARY_DIR}/subgraph_prefill_e2b-f16${OBJ_SUFFIX}.o
      ${CMAKE_CURRENT_BINARY_DIR}/forward_decode_e2b-f16${OBJ_SUFFIX}.o
      ${CMAKE_CURRENT_BINARY_DIR}/subgraph_decode_e2b-f16${OBJ_SUFFIX}.o
      ${CMAKE_CURRENT_BINARY_DIR}/CRunnerUtils.o
      ${CROSS_LIBOMP_PATH}
      -lm -lpthread -ldl
      -o ${CMAKE_BINARY_DIR}/bin/buddy-gemma4-e2b-f16-run
    DEPENDS
      ${CMAKE_CURRENT_BINARY_DIR}/buddy-gemma4-e2b-f16-main.o
      forward_prefill_e2b-f16${OBJ_SUFFIX}.o
      subgraph_prefill_e2b-f16${OBJ_SUFFIX}.o
      forward_decode_e2b-f16${OBJ_SUFFIX}.o
      subgraph_decode_e2b-f16${OBJ_SUFFIX}.o
      ${CMAKE_CURRENT_BINARY_DIR}/CRunnerUtils.o
      ${CROSS_LIBOMP_PATH}
    COMMENT "Linking buddy-gemma4-e2b-f16-run for RISC-V"
    VERBATIM)

  add_custom_target(buddy-gemma4-e2b-f16-run
    DEPENDS ${CMAKE_BINARY_DIR}/bin/buddy-gemma4-e2b-f16-run)

  # Cross-compilation deployment package
  add_custom_target(buddy-gemma4-e2b-cross-package
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/deploy
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${CMAKE_BINARY_DIR}/bin/buddy-gemma4-e2b-run
      ${CMAKE_BINARY_DIR}/bin/buddy-gemma4-e2b-f16-run
      ${CMAKE_CURRENT_BINARY_DIR}/arg0_e2b.data
      ${CMAKE_CURRENT_BINARY_DIR}/arg0_e2b-f16.data
      ${CMAKE_CURRENT_BINARY_DIR}/vocab.txt
      ${CMAKE_CURRENT_BINARY_DIR}/deploy/
    COMMENT "Creating cross-compilation deployment package"
    DEPENDS
      ${CMAKE_BINARY_DIR}/bin/buddy-gemma4-e2b-run
      ${CMAKE_BINARY_DIR}/bin/buddy-gemma4-e2b-f16-run
    VERBATIM)

else()
  # ============================================================
  # Native build (x86 or RISC-V device)
  # ============================================================

  add_library(GEMMA4_E2B STATIC forward_prefill_e2b${OBJ_SUFFIX}.o subgraph_prefill_e2b${OBJ_SUFFIX}.o forward_decode_e2b${OBJ_SUFFIX}.o subgraph_decode_e2b${OBJ_SUFFIX}.o)

  add_library(GEMMA4_E2B_F16 STATIC forward_prefill_e2b-f16${OBJ_SUFFIX}.o subgraph_prefill_e2b-f16${OBJ_SUFFIX}.o forward_decode_e2b-f16${OBJ_SUFFIX}.o subgraph_decode_e2b-f16${OBJ_SUFFIX}.o)

  SET_SOURCE_FILES_PROPERTIES(
    template.o
    PROPERTIES
    EXTERNAL_OBJECT true
    GENERATED true)

  SET_TARGET_PROPERTIES(
    GEMMA4_E2B
    PROPERTIES
    LINKER_LANGUAGE C)

  SET_SOURCE_FILES_PROPERTIES(
    template-f16.o
    PROPERTIES
    EXTERNAL_OBJECT true
    GENERATED true)

  SET_TARGET_PROPERTIES(
    GEMMA4_E2B_F16
    PROPERTIES
    LINKER_LANGUAGE C)

  # Create a distributable package (for transferring .mlir files to RVV device).
  set(DIST_PACKAGE_NAME "buddy-gemma4-e2b")
  set(DIST_PACKAGE_DIR ${CMAKE_CURRENT_BINARY_DIR}/${DIST_PACKAGE_NAME})
  set(DIST_PACKAGE_ARCHIVE ${CMAKE_CURRENT_BINARY_DIR}/${DIST_PACKAGE_NAME}.tar.zst)

  add_custom_target(buddy-gemma4-e2b-package
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DIST_PACKAGE_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill_e2b.mlir
      ${CMAKE_CURRENT_BINARY_DIR}/forward_decode_e2b.mlir
      ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill_e2b.mlir
      ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode_e2b.mlir
      ${CMAKE_CURRENT_BINARY_DIR}/arg0_e2b.data
      ${CMAKE_CURRENT_BINARY_DIR}/vocab.txt
      ${DIST_PACKAGE_DIR}/
    COMMAND sh -c "tar -I 'zstd -T0 -10' -cvf ${DIST_PACKAGE_ARCHIVE} ${DIST_PACKAGE_NAME}"
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${DIST_PACKAGE_DIR}
    COMMENT "Creating distributable package: ${DIST_PACKAGE_ARCHIVE}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS
      ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill_e2b.mlir
      ${CMAKE_CURRENT_BINARY_DIR}/forward_decode_e2b.mlir
      ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill_e2b.mlir
      ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode_e2b.mlir
      ${CMAKE_CURRENT_BINARY_DIR}/arg0_e2b.data
      ${CMAKE_CURRENT_BINARY_DIR}/vocab.txt
    USES_TERMINAL
    VERBATIM
  )

  add_executable(buddy-gemma4-e2b-run buddy-gemma4-e2b-main.cpp)

  set(GEMMA4_E2B_EXAMPLE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
  set(GEMMA4_E2B_EXAMPLE_BUILD_PATH ${CMAKE_CURRENT_BINARY_DIR})

  target_compile_definitions(buddy-gemma4-e2b-run PRIVATE
    GEMMA4_E2B_EXAMPLE_PATH="${GEMMA4_E2B_EXAMPLE_PATH}/"
    GEMMA4_E2B_EXAMPLE_BUILD_PATH="${GEMMA4_E2B_EXAMPLE_BUILD_PATH}/"
  )

  if(IS_RVV_PLATFORM)
    target_compile_options(buddy-gemma4-e2b-run PRIVATE ${RISCV_COMPILE_OPTS})
    target_link_options(buddy-gemma4-e2b-run PRIVATE ${RISCV_LINK_OPTS})
  endif()

  target_link_directories(buddy-gemma4-e2b-run PRIVATE ${LLVM_LIBRARY_DIR})

  set(BUDDY_GEMMA4_E2B_LIBS
    GEMMA4_E2B
    mlir_c_runner_utils
    omp
  )

  if(BUDDY_MLIR_USE_MIMALLOC)
    list(APPEND BUDDY_GEMMA4_E2B_LIBS mimalloc)
  endif()

  target_link_libraries(buddy-gemma4-e2b-run ${BUDDY_GEMMA4_E2B_LIBS})

  # ---- F16 Executable ----
  add_executable(buddy-gemma4-e2b-f16-run buddy-gemma4-e2b-f16-main.cpp)

  target_compile_definitions(buddy-gemma4-e2b-f16-run PRIVATE
    GEMMA4_E2B_EXAMPLE_PATH="${GEMMA4_E2B_EXAMPLE_PATH}/"
    GEMMA4_E2B_EXAMPLE_BUILD_PATH="${GEMMA4_E2B_EXAMPLE_BUILD_PATH}/"
  )

  if(IS_RVV_PLATFORM)
    target_compile_options(buddy-gemma4-e2b-f16-run PRIVATE ${RISCV_COMPILE_OPTS})
    target_link_options(buddy-gemma4-e2b-f16-run PRIVATE ${RISCV_LINK_OPTS})
  endif()

  target_link_directories(buddy-gemma4-e2b-f16-run PRIVATE ${LLVM_LIBRARY_DIR})

  set(BUDDY_GEMMA4_E2B_F16_LIBS
    GEMMA4_E2B_F16
    mlir_c_runner_utils
    omp
  )

  if(BUDDY_MLIR_USE_MIMALLOC)
    list(APPEND BUDDY_GEMMA4_E2B_F16_LIBS mimalloc)
  endif()

  target_link_libraries(buddy-gemma4-e2b-f16-run ${BUDDY_GEMMA4_E2B_F16_LIBS})

endif()  # CROSS_COMPILE_RVV
