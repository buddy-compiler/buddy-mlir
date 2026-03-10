# ==============================================================================
# Options / basic config
# ==============================================================================
option(CROSS_COMPILE_RVV "Build for RISC-V RVV target" OFF)

# ==============================================================================
# Configurable variables (default from environment, overridable by -D)
# ==============================================================================
set(BUDDY_MLIR_BUILD_DIR "$ENV{BUDDY_MLIR_BUILD_DIR}" CACHE PATH
  "Buddy MLIR host build directory (contains bin/buddy-opt)")
set(LLVM_MLIR_BUILD_DIR "$ENV{LLVM_MLIR_BUILD_DIR}" CACHE PATH
  "LLVM/MLIR host build directory (contains bin/mlir-opt, mlir-translate, llvm-as, llc)")
set(PYTHONPATH "$ENV{PYTHONPATH}" CACHE STRING
  "Python module search path")
set(BUDDY_MLIR_BUILD_CROSS_DIR "$ENV{BUDDY_MLIR_BUILD_CROSS_DIR}" CACHE PATH
  "Buddy MLIR cross build directory (contains target libraries in lib/)")
set(RISCV_GNU_TOOLCHAIN "$ENV{RISCV_GNU_TOOLCHAIN}" CACHE PATH
  "RISC-V GNU toolchain root")
set(RISCV_SYSROOT "$ENV{RISCV_SYSROOT}" CACHE PATH
  "RISC-V sysroot root path (defaults to ${RISCV_GNU_TOOLCHAIN}/sysroot when CROSS_COMPILE_RVV=ON)")
set(RISCV_OMP_SHARED "$ENV{RISCV_OMP_SHARED}" CACHE FILEPATH
  "Full path to target OpenMP shared library for RISC-V")
# Runtime path macros used by the final executable.
set(DSTP_EXAMPLE_PATH "$ENV{DSTP_EXAMPLE_PATH}" CACHE STRING
  "Runtime path prefix on target device")
set(DSTP_EXAMPLE_BUILD_PATH "$ENV{DSTP_EXAMPLE_BUILD_PATH}" CACHE STRING
  "Runtime build path prefix on target device")

# MPICH is built automatically under current binary dir.
set(MPICH_VERSION "4.3.1" CACHE STRING
  "MPICH version to build")
set(MPICH_URL "https://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz" CACHE STRING
  "MPICH source tarball URL")
set(MPICH_JOBS "4" CACHE STRING
  "Parallel build jobs for MPICH")
set(MPICH_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/mpich-install" CACHE PATH
  "MPICH installation directory")
set(MPICH_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/mpich-src" CACHE PATH
  "MPICH source directory")
set(MPICH_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/mpich-build" CACHE PATH
  "MPICH build directory")

# ==============================================================================
# Fixed options / pipelines
# ==============================================================================
set(BUFFERIZE_FULL_OPTS
  "unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries")
set(BUFFERIZE_SIMPLE_OPTS
  "bufferize-function-boundaries")
set(TOSA_PIPELINE
  "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))")

# Avoid CMake compiling test executables during configure.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

include(ExternalProject)
include(CMakeParseArguments)

# ==============================================================================
# Resolve build tools (host tools)
# ==============================================================================
if(NOT BUDDY_MLIR_BUILD_DIR STREQUAL "")
  set(BUDDY_TOOLS_BINARY_DIR "${BUDDY_MLIR_BUILD_DIR}/bin")
else()
  set(BUDDY_TOOLS_BINARY_DIR "${BUDDY_BINARY_DIR}")
endif()

if(NOT EXISTS "${BUDDY_TOOLS_BINARY_DIR}/buddy-opt")
  message(FATAL_ERROR "buddy-opt not found: ${BUDDY_TOOLS_BINARY_DIR}/buddy-opt")
endif()

if(NOT LLVM_MLIR_BUILD_DIR STREQUAL "")
  set(LLVM_TOOLS_BIN_DIR "${LLVM_MLIR_BUILD_DIR}/bin")
else()
  set(LLVM_TOOLS_BIN_DIR "${LLVM_TOOLS_BINARY_DIR}")
endif()

if(NOT EXISTS "${LLVM_TOOLS_BIN_DIR}/mlir-opt")
  message(FATAL_ERROR "mlir-opt not found: ${LLVM_TOOLS_BIN_DIR}/mlir-opt")
endif()
if(NOT EXISTS "${LLVM_TOOLS_BIN_DIR}/mlir-translate")
  message(FATAL_ERROR "mlir-translate not found: ${LLVM_TOOLS_BIN_DIR}/mlir-translate")
endif()
if(NOT EXISTS "${LLVM_TOOLS_BIN_DIR}/llvm-as")
  message(FATAL_ERROR "llvm-as not found: ${LLVM_TOOLS_BIN_DIR}/llvm-as")
endif()
if(NOT EXISTS "${LLVM_TOOLS_BIN_DIR}/llc")
  message(FATAL_ERROR "llc not found: ${LLVM_TOOLS_BIN_DIR}/llc")
endif()

set(BUDDY_OPT_DEP)
if(BUDDY_MLIR_BUILD_DIR STREQUAL "${CMAKE_BINARY_DIR}")
  set(BUDDY_OPT_DEP buddy-opt)
endif()

# ==============================================================================
# Platform-specific config
# ==============================================================================
if(CROSS_COMPILE_RVV)
  set(OPENMP_NUM_THREADS "num-threads=8")
  set(LLC_TARGET_FLAGS
    -mtriple=riscv64-unknown-linux-gnu
    -target-abi=lp64d
    -mattr=+m,+d,+v
  )

  set(CMAKE_SYSTEM_NAME Linux)
  set(CMAKE_SYSTEM_PROCESSOR riscv64)

  if(RISCV_GNU_TOOLCHAIN STREQUAL "")
    message(FATAL_ERROR
      "CROSS_COMPILE_RVV=ON requires RISCV_GNU_TOOLCHAIN. "
      "Please export it or pass -DRISCV_GNU_TOOLCHAIN=...")
  endif()

  if(RISCV_SYSROOT STREQUAL "")
    set(RISCV_SYSROOT "${RISCV_GNU_TOOLCHAIN}/sysroot")
  endif()
  set(CMAKE_SYSROOT "${RISCV_SYSROOT}")

  if(NOT EXISTS "${RISCV_SYSROOT}")
    message(FATAL_ERROR "RISCV sysroot not found: ${RISCV_SYSROOT}")
  endif()

  if(BUDDY_MLIR_BUILD_CROSS_DIR STREQUAL "")
    message(FATAL_ERROR
      "CROSS_COMPILE_RVV=ON requires BUDDY_MLIR_BUILD_CROSS_DIR. "
      "Please export it or pass -DBUDDY_MLIR_BUILD_CROSS_DIR=...")
  endif()

  if(RISCV_OMP_SHARED STREQUAL "")
    message(FATAL_ERROR
      "CROSS_COMPILE_RVV=ON requires RISCV_OMP_SHARED. "
      "Please export it or pass -DRISCV_OMP_SHARED=...")
  endif()

  set(CMAKE_FIND_ROOT_PATH
    "${RISCV_SYSROOT}"
    "${MPICH_INSTALL_DIR}"
    "${BUDDY_MLIR_BUILD_CROSS_DIR}"
  )

  set(RISCV_COMMON_FLAGS
    "--sysroot=${RISCV_SYSROOT} -march=rv64gcv -mabi=lp64d -fPIC")

  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${RISCV_COMMON_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${RISCV_COMMON_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${RISCV_COMMON_FLAGS}")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${RISCV_COMMON_FLAGS}")

  set(OPENMP_LIB "${RISCV_OMP_SHARED}")
  set(MLIR_RUNNER_UTILS_LIB
    "${BUDDY_MLIR_BUILD_CROSS_DIR}/lib/libStaticMLIRCRunnerUtils.a")

  set(MPICH_INCLUDE_DIR "${MPICH_INSTALL_DIR}/include")
  set(MPI_C_LIB "${MPICH_INSTALL_DIR}/lib/libmpi.so")
  set(MPI_CXX_LIB "${MPICH_INSTALL_DIR}/lib/libmpicxx.so")

  set(MPICH_CC "${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc")
  set(MPICH_CXX "${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++")

  if(NOT EXISTS "${MPICH_CC}")
    message(FATAL_ERROR "RISC-V gcc not found: ${MPICH_CC}")
  endif()
  if(NOT EXISTS "${MPICH_CXX}")
    message(FATAL_ERROR "RISC-V g++ not found: ${MPICH_CXX}")
  endif()

  set(MPICH_CONFIGURE_COMMAND
    <SOURCE_DIR>/configure
    CC=${MPICH_CC}
    CXX=${MPICH_CXX}
    CFLAGS=--sysroot=${RISCV_SYSROOT}\ -march=rv64gcv\ -mabi=lp64d\ -fPIC
    CXXFLAGS=--sysroot=${RISCV_SYSROOT}\ -march=rv64gcv\ -mabi=lp64d\ -fPIC
    LDFLAGS=--sysroot=${RISCV_SYSROOT}\ -march=rv64gcv\ -mabi=lp64d
    --host=riscv64-unknown-linux-gnu
    --disable-fortran
    --prefix=${MPICH_INSTALL_DIR}
  )

  if(DSTP_EXAMPLE_PATH STREQUAL "")
    set(DSTP_EXAMPLE_PATH ".")
  endif()
  if(DSTP_EXAMPLE_BUILD_PATH STREQUAL "")
    set(DSTP_EXAMPLE_BUILD_PATH ".")
  endif()

else()
  set(OPENMP_NUM_THREADS "num-threads=48")
  set(LLC_TARGET_FLAGS "-mcpu=native")

  set(OPENMP_LIB omp)
  set(MLIR_RUNNER_UTILS_LIB mlir_c_runner_utils)

  set(MPICH_INCLUDE_DIR "${MPICH_INSTALL_DIR}/include")
  set(MPI_C_LIB "${MPICH_INSTALL_DIR}/lib/libmpi.so")
  set(MPI_CXX_LIB "${MPICH_INSTALL_DIR}/lib/libmpicxx.so")

  set(MPICH_CONFIGURE_COMMAND
    <SOURCE_DIR>/configure
    --disable-fortran
    --prefix=${MPICH_INSTALL_DIR}
  )

  set(DSTP_EXAMPLE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
  set(DSTP_EXAMPLE_BUILD_PATH ${CMAKE_CURRENT_BINARY_DIR})
endif()

# ==============================================================================
# Common checks
# ==============================================================================
if(CROSS_COMPILE_RVV)
  if(NOT OPENMP_LIB)
    message(FATAL_ERROR "RISC-V OpenMP shared library not found")
  endif()
  if(NOT EXISTS "${OPENMP_LIB}")
    message(FATAL_ERROR "RISC-V OpenMP shared library does not exist: ${OPENMP_LIB}")
  endif()

  if(NOT MLIR_RUNNER_UTILS_LIB)
    message(FATAL_ERROR "RISC-V static MLIR runner utils library not found")
  endif()
  if(NOT EXISTS "${MLIR_RUNNER_UTILS_LIB}")
    message(FATAL_ERROR "RISC-V static MLIR runner utils library does not exist: ${MLIR_RUNNER_UTILS_LIB}")
  endif()
endif()

message(STATUS "CROSS_COMPILE_RVV          = ${CROSS_COMPILE_RVV}")
message(STATUS "BUDDY_MLIR_BUILD_DIR       = ${BUDDY_MLIR_BUILD_DIR}")
message(STATUS "LLVM_MLIR_BUILD_DIR        = ${LLVM_MLIR_BUILD_DIR}")
message(STATUS "BUDDY_TOOLS_BINARY_DIR     = ${BUDDY_TOOLS_BINARY_DIR}")
message(STATUS "LLVM_TOOLS_BIN_DIR         = ${LLVM_TOOLS_BIN_DIR}")
message(STATUS "PYTHONPATH                 = ${PYTHONPATH}")
message(STATUS "BUDDY_MLIR_BUILD_CROSS_DIR = ${BUDDY_MLIR_BUILD_CROSS_DIR}")
message(STATUS "RISCV_GNU_TOOLCHAIN        = ${RISCV_GNU_TOOLCHAIN}")
message(STATUS "RISCV_SYSROOT              = ${RISCV_SYSROOT}")
message(STATUS "RISCV_OMP_SHARED           = ${RISCV_OMP_SHARED}")
message(STATUS "OPENMP_LIB                 = ${OPENMP_LIB}")
message(STATUS "MLIR_RUNNER_UTILS_LIB      = ${MLIR_RUNNER_UTILS_LIB}")
message(STATUS "MPICH_VERSION              = ${MPICH_VERSION}")
message(STATUS "MPICH_URL                  = ${MPICH_URL}")
message(STATUS "MPICH_INSTALL_DIR          = ${MPICH_INSTALL_DIR}")
message(STATUS "MPICH_SOURCE_DIR           = ${MPICH_SOURCE_DIR}")
message(STATUS "MPICH_BUILD_DIR            = ${MPICH_BUILD_DIR}")
message(STATUS "MPICH_INCLUDE_DIR          = ${MPICH_INCLUDE_DIR}")
message(STATUS "MPI_C_LIB                  = ${MPI_C_LIB}")
message(STATUS "MPI_CXX_LIB                = ${MPI_CXX_LIB}")
message(STATUS "DSTP_EXAMPLE_PATH    = ${DSTP_EXAMPLE_PATH}")
message(STATUS "DSTP_EXAMPLE_BUILD_PATH = ${DSTP_EXAMPLE_BUILD_PATH}")

# ==============================================================================
# Build MPICH automatically
# ==============================================================================
ExternalProject_Add(mpich_ep
  URL ${MPICH_URL}
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/mpich-prefix
  SOURCE_DIR ${MPICH_SOURCE_DIR}
  BINARY_DIR ${MPICH_BUILD_DIR}
  CONFIGURE_COMMAND ${MPICH_CONFIGURE_COMMAND}
  BUILD_COMMAND make -j${MPICH_JOBS}
  INSTALL_COMMAND make install
  BUILD_BYPRODUCTS
  ${MPICH_INSTALL_DIR}/include/mpi.h
  ${MPICH_INSTALL_DIR}/lib/libmpi.so
  ${MPICH_INSTALL_DIR}/lib/libmpicxx.so
)

file(MAKE_DIRECTORY "${MPICH_INSTALL_DIR}")
file(MAKE_DIRECTORY "${MPICH_INCLUDE_DIR}")
file(MAKE_DIRECTORY "${MPICH_INSTALL_DIR}/lib")

add_library(mpich_c SHARED IMPORTED GLOBAL)
set_target_properties(mpich_c PROPERTIES
  IMPORTED_LOCATION "${MPI_C_LIB}"
)
add_dependencies(mpich_c mpich_ep)

add_library(mpich_cxx SHARED IMPORTED GLOBAL)
set_target_properties(mpich_cxx PROPERTIES
  IMPORTED_LOCATION "${MPI_CXX_LIB}"
)
add_dependencies(mpich_cxx mpich_ep)

# ==============================================================================
# Generate MLIR files
# ==============================================================================
add_custom_command(
  OUTPUT
  ${CMAKE_CURRENT_BINARY_DIR}/forward_decode0.mlir
  ${CMAKE_CURRENT_BINARY_DIR}/forward_decode1.mlir
  ${CMAKE_CURRENT_BINARY_DIR}/forward_decode2.mlir
  ${CMAKE_CURRENT_BINARY_DIR}/forward_decode3.mlir
  ${CMAKE_CURRENT_BINARY_DIR}/forward_decode5.mlir
  ${CMAKE_CURRENT_BINARY_DIR}/forward_decode169.mlir
  ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode0.mlir
  ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode1.mlir
  ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode2.mlir
  ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode3.mlir
  ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode5.mlir
  ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode169.mlir
  ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill0.mlir
  ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill1.mlir
  ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill2.mlir
  ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill3.mlir
  ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill5.mlir
  ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill169.mlir
  ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill0.mlir
  ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill1.mlir
  ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill2.mlir
  ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill3.mlir
  ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill5.mlir
  ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill169.mlir
  COMMAND ${CMAKE_COMMAND} -E env
  PYTHONPATH=${PYTHONPATH}
  ${Python3_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/import-deepseek-r1.py
  --output-dir ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating forward.mlir, subgraph.mlir and arg0.data..."
  VERBATIM
)

# ==============================================================================
# Pass pipelines
# ==============================================================================
set(FORWARD_PASSES
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
  -reconcile-unrealized-casts
)

set(SUBGRAPH_PASSES
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
  -reconcile-unrealized-casts
)

set(SUBGRAPH_DECODE_PASSES
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
  -reconcile-unrealized-casts
)

# ==============================================================================
# Build helper functions
# ==============================================================================
function(build_forward_prefill N)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill${N}.o
    COMMAND ${BUDDY_TOOLS_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill${N}.mlir
    -simplify-tosa-reshape |
    ${LLVM_TOOLS_BIN_DIR}/mlir-opt -pass-pipeline ${TOSA_PIPELINE} |
    ${BUDDY_TOOLS_BINARY_DIR}/buddy-opt ${FORWARD_PASSES} |
    ${LLVM_TOOLS_BIN_DIR}/mlir-translate -mlir-to-llvmir |
    ${LLVM_TOOLS_BIN_DIR}/llvm-as |
    ${LLVM_TOOLS_BIN_DIR}/llc ${LLC_TARGET_FLAGS} -filetype=obj -relocation-model=pic -O3
    -o ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill${N}.o
    DEPENDS ${BUDDY_OPT_DEP} ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill${N}.mlir
    COMMENT "Building forward_prefill${N}.o"
    VERBATIM
  )
endfunction()

function(build_subgraph_prefill N)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill${N}.o
    COMMAND ${BUDDY_TOOLS_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill${N}.mlir
    -simplify-tosa-reshape |
    ${LLVM_TOOLS_BIN_DIR}/mlir-opt -pass-pipeline ${TOSA_PIPELINE} |
    ${BUDDY_TOOLS_BINARY_DIR}/buddy-opt ${SUBGRAPH_PASSES} |
    ${LLVM_TOOLS_BIN_DIR}/mlir-translate -mlir-to-llvmir |
    ${LLVM_TOOLS_BIN_DIR}/llvm-as |
    ${LLVM_TOOLS_BIN_DIR}/llc ${LLC_TARGET_FLAGS} -filetype=obj -relocation-model=pic -O3
    -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill${N}.o
    DEPENDS ${BUDDY_OPT_DEP} ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill${N}.mlir
    COMMENT "Building subgraph0_prefill${N}.o"
    VERBATIM
  )
endfunction()

function(build_forward_decode N)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward_decode${N}.o
    COMMAND ${BUDDY_TOOLS_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_decode${N}.mlir
    -simplify-tosa-reshape |
    ${LLVM_TOOLS_BIN_DIR}/mlir-opt -pass-pipeline ${TOSA_PIPELINE} |
    ${BUDDY_TOOLS_BINARY_DIR}/buddy-opt ${FORWARD_PASSES} |
    ${LLVM_TOOLS_BIN_DIR}/mlir-translate -mlir-to-llvmir |
    ${LLVM_TOOLS_BIN_DIR}/llvm-as |
    ${LLVM_TOOLS_BIN_DIR}/llc ${LLC_TARGET_FLAGS} -filetype=obj -relocation-model=pic -O3
    -o ${CMAKE_CURRENT_BINARY_DIR}/forward_decode${N}.o
    DEPENDS ${BUDDY_OPT_DEP} ${CMAKE_CURRENT_BINARY_DIR}/forward_decode${N}.mlir
    COMMENT "Building forward_decode${N}.o"
    VERBATIM
  )
endfunction()

function(build_subgraph_decode N)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode${N}.o
    COMMAND ${BUDDY_TOOLS_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode${N}.mlir
    -simplify-tosa-reshape -simplify-tosa-matmul-scalar |
    ${LLVM_TOOLS_BIN_DIR}/mlir-opt -pass-pipeline ${TOSA_PIPELINE} |
    ${BUDDY_TOOLS_BINARY_DIR}/buddy-opt ${SUBGRAPH_DECODE_PASSES} |
    ${LLVM_TOOLS_BIN_DIR}/mlir-translate -mlir-to-llvmir |
    ${LLVM_TOOLS_BIN_DIR}/llvm-as |
    ${LLVM_TOOLS_BIN_DIR}/llc ${LLC_TARGET_FLAGS} -filetype=obj -relocation-model=pic -O3
    -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode${N}.o
    DEPENDS ${BUDDY_OPT_DEP} ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode${N}.mlir
    COMMENT "Building subgraph0_decode${N}.o"
    VERBATIM
  )
endfunction()

# ==============================================================================
# Build libraries
# ==============================================================================
foreach(N 0 1 2 3 5 169)
  build_forward_prefill(${N})
  build_subgraph_prefill(${N})
  add_library(DSR1TP_prefill${N} STATIC
    ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill${N}.o
    ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill${N}.o)

  build_forward_decode(${N})
  build_subgraph_decode(${N})
  add_library(DSR1TP_decode${N} STATIC
    ${CMAKE_CURRENT_BINARY_DIR}/forward_decode${N}.o
    ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode${N}.o)
endforeach()

set_target_properties(
  DSR1TP_prefill0 DSR1TP_prefill1 DSR1TP_prefill2
  DSR1TP_prefill3 DSR1TP_prefill5 DSR1TP_prefill169
  DSR1TP_decode0 DSR1TP_decode1 DSR1TP_decode2
  DSR1TP_decode3 DSR1TP_decode5 DSR1TP_decode169
  PROPERTIES
  LINKER_LANGUAGE C
)

# ==============================================================================
# Final executable
# ==============================================================================
add_executable(buddy-deepseek-r1-distributed buddy-deepseek-r1-distributed.cpp)

add_dependencies(buddy-deepseek-r1-distributed mpich_ep)

target_compile_definitions(buddy-deepseek-r1-distributed PRIVATE
  DEEPSEEKR1_EXAMPLE_PATH="${DSTP_EXAMPLE_PATH}/"
  DEEPSEEKR1_EXAMPLE_BUILD_PATH="${DSTP_EXAMPLE_BUILD_PATH}/"
)

target_include_directories(buddy-deepseek-r1-distributed PRIVATE
  "${MPICH_INCLUDE_DIR}"
)

if(NOT CROSS_COMPILE_RVV)
  target_link_directories(buddy-deepseek-r1-distributed PRIVATE ${LLVM_LIBRARY_DIR})
endif()

target_link_libraries(buddy-deepseek-r1-distributed PRIVATE
  DSR1TP_prefill0
  DSR1TP_prefill1
  DSR1TP_prefill2
  DSR1TP_prefill3
  DSR1TP_prefill5
  DSR1TP_prefill169
  DSR1TP_decode0
  DSR1TP_decode1
  DSR1TP_decode2
  DSR1TP_decode3
  DSR1TP_decode5
  DSR1TP_decode169
  "${MLIR_RUNNER_UTILS_LIB}"
  "${OPENMP_LIB}"
  mpich_cxx
  mpich_c
)

# ==============================================================================
# RVV packaging
# ==============================================================================
set(DEEPSEEKR1_DIST_EXE "$<TARGET_FILE:buddy-deepseek-r1-distributed>")
set(DEEPSEEKR1_RUNTIME_OMP "${RISCV_OMP_SHARED}")
set(DEEPSEEKR1_RUNTIME_MPICH_DIR "${CMAKE_CURRENT_BINARY_DIR}/mpich-install")
set(DEEPSEEKR1_RUNTIME_VOCAB "${CMAKE_SOURCE_DIR}/examples/BuddyTensorParallel/vocab.txt")

# Package 1 parameters
set(DEEPSEEKR1_PKG1_PARAM_FILES
  ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill0_arg0.data
  ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill169_arg0.data
)

# Package 2/3 common RMS parameters
set(DEEPSEEKR1_PKG23_RMS_FILES)
foreach(i 1 7 13 19 25 31 37 43 49 55 61 67 73 79 85 91 97 103 109 115 121 127 133 139 145 151 157 163)
  list(APPEND DEEPSEEKR1_PKG23_RMS_FILES
    ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill${i}_arg0.data
  )
  math(EXPR i_plus_3 "${i} + 3")
  list(APPEND DEEPSEEKR1_PKG23_RMS_FILES
    ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill${i_plus_3}_arg0.data
  )
endforeach()

# Package 2: MHA arg0, MLP arg0
set(DEEPSEEKR1_PKG2_PARAM_FILES ${DEEPSEEKR1_PKG23_RMS_FILES})
foreach(i 2 8 14 20 26 32 38 44 50 56 62 68 74 80 86 92 98 104 110 116 122 128 134 140 146 152 158 164)
  list(APPEND DEEPSEEKR1_PKG2_PARAM_FILES
    ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill${i}_arg0.data
  )
  math(EXPR i_plus_3 "${i} + 3")
  list(APPEND DEEPSEEKR1_PKG2_PARAM_FILES
    ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill${i_plus_3}_arg0.data
  )
endforeach()

# Package 3: MHA arg1, MLP arg1
set(DEEPSEEKR1_PKG3_PARAM_FILES ${DEEPSEEKR1_PKG23_RMS_FILES})
foreach(i 2 8 14 20 26 32 38 44 50 56 62 68 74 80 86 92 98 104 110 116 122 128 134 140 146 152 158 164)
  list(APPEND DEEPSEEKR1_PKG3_PARAM_FILES
    ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill${i}_arg1.data
  )
  math(EXPR i_plus_3 "${i} + 3")
  list(APPEND DEEPSEEKR1_PKG3_PARAM_FILES
    ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill${i_plus_3}_arg1.data
  )
endforeach()

function(add_deepseek_r1_package TARGET_NAME PKG_BASENAME INCLUDE_VOCAB)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs EXTRA_FILES PARAM_FILES)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(PKG_DIR "${CMAKE_CURRENT_BINARY_DIR}/${PKG_BASENAME}")
  set(PKG_ARCHIVE "${CMAKE_CURRENT_BINARY_DIR}/${PKG_BASENAME}.tgz")

  set(PKG_COPY_COMMANDS
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${PKG_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${PKG_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${DEEPSEEKR1_DIST_EXE}
    ${DEEPSEEKR1_RUNTIME_OMP}
    ${PKG_DIR}/
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${DEEPSEEKR1_RUNTIME_MPICH_DIR}
    ${PKG_DIR}/mpich-install
  )

  if(INCLUDE_VOCAB)
    list(APPEND PKG_COPY_COMMANDS
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${DEEPSEEKR1_RUNTIME_VOCAB}
      ${PKG_DIR}/
    )
  endif()

  foreach(f IN LISTS PKG_EXTRA_FILES)
    list(APPEND PKG_COPY_COMMANDS
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${f}
      ${PKG_DIR}/
    )
  endforeach()

  foreach(f IN LISTS PKG_PARAM_FILES)
    list(APPEND PKG_COPY_COMMANDS
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${f}
      ${PKG_DIR}/
    )
  endforeach()

  add_custom_command(
    OUTPUT ${PKG_ARCHIVE}
    ${PKG_COPY_COMMANDS}
    COMMAND ${CMAKE_COMMAND} -E tar "cfvz" ${PKG_ARCHIVE} --format=gnutar ${PKG_BASENAME}
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${PKG_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS
    buddy-deepseek-r1-distributed
    mpich_ep
    ${DEEPSEEKR1_RUNTIME_OMP}
    ${DEEPSEEKR1_RUNTIME_MPICH_DIR}
    ${PKG_EXTRA_FILES}
    ${PKG_PARAM_FILES}
    $<$<BOOL:${INCLUDE_VOCAB}>:${DEEPSEEKR1_RUNTIME_VOCAB}>
    COMMENT "Creating package ${PKG_ARCHIVE}"
    USES_TERMINAL
    VERBATIM
  )

  add_custom_target(${TARGET_NAME} DEPENDS ${PKG_ARCHIVE})
endfunction()

if(CROSS_COMPILE_RVV)
  add_deepseek_r1_package(
    buddy-deepseek-r1-rvv-package-rank0
    buddy-deepseek-r1-rvv-package-rank0
    TRUE
    PARAM_FILES
    ${DEEPSEEKR1_PKG1_PARAM_FILES}
  )

  add_deepseek_r1_package(
    buddy-deepseek-r1-rvv-package-rank1
    buddy-deepseek-r1-rvv-package-rank1
    FALSE
    PARAM_FILES
    ${DEEPSEEKR1_PKG2_PARAM_FILES}
  )

  add_deepseek_r1_package(
    buddy-deepseek-r1-rvv-package-rank2
    buddy-deepseek-r1-rvv-package-rank2
    FALSE
    PARAM_FILES
    ${DEEPSEEKR1_PKG3_PARAM_FILES}
  )

  add_custom_target(buddy-deepseek-r1-rvv-packages
    DEPENDS
    buddy-deepseek-r1-rvv-package-rank0
    buddy-deepseek-r1-rvv-package-rank1
    buddy-deepseek-r1-rvv-package-rank2
  )
endif()
