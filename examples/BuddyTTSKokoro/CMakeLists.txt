# Define bufferize options as variables to avoid CMake quote parsing issues.
set(BUFFERIZE_SIMPLE_OPTS "bufferize-function-boundaries")
set(TOSA_PIPELINE "builtin.module(func.func(tosa-optional-decompositions,tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))")

# Detect RISC-V Vector (RVV) platform.
if(HAVE_LOCAL_RVV)
  set(IS_RVV_PLATFORM ON)
  message(STATUS "RISC-V Vector (RVV) platform detected")
endif()

# Set platform-specific variables.
if(IS_RVV_PLATFORM)
  set(OPENMP_NUM_THREADS "num-threads=32")
  set(LLC_RISCV_ATTRS "-mattr=+m,+d,+v")
else()
  set(OPENMP_NUM_THREADS "num-threads=48")
  set(LLC_RISCV_ATTRS "")
endif()

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward_predictor.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_predictor.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/forward_vocoder.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_vocoder.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/arg0_predictor.data
         ${CMAKE_CURRENT_BINARY_DIR}/arg0_vocoder.data
         ${CMAKE_CURRENT_BINARY_DIR}/input_ids.data
         ${CMAKE_CURRENT_BINARY_DIR}/ref_s.data
  COMMAND ${CMAKE_COMMAND} -E env
          PYTHONPATH=${BUDDY_MLIR_PYTHON_PACKAGES_DIR}:$ENV{PYTHONPATH}
          ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/import-kokoro.py
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating Kokoro MLIR and data files..."
  VERBATIM
)

add_custom_target(
  buddy-kokoro-import
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/forward_predictor.mlir
          ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_predictor.mlir
          ${CMAKE_CURRENT_BINARY_DIR}/forward_vocoder.mlir
          ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_vocoder.mlir
          ${CMAKE_CURRENT_BINARY_DIR}/arg0_predictor.data
          ${CMAKE_CURRENT_BINARY_DIR}/arg0_vocoder.data
          ${CMAKE_CURRENT_BINARY_DIR}/input_ids.data
          ${CMAKE_CURRENT_BINARY_DIR}/ref_s.data
)

function(add_kokoro_mlir_object target_name mlir_file output_obj)
  add_custom_command(
    OUTPUT ${output_obj}
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/${mlir_file}
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
              -convert-bufferization-to-memref
              -cse
              -canonicalize
              -convert-linalg-to-loops
              -cse
              -lower-affine
              -memref-expand
              -arith-expand
              -convert-arith-to-llvm
              -convert-complex-to-llvm
              -finalize-memref-to-llvm
              -convert-scf-to-cf
              -convert-cf-to-llvm
              -llvm-request-c-wrappers
              -convert-arith-to-llvm
              -convert-math-to-llvm
              -convert-math-to-libm
              -convert-func-to-llvm
              -reconcile-unrealized-casts |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
            ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
            ${LLVM_TOOLS_BINARY_DIR}/llc ${LLC_RISCV_ATTRS} -filetype=obj -relocation-model=pic -O0
              -o ${CMAKE_CURRENT_BINARY_DIR}/${output_obj}
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/${mlir_file}
    COMMENT "Building ${target_name}"
    VERBATIM
  )
endfunction()

add_kokoro_mlir_object(forward_predictor.o forward_predictor.mlir forward_predictor.o)
add_kokoro_mlir_object(subgraph_predictor.o subgraph0_predictor.mlir subgraph_predictor.o)
add_kokoro_mlir_object(forward_vocoder.o forward_vocoder.mlir forward_vocoder.o)
add_kokoro_mlir_object(subgraph_vocoder.o subgraph0_vocoder.mlir subgraph_vocoder.o)

add_library(KOKORO STATIC
  forward_predictor.o
  subgraph_predictor.o
  forward_vocoder.o
  subgraph_vocoder.o
)

SET_SOURCE_FILES_PROPERTIES(
  template.o
  PROPERTIES
  EXTERNAL_OBJECT true
  GENERATED true)

SET_TARGET_PROPERTIES(
  KOKORO
  PROPERTIES
  LINKER_LANGUAGE C)

add_executable(buddy-kokoro-run buddy-kokoro-main.cpp)

set(KOKORO_EXAMPLE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(KOKORO_EXAMPLE_BUILD_PATH ${CMAKE_CURRENT_BINARY_DIR})

target_compile_definitions(buddy-kokoro-run PRIVATE
  KOKORO_EXAMPLE_PATH="${KOKORO_EXAMPLE_PATH}/"
  KOKORO_EXAMPLE_BUILD_PATH="${KOKORO_EXAMPLE_BUILD_PATH}/"
)

target_link_directories(buddy-kokoro-run PRIVATE ${LLVM_LIBRARY_DIR})

set(BUDDY_KOKORO_LIBS
  KOKORO
  mlir_c_runner_utils
  omp
)

if(BUDDY_MLIR_USE_MIMALLOC)
  list(APPEND BUDDY_KOKORO_LIBS mimalloc)
endif()

target_link_libraries(buddy-kokoro-run ${BUDDY_KOKORO_LIBS})
