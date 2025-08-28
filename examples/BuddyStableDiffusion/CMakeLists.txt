add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/arg0_text_encoder.data
         ${CMAKE_CURRENT_BINARY_DIR}/arg1_text_encoder.data
         ${CMAKE_CURRENT_BINARY_DIR}/arg0_unet.data
         ${CMAKE_CURRENT_BINARY_DIR}/arg0_vae.data
         ${CMAKE_CURRENT_BINARY_DIR}/forward_text_encoder.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_text_encoder.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/forward_unet.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_unet.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/forward_vae.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_vae.mlir
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/import-stable-diffusion.py
  --output-dir ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating forward.mlir, subgraph0.mlir and parameter files"
)

add_custom_command(
  OUTPUT forward_text_encoder.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_text_encoder.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
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
        ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_text_encoder.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_text_encoder.mlir
  COMMENT "Building forward_text_encoder.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph0_text_encoder.o
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_text_encoder.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -convert-elementwise-to-linalg
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -func-bufferize-dynamic-offset
            -conv-nhwc-fhwc-optimize
            -matmul-parallel-vectorization-optimize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -lower-affine
            -convert-scf-to-openmp
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
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_text_encoder.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_text_encoder.mlir
    COMMENT "Building subgraph0_text_encoder.o "
    VERBATIM)

add_custom_command(
  OUTPUT forward_unet.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_unet.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
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
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_unet.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_unet.mlir
  COMMENT "Building forward_unet.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph0_unet.o
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_unet.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -convert-elementwise-to-linalg
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -conv-nhwc-fhwc-optimize
            -matmul-parallel-vectorization-optimize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -lower-affine
            -convert-scf-to-openmp
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
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_unet.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_unet.mlir
    COMMENT "Building subgraph0_unet.o "
    VERBATIM)

add_custom_command(
  OUTPUT forward_vae.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_vae.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
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
        ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_vae.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_vae.mlir
  COMMENT "Building forward_vae.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph0_vae.o
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_vae.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -convert-elementwise-to-linalg
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -conv-nhwc-fhwc-optimize
            -matmul-parallel-vectorization-optimize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -lower-affine
            -convert-scf-to-openmp
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
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_vae.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_vae.mlir
    COMMENT "Building subgraph0_vae.o "
    VERBATIM)


add_library(TEXTENCODER STATIC subgraph0_text_encoder.o forward_text_encoder.o)
add_library(UNET STATIC subgraph0_unet.o forward_unet.o)
add_library(VAE STATIC subgraph0_vae.o forward_vae.o)

SET_TARGET_PROPERTIES(TEXTENCODER PROPERTIES LINKER_LANGUAGE C)
SET_TARGET_PROPERTIES(UNET PROPERTIES LINKER_LANGUAGE C)
SET_TARGET_PROPERTIES(VAE PROPERTIES LINKER_LANGUAGE C)

add_executable(buddy-stable-diffusion-run buddy-stable-diffusion-main.cpp)

set(STABLE_DIFFUSION_EXAMPLE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(STABLE_DIFFUSION_EXAMPLE_BUILD_PATH ${CMAKE_CURRENT_BINARY_DIR})

target_compile_definitions(buddy-stable-diffusion-run PRIVATE
  STABLE_DIFFUSION_EXAMPLE_PATH="${STABLE_DIFFUSION_EXAMPLE_PATH}"
  STABLE_DIFFUSION_EXAMPLE_BUILD_PATH="${STABLE_DIFFUSION_EXAMPLE_BUILD_PATH}"
)

target_link_directories(buddy-stable-diffusion-run PRIVATE ${LLVM_LIBRARY_DIR})

set(BUDDY_STABLE_DIFFUSION_LIBS TEXTENCODER UNET VAE mlir_c_runner_utils omp)

if(BUDDY_MLIR_USE_MIMALLOC)
  list(APPEND BUDDY_STABLE_DIFFUSION_LIBS mimalloc)
endif()

target_link_libraries(buddy-stable-diffusion-run ${BUDDY_STABLE_DIFFUSION_LIBS})
