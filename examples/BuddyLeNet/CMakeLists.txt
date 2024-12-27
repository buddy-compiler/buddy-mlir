add_custom_command(
  OUTPUT ${BUDDY_BINARY_DIR}/../examples/BuddyLeNet/forward.mlir
         ${BUDDY_BINARY_DIR}/../examples/BuddyLeNet/subgraph0.mlir
         ${BUDDY_BINARY_DIR}/../examples/BuddyLeNet/arg0.data
  COMMAND python3 ${BUDDY_EXAMPLES_DIR}/BuddyLeNet/buddy-lenet-import.py
          --output-dir ${BUDDY_BINARY_DIR}/../examples/BuddyLeNet
  COMMENT "Generating forward.mlir, subgraph0.mlir and parameter files"
)

add_custom_command(
  OUTPUT forward.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${BUDDY_BINARY_DIR}/../examples/BuddyLeNet/forward.mlir 
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith), empty-tensor-to-alloc-tensor, convert-elementwise-to-linalg, arith-bufferize, func.func(linalg-bufferize, tensor-bufferize), func-bufferize)" |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt 
            -pass-pipeline "builtin.module(func.func(buffer-deallocation-simplification, convert-linalg-to-loops), eliminate-empty-tensors, func.func(llvm-request-c-wrappers),convert-math-to-llvm, convert-math-to-libm, convert-scf-to-cf,  convert-arith-to-llvm, expand-strided-metadata, finalize-memref-to-llvm, convert-func-to-llvm, reconcile-unrealized-casts)" |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj  -relocation-model=pic -O0 -o ${BUDDY_BINARY_DIR}/../examples/BuddyLeNet/forward.o
  DEPENDS ${BUDDY_BINARY_DIR}/../examples/BuddyLeNet/forward.mlir
  COMMENT "Building forward.o"
  VERBATIM)

add_custom_command(
  OUTPUT subgraph0.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${BUDDY_BINARY_DIR}/../examples/BuddyLeNet/subgraph0.mlir 
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -convert-tensor-to-linalg
            -linalg-bufferize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -lower-affine
            -func-bufferize-dynamic-offset
            -arith-bufferize
            -tensor-bufferize
            -buffer-deallocation
            -finalizing-bufferize
            -convert-vector-to-scf
            -expand-strided-metadata
            -convert-vector-to-llvm
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts | 
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj  -relocation-model=pic -O0 -o ${BUDDY_BINARY_DIR}/../examples/BuddyLeNet/subgraph0.o
  DEPENDS ${BUDDY_BINARY_DIR}/../examples/BuddyLeNet/subgraph0.mlir
  COMMENT "Building subgraph0.o"
  VERBATIM)

add_library(LENET STATIC subgraph0.o forward.o)

SET_TARGET_PROPERTIES(LENET PROPERTIES LINKER_LANGUAGE C)

add_executable(buddy-lenet-run buddy-lenet-main.cpp)
target_link_directories(buddy-lenet-run PRIVATE ${LLVM_LIBRARY_DIR})

if(NOT DEFINED BUDDY_ENABLE_PNG)
  message(FATAL_ERROR "To run LeNet inference, the png library is required. Please define BUDDY_ENABLE_PNG for CMake.")
endif()
set(BUDDY_LENET_LIBS LENET mlir_c_runner_utils mlir_async_runtime mlir_runner_utils ${PNG_LIBRARIES})

target_link_libraries(buddy-lenet-run ${BUDDY_LENET_LIBS})

set(ONE_SHOT_BUFFERIZE_OPTION "bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map")
set(LOWER_TO_NVVM_OPTION "cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=fatbin")
set(CONVERT_MEMCPY_TO_GPU_OPTION "process-args=1")
set(CONVERT_MEMCPY_TO_GPU_OPTION_DISABLE_PROCESS_ARG "process-args=0")

add_custom_command(
  OUTPUT forward_gpu.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${BUDDY_BINARY_DIR}/../examples/BuddyLeNet/forward.mlir 
          -buffer-deallocation
          -canonicalize -cse -expand-strided-metadata -convert-memcpy-to-gpu -gpu-async-region |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt -llvm-request-c-wrappers --gpu-to-llvm |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj  -relocation-model=pic -O0 -o ${BUDDY_BINARY_DIR}/../examples/BuddyLeNet/forward_gpu.o
  DEPENDS ${BUDDY_BINARY_DIR}/../examples/BuddyLeNet/forward.mlir
  COMMENT "Building forward_gpu.o"
  VERBATIM)

add_custom_command(
  OUTPUT subgraph0_gpu.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${BUDDY_BINARY_DIR}/../examples/BuddyLeNet/subgraph0.mlir 
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
          -one-shot-bufferize
          -func-bufferize-dynamic-offset
          -convert-linalg-to-parallel-loops
          -canonicalize
          -gpu-map-parallel-loops
          -convert-parallel-loops-to-gpu
          -gpu-kernel-outlining
          -buffer-deallocation
          -canonicalize
          -cse |
          ${BUDDY_BINARY_DIR}/buddy-opt -convert-memcpy-to-gpu=${CONVERT_MEMCPY_TO_GPU_OPTION_DISABLE_PROCESS_ARG} -gpu-async-region -canonicalize |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt -llvm-request-c-wrappers --test-lower-to-nvvm=${LOWER_TO_NVVM_OPTION} |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj  -relocation-model=pic -O0 -o ${BUDDY_BINARY_DIR}/../examples/BuddyLeNet/subgraph0_gpu.o
  DEPENDS ${BUDDY_BINARY_DIR}/../examples/BuddyLeNet/subgraph0.mlir
  COMMENT "Building subgraph0_gpu.o"
  VERBATIM)

add_library(LENET_GPU STATIC subgraph0_gpu.o forward_gpu.o)

SET_TARGET_PROPERTIES(LENET_GPU PROPERTIES LINKER_LANGUAGE C)

add_executable(buddy-lenet-run-gpu buddy-lenet-main.cpp)
target_link_directories(buddy-lenet-run-gpu PRIVATE ${LLVM_LIBRARY_DIR})

set(BUDDY_LENET_LIBS_GPU LENET_GPU mlir_c_runner_utils mlir_async_runtime mlir_runner_utils mlir_cuda_runtime ${PNG_LIBRARIES})

target_link_libraries(buddy-lenet-run-gpu ${BUDDY_LENET_LIBS_GPU})
