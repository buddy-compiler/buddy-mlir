add_custom_command(
  OUTPUT ${BUDDY_EXAMPLES_DIR}/BuddyLeNet/forward.mlir ${BUDDY_EXAMPLES_DIR}/BuddyLeNet/subgraph1.mlir ${BUDDY_EXAMPLES_DIR}/BuddyLeNet/arg0.data
  COMMAND python3 ${BUDDY_EXAMPLES_DIR}/BuddyLeNet/buddy-lenet-import.py
  COMMENT "Generating forward.mlir, subgraph1.mlir and parameter files"
)

add_custom_command(
  OUTPUT forward.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/BuddyLeNet/forward.mlir 
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith), empty-tensor-to-alloc-tensor, convert-elementwise-to-linalg, arith-bufferize, func.func(linalg-bufferize, tensor-bufferize), func-bufferize)" |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt 
            -pass-pipeline "builtin.module(func.func(buffer-deallocation-simplification, convert-linalg-to-loops), eliminate-empty-tensors, func.func(llvm-request-c-wrappers),convert-math-to-llvm, convert-math-to-libm, convert-scf-to-cf,  convert-arith-to-llvm, expand-strided-metadata, finalize-memref-to-llvm, convert-func-to-llvm, reconcile-unrealized-casts)" |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj  -relocation-model=pic -O0 -o ${BUDDY_BINARY_DIR}/../examples/BuddyLeNet/forward.o
  DEPENDS ${BUDDY_EXAMPLES_DIR}/BuddyLeNet/forward.mlir
  COMMENT "Building forward.o"
  VERBATIM)

add_custom_command(
  OUTPUT subgraph0.ll
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${BUDDY_EXAMPLES_DIR}/BuddyLeNet/subgraph0.mlir -gpu-kernel-outlining -llvm-request-c-wrappers -convert-vector-to-scf -convert-vector-to-llvm -func-bufferize-dynamic-offset -tensor-bufferize -buffer-deallocation -finalizing-bufferize -expand-strided-metadata  -one-shot-bufferize |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt 
            -pass-pipeline "builtin.module(nvvm-attach-target{chip=sm_75 O=3},  gpu.module(convert-scf-to-cf, convert-gpu-to-nvvm, convert-arith-to-llvm), convert-scf-to-cf, gpu-to-llvm, reconcile-unrealized-casts, gpu-module-to-binary)" |
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir -o ${BUDDY_BINARY_DIR}/../examples/BuddyLeNet/subgraph0.ll
  DEPENDS ${BUDDY_EXAMPLES_DIR}/BuddyLeNet/subgraph0.mlir
  COMMENT "Building subgraph0.ll"
  VERBATIM)
  
add_custom_command(
  OUTPUT subgraph0.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang++ ${BUDDY_BINARY_DIR}/../examples/BuddyLeNet/subgraph0.ll -L/usr/local/cuda/lib64 -lcudart -O3 -c -o ${BUDDY_BINARY_DIR}/../examples/BuddyLeNet/subgraph0.o
  DEPENDS ${BUDDY_BINARY_DIR}/../examples/BuddyLeNet/subgraph0.ll
  COMMENT "Building subgraph0.o"
  VERBATIM)
  
# add_custom_command(
#   OUTPUT subgraph1.ll
#   COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${BUDDY_EXAMPLES_DIR}/BuddyLeNet/subgraph1.mlir -gpu-kernel-outlining -llvm-request-c-wrappers -convert-vector-to-scf -convert-vector-to-llvm -func-bufferize-dynamic-offset -buffer-deallocation -finalizing-bufferize -expand-strided-metadata  -one-shot-bufferize |
#           ${LLVM_MLIR_BINARY_DIR}/mlir-opt 
#             -pass-pipeline "builtin.module(nvvm-attach-target{chip=sm_75 O=3},  gpu.module(convert-scf-to-cf, convert-gpu-to-nvvm, convert-arith-to-llvm), convert-scf-to-cf, gpu-to-llvm, reconcile-unrealized-casts, gpu-module-to-binary)" |
#           ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir -o ${BUDDY_BINARY_DIR}/../examples/BuddyLeNet/subgraph1.ll
#   DEPENDS ${BUDDY_EXAMPLES_DIR}/BuddyLeNet/subgraph1.mlir
#   COMMENT "Building subgraph1.ll"
#   VERBATIM)
  
# add_custom_command(
#   OUTPUT subgraph1.o
#   COMMAND ${LLVM_MLIR_BINARY_DIR}/clang++ ${BUDDY_BINARY_DIR}/../examples/BuddyLeNet/subgraph1.ll -L/usr/local/cuda/lib64 -lcudart -O3 -c -o ${BUDDY_BINARY_DIR}/../examples/BuddyLeNet/subgraph1.o
#   DEPENDS ${BUDDY_BINARY_DIR}/../examples/BuddyLeNet/subgraph1.ll
#   COMMENT "Building subgraph1.o"
#   VERBATIM)

add_custom_command(
  OUTPUT subgraph1.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/BuddyLeNet/subgraph1.mlir 
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
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llvm-as |
          ${LLVM_MLIR_BINARY_DIR}/llc -filetype=obj  -relocation-model=pic -O0 -o ${BUDDY_BINARY_DIR}/../examples/BuddyLeNet/subgraph0.o
  DEPENDS ${BUDDY_EXAMPLES_DIR}/BuddyLeNet/subgraph0.mlir
  COMMENT "Building subgraph0.o"
  VERBATIM)

add_library(LENET STATIC subgraph0.o subgraph1.o forward.o)

SET_TARGET_PROPERTIES(LENET PROPERTIES LINKER_LANGUAGE C)

add_executable(buddy-lenet-run buddy-lenet-main.cpp)
target_link_directories(buddy-lenet-run PRIVATE ${LLVM_LIBRARY_DIR})

set(BUDDY_LENET_LIBS LENET mlir_c_runner_utils mlir_cuda_runtime ${OpenCV_LIBS})
target_link_libraries(buddy-lenet-run ${BUDDY_LENET_LIBS})
