function(BuildLeNet DEVICE_TYPE)

  add_custom_command(
    OUTPUT ${BUDDY_EXAMPLES_DIR}/BuddyLeNet/forward-${DEVICE_TYPE}.mlir 
    ${BUDDY_EXAMPLES_DIR}/BuddyLeNet/subgraph0-${DEVICE_TYPE}.mlir 
    ${BUDDY_EXAMPLES_DIR}/BuddyLeNet/subgraph1-${DEVICE_TYPE}.mlir 
    COMMAND python3 ${BUDDY_EXAMPLES_DIR}/BuddyLeNet/buddy-lenet-import.py --DEVICE_TYPE ${DEVICE_TYPE}
    COMMENT "Generating forward.mlir, subgraph.mlir, and parameter files"
  )

  add_custom_command(
    OUTPUT forward-${DEVICE_TYPE}.o
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/BuddyLeNet/forward-${DEVICE_TYPE}.mlir 
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith), empty-tensor-to-alloc-tensor, convert-elementwise-to-linalg, arith-bufferize, func.func(linalg-bufferize, tensor-bufferize), func-bufferize)" |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-opt 
              -pass-pipeline "builtin.module(func.func(buffer-deallocation-simplification, convert-linalg-to-loops), eliminate-empty-tensors, func.func(llvm-request-c-wrappers),convert-math-to-llvm, convert-math-to-libm, convert-scf-to-cf,  convert-arith-to-llvm, expand-strided-metadata, finalize-memref-to-llvm, convert-func-to-llvm, reconcile-unrealized-casts)" |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
            ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
            ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj  -relocation-model=pic -O0 -o ${BUDDY_BINARY_DIR}/../examples/BuddyLeNet/forward-${DEVICE_TYPE}.o
    DEPENDS ${BUDDY_EXAMPLES_DIR}/BuddyLeNet/forward-${DEVICE_TYPE}.mlir
    COMMENT "Building forward-${DEVICE_TYPE}.o"
    VERBATIM)

  add_custom_command(
    OUTPUT subgraph0-${DEVICE_TYPE}.o
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/BuddyLeNet/subgraph0-${DEVICE_TYPE}.mlir 
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
            ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj  -relocation-model=pic -O0 -o ${BUDDY_BINARY_DIR}/../examples/BuddyLeNet/subgraph0-${DEVICE_TYPE}.o
    DEPENDS ${BUDDY_EXAMPLES_DIR}/BuddyLeNet/subgraph0-${DEVICE_TYPE}.mlir
    COMMENT "Building subgraph0-${DEVICE_TYPE}.o"
    VERBATIM)

  set(ONE_SHOT_BUFFERIZE_OPTION "bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map")
  set(LOWER_TO_NVVM_OPTION "cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=fatbin")
  add_custom_command(
    OUTPUT subgraph1-${DEVICE_TYPE}.o
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/BuddyLeNet/subgraph1-${DEVICE_TYPE}.mlir 
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -one-shot-bufferize=${ONE_SHOT_BUFFERIZE_OPTION}
            -buffer-deallocation
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -canonicalize
            -cse |
            ${BUDDY_BINARY_DIR}/buddy-opt -convert-memcpy-to-gpu -gpu-async-region -canonicalize |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-opt -llvm-request-c-wrappers --test-lower-to-nvvm=${LOWER_TO_NVVM_OPTION} |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
            ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
            ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj  -relocation-model=pic -O0 -o ${BUDDY_BINARY_DIR}/../examples/BuddyLeNet/subgraph1-${DEVICE_TYPE}.o
    DEPENDS ${BUDDY_EXAMPLES_DIR}/BuddyLeNet/subgraph1-${DEVICE_TYPE}.mlir
    COMMENT "Building subgraph1-${DEVICE_TYPE}.o"
    VERBATIM)

  if(${DEVICE_TYPE} STREQUAL "heter")
    add_library(${DEVICE_TYPE}LENET STATIC subgraph0-${DEVICE_TYPE}.o subgraph1-${DEVICE_TYPE}.o forward-${DEVICE_TYPE}.o)
  elseif(${DEVICE_TYPE} STREQUAL "cpu")
    add_library(${DEVICE_TYPE}LENET STATIC subgraph0-${DEVICE_TYPE}.o forward-${DEVICE_TYPE}.o)
  endif()

  SET_TARGET_PROPERTIES(${DEVICE_TYPE}LENET PROPERTIES LINKER_LANGUAGE C)

  add_executable(buddy-lenet-${DEVICE_TYPE}-run buddy-lenet-main.cpp)
  target_link_directories(buddy-lenet-${DEVICE_TYPE}-run PRIVATE ${LLVM_LIBRARY_DIR})

  if(${DEVICE_TYPE} STREQUAL "cpu")
    set(BUDDY_LENET_LIBS ${DEVICE_TYPE}LENET mlir_c_runner_utils ${PNG_LIBRARIES})
  else()
    set(BUDDY_LENET_LIBS ${DEVICE_TYPE}LENET mlir_c_runner_utils mlir_async_runtime mlir_runner_utils mlir_cuda_runtime BuddyLibDIP ${PNG_LIBRARIES})
  endif()

  target_link_libraries(buddy-lenet-${DEVICE_TYPE}-run ${BUDDY_LENET_LIBS})
endfunction()

BuildLeNet("heter")
BuildLeNet("cpu")
