find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(BUDDY_TORQ_LLC_EXTRA "" CACHE STRING
    "Extra flags for llc when compiling subgraph LLVM IR")
separate_arguments(BUDDY_TORQ_LLC_EXTRA_ARGS NATIVE_COMMAND
                   "${BUDDY_TORQ_LLC_EXTRA}")

set(BUDDY_TORQ_PYTHONPATH "${CMAKE_BINARY_DIR}/python_packages:$ENV{PYTHONPATH}")

if(NOT TARGET buddy_matmul_torq)
  message(FATAL_ERROR
    "BuddyTorq requires the runtime library buddy_matmul_torq. Configure the buddy-mlir "
    "root build with -DBUDDY_RUNTIME_MATMUL_TORQ=ON and ensure TORQ-Tile is visible to "
    "find_package (e.g. -DTorqTile_DIR=/path/to/torq-tile/build, or CMAKE_PREFIX_PATH "
    "to an install prefix). Build TORQ-Tile first so TorqTileConfig.cmake exists.")
endif()

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
  COMMAND ${CMAKE_COMMAND} -E env
          "BUDDY_MLIR_BUILD_DIR=${CMAKE_BINARY_DIR}"
          "LLVM_MLIR_BUILD_DIR=${CMAKE_BINARY_DIR}/../llvm/build"
          "PYTHONPATH=${BUDDY_TORQ_PYTHONPATH}"
          ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/import_matmul_model.py
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
          python-package-buddy
          python-package-buddy-mlir
          ${CMAKE_CURRENT_SOURCE_DIR}/import_matmul_model.py
  COMMENT "Generating MLIR (matmul + TORQ external call)..."
  VERBATIM
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward-opt.mlir
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize="bufferize-function-boundaries"
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -lower-affine
            -cse
            -memref-expand
            -arith-expand
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward-opt.mlir
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
  COMMENT "Optimizing forward.mlir..."
  VERBATIM
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-opt.mlir
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -convert-elementwise-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -lower-affine
            -cse
            -memref-expand
            -arith-expand
          -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-opt.mlir
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
  COMMENT "Optimizing subgraph0.mlir..."
  VERBATIM
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward-llvm.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward-opt.mlir
            -convert-vector-to-llvm
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-func-to-llvm
            -reconcile-unrealized-casts
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward-llvm.mlir
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward-opt.mlir
  VERBATIM
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-llvm.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-opt.mlir
            -convert-vector-to-llvm
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -reconcile-unrealized-casts
          -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-llvm.mlir
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-opt.mlir
  VERBATIM
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward.ll
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-translate --mlir-to-llvmir
          ${CMAKE_CURRENT_BINARY_DIR}/forward-llvm.mlir
        -o ${CMAKE_CURRENT_BINARY_DIR}/forward.ll
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/forward-llvm.mlir
  VERBATIM
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.ll
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-translate --mlir-to-llvmir
          ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-llvm.mlir
        -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.ll
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-llvm.mlir
  VERBATIM
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/llc ${BUDDY_TORQ_LLC_EXTRA_ARGS}
          -filetype=obj -relocation-model=pic -O3
          ${CMAKE_CURRENT_BINARY_DIR}/forward.ll
        -o ${CMAKE_CURRENT_BINARY_DIR}/forward.o
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/forward.ll
  VERBATIM
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/llc ${BUDDY_TORQ_LLC_EXTRA_ARGS}
          -filetype=obj -relocation-model=pic -O3
          ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.ll
        -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.o
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.ll
  VERBATIM
)

set(MLIR_RUNNER_UTILS_LIB "${LLVM_TOOLS_BINARY_DIR}/../lib")

add_executable(buddy-torq-run
  test_mlir_runner.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.o
)

target_link_directories(buddy-torq-run PRIVATE ${MLIR_RUNNER_UTILS_LIB})
target_link_libraries(buddy-torq-run PRIVATE
  buddy_matmul_torq
  mlir_c_runner_utils
  mlir_runner_utils
)

set_target_properties(buddy-torq-run PROPERTIES
  INSTALL_RPATH "$ORIGIN/../lib:${MLIR_RUNNER_UTILS_LIB}:${CMAKE_CURRENT_BINARY_DIR}"
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH "$ORIGIN/../lib:${CMAKE_BINARY_DIR}/lib:${MLIR_RUNNER_UTILS_LIB}:${CMAKE_CURRENT_BINARY_DIR}"
)
