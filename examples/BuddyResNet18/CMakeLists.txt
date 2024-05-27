cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(BuddyResNet18)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 设置 Buddy Compiler 和 LLVM/MLIR 的路径
set(BUDDY_COMPILER_PATH /home/jack/buddy-mlir/build) # 更新为你的 buddy-mlir 路径
set(LLVM_MLIR_BINARY_DIR /home/jack/buddy-mlir/llvm/build/bin) # 更新为你的 LLVM/MLIR 可执行文件路径

# 包含 Buddy Compiler 的头文件
include_directories(${BUDDY_COMPILER_PATH}/include)

# 链接 Buddy Compiler 的库
link_directories(${BUDDY_COMPILER_PATH}/lib)

# 查找 OpenCV 包
find_package(OpenCV REQUIRED)

# 包含 OpenCV 的头文件
include_directories(${OpenCV_INCLUDE_DIRS})

# 设置生成参数文件和 MLIR 文件的命令
add_custom_command(
  OUTPUT ${CMAKE_SOURCE_DIR}/arg0.data
         ${CMAKE_SOURCE_DIR}/resnet.mlir
  COMMAND python3 ${CMAKE_SOURCE_DIR}/import-resnet18.py
  COMMENT "Generating resnet.mlir and parameter files"
)

# 设置生成 object 文件的命令
add_custom_command(
  OUTPUT resnet.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${CMAKE_SOURCE_DIR}/resnet.mlir
            -pass-pipeline
            "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith), \
            empty-tensor-to-alloc-tensor, convert-elementwise-to-linalg, arith-bufferize, \
            func.func(linalg-bufferize, tensor-bufferize), func-bufferize)" |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt
            -pass-pipeline
            "builtin.module(func.func(buffer-deallocation-simplification, convert-linalg-to-loops), \
            eliminate-empty-tensors, func.func(llvm-request-c-wrappers), \
            convert-math-to-llvm, convert-math-to-libm, convert-scf-to-cf, \
            convert-arith-to-llvm, expand-strided-metadata, finalize-memref-to-llvm, \
            convert-func-to-llvm, reconcile-unrealized-casts)" |
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llvm-as |
          ${LLVM_MLIR_BINARY_DIR}/llc -filetype=obj  -relocation-model=pic -O3 
            -o ${CMAKE_BINARY_DIR}/resnet.o
  DEPENDS ${CMAKE_SOURCE_DIR}/resnet.mlir
  COMMENT "Building resnet.o"
  VERBATIM
)

# 添加库
add_library(RESNET18 STATIC resnet.o)

SET_TARGET_PROPERTIES(RESNET18 PROPERTIES LINKER_LANGUAGE C)

# 添加可执行文件
add_executable(buddy-resnet18-run ResNet18Benchmark.cpp)
target_link_directories(buddy-resnet18-run PRIVATE ${LLVM_MLIR_LIBRARY_DIR})

# 链接库
set(BUDDY_RESNET18_LIBS RESNET18 mlir_c_runner_utils ${OpenCV_LIBS})
target_link_libraries(buddy-resnet18-run ${BUDDY_RESNET18_LIBS})
