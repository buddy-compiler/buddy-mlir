
set(BUDDY_OPT ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt)
set(BUDDY_TRANSLATE ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-translate)
set(BUDDY_LLC ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-llc)
set(INTERFACES ${BUDDY_MLIR_BUILD_DIR}/../frontend/Interfaces)

set(CMAKE_CXX_COMPILER riscv64-unknown-linux-gnu-g++)
set(CMAKE_C_COMPILER riscv64-unknown-linux-gnu-gcc)

include_directories(
  # ${BENCHMARKS_DIR}
  # ${GEMMINI_INCLUDE_DIR}
  # ${GEMMINI_INCLUDE_DIR}/../
  ${INTERFACES}
)

if (NOT DEFINED ENV{RISCV})
  message(FATAL_ERROR "Can't find RISCV environment variable(missing: RISCV_TOOLCHAIN)")
endif()

# CMAKE_C_FLAGS is set when configuring cmake.
separate_arguments(CLANG_FLAGS_LIST UNIX_COMMAND "${CMAKE_C_FLAGS}")

add_custom_command(OUTPUT buddy_matmul.o
  COMMAND ${BUDDY_OPT} ${CMAKE_CURRENT_SOURCE_DIR}/matmul.mlir 
            # -llvm-request-c-wrappers 
            # -convert-linalg-to-gemmini 
            -lower-buckyball  |
            # -convert-linalg-to-loops |  
            # -lower-gemmini 
            ${BUDDY_TRANSLATE} -buddy-to-llvmir | 
            ${BUDDY_LLC} -filetype=obj -mtriple=riscv64 
              -mattr=+buddyext,+D -float-abi=hard 
              -o buddy_matmul.o 
)
add_library(BuddyMatMul STATIC buddy_matmul.o)
set_target_properties(BuddyMatMul PROPERTIES LINKER_LANGUAGE C)

add_executable(buckyball-matmul-benchmark Main.cpp)
target_link_libraries(buckyball-matmul-benchmark
  -static
  BuddyMatMul
)
