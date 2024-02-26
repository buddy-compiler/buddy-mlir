set(MLIR_INTERFACE_SOURCES
  gemmini.cpp
)

include_directories(
  ${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include
  ${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests
)

add_library(mlirInterface SHARED
  ${MLIR_INTERFACE_SOURCES}
)