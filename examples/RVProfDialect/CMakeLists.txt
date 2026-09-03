add_library(RVProfRuntime STATIC
  RVProfRuntime.c
)

target_include_directories(RVProfRuntime PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Ensure RISC-V target
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "riscv")
  message(WARNING "RVProf is designed for RISC-V platforms. Current target: ${CMAKE_SYSTEM_PROCESSOR}")
endif()
