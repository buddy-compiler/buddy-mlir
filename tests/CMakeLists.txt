add_subdirectory(Interface)

set(RISCV_GNU_TOOLCHAIN_SYSROOT "${BUDDY_SOURCE_DIR}/thirdparty/build-riscv-gnu-toolchain/sysroot")
set(QEMU_BINARY_DIR "${BUDDY_SOURCE_DIR}/thirdparty/qemu/build/riscv64-linux-user")
set(CROSS_MLIR_RUNNER_UTILS "${BUDDY_SOURCE_DIR}/thirdparty/build-cross-mlir/lib")
set(CROSS_LLI_BIN "${BUDDY_SOURCE_DIR}/thirdparty/build-cross-clang/bin/lli")

if (NOT EXISTS ${RISCV_GNU_TOOLCHAIN_SYSROOT}
    OR NOT EXISTS ${CROSS_MLIR_RUNNER_UTILS}
    OR NOT EXISTS ${CROSS_LLI_BIN})
    message(WARNING "RISC-V envinroment is not set up. RVV tests will fail."
                    "For more info how to set up the envinroment, please refer to documention.")
endif()
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(BUDDY_TEST_DEPENDS
  FileCheck count not
  buddy-opt
  buddy-translate
  buddy-container-test
  buddy-audio-container-test
  )

if(BUDDY_ENABLE_OPENCV)
  list(APPEND BUDDY_TEST_DEPENDS buddy-image-container-test)
endif()

add_lit_testsuite(check-buddy "Running the buddy regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${BUDDY_TEST_DEPENDS}
  )
set_target_properties(check-buddy PROPERTIES FOLDER "Tests")

add_lit_testsuites(BUDDY ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${BUDDY_TEST_DEPENDS})
