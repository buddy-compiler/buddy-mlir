add_subdirectory(Interface)

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
  buddy-text-container-test
  mlir-runner
  )

if(BUDDY_ENABLE_OPENCV)
  list(APPEND BUDDY_TEST_DEPENDS buddy-image-container-test)
endif()

if(BUDDY_MLIR_ENABLE_DIP_LIB)
  list(APPEND BUDDY_TEST_DEPENDS buddy-new-image-container-test-bmp)
  if(BUDDY_ENABLE_PNG)
    list(APPEND BUDDY_TEST_DEPENDS buddy-new-image-container-test-png)
  endif()
endif()

if(BUDDY_MLIR_ENABLE_PYTHON_PACKAGES)
  list(APPEND BUDDY_TEST_DEPENDS BuddyMLIRPythonModules)
endif()

add_lit_testsuite(check-tests "Running the buddy regression tests..."
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${BUDDY_TEST_DEPENDS}
  )
set_target_properties(check-tests PROPERTIES FOLDER "Tests")

add_lit_testsuites(BUDDY ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${BUDDY_TEST_DEPENDS})

#------------------------------------------------------------------------------------
#    E2E test module
#------------------------------------------------------------------------------------

option(BUDDY_ENABLE_E2E_TESTS "Enable end-to-end (E2E) tests for Models" OFF)

if(BUDDY_ENABLE_E2E_TESTS)

  # Always enable PNG and DIP lib support
  set(BUDDY_ENABLE_PNG ON CACHE BOOL "Enable PNG support" FORCE)
  set(BUDDY_MLIR_ENABLE_DIP_LIB ON CACHE BOOL "Enable DIP library" FORCE)

  add_subdirectory(Models)
  set(BUDDY_MODELS_TEST_DEPENDS
  FileCheck count not
  buddy-opt
  buddy-translate
  mlir-runner
  buddy-lenet-run-test-cpu
  python-package
  )

  add_lit_testsuite(check-e2e "Running E2E tests for Models directory..."
    ${CMAKE_CURRENT_BINARY_DIR}/Models
    DEPENDS ${BUDDY_MODELS_TEST_DEPENDS}
    )
  set_target_properties(check-e2e PROPERTIES FOLDER "Tests")

  add_lit_testsuites(BUDDY_MODELS ${CMAKE_CURRENT_SOURCE_DIR}/Models DEPENDS ${BUDDY_MODELS_TEST_DEPENDS})

endif()
