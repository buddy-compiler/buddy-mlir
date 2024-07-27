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
  mlir-cpu-runner
  )

if(BUDDY_ENABLE_OPENCV)
  list(APPEND BUDDY_TEST_DEPENDS buddy-image-container-test)
endif()

add_lit_testsuite(check-tests "Running the buddy regression tests..."
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${BUDDY_TEST_DEPENDS}
  )
set_target_properties(check-tests PROPERTIES FOLDER "Tests")

add_lit_testsuites(BUDDY ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${BUDDY_TEST_DEPENDS})
