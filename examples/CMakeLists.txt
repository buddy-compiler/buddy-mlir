if(BUDDY_EXAMPLES)
  add_subdirectory(ConvOpt)
  add_subdirectory(Resize4D)
  add_subdirectory(DIPDialect)
  add_subdirectory(DAPDialect)
endif()

if(BUDDY_DSL_EXAMPLES)
  add_subdirectory(ToyDSL)
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(BUDDY_EXAMPLES_DEPENDS
  FileCheck count not
  buddy-opt
  buddy-translate
  )

add_lit_testsuite(check-examples "Checking the buddy-mlir examples..."
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${BUDDY_EXAMPLES_DEPENDS}
  )
set_target_properties(check-examples PROPERTIES FOLDER "Examples")

add_lit_testsuites(BUDDY-EXAMPLES ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${BUDDY_EXAMPLES_DEPENDS})
