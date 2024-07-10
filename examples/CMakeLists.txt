if(BUDDY_EXAMPLES)
  add_subdirectory(ConvOpt)
  add_subdirectory(DIPDialect)
  add_subdirectory(DAPDialect)
endif()

if(BUDDY_LLAMA_EXAMPLES)
  add_subdirectory(BuddyLlama)
endif()

if (BUDDY_BERT_EXAMPLES)
  add_subdirectory(BuddyBert)
endif()

if (BUDDY_LENET_EXAMPLES)
  add_subdirectory(BuddyLeNet)
endif()

if(BUDDY_WHISPER_EXAMPLES)
  add_subdirectory(BuddyWhisper)
endif()

if (BUDDY_MOBILENETV3_EXAMPLES)
  add_subdirectory(BuddyMobileNetV3)
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
  mlir-cpu-runner
  )

add_lit_testsuite(check-examples "Checking the buddy-mlir examples..."
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${BUDDY_EXAMPLES_DEPENDS}
  )
set_target_properties(check-examples PROPERTIES FOLDER "Examples")

add_lit_testsuites(BUDDY-EXAMPLES ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${BUDDY_EXAMPLES_DEPENDS})
