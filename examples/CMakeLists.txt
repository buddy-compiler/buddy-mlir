set(BUDDY_EXAMPLES_USE_EXTERNAL_TOOLS OFF CACHE BOOL
  "Use external buddy tools (buddy-opt/buddy-translate) from MLIR_RELEASE_PKG/bin")
if(BUDDY_EXAMPLES_USE_EXTERNAL_TOOLS)
  if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(FATAL_ERROR "Configure from buddy-mlir top-level (cmake -S <buddy-mlir> ...).")
  endif()

  if(DEFINED ENV{MLIR_RELEASE_PKG})
    list(PREPEND CMAKE_PREFIX_PATH "$ENV{MLIR_RELEASE_PKG}")
  endif()

  find_package(LLVM CONFIG REQUIRED)
  find_package(MLIR CONFIG REQUIRED)
  find_package(BuddyMLIR CONFIG REQUIRED)
  find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)

  if(NOT TARGET buddy-opt)
    add_custom_target(buddy-opt DEPENDS "${BUDDY_BINARY_DIR}/buddy-opt")
  endif()
  if(NOT TARGET buddy-translate)
    add_custom_target(buddy-translate DEPENDS "${BUDDY_BINARY_DIR}/buddy-translate")
  endif()

  include_directories(${BUDDY_MLIR_INTERFACE_DIR})
  link_directories(${BUDDY_MLIR_LIB_DIR})
endif()

if(BUDDY_EXAMPLES)
  add_subdirectory(ConvOpt)
  add_subdirectory(DIPDialect)
  add_subdirectory(DAPDialect)
endif()

if(BUDDY_LLAMA_EXAMPLES)
  add_subdirectory(BuddyLlama)
endif()

if(BUDDY_DEEPSEEKR1_EXAMPLES)
  add_subdirectory(BuddyDeepSeekR1)
endif()

if(BUDDY_QWEN3_EXAMPLES)
  add_subdirectory(BuddyQwen3)
endif()

if(BUDDY_TRANSFORMER_EXAMPLES)
  add_subdirectory(BuddyTransformer)
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

if (BUDDY_RESNET_EXAMPLES)
  add_subdirectory(BuddyResNet18)
endif()

if (BUDDY_YOLO26_EXAMPLES)
  add_subdirectory(BuddyYOLO26)
endif()

if (BUDDY_STABLE_DIFFUSION_EXAMPLES)
  add_subdirectory(BuddyStableDiffusion)
endif()

if (BUDDY_ONEDNN_EXAMPLES)
  add_subdirectory(BuddyOneDNN)
endif()

if(BUDDY_DSL_EXAMPLES)
  add_subdirectory(ToyDSL)
endif()

if(BUDDY_MLIR_ENABLE_TESTS)
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
  mlir-runner
  )

if(BUDDY_MLIR_ENABLE_PYTHON_PACKAGES AND MLIR_ENABLE_BINDINGS_PYTHON)
  list(APPEND BUDDY_TEST_DEPENDS python-package-buddy-mlir)
endif()

add_lit_testsuite(check-examples "Checking the buddy-mlir examples..."
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${BUDDY_EXAMPLES_DEPENDS}
  )
set_target_properties(check-examples PROPERTIES FOLDER "Examples")

add_lit_testsuites(BUDDY-EXAMPLES ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${BUDDY_EXAMPLES_DEPENDS})
endif()
