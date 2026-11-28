# ===- tests/Pipeline/CMakeLists.txt - Pipeline-parallel CTest suite -------===//
# ===----------------------------------------------------------------------===//

set(_BUDDY_PIPELINE_UNIT_DEPENDS)
set(_BUDDY_PIPELINE_ALL_DEPENDS)

# ------------------------------------------------------------------------------
# Layer 1A: StageManifest parser unit test (no MPI required)
# ------------------------------------------------------------------------------
add_executable(buddy-pipeline-stage-manifest-test
  StageManifestTest.cpp
)
target_compile_features(buddy-pipeline-stage-manifest-test PRIVATE cxx_std_17)
target_link_libraries(buddy-pipeline-stage-manifest-test PRIVATE
  buddy_runtime_core
)
add_test(NAME buddy-pipeline-stage-manifest
  COMMAND buddy-pipeline-stage-manifest-test
)
set_tests_properties(buddy-pipeline-stage-manifest PROPERTIES
  TIMEOUT 30
  PASS_REGULAR_EXPRESSION "StageManifest tests passed"
  LABELS "pipeline;pipeline-unit"
)
list(APPEND _BUDDY_PIPELINE_UNIT_DEPENDS buddy-pipeline-stage-manifest-test)

list(APPEND _BUDDY_PIPELINE_ALL_DEPENDS ${_BUDDY_PIPELINE_UNIT_DEPENDS})

add_custom_target(check-buddy-pipeline-unit
  COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure -L "^pipeline-unit$"
  DEPENDS ${_BUDDY_PIPELINE_UNIT_DEPENDS}
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

add_custom_target(check-buddy-pipeline
  COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure -L "^pipeline$"
  DEPENDS ${_BUDDY_PIPELINE_ALL_DEPENDS}
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)
