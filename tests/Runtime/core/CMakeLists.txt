add_library(buddy-rax-dependencies-test-plugin MODULE
  RaxDependenciesTestPlugin.cpp
)
set_target_properties(buddy-rax-dependencies-test-plugin PROPERTIES
  PREFIX ""
)

_add_test_executable(buddy-rax-dependencies-test
  RaxDependenciesHandleTest.cpp
  LINK_LIBS
    buddy_runtime_rax_dependencies
    ${CMAKE_DL_LIBS}
)
target_compile_definitions(buddy-rax-dependencies-test PRIVATE
  BUDDY_RAX_DEPENDENCY_TEST_PLUGIN="$<TARGET_FILE:buddy-rax-dependencies-test-plugin>"
)
add_dependencies(buddy-rax-dependencies-test
  buddy-rax-dependencies-test-plugin
)

# ABI fixtures run without a Triton installation. The optional full example
# separately tests a real exported kernel against Buddy-generated Softmax.
foreach(mode RANGE 0 13)
  add_library(buddy-external-kernel-plugin-${mode} MODULE
    ExternalKernelTestPlugin.cpp)
  target_include_directories(buddy-external-kernel-plugin-${mode} PRIVATE
    ${PROJECT_SOURCE_DIR}/runtime/include)
  target_compile_definitions(buddy-external-kernel-plugin-${mode} PRIVATE
    TEST_MODE=${mode})
endforeach()

_add_test_executable(buddy-rax-kernel-metadata-test
  Inputs/RaxKernelMetadataCheck.cpp
  LINK_LIBS buddy_runtime_core buddy_runtime_rax_dependencies
    buddy_runtime_external_kernel)
add_dependencies(buddy-rax-kernel-metadata-test
  buddy-rax-dependencies-test-plugin)
target_compile_definitions(buddy-rax-kernel-metadata-test PRIVATE
  BUDDY_RAX_METADATA_TEST_PLUGIN="$<TARGET_FILE:buddy-rax-dependencies-test-plugin>")
_add_test_executable(buddy-external-kernel-test ExternalKernelTest.cpp
  LINK_LIBS buddy_runtime_external_kernel)
target_compile_definitions(buddy-external-kernel-test PRIVATE
  TEST_VALID="$<TARGET_FILE:buddy-external-kernel-plugin-0>"
  TEST_BAD_ABI="$<TARGET_FILE:buddy-external-kernel-plugin-1>"
  TEST_BAD_TARGET="$<TARGET_FILE:buddy-external-kernel-plugin-2>"
  TEST_BAD_HOST="$<TARGET_FILE:buddy-external-kernel-plugin-3>"
  TEST_FAILED="$<TARGET_FILE:buddy-external-kernel-plugin-4>"
  TEST_NO_SYMBOL="$<TARGET_FILE:buddy-external-kernel-plugin-5>"
  TEST_FEATURE_PLUGIN="$<TARGET_FILE:buddy-external-kernel-plugin-6>")
target_compile_definitions(buddy-external-kernel-test PRIVATE
  TEST_NULL_QUERY="$<TARGET_FILE:buddy-external-kernel-plugin-7>"
  TEST_BAD_SIZE="$<TARGET_FILE:buddy-external-kernel-plugin-8>"
  TEST_NO_CALLBACK="$<TARGET_FILE:buddy-external-kernel-plugin-9>"
  TEST_NO_INVOKE="$<TARGET_FILE:buddy-external-kernel-plugin-10>"
  TEST_SHAPE="$<TARGET_FILE:buddy-external-kernel-plugin-11>"
  TEST_UNSUPPORTED="$<TARGET_FILE:buddy-external-kernel-plugin-12>"
  TEST_INVALID="$<TARGET_FILE:buddy-external-kernel-plugin-13>")
foreach(mode RANGE 0 13)
  add_dependencies(buddy-external-kernel-test buddy-external-kernel-plugin-${mode})
endforeach()
