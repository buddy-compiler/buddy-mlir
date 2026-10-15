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
