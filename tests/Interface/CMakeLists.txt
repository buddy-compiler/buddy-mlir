function(_add_test_executable name)
  cmake_parse_arguments(ARG
    ""
    ""
    "LINK_LIBS"
    ${ARGN})
  add_executable(${name} ${ARG_UNPARSED_ARGUMENTS})
  target_link_libraries(${name} PRIVATE
      ${ARG_LINK_LIBS})
endfunction()

add_subdirectory(core)
