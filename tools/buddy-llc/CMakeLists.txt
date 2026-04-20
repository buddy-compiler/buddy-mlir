if(NOT TARGET llc)
  message(FATAL_ERROR
    "The configured LLVM package does not export the llc executable target.")
endif()

get_target_property(BUDDY_LLC_IMPORTED_CONFIGS llc IMPORTED_CONFIGURATIONS)
if(BUDDY_LLC_IMPORTED_CONFIGS)
  list(GET BUDDY_LLC_IMPORTED_CONFIGS 0 BUDDY_LLC_IMPORTED_CONFIG)
  string(TOUPPER "${BUDDY_LLC_IMPORTED_CONFIG}" BUDDY_LLC_IMPORTED_CONFIG)
  get_target_property(BUDDY_LLC_SOURCE llc
    "IMPORTED_LOCATION_${BUDDY_LLC_IMPORTED_CONFIG}")
else()
  get_target_property(BUDDY_LLC_SOURCE llc IMPORTED_LOCATION)
endif()

if(NOT BUDDY_LLC_SOURCE)
  message(FATAL_ERROR
    "Failed to resolve the imported llc executable location.")
endif()

set(BUDDY_LLC_BINARY
  "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/buddy-llc${CMAKE_EXECUTABLE_SUFFIX}")

add_custom_command(
  OUTPUT "${BUDDY_LLC_BINARY}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
  COMMAND ${CMAKE_COMMAND} -E rm -f "${BUDDY_LLC_BINARY}"
  COMMAND ${CMAKE_COMMAND} -E copy "${BUDDY_LLC_SOURCE}" "${BUDDY_LLC_BINARY}"
  DEPENDS "${BUDDY_LLC_SOURCE}"
  VERBATIM
)

add_custom_target(buddy-llc DEPENDS "${BUDDY_LLC_BINARY}")

export_executable_symbols_for_plugins(buddy-llc)

add_llvm_tool(buddy-llc-ext
  buddy-llc-ext.cpp
  NewPMDriverExt.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  SUPPORT_PLUGINS
  )

export_executable_symbols_for_plugins(buddy-llc-ext)

install(PROGRAMS "${BUDDY_LLC_SOURCE}"
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  RENAME "buddy-llc${CMAKE_EXECUTABLE_SUFFIX}")
