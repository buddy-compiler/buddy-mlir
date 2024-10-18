add_executable(buddy-mobilenetv3-profiling buddy-mobilenetv3-main.cpp)

set(BUDDY_MOBILENETV3_PROFILING_LIBS  
  mlir_c_runner_utils 
  BuddyRuntime
  BuddyLibDIP 
  ${PNG_LIBRARIES})
target_link_libraries(buddy-mobilenetv3-profiling ${BUDDY_MOBILENETV3_PROFILING_LIBS})
set_target_properties(buddy-mobilenetv3-profiling PROPERTIES LINK_FLAGS "-rdynamic")

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/buddy-mobilenetv3-profiling
  COMMAND ${CMAKE_COMMAND} -E copy_if_different                   
    $<TARGET_FILE:buddy-mobilenetv3-profiling>
    ${CMAKE_CURRENT_SOURCE_DIR}/buddy-mobilenetv3-profiling
  DEPENDS buddy-mobilenetv3-profiling                             
  COMMENT "Copying target_executable to current source dir"
)

add_custom_target(copy_buddy_mobilenetv3 ALL
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buddy-mobilenetv3-profiling
  COMMENT "Ensuring buddy-mobilenetv3-profiling is copied after build"
)
