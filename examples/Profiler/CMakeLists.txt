

add_executable(buddy-mobilenetv3-profiling buddy-mobilenetv3-main.cpp)

set(BUDDY_MOBILENETV3_PROFILING_LIBS  
  mlir_c_runner_utils 
  BuddyRuntime
  BuddyLibDIP 
  ${PNG_LIBRARIES})
target_link_libraries(buddy-mobilenetv3-profiling ${BUDDY_MOBILENETV3_PROFILING_LIBS})


add_custom_command(
    TARGET buddy-mobilenetv3-profiling
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:buddy-mobilenetv3-profiling>
        ${CMAKE_CURRENT_SOURCE_DIR}/buddy-mobilenetv3-profiling
    COMMENT "Copying target_executable to current source dir"
)