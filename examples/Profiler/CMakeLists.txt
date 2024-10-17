

add_executable(buddy-profile buddy-mobilenetv3-main.cpp)

set(BUDDY_MOBILENETV3_LIBS  mlir_c_runner_utils BuddyLibDIP ${PNG_LIBRARIES})
target_link_libraries(buddy-profile ${BUDDY_MOBILENETV3_LIBS})
