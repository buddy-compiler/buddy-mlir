if(BUDDY_ENABLE_OPENCV)
  find_package(OpenCV REQUIRED CONFIG)
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

add_executable(correlation2D correlation2D.cpp)
target_link_libraries(correlation2D ${OpenCV_LIBS} BuddyLibDIP)

add_executable(rotation2D rotation2D.cpp)
target_link_libraries(rotation2D ${OpenCV_LIBS} BuddyLibDIP)

add_executable(resize2D resize2D.cpp)
target_link_libraries(resize2D ${OpenCV_LIBS} BuddyLibDIP)

add_executable(morph2D morph2D.cpp)
target_link_libraries(morph2D ${OpenCV_LIBS} BuddyLibDIP)

add_executable(sep_correlation2D sep_correlation2D.cpp)
target_link_libraries(sep_correlation2D ${OpenCV_LIBS} BuddyLibDIP)
