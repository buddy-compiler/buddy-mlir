if(BUDDY_ENABLE_OPENCV)
  find_package(OpenCV REQUIRED CONFIG)
  find_package(JPEG REQUIRED)
  find_package(PNG REQUIRED)
  include_directories(${OpenCV_INCLUDE_DIRS})
  include_directories(${JPEG_INCLUDE_DIRS})
  include_directories(${PNG_INCLUDE_DIRS})
endif()

add_executable(correlation2D correlation2D.cpp)
target_link_libraries(correlation2D ${OpenCV_LIBS} BuddyLibDIP)

add_executable(correlationFFT2D correlationFFT2D.cpp)
target_link_libraries(correlationFFT2D ${OpenCV_LIBS} BuddyLibDIP)

add_executable(rotation2D rotation2D.cpp)
target_link_libraries(rotation2D ${OpenCV_LIBS} ${JPEG_LIBRARY} ${PNG_LIBRARY}  BuddyLibDIP)

add_executable(resize2D resize2D.cpp)
target_link_libraries(resize2D ${OpenCV_LIBS} ${JPEG_LIBRARY} ${PNG_LIBRARY}  BuddyLibDIP)

add_executable(morph2D morph2D.cpp)
target_link_libraries(morph2D ${OpenCV_LIBS} BuddyLibDIP)

add_executable(dip-all DIPAll.cpp)
target_link_libraries(dip-all ${OpenCV_LIBS} BuddyLibDIP)

add_executable(opencv-all OpenCVAll.cpp)
target_link_libraries(opencv-all ${OpenCV_LIBS})
