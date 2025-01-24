set(DIP_LIBS ${JPEG_LIBRARY} ${PNG_LIBRARIES} BuddyLibDIP)

if(BUDDY_ENABLE_OPENCV)
  find_package(OpenCV REQUIRED CONFIG)
  include_directories(${OpenCV_INCLUDE_DIRS})

  add_executable(correlation2D correlation2D.cpp)
  target_link_libraries(correlation2D ${OpenCV_LIBS} ${DIP_LIBS})

  add_executable(correlationFFT2D correlationFFT2D.cpp)
  target_link_libraries(correlationFFT2D ${OpenCV_LIBS} ${DIP_LIBS})

  add_executable(morph2D morph2D.cpp)
  target_link_libraries(morph2D ${OpenCV_LIBS} ${DIP_LIBS})

  add_executable(dip-all DIPAll.cpp)
  target_link_libraries(dip-all ${OpenCV_LIBS} ${DIP_LIBS})

  add_executable(opencv-all OpenCVAll.cpp)
  target_link_libraries(opencv-all ${OpenCV_LIBS})
endif()

add_executable(rotation2D rotation2D.cpp)
target_link_libraries(rotation2D ${DIP_LIBS})

add_executable(rotation4D rotation4D.cpp)
target_link_libraries(rotation4D ${DIP_LIBS})

add_executable(resize2D resize2D.cpp)
target_link_libraries(resize2D ${DIP_LIBS})

add_executable(resize4D_nhwc resize4D_nhwc.cpp)
target_link_libraries(resize4D_nhwc ${DIP_LIBS})

add_executable(resize4D_nchw resize4D_nchw.cpp)
target_link_libraries(resize4D_nchw ${DIP_LIBS})
