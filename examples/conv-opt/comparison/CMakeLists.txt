add_executable(opencv-filter2D opencv-filter2D.cpp)
target_link_libraries(opencv-filter2D ${OpenCV_LIBS})

add_executable(boost_gil_conv2d boost_gil_conv2d.cpp)
find_package(Boost)
if(Boost_FOUND)
  message(STATUS "boost include path is : ${Boost_INCLUDE_DIRS}")
  include_directories(${Boost_INCLUDE_DIRS})
else()
  message(WARNING "boost not found.")
endif()
find_package(PNG REQUIRED)
target_link_libraries(boost_gil_conv2d 
  PNG::PNG 
  ${BOOST_LIBRARIES}
  ${OpenCV_LIBS})
