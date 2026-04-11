add_subdirectory(FrontendGen)
add_subdirectory(Interfaces)

add_library(buddy_frontend_interface INTERFACE)
target_include_directories(buddy_frontend_interface INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Interfaces/buddy>
  $<INSTALL_INTERFACE:include/buddy-mlir>
)
target_link_libraries(buddy_frontend_interface INTERFACE LLVMSupport)
target_compile_features(buddy_frontend_interface INTERFACE cxx_std_17)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Interfaces/buddy
        DESTINATION include/buddy-mlir
        COMPONENT buddy_dev)
install(TARGETS buddy_frontend_interface
        EXPORT BuddyMLIRTargets
        COMPONENT buddy_dev)


if(BUDDY_MLIR_ENABLE_PYTHON_PACKAGES)
  add_subdirectory(Python)
endif()
