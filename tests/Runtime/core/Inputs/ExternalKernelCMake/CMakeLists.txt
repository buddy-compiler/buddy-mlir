cmake_minimum_required(VERSION 3.20)
project(BuddyExternalKernelConsumer LANGUAGES CXX)
include(GNUInstallDirs)
# The public target must raise a downstream project's older language request.
set(CMAKE_CXX_STANDARD 14)

set(CONSUMER_MODE "build-tree" CACHE STRING "build-tree or installed")
if(CONSUMER_MODE STREQUAL "build-tree")
  if(NOT BUDDY_SOURCE_DIR)
    message(FATAL_ERROR "Set BUDDY_SOURCE_DIR to the Buddy checkout")
  endif()
  set(BUDDY_BINARY_DIR "${CMAKE_BINARY_DIR}")
  # Headers only: no build-host C/C++ executable is run by this fixture.
  include_directories("${BUDDY_SOURCE_DIR}/llvm/mlir/include")
  add_subdirectory("${BUDDY_SOURCE_DIR}/runtime" runtime)
  # Exercise the real main-package template as well as the runtime-only package.
  include(CMakePackageConfigHelpers)
  configure_package_config_file(
    "${BUDDY_SOURCE_DIR}/cmake/BuddyMLIRConfig.cmake.in"
    "${CMAKE_BINARY_DIR}/BuddyMLIRConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/BuddyMLIR")
  install(FILES "${CMAKE_BINARY_DIR}/BuddyMLIRConfig.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/BuddyMLIR"
    COMPONENT buddy_runtime)
elseif(CONSUMER_MODE STREQUAL "main-package")
  find_package(BuddyMLIR CONFIG REQUIRED)
else()
  find_package(BuddyExternalKernel CONFIG REQUIRED)
endif()

# Deliberately rely on the imported target for includes, C++17, and dl linkage.
add_executable(external-kernel-consumer Consumer.cpp)
target_link_libraries(external-kernel-consumer PRIVATE
  BuddyMLIR::buddy_runtime_external_kernel)

# Shared Runner/plugin consumers must also be able to link the static archive.
add_library(external-kernel-shared-consumer SHARED Consumer.cpp)
target_link_libraries(external-kernel-shared-consumer PRIVATE
  BuddyMLIR::buddy_runtime_external_kernel)

add_executable(external-kernel-shared-loader SharedLoader.cpp)
target_link_libraries(external-kernel-shared-loader PRIVATE ${CMAKE_DL_LIBS})
