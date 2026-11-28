# -----------------------------------------------------------------------------
# External-call kernel libraries used by Python-driven execution.
# -----------------------------------------------------------------------------
add_subdirectory(external_kernels)

# RAX schema + FlatBuffers interface target (used by rax-pack, rax-inspect,
# and the model-agnostic runtime core).
add_subdirectory(rax)

# Optional AOT host kernels, with a Buddy-generated fallback at the call site.
add_library(buddy_runtime_external_kernel STATIC core/ExternalKernel.cpp)
add_library(BuddyMLIR::buddy_runtime_external_kernel ALIAS
            buddy_runtime_external_kernel)
set_target_properties(buddy_runtime_external_kernel PROPERTIES
  POSITION_INDEPENDENT_CODE ON)
target_include_directories(buddy_runtime_external_kernel PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/buddy-mlir>)
target_link_libraries(buddy_runtime_external_kernel PRIVATE ${CMAKE_DL_LIBS})
target_compile_features(buddy_runtime_external_kernel PUBLIC cxx_std_17)
install(TARGETS buddy_runtime_external_kernel EXPORT BuddyExternalKernelTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT buddy_runtime)
install(EXPORT BuddyExternalKernelTargets
        FILE BuddyExternalKernelTargets.cmake
        NAMESPACE BuddyMLIR::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/BuddyExternalKernel
        COMPONENT buddy_runtime)
install(FILES cmake/BuddyExternalKernelConfig.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/BuddyExternalKernel
        COMPONENT buddy_runtime)

# RAX runtime dependencies must remain loaded while a runner or server model is
# alive. Keep the dlopen lifecycle in a shared module used by both frontends.
add_library(buddy_runtime_rax_dependencies STATIC
  core/RaxDependenciesHandle.cpp
)
target_include_directories(buddy_runtime_rax_dependencies PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/buddy-mlir>
)
target_link_libraries(buddy_runtime_rax_dependencies PRIVATE
  ${CMAKE_DL_LIBS}
)
target_compile_features(buddy_runtime_rax_dependencies PUBLIC cxx_std_17)
install(TARGETS buddy_runtime_rax_dependencies
        EXPORT BuddyMLIRTargets
        COMPONENT buddy_runtime)

# Model-agnostic runtime headers: InferenceRunner, ModelManifest, BufferPool.
# Public includes live under runtime/include/ → #include "buddy/runtime/core/*.h"
# (CMake stays here — not under include/, which should remain installable headers only.)
add_library(buddy_runtime_core INTERFACE)
target_include_directories(buddy_runtime_core INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/buddy-mlir>
)
target_link_libraries(buddy_runtime_core INTERFACE
  buddy_rax
)
target_compile_features(buddy_runtime_core INTERFACE cxx_std_17)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION include/buddy-mlir
        COMPONENT buddy_runtime)
install(TARGETS buddy_runtime_core
        EXPORT BuddyMLIRTargets
        COMPONENT buddy_runtime)

# LLM runtime utilities (text generation, interactive REPL).
add_subdirectory(llm)
