# Locate FlatBuffers headers. Prefer the same prefix as flatc, or the active
# tt-mlir toolchain, so generated RAX headers match the headers used by C++.
find_path(FLATBUFFERS_INCLUDE_DIR flatbuffers/flatbuffers.h
  HINTS ${BUDDY_FLATBUFFERS_INCLUDE_HINTS} /usr/include /usr/local/include
  DOC "FlatBuffers include directory"
)
if(NOT FLATBUFFERS_INCLUDE_DIR)
  message(FATAL_ERROR
    "FlatBuffers headers not found. Install:\n"
    "  sudo apt install libflatbuffers-dev")
endif()
message(STATUS "Found FlatBuffers headers: ${FLATBUFFERS_INCLUDE_DIR}")

# A cross compiler searches its sysroot, not the host's /usr/include, and
# CMake does not pass implicit include directories such as /usr/include with
# -I. Header-only FlatBuffers is reached there through a directory of its own
# (as for the RVV plugins in buddy_model.cmake), as /usr/include holds host
# headers.
set(BUDDY_RAX_FLATBUFFERS_INCLUDE_DIR "${FLATBUFFERS_INCLUDE_DIR}")
if(CMAKE_CROSSCOMPILING)
  set(BUDDY_RAX_FLATBUFFERS_INCLUDE_DIR "${CMAKE_BINARY_DIR}/flatbuffers-include")
  file(MAKE_DIRECTORY "${BUDDY_RAX_FLATBUFFERS_INCLUDE_DIR}")
  file(CREATE_LINK "${FLATBUFFERS_INCLUDE_DIR}/flatbuffers"
       "${BUDDY_RAX_FLATBUFFERS_INCLUDE_DIR}/flatbuffers" SYMBOLIC)
endif()

# flatc emits rax_generated.h; CMake renames to RAX.h for #include "buddy/runtime/rax/RAX.h"
set(RAX_INCLUDE_ROOT "${CMAKE_BINARY_DIR}/runtime/include")
set(RAX_GEN_DIR "${RAX_INCLUDE_ROOT}/buddy/runtime/rax")
set(RAX_H "${RAX_GEN_DIR}/RAX.h")

file(MAKE_DIRECTORY "${RAX_GEN_DIR}")

add_custom_command(
  OUTPUT "${RAX_H}"
  COMMAND "${FLATC_EXECUTABLE}" --cpp --gen-all -o "${RAX_GEN_DIR}"
          "${CMAKE_CURRENT_SOURCE_DIR}/rax.fbs"
  COMMAND "${CMAKE_COMMAND}" -E rename
          "${RAX_GEN_DIR}/rax_generated.h"
          "${RAX_H}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/rax.fbs"
  COMMENT "Generating buddy/runtime/rax/RAX.h from rax.fbs"
  VERBATIM
)

add_custom_target(buddy-rax-gen DEPENDS "${RAX_H}")

add_library(buddy_rax INTERFACE)
add_dependencies(buddy_rax buddy-rax-gen)

target_include_directories(buddy_rax INTERFACE
  "${BUDDY_RAX_FLATBUFFERS_INCLUDE_DIR}"  # flatbuffers headers (libflatbuffers-dev)
  "${RAX_INCLUDE_ROOT}"         # buddy/runtime/rax/RAX.h (build tree)
)

install(DIRECTORY ${RAX_INCLUDE_ROOT}/buddy
        DESTINATION include/buddy-mlir/
        COMPONENT buddy_runtime)
install(TARGETS buddy_rax
        EXPORT BuddyMLIRTargets
        COMPONENT buddy_runtime)
