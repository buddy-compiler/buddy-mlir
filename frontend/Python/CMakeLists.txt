# Recursively retrieve all python files from the current directory.
file(GLOB_RECURSE ALL_PY_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.py")

foreach(FILE ${ALL_PY_FILES})
    # Get the directory of the current file.
    get_filename_component(DIR "${FILE}" DIRECTORY)
    # Set the destination directory for the target file.
    set(DEST "${BUDDY_MLIR_PYTHON_PACKAGES_DIR}/buddy/compiler/${DIR}")
    # Copy the file into the destination directory.
    file(COPY ${FILE} DESTINATION ${DEST})
endforeach()

# -----------------------------------------------------------------------------
# Runtime helper libraries (shared) for Python-driven execution.
# -----------------------------------------------------------------------------
add_library(buddy_rng_utils SHARED runtime/buddy_rng_utils.cpp)
target_compile_features(buddy_rng_utils PRIVATE cxx_std_17)
set_target_properties(
  buddy_rng_utils
  PROPERTIES
    OUTPUT_NAME "buddy_rng_utils"
    LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY}
)
