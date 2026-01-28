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
