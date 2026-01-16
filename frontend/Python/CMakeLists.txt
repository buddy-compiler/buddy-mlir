# Recursively retrieve all python files from the current directory.
file(GLOB_RECURSE ALL_PY_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.py")

set(PYTHON_OUTPUT_FILES "")
foreach(FILE ${ALL_PY_FILES})
    get_filename_component(REL_DIR "${FILE}" DIRECTORY)
    set(SRC "${CMAKE_CURRENT_SOURCE_DIR}/${FILE}")
    set(DEST_DIR "${BUDDY_MLIR_PYTHON_PACKAGES_DIR}/buddy/compiler/${REL_DIR}")
    set(DEST_FILE "${BUDDY_MLIR_PYTHON_PACKAGES_DIR}/buddy/compiler/${FILE}")

    add_custom_command(
        OUTPUT "${DEST_FILE}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${DEST_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${SRC}" "${DEST_FILE}"
        DEPENDS "${SRC}"
        COMMENT "Syncing ${FILE}"
        VERBATIM
    )

    list(APPEND PYTHON_OUTPUT_FILES "${DEST_FILE}")
endforeach()

add_custom_target(python-package
    ALL DEPENDS ${PYTHON_OUTPUT_FILES}
    COMMENT "Syncing python package files"
)
