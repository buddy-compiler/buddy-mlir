# Re-run CMake when python files are added/removed
file(GLOB_RECURSE ALL_PY_FILES
     CONFIGURE_DEPENDS
     RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
     "*.py")

set(PYTHON_COPY_OUTPUTS "")

foreach(FILE ${ALL_PY_FILES})
    get_filename_component(DIR "${FILE}" DIRECTORY)

    set(SRC "${CMAKE_CURRENT_SOURCE_DIR}/${FILE}")
    set(DST "${BUDDY_MLIR_PYTHON_PACKAGES_DIR}/buddy/compiler/${FILE}")

    list(APPEND PYTHON_COPY_OUTPUTS "${DST}")

    add_custom_command(
        OUTPUT "${DST}"
        COMMAND ${CMAKE_COMMAND} -E make_directory
                "${BUDDY_MLIR_PYTHON_PACKAGES_DIR}/buddy/compiler/${DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${SRC}" "${DST}"
        DEPENDS "${SRC}"
        COMMENT "Copying ${FILE}"
    )
endforeach()

add_custom_target(copy_python_files ALL
    DEPENDS ${PYTHON_COPY_OUTPUTS}
)
