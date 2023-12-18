include(AddMLIRPython)

declare_mlir_python_sources(BuddyPythonSources)
declare_mlir_python_sources(BuddyPythonSources.Compiler
    ADD_TO_PARENT BuddyPythonSources
    ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/buddy"
    SOURCES
        compiler/__init__.py
        compiler/frontend.py
        compiler/ops/linalg.py
        compiler/ops/math.py
        compiler/ops/tosa.py
)

add_mlir_python_modules(BuddyPythonModules
  ROOT_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/python_packages/buddy"
  INSTALL_PREFIX "python_packages/buddy"
  DECLARED_SOURCES BuddyPythonSources
)
