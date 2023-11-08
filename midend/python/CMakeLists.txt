include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `mlir_standalone`
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=buddy_mlir.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(BuddyMLIRPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT BuddyMLIRPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/buddy_mlir"
  TD_FILE dialects/GemminiOps.td
  SOURCES
    dialects/gemmini.py
  DIALECT_NAME gemmini)

################################################################################
# Extensions
################################################################################

declare_mlir_python_extension(BuddyMLIRPythonSources.Extension
  MODULE_NAME _gemminiDialects
  ADD_TO_PARENT BuddyMLIRPythonSources
  SOURCES
    BuddyMLIRModule.cpp
  EMBED_CAPI_LINK_LIBS
    BuddyMLIRCAPI
)

################################################################################
# Generate packages and shared library
# Downstreams typically will not use these, but they are useful for local
# testing.
################################################################################


################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(BuddyMLIRPythonCAPI
  INSTALL_COMPONENT BuddyMLIRPythonModules
  INSTALL_DESTINATION python_packages/mlir_core/buddy_mlir/_mlir_libs
  OUTPUT_DIRECTORY "${BUDDY_BUILD_DIR}/midend/python_packages/mlir_core/buddy_mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    BuddyMLIRPythonSources
    MLIRPythonSources
    
    MLIRPythonExtension.RegisterEverything
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(BuddyMLIRPythonModules
  ROOT_PREFIX "${BUDDY_BUILD_DIR}/midend/python_packages/mlir_core/buddy_mlir"
  INSTALL_PREFIX "python_packages/mlir_core/buddy_mlir"
  DECLARED_SOURCES
    BuddyMLIRPythonSources
    
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    BuddyMLIRPythonCAPI
    MLIRPythonCAPI
  )