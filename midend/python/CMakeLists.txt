include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `mlir_standalone`
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=buddy_mlir.")

################################################################################
# Structural groupings.
################################################################################

declare_mlir_python_sources(BuddyMLIRPythonSources)
declare_mlir_python_sources(BuddyMLIRPythonSources.Dialects
  ADD_TO_PARENT BuddyMLIRPythonSources)

################################################################################
# Dialect bindings
################################################################################

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT BuddyMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/buddy_mlir"
  TD_FILE dialects/BudBinding.td
  SOURCES
    dialects/bud.py
  DIALECT_NAME bud)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT BuddyMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/buddy_mlir"
  TD_FILE dialects/DAPBinding.td
  SOURCES
    dialects/dap.py
  DIALECT_NAME dap)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT BuddyMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/buddy_mlir"
  TD_FILE dialects/DIPBinding.td
  SOURCES
    dialects/dip.py
  DIALECT_NAME dip)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT BuddyMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/buddy_mlir"
  TD_FILE dialects/GemminiBinding.td
  SOURCES
    dialects/gemmini.py
  DIALECT_NAME gemmini)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT BuddyMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/buddy_mlir"
  TD_FILE dialects/RVVBinding.td
  SOURCES
    dialects/rvv.py
  DIALECT_NAME rvv)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT BuddyMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/buddy_mlir"
  TD_FILE dialects/ScheBinding.td
  SOURCES
    dialects/sche.py
  DIALECT_NAME sche)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT BuddyMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/buddy_mlir"
  TD_FILE dialects/VectorExpBinding.td
  SOURCES
    dialects/vector_exp.py
  DIALECT_NAME vector_exp)

################################################################################
# Python extensions.
# The sources for these are all in lib/python/Bindings, but since they have to
# be rebuilt for each package and integrate with the source setup here, we
# just reference them here instead of having ordered, cross package target
# dependencies.
################################################################################

declare_mlir_python_extension(BuddyMLIRPythonSources.Dialects.Bud.Pybind
  MODULE_NAME _budDialects
  ADD_TO_PARENT BuddyMLIRPythonSources.Dialects.bud
  SOURCES
    Bindings/DialectBud.cpp
  EMBED_CAPI_LINK_LIBS
    BuddyMLIRCAPI
)

declare_mlir_python_extension(BuddyMLIRPythonSources.Dialects.DAP.Pybind
  MODULE_NAME _dapDialects
  ADD_TO_PARENT BuddyMLIRPythonSources.Dialects.dap
  SOURCES
    Bindings/DialectDAP.cpp
  EMBED_CAPI_LINK_LIBS
    BuddyMLIRCAPI
)

declare_mlir_python_extension(BuddyMLIRPythonSources.Dialects.DIP.Pybind
  MODULE_NAME _dipDialects
  ADD_TO_PARENT BuddyMLIRPythonSources.Dialects.dip
  SOURCES
    Bindings/DialectDIP.cpp
  EMBED_CAPI_LINK_LIBS
    BuddyMLIRCAPI
)

declare_mlir_python_extension(BuddyMLIRPythonSources.Dialects.Gemmini.Pybind
  MODULE_NAME _gemminiDialects
  ADD_TO_PARENT BuddyMLIRPythonSources.Dialects.gemmini
  SOURCES
    Bindings/DialectGemmini.cpp
  EMBED_CAPI_LINK_LIBS
    BuddyMLIRCAPI
)

declare_mlir_python_extension(BuddyMLIRPythonSources.Dialects.RVV.Pybind
  MODULE_NAME _rvvDialects
  ADD_TO_PARENT BuddyMLIRPythonSources.Dialects.rvv
  SOURCES
    Bindings/DialectRVV.cpp
  EMBED_CAPI_LINK_LIBS
    BuddyMLIRCAPI
)

declare_mlir_python_extension(BuddyMLIRPythonSources.Dialects.Sche.Pybind
  MODULE_NAME _scheDialects
  ADD_TO_PARENT BuddyMLIRPythonSources.Dialects.sche
  SOURCES
    Bindings/DialectSche.cpp
  EMBED_CAPI_LINK_LIBS
    BuddyMLIRCAPI
)

declare_mlir_python_extension(BuddyMLIRPythonSources.Dialects.VectorExp.Pybind
  MODULE_NAME _vector_expDialects
  ADD_TO_PARENT BuddyMLIRPythonSources.Dialects.vector_exp
  SOURCES
    Bindings/DialectVectorExp.cpp
  EMBED_CAPI_LINK_LIBS
    BuddyMLIRCAPI
)

# declare_mlir_python_extension(BuddyMLIRPythonSources.Extension
#   MODULE_NAME _buddyRegisterEverything
#   ADD_TO_PARENT BuddyMLIRPythonSources
#   SOURCES
#     Bindings/RegisterEverything.cpp
#   EMBED_CAPI_LINK_LIBS
#     BuddyMLIRCAPI
# )

################################################################################
# Common CAPI dependency DSO.
# All python extensions must link through one DSO which exports the CAPI, and
# this must have a globally unique name amongst all embeddors of the python
# library since it will effectively have global scope.
#
# The presence of this aggregate library is part of the long term plan, but its
# use needs to be made more flexible.
#
# TODO: Upgrade to the aggregate utility in https://reviews.llvm.org/D106419
# once ready.
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