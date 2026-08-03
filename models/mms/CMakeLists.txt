include(${CMAKE_SOURCE_DIR}/tools/buddy-codegen/cmake/buddy_model.cmake)
include(ProcessorCount)
ProcessorCount(_BUDDY_MMS_DEFAULT_COMPILE_JOBS)
if(_BUDDY_MMS_DEFAULT_COMPILE_JOBS EQUAL 0) set(_BUDDY_MMS_DEFAULT_COMPILE_JOBS 1) endif()
set(BUDDY_MMS_SPEC "${CMAKE_CURRENT_SOURCE_DIR}/specs/f32.json" CACHE FILEPATH "Variant spec")
set(BUDDY_MMS_HF_CONFIG "" CACHE FILEPATH "Optional HF config.json")
set(BUDDY_MMS_LOCAL_MODEL "" CACHE PATH "Optional local HF model dir")
set(BUDDY_MMS_COMPILE_JOBS "${_BUDDY_MMS_DEFAULT_COMPILE_JOBS}" CACHE STRING "Parallel jobs")
set(MMS_BUILD_DIR "" CACHE PATH "Mode A: pre-built .o")
set(MMS_MLIR_DIR "" CACHE PATH "Mode B: pre-generated MLIR")
if(NOT Python3_EXECUTABLE) find_package(Python3 COMPONENTS Interpreter REQUIRED) endif()
buddy_add_model(NAME mms SPEC "${BUDDY_MMS_SPEC}" RUNNER_SRC MMSRunner.cpp
  HF_CONFIG "${BUDDY_MMS_HF_CONFIG}" LOCAL_MODEL "${BUDDY_MMS_LOCAL_MODEL}"
  BUILD_DIR "${MMS_BUILD_DIR}" MLIR_DIR "${MMS_MLIR_DIR}" COMPILE_JOBS "${BUDDY_MMS_COMPILE_JOBS}")
