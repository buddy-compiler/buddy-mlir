# DeepSeek R1 tensor-parallel package. This build graph intentionally lives
# outside buddy_add_model(): TP=2 produces rank-local plans, parameter packs,
# scheduled RHAL modules, and RAX artifacts rather than one ordinary model.

include(${CMAKE_SOURCE_DIR}/tools/buddy-codegen/cmake/buddy_model.cmake)
include(ProcessorCount)

if(NOT BUDDY_RUNTIME_ENABLE_MPI)
  message(FATAL_ERROR
    "models/deepseek_r1_tp2 requires BUDDY_RUNTIME_ENABLE_MPI=ON")
endif()
if(NOT BUDDY_MLIR_ENABLE_PYTHON_PACKAGES)
  message(FATAL_ERROR
    "models/deepseek_r1_tp2 requires BUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON")
endif()
if(IS_RVV_CROSSCOMPILE)
  message(FATAL_ERROR
    "models/deepseek_r1_tp2 has only been validated for native builds")
endif()
if(NOT Python3_EXECUTABLE)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()

ProcessorCount(_BUDDY_DSR1_TP2_DEFAULT_COMPILE_JOBS)
if(_BUDDY_DSR1_TP2_DEFAULT_COMPILE_JOBS EQUAL 0)
  set(_BUDDY_DSR1_TP2_DEFAULT_COMPILE_JOBS 1)
endif()

set(BUDDY_DSR1_TP2_SPEC
  "${CMAKE_SOURCE_DIR}/models/deepseek_r1/specs/f32.json"
  CACHE FILEPATH "DeepSeek R1 source model spec reused by the TP=2 package")
set(BUDDY_DSR1_TP2_HF_CONFIG ""
  CACHE FILEPATH "Optional HuggingFace config.json for DeepSeek R1 TP=2")
set(BUDDY_DSR1_TP2_LOCAL_MODEL ""
  CACHE PATH "Optional local HuggingFace DeepSeek R1 model directory")
set(BUDDY_DSR1_TP2_COMPILE_JOBS
  "${_BUDDY_DSR1_TP2_DEFAULT_COMPILE_JOBS}"
  CACHE STRING "Parallel MLIR compilation jobs for DeepSeek R1 TP=2")
set(BUDDY_DSR1_TP2_LLC_ATTRS "-mcpu=native"
  CACHE STRING "llc target attributes for DeepSeek R1 TP=2")

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
          "import json,sys; v=json.load(open(sys.argv[1]))['variant']; print({'f32':'float32','f16':'float16','bf16':'bfloat16'}.get(v,''))"
          "${BUDDY_DSR1_TP2_SPEC}"
  OUTPUT_VARIABLE _BUDDY_DSR1_TP2_PACK_DTYPE
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _BUDDY_DSR1_TP2_SPEC_RESULT)
if(NOT _BUDDY_DSR1_TP2_SPEC_RESULT EQUAL 0 OR
   _BUDDY_DSR1_TP2_PACK_DTYPE STREQUAL "")
  message(FATAL_ERROR
    "DeepSeek R1 TP=2 currently supports only plain f32/f16/bf16 specs")
endif()

set(_TP2_BIN "${CMAKE_CURRENT_BINARY_DIR}")
set(_TP2_GEN "${_TP2_BIN}/generated")
set(_TP2_LAYER "${_TP2_BIN}/layer_partitioned")
set(_TP2_OBJ "${_TP2_BIN}/obj_partitioned")
set(_TP2_CONFIG "${_TP2_GEN}/config.json")
set(_TP2_SESSION_H
  "${_TP2_GEN}/buddy/runtime/models/ModelSession.h")
set(_TP2_SESSION_CPP "${_TP2_GEN}/ModelSession.cpp")
set(_TP2_IMPORT_STAMP "${_TP2_BIN}/.buddy_import_done")
set(_TP2_RAX_SHIMS "${_TP2_GEN}/RaxShims.cpp")
set(_TP2_MODEL_SO "${_TP2_BIN}/deepseek_r1_tp2_model.so")
set(_TP2_RUNNER_SO_NAME "deepseek_r1_tp2_runner.so")

set(_TP2_GEN_CONFIG_CMD
  "${Python3_EXECUTABLE}" "${BUDDY_CODEGEN_DIR}/gen_config.py"
  --spec "${BUDDY_DSR1_TP2_SPEC}" -o "${_TP2_CONFIG}")
if(BUDDY_DSR1_TP2_HF_CONFIG)
  list(APPEND _TP2_GEN_CONFIG_CMD
    --hf-config "${BUDDY_DSR1_TP2_HF_CONFIG}")
endif()
add_custom_command(
  OUTPUT "${_TP2_CONFIG}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${_TP2_GEN}"
  COMMAND ${_TP2_GEN_CONFIG_CMD}
  DEPENDS
    "${BUDDY_DSR1_TP2_SPEC}"
    "${BUDDY_CODEGEN_DIR}/gen_config.py"
  COMMENT "[deepseek_r1_tp2] Generating config.json"
  VERBATIM)

add_custom_command(
  OUTPUT "${_TP2_SESSION_H}" "${_TP2_SESSION_CPP}"
  COMMAND "${Python3_EXECUTABLE}" "${BUDDY_CODEGEN_DIR}/gen_session.py"
          --config "${_TP2_CONFIG}" --output-dir "${_TP2_GEN}"
  DEPENDS "${_TP2_CONFIG}" "${BUDDY_CODEGEN_DIR}/gen_session.py"
  COMMENT "[deepseek_r1_tp2] Generating model constants"
  VERBATIM)

add_library(buddy_models_deepseek_r1_tp2 STATIC
  DeepSeekR1TPRunner.cpp
  DeepSeekR1RaxRunner.cpp
  DeepSeekR1RaxSession.cpp
  "${_TP2_SESSION_H}"
)
set_target_properties(buddy_models_deepseek_r1_tp2 PROPERTIES
  POSITION_INDEPENDENT_CODE ON)
target_include_directories(buddy_models_deepseek_r1_tp2 PUBLIC
  "${_TP2_GEN}"
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${BUDDY_SOURCE_DIR}/frontend/Interfaces>
  $<BUILD_INTERFACE:${BUDDY_BINARY_DIR}/frontend/Interfaces>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/buddy-mlir>
)
target_compile_features(buddy_models_deepseek_r1_tp2 PUBLIC cxx_std_17)
target_link_libraries(buddy_models_deepseek_r1_tp2 PUBLIC
  buddy_runtime_core
  buddy_runtime_execution
  buddy_runtime_llm
  buddy_runtime_mpi
  ${CMAKE_DL_LIBS}
  LLVMSupport
)

add_library(buddy_models_deepseek_r1_tp2_runner SHARED
  DeepSeekR1TPRunnerPlugin.cpp)
set_target_properties(buddy_models_deepseek_r1_tp2_runner PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${_TP2_BIN}"
  RUNTIME_OUTPUT_DIRECTORY "${_TP2_BIN}"
  OUTPUT_NAME "deepseek_r1_tp2_runner"
  PREFIX "")
target_link_libraries(buddy_models_deepseek_r1_tp2_runner PRIVATE
  buddy_models_deepseek_r1_tp2)
target_compile_features(buddy_models_deepseek_r1_tp2_runner PRIVATE cxx_std_17)

set(_TP2_RUNTIME_PLANS)
set(_TP2_PARAMETER_PACKS)
set(_TP2_RUNTIME_PLAN_ARGS)
foreach(_rank RANGE 0 1)
  foreach(_phase IN ITEMS forward_prefill forward_decode)
    set(_plan "${_TP2_LAYER}/runtime/rank${_rank}_${_phase}.json")
    list(APPEND _TP2_RUNTIME_PLANS "${_plan}")
    list(APPEND _TP2_RUNTIME_PLAN_ARGS --runtime-plan "${_plan}")
  endforeach()
  list(APPEND _TP2_PARAMETER_PACKS
    "${_TP2_BIN}/rank${_rank}_params_${_BUDDY_DSR1_TP2_PACK_DTYPE}.data")
endforeach()

set(_TP2_IMPORT_ENV
  "${CMAKE_COMMAND}" -E env
  "PYTHONPATH=${CMAKE_BINARY_DIR}/python_packages")
if(BUDDY_DSR1_TP2_LOCAL_MODEL)
  list(APPEND _TP2_IMPORT_ENV
    "DEEPSEEKR1_MODEL_PATH=${BUDDY_DSR1_TP2_LOCAL_MODEL}")
endif()
set(_TP2_IMPORT_ARGS
  --experimental-template-partitioned
  --skip-full-mlir
  --tensor-parallel-size 2)
if(BUDDY_MODEL_REUSE_WEIGHTS)
  list(APPEND _TP2_IMPORT_ARGS --reuse-existing-weights)
endif()
set(_TP2_IMPORT_DEPS
  "${_TP2_CONFIG}"
  "${BUDDY_CODEGEN_DIR}/import_model.py")
if(TARGET python-package-buddy)
  list(APPEND _TP2_IMPORT_DEPS python-package-buddy)
endif()
add_custom_command(
  OUTPUT
    "${_TP2_IMPORT_STAMP}"
    ${_TP2_RUNTIME_PLANS}
    ${_TP2_PARAMETER_PACKS}
  BYPRODUCTS
    "${_TP2_LAYER}/partition_manifest.json"
    "${_TP2_LAYER}/forward_prefill.mlir"
    "${_TP2_LAYER}/forward_decode.mlir"
  COMMAND ${_TP2_IMPORT_ENV}
          "${Python3_EXECUTABLE}" "${BUDDY_CODEGEN_DIR}/import_model.py"
          --config "${_TP2_CONFIG}" --output-dir "${_TP2_BIN}"
          ${_TP2_IMPORT_ARGS}
  COMMAND "${CMAKE_COMMAND}" -E touch "${_TP2_IMPORT_STAMP}"
  DEPENDS ${_TP2_IMPORT_DEPS}
  COMMENT "[deepseek_r1_tp2] Importing rank-local TP=2 artifacts"
  VERBATIM)

add_custom_command(
  OUTPUT "${_TP2_RAX_SHIMS}"
  COMMAND "${Python3_EXECUTABLE}" "${BUDDY_CODEGEN_DIR}/gen_rax_shims.py"
          ${_TP2_RUNTIME_PLAN_ARGS} -o "${_TP2_RAX_SHIMS}"
  DEPENDS
    "${_TP2_IMPORT_STAMP}"
    ${_TP2_RUNTIME_PLANS}
    "${BUDDY_CODEGEN_DIR}/gen_rax_shims.py"
  COMMENT "[deepseek_r1_tp2] Generating RAX ABI shims"
  VERBATIM)

set(_TP2_OPENMP_ARGS)
if(BUDDY_OPENMP_RUNTIME_LIBRARY)
  list(APPEND _TP2_OPENMP_ARGS
    --openmp-runtime-lib "${BUDDY_OPENMP_RUNTIME_LIBRARY}")
endif()
add_custom_command(
  OUTPUT "${_TP2_MODEL_SO}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${_TP2_OBJ}"
  COMMAND "${Python3_EXECUTABLE}" "${BUDDY_CODEGEN_DIR}/compile_pipeline.py"
          --config "${_TP2_CONFIG}"
          --compile-partitioned
          --link
          --mlir-dir "${_TP2_LAYER}"
          --output-dir "${_TP2_OBJ}"
          --output-so "${_TP2_MODEL_SO}"
          --buddy-opt "${BUDDY_BINARY_DIR}/buddy-opt"
          --llvm-tools-dir "${LLVM_TOOLS_BINARY_DIR}"
          "--llc-attrs=${BUDDY_DSR1_TP2_LLC_ATTRS}"
          --cxx "${CMAKE_CXX_COMPILER}"
          --llvm-lib-dir "${LLVM_LIBRARY_DIR}"
          ${_TP2_OPENMP_ARGS}
          ${_TP2_RUNTIME_PLAN_ARGS}
          --rax-shims "${_TP2_RAX_SHIMS}"
          -j "${BUDDY_DSR1_TP2_COMPILE_JOBS}"
  COMMAND "${Python3_EXECUTABLE}" "${BUDDY_CODEGEN_DIR}/gen_rax_shims.py"
          ${_TP2_RUNTIME_PLAN_ARGS}
          --validate-library "${_TP2_MODEL_SO}"
  DEPENDS
    buddy-opt
    "${_TP2_CONFIG}"
    "${_TP2_IMPORT_STAMP}"
    "${_TP2_RAX_SHIMS}"
    ${_TP2_RUNTIME_PLANS}
    "${BUDDY_CODEGEN_DIR}/compile_pipeline.py"
    "${BUDDY_CODEGEN_DIR}/gen_rax_shims.py"
  COMMENT "[deepseek_r1_tp2] Compiling deepseek_r1_tp2_model.so"
  VERBATIM)
add_custom_target(deepseek_r1_tp2_model_so DEPENDS "${_TP2_MODEL_SO}")

set(_TP2_VOCAB_SRC "${CMAKE_SOURCE_DIR}/examples/BuddyDeepSeekR1/vocab.txt")
set(_TP2_VOCAB "${_TP2_BIN}/vocab.txt")
add_custom_command(
  OUTPUT "${_TP2_VOCAB}"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
          "${_TP2_VOCAB_SRC}" "${_TP2_VOCAB}"
  DEPENDS "${_TP2_VOCAB_SRC}"
  COMMENT "[deepseek_r1_tp2] Copying vocab.txt"
  VERBATIM)

set(_TP2_RAX_PACK_ARGS)
if(BUDDY_RAX_EMBED_PAYLOAD)
  list(APPEND _TP2_RAX_PACK_ARGS --embed-payload)
endif()
set(_TP2_RAX_FILES)
foreach(_rank RANGE 0 1)
  set(_prefill_plan
    "${_TP2_LAYER}/runtime/rank${_rank}_forward_prefill.json")
  set(_decode_plan
    "${_TP2_LAYER}/runtime/rank${_rank}_forward_decode.json")
  set(_rank_rhal "${_TP2_BIN}/rank${_rank}.rhal.mlir")
  set(_rank_rax "${_TP2_BIN}/rank${_rank}.rax")
  list(GET _TP2_PARAMETER_PACKS ${_rank} _rank_parameter_pack)

  add_custom_command(
    OUTPUT "${_rank_rhal}"
    COMMAND "${Python3_EXECUTABLE}" "${BUDDY_CODEGEN_DIR}/gen_manifest.py"
            --config "${_TP2_CONFIG}"
            --runtime-plan "forward_prefill=${_prefill_plan}"
            --runtime-plan "forward_decode=${_decode_plan}"
            --kernel-library "deepseek_r1_tp2_model.so"
            --runner-library "${_TP2_RUNNER_SO_NAME}"
            -o "${_rank_rhal}"
    DEPENDS
      "${_TP2_IMPORT_STAMP}"
      "${_prefill_plan}"
      "${_decode_plan}"
      "${_TP2_CONFIG}"
      "${BUDDY_CODEGEN_DIR}/gen_manifest.py"
    COMMENT "[deepseek_r1_tp2] Generating rank${_rank}.rhal.mlir"
    VERBATIM)

  add_custom_command(
    OUTPUT "${_rank_rax}"
    COMMAND "$<TARGET_FILE:rax-pack>" "${_rank_rhal}"
            -o "${_rank_rax}" ${_TP2_RAX_PACK_ARGS}
    DEPENDS
      rax-pack
      "${_rank_rhal}"
      "${_rank_parameter_pack}"
      "${_TP2_MODEL_SO}"
      buddy_models_deepseek_r1_tp2_runner
      "${_TP2_VOCAB}"
    WORKING_DIRECTORY "${_TP2_BIN}"
    COMMENT "[deepseek_r1_tp2] Packing rank${_rank}.rax"
    VERBATIM)
  list(APPEND _TP2_RAX_FILES "${_rank_rax}")
endforeach()

add_custom_target(deepseek_r1_tp2_rax
  DEPENDS ${_TP2_RAX_FILES} "${_TP2_VOCAB}"
  COMMENT "DeepSeek R1 TP=2 rank-local RAX package -> ${_TP2_BIN}")

install(FILES
  include/buddy/runtime/models/DeepSeekR1TPRunner.h
  include/buddy/runtime/models/DeepSeekR1RaxRunner.h
  include/buddy/runtime/models/DeepSeekR1RaxSession.h
  DESTINATION include/buddy-mlir/buddy/runtime/models
  COMPONENT buddy_runtime)
install(TARGETS buddy_models_deepseek_r1_tp2
  EXPORT BuddyMLIRTargets
  COMPONENT buddy_runtime)
