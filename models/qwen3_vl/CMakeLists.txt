# ══════════════════════════════════════════════════════════════════════════════
# Qwen3-VL — self-contained model integration (NOT buddy_add_model)
#
# Qwen3-VL is a vision-language model (ViT vision encoder + DeepStack -> patch
# merger -> dense Qwen3 text decoder with interleaved MRoPE). The multimodal
# pre/post-processing and DeepStack feature injection don't fit the plain
# prefill/decode buddy_add_model LLM pipeline, so the codegen is wired here
# directly (mirroring the self-contained models/whisper design), producing the
# same runtime contract: a runner plugin + a .rax loaded by buddy-cli.
#
# Targets:
#   buddy_models_qwen3_vl(_runner)  - runner static lib + C-ABI plugin (always)
#   qwen3_vl_rax                    - import -> compile -> shims -> stage -> .rax
#                                     (needs BUDDY_QWEN3_VL_MODEL_PATH + python)
# ══════════════════════════════════════════════════════════════════════════════

# ── Runner static library + C-ABI plugin ──────────────────────────────────────
set(LIB_TARGET buddy_models_qwen3_vl)

add_library(${LIB_TARGET} STATIC
  "${CMAKE_CURRENT_SOURCE_DIR}/Qwen3VLRunner.cpp"
)
target_include_directories(${LIB_TARGET} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${BUDDY_SOURCE_DIR}/frontend/Interfaces>
  $<BUILD_INTERFACE:${BUDDY_BINARY_DIR}/frontend/Interfaces>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/buddy-mlir>
)
target_compile_features(${LIB_TARGET} PUBLIC cxx_std_17)
target_link_libraries(${LIB_TARGET} PUBLIC
  buddy_runtime_core
  buddy_runtime_llm
  ${CMAKE_DL_LIBS}
  LLVMSupport
)

set(RUNNER_PLUGIN_TARGET buddy_models_qwen3_vl_runner)
add_library(${RUNNER_PLUGIN_TARGET} SHARED
  "${CMAKE_CURRENT_SOURCE_DIR}/Qwen3VLRunnerPlugin.cpp"
)
set_target_properties(${RUNNER_PLUGIN_TARGET} PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  OUTPUT_NAME "qwen3_vl_runner"
  PREFIX ""
)
target_link_libraries(${RUNNER_PLUGIN_TARGET} PRIVATE ${LIB_TARGET})
target_compile_features(${RUNNER_PLUGIN_TARGET} PRIVATE cxx_std_17)

# ── Codegen pipeline: import -> compile -> shims -> stage -> .rax ──────────────
if(NOT Python3_EXECUTABLE)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()

set(BUDDY_QWEN3_VL_MODEL_PATH "" CACHE PATH
  "Local HuggingFace Qwen3-VL-2B-Instruct directory (required to build qwen3_vl_rax)")
set(BUDDY_QWEN3_VL_SPEC
  "${CMAKE_CURRENT_SOURCE_DIR}/specs/instruct_2b.json"
  CACHE FILEPATH "Qwen3-VL variant spec JSON")

set(_Q_CG  "${CMAKE_CURRENT_SOURCE_DIR}/codegen")
set(_Q_CODEGEN "${_Q_CG}/qwen3_vl_codegen.py")
set(_Q_ART "${CMAKE_CURRENT_BINARY_DIR}/artifacts")
set(_Q_VIS "${_Q_ART}/vision")
set(_Q_DEC "${_Q_ART}/decoder_rt")
set(_Q_OMP "${LLVM_BINARY_DIR}/runtimes/runtimes-bins/openmp/runtime/src")
set(_Q_TEST_IMG "${CMAKE_CURRENT_SOURCE_DIR}/test_text.png")
set(_Q_PYTHONPATH "${CMAKE_BINARY_DIR}/python_packages")
if(DEFINED ENV{PYTHONPATH})
  set(_Q_PYTHONPATH "${_Q_PYTHONPATH}:$ENV{PYTHONPATH}")
endif()

set(_Q_ENV
  BUDDY_MLIR_BUILD_DIR=${CMAKE_BINARY_DIR}
  LLVM_MLIR_BUILD_DIR=${LLVM_BINARY_DIR}
  PYTHONPATH=${_Q_PYTHONPATH}
  CUDA_VISIBLE_DEVICES=
  QWEN3_VL_OUT_DIR=${_Q_ART}
  QWEN3_VL_PKG=${CMAKE_CURRENT_BINARY_DIR}
  QWEN3_VL_SPEC=${BUDDY_QWEN3_VL_SPEC}
  QWEN3_VL_MODEL_PATH=${BUDDY_QWEN3_VL_MODEL_PATH})

# Stage 1: PyTorch -> MLIR + weights (vision encoder + runtime-position decoder).
add_custom_command(
  OUTPUT ${_Q_VIS}/vision_forward.mlir ${_Q_VIS}/vision_subgraph0.mlir
         ${_Q_VIS}/vision_arg0.data
         ${_Q_DEC}/decoder_forward.mlir ${_Q_DEC}/decoder_subgraph0.mlir
         ${_Q_DEC}/decoder_arg0.data ${_Q_DEC}/embed_table.bin
  COMMAND ${CMAKE_COMMAND} -E make_directory ${_Q_ART}
  COMMAND ${CMAKE_COMMAND} -E env ${_Q_ENV}
          ${Python3_EXECUTABLE} ${_Q_CODEGEN} import-vision
  COMMAND ${CMAKE_COMMAND} -E env ${_Q_ENV}
          ${Python3_EXECUTABLE} ${_Q_CODEGEN} import-decoder-rt --seq-len 160
  DEPENDS ${_Q_CODEGEN} ${BUDDY_QWEN3_VL_SPEC} ${_Q_TEST_IMG}
  COMMENT "[qwen3_vl] import PyTorch -> MLIR + weights"
  VERBATIM)

# Stage 2: MLIR -> object (via the piped buddy compile pipeline helper).
function(_qwen3vl_obj dir name)
  add_custom_command(
    OUTPUT ${dir}/${name}.o
    COMMAND bash ${_Q_CG}/lower_to_obj.sh
            $<TARGET_FILE:buddy-opt> ${LLVM_TOOLS_BINARY_DIR}
            ${dir}/${name}.mlir ${dir}/${name}.o
    DEPENDS ${dir}/${name}.mlir ${_Q_CG}/lower_to_obj.sh buddy-opt
    COMMENT "[qwen3_vl] compile ${name}.mlir -> .o"
    VERBATIM)
endfunction()
_qwen3vl_obj(${_Q_VIS} vision_subgraph0)
_qwen3vl_obj(${_Q_VIS} vision_forward)
_qwen3vl_obj(${_Q_DEC} decoder_subgraph0)
_qwen3vl_obj(${_Q_DEC} decoder_forward)

# Stage 3: flat-C shim shared libraries around the compiled kernels.
function(_qwen3vl_shim out src obj1 obj2)
  add_custom_command(
    OUTPUT ${out}
    COMMAND ${CMAKE_CXX_COMPILER} -shared -fPIC -std=c++17 -O2
            -I${BUDDY_SOURCE_DIR}/frontend/Interfaces
            ${src} ${obj1} ${obj2}
            -L${LLVM_LIBRARY_DIR} -lmlir_c_runner_utils -L${_Q_OMP} -lomp
            -Wl,-rpath,${LLVM_LIBRARY_DIR} -Wl,-rpath,${_Q_OMP} -o ${out}
    DEPENDS ${src} ${obj1} ${obj2}
    COMMENT "[qwen3_vl] link ${out}"
    VERBATIM)
endfunction()
_qwen3vl_shim(${_Q_VIS}/vision_shim.so ${_Q_CG}/vision_shim.cpp
              ${_Q_VIS}/vision_forward.o ${_Q_VIS}/vision_subgraph0.o)
_qwen3vl_shim(${_Q_DEC}/decoder_shim.so ${_Q_CG}/decoder_shim.cpp
              ${_Q_DEC}/decoder_forward.o ${_Q_DEC}/decoder_subgraph0.o)

# Stage 4: stage the package + RHAL manifest + rax-pack.
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/qwen3_vl.rax
  COMMAND ${CMAKE_COMMAND} -E env ${_Q_ENV}
          RAX_PACK=$<TARGET_FILE:rax-pack>
          QWEN3_VL_RUNNER_SO=$<TARGET_FILE:${RUNNER_PLUGIN_TARGET}>
          ${Python3_EXECUTABLE} ${_Q_CODEGEN} stage
  DEPENDS ${_Q_CODEGEN}
          ${_Q_VIS}/vision_shim.so ${_Q_DEC}/decoder_shim.so
          ${_Q_VIS}/vision_arg0.data ${_Q_DEC}/decoder_arg0.data
          ${_Q_DEC}/embed_table.bin ${RUNNER_PLUGIN_TARGET} rax-pack
  COMMENT "[qwen3_vl] stage package + rax-pack -> qwen3_vl.rax"
  VERBATIM)

add_custom_target(qwen3_vl_rax
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/qwen3_vl.rax)
