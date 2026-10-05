# ===- CMakeLists.txt - BGE-M3 dense embedding model (optimized build) ------===
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===----------------------------------------------------------------------===
#
# Self-contained BGE-M3 build (issue #888, phase 2). Unlike the generic
# buddy_add_model path in tools/buddy-codegen/cmake/buddy_model.cmake, the
# whole lowering pipeline lives here so BGE-M3-specific optimizations stay
# inside this example and the shared build files remain untouched:
#
#   import (models/bge_m3 codegen) -> forward.o / subgraph0.o
#   -> bge_m3_model.so -> runner/embedding plugins -> bge_m3.rax
#
# Optimized subgraph0 pipeline (x86, all gated by cos = 1.0):
#   * llc -mcpu=native              (was silently scalar SSE2 before)
#   * -bge-m3-batchmatmul-transpose-b-vec  (guarded attention Q@K^T /
#     scores@V vectorization; #12, ~-10%)
#   * -linalg-transpose-tile=tile-size=8   (tiled weight transposes; #13,
#     ~-13% on top, single-thread cycles -75%)
#
# Build with:
#   ninja bge-m3-optimized
# Output: <build>/examples/BuddyBgeM3/bge_m3.rax
#
# ===----------------------------------------------------------------------===

# Reuse the BGE-M3 importer / manifest generator / runner sources from
# models/bge_m3; only the lowering pipeline differs.
set(BGE_M3_SRC "${CMAKE_SOURCE_DIR}/models/bge_m3")
set(BGE_M3_SPEC "${BGE_M3_SRC}/specs/base.json")
set(BGE_M3_PY_PKG_ROOT "${CMAKE_BINARY_DIR}/python_packages")
set(BGE_M3_IMPORT_PYTHONPATH
    "${BGE_M3_PY_PKG_ROOT}:$ENV{PYTHONPATH}")
if(NOT DEFINED BUDDY_BGE_M3_MODEL_PATH)
  set(BUDDY_BGE_M3_MODEL_PATH "" CACHE PATH
    "Local HuggingFace-format BGE-M3 snapshot (tokenizer source)")
endif()

if(NOT Python3_EXECUTABLE)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()

# ── Target platform: llc attributes / link toolchain ─────────────────────
if(IS_RVV_CROSSCOMPILE)
  message(WARNING
    "[bge-m3-optimized] RVV cross-compile is handled by build_seq.sh; "
    "this target only supports the native toolchain.")
elseif(HAVE_LOCAL_RVV)
  set(BGE_M3_LLC_ATTRS -mcpu=native -mattr=+m,+d,+v)
else()
  set(BGE_M3_LLC_ATTRS -mcpu=native)
endif()

# ── Stage 1: import (forward.mlir / subgraph0.mlir / arg0.data) ──────────
add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
    ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
    ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
  COMMAND ${CMAKE_COMMAND} -E env
          "PYTHONPATH=${BGE_M3_IMPORT_PYTHONPATH}"
          ${Python3_EXECUTABLE}
          "${BGE_M3_SRC}/codegen/import-bge-m3.py"
          --spec "${BGE_M3_SPEC}"
          --output-dir "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS
    "${BGE_M3_SRC}/codegen/import-bge-m3.py"
    "${BGE_M3_SPEC}"
  COMMENT "[bge-m3-optimized] Stage 1: importing BGE-M3 -> MLIR + weights"
  VERBATIM
)

# ── Stage 2: forward.mlir -> forward.o ────────────────────────────────────
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, \
            tosa-to-linalg, tosa-to-tensor, tosa-to-arith), \
            empty-tensor-to-alloc-tensor, convert-elementwise-to-linalg)" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -pass-pipeline "builtin.module(func.func(\
            buffer-deallocation-simplification, convert-linalg-to-loops), \
            matmul-parallel-vectorization-optimize, batchmatmul-optimize, \
            eliminate-empty-tensors, func.func(llvm-request-c-wrappers), \
            convert-scf-to-openmp, convert-openmp-to-llvm, \
            convert-math-to-llvm, convert-math-to-libm, convert-scf-to-cf, \
            convert-arith-to-llvm, expand-strided-metadata, \
            finalize-memref-to-llvm, convert-func-to-llvm, \
            reconcile-unrealized-casts)" |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic
            ${BGE_M3_LLC_ATTRS} -O0
            -o ${CMAKE_CURRENT_BINARY_DIR}/forward.o
  DEPENDS
    buddy-opt
    ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
  COMMENT "[bge-m3-optimized] Stage 2: forward.mlir -> forward.o"
  VERBATIM
)

# ── Stage 2: subgraph0.mlir -> subgraph0.o (optimized pipeline) ───────────
# Attention batch matmuls: guarded BGE-M3 transposed-B vectorization.
# Weight transposes: 8x8 tiled cache-friendly copies.
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, \
            tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -test-linalg-transform-patterns=test-decompose-pad-tensor |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -arith-expand
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize=bufferize-function-boundaries
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -matmul-parallel-vectorization-optimize
            -batchmatmul-optimize
            -bge-m3-batchmatmul-transpose-b-vec=vector-size=64
            -linalg-transpose-tile=tile-size=8
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -lower-affine
            -convert-scf-to-openmp
            -convert-linalg-to-loops
            -expand-strided-metadata
            -lower-affine
            -cse
            -convert-vector-to-llvm
            -memref-expand
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic
            ${BGE_M3_LLC_ATTRS} -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.o
  DEPENDS
    buddy-opt
    ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
  COMMENT "[bge-m3-optimized] Stage 2: subgraph0.mlir -> subgraph0.o"
  VERBATIM
)

# ── Stage 3: link bge_m3_model.so ────────────────────────────────────────
# libomp lives in <llvm>/lib/<host-triple>; prefer the located runtime
# library and bake its directory into rpath so the dlopen'ed payload copy
# resolves it even though only the model .so is embedded.
set(BGE_M3_OMP_LINK_ARGS)
set(BGE_M3_OMP_RPATH_ARGS)
if(BUDDY_OPENMP_RUNTIME_LIBRARY)
  list(APPEND BGE_M3_OMP_LINK_ARGS "${BUDDY_OPENMP_RUNTIME_LIBRARY}")
  get_filename_component(_BGE_M3_OMP_DIR
    "${BUDDY_OPENMP_RUNTIME_LIBRARY}" DIRECTORY)
  list(APPEND BGE_M3_OMP_RPATH_ARGS "-Wl,-rpath,${_BGE_M3_OMP_DIR}")
else()
  list(APPEND BGE_M3_OMP_LINK_ARGS -lomp)
  list(APPEND BGE_M3_OMP_RPATH_ARGS
    "-Wl,-rpath,${LLVM_LIBRARY_DIR}/${LLVM_HOST_TRIPLE}")
endif()

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/bge_m3_model.so
  COMMAND ${CMAKE_CXX_COMPILER}
            -shared -fPIC
            -Wl,-soname,bge_m3_model.so
            -Wl,--allow-multiple-definition
            -o ${CMAKE_CURRENT_BINARY_DIR}/bge_m3_model.so
            ${CMAKE_CURRENT_BINARY_DIR}/forward.o
            ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.o
            -L${LLVM_LIBRARY_DIR}
            ${BGE_M3_OMP_LINK_ARGS}
            -lmlir_c_runner_utils
            ${BGE_M3_OMP_RPATH_ARGS}
            "-Wl,-rpath,${LLVM_LIBRARY_DIR}"
            -Wl,-rpath,'$ORIGIN' -lm
  DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/forward.o
    ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.o
  COMMENT "[bge-m3-optimized] Stage 3: linking bge_m3_model.so"
  VERBATIM
)

# ── Runner / embedding plugins (sources shared with models/bge_m3) ───────
add_library(bge_m3_runner_opt_core STATIC
  "${BGE_M3_SRC}/BgeM3Runner.cpp"
  "${BGE_M3_SRC}/BgeM3Runtime.cpp"
  "${BGE_M3_SRC}/BgeM3EmbeddingModel.cpp"
)
target_include_directories(bge_m3_runner_opt_core PUBLIC
  "${BGE_M3_SRC}/include"
  "${CMAKE_SOURCE_DIR}/frontend/Interfaces"
  "${CMAKE_BINARY_DIR}/frontend/Interfaces"
)
target_compile_features(bge_m3_runner_opt_core PUBLIC cxx_std_17)
target_link_libraries(bge_m3_runner_opt_core PUBLIC
  buddy_runtime_core
  ${CMAKE_DL_LIBS}
  LLVMSupport
)

add_library(bge_m3_runner_opt SHARED
  "${BGE_M3_SRC}/BgeM3RunnerPlugin.cpp")
set_target_properties(bge_m3_runner_opt PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  OUTPUT_NAME "bge_m3_runner"
  PREFIX "")
target_compile_features(bge_m3_runner_opt PRIVATE cxx_std_17)
target_link_libraries(bge_m3_runner_opt PRIVATE bge_m3_runner_opt_core)

add_library(bge_m3_embedding_opt SHARED
  "${BGE_M3_SRC}/BgeM3EmbeddingModelPlugin.cpp")
set_target_properties(bge_m3_embedding_opt PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  OUTPUT_NAME "bge_m3_embedding"
  PREFIX "")
target_compile_features(bge_m3_embedding_opt PRIVATE cxx_std_17)
target_link_libraries(bge_m3_embedding_opt PRIVATE bge_m3_runner_opt_core)

# ── Stage 4: manifest + rax-pack ─────────────────────────────────────────
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tokenizer.json
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${BUDDY_BGE_M3_MODEL_PATH}/tokenizer.json"
          "${CMAKE_CURRENT_BINARY_DIR}/tokenizer.json"
  DEPENDS "${BUDDY_BGE_M3_MODEL_PATH}/tokenizer.json"
  COMMENT "[bge-m3-optimized] Copying tokenizer.json"
  VERBATIM
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/bge_m3.mlir
  COMMAND ${Python3_EXECUTABLE}
          "${BGE_M3_SRC}/codegen/gen_bge_m3_manifest.py"
          --spec "${BGE_M3_SPEC}"
          --runner-library "bge_m3_runner.so"
          --embedding-library "bge_m3_embedding.so"
          -o "${CMAKE_CURRENT_BINARY_DIR}/bge_m3.mlir"
  DEPENDS
    "${BGE_M3_SRC}/codegen/gen_bge_m3_manifest.py"
    "${BGE_M3_SPEC}"
  COMMENT "[bge-m3-optimized] Stage 4: generating bge_m3.mlir manifest"
  VERBATIM
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/bge_m3.rax
  COMMAND ${BUDDY_BINARY_DIR}/rax-pack
          ${CMAKE_CURRENT_BINARY_DIR}/bge_m3.mlir
          -o ${CMAKE_CURRENT_BINARY_DIR}/bge_m3.rax
          --embed-payload
  DEPENDS
    rax-pack
    ${CMAKE_CURRENT_BINARY_DIR}/bge_m3.mlir
    ${CMAKE_CURRENT_BINARY_DIR}/bge_m3_model.so
    ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
    ${CMAKE_CURRENT_BINARY_DIR}/tokenizer.json
    bge_m3_runner_opt
    bge_m3_embedding_opt
  COMMENT "[bge-m3-optimized] Stage 4: packing bge_m3.rax"
  VERBATIM
)

add_custom_target(bge-m3-optimized
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/bge_m3.rax
  COMMENT "BGE-M3 optimized .rax -> ${CMAKE_CURRENT_BINARY_DIR}/bge_m3.rax"
)
