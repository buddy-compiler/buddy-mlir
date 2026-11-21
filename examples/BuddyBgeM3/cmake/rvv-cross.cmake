# ===- rvv-cross.cmake - BGE-M3 runner/embedding plugins for riscv64 -===//
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
# ===----------------------------------------------------------------------===//
#
# Cross-compiles the BGE-M3 runner/embedding plugins for riscv64, mirroring
# _buddy_rvv_cross_plugins in tools/buddy-codegen/cmake/buddy_model.cmake:
# the host clang++ compiles the plugin + runner-core sources with
# --target=riscv64 --sysroot/--gcc-toolchain, and links each plugin against
# the riscv64 libLLVMSupport.a / libLLVMDemangle.a of RISCV_LLVM_BUILD_DIR.
#
# Reads from the caller:
#   BGE_M3_SRC, BGE_M3_CROSS_OPTS, RISCV_LLVM_BUILD_DIR,
#   FLATBUFFERS_INCLUDE_DIR, BGE_M3_CROSS_CXX
#   bge_m3_runner_opt_core  (for its INCLUDE_DIRECTORIES)
# Appends the produced .so files to BGE_M3_CROSS_PLUGIN_FILES (parent scope).
#
# ===----------------------------------------------------------------------===//

set(_bgem3_dir "${CMAKE_CURRENT_BINARY_DIR}/rv64")
file(MAKE_DIRECTORY "${_bgem3_dir}")

# Header-only FlatBuffers: keep it reachable under its own directory, so the
# cross compile does not pull in host headers through /usr/include.
set(_bgem3_fb_dir "${_bgem3_dir}/flatbuffers-include")
file(MAKE_DIRECTORY "${_bgem3_fb_dir}")
file(CREATE_LINK "${FLATBUFFERS_INCLUDE_DIR}/flatbuffers"
     "${_bgem3_fb_dir}/flatbuffers" SYMBOLIC)

set(_bgem3_incs
  "-I${RISCV_LLVM_BUILD_DIR}/include"
  "-I${_bgem3_fb_dir}"
  "-I$<JOIN:$<FILTER:$<TARGET_PROPERTY:bge_m3_runner_opt_core,INCLUDE_DIRECTORIES>,EXCLUDE,^/usr(/local)?/include/?$>,$<SEMICOLON>-I>")

set(_bgem3_cxxflags -std=c++17 -O2 -DNDEBUG -fPIC -march=rv64gcv)

# Compile the runner core sources once; both plugins link them.
set(_bgem3_core_srcs
  "${BGE_M3_SRC}/BgeM3Runner.cpp"
  "${BGE_M3_SRC}/BgeM3Runtime.cpp"
  "${BGE_M3_SRC}/BgeM3EmbeddingModel.cpp")
set(_bgem3_core_objs)
foreach(_src ${_bgem3_core_srcs})
  get_filename_component(_name "${_src}" NAME_WE)
  set(_obj "${_bgem3_dir}/${_name}.o")
  add_custom_command(
    OUTPUT "${_obj}"
    COMMAND "${BGE_M3_CROSS_CXX}" ${BGE_M3_CROSS_OPTS} ${_bgem3_cxxflags}
            ${_bgem3_incs}
            -MD -MF "${_obj}.d" -c "${_src}" -o "${_obj}"
    DEPENDS "${_src}"
    DEPFILE "${_obj}.d"
    COMMENT "[bge-m3-optimized] riscv64: compiling ${_name}"
    COMMAND_EXPAND_LISTS
    VERBATIM)
  list(APPEND _bgem3_core_objs "${_obj}")
endforeach()

# Compile + link one plugin.  out_name = basename without .so.
function(_bgem3_cross_plugin _out_name _src)
  get_filename_component(_name "${_src}" NAME_WE)
  set(_obj "${_bgem3_dir}/${_name}.o")
  add_custom_command(
    OUTPUT "${_obj}"
    COMMAND "${BGE_M3_CROSS_CXX}" ${BGE_M3_CROSS_OPTS} ${_bgem3_cxxflags}
            ${_bgem3_incs}
            -MD -MF "${_obj}.d" -c "${_src}" -o "${_obj}"
    DEPENDS "${_src}"
    DEPFILE "${_obj}.d"
    COMMENT "[bge-m3-optimized] riscv64: compiling ${_name}"
    COMMAND_EXPAND_LISTS
    VERBATIM)
  set(_so "${CMAKE_CURRENT_BINARY_DIR}/${_out_name}${CMAKE_SHARED_LIBRARY_SUFFIX}")
  add_custom_command(
    OUTPUT "${_so}"
    COMMAND "${BGE_M3_CROSS_CXX}" ${BGE_M3_CROSS_OPTS} -shared -fPIC -o "${_so}"
            "${_obj}" ${_bgem3_core_objs}
            "${RISCV_LLVM_BUILD_DIR}/lib/libLLVMSupport.a"
            "${RISCV_LLVM_BUILD_DIR}/lib/libLLVMDemangle.a"
            -ldl -lpthread
    DEPENDS "${_obj}" ${_bgem3_core_objs}
    COMMENT "[bge-m3-optimized] riscv64: linking ${_out_name}"
    VERBATIM)
  set(_so_abs "${_so}" PARENT_SCOPE)
endfunction()

_bgem3_cross_plugin(bge_m3_runner "${BGE_M3_SRC}/BgeM3RunnerPlugin.cpp")
set(BGE_M3_CROSS_PLUGIN_FILES "${_so_abs}")
_bgem3_cross_plugin(bge_m3_embedding "${BGE_M3_SRC}/BgeM3EmbeddingModelPlugin.cpp")
list(APPEND BGE_M3_CROSS_PLUGIN_FILES "${_so_abs}")
set(BGE_M3_CROSS_PLUGIN_FILES "${BGE_M3_CROSS_PLUGIN_FILES}" PARENT_SCOPE)
