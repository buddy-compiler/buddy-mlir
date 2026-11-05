#===- buddy_rvv_vlen.cmake - The VLEN of RISC-V Vector model builds ------===#
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
#===----------------------------------------------------------------------===#
#
# The VLEN of a RISC-V Vector model build is set in one place,
# BUDDY_RISCV_VLEN (empty: any VLEN). Everything else follows it: the llc
# options of the model's code, the VIR lowering
# (compile_pipeline.kernels_lower_vir), the checks of code written for one
# VLEN (the A100 code of "prefill_ime", the A210 kernels), and the VLEN check
# of the model's libraries (runtime/rvv/BuddyRvvVlen.c).
#
#   buddy_rvv_vlen_llc_attrs(<var> LLC_ATTRS <llc options> [VLEN <vlen>]
#                            [CONTEXT <name>])
#
# sets <var> to the llc options with the VLEN options of <vlen> (+v,
# +zvl<vlen>b, -riscv-v-vector-bits-max=<vlen>) when it is given: a VLEN is
# for code with V, and the +v makes it seen when only an -mcpu would imply V
# (compile_pipeline.rvv_enabled). The llc options must not
# have VLEN options of their own (a model's LLC_ATTRS): they stop the
# configuration.
#
#   buddy_rvv_vlen_required(<var> LLC_ATTRS <llc options> PYTHON <python>
#                           CODEGEN_DIR <tools/buddy-codegen> [CONTEXT <name>])
#
# sets <var> to the exact VLEN of the final llc options, read as llc reads
# them (compile_pipeline.exact_rvv_vlen), or to "" (not RVV, or no exact
# VLEN: code valid on any VLEN). Options that disagree stop the
# configuration.
#
#===----------------------------------------------------------------------===#

function(buddy_rvv_vlen_llc_attrs out)
  cmake_parse_arguments(A "" "LLC_ATTRS;VLEN;CONTEXT" "" ${ARGN})
  if(A_LLC_ATTRS MATCHES "(\\+zvl[0-9]+b|-riscv-v-vector-bits-(min|max))")
    message(FATAL_ERROR
      "${A_CONTEXT}: the llc options \"${A_LLC_ATTRS}\" set a VLEN "
      "(${CMAKE_MATCH_1}); the VLEN of a build is BUDDY_RISCV_VLEN.")
  endif()
  set(_attrs "${A_LLC_ATTRS}")
  if(A_VLEN)
    string(APPEND _attrs
      " -mattr=+v,+zvl${A_VLEN}b -riscv-v-vector-bits-max=${A_VLEN}")
    string(STRIP "${_attrs}" _attrs)
  endif()
  set(${out} "${_attrs}" PARENT_SCOPE)
endfunction()

function(buddy_rvv_vlen_required out)
  cmake_parse_arguments(A "" "LLC_ATTRS;PYTHON;CODEGEN_DIR;CONTEXT" "" ${ARGN})
  execute_process(
    COMMAND "${A_PYTHON}" -c
            "import sys; sys.path.insert(0, sys.argv[1]); import compile_pipeline as c; v = c.exact_rvv_vlen(sys.argv[2].split()); print('' if v is None else v)"
            "${A_CODEGEN_DIR}" "${A_LLC_ATTRS}"
    OUTPUT_VARIABLE _vlen
    ERROR_VARIABLE _error
    RESULT_VARIABLE _status
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _status EQUAL 0)
    string(REGEX MATCH "[^\n]+\n*$" _error "${_error}")
    string(STRIP "${_error}" _error)
    message(FATAL_ERROR
      "${A_CONTEXT}: the llc options \"${A_LLC_ATTRS}\": ${_error}")
  endif()
  set(${out} "${_vlen}" PARENT_SCOPE)
endfunction()
