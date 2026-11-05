#===- buddy_rvv_vlen.cmake - The VLEN check of RVV model libraries -------===#
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
# RISC-V Vector code of an exact VLEN computes wrong results on another VLEN
# (runtime/rvv/BuddyRvvVlen.c). The VLEN a model is for comes from the llc
# options its code is compiled with, the same reading as the VIR lowering
# (compile_pipeline.exact_rvv_vlen), and is 1024 with "prefill_ime" (the
# SpacemiT K3 A100 code, compile_pipeline.A100_VLEN).
#
#   buddy_rvv_vlen_required(<var> LLC_ATTRS <llc options> PYTHON <python>
#                           CODEGEN_DIR <tools/buddy-codegen>
#                           [PREFILL_IME <bool>] [CONTEXT <name>])
#
# sets <var> to that VLEN, or to "" (not RVV, or no exact VLEN: code valid on
# any VLEN). llc options that disagree, or an exact VLEN other than 1024 with
# "prefill_ime", stop the configuration.
#
#===----------------------------------------------------------------------===#

function(buddy_rvv_vlen_required out)
  cmake_parse_arguments(A "" "LLC_ATTRS;PYTHON;CODEGEN_DIR;PREFILL_IME;CONTEXT"
                        "" ${ARGN})
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
  if(A_PREFILL_IME)
    if(_vlen AND NOT _vlen EQUAL 1024)
      message(FATAL_ERROR
        "${A_CONTEXT}: \"prefill_ime\" needs the VLEN of the K3 A100 cores, "
        "1024; the llc options give ${_vlen}.")
    endif()
    set(_vlen 1024)
  endif()
  set(${out} "${_vlen}" PARENT_SCOPE)
endfunction()
