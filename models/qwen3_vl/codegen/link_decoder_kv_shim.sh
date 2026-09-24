#!/usr/bin/env bash
# ===- link_decoder_kv_shim.sh - Link prefill+decode into decoder_shim.so -===//
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
# Link the four KV decoder objects plus decoder_kv_shim.cpp into one shared
# library (normally named decoder_shim.so so stage/rax/runner paths stay
# unchanged).
#
# Why objcopy is required
# -----------------------
# Both the prefill and decode MLIR graphs export the same entry symbols for
# their fused compute subgraph:
#   subgraph0
#   _mlir_ciface_subgraph0
# Linking them into one .so without renaming makes one definition win and the
# other call the wrong body (historically: decode invoking the prefill
# subgraph → SIGSEGV). We rename before link:
#   prefill -> subgraph0_prefill / _mlir_ciface_subgraph0_prefill
#   decode  -> subgraph0_decode  / _mlir_ciface_subgraph0_decode
# The forward_* objects call those ciface symbols; redefine-sym updates the
# relocations in both forward and subgraph objects.
#
# Usage (host or cross; pass linker flags after --):
#   link_decoder_kv_shim.sh <cxx> <frontend_inc> <kv_dir> <shim.cpp> <out.so> \
#       -- [extra cxx flags and libs...]
#
# <kv_dir> must contain:
#   decoder_prefill_forward.o  decoder_prefill_subgraph0.o
#   decoder_decode_forward.o   decoder_decode_subgraph0.o
#
# ===----------------------------------------------------------------------===//
set -euo pipefail
if [[ $# -lt 5 ]]; then
  echo "usage: $0 <cxx> <frontend_inc> <kv_dir> <shim.cpp> <out.so> -- [link-args...]" >&2
  exit 2
fi
CXX=$1
INC=$2
KV=$3
SHIM=$4
OUT=$5
shift 5
if [[ "${1:-}" == "--" ]]; then
  shift
fi

# Prefer llvm-objcopy next to the clang++ used for the RISC-V link.
OBJCOPY="$(dirname "$CXX")/llvm-objcopy"
if [[ ! -x "$OBJCOPY" ]]; then
  OBJCOPY="${LLVM_OBJCOPY:-llvm-objcopy}"
fi
if ! command -v "$OBJCOPY" >/dev/null 2>&1 && [[ ! -x "$OBJCOPY" ]]; then
  echo "error: llvm-objcopy not found (needed to rename subgraph0 symbols)" >&2
  exit 1
fi

for f in decoder_prefill_forward.o decoder_prefill_subgraph0.o \
         decoder_decode_forward.o decoder_decode_subgraph0.o; do
  if [[ ! -f "$KV/$f" ]]; then
    echo "error: missing $KV/$f" >&2
    exit 1
  fi
done

# Work on copies so repeated links do not double-rename already patched .o files.
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cp "$KV/decoder_prefill_forward.o" "$work/prefill_forward.o"
cp "$KV/decoder_prefill_subgraph0.o" "$work/prefill_subgraph0.o"
cp "$KV/decoder_decode_forward.o" "$work/decode_forward.o"
cp "$KV/decoder_decode_subgraph0.o" "$work/decode_subgraph0.o"

for f in prefill_forward.o prefill_subgraph0.o; do
  "$OBJCOPY" \
    --redefine-sym subgraph0=subgraph0_prefill \
    --redefine-sym _mlir_ciface_subgraph0=_mlir_ciface_subgraph0_prefill \
    "$work/$f"
done
for f in decode_forward.o decode_subgraph0.o; do
  "$OBJCOPY" \
    --redefine-sym subgraph0=subgraph0_decode \
    --redefine-sym _mlir_ciface_subgraph0=_mlir_ciface_subgraph0_decode \
    "$work/$f"
done

"$CXX" -shared -fPIC -std=c++17 -O2 -I"$INC" \
  "$SHIM" \
  "$work/prefill_forward.o" "$work/prefill_subgraph0.o" \
  "$work/decode_forward.o" "$work/decode_subgraph0.o" \
  "$@" \
  -o "$OUT"
echo "[link] wrote $OUT"
