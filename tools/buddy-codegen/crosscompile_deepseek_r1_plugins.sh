#!/bin/bash
# Cross-compile the DeepSeek R1 runner/serving C++ plugins for RISC-V.
#
# Why this exists: with IS_RVV_CROSSCOMPILE=ON the CMake runner/serving
# plugin targets are still compiled by the host compiler, producing x86-64
# .so files that cannot be dlopen'd on the RISC-V board. Until the CMake
# plumbing is fixed, build them manually with the RISC-V toolchain and
# re-run the deepseek_r1_rax (Stage 4) target so the plugins are packed
# into the .rax.
#
# Usage: edit SRC/BLD/TC below if needed, then:
#   bash tools/buddy-codegen/crosscompile_deepseek_r1_plugins.sh
#   cmake --build <riscv-build-dir> --target deepseek_r1_rax
set -euo pipefail

SRC=${SRC:-/tmp/buddy-deepseek-rax-inplace-pr}          # repo worktree
BLD=${BLD:-$SRC/build-riscv-packed-f32-8}               # riscv build dir
TC=${TC:-/home/huangguoning/code/buddy-mlir/build-rv/thirdparty/riscv-gnu-toolchain}
LLVM_RV_LIBS=${LLVM_RV_LIBS:-/home/huangguoning/code/buddy-mlir/llvm/build-cross-mlir-rv/lib}
FB_INCLUDE=${FB_INCLUDE:-/tmp/buddy-flatbuffers-include}
CXX=$TC/bin/riscv64-unknown-linux-gnu-g++

LLVM_INC="-I$(readlink -f "$SRC"/llvm/llvm/include) -I$(readlink -f "$SRC"/llvm/build/include) -I$(readlink -f "$SRC"/llvm/mlir/include) -I$(readlink -f "$SRC"/llvm/build/tools/mlir/include"

COMMON_FLAGS="-O2 -fPIC -fno-semantic-interposition -std=gnu++17 -UNDEBUG
 -D_DEBUG -D_GLIBCXX_ASSERTIONS -D_GLIBCXX_USE_CXX11_ABI=1
 -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS
 -w"

INCLUDES="-I$SRC/midend/include -I$SRC/midend/include/Interface -I$SRC/midend/include/Dialect
 -I$BLD/midend/include/Dialect -I$BLD -I$SRC/lib -I$SRC/thirdparty/include
 -I$SRC/frontend/Interfaces $LLVM_INC
 -I$BLD/models/deepseek_r1/generated -I$SRC/models/deepseek_r1/include
 -I$BLD/bin/frontend/Interfaces -I$SRC/runtime/include -I$BLD/runtime/include
 -I$FB_INCLUDE"

# Static libstdc++/libgcc: the board's libstdc++ predates the toolchain's
# GLIBCXX version. LLVMSupport is needed for llvm::json/Support used by the
# manifest and chat-template code.
LINK_FLAGS="-static-libstdc++ -static-libgcc -Wl,-rpath,\$ORIGIN -lm $LLVM_RV_LIBS/libLLVMSupport.a"

# Sources shared by both plugins (the runner library).
LIB_SOURCES="$BLD/models/deepseek_r1/generated/ModelSession.cpp
 $SRC/models/deepseek_r1/DeepSeekR1Runner.cpp
 $SRC/models/deepseek_r1/DeepSeekR1ResidentModel.cpp
 $SRC/runtime/llm/TextGeneration.cpp
 $SRC/runtime/llm/InteractiveSession.cpp"

build_plugin() {
  local out=$1; shift
  echo "[cross] building $out"
  $CXX $COMMON_FLAGS $INCLUDES "$@" $LIB_SOURCES -shared -o "$out" $LINK_FLAGS
}

build_plugin "$BLD/models/deepseek_r1/deepseek_r1_runner.so" \
  "$SRC/models/deepseek_r1/DeepSeekR1RunnerPlugin.cpp"
build_plugin "$BLD/models/deepseek_r1/deepseek_r1_serving.so" \
  "$SRC/models/deepseek_r1/DeepSeekR1ResidentModelPlugin.cpp"

file "$BLD/models/deepseek_r1/deepseek_r1_runner.so" \
     "$BLD/models/deepseek_r1/deepseek_r1_serving.so"
