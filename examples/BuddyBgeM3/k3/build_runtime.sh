#!/usr/bin/env bash
# ===- build_runtime.sh - Build BGE-M3 runtime plugins (once, all seqs) -===//
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
# Products (all under $OUT/runtime/):
#   libbuddy_models_bge_m3.a  <- BgeM3Runner.cpp / BgeM3Runtime.cpp /
#                                 BgeM3EmbeddingModel.cpp
#   bge_m3_runner.so          <- BgeM3RunnerPlugin.cpp (buddy-cli)
#   bge_m3_embedding.so       <- BgeM3EmbeddingModelPlugin.cpp (buddy-server)
#
# Pure shell, no python / CMake: only prebuilt clang++ / llvm-ar.
#
# Usage: bash build_runtime.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

RT="$OUT/runtime"
mkdir -p "$RT"

CXX="$LLVMBIN/clang++"
AR="$LLVMBIN/llvm-ar"

# Preflight: tools and generated headers.
miss=0
for t in "$CXX" "$AR" "$BUDDYBIN/rax-pack"; do
  [ -x "$t" ] || { echo "missing tool: $t"; miss=1; }
done
[ "$miss" = 0 ] || {
  echo "-> build buddy-mlir first, e.g. ninja -C $BUILD rax-pack"
  exit 1
}
for d in "$BUILD/midend/include/Dialect"; do
  [ -d "$d" ] || {
    echo "missing generated headers $d (run ninja -C $BUILD first)"
    exit 1
  }
done

INC=(
  -I"$REPO/midend/include"
  -I"$REPO/midend/include/Interface"
  -I"$REPO/midend/include/Dialect"
  -I"$BUILD/midend/include/Dialect"
  -I"$BUILD"
  -I"$REPO/lib"
  -I"$REPO/thirdparty/include"
  -I"$REPO/frontend/Interfaces"
  -I"$LLVM/include"
  -I"$BUILD/models/bge_m3/generated"
  -I"$REPO/models/bge_m3/include"
  -I"$BUILD/bin/frontend/Interfaces"
  -I"$REPO/runtime/include"
  -I"$BUILD/runtime/include"
)
DEFS=(
  -DNDEBUG -D_GLIBCXX_USE_CXX11_ABI=1 -D__STDC_CONSTANT_MACROS
  -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS
)
FLAGS=(
  -std=c++17 -fPIC -fno-semantic-interposition
  -fvisibility-inlines-hidden -O2
)

M="$REPO/models/bge_m3"

echo "[runtime] 1/3 compile model runtime objects"
for f in BgeM3Runner.cpp BgeM3Runtime.cpp BgeM3EmbeddingModel.cpp; do
  echo "        - $f"
  "$CXX" "${DEFS[@]}" "${INC[@]}" "${FLAGS[@]}" \
    -c "$M/$f" -o "$RT/${f%.cpp}.o"
done

echo "[runtime] 2/3 archive libbuddy_models_bge_m3.a"
rm -f "$RT/libbuddy_models_bge_m3.a"
"$AR" qc "$RT/libbuddy_models_bge_m3.a" \
  "$RT/BgeM3Runner.o" "$RT/BgeM3Runtime.o" "$RT/BgeM3EmbeddingModel.o"

echo "[runtime] 3/3 link plugin .so"
LIBS=("$LLVMLIB/libLLVMSupport.a" -lrt -ldl -lpthread -lm)
# libz is optional (LLVM built with -DLLVM_ENABLE_ZLIB=OFF may not need it).
Z="$(find /usr/lib /lib -maxdepth 4 -name 'libz.so' -print -quit \
  2>/dev/null || true)"
[ -n "$Z" ] && LIBS+=("$Z")
LIBS+=("$LLVMLIB/libLLVMDemangle.a")

build_plugin() { # $1=plugin.cpp  $2=out.so
  local src="$1" so="$2"
  "$CXX" "${DEFS[@]}" "${INC[@]}" "${FLAGS[@]}" \
    -c "$M/$src" -o "$RT/${src%.cpp}.o"
  "$CXX" -fPIC -shared -Wl,-soname,"$so" -Wl,-z,defs -Wl,-z,nodelete \
    -o "$RT/$so" "$RT/${src%.cpp}.o" "$RT/libbuddy_models_bge_m3.a" \
    "${LIBS[@]}"
  echo "        -> $RT/$so"
}
build_plugin BgeM3RunnerPlugin.cpp         bge_m3_runner.so
build_plugin BgeM3EmbeddingModelPlugin.cpp bge_m3_embedding.so

echo "[runtime] done -> $RT"
ls -lh "$RT"/*.so

