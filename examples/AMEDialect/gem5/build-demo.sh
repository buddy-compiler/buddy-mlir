#!/usr/bin/env bash
# Link the initialized, printing MLIR demo as a freestanding Linux SE ELF.
set -euo pipefail

ame_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ame_repo_dir=$(cd -- "$ame_script_dir/../../.." && pwd)
ame_output_dir=${1:-"$ame_repo_dir/build/ame-gem5-demo"}
mkdir -p -- "$ame_output_dir"
ame_output_dir=$(cd -- "$ame_output_dir" && pwd)

BUDDY_OPT=${BUDDY_OPT:-"$ame_repo_dir/build/bin/buddy-opt"}
BUDDY_TRANSLATE=${BUDDY_TRANSLATE:-"$ame_repo_dir/build/bin/buddy-translate"}
BUDDY_LLC=${BUDDY_LLC:-"$ame_repo_dir/build/bin/buddy-llc"}
LLVM_MC=${LLVM_MC:-"$ame_repo_dir/llvm/build/bin/llvm-mc"}
CLANG=${CLANG:-"$ame_repo_dir/llvm/build/bin/clang"}
RISCV_LD=${RISCV_LD:-riscv64-unknown-linux-gnu-ld}

for ame_tool in "$BUDDY_OPT" "$BUDDY_TRANSLATE" "$BUDDY_LLC" \
    "$LLVM_MC" "$CLANG" "$RISCV_LD"; do
  if ! command -v -- "$ame_tool" >/dev/null 2>&1; then
    echo "Tool not found: $ame_tool" >&2
    exit 1
  fi
done

cp -- "$ame_script_dir/../ame-matmul-demo.mlir" \
  "$ame_output_dir/ame-matmul-demo.mlir"
"$BUDDY_OPT" "$ame_output_dir/ame-matmul-demo.mlir" --lower-ame \
  -o "$ame_output_dir/ame-matmul-demo-lowered.mlir"
"$BUDDY_OPT" "$ame_output_dir/ame-matmul-demo-lowered.mlir" \
  --convert-linalg-to-loops --lower-affine --convert-scf-to-cf \
  --expand-strided-metadata --lower-affine --convert-cf-to-llvm \
  --convert-arith-to-llvm --convert-math-to-llvm --convert-func-to-llvm \
  --finalize-memref-to-llvm --reconcile-unrealized-casts \
  -o "$ame_output_dir/ame-matmul-demo-llvm.mlir"
"$BUDDY_TRANSLATE" --buddy-to-llvmir \
  "$ame_output_dir/ame-matmul-demo-llvm.mlir" \
  -o "$ame_output_dir/ame-matmul-demo.ll"

ame_features=+m,+xame,+xame-gem5
"$BUDDY_LLC" "$ame_output_dir/ame-matmul-demo.ll" \
  -mtriple=riscv64 -mattr="$ame_features" -target-abi=lp64 \
  -verify-machineinstrs -filetype=asm -o "$ame_output_dir/ame-matmul-demo.s"
"$BUDDY_LLC" "$ame_output_dir/ame-matmul-demo.ll" \
  -mtriple=riscv64 -mattr="$ame_features" -target-abi=lp64 \
  -verify-machineinstrs -filetype=obj -o "$ame_output_dir/ame-matmul-demo.o"

"$LLVM_MC" -triple=riscv64 -mattr="$ame_features" -target-abi=lp64 \
  -filetype=obj "$ame_script_dir/demo-start.S" \
  -o "$ame_output_dir/demo-start.o"
"$CLANG" --target=riscv64-unknown-linux-gnu -march=rv64im -mabi=lp64 \
  -O2 -ffreestanding -fno-builtin -fno-stack-protector -fno-pic -fno-pie \
  -fno-vectorize -fno-slp-vectorize -mcmodel=medany -msmall-data-limit=0 \
  -Wall -Wextra -Werror -c "$ame_script_dir/print.c" \
  -o "$ame_output_dir/print.o"
"$RISCV_LD" -m elf64lriscv -static -e _start --build-id=none \
  --no-dynamic-linker --no-relax "$ame_output_dir/demo-start.o" \
  "$ame_output_dir/print.o" "$ame_output_dir/ame-matmul-demo.o" \
  -o "$ame_output_dir/ame-matmul-demo.elf"

echo "ELF built: $ame_output_dir/ame-matmul-demo.elf"
echo "Run it with AME_gem5 and $ame_script_dir/run-se.py to print and verify C."
