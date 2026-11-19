#!/usr/bin/env bash
# Build a standalone SE workload from the existing AME matmul example.
set -euo pipefail

ame_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ame_repo_dir=$(cd -- "$ame_script_dir/../../.." && pwd)
ame_output_dir=${1:-"$ame_repo_dir/build/ame-gem5"}
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

# AME_gem5 fixes its logical square at 4x4; compile the matching source example.
cp -- "$ame_script_dir/../ame-matmul-4x4.mlir" \
  "$ame_output_dir/ame-matmul-4x4.mlir"
"$BUDDY_OPT" "$ame_output_dir/ame-matmul-4x4.mlir" \
  --lower-ame --convert-func-to-llvm --reconcile-unrealized-casts \
  -o "$ame_output_dir/ame-matmul-llvm.mlir"
"$BUDDY_TRANSLATE" --buddy-to-llvmir "$ame_output_dir/ame-matmul-llvm.mlir" \
  -o "$ame_output_dir/ame-matmul.ll"

ame_features=+m,+xame,+xame-gem5
"$BUDDY_LLC" "$ame_output_dir/ame-matmul.ll" \
  -mtriple=riscv64 -mattr="$ame_features" -target-abi=lp64 \
  -verify-machineinstrs -filetype=asm -o "$ame_output_dir/ame-matmul.s"
"$BUDDY_LLC" "$ame_output_dir/ame-matmul.ll" \
  -mtriple=riscv64 -mattr="$ame_features" -target-abi=lp64 \
  -verify-machineinstrs -filetype=obj -o "$ame_output_dir/ame-matmul.o"

# The runtime has XAME assembly; a stock GNU assembler cannot assemble it.
"$LLVM_MC" -triple=riscv64 -mattr="$ame_features" -target-abi=lp64 \
  -filetype=obj "$ame_script_dir/runtime.S" -o "$ame_output_dir/runtime.o"
# Only the checker is C. All matrix computation comes from the MLIR kernel.
"$CLANG" --target=riscv64-unknown-linux-gnu -march=rv64im -mabi=lp64 \
  -O2 -ffreestanding -fno-builtin -fno-stack-protector -fno-pic -fno-pie \
  -fno-vectorize -fno-slp-vectorize -mcmodel=medany -msmall-data-limit=0 \
  -Wall -Wextra -Werror -c "$ame_script_dir/main.c" -o "$ame_output_dir/main.o"
"$RISCV_LD" -m elf64lriscv -static -e _start --build-id=none \
  --no-dynamic-linker --no-relax "$ame_output_dir/runtime.o" \
  "$ame_output_dir/main.o" "$ame_output_dir/ame-matmul.o" \
  -o "$ame_output_dir/ame-matmul.elf"

echo "ELF built: $ame_output_dir/ame-matmul.elf"
echo "Run it with AME_gem5 and $ame_script_dir/run-se.py; building alone is not a numerical PASS."
