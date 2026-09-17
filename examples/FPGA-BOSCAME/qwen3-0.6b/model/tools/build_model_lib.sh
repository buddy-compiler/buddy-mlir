#!/usr/bin/env bash
# Build the static library the model graph calls into, then prove the call chain
# links: generated adapters + archive, checked for resolved symbols.
#
# The library is the union of
#   * the shared 72-case kernels at the model's shapes (linear, rmsnorm), and
#   * the model-only specialisations from model/tools/model_kernel_cases.py
#     (attention over the full cache, runtime-position mask and KV slot).
# Nothing here rebuilds kernels; it snapshots already-audited objects.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
MODEL="$REPO/examples/FPGA-BOSCAME/qwen3-0.6b/model"
TRITON_BUILD="${TRITON_BUILD:-$REPO/examples/FPGA-BOSCAME/qwen3-0.6b/triton/build}"
LLVM_BIN="${LLVM_BIN:-$REPO/llvm/build-2d26/bin}"
OUT="${OUT:-$MODEL/build/model-lib}"
ADAPTERS="${ADAPTERS:-$MODEL/build/triton-call/full-28l/qwen_triton_adapters.c}"

# The adapter file is the authority: never silently fall back to a different
# shape set or a historical full-28l report when OUT/ADAPTERS are overridden.
if [[ ! -f "$ADAPTERS" ]]; then
  echo "missing generated adapters: $ADAPTERS" >&2
  exit 1
fi
CASES=()
while read -r case; do
  [[ -n "$case" ]] && CASES+=(--case "$case")
done < <(grep -o '_mlir_ciface_kernel_[a-z0-9_]*' "$ADAPTERS" \
           | sed 's/_mlir_ciface_kernel_//' | sort -u)
if [[ ${#CASES[@]} == 0 ]]; then
  echo "no Triton kernel references in $ADAPTERS" >&2
  exit 1
fi
echo "discovered $((${#CASES[@]} / 2)) kernels from $(basename "$ADAPTERS")"
for ((i = 0; i < ${#CASES[@]}; i += 2)); do
  case="${CASES[$((i + 1))]}"
  if [[ ! -f "$TRITON_BUILD/$case/nr/kernel.o" ]]; then
    echo "missing NR build for $case; build it first" >&2
    exit 1
  fi
done

rm -rf "$OUT"
python3 -B "$MODEL/tools/archive_kernels.py" \
  --triton-build "$TRITON_BUILD" "${CASES[@]}" \
  --output "$OUT" --ar "$LLVM_BIN/llvm-ar" --nm "$LLVM_BIN/llvm-nm"

# Compile the generated ABI adapters for RISC-V and link them with the archive
# plus a probe that calls every replaced symbol.
"$LLVM_BIN/clang" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64d \
  -mcmodel=medany -O2 -ffreestanding -fno-builtin -Wall -Wextra -Werror \
  -I"$REPO/examples/FPGA-BOSCAME/qwen3-0.6b" \
  -I"$REPO/examples/FPGA-BOSCAME/common/nr" \
  -I"$REPO/examples/FPGA-BOSCAME/common/uart" \
  -c "$ADAPTERS" -o "$OUT/qwen_triton_adapters.o"

python3 - "$OUT" "$ADAPTERS" <<'PY'
import sys
from pathlib import Path
out = Path(sys.argv[1])
import json
import re
adapters = Path(sys.argv[2])
replacement = json.loads((adapters.parent / "triton-call-replacement.json").read_text())
symbols = sorted(re.findall(r"^void (_mlir_ciface_[A-Za-z_0-9]+)\(",
                            adapters.read_text(), re.MULTILINE))
expected = sorted("_mlir_ciface_" + symbol for symbol in replacement["distinct_symbols"])
if symbols != expected:
    raise SystemExit("adapter definitions disagree with their replacement report")
lines = [
    "/* Link probe for the graph's external calls. Never flashed, never run. */",
    "/* Taking each replaced symbol's address forces the archive to resolve it,  */",
    "/* so a missing or renamed kernel fails the link, not the numerics.       */",
]
lines += [f"extern void {symbol}(void);" for symbol in symbols]
lines.append("")
lines.append("typedef void (*qwen_external_fn)(void);")
lines.append("qwen_external_fn const qwen_external_symbols[] = {")
lines += [f"  (qwen_external_fn){symbol}," for symbol in symbols]
lines.append("};")
lines.append("unsigned const qwen_external_symbol_count =")
lines.append("    sizeof(qwen_external_symbols) / sizeof(qwen_external_symbols[0]);")
(out / "external-link-probe.c").write_text("\n".join(lines) + "\n")
print(f"probe references {len(symbols)} replaced symbols")
PY

"$LLVM_BIN/clang" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64d \
  -mcmodel=medany -O2 -ffreestanding -fno-builtin -Wall -Wextra -Werror \
  -I"$REPO/examples/FPGA-BOSCAME/qwen3-0.6b" \
  -I"$REPO/examples/FPGA-BOSCAME/common/nr" \
  -I"$REPO/examples/FPGA-BOSCAME/common/uart" \
  -c "$OUT/external-link-probe.c" -o "$OUT/external-link-probe.o"

"${LD_LLD:-ld.lld}" -m elf64lriscv -r -Map="$OUT/external-link.map" \
  -o "$OUT/external-linked.o" \
  "$OUT/qwen_triton_adapters.o" "$OUT/external-link-probe.o" \
  "$OUT/libqwen_triton.a"

echo "--- adapters + probe + archive linked: $OUT/external-linked.o"
echo "--- undefined symbols after link (must be runtime/libm only):"
"$LLVM_BIN/llvm-nm" --undefined-only "$OUT/external-linked.o"
python3 - "$OUT" "$LLVM_BIN/llvm-nm" <<'PY'
import json
import subprocess
import sys
from pathlib import Path
out = Path(sys.argv[1])
lines = subprocess.check_output([sys.argv[2], '--undefined-only',
                                 str(out / 'external-linked.o')], text=True).splitlines()
undefined = sorted(line.split()[-1] for line in lines if line.strip())
unresolved_kernels = [name for name in undefined
                      if name.startswith(('_mlir_ciface_', 'triton_', 'qwen_graph_'))]
report = {'undefined_symbols': undefined, 'unresolved_kernel_symbols': unresolved_kernels,
          'status': 'FAIL' if unresolved_kernels else 'PASS',
          'scope': 'relocatable link; runtime/libm resolution is checked by final ELF audit'}
(out / 'external-link.json').write_text(json.dumps(report, indent=2) + '\n')
if unresolved_kernels:
    raise SystemExit('unresolved kernel symbols: ' + ', '.join(unresolved_kernels))
PY
echo "--- resolved Triton symbols:"
"$LLVM_BIN/llvm-nm" --defined-only "$OUT/external-linked.o" \
  | grep -c "_mlir_ciface_kernel_"
