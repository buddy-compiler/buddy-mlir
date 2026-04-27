#!/usr/bin/env python3
"""Patch subgraph0_decode_e2b.mlir for constant-folded cumlen.

torch._dynamo tracing constant-folds cumulative_length into the attention
mask, so the mask only works for the traced cumlen value (1). This script
replaces the baked cumlen computation with the actual cumlen input arg.

Historical pattern (older lowering):
  %N = "tosa.const"() <{values = dense<0> : tensor<1xi64>}>
  %M = "tosa.const"() <{values = dense<1> : tensor<1xi64>}>
  %R = tosa.add %N, %M
→ replace with: tosa.add %arg16, %M

Current pattern (tensor.generate for arange(0), then +1):
  %g = tensor.generate { ... } : tensor<1xi64>
  %c = "tosa.const"() <{values = dense<1> : tensor<1xi64>}>
  %R = tosa.add %g, %c
→ replace with: tosa.add %arg16, %c

Only the first attention block still bakes constants; later layers already use
%arg19, %arg40, ... We patch only adds whose second operand is the dense<1>
const on the previous line (not an existing %argN).
"""

import re
import sys

CUMLEN_ARG = "%arg16"

mlir_path = (
    sys.argv[1]
    if len(sys.argv) > 1
    else (
        "/home/zhuxinye/buddy-mlir/build/examples/BuddyGemma4/"
        "subgraph0_decode_e2b.mlir"
    )
)

with open(mlir_path) as f:
    content = f.read()

lines = content.split("\n")

# Regex: tosa.add %generated..., %op : (tensor<1xi64>, tensor<1xi64>) [-> ...]
RE_ADD_GEN = re.compile(
    r"(%\w+) = tosa\.add (%generated(?:_\d+)?), (%\w+) : "
    r"\(tensor<1xi64>, tensor<1xi64>\)(?: -> tensor<1xi64>)?"
)
# Previous line: const 1 for cumlen offset (kv_length = 1 at trace)
RE_CONST_ONE = re.compile(
    r'(%\w+) = "tosa\.const"\(\) <\{values = dense<1> : tensor<1xi64>\}>'
)

patch_count = 0
i = 0
while i < len(lines):
    # --- New IR: tensor.generate + const1 + tosa.add ---
    m_add = RE_ADD_GEN.match(lines[i].strip())
    if m_add and i > 0:
        _result, gen, op2 = m_add.groups()
        m_const = RE_CONST_ONE.match(lines[i - 1].strip())
        if m_const and m_const.group(1) == op2:
            old_line = lines[i]
            new_line = old_line.replace(
                f"tosa.add {gen}, {op2}",
                f"tosa.add {CUMLEN_ARG}, {op2}",
            )
            lines[i] = new_line
            patch_count += 1
            print(f"  Line {i + 1}: {old_line.strip()} -> {new_line.strip()}")
            i += 1
            continue

    # --- Legacy: const0 + const1 + add (dense<0> / dense<1> on 1xi64) ---
    if i + 2 < len(lines):
        l0 = lines[i].strip()
        l1 = lines[i + 1].strip()
        l2 = lines[i + 2].strip()

        m0 = re.match(
            r'(%\w+) = "tosa\.const"\(\) <\{values = dense<0> : '
            r"tensor<1xi64>\}>",
            l0,
        )
        m1 = re.match(
            r'(%\w+) = "tosa\.const"\(\) <\{values = dense<1> : '
            r"tensor<1xi64>\}>",
            l1,
        )
        if m0 and m1:
            var0 = m0.group(1)
            var1 = m1.group(1)
            m2 = re.match(
                rf"(%\w+) = tosa\.add {re.escape(var0)}, {re.escape(var1)}",
                l2,
            )
            if m2:
                old_line = lines[i + 2]
                new_line = old_line.replace(
                    f"tosa.add {var0}, {var1}",
                    f"tosa.add {CUMLEN_ARG}, {var1}",
                )
                lines[i + 2] = new_line
                patch_count += 1
                print(
                    f"  Line {i + 3}: {old_line.strip()} -> {new_line.strip()}"
                )
                i += 3
                continue
    i += 1

if patch_count == 0:
    print("WARNING: No patches applied!")
    sys.exit(1)

print(f"\nApplied {patch_count} patches")

with open(mlir_path, "w") as f:
    f.write("\n".join(lines))

print(f"Saved to {mlir_path}")
