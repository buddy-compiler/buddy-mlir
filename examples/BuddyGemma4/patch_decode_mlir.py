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

import argparse
import os
import re
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Patch Gemma4 decode MLIR to use runtime cumulative length."
    )
    parser.add_argument("mlir_path", type=str, help="Path to decode MLIR file.")
    parser.add_argument(
        "--cumlen-arg",
        type=str,
        default="%arg16",
        help="SSA value to use for runtime cumulative length.",
    )
    parser.add_argument(
        "--strict-no-change",
        action="store_true",
        help="Return non-zero when no modifications were applied.",
    )
    return parser.parse_args()


args = parse_args()
CUMLEN_ARG = args.cumlen_arg
mlir_path = args.mlir_path

if not os.path.exists(mlir_path):
    print(f"ERROR: file does not exist: {mlir_path}")
    sys.exit(2)

with open(mlir_path) as f:
    content = f.read()

lines = content.split("\n")

# --- Pass 1: collect all SSA values that are tensor.generate results ---
# Handles both:
#   %generated = tensor.generate  { ... }         (unquoted, named)
#   %N = "tensor.generate"() ({ ... })            (quoted, numeric SSA)
RE_GEN_UNQUOTED = re.compile(r"(%\w+) = tensor\.generate\b")
RE_GEN_QUOTED = re.compile(r'(%\w+) = "tensor\.generate"\(\)')
generate_vars: set[str] = set()
for line in lines:
    s = line.strip()
    for pat in (RE_GEN_UNQUOTED, RE_GEN_QUOTED):
        m = pat.match(s)
        if m:
            generate_vars.add(m.group(1))

# Previous line: const 1 for cumlen offset (kv_length = 1 at trace)
RE_CONST_ONE = re.compile(
    r'(%\w+) = (?:"tosa\.const"\(\)|tosa\.const) '
    r"<\{values = dense<1> : tensor<1xi64>\}>"
)

# tosa.add with unquoted syntax:  %R = tosa.add %A, %B : (...)
RE_ADD_UNQUOTED = re.compile(
    r"(%\w+) = tosa\.add (%\w+), (%\w+) : "
    r"\(tensor<1xi64>, tensor<1xi64>\)(?: -> tensor<1xi64>)?"
)
# tosa.add with quoted syntax:  %R = "tosa.add"(%A, %B) : (...)
RE_ADD_QUOTED = re.compile(
    r'(%\w+) = "tosa\.add"\((%\w+), (%\w+)\) : '
    r"\(tensor<1xi64>, tensor<1xi64>\)(?: -> tensor<1xi64>)?"
)

patch_count = 0
i = 0
while i < len(lines):
    s = line_strip = lines[i].strip()

    # --- Pattern A: tensor.generate + const1 + tosa.add (any syntax) ---
    for re_add in (RE_ADD_UNQUOTED, RE_ADD_QUOTED):
        m_add = re_add.match(line_strip)
        if m_add and i > 0:
            _result, lhs, rhs = m_add.groups()
            m_const = RE_CONST_ONE.match(lines[i - 1].strip())
            if m_const and m_const.group(1) == rhs and lhs in generate_vars:
                old_line = lines[i]
                # Replace first occurrence of the lhs variable in the add
                new_line = old_line.replace(lhs, CUMLEN_ARG, 1)
                lines[i] = new_line
                patch_count += 1
                print(
                    f"  Line {i + 1}: {old_line.strip()} -> {new_line.strip()}"
                )
                break

    # --- Pattern B: const0 + const1 + unquoted add (legacy f32 style) ---
    if i + 2 < len(lines):
        l0 = lines[i].strip()
        l1 = lines[i + 1].strip()
        l2 = lines[i + 2].strip()

        m0 = re.match(
            r'(%\w+) = (?:"tosa\.const"\(\)|tosa\.const) '
            r"<\{values = dense<0> : "
            r"tensor<1xi64>\}>",
            l0,
        )
        m1 = re.match(
            r'(%\w+) = (?:"tosa\.const"\(\)|tosa\.const) '
            r"<\{values = dense<1> : "
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
    # Already-patched files are expected on incremental rebuilds.
    already_patched = CUMLEN_ARG in content
    if already_patched and not args.strict_no_change:
        print("No patch needed (file appears already patched).")
        sys.exit(0)
    else:
        print("WARNING: No patches applied!")
        if args.strict_no_change:
            sys.exit(1)
        print("Continuing without changes.")
        sys.exit(0)

print(f"\nApplied {patch_count} patches")

with open(mlir_path, "w") as f:
    f.write("\n".join(lines))

print(f"Saved to {mlir_path}")
