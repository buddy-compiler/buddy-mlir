#!/usr/bin/env python3
"""Patch subgraph0_decode_e2b.mlir to replace constant-folded cumlen with dynamic input.

The torch._dynamo tracing constant-folds cumulative_length into the attention mask,
making the mask only work for the traced cumlen value (1). This script replaces
the hardcoded cumlen computation with a reference to the actual cumlen input arg.

The pattern to fix:
  %N = "tosa.const"() <{values = dense<0> : tensor<1xi64>}>   ← was arange(0) or cumlen
  %M = "tosa.const"() <{values = dense<1> : tensor<1xi64>}>   ← was kv_length or cumlen
  %R = tosa.add %N, %M                                         ← was cache_pos or new_cumlen

We replace the constant representing the old cumlen value with the first cumlen input arg
(%arg16 in the subgraph).
"""
import re
import sys

mlir_path = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/home/zhuxinye/buddy-mlir/build/examples/BuddyGemma4/subgraph0_decode_e2b.mlir"
)

with open(mlir_path, "r") as f:
    content = f.read()

lines = content.split("\n")

# Find the three "const 0 + const 1" patterns near the top of the function (lines 70-100)
# These compute: cache_position, new_cumlen, and sliding_window_cumlen
# All should use %arg16 (the first cumlen/cache_position input) instead of const 0.

# Pattern: three consecutive lines like:
#   %X = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
#   %Y = "tosa.const"() <{values = dense<1> : tensor<1xi64>}> : () -> tensor<1xi64>
#   %Z = tosa.add %X, %Y : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
# We replace %Z = tosa.add %X, %Y with %Z = tosa.add %arg16, %Y
# (replacing const_0 with arg16, keeping const_1 for kv_length)

patch_count = 0
i = 0
while i < len(lines):
    # Look for the pattern: const 0, const 1, add
    if i + 2 < len(lines):
        l0 = lines[i].strip()
        l1 = lines[i + 1].strip()
        l2 = lines[i + 2].strip()

        m0 = re.match(
            r'(%\w+) = "tosa\.const"\(\) <\{values = dense<0> : tensor<1xi64>\}>',
            l0,
        )
        m1 = re.match(
            r'(%\w+) = "tosa\.const"\(\) <\{values = dense<1> : tensor<1xi64>\}>',
            l1,
        )
        if m0 and m1:
            var0 = m0.group(1)
            var1 = m1.group(1)
            m2 = re.match(
                rf"(%\w+) = tosa\.add {re.escape(var0)}, {re.escape(var1)}", l2
            )
            if m2:
                result_var = m2.group(1)
                # Replace: use %arg16 instead of the const 0
                # %Z = tosa.add %arg16, %Y (where %Y = const 1 = kv_length)
                old_line = lines[i + 2]
                new_line = old_line.replace(
                    f"tosa.add {var0}, {var1}", f"tosa.add {var0}, %arg16"
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
