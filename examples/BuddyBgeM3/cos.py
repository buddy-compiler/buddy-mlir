#!/usr/bin/env python3
# ===- cos.py - Cosine similarity gate (pure stdlib, zero deps) -----------===//
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
# Usage:
#   python3 cos.py <a.txt|json> <b.txt|json>
#
# Inputs can be:
#   - raw buddy-cli output (extract from first '[' to last ']')
#   - a JSON array file
#
# Criterion: before/after embeddings must have cosine > 0.999.
import json
import math
import sys


def load(path):
    s = open(path, "r").read()
    i, j = s.find("["), s.rfind("]")
    if i < 0 or j < 0:
        raise SystemExit(f"{path}: vector array not found")
    return json.loads(s[i : j + 1])


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    a, b = load(sys.argv[1]), load(sys.argv[2])
    if len(a) != len(b):
        raise SystemExit(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    cos = dot / (na * nb) if na and nb else float("nan")
    print(f"cos = {cos:.9f}   (dim={len(a)})")
    print("PASS" if cos > 0.999 else "FAIL")
    return 0 if cos > 0.999 else 1


if __name__ == "__main__":
    sys.exit(main())

