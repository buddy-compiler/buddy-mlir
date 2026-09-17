#!/usr/bin/env python3
"""Link two production INT8 Triton linear builds into an AME A/B benchmark.

Uses the same snapshot, provenance, symbol renaming, and common NR runtime
linker as dequantization. --full-input-scan checks every B byte (outside timing).
The default checks all A and samples every B row, with full boundary rows.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dequant_ab import main

if __name__ == "__main__":
    main(benchmark="ame")
