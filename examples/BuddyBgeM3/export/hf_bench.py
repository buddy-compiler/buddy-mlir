#!/usr/bin/env python3
# ===- hf_bench.py - HF reference BGE-M3 latency benchmark ----------------===//
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
# Mirrors the buddy BGE-M3 dense path: AutoModel -> last_hidden_state[:, 0]
# (CLS) -> L2 norm. Tokenizer settings MUST match the buddy side:
#   padding="max_length", truncation=True, max_length=<spec max_seq_len>.
#
# Usage:
#   python3 hf_bench.py --model-dir "$LOCAL_BGE_M3" \
#       --max-length 512 --repeat 10 --threads 4
import argparse
import statistics
import time

import torch
from transformers import AutoModel, AutoTokenizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument(
        "--threads",
        type=int,
        default=None,
        help="torch.set_num_threads (align with the buddy side)",
    )
    ap.add_argument(
        "--text", default="The quick brown fox jumps over the lazy dog."
    )
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModel.from_pretrained(args.model_dir)
    model.eval()

    enc = tok(
        args.text,
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )

    # Warm-up (excluded from stats).
    with torch.no_grad():
        model(**enc)

    lat_ms = []
    with torch.no_grad():
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            out = model(**enc)
            cls_vec = out.last_hidden_state[:, 0, :]
            cls_vec = cls_vec / cls_vec.norm(dim=-1, keepdim=True)
            lat_ms.append((time.perf_counter() - t0) * 1000.0)

    steady = lat_ms[1:] if len(lat_ms) > 1 else lat_ms
    med = statistics.median(steady)
    print(
        f"[hf-bench] threads={torch.get_num_threads()} "
        f"max_length={args.max_length} repeat={args.repeat}"
    )
    print(f"[hf-bench] latencies(ms) = {[round(x, 2) for x in lat_ms]}")
    print(f"[hf-bench] steady-state median = {med:.2f} ms (n={len(steady)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

