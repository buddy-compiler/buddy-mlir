#!/usr/bin/env python3
# ===- gen_specs.py - Generate BGE-M3 128/256/512 spec variants -----------===//
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
# Creates models/bge_m3/specs/seq{128,256,512}.json from base.json by
# changing max_seq_len (fixed-shape compile) and model_id.
#
# Usage (from the repo root):
#   python3 examples/BuddyBgeM3/export/gen_specs.py
import json
from pathlib import Path

# export/gen_specs.py -> examples/BuddyBgeM3/export -> repo root.
REPO = Path(__file__).resolve().parents[3]
BASE = REPO / "models" / "bge_m3" / "specs" / "base.json"


def main() -> int:
    base = json.loads(BASE.read_text())
    for seq_len in (128, 256, 512):
        spec = dict(base)
        spec["max_seq_len"] = seq_len
        spec["model_id"] = f"bge_m3_seq{seq_len}"
        out = BASE.with_name(f"seq{seq_len}.json")
        out.write_text(json.dumps(spec, indent=2) + "\n")
        print(f"[gen] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

