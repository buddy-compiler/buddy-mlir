#!/usr/bin/env python3
# ===- bge_m3_tokenize.py - BGE-M3 tokenizer helper -----------------------===//
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

import argparse
import os
import sys

os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from transformers import AutoTokenizer
from transformers.utils import logging

logging.set_verbosity_error()


def main() -> int:
    parser = argparse.ArgumentParser(description="Tokenize text for BGE-M3")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--max-len", type=int, required=True)
    args = parser.parse_args()

    with open(args.input_file, encoding="utf-8") as f:
        text = f.read()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=args.max_len,
        return_tensors=None,
    )

    print(" ".join(str(x) for x in encoded["input_ids"]))
    print(" ".join(str(x) for x in encoded["attention_mask"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
