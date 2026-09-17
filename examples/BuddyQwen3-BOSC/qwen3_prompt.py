#!/usr/bin/env python3
"""Prepare Qwen3 chat-template token IDs for Buddy layer-wise imports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def _qwen3_0_6b_dir() -> Path:
    env = os.environ.get("QWEN3_0_6B_DIR")
    if env:
        return Path(env)
    return HERE.parents[1].parent / "qwen3-0.6b"


QWEN3_0_6B_DIR = _qwen3_0_6b_dir()
DEFAULT_TOKENIZER_DIR = (
    Path(os.environ["QWEN3_0_6B_TOKENIZER_DIR"])
    if "QWEN3_0_6B_TOKENIZER_DIR" in os.environ
    else QWEN3_0_6B_DIR / "tokenizer" / "data"
)
QWEN3_BOS_ID = 151643


def parse_token_ids(text: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("token IDs must be a comma-separated integer list") from exc
    if not values:
        raise ValueError("token ID list must not be empty")
    return values


def left_pad_token_ids(
    token_ids: list[int], capacity: int, pad_token_id: int = QWEN3_BOS_ID
) -> list[int]:
    if capacity <= 0 or not token_ids or len(token_ids) > capacity:
        raise ValueError("token IDs must fit a positive padded capacity")
    return [pad_token_id] * (capacity - len(token_ids)) + token_ids


def build_left_padded_inputs(
    token_ids: list[int], prefill_capacity: int, max_cache_len: int
) -> dict[str, np.ndarray]:
    padded_ids = left_pad_token_ids(token_ids, prefill_capacity)
    if max_cache_len <= prefill_capacity:
        raise ValueError("max cache length must exceed padded prefill capacity")
    masked_prefix = prefill_capacity - len(token_ids)
    minimum = np.finfo(np.float32).min
    prefill_mask = np.triu(
        np.full(
            (prefill_capacity, prefill_capacity), minimum, dtype=np.float32
        ),
        k=1,
    )
    prefill_mask[:, :masked_prefix] = minimum
    decode_mask = np.full(max_cache_len, minimum, dtype=np.float32)
    decode_mask[masked_prefix : prefill_capacity + 1] = np.float32(0.0)
    write_mask = np.zeros(max_cache_len, dtype=np.float32)
    write_mask[prefill_capacity] = np.float32(1.0)
    return {
        "input_ids": np.asarray(padded_ids, dtype=np.int64).reshape(
            1, prefill_capacity
        ),
        "prompt_length": np.asarray([len(token_ids)], dtype=np.int32),
        "prefill_mask": prefill_mask.reshape(
            1, 1, prefill_capacity, prefill_capacity
        ),
        "decode_mask": decode_mask.reshape(1, 1, 1, max_cache_len),
        "cache_write_mask": write_mask.reshape(1, 1, max_cache_len, 1),
    }


def tokenize_chat_prompt(
    prompt: str,
    tokenizer_dir: Path = DEFAULT_TOKENIZER_DIR,
    system_prompt: str = "Qwen3",
    enable_thinking: bool = False,
) -> list[int]:
    """Match the Qwen3 C runner's BOS + chat-template prompt layout."""
    if not prompt:
        raise ValueError("prompt must not be empty")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, local_files_only=True
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        return_dict=True,
    )
    return [QWEN3_BOS_ID, *[int(token) for token in encoded["input_ids"]]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--system-prompt", default="Qwen3")
    parser.add_argument("--enable-thinking", action="store_true")
    args = parser.parse_args()
    token_ids = tokenize_chat_prompt(
        args.prompt,
        args.tokenizer_dir,
        args.system_prompt,
        args.enable_thinking,
    )
    print(json.dumps({
        "prompt": args.prompt,
        "system_prompt": args.system_prompt,
        "enable_thinking": args.enable_thinking,
        "prefill_len": len(token_ids),
        "token_ids": token_ids,
        "token_ids_csv": ",".join(str(token) for token in token_ids),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
