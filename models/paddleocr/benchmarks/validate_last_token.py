#!/usr/bin/env python3
"""Check that final-position LM-head projection preserves reference logits."""

import argparse

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="OCR:")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    image = Image.open(args.image).convert("RGB")
    inputs = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": args.prompt},
                ],
            }
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    with torch.inference_mode():
        full = model(**inputs, use_cache=False).logits[:, -1:, :].clone()
        original = model.lm_head.forward
        model.lm_head.forward = lambda x: original(x[:, -1:, :])
        final_only = model(**inputs, use_cache=False).logits

    difference = (full - final_only).abs()
    print(f"input_tokens={inputs['input_ids'].shape[1]}")
    print(f"vision_patches={inputs['pixel_values'].shape[0]}")
    print(f"logits_shape={list(final_only.shape)}")
    print(f"max_abs_error={difference.max().item()}")
    print(f"same_top1={full.argmax(-1).item() == final_only.argmax(-1).item()}")
    if not torch.allclose(full, final_only, atol=1e-4, rtol=1e-5):
        raise SystemExit("last-position logits exceed tolerance")


if __name__ == "__main__":
    main()
