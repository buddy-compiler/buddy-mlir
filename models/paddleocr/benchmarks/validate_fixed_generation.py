#!/usr/bin/env python3
"""Check a fixed-shape recompute strategy against cached OCR generation."""

import argparse

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="OCR:")
    parser.add_argument("--new-tokens", type=int, default=4)
    parser.add_argument("--precision", choices=("f32", "f16"), default="f16")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    dtype = torch.float16 if args.precision == "f16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
        dtype=dtype,
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
    base = inputs["input_ids"][0].tolist()
    grid = inputs["image_grid_thw"]
    fixed_len = len(base) + args.new_tokens

    with torch.inference_mode():
        cached = model.generate(
            **inputs,
            max_new_tokens=args.new_tokens,
            use_cache=True,
            do_sample=False,
        )[0, len(base) :].tolist()
        generated = []
        for _ in range(args.new_tokens):
            active = base + generated
            active_ids = torch.tensor([active])
            active_pos, _ = model.get_rope_index(
                active_ids,
                image_grid_thw=grid,
                attention_mask=torch.ones_like(active_ids),
            )
            ids = torch.zeros((1, fixed_len), dtype=torch.long)
            mask = torch.zeros_like(ids)
            pos = torch.zeros((3, 1, fixed_len), dtype=torch.long)
            ids[0, : len(active) - 1] = torch.tensor(active[:-1])
            ids[0, -1] = active[-1]
            mask[0, : len(active) - 1] = 1
            mask[0, -1] = 1
            pos[:, :, : len(active) - 1] = active_pos[:, :, :-1]
            pos[:, :, -1] = active_pos[:, :, -1]
            output = model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=pos,
                pixel_values=inputs["pixel_values"],
                image_grid_thw=grid,
                use_cache=False,
            )
            generated.append(int(output.logits[0, -1].argmax()))

    print("cached:", cached)
    print("fixed: ", generated)
    if cached != generated:
        raise SystemExit("fixed-shape generation differs from cached reference")


if __name__ == "__main__":
    main()
