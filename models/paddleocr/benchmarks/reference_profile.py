#!/usr/bin/env python3
"""Profile PaddleOCR-VL's reference image-to-text path with fixed OCR inputs.

The timings are for the installed Transformers/PyTorch implementation, not RAX.
Keep the image, prompt, dtype, thread count, and generated token count identical
when comparing another implementation.
"""

import argparse
import hashlib
import json
import resource
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


def timed_module(module, name, times, state):
    original = module.forward

    def wrapped(*args, **kwargs):
        start = time.perf_counter()
        result = original(*args, **kwargs)
        elapsed = time.perf_counter() - start
        if state["record"]:
            stage = name
            if name in ("text", "lm_head"):
                x = kwargs.get("inputs_embeds")
                if x is None and args:
                    x = args[0]
                if x is not None:
                    stage += "_prefill" if x.shape[-2] > 1 else "_decode"
            times[stage].append(elapsed)
            if name == "lm_head":
                logits = result[:, -1, :].float().contiguous().cpu()
                if state["save_logits"]:
                    state["logits"].append(logits.clone())
                state["last_logits_sha256"] = hashlib.sha256(
                    logits.numpy().tobytes()
                ).hexdigest()
                state["last_logits_top5"] = torch.topk(
                    logits[0], 5
                ).indices.tolist()
        return result

    module.forward = wrapped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", default="OCR:")
    parser.add_argument("--dtype", choices=("bf16", "f32"), default="bf16")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument(
        "--logits-output",
        type=Path,
        help="Save last-position logits from each generation step",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
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
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    times = defaultdict(list)
    state = {"record": False, "save_logits": args.logits_output is not None}
    for name, module in (
        ("vision", model.visual),
        ("projector", model.mlp_AR),
        ("text", model.model),
        ("lm_head", model.lm_head),
    ):
        timed_module(module, name, times, state)

    runs = []
    saved_logits = []
    for run in range(args.warmups + args.repetitions):
        state["record"] = run >= args.warmups
        state["logits"] = []
        times.clear()
        start = time.perf_counter()
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        preprocess_s = time.perf_counter() - start
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=args.use_cache,
            )
        generate_s = time.perf_counter() - start
        if not state["record"]:
            continue
        if args.logits_output is not None:
            saved_logits.append(torch.cat(state["logits"], dim=0))
        generated_ids = output[0, inputs["input_ids"].shape[1] :].tolist()
        runs.append(
            {
                "preprocess_s": preprocess_s,
                "generate_s": generate_s,
                "end_to_end_s": preprocess_s + generate_s,
                "images_per_s": 1.0 / (preprocess_s + generate_s),
                "stage_s": {k: sum(v) for k, v in times.items()},
                "stage_calls": {k: len(v) for k, v in times.items()},
                "input_tokens": int(inputs["input_ids"].shape[1]),
                "vision_patches": int(inputs["pixel_values"].shape[0]),
                "generated_ids": generated_ids,
                "generated_text": processor.decode(
                    generated_ids, skip_special_tokens=True
                ),
                "last_logits_sha256": state["last_logits_sha256"],
                "last_logits_top5": state["last_logits_top5"],
                "peak_rss_kib": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss,
            }
        )
    if args.logits_output is not None:
        args.logits_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(saved_logits, args.logits_output)
    print(
        json.dumps(
            {
                "implementation": "transformers_reference",
                "model": str(args.model.resolve()),
                "image": str(args.image.resolve()),
                "prompt": args.prompt,
                "dtype": args.dtype,
                "threads": args.threads,
                "max_new_tokens": args.max_new_tokens,
                "use_cache": args.use_cache,
                "warmups": args.warmups,
                "runs": runs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
