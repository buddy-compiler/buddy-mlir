#!/usr/bin/env python3
# ===- run-subgraphs-python.py -----------------------------------------------
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===

import argparse
import ctypes
import os
import subprocess
import time
from pathlib import Path

import numpy as np

KV_COUNT = 56
MAX_TOKEN_LENGTH = 1024
EOS_TOKEN_ID = 151643
PARAM_SIZE = 1777088064

rt = None


def _init_mlir_bindings():
    global rt
    from mlir import runtime as _rt

    rt = _rt


def _build_forward_output_descriptor():
    class ForwardOutputDescriptor(ctypes.Structure):
        _fields_ = [
            (str(i), rt.make_nd_memref_descriptor(4, rt.as_ctype(np.float32)))
            for i in range(KV_COUNT)
        ] + [
            (
                str(KV_COUNT),
                rt.make_nd_memref_descriptor(3, rt.as_ctype(np.float32)),
            )
        ]

    return ForwardOutputDescriptor


def _memref_desc(array: np.ndarray):
    return rt.get_ranked_memref_descriptor(array)


def _memref_to_numpy(desc):
    return rt.ranked_memref_to_numpy(ctypes.pointer(desc))


def _sample_next_token(
    logits: np.ndarray,
    temperature: float,
    top_p: float,
    rng: np.random.Generator,
) -> int:
    if temperature <= 0:
        return int(np.argmax(logits))

    scaled = logits.astype(np.float64) / temperature
    scaled -= np.max(scaled)
    probs = np.exp(scaled)
    probs_sum = probs.sum()
    if probs_sum <= 0:
        return int(np.argmax(logits))
    probs /= probs_sum

    if 0 < top_p < 1.0:
        sorted_idx = np.argsort(-probs)
        sorted_probs = probs[sorted_idx]
        cum = np.cumsum(sorted_probs)
        keep_count = int(np.searchsorted(cum, top_p, side="left")) + 1
        keep_count = max(1, keep_count)
        keep_idx = sorted_idx[:keep_count]
        keep_probs = probs[keep_idx]
        keep_probs /= keep_probs.sum()
        return int(rng.choice(keep_idx, p=keep_probs))

    return int(rng.choice(len(probs), p=probs))


def _resolve_path(path_arg: Path, repo_root: Path) -> Path:
    path_arg = path_arg.expanduser()
    if path_arg.is_absolute():
        return path_arg.resolve()
    cwd_path = (Path.cwd() / path_arg).resolve()
    if cwd_path.exists():
        return cwd_path
    return (repo_root / path_arg).resolve()


def _apply_omp_env(args):
    if args.omp_num_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(args.omp_num_threads)
    if args.omp_proc_bind is not None:
        os.environ["OMP_PROC_BIND"] = args.omp_proc_bind
    if args.omp_places is not None:
        os.environ["OMP_PLACES"] = args.omp_places
    if args.kmp_affinity is not None:
        os.environ["KMP_AFFINITY"] = args.kmp_affinity


def _ensure_aot_runtime_lib(artifact_dir: Path) -> Path:
    so_path = artifact_dir / "libdeepseek_forward_runtime.so"
    if so_path.exists():
        return so_path

    objs = [
        artifact_dir / "forward_prefill.o",
        artifact_dir / "forward_decode.o",
        artifact_dir / "subgraph_prefill.o",
        artifact_dir / "subgraph_decode.o",
    ]
    missing = [str(p) for p in objs if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing object files required for AOT runtime:\n  "
            + "\n  ".join(missing)
        )

    cmd = ["clang++", "-shared", *[str(p) for p in objs], "-o", str(so_path)]
    subprocess.check_call(cmd)
    return so_path


def _run_aot(
    artifact_dir: Path,
    llvm_build_dir: Path,
    tokenizer,
    params: np.ndarray,
    prefill_input: np.ndarray,
    token_cnt: int,
    eos_id: int,
    args,
    rng: np.random.Generator,
):
    runtime_so = _ensure_aot_runtime_lib(artifact_dir)

    libdir = llvm_build_dir / "lib"
    rtld_global = getattr(os, "RTLD_GLOBAL", 0)
    for so in [
        libdir / "libomp.so",
        libdir / "libmlir_c_runner_utils.so",
        libdir / "libmlir_runner_utils.so",
    ]:
        ctypes.CDLL(str(so), mode=rtld_global)
    runtime_lib = ctypes.CDLL(str(runtime_so), mode=rtld_global)

    prefill_func = runtime_lib._mlir_ciface_forward_prefill
    decode_func = runtime_lib._mlir_ciface_forward_decode

    ForwardOutputDescriptor = _build_forward_output_descriptor()

    params_desc = _memref_desc(params)
    prefill_desc = _memref_desc(prefill_input)
    decode_input = np.zeros((1, 1), dtype=np.int64)
    decode_input_desc = _memref_desc(decode_input)
    cache_pos = np.zeros((1,), dtype=np.int64)
    cache_pos_desc = _memref_desc(cache_pos)

    state_in = ForwardOutputDescriptor()
    state_out = ForwardOutputDescriptor()

    print("[Python] prefill...")
    t0 = time.time()
    prefill_func(
        ctypes.byref(state_in),
        ctypes.byref(params_desc),
        ctypes.byref(prefill_desc),
    )
    prefill_s = time.time() - t0

    prefill_logits = _memref_to_numpy(getattr(state_in, str(KV_COUNT)))
    next_token = _sample_next_token(
        prefill_logits[0, token_cnt - 1, :], args.temperature, args.top_p, rng
    )

    generated = [next_token]
    print(
        tokenizer.decode(
            [next_token],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ),
        end="",
        flush=True,
    )

    decode_time = 0.0
    for _ in range(args.max_new_tokens - 1):
        if token_cnt >= MAX_TOKEN_LENGTH or next_token == eos_id:
            break

        decode_input[0, 0] = next_token
        cache_pos[0] = token_cnt

        decode_args = [
            ctypes.byref(state_out),
            ctypes.byref(params_desc),
            ctypes.byref(decode_input_desc),
            ctypes.byref(cache_pos_desc),
        ]
        for i in range(KV_COUNT):
            decode_args.append(ctypes.byref(getattr(state_in, str(i))))

        t0 = time.time()
        decode_func(*decode_args)
        decode_time += time.time() - t0

        logits = _memref_to_numpy(getattr(state_out, str(KV_COUNT)))
        next_token = _sample_next_token(
            logits[0, 0, :], args.temperature, args.top_p, rng
        )
        generated.append(next_token)
        token_cnt += 1

        if next_token == eos_id:
            break
        print(
            tokenizer.decode(
                [next_token],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ),
            end="",
            flush=True,
        )

        state_in, state_out = state_out, state_in

    print("\n")
    print(f"[Python] prefill time: {prefill_s:.3f}s")
    if len(generated) > 1:
        print(
            f"[Python] decode speed: {(len(generated)-1)/max(decode_time,1e-9):.2f} tok/s"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Python end-to-end inference by executing exported Buddy subgraphs"
    )
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("build/examples/BuddyDeepSeekR1"),
        help="Directory containing forward_prefill.mlir/forward_decode.mlir/arg0.data",
    )
    parser.add_argument(
        "--llvm-build-dir",
        type=Path,
        default=Path("llvm/build"),
        help="LLVM build dir that provides MLIR runtime shared libs",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.environ.get(
            "DEEPSEEKR1_MODEL_PATH", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        ),
        help="HF model id or local model path (for tokenizer)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--eos-id", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--system-prompt", type=str, default="You are a helpful assistant."
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable tokenizer chat template and use raw --prompt encoding.",
    )
    parser.add_argument("--omp-num-threads", type=int, default=None)
    parser.add_argument("--omp-proc-bind", type=str, default=None)
    parser.add_argument("--omp-places", type=str, default=None)
    parser.add_argument("--kmp-affinity", type=str, default=None)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    _init_mlir_bindings()
    _apply_omp_env(args)

    repo_root = Path(__file__).resolve().parents[2]
    artifact_dir = _resolve_path(args.artifact_dir, repo_root)
    llvm_build_dir = _resolve_path(args.llvm_build_dir, repo_root)

    artifact_dir.mkdir(parents=True, exist_ok=True)

    param_file = artifact_dir / "arg0.data"

    missing = [
        str(p)
        for p in [
            artifact_dir / "forward_prefill.mlir",
            artifact_dir / "forward_decode.mlir",
            param_file,
        ]
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing artifacts:\n  "
            + "\n  ".join(missing)
            + "\nPlease generate them under --artifact-dir before running."
        )

    params = np.fromfile(param_file, dtype=np.float32)
    if params.size != PARAM_SIZE:
        raise ValueError(
            f"Unexpected param size: {params.size}, expect {PARAM_SIZE}"
        )
    params = params.reshape((PARAM_SIZE,))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if not args.no_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": args.prompt})
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    else:
        input_ids = tokenizer.encode(args.prompt, add_special_tokens=True)

    if not input_ids:
        raise ValueError("Prompt tokenization produced empty input.")
    input_ids = input_ids[:MAX_TOKEN_LENGTH]
    token_cnt = len(input_ids)

    eos_id = (
        args.eos_id
        if args.eos_id is not None
        else (
            int(tokenizer.eos_token_id)
            if tokenizer.eos_token_id is not None
            else EOS_TOKEN_ID
        )
    )
    rng = np.random.default_rng(args.seed)

    prefill_input = np.zeros((1, MAX_TOKEN_LENGTH), dtype=np.int64)
    prefill_input[0, :token_cnt] = np.asarray(input_ids, dtype=np.int64)

    print("[Python] runtime: aot")
    _run_aot(
        artifact_dir,
        llvm_build_dir,
        tokenizer,
        params,
        prefill_input,
        token_cnt,
        eos_id,
        args,
        rng,
    )


if __name__ == "__main__":
    main()
