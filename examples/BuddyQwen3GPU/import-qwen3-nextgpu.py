#!/usr/bin/env python3
import argparse
import os
import struct
import numpy as np
import torch
from transformers import AutoModelForCausalLM


def to_f32(x: torch.Tensor) -> np.ndarray:
    return x.detach().to(torch.float32).contiguous().cpu().numpy()


def main():
    ap = argparse.ArgumentParser(
        description="Export Qwen3 weights for BuddyQwen3GPU kernels"
    )
    ap.add_argument(
        "--model-path",
        type=str,
        default=os.environ.get("Qwen3_0_6B_MODEL_PATH", ""),
    )
    ap.add_argument("--output", type=str, default="qwen3_nextgpu_weights.bin")
    args = ap.parse_args()
    # seq_len is fixed to match the compiled MLIR kernels
    args.seq_len = 512

    if not args.model_path:
        raise SystemExit(
            "Please set --model-path or export Qwen3_0_6B_MODEL_PATH"
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    sd = model.state_dict()
    cfg = model.config

    hidden = int(cfg.hidden_size)
    ffn = int(cfg.intermediate_size)
    layers = int(cfg.num_hidden_layers)
    vocab = int(cfg.vocab_size)

    # Kernel target dimensions in BuddyQwen3GPU MLIR.
    # Hidden/FFN stay at native Qwen3-0.6B sizes; attention is GQA with
    # 16 query heads and 8 KV heads in the kernels.
    target_hidden = 1024
    target_ffn = 3072

    if hidden != target_hidden or ffn != target_ffn:
        raise RuntimeError(
            f"Unexpected model dims: hidden={hidden}, ffn={ffn}, expected hidden={target_hidden}, ffn={target_ffn}"
        )

    header = struct.pack(
        "<8s7I",
        b"BQ3GPUW\0",
        2,  # version
        layers,
        vocab,
        target_hidden,
        target_ffn,
        args.seq_len,
        8,  # head_num
    )

    with open(args.output, "wb") as f:
        f.write(header)

        # Global tensors.
        emb = to_f32(sd["model.embed_tokens.weight"])
        final_norm = to_f32(sd["model.norm.weight"])
        lm_head = to_f32(sd["lm_head.weight"]).T  # [hidden, vocab]

        emb.tofile(f)
        final_norm.tofile(f)
        lm_head.tofile(f)

        # Per-layer tensors.
        for i in range(layers):
            prefix = f"model.layers.{i}."

            norm1 = to_f32(sd[prefix + "input_layernorm.weight"])
            norm2 = to_f32(sd[prefix + "post_attention_layernorm.weight"])

            # Qwen3-0.6B grouped-query attention:
            # q_proj: [2048, 1024] (16 query heads), k/v: [1024, 1024] (8 kv heads),
            # o_proj: [1024, 2048].
            # Export directly to kernel matmul layout [in, out].
            q_full = to_f32(
                sd[prefix + "self_attn.q_proj.weight"]
            )  # [2048, 1024]
            k_w = to_f32(sd[prefix + "self_attn.k_proj.weight"])
            v_w = to_f32(sd[prefix + "self_attn.v_proj.weight"])
            o_full = to_f32(
                sd[prefix + "self_attn.o_proj.weight"]
            )  # [1024, 2048]
            q_w = q_full
            o_w = o_full

            gate = to_f32(sd[prefix + "mlp.gate_proj.weight"])
            up = to_f32(sd[prefix + "mlp.up_proj.weight"])
            down = to_f32(sd[prefix + "mlp.down_proj.weight"])

            # Convert torch linear weight [out, in] to kernel matmul weight [in, out].
            wq = q_w.T
            wk = k_w.T
            wv = v_w.T
            wo = o_w.T
            q_norm = to_f32(sd[prefix + "self_attn.q_norm.weight"])
            k_norm = to_f32(sd[prefix + "self_attn.k_norm.weight"])
            ffn_gate = gate.T
            ffn_up = up.T
            ffn_down = down.T

            norm1.tofile(f)
            wq.tofile(f)
            wk.tofile(f)
            wv.tofile(f)
            q_norm.tofile(f)
            k_norm.tofile(f)
            wo.tofile(f)
            norm2.tofile(f)
            ffn_gate.tofile(f)
            ffn_up.tofile(f)
            ffn_down.tofile(f)

    print(f"Exported weights to: {os.path.abspath(args.output)}")
    print(
        f"layers={layers}, vocab={vocab}, hidden={target_hidden}, ffn={target_ffn}, seq={args.seq_len}, heads=8"
    )


if __name__ == "__main__":
    main()
