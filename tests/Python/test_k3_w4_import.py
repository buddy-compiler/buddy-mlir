# RUN: %PYTHON %s 2>&1 | FileCheck %s
#
# Variant w4g32 in import_model.py on a tiny random Qwen2 model (no
# download): k3_w4 rewrites every Linear layer and the attention of the
# prefill and decode graphs into kernel calls, writes the kernels, and the
# weights it extracts have the sizes gen_config.py computed from the HF
# config.

import json
import os
import re
import sys
import tempfile
from collections import Counter

# Import buddy/MLIR first to avoid LLVM option conflicts.
from buddy.compiler.frontend import DynamoCompiler  # noqa: F401

sys.path.insert(
    0, os.path.join(os.environ["BUDDY_SRC_ROOT"], "tools", "buddy-codegen")
)
import gen_config  # noqa: E402
import import_model  # noqa: E402
import torch  # noqa: E402
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

torch.manual_seed(0)
hf = Qwen2Config(
    vocab_size=384,
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=64,
    tie_word_embeddings=False,
    architectures=["Qwen2ForCausalLM"],
)
model = Qwen2ForCausalLM(hf).eval()

work = tempfile.mkdtemp()
hf_path = os.path.join(work, "config.json")
with open(hf_path, "w") as f:
    json.dump(hf.to_dict(), f)
spec = {
    "hf_model_path": "test/qwen2-tiny",
    "model_family": "deepseek_r1",
    "variant": "w4g32",
    "max_token_len": 64,
    "prefill_chunk": 32,
    "num_threads": 4,
}
config = gen_config.gen_config(spec, hf_path)
for w in config["weights"]:
    print(f"{w['tag']}: {w['num_elements']} {w['element_type']} {w['file']}")
pipelines = config["compilation"]["pipelines"]
print("pipelines:", ", ".join(f"{k}={v}" for k, v in pipelines.items()))
# f32: embedding 384 x 256; per layer 2 norms and the q / k / v biases
# (256 + 128 + 128); final norm; inv_freq 32
# CHECK: f32_params: 100640 f32 arg0-w4g32-f32.data
# int4, 9 / 16 byte per element: per layer [256, 512] q / k / v, [256, 256]
# o, 2 x [256, 512] gate / up, [512, 256] down; [256, 384] lm_head
# CHECK-NEXT: i8_params: 718848 i8 arg0-w4g32-q4.data
# CHECK-NEXT: pipelines: {{.*}}k3_kernels=kernels


def error(path=hf_path, **extra):
    try:
        c = gen_config.gen_config({**spec, **extra}, path)
    except ValueError as e:
        return f"ValueError: {e}"
    return f"prefill_chunk {c['prefill_chunk']}"


# The kernels' constraints are checked when the config is made.
# CHECK: no chunks: ValueError: w4g32 needs prefill_chunk
print("no chunks:", error(prefill_chunk=0))
# CHECK-NEXT: chunk 48: ValueError: w4g32 needs prefill_chunk to be a multiple of 32, got 48
print("chunk 48:", error(prefill_chunk=48))
# CHECK-NEXT: chunk 64: prefill_chunk 64
print("chunk 64:", error(prefill_chunk=64))
# CHECK-NEXT: chunk true: prefill_chunk 64
print("chunk true:", error(prefill_chunk=True))
odd_path = os.path.join(work, "head_dim_40.json")
with open(odd_path, "w") as f:
    json.dump({**hf.to_dict(), "head_dim": 40}, f)
# CHECK-NEXT: head_dim 40: ValueError: w4g32 needs a head_dim that is a multiple of 16, got 40
print("head_dim 40:", error(odd_path))

prefill, decode, _ = import_model.compile_chunk_graphs(model, config)
import_model.apply_pre_transforms(prefill[0], decode[0])
import_model.apply_k3_w4(prefill[0], decode[0], config, work)


def ops(graph):
    return Counter(type(n).__name__ for n in graph.body)


for name, graph in (("prefill", prefill[0]), ("decode", decode[0])):
    count = ops(graph)
    print(
        f"{name}: CallExternalOp {count['CallExternalOp']}, "
        f"MatmulOp {count['MatmulOp']}, AddMMOp {count['AddMMOp']}, "
        f"attention {count['ScaledDotProductFlashAttentionForCpuOp']}"
    )
# CHECK: prefill: {{.*}}MatmulOp 0, AddMMOp 0, attention 0
# CHECK-NEXT: decode: {{.*}}MatmulOp 0, AddMMOp 0, attention 0

# The kernels write no argument but the attention's KV caches (arguments 3
# and 4), so bufferization copies none of their arguments.
for name, graph in (("prefill", prefill[0]), ("decode", decode[0])):
    written = sorted(
        {
            (n.call_func_name.split("_")[1], str(n.written_args))
            for n in graph.body
            if type(n).__name__ == "CallExternalOp"
        }
    )
    print(f"{name} written_args:", ", ".join(f"{k} {w}" for k, w in written))
# CHECK: prefill written_args: attn [3, 4], q4 []
# CHECK-NEXT: decode written_args: attn [3, 4], q4 []

with open(os.path.join(work, "k3_kernels-w4g32.mlir")) as f:
    kernels = f.read()
for name in sorted(re.findall(r"func\.func @(\w+)\(", kernels)):
    print(name)
# The RMSNorms before q / k / v, gate / up and the decode lm_head are
# computed by those kernels. The lm_head of a prefill chunk gets its last
# row only (one row, after the final norm of the whole chunk), in a kernel
# of its own that the session can switch off (buddy_set_prefill_logits).
# CHECK: buddy_set_prefill_logits{{$}}
# CHECK-NEXT: k3_attn_m1_h4_kv2_d64_c64{{$}}
# CHECK-NEXT: k3_attn_m32_h4_kv2_d64_c64
# CHECK-NEXT: k3_q4_glu_m1_k256_n512_rms
# CHECK-NEXT: k3_q4_glu_m32_k256_n512_rms
# CHECK-NEXT: k3_q4_multi_m1_k256_n256_128_128_b_rms
# CHECK-NEXT: k3_q4_multi_m32_k256_n256_128_128_b_rms
# CHECK-NEXT: k3_q4_plain_m1_k256_n256
# CHECK-NEXT: k3_q4_plain_m1_k256_n384_prefill_logits{{$}}
# CHECK-NEXT: k3_q4_plain_m1_k256_n384_rms{{$}}
# CHECK-NEXT: k3_q4_plain_m1_k512_n256
# CHECK-NEXT: k3_q4_plain_m32_k256_n256
# CHECK-NEXT: k3_q4_plain_m32_k512_n256

buckets = import_model.extract_k3_weights(prefill[0], config)
print({k: (len(v), str(v.dtype)) for k, v in buckets.items()})
# CHECK: {'f32_params': (100640, 'float32'), 'i8_params': (718848, 'int8')}


# "prefill_ime": the prefill kernels of 64 rows run on the matrix engine, on
# a second, IME-layout copy of their weights, and so does the prefill
# attention; the LM head (one row in prefill) and decode keep the tile layout.
# CHECK: ime chunk 32: ValueError: prefill_ime needs prefill_chunk 64, got 32
print("ime chunk 32:", error(prefill_ime=True))
# CHECK-NEXT: ime f32: ValueError: prefill_ime needs the variant w4g32
print(
    "ime f32:",
    error(variant="f32", prefill_ime=True, weights_override={"total": 1}),
)
ime_config = gen_config.gen_config(
    {**spec, "prefill_chunk": 64, "prefill_ime": True}, hf_path
)
for w in ime_config["weights"]:
    print(f"ime {w['tag']}: {w['num_elements']}")
ime_pipelines = ime_config["compilation"]["pipelines"]
print("ime pipelines:", ", ".join(f"{k}={v}" for k, v in ime_pipelines.items()))
# the layers' int4 twice, the LM head once: (2 x 2 x 589824 + 98304) x 9 / 16
# CHECK: ime f32_params: 100640
# CHECK-NEXT: ime i8_params: 1382400
# CHECK-NEXT: ime pipelines: {{.*}}k3_kernels=kernels_a100, k3_kernels_ime=kernels_ime

ime_work = tempfile.mkdtemp()
prefill, decode, _ = import_model.compile_chunk_graphs(model, ime_config)
import_model.apply_pre_transforms(prefill[0], decode[0])
import_model.apply_k3_w4(prefill[0], decode[0], ime_config, ime_work)
for name in ("k3_kernels-w4g32.mlir", "k3_kernels_ime-w4g32.mlir"):
    with open(os.path.join(ime_work, name)) as f:
        text = f.read()
    defined = sorted(
        re.findall(r"func\.func (?:private )?@(\w+)\(.*\{$", text, re.M)
    )
    print(
        name + ":", ", ".join(d for d in defined if "_m64_" in d or "ime" in d)
    )
# CHECK: k3_kernels-w4g32.mlir: k3_attn_m64_h4_kv2_d64_c64_ime, k3_q4_glu_m64_k256_n512_ime_rms, k3_q4_multi_m64_k256_n256_128_128_b_ime_rms, k3_q4_plain_m64_k256_n256_ime, k3_q4_plain_m64_k512_n256_ime
# CHECK-NEXT: k3_kernels_ime-w4g32.mlir: k3_attn_ime_pv, k3_attn_ime_qk, k3_ime_hp_step, k3_q4_glu_m64_k256_n512_ime_rms__tile, k3_q4_multi_m64_k256_n256_128_128_b_ime_rms__tile, k3_q4_plain_m64_k256_n256_ime__tile, k3_q4_plain_m64_k512_n256_ime__tile
buckets = import_model.extract_k3_weights(prefill[0], ime_config)
print("ime weights:", {k: len(v) for k, v in buckets.items()})
# CHECK: ime weights: {'f32_params': 100640, 'i8_params': 1382400}
