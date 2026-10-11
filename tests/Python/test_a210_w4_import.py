# RUN: %PYTHON %s 2>&1 | FileCheck %s
#
# Variant w4g32 with "kernels": "a210" in import_model.py on a tiny random
# Qwen2 model (no download): the graph rewrite of k3_w4 with the A210
# kernels (graph/transform/a210_w4.py), written to a210_kernels-w4g32.mlir,
# the thread weights of the spec in their specs, and the weights the size
# gen_config.py computed. The spec fields are checked by gen_config.py.

import json
import os
import re
import sys
import tempfile

# Import buddy/MLIR first to avoid LLVM option conflicts.
from buddy.compiler.frontend import DynamoCompiler  # noqa: F401

sys.path.insert(
    0, os.path.join(os.environ["BUDDY_SRC_ROOT"], "tools", "buddy-codegen")
)
import gen_config  # noqa: E402
import import_model  # noqa: E402
import torch  # noqa: E402
from buddy.compiler.graph.transform import a210_w4  # noqa: E402
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
weights = {"prefill": [1, 1, 3, 3], "decode": [1, 1, 0.5, 0.5]}
spec = {
    "hf_model_path": "test/qwen2-tiny",
    "model_family": "deepseek_r1",
    "variant": "w4g32",
    "kernels": "a210",
    "thread_weights": weights,
    "thread_cores": ["c908", "c908", "c920", "c920"],
    "max_token_len": 64,
    "prefill_chunk": 32,
    "num_threads": 4,
    "thread_pool": True,
}
config = gen_config.gen_config(spec, hf_path)
for w in config["weights"]:
    print(f"{w['tag']}: {w['num_elements']} {w['element_type']} {w['file']}")
pipelines = config["compilation"]["pipelines"]
print("pipelines:", ", ".join(f"{k}={v}" for k, v in pipelines.items()))
print("kernels:", config["kernels"], config["thread_weights"])
print("cores:", config["thread_cores"])
print("layout:", import_model.w4g32_layout(config))
# the same buffers as with the K3 kernels (test_k3_w4_import.py)
# CHECK: f32_params: 100640 f32 arg0-w4g32-f32.data
# CHECK-NEXT: i8_params: 718848 i8 arg0-w4g32-q4.data
# CHECK-NEXT: pipelines: {{.*}}subgraph_decode=subgraph_decode, a210_kernels=kernels{{$}}
# CHECK-NEXT: kernels: a210 {'prefill': [1, 1, 3, 3], 'decode': [1, 1, 0.5, 0.5]}
# CHECK-NEXT: cores: ['c908', 'c908', 'c920', 'c920']
# the A210 version includes the K3 one (k3_w4.quantize_q4 makes the values)
# CHECK-NEXT: layout: a210:1.k3-1


def error(**extra):
    try:
        c = gen_config.gen_config({**spec, **extra}, hf_path)
    except ValueError as e:
        return f"ValueError: {e}"
    return f"kernels {c['kernels']}"


# CHECK: kernels x86: ValueError: kernels must be one of ('k3', 'a210'), got 'x86'
print("kernels x86:", error(kernels="x86"))
# CHECK-NEXT: kernels f32: ValueError: kernels needs the variant w4g32
print("kernels f32:", error(variant="f32", weights_override={"total": 1}))
# CHECK-NEXT: a210 ime: ValueError: prefill_ime needs the kernels k3
print("a210 ime:", error(prefill_ime=True, prefill_chunk=64))
# CHECK-NEXT: weights k3: ValueError: thread_weights needs the kernels a210
print("weights k3:", error(kernels="k3"))
# CHECK-NEXT: weights 3: ValueError: thread_weights decode must be 4 positive numbers (num_threads), got [1, 1, 1]
print("weights 3:", error(thread_weights={"decode": [1, 1, 1]}))
# CHECK-NEXT: weights key: ValueError: thread_weights must be {"prefill": [...], "decode": [...]}
print("weights key:", error(thread_weights={"attention": [1, 1, 1, 1]}))
# CHECK-NEXT: no weights: kernels a210
print("no weights:", error(thread_weights=None))
# CHECK-NEXT: cores k3: ValueError: thread_cores needs the kernels a210
print("cores k3:", error(kernels="k3", thread_weights=None))
# CHECK-NEXT: cores x: ValueError: thread_cores must be 4 of ('c908', 'c920') (num_threads), got ['c908', 'c920']
print("cores x:", error(thread_cores=["c908", "c920"]))
# Per-thread fields need the pinned threads of the thread pool.
# CHECK-NEXT: weights no pool: ValueError: thread_weights needs thread_pool true
print("weights no pool:", error(thread_pool=False))
# CHECK-NEXT: cores no pool: ValueError: thread_cores needs thread_pool true
print("cores no pool:", error(thread_pool=False, thread_weights=None))
# CHECK-NEXT: kernels null: kernels k3
print(
    "kernels null:", error(kernels=None, thread_weights=None, thread_cores=None)
)

prefill, decode, _ = import_model.compile_chunk_graphs(model, config)
import_model.apply_pre_transforms(prefill[0], decode[0])
import_model.apply_k3_w4(prefill[0], decode[0], config, work)
print("k3 kernels written:", os.path.exists(f"{work}/k3_kernels-w4g32.mlir"))
# CHECK: k3 kernels written: False
with open(os.path.join(work, "a210_kernels-w4g32.mlir")) as f:
    kernels = f.read()
for name in sorted(re.findall(r"func\.func @(\w+)\(", kernels)):
    print(name)
# CHECK: a210_attn_m1_h4_kv2_d64_c64{{$}}
# CHECK-NEXT: a210_attn_m32_h4_kv2_d64_c64
# CHECK-NEXT: a210_q4_glu_m1_k256_n512_rms
# CHECK-NEXT: a210_q4_glu_m32_k256_n512_rms
# CHECK-NEXT: a210_q4_multi_m1_k256_n256_128_128_b_rms
# CHECK-NEXT: a210_q4_multi_m32_k256_n256_128_128_b_rms
# CHECK-NEXT: a210_q4_plain_m1_k256_n256
# CHECK-NEXT: a210_q4_plain_m1_k256_n384_prefill_logits{{$}}
# CHECK-NEXT: a210_q4_plain_m1_k256_n384_rms{{$}}
# CHECK-NEXT: a210_q4_plain_m1_k512_n256
# CHECK-NEXT: a210_q4_plain_m32_k256_n256
# CHECK-NEXT: a210_q4_plain_m32_k512_n256
# CHECK-NEXT: buddy_set_prefill_logits{{$}}
tiles = sorted(
    set(re.findall(r"@(a210_q4_plain_m32_k256_n256__tile\w*)\(", kernels))
)
print("prefill tiles:", tiles)
# CHECK: prefill tiles: ['a210_q4_plain_m32_k256_n256__tile_r4a2_generic', 'a210_q4_plain_m32_k256_n256__tile_r4a2_sifive_u74']

# The weights are packed in the A210 layout: the o projection of layer 0
# (a [256, 256] weight in a parameter of its own).
buckets = import_model.extract_k3_weights(prefill[0], config)
print({k: (len(v), str(v.dtype)) for k, v in buckets.items()})
# CHECK: {'f32_params': (100640, 'float32'), 'i8_params': (718848, 'int8')}
o_proj = model.model.layers[0].self_attn.o_proj.weight.detach().numpy().T
packed = a210_w4.pack_q4(o_proj)
print(
    "o_proj packed in i8_params:",
    packed.tobytes() in buckets["i8_params"].tobytes(),
)
# CHECK: o_proj packed in i8_params: True

# Switching the kernel collection never reuses the weights of the other
# one: K3 and A210 weight files have the same sizes but other bytes, and the
# manifests record different layouts.
k3_spec = {
    k: v
    for k, v in spec.items()
    if k not in ("kernels", "thread_weights", "thread_cores")
}
k3_config = gen_config.gen_config(k3_spec, hf_path)
switch_work = tempfile.mkdtemp()
for entry in import_model.expected_weight_entries(k3_config, switch_work):
    with open(entry["path"], "wb") as f:
        f.truncate(entry["bytes_per_element"] * entry["num_elements"])
import_model.write_weights_manifest(k3_config, switch_work)
print(
    "k3 reuses k3:",
    import_model.can_reuse_existing_weights(k3_config, switch_work),
)
print(
    "a210 reuses k3:",
    import_model.can_reuse_existing_weights(config, switch_work),
)
# CHECK: k3 reuses k3: True
# CHECK-NEXT: a210 reuses k3: False
