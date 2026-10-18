# RUN: %PYTHON %s buddy-opt %openmp_runtime_dir 2>&1 | FileCheck %s
#
# Variant w4g32 in import_model.py on a tiny random Qwen3-MoE model (no
# download): the experts traced as buddy::moe_experts compute what HF
# computes; k3_w4 rewrites each MoE block of the prefill and decode graphs
# into a "moe" kernel with the RMSNorm before it (and in decode the residual
# add after it) fused; both graphs get the same packed expert weights, and
# the kernels run on them (on the host) give HF's MoE blocks.

import ctypes
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from collections import Counter

# Import buddy/MLIR first to avoid LLVM option conflicts.
from buddy.compiler.frontend import DynamoCompiler  # noqa: F401

sys.path.insert(
    0, os.path.join(os.environ["BUDDY_SRC_ROOT"], "tools", "buddy-codegen")
)
import compile_pipeline  # noqa: E402
import gen_config  # noqa: E402
import import_model  # noqa: E402
import numpy  # noqa: E402
import torch  # noqa: E402
from buddy.compiler.graph.transform import k3_w4  # noqa: E402
from buddy.compiler.moe import use_buddy_moe_experts  # noqa: E402
from buddy_mlir import ir  # noqa: E402
from buddy_mlir.execution_engine import ExecutionEngine  # noqa: E402
from buddy_mlir.runtime import (  # noqa: E402
    get_ranked_memref_descriptor,
    make_nd_memref_descriptor,
    ranked_memref_to_numpy,
)
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM  # noqa: E402

BUDDY_OPT, OMP_DIR = sys.argv[1], sys.argv[2]
THREADS = 4
CHUNK = 32

torch.manual_seed(0)
hf = Qwen3MoeConfig(
    vocab_size=384,
    hidden_size=256,
    moe_intermediate_size=128,
    num_experts=8,
    num_experts_per_tok=2,
    norm_topk_prob=True,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    max_position_embeddings=64,
    tie_word_embeddings=False,
    architectures=["Qwen3MoeForCausalLM"],
)
model = Qwen3MoeForCausalLM(hf).eval()

work = tempfile.mkdtemp()
hf_path = os.path.join(work, "config.json")
with open(hf_path, "w") as f:
    json.dump(hf.to_dict(), f)
spec = {
    "hf_model_path": "test/qwen3-moe-tiny",
    "model_family": "deepseek_r1",
    "variant": "w4g32",
    "max_token_len": 64,
    "prefill_chunk": CHUNK,
    "num_threads": THREADS,
}
config = gen_config.gen_config(spec, hf_path)
for w in config["weights"]:
    print(f"{w['tag']}: {w['num_elements']}")
# f32: embedding 384 x 256; per layer 2 norms, q_norm / k_norm (64 each)
# and the router [256, 8]; final norm; inv_freq 32
# CHECK: f32_params: 103968
# int4, 9 / 16 byte per element: per layer [256, 256] q, [256, 128] k / v,
# [256, 256] o, 8 experts x (2 x [256, 128] gate / up, [128, 256] down);
# [256, 384] lm_head
# CHECK-NEXT: i8_params: 1161216

# The kernel computes HF Qwen3-MoE's routing, normalized top-k weights
# included: a model without them is refused when the config is made.
unnormalized = os.path.join(work, "unnormalized.json")
with open(unnormalized, "w") as f:
    json.dump({**hf.to_dict(), "norm_topk_prob": False}, f)
try:
    gen_config.gen_config(spec, unnormalized)
except ValueError as e:
    print("norm_topk_prob false: ValueError:", e)
# CHECK: norm_topk_prob false: ValueError: w4g32 needs norm_topk_prob

# buddy::moe_experts computes HF's experts.
ids = torch.randint(0, hf.vocab_size, (1, 12))
with torch.no_grad():
    want = model(ids).logits
    use_buddy_moe_experts(model)
    got = model(ids).logits
print("moe_experts logits match HF:", torch.allclose(got, want, atol=1e-5))
# CHECK: moe_experts logits match HF: True

# A graph with another routing than the kernel's is refused by the rewrite
# too (here the top-k weights not normalized).
torch.manual_seed(0)
other = Qwen3MoeForCausalLM(
    Qwen3MoeConfig(**{**hf.to_dict(), "norm_topk_prob": False})
).eval()
use_buddy_moe_experts(other)
_, other_decode, _ = import_model.compile_chunk_graphs(other, config)
import_model.apply_pre_transforms(other_decode[0], other_decode[0])
try:
    k3_w4.k3_w4_rewrite(
        other_decode[0], 1, other_decode[0].inputs[1].name, THREADS
    )
except RuntimeError as e:
    print("unnormalized:", e)
# CHECK: unnormalized: k3_w4: moe_experts: not the routing of the moe kernel

prefill, decode, _ = import_model.compile_chunk_graphs(model, config)
import_model.apply_pre_transforms(prefill[0], decode[0])
import_model.apply_k3_w4(prefill[0], decode[0], config, work)
graphs = (("prefill", prefill[0]), ("decode", decode[0]))

for name, graph in graphs:
    count = Counter(type(n).__name__ for n in graph.body)
    moe_calls = sum(
        n.call_func_name.startswith("k3_moe_")
        for n in graph.body
        if type(n).__name__ == "CallExternalOp"
    )
    print(
        f"{name}: moe calls {moe_calls}, MoEExpertsOp "
        f"{count['MoEExpertsOp']}, TopkOp {count['TopkOp']}, f32 MatmulOp "
        f"{count['MatmulOp']}"
    )
# The routing nodes, the routers' matmuls included, are gone: the kernels
# route.
# CHECK: prefill: moe calls 2, MoEExpertsOp 0, TopkOp 0, f32 MatmulOp 0
# CHECK-NEXT: decode: moe calls 2, MoEExpertsOp 0, TopkOp 0, f32 MatmulOp 0

with open(os.path.join(work, "k3_kernels-w4g32.mlir")) as f:
    kernels = f.read()
for name in sorted(re.findall(r"func\.func @(k3_moe\w+)\(", kernels)):
    print(name)
# CHECK: k3_moe_m1_k256_i128_e8_t2_rms_res{{$}}
# CHECK-NEXT: k3_moe_m32_k256_i128_e8_t2_rms{{$}}

buckets = import_model.extract_k3_weights(prefill[0], config)
print({k: len(v) for k, v in buckets.items()})
# CHECK: {'f32_params': 103968, 'i8_params': 1161216}


def params(graph):
    return dict(
        zip((p.name for p in graph.params), graph._params_ref, strict=True)
    )


def moe_calls(graph):
    return [
        n
        for n in graph.body
        if type(n).__name__ == "CallExternalOp"
        and n.call_func_name.startswith("k3_moe_")
    ]


# Both graphs read the same packed experts (pack_moe once per block, the
# cache of k3_w4_rewrite), those of the model's experts.
pp, dp = params(prefill[0]), params(decode[0])
same = all(
    torch.equal(pp[str(a)], dp[str(b)])
    for c, d in zip(moe_calls(prefill[0]), moe_calls(decode[0]), strict=True)
    for a, b in zip(c.args[1:5], d.args[1:5], strict=True)
)
print("prefill and decode MoE weights equal:", same)
# CHECK: prefill and decode MoE weights equal: True


def jit(specs):
    stages = compile_pipeline.build_stages("kernels", THREADS, "", "w4g32")
    passes = next(args for tool, args in stages if tool == "buddy-opt")
    llvm = subprocess.run(
        [BUDDY_OPT, *passes],
        input=k3_w4.gen_kernels(specs),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    with ir.Context():
        return ExecutionEngine(
            ir.Module.parse(llvm),
            opt_level=2,
            shared_libs=[os.path.join(OMP_DIR, "libomp.so")],
            enable_pic=platform.machine().lower() == "riscv64",
        )


def ptr(x):
    return ctypes.pointer(ctypes.pointer(x))


def run(ee, name, *arrays):
    res = make_nd_memref_descriptor(2, ctypes.c_float)()
    args = [ptr(get_ranked_memref_descriptor(a)) for a in arrays]
    ee.invoke(name, ptr(res), *args)
    return ranked_memref_to_numpy([res])


# Each MoE call with its graph's arguments: x (before the RMSNorm), the
# router, the packed gate / up and down, the norm weight (and the residual
# h): HF's post-attention RMSNorm and MoE block with its experts' weights
# rounded to int4 (plus h), up to the kernel's int8 activations.
def int4(w):
    """w [.., N, K] (HF's layout) rounded as the kernels' int4 weights."""
    q = [k3_w4.dequantize_q4(*k3_w4.quantize_q4(m.T.numpy())).T for m in w]
    return torch.from_numpy(numpy.stack(q).astype(numpy.float32))


for block in model.model.layers:
    experts = block.mlp.experts
    with torch.no_grad():
        experts.gate_up_proj.copy_(int4(experts.gate_up_proj))
        experts.down_proj.copy_(int4(experts.down_proj))

rng = numpy.random.default_rng(0)
for name, graph in graphs:
    p = params(graph)
    rows = 1 if name == "decode" else CHUNK
    base = k3_w4.moe_spec(rows, 256, 128, 8, 2, THREADS)
    spec = dict(base, name=base["name"] + "_rms", norm_eps=hf.rms_norm_eps)
    if rows == 1:
        spec = dict(spec, name=spec["name"] + "_res", residual=True)
    ee = jit([spec])
    for layer, c in enumerate(moe_calls(graph)):
        assert c.call_func_name == spec["name"], c.call_func_name
        weights = [p[str(a)].detach().contiguous().numpy() for a in c.args[1:5]]
        x = (3 * rng.standard_normal((rows, 256))).astype(numpy.float32)
        h = rng.standard_normal((rows, 256)).astype(numpy.float32)
        y = run(ee, spec["name"], x, *weights, *([h] if rows == 1 else []))
        if rows == 1:
            y = y - h
        block = model.model.layers[layer]
        with torch.no_grad():
            xn = block.post_attention_layernorm(torch.from_numpy(x)[None])
            want = block.mlp(xn)[0].numpy()
        err = numpy.abs(y - want).max() / numpy.abs(want).max()
        print(f"{name} layer {layer}: error {'ok' if err < 0.03 else err}")
# CHECK: prefill layer 0: error ok
# CHECK-NEXT: prefill layer 1: error ok
# CHECK-NEXT: decode layer 0: error ok
# CHECK-NEXT: decode layer 1: error ok
