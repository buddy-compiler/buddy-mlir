# RUN: %PYTHON %s 2>&1 | FileCheck %s
#
# The memory options of the model spec, `arena` and `hugepages`
# (docs/ModelMemoryOptions.md), in the buddy-codegen generators: the derived
# config, the passes compile_pipeline.py runs and the generated
# ModelSession. Without them nothing changes.

import json
import os
import re
import sys
import tempfile

sys.path.insert(
    0, os.path.join(os.environ["BUDDY_SRC_ROOT"], "tools", "buddy-codegen")
)
import compile_pipeline  # noqa: E402
import gen_config  # noqa: E402
import gen_session  # noqa: E402

# A small Qwen2-like model: 4 layers (8 KV caches), 2 KV heads.
HF = {
    "architectures": ["Qwen2ForCausalLM"],
    "hidden_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_hidden_layers": 4,
    "vocab_size": 1000,
    "eos_token_id": 2,
}
hf_path = os.path.join(tempfile.mkdtemp(), "config.json")
with open(hf_path, "w") as f:
    json.dump(HF, f)


def config(**extra):
    spec = {
        "hf_model_path": "test/qwen2-tiny",
        "model_family": "deepseek_r1",
        "variant": "f32",
        "max_token_len": 256,
        "weights_override": {"total": 1000},
        **extra,
    }
    return gen_config.gen_config(spec, hf_path)


def error(**extra):
    try:
        config(**extra)
    except ValueError as e:
        return f"ValueError: {e}"
    return "no error"


# CHECK: off: arena False hugepages False
c = config()
print("off:", "arena", c["arena"], "hugepages", c["hugepages"])
# CHECK: on: arena True hugepages True
c = config(arena=True, hugepages=True, prefill_chunk=32)
print("on:", "arena", c["arena"], "hugepages", c["hugepages"])
# CHECK: not a bool: ValueError: arena must be true or false, got 1
print("not a bool:", error(arena=1))
# CHECK: no chunks: ValueError: arena needs prefill_chunk: a prefill call over max_token_len positions allocates too much to keep all of it
print("no chunks:", error(arena=True))
# CHECK: hugepages alone: True
print("hugepages alone:", config(hugepages=True)["hugepages"])
# CHECK: tiered: ValueError: arena and tiered_kv_cache are mutually exclusive
print("tiered:", error(arena=True, tiered_kv_cache=True, cache_sizes=[256]))
# "thread_pool" (docs/ModelThreadPool.md) is validated the same way.
# CHECK: thread_pool: False True
print(
    "thread_pool:",
    config()["thread_pool"],
    config(thread_pool=True)["thread_pool"],
)
# CHECK: thread_pool not a bool: ValueError: thread_pool must be true or false, got 'yes'
print("thread_pool not a bool:", error(thread_pool="yes"))


# compile_pipeline.py: with the arena, the allocations call the generic MLIR
# functions and the buffer deallocation passes are gone, in every pipeline.
def passes(pipeline_type, arena):
    stages = compile_pipeline.build_stages(
        pipeline_type, 4, "", "f32", arena=arena
    )
    args = [a for tool, a in stages if tool == "buddy-opt"]
    flat = [a for stage in args for a in stage]
    lower = [a for a in flat if a.startswith("-finalize-memref-to-llvm")]
    dealloc = [a for a in flat if "deallocation" in a]
    return f"{' '.join(lower)} | deallocation passes {len(dealloc)}"


for pipeline_type in ("standard", "subgraph", "subgraph_decode", "forward"):
    for arena in (False, True):
        print(f"{pipeline_type} arena={arena}: {passes(pipeline_type, arena)}")
# CHECK: standard arena=False: -finalize-memref-to-llvm | deallocation passes 3
# CHECK: standard arena=True: -finalize-memref-to-llvm=use-generic-functions=true | deallocation passes 0
# CHECK: subgraph arena=False: -finalize-memref-to-llvm | deallocation passes 3
# CHECK: subgraph arena=True: -finalize-memref-to-llvm=use-generic-functions=true | deallocation passes 0
# CHECK: subgraph_decode arena=False: -finalize-memref-to-llvm | deallocation passes 3
# CHECK: subgraph_decode arena=True: -finalize-memref-to-llvm=use-generic-functions=true | deallocation passes 0
# CHECK: forward arena=False: -finalize-memref-to-llvm | deallocation passes 0
# CHECK: forward arena=True: -finalize-memref-to-llvm=use-generic-functions=true | deallocation passes 0


# The session: buddy_arena_reset is resolved when the model is loaded and
# called before every forward call; the results are released, not freed.
def function(impl, name):
    return re.search(
        rf"\n[^\n]*{re.escape(name)}\(.*?\n}}\n", impl, re.S
    ).group(0)


def arena_lines(impl):
    out = []
    for line in impl.splitlines():
        line = line.strip()
        if (
            "arenaReset" in line
            or "Fn(" in line
            and "impl_->" in line
            or line.startswith("releaseDecodeABI(abi)")
        ):
            out.append(line)
    return out


plain = gen_session.gen_impl(config())
words = ("arenaReset", "releaseDecodeABI", "adviseHugePages", "sys/mman.h")
# CHECK: plain: arenaReset 0, releaseDecodeABI 0, adviseHugePages 0, sys/mman.h 0
print("plain:", ", ".join(f"{w} {plain.count(w)}" for w in words))

impl = gen_session.gen_impl(config(arena=True, prefill_chunk=32))
print("--- arena")
print("\n".join(arena_lines(impl)))
# CHECK-LABEL: --- arena
# CHECK: releaseDecodeABI(abi);
# CHECK: void (*arenaReset)() = nullptr;
# CHECK: arenaReset = reinterpret_cast<void (*)()>(
# CHECK: impl_->arenaReset();
# CHECK-NEXT: impl_->prefillFn(
# CHECK: impl_->arenaReset();
# CHECK-NEXT: impl_->decodeFn(
print(function(impl, "void resetDecodeResultABI"))
# CHECK-LABEL: void resetDecodeResultABI(DecodeABI &abi, intptr_t kvShape[4],
# CHECK-NEXT: intptr_t logitsShape[3], intptr_t pshape[1]) {
# CHECK-NEXT: releaseDecodeABI(abi);
# CHECK-NEXT: destroyDecodeABI(abi);

# Huge pages: the weight buffer is advised before the weights are read in.
impl = gen_session.gen_impl(config(hugepages=True))
print(function(impl, "void ModelSession::loadWeights"))
# CHECK-LABEL: void ModelSession::loadWeights(
# CHECK: params_ = std::make_unique<MemRef<float, 1>>(shape);
# CHECK-NEXT: adviseHugePages(params_->getData(), sizeof(float) * params_->getSize());
# CHECK-NEXT: std::ifstream f(paths[0], std::ios::binary);
