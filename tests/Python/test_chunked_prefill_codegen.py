# RUN: %PYTHON %s 2>&1 | FileCheck %s
#
# Chunked prefill (spec field `prefill_chunk`) in the buddy-codegen
# generators: the derived config, the generated ModelSession::prefill() and
# the manifest. Without `prefill_chunk` nothing changes.

import json
import os
import re
import sys
import tempfile

sys.path.insert(
    0, os.path.join(os.environ["BUDDY_SRC_ROOT"], "tools", "buddy-codegen")
)
import gen_config  # noqa: E402
import gen_manifest  # noqa: E402
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


# CHECK: off: 0 deepseek_r1_f32
c = config()
print("off:", c["prefill_chunk"], c["model_id"])
# CHECK: default: 64 deepseek_r1_f32_prefill_chunk64
c = config(prefill_chunk=True)
print("default:", c["prefill_chunk"], c["model_id"])
# CHECK: 32: 32
print("32:", config(prefill_chunk=32)["prefill_chunk"])
# CHECK: too long: ValueError: prefill_chunk (512) must not exceed max_token_len (256)
print("too long:", error(prefill_chunk=512))
# CHECK: negative: ValueError: prefill_chunk must be true/false or a non-negative integer, got -1
print("negative:", error(prefill_chunk=-1))
# CHECK: tiered: ValueError: prefill_chunk and tiered_kv_cache are mutually exclusive
print(
    "tiered:", error(prefill_chunk=64, tiered_kv_cache=True, cache_sizes=[256])
)

# Without chunks: the PrefillABI and the single forward_prefill call.
plain = gen_session.gen_impl(config())


def summary(impl):
    return (
        f"PrefillABI {int('struct PrefillABI {' in impl)}, "
        f"PrefillFn = DecodeFn {int('using PrefillFn = DecodeFn;' in impl)}, "
        f"1 x max_token_len prefill logits "
        f"{int('cfg_.maxTokenLen, cfg_.vocabSize}' in impl)}"
    )


# CHECK: plain: PrefillABI 1, PrefillFn = DecodeFn 0, 1 x max_token_len prefill logits 1
print("plain:", summary(plain))

# With chunks: forward_prefill has the decode ABI and prefill() loops.
impl = gen_session.gen_impl(config(prefill_chunk=32))
# CHECK: chunked: PrefillABI 0, PrefillFn = DecodeFn 1, 1 x max_token_len prefill logits 0
print("chunked:", summary(impl))
prefill = re.search(r"void ModelSession::prefill\(.*?\n}\n", impl, re.S).group(
    0
)
print(prefill)
# CHECK-LABEL: void ModelSession::prefill(Text<size_t, 2> &tokens) {
# CHECK: constexpr int C = 32;
# CHECK: const int n = n0 < C ? n0 - 1 : n0;
# CHECK: std::memset(state.kv(i).getData(), 0,
# CHECK: while (n > 0) {
# CHECK-NEXT: if (start + C > n)
# CHECK-NEXT: start = std::max(0, n - C);
# CHECK: cachePosition_->getData()[0] = (long long)start;
# CHECK: impl_->prefillFn(
# CHECK-NEXT: &result, params_.get(), impl_->chunkTokens.get(), cachePosition_.get(),
# CHECK: const int row = std::min(n, start + C) - 1 - start;
# CHECK: intptr_t logitsShape[3] = {1, C, cfg_.vocabSize};
# CHECK: if (n < n0) {
# CHECK-NEXT: position_ = n;
# CHECK-NEXT: decode((int)tokens.getData()[n0 - 1]);

# The manifest: chunk tokens, their start position, chunk logits.
manifest = gen_manifest.gen_manifest(config(prefill_chunk=32), "model.so")
for line in manifest.splitlines():
    if "prefill" in line and ("rhal.buffer" in line or "inputs" in line):
        print(line.strip())
# CHECK: rhal.buffer @prefill_tokens {space = "host", type = tensor<1x32xi64>}
# CHECK: rhal.buffer @logits_prefill {space = "host", type = tensor<1x32x1000xf32>}
