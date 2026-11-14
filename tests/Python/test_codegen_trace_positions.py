# RUN: %PYTHON %s 2>&1 | FileCheck %s
#
# import_model traces every graph at cache positions inside the KV cache it
# hands the model, whatever max_token_len is. (A trace past the cache does
# not necessarily fail -- it runs on fake tensors -- so the positions are
# recorded and checked.) A tiny random Qwen2 model, no download.

import os
import sys

# Import buddy/MLIR first to avoid LLVM option conflicts.
from buddy.compiler.frontend import DynamoCompiler

sys.path.insert(
    0, os.path.join(os.environ["BUDDY_SRC_ROOT"], "tools", "buddy-codegen")
)
import import_model  # noqa: E402
import torch  # noqa: E402
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

torch.manual_seed(0)
model = Qwen2ForCausalLM(
    Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
).eval()

traces = []
importer = DynamoCompiler.importer


def recording_importer(self, model, *args, **kwargs):
    position = kwargs.get("cache_position")
    cache = kwargs.get("past_key_values")
    if position is not None:
        traces.append(
            (
                self._func_name,
                int(position.min()),
                int(position.max()),
                cache.max_cache_len,
            )
        )
    return importer(self, model, *args, **kwargs)


DynamoCompiler.importer = recording_importer

MAX_TOKEN_LEN = 16
import_model.compile_graphs(model, {"shape": {"max_token_len": MAX_TOKEN_LEN}})
for chunk in (8, MAX_TOKEN_LEN):
    import_model.compile_chunk_graphs(
        model,
        {"shape": {"max_token_len": MAX_TOKEN_LEN}, "prefill_chunk": chunk},
    )

for func, lo, hi, cache_len in traces:
    where = "in range" if 0 <= lo <= hi < cache_len else "OUT OF RANGE"
    print(f"{func}: positions {lo}..{hi}, cache {cache_len}: {where}")
# compile_graphs
# CHECK: forward_decode: positions 1..1, cache 16: in range
# compile_chunk_graphs, chunks of 8 and 16
# CHECK: forward_prefill: positions 0..7, cache 16: in range
# CHECK: forward_decode: positions 1..1, cache 16: in range
# CHECK: forward_prefill: positions 0..15, cache 16: in range
# CHECK: forward_decode: positions 1..1, cache 16: in range
# CHECK-NOT: OUT OF RANGE
