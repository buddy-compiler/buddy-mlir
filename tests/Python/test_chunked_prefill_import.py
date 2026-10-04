# RUN: %PYTHON %s 2>&1 | FileCheck %s
#
# import_model.compile_chunk_graphs traces forward_prefill on its own, empty
# KV cache, so that a chunk as long as the cache (prefill_chunk ==
# max_token_len) is in range. A tiny random Qwen2 model, no download.

import os
import sys

# Import buddy/MLIR first to avoid LLVM option conflicts.
from buddy.compiler.frontend import DynamoCompiler  # noqa: F401

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
        max_position_embeddings=256,
    )
).eval()

MAX_TOKEN_LEN = 32


def shapes(graph):
    """Shapes of the token ids input and of the logits output."""
    ids = [
        tuple(n.tensor_meta["shape"])
        for n in graph.inputs
        if str(n.tensor_meta["dtype"]).lower().endswith("int64")
        and len(n.tensor_meta["shape"]) == 2
    ]
    out = graph.body[-1]
    logits = [
        tuple(graph.node_table[a].tensor_meta["shape"])
        for a in out.args
        if len(graph.node_table[a].tensor_meta["shape"]) == 3
    ]
    return ids, logits


for chunk in (8, MAX_TOKEN_LEN):
    config = {
        "shape": {"max_token_len": MAX_TOKEN_LEN},
        "prefill_chunk": chunk,
    }
    prefill, decode, params = import_model.compile_chunk_graphs(model, config)
    print(f"chunk {chunk}: prefill {shapes(prefill[0])}")
    print(f"chunk {chunk}: decode {shapes(decode[0])}")
# CHECK: chunk 8: prefill ([(1, 8)], [(1, 8, 128)])
# CHECK: chunk 8: decode ([(1, 1)], [(1, 1, 128)])
# CHECK: chunk 32: prefill ([(1, 32)], [(1, 32, 128)])
# CHECK: chunk 32: decode ([(1, 1)], [(1, 1, 128)])

try:
    import_model.compile_chunk_graphs(
        model,
        {"shape": {"max_token_len": MAX_TOKEN_LEN}, "prefill_chunk": 33},
    )
except ValueError as e:
    print("ValueError:", e)
# CHECK: ValueError: prefill_chunk (33) must be in 1 .. max_token_len (32)
