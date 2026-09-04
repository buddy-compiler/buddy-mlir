#!/usr/bin/env python3
# ===- partition_strategy.py - Mistral layer split strategy ----------------===//
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===----------------------------------------------------------------------===//

from buddy.compiler.graph import SplitStrategy
from buddy.compiler.graph.operation import FlashAttentionForCpuPrefillOp
from buddy.compiler.graph.operation import GQAAttentionFusedOp

# === ABI output-order fix ====================================================
# The shared tools/buddy-codegen/import_model.py applies a DeepSeek-style prefill
# output remap (adjacent K/V swap) when exporting layer-partitioned MLIR. That
# swap assumes the model's natural graph output order is [V0,K0,V1,K1,...,logits]
# (as produced by the DeepSeek trace). Mistral is a Llama-family decoder whose
# natural order is [logits,K0,V0,K1,V1,...], so the swap mangles the exported
# forward_prefill.mlir return order to [K0,logits,K1,V0,...], which no longer
# matches gen_session.py's PrefillABI ([kv0..kv{n-1}, logits]). ModelSession::
# prefill then memcpy's maxTokenLen*vocabSize bytes out of the last 4MB KV slot,
# which segfaults. We restore the order gen_session.py expects by re-mapping the
# remap from "adjacent swap" to "move logits to the end".

# The generic decode export has the same ABI gap, in reverse: gen_session.py's
# DecodeABI interleaves a dummy input between each (K,V) pair (dummies for 2n
# KV tensors) and also returns cachePositionOut + the dummies, while the generic
# driver emits a contiguous decode wrapper. We rewrite the text of the emitted
# forward_decode.mlir wrapper to match DecodeABI. Everything (n, KV type, logits
# type) is derived from the module text, so the same code works for any Llama
# family decoder regardless of kv count, head shape or vocab size.


def _transform_decode_text(module_text):
    """Rewrite the public @forward_decode wrapper to the DecodeABI layout.

    The wrapper emitted by PartitionedGraphDriver has contiguous kv args
    (weights, token, cachePos, kv0..kv{n-1}) and returns [kv0..kv{n-1}, logits].
    gen_session.py's DecodeABI expects, after cachePos, the interleaved order
    kv0, kv1, dummy0, kv2, kv3, dummy1, ..., dummy_{d-1}, kv_{n-2}, kv_{n-1}
    (d = n/2 - 1 dummies) and the return tuple [cachePosOut, kv0, kv1, dummy0,
    ..., kv_{n-1}, logits]. Only the public wrapper is rewritten; the private
    @subgraph0_decode0 keeps its contiguous signature.
    """
    import re

    DUMMY = "memref<1xi64>"

    def split_top(s):
        s = s.strip()
        if s.startswith("(") and s.endswith(")"):
            s = s[1:-1]
        out, depth, cur = [], 0, ""
        for ch in s:
            if ch in "<([" :
                depth += 1
            elif ch in ">)]":
                depth -= 1
            if ch == "," and depth == 0:
                out.append(cur)
                cur = ""
            else:
                cur += ch
        if cur.strip():
            out.append(cur)
        return out

    m = re.search(
        r"func\.func @forward_decode(\(.*?\))\s*->\s*\((.*?)\)\s*\{",
        module_text, re.S)
    if m is None:
        raise ValueError("_transform_decode_text: no func.func @forward_decode")

    args = [a.strip() for a in split_top(m.group(1))]
    rets = [t.strip() for t in split_top(m.group(2))]

    n_kv = len(rets) - 1
    if len(args) != 3 + n_kv:
        raise ValueError(
            f"_transform_decode_text: unexpected arg count {len(args)}")
    KV = args[3].split(":", 1)[1].strip()
    LOGITS = rets[-1]
    if not all(a.split(":", 1)[1].strip() == KV for a in args[3:3 + n_kv]):
        raise ValueError("_transform_decode_text: kv args not contiguous")
    if n_kv < 4 or n_kv % 2 != 0:
        raise ValueError(f"_transform_decode_text: bad n_kv={n_kv}")
    n_dummy = n_kv // 2 - 1

    # new interleaved arg positions
    kv_newpos = {0: 3, 1: 4}
    for i in range(n_dummy):
        kv_newpos[2 + 2 * i] = 6 + 3 * i
        kv_newpos[3 + 2 * i] = 7 + 3 * i
    dummy_newpos = {i: 5 + 3 * i for i in range(n_dummy)}

    new_args = [None] * (3 + n_kv + n_dummy)
    new_args[0], new_args[1], new_args[2] = args[0], args[1], args[2]
    for k in range(n_kv):
        p = kv_newpos[k]
        new_args[p] = f"%arg{p}: {KV}"
    for i in range(n_dummy):
        p = dummy_newpos[i]
        new_args[p] = f"%arg{p}: {DUMMY}"
    if any(a is None for a in new_args):
        raise ValueError("_transform_decode_text: holes in new arg list")

    new_rets = [DUMMY]
    for i in range(n_dummy):
        new_rets.extend([KV, KV, DUMMY])
    new_rets.extend([KV, KV, LOGITS])
    if len(new_rets) != 1 + n_kv + n_dummy + 1:
        raise ValueError("_transform_decode_text: bad new return count")

    # body rewrite
    func_start = m.start()
    brace_open = module_text.find("{", m.start())
    depth = 0
    close = None
    for j in range(brace_open, len(module_text)):
        if module_text[j] == "{":
            depth += 1
        elif module_text[j] == "}":
            depth -= 1
            if depth == 0:
                close = j
                break
    if close is None:
        raise ValueError("_transform_decode_text: unbalanced body")
    body = module_text[brace_open + 1:close]
    func_end = close + 1

    cm = re.search(
        r"%(\w+)(?::\d+)?\s*=\s*(?:func\.)?call @subgraph0_decode0\(", body, re.S)
    if cm is None:
        raise ValueError("_transform_decode_text: no call @subgraph0_decode0")
    res_var = cm.group(1)

    # single-pass cast-source rename (no sequential re.sub cascade)
    old_range = set(range(3, 3 + n_kv))

    def _map(mo):
        src = mo.group(2)  # %argN
        try:
            k = int(src[4:])
        except ValueError:
            return mo.group(0)
        if k not in old_range:
            return mo.group(0)
        return f"%arg{kv_newpos[k - 3]}"

    body2 = re.sub(
        r"(memref\.cast )(%arg\d+)( : " + re.escape(KV) + r")",
        lambda mo: mo.group(1) + _map(mo) + mo.group(3),
        body)

    retm = re.search(r"(?m)^\s*return (.+)$", body2)
    if retm is None:
        raise ValueError("_transform_decode_text: no return in body")
    res_idx = [int(o.split("#")[1])
               for o in retm.group(1).split(" : ")[0].split(",")]
    kv_residx = {k: res_idx[k] for k in range(n_kv)}
    logits_residx = res_idx[n_kv]

    new_return_ops = ["%arg2"]  # cachePositionOut passthrough
    for i in range(n_dummy):
        new_return_ops.append(f"%{res_var}#{kv_residx[2 * i]}")
        new_return_ops.append(f"%{res_var}#{kv_residx[2 * i + 1]}")
        new_return_ops.append(f"%arg{dummy_newpos[i]}")
    new_return_ops.append(f"%{res_var}#{kv_residx[n_kv - 2]}")
    new_return_ops.append(f"%{res_var}#{kv_residx[n_kv - 1]}")
    new_return_ops.append(f"%{res_var}#{logits_residx}")
    if len(new_return_ops) != len(new_rets):
        raise ValueError("_transform_decode_text: return ops/type mismatch")

    ret_line = "  return " + ", ".join(new_return_ops) + " : " + \
        ", ".join(new_rets) + "\n"
    new_body = body2[:retm.start()] + ret_line + body2[retm.end():]

    new_sig = "func.func @forward_decode(" + ", ".join(new_args) + ") -> (" + \
        ", ".join(new_rets) + ") {"
    return module_text[:func_start] + new_sig + new_body + \
        module_text[func_end - 1:]


def _patch_combined_graph_abi():
    try:
        from buddy.compiler.graph.partitioned_graph_driver import (
            PartitionedGraphDriver,
        )
    except Exception:
        return

    _orig = PartitionedGraphDriver.construct_combined_main_graph

    def _construct(self, do_param_pack=False, output_index_remap=None):
        if output_index_remap is not None:
            n = len(output_index_remap)
            kv_count = n - 1
            is_deepseek_swap = (
                n >= 3
                and (n - 1) % 2 == 0
                and all(
                    (output_index_remap[i] == (i ^ 1)) if i < kv_count else
                    (output_index_remap[i] == i)
                    for i in range(n)
                )
            )
            if is_deepseek_swap:
                # natural order is [logits, kv0, kv1, ..., kv_{n-2}]; reorder
                # to [kv0, kv1, ..., kv_{n-2}, logits].
                output_index_remap = list(range(1, n)) + [0]
            return _orig(self, do_param_pack, output_index_remap)
        # decode combined graph: rewrite the emitted wrapper text to DecodeABI
        combined = _orig(self, do_param_pack)
        return _transform_decode_text(str(combined))

    PartitionedGraphDriver.construct_combined_main_graph = _construct


_patch_combined_graph_abi()


def layer_split_strategy(kind: str) -> SplitStrategy:
    """Return the Mistral (MistralForCausalLM, 32 layers) layer split strategy.

    Mistral-7B-Instruct-v0.2 is a dense Llama-family decoder (no MoE routing
    PowOp like DeepSeek R1). The fusion pass produces exactly one fused
    attention op per decoder layer -- FlashAttentionForCpuPrefillOp in the
    prefill graph (after `flash_attention_prefill`) and GQAAttentionFusedOp in
    the decode graph (after `gqa_attention_fusion`) -- so a vertical split on
    those boundaries yields one subgraph per layer. parallel_num stays 1:
    weights and tensor shapes are never horizontally sharded.
    """
    if kind == "prefill":
        return SplitStrategy(
            name="mistral_prefill_layers",
            parallel_num=1,
            ops_count=[],
            stage_boundary_op=FlashAttentionForCpuPrefillOp,
            stage_boundary_op_num=32,
        )
    if kind == "decode":
        return SplitStrategy(
            name="mistral_decode_layers",
            parallel_num=1,
            ops_count=[],
            stage_boundary_op=GQAAttentionFusedOp,
            stage_boundary_op_num=32,
        )
    raise ValueError(f"unknown Mistral layer split kind: {kind}")
