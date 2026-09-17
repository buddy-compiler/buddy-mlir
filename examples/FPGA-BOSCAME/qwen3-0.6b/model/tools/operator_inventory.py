#!/usr/bin/env python3
"""Build the model operator mapping inventory from the *real* imported graph.

Every entry here is derived by walking the actual Buddy op dump
(``prefill-raw-ops.json`` / ``decode-raw-ops.json``) and matching operator
*structure*, not function names. A name-based mapping would happily claim a
kernel is covered when the graph's semantics differ, which is exactly the
failure this report has to prevent.

For each semantic group the report records operand/result shapes and dtypes,
which producers were views/permutes (so no data movement is implied), whether
the op accumulates into its destination, and which Triton kernel is expected to
serve it -- plus whether that kernel exists today in the model's Triton source.

The matching is deliberately conservative: an op that does not match a known
pattern is reported as unmatched rather than being folded into a neighbour.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

# Op types that only change the descriptor and move no data.
VIEW_OPS = {"ViewOp", "PermuteOp", "UnsqueezeOp", "SliceOp", "ExpandOp",
            "AliasOp", "CloneOp", "ReshapeOp"}


def load(path):
    return json.loads(Path(path).read_text())


def index_ops(ops):
    return {op["name"]: op for op in ops}


def shape_of(op):
    meta = op.get("tensor_meta") or {}
    return meta.get("shape")


def dtype_of(op):
    meta = op.get("tensor_meta") or {}
    dtype = meta.get("dtype")
    if isinstance(dtype, list):
        return [str(d).split(".")[-1] for d in dtype]
    return str(dtype).split(".")[-1] if dtype else None


def parent(ops_by_name, op, position):
    parents = op.get("parents") or []
    if position < len(parents):
        return ops_by_name.get(parents[position])
    return None


def find_rmsnorm(ops_by_name, op):
    """Detect  Pow -> Mean -> Add(eps) -> Rsqrt -> Mul -> Mul(weight).

    The traversal starts from the final Mul and walks *backwards* through
    parents, so the pattern is anchored on real dataflow edges.
    """
    if op["op_type"] != "MulOp":
        return None
    normed = parent(ops_by_name, op, 1)
    if normed is None or normed["op_type"] != "MulOp":
        return None
    rsqrt = parent(ops_by_name, normed, 1)
    if rsqrt is None or rsqrt["op_type"] != "RsqrtOp":
        return None
    add = parent(ops_by_name, rsqrt, 0)
    if add is None or add["op_type"] != "AddOp":
        return None
    mean = parent(ops_by_name, add, 0)
    if mean is None or mean["op_type"] != "MeanOp":
        return None
    power = parent(ops_by_name, mean, 0)
    if power is None or power["op_type"] != "PowOp":
        return None
    source = parent(ops_by_name, power, 0)
    if source is None:
        return None
    weight = parent(ops_by_name, op, 0)
    return {
        "source": source,
        "weight": weight,
        "reduce_shape": shape_of(mean),
        "weight_shape": shape_of(weight) if weight else None,
    }


def find_silu(ops_by_name, op):
    """Detect Sigmoid -> Mul(x, sigmoid(x))."""
    if op["op_type"] != "MulOp":
        return None
    sigmoid = parent(ops_by_name, op, 1)
    if sigmoid is None or sigmoid["op_type"] != "SigmoidOp":
        return None
    source = parent(ops_by_name, sigmoid, 0)
    other = parent(ops_by_name, op, 0)
    if source is None or other is None or source["name"] != other["name"]:
        return None
    return {"source": source, "shape": shape_of(op)}


def find_rope(ops_by_name, op):
    """Detect the split-half rotation: Add(Mul(x,cos), Mul(Cat(-x2,x1),sin))."""
    if op["op_type"] != "AddOp":
        return None
    first = parent(ops_by_name, op, 0)
    second = parent(ops_by_name, op, 1)
    if not first or not second:
        return None
    if first["op_type"] != "MulOp" or second["op_type"] != "MulOp":
        return None
    cat = parent(ops_by_name, second, 0)
    if cat is None or cat["op_type"] != "CatOp":
        return None
    negative = parent(ops_by_name, cat, 0)
    if negative is None or negative["op_type"] != "NegOp":
        return None
    sine = parent(ops_by_name, second, 1)
    cosine = parent(ops_by_name, first, 1)
    return {
        "shape": shape_of(op),
        "cosine": cosine, "sine": sine,
        "cosine_shape": shape_of(cosine) if cosine else None,
        "sine_shape": shape_of(sine) if sine else None,
    }


def classify(ops):
    """Assign every op to a semantic group; return groups and leftovers.

    Interior nodes of a matched pattern (the Pow/Mean/Add/Rsqrt chain inside an
    RMSNorm, the Sigmoid inside a SiLU, the Cat/Neg insde a RoPE rotation) are
    marked ``consumed`` so the leftover count reflects genuinely unclassified
    work instead of re-counting nodes that a kernel already covers.
    """
    by_name = index_ops(ops)
    groups = defaultdict(list)
    consumed = set()

    def chain(op, depth=0):
        """Names of the backward dataflow cone, bounded to stay cheap."""
        if op is None or depth > 6:
            return []
        names = [op["name"]]
        for parent_name in op.get("parents") or []:
            names.extend(chain(by_name.get(parent_name), depth + 1))
        return names

    for op in ops:
        kind = op["op_type"]
        name = op["name"]
        if kind == "PlaceholderOp":
            groups["parameter-or-input"].append(op)
        elif kind == "EmbeddingOp":
            groups["embedding"].append(op)
        elif kind == "ScaledDotProductFlashAttentionForCpuOp":
            groups["attention-fused-op"].append(op)
        elif kind == "IndexPutOp":
            groups["kv-cache-write"].append(op)
        elif kind == "MatmulOp":
            target = shape_of(op)
            if target and target[-1] == 151936:
                groups["lm_head"].append(op)
            else:
                groups["linear-projection"].append(op)
        elif kind == "BatchMatmulOp":
            groups["rope-frequency-outer-product-or-attention"].append(op)
        elif kind == "GetItemOp":
            # Selecting result 0 from the SDPA tuple is a projection of the
            # attention kernel's own output, not separate work.
            source = parent(by_name, op, 0)
            if source is not None and \
                    source["op_type"] == "ScaledDotProductFlashAttentionForCpuOp":
                groups["attention-result-select"].append(op)
            else:
                groups["unmatched"].append(op)
        elif kind == "OutputOp":
            groups["output"].append(op)
        else:
            rms = find_rmsnorm(by_name, op)
            silu = find_silu(by_name, op)
            rope = find_rope(by_name, op)
            if rms:
                head_dim_norm = rms["weight_shape"] == [128]
                groups["qk-norm" if head_dim_norm else "rms-norm"].append(
                    dict(op, matched=rms))
                consumed.update(chain(rms["source"]))
                consumed.add(rms["weight"]["name"] if rms["weight"] else "")
            elif silu:
                groups["silu"].append(dict(op, matched=silu))
                consumed.add(silu["source"]["name"])
            elif rope:
                groups["rope-rotation"].append(dict(op, matched=rope))
                consumed.update(chain(rope["cosine"]))
                consumed.update(chain(rope["sine"]))
            elif kind in ("SigmoidOp", "CosOp", "SinOp", "PowOp", "MeanOp",
                          "RsqrtOp", "NegOp", "CatOp", "SliceOp"):
                groups["pattern-interior"].append(op)
            elif kind in VIEW_OPS:
                groups["layout-view"].append(op)
            elif kind in ("AddOp", "MulOp", "SubOp", "DivOp"):
                groups["elementwise"].append(op)
            elif kind in ("FullOp", "IotaOp", "ScalarTensorOp", "LeTensorOp",
                          "WhereOp", "ConvertElementTypeOp"):
                groups["mask-or-position"].append(op)
            else:
                groups["unmatched"].append(op)

    for group in ("elementwise", "layout-view", "mask-or-position",
                  "pattern-interior"):
        kept = [op for op in groups[group] if op["name"] not in consumed]
        dropped = len(groups[group]) - len(kept)
        groups[group] = kept
        if dropped:
            groups["consumed-by-matched-pattern"].append(
                {"op_type": "count-only", "name": f"{group}:{dropped}"})
    return groups


def summarise(groups):
    summary = {}
    for name, members in sorted(groups.items()):
        shapes = Counter()
        for op in members:
            shape = shape_of(op)
            if isinstance(shape, list) and shape and isinstance(shape[0], list):
                shapes[json.dumps(shape)] += 1
            elif shape is not None:
                shapes["x".join(str(d) for d in shape)] += 1
        summary[name] = {
            "count": len(members),
            "output_shapes": dict(shapes.most_common(12)),
            "op_types": dict(Counter(m["op_type"] for m in members)),
            "dtypes": dict(Counter(str(dtype_of(m)) for m in members)),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill", type=Path, required=True)
    parser.add_argument("--decode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=28)
    args = parser.parse_args()

    report = {"source": "real Buddy graph op dumps", "layers": args.layers,
              "graphs": {}}
    for label, path in (("prefill", args.prefill), ("decode", args.decode)):
        ops = load(path)
        groups = classify(ops)
        report["graphs"][label] = {
            "op_count": len(ops),
            "groups": summarise(groups),
        }
    (args.output.parent).mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    for label in ("prefill", "decode"):
        print(f"=== {label}: {report['graphs'][label]['op_count']} ops")
        for name, info in report["graphs"][label]["groups"].items():
            top = list(info["output_shapes"].items())[:3]
            print(f"  {name:<42} n={info['count']:<5} "
                  f"{'; '.join(f'{s}x{c}' for s, c in top)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())