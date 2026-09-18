"""Regressions for graph matching, result lifetimes, and numerical evidence."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

MODEL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL / "tools"))
import triton_call_replace as tcr
from run_graph_host import compare_reference, error_stats
from quant_model_reference import forward
from reference_fp32 import blocked_dot
from audit_attention_native_ir import audit as audit_native_key


def node(kind, name, shape, args=(), dtype="TensorDType.Float32", **kwargs):
    value = type(kind, (), {})()
    value.name, value.args, value.kwargs = name, list(args), kwargs
    value.tensor_meta = {"shape": list(shape), "dtype": dtype}
    value._parents, value._children = [], []
    return value


def graph(*nodes):
    table = {value.name: value for value in nodes}
    for value in nodes:
        value._parents = [arg for arg in value.args if isinstance(arg, str) and arg in table]
        for parent in value._parents:
            table[parent]._children.append(value.name)
    return SimpleNamespace(node_table=table, body=list(nodes), _func_name="forward_prefill")


class MatchContracts(unittest.TestCase):
    def test_quantization_shares_only_readonly_views_of_the_same_rmsnorm(self):
        x = node("PlaceholderOp", "x", [1, 2, 4])
        w = node("PlaceholderOp", "w", [4])
        power = node("PowOp", "power", [1, 2, 4], ["x", 2])
        mean = node("MeanOp", "mean", [1, 2, 1], ["power", [-1], True])
        add = node("AddOp", "eps", [1, 2, 1], ["mean", 1e-6])
        rsqrt = node("RsqrtOp", "rsqrt", [1, 2, 1], ["eps"])
        scaled = node("MulOp", "scaled", [1, 2, 4], ["x", "rsqrt"])
        norm = node("MulOp", "norm", [1, 2, 4], ["w", "scaled"])
        a = node("ViewOp", "a", [2, 4], ["norm", [2, 4]])
        b = node("ViewOp", "b", [2, 4], ["norm", [2, 4]])
        qa = node("MatmulOp", "qa", [2, 4], ["a", "weight"])
        qb = node("MatmulOp", "qb", [2, 4], ["b", "weight"])
        g = graph(x, w, power, mean, add, rsqrt, scaled, norm, a, b, qa, qb)
        index = {"rmsnorm_2x4": {"name": "rmsnorm_2x4"}}
        key = tcr.activation_quantization_key(g, a, index)
        self.assertIsNotNone(key)
        self.assertEqual(key, tcr.activation_quantization_key(g, b, index))
        b.tensor_meta["shape"] = [4, 2]
        self.assertIsNone(tcr.activation_quantization_key(g, a, index))
        b.tensor_meta["shape"] = [2, 4]
        for kind in ("SliceOp", "PermuteOp", "ExpandOp", "CallOp"):
            b.__class__ = type(kind, (), {})
            self.assertIsNone(tcr.activation_quantization_key(g, a, index))
        b.__class__ = type("ViewOp", (), {})
        qb.__class__ = type("CallOp", (), {})
        self.assertIsNone(tcr.activation_quantization_key(g, a, index))
        qb.__class__ = type("MatmulOp", (), {})
        power.args[1] = 3
        self.assertIsNone(tcr.activation_quantization_key(g, a, index))

    def test_actual_bufferized_key_stride_is_required(self):
        # A tensor has the expected logical shape even when its lowered memref
        # describes a transpose. Validate the actual ABI, not merely the shape.
        prefix = "qwen_graph_attn_qk_attention_qk_position_native_16x1x512x128"
        arguments = ["ptr null", "ptr null", "i64 0"] + ["i64 1"] * 8
        key = ["ptr null", "ptr null", "i64 0", "i64 1", "i64 16",
               "i64 512", "i64 128", "i64 1048576", "i64 65536", "i64 128", "i64 1"]
        arguments += key + ["ptr null", "ptr null", "i64 0", "i64 1", "i64 1"]
        arguments += ["ptr null", "ptr null", "i64 0"] + ["i64 1"] * 8
        ir = "define void @forward_decode() {\n  call void @" + prefix + "(" + ", ".join(arguments) + ")\n  ret void\n}\n"
        self.assertEqual(audit_native_key(ir, 1)[0]["key_strides"], [1048576, 65536, 128, 1])
        # Same shape, incompatible physical storage must not link to this QK.
        with self.assertRaises(ValueError):
            audit_native_key(ir.replace("i64 65536, i64 128, i64 1", "i64 65536, i64 1, i64 512"), 1)
        with self.assertRaises(ValueError):
            audit_native_key(ir, 2)

    def test_dynamic_attention_requires_full_manifest_contract(self):
        case = {"kernel_module": "kernels_position",
                "arguments": [{"name": n, "rank": r, "dtype": d} for n, r, d in
                              [("A", 3, "f32"), ("B", 3, "f32"),
                               ("Position", 1, "i32"), ("C", 3, "f32")]],
                "constexprs": dict(M=16, N=512, K=128, BM=16, BN=16, BK=64, QK=True),
                "grid": [1, 32, 16]}
        self.assertIsNone(tcr.attention_position_case_error(case, 16, 16, 512, 128, True))
        for field, invalid in (("kernel_module", "kernels"), ("grid", [1, 1, 16]),
                               ("constexprs", dict(case["constexprs"], N=16)),
                               ("arguments", case["arguments"][:2]+case["arguments"][3:])):
            altered = dict(case, **{field: invalid})
            self.assertIsNotNone(tcr.attention_position_case_error(altered, 16, 16, 512, 128, True))
        case["arguments"][2]["dtype"] = "i64"
        self.assertIsNotNone(tcr.attention_position_case_error(case, 16, 16, 512, 128, True))

    def test_native_key_contract_cannot_accept_transposed_kernel(self):
        case = {"kernel_module": "kernels_position_native",
                "arguments": [{"name": n, "rank": r, "dtype": d} for n, r, d in
                              [("A", 3, "f32"), ("B", 3, "f32"),
                               ("Position", 1, "i32"), ("C", 3, "f32")]],
                "constexprs": dict(M=1, N=512, K=128, BM=1, BN=16, BK=64, QK=True),
                "grid": [1, 32, 16]}
        self.assertIsNone(tcr.attention_position_case_error(case, 16, 1, 512, 128, True, True))
        self.assertIsNotNone(tcr.attention_position_case_error(case, 16, 1, 512, 128, True, False))
        case["kernel_module"] = "kernels_position"
        self.assertIsNotNone(tcr.attention_position_case_error(case, 16, 1, 512, 128, True, True))

    def test_square_linear_rejects_identity_permutation(self):
        a = node("PlaceholderOp", "a", [2, 4])
        w = node("PlaceholderOp", "w", [4, 4])
        transpose = node("PermuteOp", "transpose", [4, 4], ["w", [1, 0]])
        out = node("MatmulOp", "out", [2, 4], ["a", "transpose"])
        g = graph(a, w, transpose, out)
        index = {"matmul_2x4x4_f32": {"name": "matmul_2x4x4_f32"}}
        self.assertNotIn("reason", tcr.match_linear(g, out, index))
        transpose.args[1] = [0, 1]
        self.assertIn("reason", tcr.match_linear(g, out, index))

    def test_rmsnorm_rejects_different_math(self):
        x = node("PlaceholderOp", "x", [1, 2, 4])
        w = node("PlaceholderOp", "w", [4])
        power = node("PowOp", "power", [1, 2, 4], ["x", 2])
        mean = node("MeanOp", "mean", [1, 2, 1], ["power", [-1], True])
        add = node("AddOp", "eps", [1, 2, 1], ["mean", 1e-6])
        rsqrt = node("RsqrtOp", "rsqrt", [1, 2, 1], ["eps"])
        scaled = node("MulOp", "scaled", [1, 2, 4], ["x", "rsqrt"])
        out = node("MulOp", "out", [1, 2, 4], ["w", "scaled"])
        g = graph(x, w, power, mean, add, rsqrt, scaled, out)
        index = {"rmsnorm_2x4": {"name": "rmsnorm_2x4"}}
        self.assertNotIn("reason", tcr.match_rmsnorm(g, out, index))
        for target, pos, invalid in ((power, 1, 3), (mean, 1, [1]),
                                     (mean, 2, False), (add, 1, 1e-5)):
            original = target.args[pos]
            target.args[pos] = invalid
            self.assertIn("reason", tcr.match_rmsnorm(g, out, index))
            target.args[pos] = original

    def test_returned_outputs_do_not_alias(self):
        """Compile generated adapters and keep two equal-shaped results live."""
        kernel = {"adapter_entry": "kernel", "arguments": [{"rank": 1}, {"rank": 1}]}
        types = {}
        g = SimpleNamespace(_func_name="prefill")
        for name in ("first", "second"):
            symbol = tcr.returning_symbol(g, SimpleNamespace(name=name), "shape")
            types[symbol] = {"case": kernel, "kind": "silu", "shape": [1],
                             "graph_ranks": [1], "result_rank": 1,
                             "result_shape": [1], "entry_args": ["a1", "a0"]}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "adapters.c"
            tcr.generate_adapters(types, source)
            with source.open("a") as handle:
                handle.write('''
void kernel(MemRef1 *in, MemRef1 *out) {
  ((float *)out->aligned)[0] = ((float *)in->aligned)[0];
}
int main(void) {
  float x = 2, y = 9;
  MemRef1 a = make_1(&x, 1), b = make_1(&y, 1), first, second;
  _mlir_ciface_shape__prefill_first(&first, &a);
  _mlir_ciface_shape__prefill_second(&second, &b);
  return first.aligned == second.aligned ||
    ((float *)first.aligned)[0] != 2 || ((float *)second.aligned)[0] != 9;
}
''')
            executable = directory / "check"
            subprocess.run(["cc", "-std=c11", "-DHOST_TEST", "-I", str(MODEL.parent),
                            str(source), "-o", str(executable)], check=True)
            subprocess.run([str(executable)], check=True)


class EvidenceContracts(unittest.TestCase):
    def test_nr_profile_models_single_rounding_fma(self):
        a = np.float32(1 + 2 ** -23)
        b = np.float32(1 - 2 ** -23)
        left = np.array([[[1, a]]], np.float32)
        right = np.array([[[-1], [b]]], np.float32)
        self.assertEqual(float(blocked_dot(left, right)[0, 0, 0]), 0)
        self.assertEqual(float(blocked_dot(left, right, profile="nr-fpga")[0, 0, 0]), -(2 ** -46))

    def test_shape_and_nonfinite_errors_are_not_success(self):
        self.assertIn("shape_mismatch", error_stats(np.ones((2, 3)), np.ones((3, 2))))
        self.assertIn("nonfinite", error_stats([float("nan")], [1]))

    def test_matching_later_tokens_does_not_hide_prefill_divergence(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            np.savez(directory / "arrays.npz", prefill_logits=[[[2., 0.]]], decode_logits_0=[1., 0.])
            (directory / "reference.json").write_text(json.dumps({
                "generated_ids": [0], "decode_steps_recorded": [
                    {"input_token": 0, "cache_position": 1}]}))
            report = {"prefill": {"argmax_last": 1}, "generated_ids": [0],
                      "decode_steps_recorded": [{"input_token": 1, "cache_position": 1}]}
            result = compare_reference({"prefill_logits": np.array([[[0., 2.]]]),
                                        "decode_logits_0": np.array([1., 0.])}, report, directory)
            self.assertFalse(result["token_trajectory_match"])
            self.assertEqual(result["decode_same_input_context"], [False])

    def test_layer_capture_keeps_prefill_and_decode_shapes_separate(self):
        class Weights:
            scheme = {"enabled": False}
            fp32 = {key: np.ones(width, np.float32) for key, width in (
                ("model.layers.0.input_layernorm.weight", 4),
                ("model.layers.0.post_attention_layernorm.weight", 4),
                ("model.layers.0.self_attn.q_norm.weight", 128),
                ("model.layers.0.self_attn.k_norm.weight", 128),
                ("model.norm.weight", 4))}
            def embed(self, ids):
                return np.ones((len(ids), 4), np.float32)
            def linear(self, value, name):
                width = 128 if name.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight")) else 4
                return np.zeros((len(value), width), np.float32)
        config = {"rms_norm_eps": 1e-6, "rope_theta": 1e6, "num_hidden_layers": 1,
                  "num_key_value_heads": 1, "num_attention_heads": 1, "head_dim": 128}
        _, arrays = forward([0, 1, 2], Weights(), config, 2, 8, 2, {0})
        self.assertEqual(arrays["prefill_layer_0_hidden"].shape, (3, 4))
        self.assertEqual(arrays["decode_layer_0_hidden"].shape, (2, 1, 4))
        self.assertEqual(arrays["kv_key_used"].shape, (1, 5, 1, 128))


if __name__ == "__main__":
    unittest.main()
