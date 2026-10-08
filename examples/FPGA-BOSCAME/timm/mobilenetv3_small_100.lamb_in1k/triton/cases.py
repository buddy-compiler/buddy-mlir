"""Single inventory for MobileNetV3 Triton static specializations."""

import argparse
import json
from math import prod
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT.parent
BLOCK = 128
SHAPES = ((1, 24, 28, 28), (1, 40, 14, 14),
          (1, 48, 14, 14), (1, 96, 7, 7))
RELU_SHAPES = (
    (1, 16, 56, 56), (1, 8, 1, 1), (1, 72, 56, 56),
    (1, 72, 28, 28), (1, 88, 28, 28), (1, 24, 1, 1),
    (1, 64, 1, 1), (1, 32, 1, 1), (1, 40, 1, 1),
    (1, 72, 1, 1), (1, 144, 1, 1),
)
HARDSIGMOID_SHAPES = tuple((1, count, 1, 1) for count in (16, 96, 240, 120, 144, 288, 576))
HARDSWISH_SHAPES = (
    (1, 16, 112, 112), (1, 96, 28, 28), (1, 96, 14, 14),
    (1, 240, 14, 14), (1, 120, 14, 14), (1, 144, 14, 14),
    (1, 576, 7, 7), (1, 288, 14, 14), (1, 288, 7, 7), (1, 1024, 1, 1),
)
SE_MUL_SHAPES = (
    (1, 16, 56, 56), (1, 96, 14, 14), (1, 240, 14, 14),
    (1, 120, 14, 14), (1, 144, 14, 14), (1, 288, 7, 7), (1, 576, 7, 7),
)
MEAN_HW_SHAPES = SE_MUL_SHAPES
FAMILY_SHAPES = {"residual_add": SHAPES, "relu": RELU_SHAPES,
                 "hardsigmoid": HARDSIGMOID_SHAPES, "hardswish": HARDSWISH_SHAPES,
                 "se_mul": SE_MUL_SHAPES, "mean_hw": MEAN_HW_SHAPES}
DWCONV_CONFIGS = (
    (16, 112, 3, 2, 1), (72, 56, 3, 2, 1), (88, 28, 3, 1, 1),
    (96, 28, 5, 2, 2), (240, 14, 5, 1, 2), (120, 14, 5, 1, 2),
    (144, 14, 5, 1, 2), (288, 14, 5, 2, 2), (576, 7, 5, 1, 2),
)
PWCONV_CONFIGS = (
    (16, 8, 1, 1), (8, 16, 1, 1), (16, 16, 56, 56), (16, 72, 56, 56),
    (72, 24, 28, 28), (24, 88, 28, 28), (88, 24, 28, 28), (24, 96, 28, 28),
    (96, 24, 1, 1), (24, 96, 1, 1), (96, 40, 14, 14), (40, 240, 14, 14),
    (240, 64, 1, 1), (64, 240, 1, 1), (240, 40, 14, 14), (40, 120, 14, 14),
    (120, 32, 1, 1), (32, 120, 1, 1), (120, 48, 14, 14), (48, 144, 14, 14),
    (144, 40, 1, 1), (40, 144, 1, 1), (144, 48, 14, 14), (48, 288, 14, 14),
    (288, 72, 1, 1), (72, 288, 1, 1), (288, 96, 7, 7), (96, 576, 7, 7),
    (576, 144, 1, 1), (144, 576, 1, 1), (576, 96, 7, 7), (576, 1024, 1, 1),
)
FAMILIES = (*FAMILY_SHAPES, "linear", "depthwise_conv2d", "pointwise_conv2d", "conv_stem")


def launch_template(family):
    if family not in FAMILIES:
        raise ValueError(f"Unknown family: {family}")
    return "launch.c.in" if family == "residual_add" else f"launch_{family}.c.in"


def inventory(family="residual_add"):
    if family not in FAMILIES:
        raise ValueError(f"Unknown family: {family}")
    if family == "conv_stem":
        name = "conv_stem_cin3_cout16_h224_k3_s2_p1"
        m, n, k = 112 * 112, 16, 3 * 3 * 3
        bm, bn, bk = 16, 16, 32
        steps, u = k + 2, 2.0**-24
        return [{
            "name": name, "family": family, "kernel": family, "symbol": name,
            "dtype": "float32", "layout": "NCHW contiguous; OIHW Weight[Cout,Cin,KH,KW]",
            "shapes": [[1, 3, 224, 224]], "weight_shape": [16, 3, 3, 3],
            "bias_shape": [16], "output_shape": [1, 16, 112, 112],
            "kernel_size": [3, 3], "stride": [2, 2], "padding": [1, 1],
            "dilation": [1, 1], "groups": 1,
            "mnk": {"M": m, "N": n, "K": k},
            "logical_addressing": {
                "k": "(ic*KH+kh)*KW+kw", "m": "oh*OW+ow",
                "A[m,k]": "X[(ic*H+oh*SH+kh-PH)*W+ow*SW+kw-PW], or zero outside H/W",
                "B[k,n]": "Weight[n*(Cin*KH*KW)+k]", "Out[m,n]": "Out[n*(OH*OW)+m]"},
            "constexprs": {"CIN": 3, "COUT": 16, "H": 224, "W": 224,
                           "KH": 3, "KW": 3, "SH": 2, "SW": 2, "PH": 1, "PW": 1,
                           "OH": 112, "OW": 112, "BM": bm, "BN": bn, "BK": bk},
            "signature": {"X": "*fp32", "Weight": "*fp32", "Bias": "*fp32", "Out": "*fp32"},
            "grid": [(m + bm - 1) // bm, (n + bn - 1) // bn, 1],
            "grid_axes": ["output_spatial_tile", "output_channel_tile", "unused"],
            "tail": {"M": m % bm, "N": n % bn, "K": k % bk},
            "tolerance": {"formula": "gamma_(K+2)*(sum_valid(abs(X*Weight))+abs(Bias[oc])) + (K+2)*2^-149",
                          "unit_roundoff": u, "gamma": steps * u / (1 - steps * u),
                          "absolute_floor": steps * 2.0**-149},
        }]
    if family == "pointwise_conv2d":
        cases = []
        for cin, cout, h, w in PWCONV_CONFIGS:
            m, n, k = h * w, cout, cin
            bm, bn, bk = min(16, 1 << (m - 1).bit_length()), 16, 32
            name = f"pwconv_cin{cin}_cout{cout}_h{h}_w{w}"
            steps, u = k + 2, 2.0**-24
            cases.append({
                "name": name, "family": family, "kernel": family, "symbol": name,
                "dtype": "float32", "layout": "NCHW contiguous; OIHW Weight[Cout,Cin,1,1]",
                "shapes": [[1, cin, h, w]], "weight_shape": [cout, cin, 1, 1],
                "bias_shape": [cout], "output_shape": [1, cout, h, w],
                "kernel_size": [1, 1], "stride": [1, 1], "padding": [0, 0],
                "dilation": [1, 1], "groups": 1,
                "mnk": {"M": m, "N": n, "K": k},
                "logical_addressing": {"A[m,k]": "X[k*(H*W)+m]",
                                       "B[k,n]": "Weight[n*Cin+k]",
                                       "Out[m,n]": "Out[n*(H*W)+m]"},
                "constexprs": {"CIN": cin, "COUT": cout, "H": h, "W": w,
                               "BM": bm, "BN": bn, "BK": bk},
                "signature": {"X": "*fp32", "Weight": "*fp32", "Bias": "*fp32", "Out": "*fp32"},
                "grid": [(m + bm - 1) // bm, (n + bn - 1) // bn, 1],
                "grid_axes": ["spatial_tile", "output_channel_tile", "unused"],
                "tail": {"M": m % bm, "N": n % bn, "K": k % bk},
                "tolerance": {"formula": "gamma_(Cin+2)*(sum_ic(abs(X[ic,h,w]*Weight[oc,ic]))+abs(Bias[oc])) + (Cin+2)*2^-149",
                              "unit_roundoff": u, "gamma": steps * u / (1 - steps * u),
                              "absolute_floor": steps * 2.0**-149},
            })
        return cases
    if family == "depthwise_conv2d":
        cases = []
        for c, h, k, stride, pad in DWCONV_CONFIGS:
            output = (h + 2 * pad - k) // stride + 1
            name = f"dwconv_c{c}_h{h}_k{k}_s{stride}_p{pad}"
            steps, u = k * k + 2, 2.0**-24
            cases.append({
                "name": name, "family": family, "kernel": family, "symbol": name,
                "dtype": "float32", "layout": "NCHW contiguous; OIHW Weight[C,1,KH,KW]",
                "shapes": [[1, c, h, h]], "weight_shape": [c, 1, k, k],
                "bias_shape": [c], "output_shape": [1, c, output, output],
                "groups": c, "dilation": [1, 1], "stride": [stride, stride], "padding": [pad, pad],
                "constexprs": {"C": c, "H": h, "W": h, "KH": k, "KW": k,
                               "SH": stride, "SW": stride, "PH": pad, "PW": pad,
                               "OH": output, "OW": output, "BLOCK": BLOCK},
                "signature": {"X": "*fp32", "Weight": "*fp32", "Bias": "*fp32", "Out": "*fp32"},
                "grid": [(output * output + BLOCK - 1) // BLOCK, c, 1],
                "grid_axes": ["output_spatial_tile", "channel", "unused"],
                "tail": output * output % BLOCK,
                "tolerance": {"formula": "gamma_(KH*KW+2)*(sum_valid(abs(X*Weight))+abs(Bias[c])) + (KH*KW+2)*2^-149",
                              "unit_roundoff": u, "gamma": steps * u / (1 - steps * u),
                              "absolute_floor": steps * 2.0**-149},
            })
        return cases
    if family == "linear":
        m, n, k = 1, 1000, 1024
        # Same supported FP32 dot blocking as Qwen: BM<=16, BN=16,
        # BK=lowest set bit of K (one full reduction for K=1024).
        bm, bn, bk = 1 << (min(m, 16) - 1).bit_length(), 16, k & -k
        name = f"linear_m{m}_n{n}_k{k}"
        u = 2.0**-24
        steps = k + 2  # product, K-1 additions, bias, reference-rounding margin
        return [{
            "name": name, "family": family, "kernel": "linear", "symbol": name,
            "dtype": "float32", "layout": "row-major Input[M,K], Weight[N,K], Output[M,N]",
            "shapes": [[m, k]], "weight_shape": [n, k], "bias_shape": [n],
            "output_shape": [m, n],
            "constexprs": {"M": m, "N": n, "K": k, "BM": bm, "BN": bn, "BK": bk},
            "signature": {"X": "*fp32", "Weight": "*fp32", "Bias": "*fp32", "Out": "*fp32"},
            "grid": [(m + bm - 1) // bm, (n + bn - 1) // bn, 1],
            "grid_axes": ["row_tile", "output_channel_tile", "unused"],
            "tail": {"M": m % bm, "N": n % bn, "K": k % bk},
            "tail_strategy": "overlap final BN weight reads; disjoint stores before/after N-BN",
            "tolerance": {"formula": "gamma_(K+2)*(sum_k(abs(X[m,k]*Weight[n,k]))+abs(Bias[n])) + (K+2)*2^-149",
                          "unit_roundoff": u, "gamma": steps * u / (1 - steps * u),
                          "absolute_floor": steps * 2.0**-149},
        }]
    cases = {}
    prefix = "add" if family == "residual_add" else family
    signature = {"X": "*fp32", "Out": "*fp32"} if family != "residual_add" else {
        "X": "*fp32", "Y": "*fp32", "Out": "*fp32"}
    if family == "se_mul":
        signature = {"X": "*fp32", "Scale": "*fp32", "Out": "*fp32"}
    for shape in FAMILY_SHAPES[family]:
        count = prod(shape)
        key = shape if family in ("se_mul", "mean_hw") else count
        if key not in cases:
            cases[key] = {
                "name": f"{prefix}_count{count}", "family": family,
                "kernel": family, "symbol": f"{family}_count{count}",
                "dtype": "float32", "layout": "NCHW contiguous", "shapes": [],
                "constexprs": {"COUNT": count, "BLOCK": BLOCK},
                "signature": dict(signature),
                "grid": [(count + BLOCK - 1) // BLOCK, 1, 1],
                "tail": count % BLOCK,
            }
            if family == "se_mul":
                _, channels, height, width = shape
                spatial = height * width
                name = f"se_mul_c{channels}_h{height}_w{width}"
                cases[key].update(
                    name=name, symbol=name, scale_shape=[1, channels, 1, 1],
                    constexprs={"COUNT": count, "SPATIAL": spatial, "BLOCK": BLOCK},
                    grid=[(spatial + BLOCK - 1) // BLOCK, channels, 1],
                    tail=spatial % BLOCK,
                    grid_axes=["spatial_tile", "channel", "unused"],
                )
            elif family == "mean_hw":
                _, channels, height, width = shape
                spatial = height * width
                block = 1 << (spatial - 1).bit_length()
                name = f"mean_hw_c{channels}_h{height}_w{width}"
                u = 2.0**-24
                cases[key].update(
                    name=name, symbol=name, output_shape=[1, channels, 1, 1],
                    constexprs={"COUNT": count, "SPATIAL": spatial, "BLOCK": block},
                    grid=[channels, 1, 1], tail=spatial % block,
                    grid_axes=["channel", "unused", "unused"],
                    reduction_axes=[2, 3], keepdim=True,
                    tolerance={"formula": "gamma_(SPATIAL+1)*mean(abs(X[channel])) + 2^-149",
                               "unit_roundoff": u,
                               "gamma": (spatial + 1) * u / (1 - (spatial + 1) * u),
                               "absolute_floor": 2.0**-149},
                )
        cases[key]["shapes"].append(list(shape))
    return list(cases.values())


def select(names=None, family="residual_add"):
    cases = inventory(family)
    unknown = set(names or ()) - {c["name"] for c in cases}
    if unknown:
        raise ValueError(f"Unknown cases: {sorted(unknown)}")
    return [c for c in cases if not names or c["name"] in names]


def generate(cases=None):
    for case in cases or inventory():
        template = (ROOT / launch_template(case["family"])).read_text()
        directory = MODEL_ROOT / case["name"]
        directory.mkdir(parents=True, exist_ok=True)
        launch = template.replace("@CASE@", case["name"])
        for token, value in case["constexprs"].items():
            launch = launch.replace(f"@{token}@", str(value))
        if case["family"] in ("se_mul", "mean_hw"):
            _, channels, height, width = case["shapes"][0]
            for token, value in (("C", channels), ("H", height), ("W", width)):
                launch = launch.replace(f"@{token}@", str(value))
        (directory / "launch.c").write_text(launch)
        (directory / "metadata.json").write_text(json.dumps(case, indent=2) + "\n")
        family_flag = "" if case["family"] == "residual_add" else f" --family={case['family']}"
        (directory / "makefile").write_text(
            "# Generated by ../triton/cases.py\n"
            "TRITON_PYTHON ?= python3\n"
            ".PHONY: all check run\n"
            f"all:\n\t$(TRITON_PYTHON) ../triton/build.py{family_flag} --nr\n"
            f"check:\n\t$(TRITON_PYTHON) ../triton/build.py{family_flag} --host\n"
            "run: all\n"
            f"\t../../../fpga_run.sh ../triton/build/{case['name']}/nr/{case['name']}.bin "
            "--fpga=5 --capture-seconds=120 --completion-marker='[nr] RA returned:'\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--family", choices=FAMILIES, default="residual_add")
    args = parser.parse_args()
    if args.generate:
        generate(inventory(args.family))
    print(json.dumps(inventory(args.family), indent=2))
