#!/usr/bin/env python3
"""Build one operator family; gate NR builds on every case's real host check."""

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cases import FAMILIES, ROOT, MODEL_ROOT, generate, inventory, launch_template
from export import compile_case, sha

BUILD = ROOT / "build"
VALIDATION = MODEL_ROOT / "validation/residual-add"
FPGA_ROOT = MODEL_ROOT.parents[1]
TOOLS = FPGA_ROOT / "tools"
LOWER = [
    "--convert-linalg-to-loops", "--expand-strided-metadata", "--lower-affine",
    "--convert-vector-to-scf", "--convert-vector-to-llvm",
    "--convert-math-to-llvm", "--convert-scf-to-cf", "--convert-cf-to-llvm",
    "--convert-arith-to-llvm", "--convert-index-to-llvm", "--convert-ub-to-llvm",
    "--memref-expand", "--finalize-memref-to-llvm", "--convert-func-to-llvm",
    "--reconcile-unrealized-casts",
]


def validation_directory(family):
    return MODEL_ROOT / "validation" / family.replace("_", "-")


def command(args, log=None, input_text=None):
    argv = list(map(str, args))
    result = subprocess.run(argv, input=input_text, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if log is not None:
        Path(log).write_text(shlex.join(argv) + "\n" + result.stdout)
    if result.returncode:
        raise RuntimeError(f"{shlex.join(argv)}\n{result.stdout}")
    return result.stdout


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def configuration():
    text = command(["make", "--no-print-directory", "-s", "-f",
                    MODEL_ROOT / "common.mk", "print-config", f"PYTHON={sys.executable}"])
    return json.loads(text)


def tools_provenance(config):
    result = {}
    for name in ("BUDDY_OPT", "BUDDY_TRANSLATE", "HOST_CC", "LLC", "RISCV_CC",
                 "RISCV_LD", "RISCV_OBJCOPY", "RISCV_OBJDUMP"):
        executable = shutil.which(shlex.split(config[name])[0])
        if executable is None:
            raise RuntimeError(f"Missing tool {name}: {config[name]}")
        path = Path(executable).resolve()
        result[name] = {"path": executable, "resolved_path": str(path), "sha256": sha(path),
                        "version": command([executable, "--version"]).splitlines()[0]}
    return result


def source_fingerprint(case, config, tool_info):
    paths = [ROOT / name for name in ("kernels.py", "cases.py", "export.py", "build.py",
                                    launch_template(case["family"]))]
    if case["family"] in ("hardsigmoid", "hardswish", "se_mul", "mean_hw", "linear", "depthwise_conv2d", "pointwise_conv2d", "conv_stem"):
        paths.append(ROOT / f"verify_{case['family']}.py")
    paths += [MODEL_ROOT / name for name in ("common.mk", "support.c", "support.h")]
    paths += [MODEL_ROOT / case["name"] / name for name in ("launch.c", "metadata.json")]
    paths += sorted((FPGA_ROOT / "common/nr").glob("*.[chS]"))
    paths += [FPGA_ROOT / "common/nr/nr.ld", FPGA_ROOT / "common/nr/nr.mk",
              FPGA_ROOT / "common/toolchain.mk"]
    paths += sorted((FPGA_ROOT / "common/nr").glob("*.inc"))
    paths += sorted((FPGA_ROOT / "common/uart").glob("*.h"))
    paths += [TOOLS / name for name in ("ame_to_word.py", "restrict_fpga_assembly.py", "check_nr_elf.py", "nr_isa.py")]
    sources = {str(p.relative_to(FPGA_ROOT)): sha(p) for p in paths}
    manifest = json.loads((BUILD / case["name"] / "frontend.json").read_text())
    text = json.dumps({"sources": sources, "config": config, "tools": tool_info,
                       "frontend": manifest}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest(), sources


def lower(case, config, target):
    export_dir = BUILD / case["name"]
    out = export_dir / target
    out.mkdir(parents=True, exist_ok=True)
    opt = shlex.split(config["BUDDY_OPT"])
    command([*opt, export_dir / "kernel.linalg.mlir", "--empty-tensor-to-alloc-tensor",
             "--one-shot-bufferize=allow-return-allocs-from-loops=true buffer-alignment=64",
             "--canonicalize", "--cse", "--buffer-loop-hoisting",
             "--promote-buffers-to-stack=max-alloc-size-in-bytes=65536",
             "-o", out / "bufferized.mlir"], log=out / "bufferize.log")
    lowered = out / "bufferized.mlir"
    if case["family"] == "linear":
        bufferized = lowered.read_text()
        # Only compiler-generated, bounded local tiles are permitted. The
        # caller supplies physical [N,K]; never allocate a full weight transpose.
        bm, bn, bk = (case["constexprs"][k] for k in ("BM", "BN", "BK"))
        tile_shapes = {(bm, bk), (bk, bn), (bn,), (bm, bn)}
        allocations = re.findall(r"memref\.alloca\([^\n]*memref<([0-9x]+)xf32>", bufferized)
        if (re.search(r"memref\.alloc\(", bufferized)
                or len(allocations) != bufferized.count("memref.alloca(")
                or any(tuple(map(int, shape.split("x"))) not in tile_shapes
                       for shape in allocations)
                or re.search(r"(?:memref|linalg)\.transpose", bufferized)):
            raise RuntimeError("Linear requires bounded local dot tiles without a weight transpose")
    if case["family"] in ("linear", "pointwise_conv2d", "conv_stem") and target == "nr":
        vectorized = out / "vectorized.mlir"
        pass_name = {"linear": "LINEAR_FP32_PASS", "pointwise_conv2d": "PWCONV_FP32_PASS",
                     "conv_stem": "STEM_FP32_PASS"}[case["family"]]
        command([*opt, lowered, *shlex.split(config[pass_name]),
                 "--canonicalize", "--cse", "-o", vectorized], log=out / "vectorize.log")
        contents = vectorized.read_text()
        if "vector.fma" not in contents or "linalg.matmul" in contents:
            raise RuntimeError("Buddy did not lower dot tiles to vector FMA")
        if case["family"] == "linear" and re.search(r"vector\.(?:multi_)?reduction", contents):
            raise RuntimeError("Linear must accumulate across K in independent output lanes")
        lowered = vectorized
    command([*opt, lowered, *LOWER,
             "-o", out / "kernel.llvm.mlir"], log=out / "lower.log")
    llvm_ir = out / "kernel.ll"
    command([*shlex.split(config["BUDDY_TRANSLATE"]), "--buddy-to-llvmir",
             out / "kernel.llvm.mlir", "-o", llvm_ir], log=out / "translate.log")
    text = llvm_ir.read_text()
    definition = re.search(r"define\s+void\s+@" + case["symbol"] + r"\((.*?)\)", text, re.S)
    expected = ["i64", "ptr"] * len(case["signature"]) + ["i32"] * 6
    if not definition or [p.strip().split()[0] for p in definition[1].split(",")] != expected:
        raise RuntimeError("Unexpected Triton unranked-memref/grid ABI")
    if re.search(r"\b(?:call|declare)\b[^\n]*@(?:malloc|aligned_alloc|free)\b", text):
        raise RuntimeError("Unexpected per-grid heap allocation remains")
    return out


def artifacts(case, out, image):
    paths = [BUILD / case["name"] / name for name in ("kernel.ttir", "kernel.linalg.mlir", "adapter.c", "frontend.json")]
    paths += [out / "kernel.ll", out / "kernel.llvm.mlir", image]
    return {str(p.relative_to(MODEL_ROOT)): sha(p) for p in paths}


def check_host(case, config, fingerprint, sources, force):
    out = BUILD / case["name"] / "host"
    result_file = out / "result.json"
    if not force and result_file.exists():
        previous = json.loads(result_file.read_text())
        if previous.get("fingerprint") == fingerprint and previous.get("status") == "PASS":
            if all((MODEL_ROOT / p).is_file() and sha(MODEL_ROOT / p) == digest
                   for p, digest in previous["artifacts"].items()):
                print("host PASS (verified existing evidence) " + case["name"], flush=True)
                return previous
    out = lower(case, config, "host")
    image = out / "check"
    command([*shlex.split(config["HOST_CC"]), "-O2", "-ffp-contract=off",
             "-fno-vectorize", "-fno-slp-vectorize", "-DHOST_TEST", "-I", MODEL_ROOT,
             out / "kernel.ll", BUILD / case["name"] / "adapter.c",
             MODEL_ROOT / case["name"] / "launch.c", MODEL_ROOT / "support.c",
             "-o", image], log=out / "compile.log")
    result = command([image], log=out / "output.log")
    passed = re.search(r"verify " + re.escape(case["name"]) +
                       r": PASS errors=00000000 max_abs_error_f32_bits=([0-9a-fA-F]{8})", result)
    if not passed or (case["family"] not in ("mean_hw", "linear", "depthwise_conv2d", "pointwise_conv2d", "conv_stem") and passed[1] != "00000000"):
        raise RuntimeError(f"Independent C oracle did not pass: {case['name']}")
    oracle = ("Independent C x>0 ? x : 0, exact finite FP32 equality; two patterns with negatives, "
              "+0.0, -0.0, positives and +/-FLT_MAX; both output zero signs accepted"
              if case["family"] == "relu" else
              "Independent C X[i]+Y[i], exact FP32 equality, two signed nonzero patterns")
    if case["family"] == "hardsigmoid":
        oracle = ("Independent C clamp(X[i]+3,0,6)/6; exact FP32 equality; three patterns "
                  "cover all five regions and adjacent FP32 values at +/-3, signed zeros, "
                  "extremes and infinities; NaNs compared by classification")
    elif case["family"] == "hardswish":
        oracle = ("Independent C X[i]*clamp(X[i]+3,0,6)/6 with FP32 multiplication before division; "
                  "exact FP32 equality; three patterns cover all five regions, adjacent FP32 "
                  "values at +/-3, signed zeros, extremes and overflow; NaNs by classification "
                  "and infinities by signed equality")
    elif case["family"] == "se_mul":
        oracle = ("Independent C nested c/h/w loops: X[(c*H+h)*W+w]*Scale[c]; "
                  "Scale has exactly C elements; exact finite FP32 equality; unique and "
                  "reversed channel scales, signed values/zeros, last-channel-only pattern")
    elif case["family"] == "mean_hw":
        oracle = ("Independent C double accumulation in nested c/h/w loops, divided by H*W; "
                  "per-channel bound gamma_(H*W+1)*mean(abs(X))+2^-149; signed zeros, "
                  "channel constants, first/last impulses, cancellation and non-dyadic inputs")
    elif case["family"] == "linear":
        oracle = ("Independent C FP64 m/n/k loops, reading Weight[n*K+k] and adding Bias[n]; "
                  "gamma_(K+2)*(sum(abs(X*Weight))+abs(Bias))+(K+2)*2^-149; "
                  "bias-only, first/middle/last K basis, signed non-dyadic, cancellation, "
                  "last-output-channel-only patterns")
    elif case["family"] == "depthwise_conv2d":
        oracle = ("Independent C FP64 c/oh/ow/kh/kw loops, NCHW input and OIHW Weight[C,1,KH,KW]; "
                  "explicit zero padding, stride and Bias[c]; per-output forward error bound; "
                  "bias-only, spatial/channel ramps, four corner impulses, last-channel-only, "
                  "asymmetric one-hot weights, cancellation and signed non-dyadic patterns")
    elif case["family"] == "pointwise_conv2d":
        oracle = ("Independent C FP64 oc/h/w/ic Conv2D loops: X[(ic*H+h)*W+w]*Weight[oc*Cin+ic]+Bias[oc]; "
                  "bias-only, spatial/channel ramps, first/last input channel, one-hot channel routing, "
                  "corner impulses, cancellation, signed non-dyadic and last-output-only patterns; "
                  "gamma_(Cin+2) forward error bound")
    elif case["family"] == "conv_stem":
        oracle = ("Independent C FP64 oc/oh/ow/ic/kh/kw loops (batch=1), NCHW input, OIHW weights, "
                  "explicit zero padding and stride=2, fused Bias[oc]; bias-only, channel/spatial ramps, "
                  "all 27 one-hot kernel taps, corner impulses, boundary stripes, constant valid-pixel "
                  "counts, cancellation, signed non-dyadic values and last-output-only patterns")
    record = {"case": case["name"], "status": "PASS", "fingerprint": fingerprint,
              "sources": sources, "artifacts": artifacts(case, out, image),
              "oracle": oracle,
              "bounds": "Host PROT_NONE page immediately after COUNT; prefix canaries; inputs unchanged",
              "output": result}
    record["artifacts"][str((out / "output.log").relative_to(MODEL_ROOT))] = sha(out / "output.log")
    if case["family"] == "se_mul":
        record["bounds"] = ("Host PROT_NONE pages immediately after X[COUNT], Out[COUNT] and "
                            "Scale[C]; prefix canaries; X and Scale unchanged bitwise")
    elif case["family"] in ("mean_hw", "linear", "depthwise_conv2d", "pointwise_conv2d", "conv_stem"):
        metrics = re.search(case["family"] + r" metrics max_abs_error=(\S+) max_error_over_bound=(\S+) max_bound=(\S+)", result)
        if not metrics:
            raise RuntimeError("Missing double-oracle reduction metrics")
        record["metrics"] = dict(zip(("max_abs_error", "max_error_over_bound", "max_bound"),
                                     map(float, metrics.groups()), strict=True))
        if (not all(math.isfinite(v) and v >= 0 for v in record["metrics"].values())
                or record["metrics"]["max_error_over_bound"] > 1):
            raise RuntimeError("Invalid or failing reduction error metrics")
        record["tolerance"] = case["tolerance"]
        record["bounds"] = "Host PROT_NONE pages immediately after X[COUNT] and Out[C]; prefix canaries; input unchanged bitwise"
        if case["family"] == "linear":
            record["bounds"] = ("Host PROT_NONE pages immediately after X[M*K], Weight[N*K], Bias[N], Out[M*N]; "
                                "prefix canaries; X/Weight/Bias unchanged bitwise")
        elif case["family"] == "depthwise_conv2d":
            record["bounds"] = ("Host PROT_NONE pages immediately after X[C*H*W], Weight[C*KH*KW], "
                                "Bias[C], Out[C*OH*OW]; prefix canaries; X/Weight/Bias unchanged bitwise")
        elif case["family"] == "pointwise_conv2d":
            record["bounds"] = ("Host PROT_NONE pages immediately after X[Cin*H*W], Weight[Cout*Cin], "
                                "Bias[Cout], Out[Cout*H*W]; prefix canaries; X/Weight/Bias unchanged bitwise")
        elif case["family"] == "conv_stem":
            record["bounds"] = ("Host PROT_NONE pages immediately after X[Cin*H*W], Weight[Cout*Cin*KH*KW], "
                                "Bias[Cout], Out[Cout*OH*OW]; prefix canaries; X/Weight/Bias unchanged bitwise")
    if case["family"] in ("hardsigmoid", "hardswish", "se_mul", "mean_hw", "linear", "depthwise_conv2d", "pointwise_conv2d", "conv_stem"):
        verifier = importlib.import_module(f"verify_{case['family']}")
        record["pytorch"] = verifier.verify(case, out, config, command)
        for name in ("pytorch-check.so", "pytorch.json"):
            record["artifacts"][str((out / name).relative_to(MODEL_ROOT))] = sha(out / name)
    write_json(result_file, record)
    print(result.strip(), flush=True)
    return record


def build_nr(case, config, host_record):
    previous_file = BUILD / case["name"] / "nr/result.json"
    if previous_file.exists():
        previous = json.loads(previous_file.read_text())
        if (previous.get("build") == "PASS"
                and previous.get("host_fingerprint") == host_record["fingerprint"]
                and Path(previous["elf"]).is_file()
                and sha(previous["elf"]) == previous["elf_sha256"]
                and all((MODEL_ROOT / p).is_file() and sha(MODEL_ROOT / p) == digest
                        for p, digest in previous["artifacts"].items())):
            print("NR build PASS (verified existing artifacts) " + case["name"], flush=True)
            return previous
    out = lower(case, config, "nr")
    vector_host = None
    if case["family"] in ("linear", "pointwise_conv2d", "conv_stem"):
        # Execute the SAME vectorized LLVM subsequently compiled to RISC-V.
        # Scalar Host PASS alone does not validate the NR vectorized dot.
        vector_image = out / "vector-host-check"
        command([*shlex.split(config["HOST_CC"]), "-O2", "-ffp-contract=off",
                 "-fno-vectorize", "-fno-slp-vectorize", "-DHOST_TEST", "-I", MODEL_ROOT,
                 out / "kernel.ll", BUILD / case["name"] / "adapter.c",
                 MODEL_ROOT / case["name"] / "launch.c", MODEL_ROOT / "support.c",
                 "-o", vector_image], log=out / "vector-host-compile.log")
        result = command([vector_image], log=out / "vector-host.log")
        if f"verify {case['name']}: PASS errors=00000000" not in result:
            raise RuntimeError("Independent C oracle rejected NR vectorized " + case["family"])
        vector_host = {"status": "PASS", "output": result,
                       "pytorch": importlib.import_module("verify_" + case["family"]).verify(case, out, config, command)}
        print(result.strip(), flush=True)
    # Scalar FP32 is supported by NR; vector tensor types are scalarized by LLVM.
    llc_flags = ["-O2", "-filetype=asm", "-mtriple=riscv64", "-target-abi=lp64d",
                 "-mattr=+m,+a,+f,+d,+c,-v", "-code-model=medium"]
    if case["family"] == "linear":
        llc_flags = ["-O2", "-filetype=asm", "-mtriple=riscv64", "-target-abi=lp64d",
                     *shlex.split(config["LINEAR_RVV_FLAGS"]), "-code-model=medium"]
    elif case["family"] in ("pointwise_conv2d", "conv_stem"):
        llc_flags = ["-O2", "-filetype=asm", "-mtriple=riscv64", "-target-abi=lp64d",
                     *shlex.split(config["STEM_RVV_FLAGS" if case["family"] == "conv_stem" else "PWCONV_RVV_FLAGS"]), "-code-model=medium"]
    command([*shlex.split(config["LLC"]), out / "kernel.ll", *llc_flags,
             "-o", out / "kernel.s"], log=out / "llc.log")
    if case["family"] in ("linear", "pointwise_conv2d", "conv_stem") and not re.search(r"\bvfmacc\.", (out / "kernel.s").read_text()):
        raise RuntimeError("RISC-V assembly lost RVV FP32 dot accumulation")
    if case["family"] == "linear" and re.search(r"\bvfred[uo]sum\.", (out / "kernel.s").read_text()):
        raise RuntimeError("Linear emitted a horizontal FP reduction unsupported on patch6")
    encoded = command([sys.executable, TOOLS / "ame_to_word.py"],
                       input_text=(out / "kernel.s").read_text())
    constrained = command([sys.executable, TOOLS / "restrict_fpga_assembly.py"], input_text=encoded)
    (out / "kernel.nr.S").write_text(constrained)
    cc = shlex.split(config["RISCV_CC"])
    flags = [*shlex.split(config["NR_CFLAGS"]), "-I", str(MODEL_ROOT)]
    objects = []
    for source, name in ((out / "kernel.nr.S", "kernel"),
                         (BUILD / case["name"] / "adapter.c", "adapter"),
                         (MODEL_ROOT / case["name"] / "launch.c", "launch"),
                         (MODEL_ROOT / "support.c", "support")):
        obj = out / (name + ".o")
        extra = ["-march=rv64gcv_zicbom_zvl512b"] if case["family"] in ("linear", "pointwise_conv2d", "conv_stem") and name == "kernel" else []
        command([*cc, *flags, *extra, "-c", source, "-o", obj], log=out / (name + "-compile.log"))
        objects.append(obj)
    elf = out / (case["name"] + ".elf")
    image = out / (case["name"] + ".bin")
    command([*shlex.split(config["RISCV_LD"]), "--gc-sections", "-T", config["NR_LINKER_SCRIPT"],
             "-Map=" + str(out / "kernel.map"), "-o", elf,
             *objects, *shlex.split(config["NR_OBJECTS"])], log=out / "link.log")
    command([sys.executable, TOOLS / "check_nr_elf.py", elf,
             "--objdump", config["RISCV_OBJDUMP"], "--output", out / "elf-audit.json"],
            log=out / "audit.log")
    command([*shlex.split(config["RISCV_OBJCOPY"]), "-O", "binary", elf, image])
    record = {"case": case["name"], "build": "PASS", "fpga": "NOT EXECUTED",
              "host_fingerprint": host_record["fingerprint"], "llc_flags": llc_flags,
              "artifacts": artifacts(case, out, image),
              "elf": str(elf), "bin": str(image), "elf_sha256": sha(elf),
              "audit": json.loads((out / "elf-audit.json").read_text())}
    if vector_host is not None:
        record["vector_host"] = vector_host
        extra_artifacts = ["vectorized.mlir", "vector-host-check", "vector-host.log",
                           "pytorch-check.so", "pytorch.json", "kernel.nr.S", "elf-audit.json"]
        if case["family"] == "linear":
            extra_artifacts.insert(0, "bufferized.mlir")
        for name in extra_artifacts:
            record["artifacts"][str((out / name).relative_to(MODEL_ROOT))] = sha(out / name)
    write_json(out / "result.json", record)
    print("NR build PASS " + case["name"], flush=True)
    return record


def run_board(case, nr_record):
    validation = validation_directory(case["family"])
    argv = [str(FPGA_ROOT / "fpga_run.sh"), nr_record["bin"], "--fpga=5",
            "--capture-seconds=120", "--completion-marker=[nr] RA returned:"]
    print(shlex.join(argv), flush=True)
    log = validation / (case["name"] + ".fpga.log")
    with log.open("w") as stream:
        process = subprocess.run(argv, text=True, stdout=stream, stderr=subprocess.STDOUT)
    output = log.read_text()
    passed = (process.returncode == 0 and f"verify {case['name']}: PASS errors=00000000" in output
              and "[nr] RA returned: PASS" in output)
    status = "PASS" if passed else ("FAIL" if "[nr]" in output else "NOT EXECUTED")
    record = {"case": case["name"], "status": status, "command": argv,
              "returncode": process.returncode, "log": str(log), "log_sha256": sha(log),
              "bin_sha256": sha(nr_record["bin"])}
    write_json(validation / (case["name"] + ".fpga.json"), record)
    print(output, flush=True)
    return record


def run_all_boards(cases, nr_results):
    validation = validation_directory(cases[0]["family"])
    results = [{"case": c["name"], "status": "NOT EXECUTED"} for c in cases]
    summary = {"status": "INCOMPLETE", "count": len(cases), "results": results}
    write_json(validation / "fpga.json", summary)
    for index, (case, nr_record) in enumerate(zip(cases, nr_results, strict=True)):
        results[index] = run_board(case, nr_record)
        write_json(validation / "fpga.json", summary)
        if results[index]["status"] != "PASS":
            raise RuntimeError(f"FPGA {results[index]['status']}: {case['name']}; "
                               f"remaining cases NOT EXECUTED; see {validation}")
    summary["status"] = "PASS"
    write_json(validation / "fpga.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", action="store_true")
    parser.add_argument("--nr", action="store_true")
    parser.add_argument("--run", action="store_true", help="Build and run the selected family with the existing fpga_run.sh")
    parser.add_argument("--family", choices=FAMILIES, default="residual_add")
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    if args.print_config:
        print(json.dumps({k.removeprefix("MOBILENET_CFG_"): v for k, v in os.environ.items()
                          if k.startswith("MOBILENET_CFG_")}))
        return
    if not (args.host or args.nr or args.run):
        parser.error("select --host, --nr or --run")
    cases = inventory(args.family)
    generate(cases)
    config = configuration()
    tool_info = tools_provenance(config)
    validation = validation_directory(args.family)
    validation.mkdir(parents=True, exist_ok=True)
    write_json(validation / "toolchain.json", {"configuration": config, "tools": tool_info})
    host_results = []
    for case in cases:
        compile_case(case, BUILD)
        fingerprint, sources = source_fingerprint(case, config, tool_info)
        host_results.append(check_host(case, config, fingerprint, sources, args.host))
    write_json(validation / "host.json", {"status": "PASS", "count": len(cases), "results": host_results})
    print(f"All {len(cases)} {args.family} host cases PASS", flush=True)
    if args.nr or args.run:
        # Only after every host oracle has passed, build NR runtime and binaries.
        command(["make", "--no-print-directory", "-s", "-B", "-f",
                 MODEL_ROOT / "common.mk", "runtime"], log=BUILD / "runtime-build.log")
        nr_results = [build_nr(c, config, h) for c, h in zip(cases, host_results, strict=True)]
        write_json(validation / "nr.json", {"build": "PASS", "count": len(cases), "results": nr_results})
        if args.run:
            run_all_boards(cases, nr_results)


if __name__ == "__main__":
    main()
