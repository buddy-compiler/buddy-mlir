#!/usr/bin/env python3
"""Build a bare-metal NR image that runs the compiled model graph on the board.

Everything here exists to make one honest claim testable: that the *compiled
Qwen3 graph*, with its operators replaced by calls into the Triton static
library, executes on the board and produces the token the FP32 reference
produced.

What the image is made of:

  * ``forward_prefill`` lowered from the Buddy graph to a RISC-V object with the
    external calls left unresolved (see tools/lower_model_nr.py);
  * the generated ABI wrappers and the Triton archive that satisfy them;
  * the shared NR runtime (crt, runtime, math, copy, AME sync);
  * a small ``main`` that fills the fixed 16-token prompt, calls the graph,
    takes the argmax of the logits and prints it over UART.

The parameters are **not** in the image. They live in a NOLOAD ``.workspace``
array whose address the loader fills from a separate DDR segment (see
``validation/board/ddr-load-path.json``); the image only carries the code and a
descriptor pointing at that address. That keeps the uploaded image small and is
also the only way the 653 MiB of FP32 parameters can reach the board through the
multi-segment loader.

The graph's own ABI is taken from the lowered module rather than guessed: the
subgraph entry is ``_mlir_ciface_forward_prefill(<results struct>*, <18
descriptors>)`` where the 18 are the 14 parameters in the compiled layout order
followed by ids, cache position, K cache and V cache.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

# The compiled parameter order for one layer, which is exactly the order
# validation/weight-layout.json resolved (verified: run_graph_host.py reports
# param_order_matches_layout).
PARAM_ORDER = [
    "model.embed_tokens.weight", "model.layers.0.input_layernorm.weight",
    "model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.q_norm.weight",
    "model.layers.0.self_attn.k_proj.weight", "model.layers.0.self_attn.k_norm.weight",
    "model.layers.0.self_attn.v_proj.weight", "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.post_attention_layernorm.weight",
    "model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight", "model.norm.weight", "_rotary_inv_freq",
]


DECODE_BODY = []


def decode_body(prompt_len, cache_len, head_dim, steps, vocab):
    """Decode steps, sharing the cache descriptors the graph returns.

    The decode graph takes the K/V caches as arguments and returns updated ones.
    Bufferization makes it write into a fresh buffer rather than the caller's, so
    the returned descriptors -- not the input ones -- are what the next step must
    read. Feeding the original descriptors back would silently decode against a
    stale cache.
    """
    if steps <= 0:
        return []
    return [
        "",
        "  /* ---- decode steps ---- */",
        "  MemRef2 m_ids_dec = make_2(input_ids, 1, 1);",
        "  MemRef1 m_pos_dec = make_1((void *)cache_position, 1);",
        "  MemRef4 m_k_in = results.key;",
        "  MemRef4 m_v_in = results.value;",
        "  for (unsigned step = 0; step < %d; ++step) {" % steps,
        "    input_ids[0] = (long long)best;",
        "    cache_position[0] = %d + (long long)step;" % prompt_len,
        "    GraphResults dec;",
        "    dec.position = make_1((void *)cache_position, 1);",
        f"    dec.key = make_4(k_cache_f, 1, 8, {cache_len}, {head_dim});",
        f"    dec.value = make_4(v_cache_f, 1, 8, {cache_len}, {head_dim});",
        "    dec.logits = make_3(logits_f, 1, 1, %d);" % vocab,
        "    _mlir_ciface_forward_decode(&dec, &p0, &p1, &p2, &p3, &p4, &p5,"
        "        &p6, &p7, &p8, &p9, &p10, &p11, &p12, &p13, &m_ids_dec,"
        "        &m_pos_dec, &m_k_in, &m_v_in);",
        "    const float *step_logits =",
        "        (const float *)dec.logits.aligned + dec.logits.offset;",
        "    best = 0; best_value = step_logits[0];",
        "    for (unsigned i = 1; i < %d; ++i)" % vocab,
        "      if (step_logits[i] > best_value) { best_value = step_logits[i]; best = i; }",
        '    nr_puts("[model] decode step ");',
        "    nr_hex32(step);",
        '    nr_puts(" position=");',
        "    nr_hex32((uint32_t)cache_position[0]);",
        '    nr_puts(" token=");',
        "    nr_hex32(best);",
        '    nr_puts(" logit_bits=");',
        "    { union { float f; uint32_t u; } v = { best_value }; nr_hex32(v.u); }",
        '    nr_puts("\\r\\n");',
        "    /* next step reads the cache the graph just returned */",
        "    m_k_in = dec.key;",
        "    m_v_in = dec.value;",
        "  }",
    ]


def workspace_declarations(entries):
    """Emit NOLOAD .workspace arrays, one explicit asm block per array.

    Symbol names get a `_raw` suffix and a typed pointer alias is declared, so
    the arrays can be indexed without casting at every use.
    """
    lines = []
    for name, size in entries:
        lines += [
            f'__asm__(".section .workspace,\\"aw\\",@nobits\\n"',
            f'        ".balign 64\\n"',
            f'        "{name}_raw:\\n"',
            f'        ".skip {size}\\n"',
            f'        ".previous\\n");',
            f'extern unsigned char {name}_raw[] __asm__("{name}_raw");',
        ]
    return lines


def main_source(layout, prompt_ids, params_bytes, cache_len, layers, head_dim,
                decode_steps=0, vocab=151936):
    """Emit the bare-metal entry.

    Only control flow, descriptor construction and UART output live here; every
    floating point result comes from the compiled graph.
    """
    # The layout's section order *is* the compiled parameter order (verified
    # independently by run_graph_host.py's param_order_matches_layout), so index
    # into it rather than matching names -- the shared embedding/lm_head matrix
    # is named either way depending on which tensor the content match found.
    ordered = layout["sections"]
    if len(ordered) != len(PARAM_ORDER) + (0 if layers == 1 else 0):
        pass
    args = []
    for name, section in zip(PARAM_ORDER, ordered):
        shape = section["shape"]
        # `weights` is a float*, so the offset must be in ELEMENTS. Using the
        # byte offset here advances 4x too far and every parameter except the
        # first is read from outside the loaded segment.
        offset = section["offset_elements"]
        rank = len(shape)
        dims = ", ".join(str(d) for d in shape)
        size = section["bytes"]
        view = f"weights + {offset}"
        if rank == 1:
            args.append(f"  MemRef1 p{len(args)} = make_1({view}, {shape[0]});")
        else:
            strides = []
            for i in range(rank):
                stride = 1
                for d in shape[i + 1:]:
                    stride *= d
                strides.append(str(stride))
            args.append(
                f"  MemRef{rank} p{len(args)} = make_{rank}({view}, {dims});")
    # ids, cache position, K cache, V cache -- after the parameters, matching the
    # compiled subgraph's argument order.
    args.append("  MemRef2 m_ids = make_2(input_ids, 1, %d);" % len(prompt_ids))
    args.append("  MemRef1 m_pos = make_1((void *)cache_position, 1);")
    # the graph's cache ABI is memref<1x8x{cap}x{head_dim}> per layer
    args.append(f"  MemRef4 m_k = make_4(k_cache_f, 1, 8, {cache_len}, {head_dim});")
    args.append(f"  MemRef4 m_v = make_4(v_cache_f, 1, 8, {cache_len}, {head_dim});")
    # Derived, never hand-written: the arity of this call was already wrong once
    # when it was typed out by hand.
    descriptor_ranks = [len(section["shape"]) for section in ordered]
    input_ranks = descriptor_ranks + [2, 1, 4, 4]
    prototype = []
    line = "extern void _mlir_ciface_forward_prefill(GraphResults *"
    for rank in input_ranks:
        piece = f", MemRef{rank} *"
        if len(line) + len(piece) > 76:
            prototype.append(line)
            line = "   "
        line += piece
    prototype.append(line + ");")
    if decode_steps:
        decode_prototype = [l.replace("_forward_prefill", "_forward_decode")
                            for l in prototype]
    else:
        decode_prototype = []
    call_arguments = ["&results"] + [f"&p{i}" for i in range(len(descriptor_ranks))] \
        + ["&m_ids", "&m_pos", "&m_k", "&m_v"]
    call_lines = []
    line = "  _mlir_ciface_forward_prefill("
    for index, argument in enumerate(call_arguments):
        piece = argument + (", " if index + 1 < len(call_arguments) else ");")
        if len(line) + len(piece) > 76:
            call_lines.append(line.rstrip())
            line = "      "
        line += piece
    call_lines.append(line)

    global DECODE_BODY
    DECODE_BODY = decode_body(len(prompt_ids), cache_len, head_dim,
                              decode_steps, vocab)
    return "\n".join([
        '#include "support.h"',
        '#include "nr_runtime.h"',
        "",
        "/* These live in the linker's NOLOAD .workspace section, so the uploaded",
        " * image does not carry them. The explicit `@nobits` is required: a plain",
        " * `section(\".workspace\")` attribute emits SHT_PROGBITS and the linker",
        " * then refuses the section type mismatch and counts the bytes into the",
        " * image. support.c solves the same problem the same way. */",
        *workspace_declarations([
            ("weight_arena", params_bytes),
            ("k_cache", layers * 8 * cache_len * head_dim * 4),
            ("v_cache", layers * 8 * cache_len * head_dim * 4),
            ("logits", 1 * 1 * 151936 * 4),
            ("cache_position", 8),
            ("input_ids", len(prompt_ids) * 8),
        ]),
        "static float *const weights = (float *)weight_arena_raw;",
        "static float *const k_cache_f = (float *)k_cache_raw;",
        "static float *const v_cache_f = (float *)v_cache_raw;",
        "static float *const logits_f = (float *)logits_raw;",
        "static long long *const cache_position = (long long *)cache_position_raw;",
        "static long long *const input_ids = (long long *)input_ids_raw;",
        "",
        "typedef struct {",
        "  MemRef1 position;",
        "  MemRef4 key;",
        "  MemRef4 value;",
        "  MemRef3 logits;",
        "} GraphResults;",
        "",
        *prototype,
        *decode_prototype,
        "",
        f"static const long long prompt[{len(prompt_ids)}] = {{"
        + ", ".join(str(v) for v in prompt_ids) + "};",
        "",
        "int launch(void) {",
        *args,
        "  for (unsigned i = 0; i < sizeof(prompt) / sizeof(prompt[0]); ++i)",
        "    input_ids[i] = prompt[i];",
        "  cache_position[0] = 0;",
        f"  for (unsigned i = 0; i < {layers * 8 * cache_len * head_dim}; ++i) {{",
        "    k_cache_f[i] = 0.0f; v_cache_f[i] = 0.0f;",
        "  }",
        "  GraphResults results;",
        "  results.position = make_1((void *)cache_position, 1);",
        f"  results.key = make_4(k_cache_f, 1, 8, {cache_len}, {head_dim});",
        f"  results.value = make_4(v_cache_f, 1, 8, {cache_len}, {head_dim});",
        "  results.logits = make_3(logits_f, 1, 1, 151936);",
        *call_lines,
        "  /* Read the logits from the descriptor the graph returned. A result",
        "   * descriptor is an output slot: the caller supplies storage for the",
        "   * slot, but the callee decides what it points at (here an allocation",
        "   * of its own), so reading the static buffer handed in reads zeros. */",
        "  const float *out_logits =",
        "      (const float *)results.logits.aligned + results.logits.offset;",
        "  unsigned best = 0; float best_value = out_logits[0];",
        f"  for (unsigned i = 1; i < {vocab}; ++i)",
        "    if (out_logits[i] > best_value) { best_value = out_logits[i]; best = i; }",
        '  nr_puts("[model] prefill argmax token=");',
        "  nr_hex32(best);",
        '  nr_puts(" logit_bits=");',
        "  { union { float f; uint32_t u; } v = { best_value }; nr_hex32(v.u); }",
        '  nr_puts("\\r\\n");',
        *DECODE_BODY,
        "  return 0;",
        "}",
        "",
    ])


def build(args, layout):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    llvm_bin = args.repo_root / "llvm/build-2d26/bin"
    qwen = args.repo_root / "examples/FPGA-BOSCAME/qwen3-0.6b"
    common = args.repo_root / "examples/FPGA-BOSCAME/common"
    nr = common / "nr"

    params_bytes = 0
    for section in layout["sections"]:
        params_bytes = max(params_bytes, section["offset_bytes"] + section["bytes"])
    params_bytes = (params_bytes + 63) & ~63

    source = output / "model_main.c"
    source.write_text(main_source(layout, args.prompt_ids, params_bytes,
                                  args.max_cache_len, args.layers, args.head_dim,
                                  decode_steps=args.decode_steps))

    flags = ["--target=riscv64-unknown-elf", "-march=rv64gc_zicbom", "-mabi=lp64d",
             "-mcmodel=medany", "-O2", "-nostdlib", "-ffreestanding", "-fno-builtin",
             "-fno-pie", "-fno-vectorize", "-fno-slp-vectorize", "-ffp-contract=off",
             f"-I{qwen}", f"-I{nr}", f"-I{common}/uart"]
    steps = {}

    def run(command, **kwargs):
        """Run a tool, capturing output unless the caller redirected it."""
        redirected = any(k in kwargs for k in ("stdin", "stdout", "stderr"))
        if redirected:
            result = subprocess.run([str(c) for c in command], **kwargs)
            result.stdout = result.stdout or ""
            result.stderr = result.stderr or ""
            return result
        return subprocess.run([str(c) for c in command], capture_output=True,
                              text=True, **kwargs)

    result = run([llvm_bin / "clang", *flags, "-c", source, "-o", output / "model_main.o"])
    steps["compile_main"] = {"returncode": result.returncode,
                             "stderr": result.stderr[-1500:]}
    if result.returncode:
        return {"status": "FAILED compiling main", "steps": steps}

    graph_object = args.graph_object
    if args.graph_ir is not None:
        # The graph's own LLVM IR must go through the same assembly
        # post-processing the handwritten operator pipeline uses: the AME
        # encoder rewrites the accelerator instructions and
        # restrict_fpga_assembly.py inserts the fences the board requires around
        # vector memory accesses. Compiling llc straight to an object skips both
        # and the ELF audit rejects it.
        assembly = output / "forward_prefill.s"
        encoded = output / "forward_prefill.encoded.s"
        restricted = output / "forward_prefill.nr.S"
        tools = args.repo_root / "examples/FPGA-BOSCAME/tools"
        run([llvm_bin / "llc", args.graph_ir, "-O2", "-filetype=asm",
             "-mtriple=riscv64", "-target-abi=lp64d",
             "-mattr=+m,+a,+f,+d,+c,-v,+xboscame", "-code-model=medium",
             "-o", assembly])
        result = run([sys.executable, tools / "ame_to_word.py"],
                     stdin=assembly.open("rb"), stdout=encoded.open("wb"))
        if result.returncode:
            return {"status": "FAILED encoding AME instructions",
                    "steps": {"ame_to_word": {"returncode": result.returncode,
                                              "stderr": result.stderr[-800:]}}}
        result = run([sys.executable, tools / "restrict_fpga_assembly.py"],
                     stdin=encoded.open("rb"), stdout=restricted.open("wb"))
        if result.returncode:
            return {"status": "FAILED restricting assembly",
                    "steps": {"restrict": {"returncode": result.returncode,
                                           "stderr": result.stderr[-800:]}}}
        graph_object = output / "forward_prefill.nr.o"
        result = run([llvm_bin / "clang", *flags, "-c", restricted,
                      "-o", graph_object])
        steps["graph_object"] = {"returncode": result.returncode,
                                 "stderr": result.stderr[-1500:]}
        if result.returncode:
            return {"status": "FAILED assembling the graph", "steps": steps}

    if args.decode_ir is not None:
        decode_object = output / "forward_decode.nr.o"
        decode_assembly = output / "forward_decode.s"
        result = run([llvm_bin / "llc", args.decode_ir, "-O2", "-filetype=asm",
                      "-mtriple=riscv64", "-target-abi=lp64d",
                      "-mattr=+m,+a,+f,+d,+c,-v,+xboscame", "-code-model=medium",
                      "-o", decode_assembly])
        if result.returncode:
            steps["decode_llc"] = {"returncode": result.returncode,
                                   "stderr": result.stderr[-1500:]}
            return {"status": "FAILED compiling the decode graph", "steps": steps}
        for tool, source_suffix, target in (
                ("ame_to_word.py", "s", "encoded.s"),
                ("restrict_fpga_assembly.py", "encoded.s", "nr.S")):
            handle_in = (output / f"forward_decode.{source_suffix}").open("rb")
            handle_out = (output / f"forward_decode.{target}").open("wb")
            result = run([sys.executable, tools / tool], stdin=handle_in,
                         stdout=handle_out)
            handle_in.close(); handle_out.close()
            if result.returncode:
                steps[f"decode_{tool}"] = {"returncode": result.returncode,
                                           "stderr": result.stderr[-1500:]}
                return {"status": f"FAILED in {tool} for the decode graph",
                        "steps": steps}
        raw_decode_object = output / "forward_decode.raw.o"
        result = run([llvm_bin / "clang", *flags, "-c",
                      output / "forward_decode.nr.S", "-o", raw_decode_object])
        if result.returncode:
            steps["decode_assemble"] = {"returncode": result.returncode,
                                        "stderr": result.stderr[-1500:]}
            return {"status": "FAILED assembling the decode graph", "steps": steps}
        # Each graph lowerer emits its own `dealloc_helper` (from
        # bufferization-lower-deallocations), so two entry points compiled
        # separately collide at link time. Renaming it in the decode object is
        # enough: the helper is module-local in purpose, and objcopy rewrites the
        # references inside that object too.
        result = run([llvm_bin / "llvm-objcopy",
                      "--redefine-sym", "dealloc_helper=dealloc_helper_decode",
                      "--redefine-sym",
                      "_mlir_ciface_dealloc_helper=_mlir_ciface_dealloc_helper_decode",
                      raw_decode_object, decode_object])
        if result.returncode or not decode_object.is_file():
            steps["decode_rename"] = {"returncode": result.returncode,
                                      "stderr": result.stderr[-1500:]}
            return {"status": "FAILED renaming the decode helper", "steps": steps}
    else:
        decode_object = None

    adapters = args.adapters
    result = run([llvm_bin / "clang", *flags, "-c", adapters,
                  "-o", output / "qwen_triton_adapters.o"])
    steps["compile_adapters"] = {"returncode": result.returncode,
                                 "stderr": result.stderr[-1500:]}
    if result.returncode:
        return {"status": "FAILED compiling adapters", "steps": steps}

    runtime_sources = ["crt.S", "nr_runtime.c", "nr_math.c", "ame_sync.c", "nr_copy.S"]
    runtime_objects = []
    for name in runtime_sources:
        obj = output / (Path(name).stem + ".o")
        source_path = nr / name
        source_flags = flags
        # nr_copy.S uses the RVV copy loop, so it alone needs V enabled; this
        # mirrors the handwritten operator pipeline's rule for the same file.
        if name == "nr_copy.S":
            source_flags = ["-march=rv64gcv_zicbom"
                            if f.startswith("-march=") else f for f in flags]
        command = [llvm_bin / "clang", *source_flags, "-c", source_path, "-o", obj]
        result = run(command)
        if result.returncode:
            steps[f"runtime_{name}"] = {"returncode": result.returncode,
                                        "stderr": result.stderr[-800:]}
            return {"status": f"FAILED building NR runtime {name}", "steps": steps}
        runtime_objects.append(obj)

    # support.c deliberately keeps a 640 MiB NOLOAD arena for the operator
    # examples; linking it here would collide with the model's own workspace and
    # overflow HIGH. The model main uses only nr_runtime.c entry points.
    elf = output / "qwen_model.elf"
    # ld.lld explicitly, as the handwritten operator pipeline does: driving the
    # link through clang picks the host ld, which has no riscv emulation.
    result = run(["ld.lld", "-m", "elf64lriscv", "--gc-sections",
                  "-T", nr / "nr.ld", f"-Map={output / 'qwen_model.map'}",
                  "-o", elf,
                  output / "model_main.o", graph_object,
                  output / "qwen_triton_adapters.o",
                  *( [decode_object] if decode_object is not None else [] ),
                  *runtime_objects, args.archive])
    steps["link"] = {"returncode": result.returncode, "stderr": result.stderr[-3000:]}
    if result.returncode or not elf.is_file():
        return {"status": "FAILED linking the bare-metal image", "steps": steps}

    binary = output / "qwen_model.bin"
    run([llvm_bin / "llvm-objcopy", "-O", "binary", elf, binary])
    audit = run([sys.executable,
                 args.repo_root / "examples/FPGA-BOSCAME/tools/check_nr_elf.py",
                 elf, "--objdump", llvm_bin / "llvm-objdump",
                 "--output", output / "elf-audit.json"])
    steps["audit"] = {"returncode": audit.returncode, "stdout": audit.stdout[-1500:],
                      "stderr": audit.stderr[-1500:]}
    undefined = run([llvm_bin / "llvm-nm", "--undefined-only", elf])
    sizes = run([llvm_bin / "llvm-size", elf])

    return {
        "status": "PASS" if audit.returncode == 0 else "built but the ELF audit failed",
        "steps": steps,
        "image": {"elf": str(elf), "bytes": elf.stat().st_size,
                  "bin": str(binary), "bin_bytes": binary.stat().st_size},
        "parameter_buffer_bytes": params_bytes,
        "undefined_symbols": sorted({l.split()[-1] for l in undefined.stdout.splitlines()
                                     if l.strip()}),
        "sizes": sizes.stdout,
    }


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--graph-object", type=Path, default=None,
                        help="prebuilt graph object (ignored when --graph-ir is given)")
    parser.add_argument("--decode-ir", type=Path, default=None,
                        help="LLVM IR for forward_decode; adds 8 decode steps")
    parser.add_argument("--decode-steps", type=int, default=0)
    parser.add_argument("--graph-ir", type=Path, default=None,
                        help="graph LLVM IR; assembled through ame_to_word.py and "
                             "restrict_fpga_assembly.py like the operator pipeline")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--adapters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--max-cache-len", type=int, default=128)
    parser.add_argument("--prompt-ids", default="151644,872,198,3838,374,9625,30,"
                                               "151645,198,151644,77091,198,"
                                               "151667,271,151668,271")
    args = parser.parse_args()
    args.prompt_ids = [int(v) for v in args.prompt_ids.split(",") if v.strip()]
    layout = json.loads(args.layout.read_text())
    report = build(args, layout)
    (args.output / "model-image.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("steps", "sizes")}, indent=2))
    for name, step in report.get("steps", {}).items():
        if step.get("returncode"):
            print(f"--- {name} failed ---")
            print(step.get("stderr", "")[:1500])
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(cli())
