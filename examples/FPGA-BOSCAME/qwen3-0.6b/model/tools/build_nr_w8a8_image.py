#!/usr/bin/env python3
"""Bare-metal NR image for the fully-quantised 28-layer Qwen3 graph.

The FP32 image builder hand-wrote its 14 parameter descriptors. That does not
scale: the W8A8 graph takes 508 parameters and 1578 workspace buffers, so every
descriptor, every workspace declaration and the single call are generated from the
rewrite report and the parameter segment's recorded offsets.

Layout of the generated program:

  * the parameter arena is a NOLOAD ``.workspace`` array whose bytes arrive as
    their own DDR segment (build_nr_w8a8_segment.py). Each parameter descriptor
    points at ``arena + offset`` for the offset the segment builder recorded.
  * the W8A8 workspace is likewise NOLOAD: 4 buffers per linear plus one per
    embedding, all caller-owned because one-shot bufferize treats an external
    call's operands as read-only. The accumulators are zeroed before each call --
    the int8 matmul accumulates into C rather than writing it.
  * the KV caches, the input ids and the cache position are ordinary workspace.
  * one call into ``_mlir_ciface_forward_prefill`` (and ``_forward_decode`` when
    decode steps are requested), then argmax over the vocabulary, printed to UART.

Nothing here computes model arithmetic; every value comes from a compiled kernel.
"""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
import subprocess
import sys

CTYPE = {"TensorDType.Int8": "int8_t", "TensorDType.Int32": "int32_t",
         "TensorDType.Int64": "int64_t", "TensorDType.Float32": "float"}
NP = {"TensorDType.Int8": "np.int8", "TensorDType.Int32": "np.int32"}


def resolve_linker(requested, llvm):
    """Avoid old RISC-V --wrap/relaxation symbol corruption (observed LLD 15)."""
    candidates = [str(requested)] if requested else [str(llvm / "ld.lld"), "ld.lld-20", "ld.lld"]
    failures = []
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if not resolved:
            failures.append(candidate + ": absent")
            continue
        resolved = str(Path(resolved).absolute())
        version = subprocess.check_output([resolved, "--version"], text=True).strip()
        match = re.search(r"\bLLD (\d+)\.", version)
        if not match or int(match[1]) < 20:
            failures.append(candidate + ": unsupported " + version)
            continue
        return {"path": resolved, "binary_path": str(Path(resolved).resolve()), "version": version,
                "sha256": hashlib.sha256(Path(resolved).read_bytes()).hexdigest()}
    raise ValueError("NR model linking requires LLD >= 20; pass --linker explicitly. " + "; ".join(failures))


def fixed_prompt_bytes(args):
    text = getattr(args, 'prompt_text', None)
    path = getattr(args, 'prompt_file', None)
    if text is not None and path is not None:
        raise ValueError("choose either prompt_text or prompt_file")
    if text is None and path is None:
        return None
    data = path.read_bytes() if path is not None else text.encode('utf-8')
    data.decode('utf-8', errors='strict')
    if not data or len(data) > 2048:
        raise ValueError("fixed prompt must contain 1..2048 UTF-8 bytes")
    return data


def fixed_prompt_source(args, data, tokenizer_bytes):
    """Board text control only; expected IDs check the encoder, never feed it."""
    capacity = getattr(args, 'text_output_bytes', 65536)
    return [
        'static const uint8_t fixed_user_text[] = {' + ','.join(map(str, data)) + '};',
        'static const uint32_t expected_prompt_ids[] = {' + ','.join(map(str, args.prompt_ids)) + '};',
        f'#define TEXT_OUTPUT_CAPACITY {capacity}',
        r'''
typedef struct { uint8_t data[TEXT_OUTPUT_CAPACITY]; size_t length; int overflow; } TextOutput;
static TextOutput text_output;
static void append_text(void *context, const uint8_t *bytes, size_t count) {
  TextOutput *out = (TextOutput *)context;
  if (count > sizeof(out->data) - out->length) { out->overflow = 1; return; }
  memcpy(out->data + out->length, bytes, count);
  out->length += count;
}
static void text_hex(const uint8_t *bytes, size_t count) {
  static const char digits[] = "0123456789abcdef";
  for (size_t i = 0; i < count; ++i) {
    char pair[3] = {digits[bytes[i] >> 4], digits[bytes[i] & 15], 0};
    nr_puts(pair);
  }
}
static int record_text_token(const QwenTokenizerResource *resource,
                             QwenUtf8Decoder *utf8, unsigned step, unsigned token) {
  size_t previous = text_output.length;
  if (qwen_decode_token(resource, utf8, token, 1, append_text, &text_output) ||
      text_output.overflow) {
    nr_puts("[text] decode or output capacity: FAIL\r\n"); return 1;
  }
  nr_puts("[text] prediction="); nr_hex32(step);
  nr_puts(" token="); nr_hex32(token);
  nr_puts(" eos="); nr_hex32(token == 151643 || token == 151645);
  nr_puts(" bytes="); nr_hex32(text_output.length - previous);
  nr_puts(" hex="); text_hex(text_output.data + previous, text_output.length - previous);
  nr_puts("\r\n");
  return 0;
}
int launch(void) {
  QwenTokenizerResource resource;
''',
        f'  if (qwen_tokenizer_open(&resource, tokenizer_blob_raw, {tokenizer_bytes})) {{',
        '    nr_puts("[text] tokenizer resource: FAIL\\r\\n"); return 1; }',
        r'''
  static uint8_t prompt[8192];
  static uint32_t encoded_ids[2048];
  size_t bytes = 0, count = 0;
  uint64_t encode_begin = nr_cycles();
  if (qwen_chat_single_turn(prompt, sizeof(prompt), &bytes, 0, 0, 0,
                            fixed_user_text, sizeof(fixed_user_text), 0) ||
      qwen_encode(&resource, prompt, bytes, encoded_ids, 2048, &count)) {
    nr_puts("[text] template or tokenizer: FAIL\r\n"); return 1;
  }
  uint64_t encode_cycles = nr_cycles() - encode_begin;
  nr_puts("[text] prompt count="); nr_hex32(count); nr_puts(" ids=");
  for (size_t i = 0; i < count; ++i) {
    if (i) nr_puts(" "); nr_hex32(encoded_ids[i]);
  }
  nr_puts(" encode_cycles="); nr_hex64(encode_cycles); nr_puts("\r\n");
''',
        f'  if (count != {args.prefill_len}) {{',
        '    nr_puts("[text] prompt length differs from compiled prefill: FAIL\\r\\n"); return 1; }',
        r'''
  for (size_t i = 0; i < count; ++i) {
    if (encoded_ids[i] != expected_prompt_ids[i]) {
      nr_puts("[text] prompt IDs differ from expected reference: FAIL\r\n"); return 1;
    }
  }
  nr_puts("verify fixed prompt tokenizer: PASS\r\n");
  /* The actual encoder output feeds the graph. The expected array is read only
   * above for validation; it never supplies model operands. */
  for (size_t i = 0; i < count; ++i) input_ids[i] = encoded_ids[i];
  reset_cache();
  text_output.length = 0; text_output.overflow = 0;
  QwenUtf8Decoder utf8 = {{0,0,0,0},0,0};
  unsigned token = 0; float score = 0;
  nr_puts("[text] mode=fixed-validation; EOS is recorded, never early-stops\r\n");
  if (run_prefill(0, &token, &score, 1)) return 1;
  if (record_text_token(&resource, &utf8, 0, token)) return 1;
''',
        *([f'  for (unsigned step = 0; step < {args.decode_steps}; ++step) {{',
           '    input_ids[0] = token;',
           f'    if (run_decode({args.prefill_len} + step, &token, &score, 1)) return 1;',
           '    if (record_text_token(&resource, &utf8, step + 1, token)) return 1;',
           '  }'] if args.decode_ir else []),
        r'''
  size_t previous = text_output.length;
  qwen_decode_finish(&utf8, append_text, &text_output);
  if (text_output.overflow) { nr_puts("[text] output capacity: FAIL\r\n"); return 1; }
  nr_puts("[text] finish hex=");
  text_hex(text_output.data + previous, text_output.length - previous);
  nr_puts("\r\n[text] output bytes="); nr_hex32(text_output.length);
  nr_puts(" hex="); text_hex(text_output.data, text_output.length);
  nr_puts("\r\n[text] BEGIN\r\n");
  /* A single length-delimited UART write keeps model traces outside the text.
   * UTF-8 was incrementally decoded on board after each actual prediction. */
  nr_write(text_output.data, text_output.length);
  nr_puts("\r\n[text] END\r\nverify fixed text validation: PASS\r\n");
  return 0;
}
''']


def align(n, a=64):
    return (n + a - 1) // a * a


def element_size(dtype):
    return {"TensorDType.Int8": 1, "TensorDType.Int32": 4,
            "TensorDType.Int64": 8, "TensorDType.Float32": 4}[dtype]


def prepare_reference(args, output):
    """Optional validation data in the final image; never in the kernel archive."""
    if not args.reference_arrays:
        return None
    import math
    if any(not math.isfinite(x) or x < 0 for x in
           (args.reference_atol, args.reference_mean_atol)):
        raise ValueError("reference error thresholds must be finite and nonnegative")
    if args.interactive:
        raise ValueError("fixed numeric oracle cannot be used for arbitrary interactive prompts")
    metadata_path = (getattr(args, 'reference_metadata', None)
                     or args.reference_arrays.with_name('quant-reference.json'))
    metadata = json.loads(metadata_path.read_text())
    for name, expected in (('prompt_ids', args.prompt_ids), ('layers', args.layers),
                           ('capacity', args.cache_len)):
        if metadata.get(name) != expected:
            raise ValueError(f"reference metadata {name} differs from image")
    if metadata.get('arithmetic_profile') != 'nr-fpga':
        raise ValueError("FPGA oracle requires explicit nr-fpga arithmetic_profile")
    trajectory = metadata.get('decode_steps_recorded', [])
    if len(trajectory) < args.decode_steps:
        raise ValueError("reference metadata lacks requested decode trajectory")
    import numpy as np
    total = args.prefill_len + args.decode_steps
    with np.load(args.reference_arrays) as arrays:
        logits = [arrays['prefill_logits'].reshape(-1, 151936)[-1]]
        logits += [arrays[f'decode_logits_{i}'].reshape(151936)
                   for i in range(args.decode_steps)]
        predicted = [int(np.argmax(row)) for row in logits]
        if metadata.get('prefill_argmax_last') != predicted[0]:
            raise ValueError("reference metadata prefill token differs from arrays")
        for step, row in enumerate(trajectory[:args.decode_steps]):
            expected = {'step': step, 'cache_position': args.prefill_len + step,
                        'input_token': predicted[step],
                        'generated_token': predicted[step + 1]}
            if any(row.get(k) != value for k, value in expected.items()):
                raise ValueError(f"reference metadata decode trajectory differs at step {step}")
        caches = []
        for name in ('kv_key_used', 'kv_value_used'):
            a = arrays[name]
            if a.shape == (args.layers, total, 8, 128):
                a = a.transpose(0, 2, 1, 3)
            if a.shape != (args.layers, 8, total, 128):
                raise ValueError(f"reference {name} shape {a.shape} does not match image")
            caches.append(a)
        values = np.concatenate([np.stack(logits).ravel(), *(a.ravel() for a in caches)]).astype('<f4')
        if not np.isfinite(values).all():
            raise ValueError("reference contains non-finite values")
    blob = output / 'numeric-reference.bin'
    blob.write_bytes(values.astype('<f4').tobytes())
    asm = '.section .rodata.qwen_reference,"a",@progbits\n.balign 64\n'
    asm += '.globl model_reference_raw\nmodel_reference_raw:\n'
    asm += '.incbin ' + json.dumps(str(blob.resolve())) + '\n'
    (output / 'numeric-reference.S').write_text(asm)
    record = {'schema_version': 1, 'layers': args.layers, 'prefill_len': args.prefill_len,
              'decode_steps': args.decode_steps, 'vocab_size': 151936,
              'source': str(args.reference_arrays),
              'source_sha256': hashlib.sha256(args.reference_arrays.read_bytes()).hexdigest(),
              'metadata_source': str(metadata_path),
              'metadata_sha256': hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
              'prompt_ids': args.prompt_ids,
              'arithmetic_profile': metadata['arithmetic_profile'],
              'capacity': args.cache_len,
              'trajectory_verified': True,
              'blob_sha256': hashlib.sha256(blob.read_bytes()).hexdigest(),
              'bytes': blob.stat().st_size, 'layout': 'logits[steps+1,V], K[L,8,T,128], V[L,8,T,128]',
              'max_abs_tolerance': args.reference_atol,
              'mean_abs_tolerance': args.reference_mean_atol,
              'purpose': 'validation only; not used to choose tokens or update model state'}
    (output / 'numeric-reference.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def emit_workspace(name, size):
    return [
        f'__asm__(".section .workspace,\\"aw\\",@nobits\\n"',
        f'        ".balign 64\\n"',
        f'        "{name}_raw:\\n"',
        f'        ".skip {size}\\n"',
        f'        "{name}_end:\\n"',
        f'        ".previous\\n");',
        f'extern unsigned char {name}_raw[] __asm__("{name}_raw");',
    ]


def check_contract(report, segment, args):
    """Match generated descriptors to actual imported ABI AND final LLVM ranks.

    LLVM opaque pointers erase dtype and static extent information. Checking
    ranks alone can accept a cap512 graph backed by a cap32 allocation.
    """
    if getattr(args, 'profile_progress', False) and not getattr(args, 'profile_kernels', False):
        raise ValueError('--profile-progress requires --profile-kernels')
    if getattr(args, 'profile_probe', None):
        if not (getattr(args, 'profile_kernels', False) and getattr(args, 'profile_progress', False)):
            raise ValueError('--profile-probe requires --profile-kernels and --profile-progress')
        from kernel_profile import parse_probe
        probe = parse_probe(args.profile_probe)
        prototypes = re.findall(r'^extern void (_mlir_ciface_kernel_\w+)\(',
                                args.adapters.read_text(), re.MULTILINE)
        if probe['symbol'] not in prototypes:
            raise ValueError('--profile-probe symbol absent from typed adapters: ' + probe['symbol'])
    scalar_dtype = {"TensorDType.Float32": "f32", "TensorDType.Int8": "i8",
                    "TensorDType.Int32": "i32", "TensorDType.Int64": "i64"}
    if not 1 <= args.layers <= 28 or args.head_dim != 128:
        raise ValueError("Qwen3 needs 1..28 layers and head_dim=128")
    if not 1 <= args.cache_len <= 0x7fffffff:
        raise ValueError("cache capacity must fit positive int32 positions")
    if args.prefill_len != len(args.prompt_ids) or args.prefill_len < 1:
        raise ValueError("prompt ID count must equal prefill_len")
    if any(t < 0 or t >= 151936 for t in args.prompt_ids):
        raise ValueError("prompt token outside vocabulary")
    if args.decode_steps < 0 or args.prefill_len + args.decode_steps > args.cache_len:
        raise ValueError("prefill + decode exceeds cache capacity")
    if args.decode_steps and not args.decode_ir:
        raise ValueError("decode_steps requires decode IR; use --decode-steps=0 for prefill-only")
    if args.graph != "prefill":
        raise ValueError("image entry graph must be prefill; decode IR is a separate entry")
    if args.interactive and (not args.decode_ir or not args.tokenizer_blob):
        raise ValueError("interactive mode requires decode IR and tokenizer resource")
    prompt = fixed_prompt_bytes(args)
    if prompt is not None:
        if args.interactive:
            raise ValueError("fixed prompt and interactive modes are mutually exclusive")
        if not args.tokenizer_blob:
            raise ValueError("fixed prompt requires tokenizer resource")
        if args.prefill_len > 2048:
            raise ValueError("fixed prompt encoder capacity is 2048 tokens")
        if not 1 <= getattr(args, 'text_output_bytes', 65536) <= 1048576:
            raise ValueError("text_output_bytes must be 1..1048576")
    if args.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if getattr(args, "uart_probe", False):
        raise ValueError("use common/nr/probes UART RX probe; RA cannot read UART registers")
    graphs = [(args.graph, args.graph_ir)]
    if args.decode_ir:
        graphs.append(("decode", args.decode_ir))
    placement = {p["name"]: p for p in segment["placement"]}
    if len(placement) != len(segment["placement"]) or segment["bytes"] <= 0:
        raise ValueError("invalid or duplicate parameter segment placement")
    occupied = sorted((p["offset_bytes"], p["offset_bytes"] + p["bytes"])
                      for p in segment["placement"])
    if any(a[1] > b[0] for a, b in zip(occupied, occupied[1:])):
        raise ValueError("parameter segment placements overlap")

    def record(shape, dtype, node=None):
        return {"shape": shape, "dtype": dtype, "node": node}

    def compare_abi(actual, expected, label):
        if len(actual) != len(expected):
            raise ValueError(f"{label}: descriptor count differs from generated ABI")
        offset = 0
        for index, (a, e) in enumerate(zip(actual, expected)):
            rank = len(e["shape"])
            size = 8 * (3 + 2 * rank)
            if (a.get("index") != index or a.get("shape") != e["shape"]
                    or a.get("dtype") != e["dtype"] or a.get("rank") != rank
                    or a.get("descriptor_bytes_lp64") != size
                    or (e["node"] is not None and a.get("node") != e["node"])):
                raise ValueError(f"{label}[{index}]: shape/dtype/order differs from generated descriptor: {a} vs {e}")
            if "aggregate_offset_bytes_lp64" in a and a["aggregate_offset_bytes_lp64"] != offset:
                raise ValueError(f"{label}[{index}]: aggregate offset differs from C struct")
            offset += size
        return offset

    evidence = {}
    for kind, ir in graphs:
        graph = report["graphs"][kind]
        if graph["parameters"] != report["graphs"][args.graph]["parameters"]:
            raise ValueError("prefill/decode parameter order, shape or dtype differs")
        for p in graph["parameters"]:
            q = placement.get(p["name"])
            size = element_size(p["dtype"])
            if not 1 <= len(p["shape"]) <= 4 or any(d <= 0 for d in p["shape"]):
                raise ValueError(f"unsupported static parameter shape: {p}")
            for d in p["shape"]:
                size *= d
            if (q is None or q["shape"] != p["shape"] or q["bytes"] != size
                    or q["dtype"] != scalar_dtype[p["dtype"]] or q["offset_bytes"] < 0
                    or q["offset_bytes"] % 64
                    or q["offset_bytes"] + size > segment["bytes"]):
                raise ValueError(f"parameter segment mismatch: {p['name']}")
        text = ir.read_text()
        name = "forward_" + kind
        signature = next((line for line in text.splitlines()
                          if line.startswith("define ") and f"@{name}(" in line), None)
        if signature is None:
            raise ValueError(f"{ir}: missing graph definition {name}")
        ranks = [int(n) for n in re.findall(
            r"\{ ptr, ptr, i64, \[(\d+) x i64\], \[\d+ x i64\] \}",
            signature.split("@", 1)[0])]
        expected = [1, 4, 4] * args.layers + [3]
        if ranks != expected:
            raise ValueError(f"{ir}: result ranks {ranks} != {expected}")
        wrapper_marker = f"define void @_mlir_ciface_{name}("
        if wrapper_marker not in text:
            raise ValueError(f"{ir}: missing CIFACE graph definition {name}")
        wrapper = text.split(wrapper_marker, 1)[1].split("\n}", 1)[0]
        inputs = [int(n) for n in re.findall(
            r"load \{ ptr, ptr, i64, \[(\d+) x i64\], \[\d+ x i64\] \}", wrapper)]
        expected_inputs = ([len(p["shape"]) for p in graph["parameters"]]
                           + [2] + [1, 4, 4] * args.layers
                           + [len(w["shape"]) for w in graph.get("workspace_inputs", [])])
        if inputs != expected_inputs:
            raise ValueError(f"{ir}: CIFACE input ranks/order differ from report")
        abi = graph.get("entry_abi")
        if not abi or abi.get("entry") != name:
            raise ValueError(f"{kind}: missing actual imported entry_abi; regenerate the graph report")
        length = args.prefill_len if kind == "prefill" else 1
        kv = [1, 8, args.cache_len, args.head_dim]
        generated_inputs = [record(p["shape"], scalar_dtype[p["dtype"]], p["name"])
                            for p in graph["parameters"]]
        generated_inputs += [record([1, length], "i64")]
        generated_inputs += [record([1], "i64"), record(kv, "f32"), record(kv, "f32")] * args.layers
        for w in graph.get("workspace_inputs", []):
            if not 1 <= len(w["shape"]) <= 4 or any(d <= 0 for d in w["shape"]):
                raise ValueError(f"unsupported static workspace shape: {w}")
            generated_inputs.append(record(w["shape"], scalar_dtype[w["dtype"]], w["name"]))
        compare_abi(abi.get("inputs", []), generated_inputs, f"{kind} inputs")
        outputs = abi.get("outputs", [])
        # The graph may slice the last hidden token before lm_head. Both [1,1,V]
        # and [1,prefill_len,V] are legal; collect() deliberately reads the last.
        if not outputs or outputs[-1].get("shape") not in ([1, 1, 151936], [1, length, 151936]):
            raise ValueError(f"{kind}: unexpected logits shape in actual imported ABI")
        generated_outputs = [record([1], "i64"), record(kv, "f32"), record(kv, "f32")] * args.layers
        generated_outputs.append(record(outputs[-1]["shape"], "f32"))
        result_bytes = compare_abi(outputs, generated_outputs, f"{kind} outputs")
        if (abi.get("parameter_count") != len(graph["parameters"])
                or abi.get("result_descriptor_count") != len(generated_outputs)
                or abi.get("result_aggregate_bytes_lp64") != result_bytes):
            raise ValueError(f"{kind}: imported ABI aggregate metadata differs from C layout")
        evidence[kind] = {"result_ranks": ranks, "input_ranks": inputs,
                          "result_bytes": result_bytes, "shape_dtype_checked": True,
                          "cache_shape": kv, "input_ids_shape": [1, length],
                          "logits_shape": outputs[-1]["shape"]}
    return evidence


def generate(report, segment, args):
    """Emit ABI glue and session control; arithmetic stays in the compiled graph."""
    contract = check_contract(report, segment, args)
    layers, capacity, width = args.layers, args.cache_len, args.head_dim
    kv_elements = 8 * capacity * width
    graph = report["graphs"][args.graph]
    dec_graph = report["graphs"].get("decode") if args.decode_ir else None
    tokenizer_bytes = args.tokenizer_blob.stat().st_size if args.tokenizer_blob else 0
    prompt = fixed_prompt_bytes(args)
    offsets = {p["name"]: p["offset_bytes"] for p in segment["placement"]}
    out = ['#include "support.h"', '#include "nr_runtime.h"']
    if getattr(args, 'intermediate_arrays', None):
        out += ['extern void qwen_intermediate_begin(unsigned, unsigned);',
                'extern int qwen_intermediate_end(void);']
    if getattr(args, 'profile_kernels', False):
        out += ['extern void qwen_profile_reset(void);',
                'extern void qwen_profile_report(unsigned position);']
    if args.interactive or prompt is not None:
        out.append('#include "tokenizer_resource.h"')
    out += [f'#define LAYERS {layers}', f'#define CAPACITY {capacity}',
            f'#define KV_ELEMENTS {kv_elements}', '#define VOCAB 151936',
            'typedef struct { MemRef1 position; MemRef4 key, value; } CacheResult;',
            'typedef struct { CacheResult cache[LAYERS]; MemRef3 logits; } GraphResults;',
            '_Static_assert(sizeof(CacheResult) == 216, "cache result ABI");',
            '_Static_assert(sizeof(GraphResults) == LAYERS * 216 + 72, "graph result ABI");']
    for name, size in (("weight_arena", segment["bytes"]),
                       ("k_cache", layers * kv_elements * 4),
                       ("v_cache", layers * kv_elements * 4),
                       ("input_ids", args.prefill_len * 8), ("cache_position", 64)):
        out += emit_workspace(name, align(size))
    if tokenizer_bytes:
        out += emit_workspace("tokenizer_blob", align(tokenizer_bytes))
    out += ['static float *const k_cache_f = (float *)k_cache_raw;',
            'static float *const v_cache_f = (float *)v_cache_raw;',
            'static int64_t *const input_ids = (int64_t *)input_ids_raw;',
            'static int64_t *const cache_position = (int64_t *)cache_position_raw;']
    all_graphs = [("prefill", graph)] + ([("decode", dec_graph)] if dec_graph else [])
    ws = {}
    for kind, g in all_graphs:
        cursor, entries = 0, {}
        for e in g.get("workspace_inputs", []):
            size = element_size(e["dtype"])
            for dim in e["shape"]:
                size *= dim
            cursor = align(cursor)
            entries[e["name"]] = (cursor, size)
            cursor += size
        ws[kind] = (entries, align(cursor))
        out += emit_workspace(f"ws_{kind}", max(64, align(cursor)))
        proto = ([f"MemRef{len(p['shape'])} *" for p in g["parameters"]]
                 + ["MemRef2 *"] + ["MemRef1 *", "MemRef4 *", "MemRef4 *"] * layers
                 + [f"MemRef{len(e['shape'])} *" for e in g.get("workspace_inputs", [])])
        out.append(f'extern void _mlir_ciface_forward_{kind}(GraphResults *, '
                   + ', '.join(proto) + ');')
    if getattr(args, 'reference_arrays', None):
        total = args.prefill_len + args.decode_steps
        out += [f'#define REF_LENGTH {total}', f'#define REF_LOGITS_ROWS {args.decode_steps+1}',
                'extern const float model_reference_raw[];',
                'static int compare_reference(const GraphResults *, unsigned, const float *, int64_t);']
    out += [r"""
static void zero_bytes(void *p, size_t n) { memset(p, 0, n); }
/* Returned buffers may alias inputs or belong to the graph's scoped heap.
 * Preserve every layer before releasing that scope. Honor offsets/strides. */
static int retain_cache(float *dst, const MemRef4 *m) {
  if (!m->aligned || m->sizes[0] != 1 || m->sizes[1] != 8 ||
      m->sizes[2] != CAPACITY || m->sizes[3] != 128) return -1;
  const float *src = (const float *)m->aligned + m->offset;
  if (m->strides[3] == 1 && m->strides[2] == 128 &&
      m->strides[1] == CAPACITY * 128) {
    if (src != dst) nr_copy_bytes(dst, src, KV_ELEMENTS * sizeof(float));
  } else {
    for (unsigned h = 0; h < 8; ++h)
      for (unsigned t = 0; t < CAPACITY; ++t)
        for (unsigned d = 0; d < 128; ++d)
          dst[(h * CAPACITY + t) * 128 + d] =
              src[h * m->strides[1] + t * m->strides[2] + d * m->strides[3]];
  }
  return 0;
}
static void float_bits(float f) {
  union { float f; uint32_t u; } bits = {f}; nr_hex32(bits.u);
}
static uint64_t selection_cycles, cache_retention_cycles;
static int collect(GraphResults *r, unsigned position, unsigned *token,
                   float *score, int trace) {
  uint64_t selection_begin = nr_cycles();
  if (!r->logits.aligned || r->logits.sizes[0] != 1 ||
      r->logits.sizes[1] < 1 || r->logits.sizes[2] != VOCAB) return -1;
  const float *scores = (const float *)r->logits.aligned + r->logits.offset +
                        (r->logits.sizes[1]-1) * r->logits.strides[1];
  *token = 0; *score = scores[0];
  for (unsigned i = 0; i < VOCAB; ++i) {
    float value = scores[i * r->logits.strides[2]];
    if (!(value <= 3.402823466e38f && value >= -3.402823466e38f)) return -1;
    if (value > *score) { *score = value; *token = i; }
  }
  uint64_t retention_begin = nr_cycles();
  selection_cycles = retention_begin - selection_begin;
  for (unsigned l = 0; l < LAYERS; ++l) {
    if (retain_cache(k_cache_f + l * KV_ELEMENTS, &r->cache[l].key) ||
        retain_cache(v_cache_f + l * KV_ELEMENTS, &r->cache[l].value)) return -1;
  }
  cache_retention_cycles = nr_cycles() - retention_begin;
#ifdef REF_LENGTH
  int numeric_status = compare_reference(r, position, scores, r->logits.strides[2]);
#else
  int numeric_status = 0;
#endif
  if (trace) {
    nr_puts("[model] selected position="); nr_hex32(position);
    static const unsigned ids[] = {49000,374,264,3146,7407,304,4787,5159,11};
    for (unsigned i = 0; i < sizeof(ids)/sizeof(ids[0]); ++i) {
      nr_puts(" "); nr_hex32(ids[i]); nr_puts(":");
      float_bits(scores[ids[i] * r->logits.strides[2]]);
    }
    nr_puts("\r\n");
    for (unsigned l = 0; l < LAYERS; ++l) {
      nr_puts("[model] kv position="); nr_hex32(position);
      nr_puts(" layer="); nr_hex32(l);
      nr_puts(" k="); float_bits(k_cache_f[l * KV_ELEMENTS + position * 128]);
      nr_puts(" v="); float_bits(v_cache_f[l * KV_ELEMENTS + position * 128]);
      nr_puts("\r\n");
    }
  }
  return numeric_status;
}
static void reset_cache(void) {
  zero_bytes(k_cache_f, LAYERS * KV_ELEMENTS * sizeof(float));
  zero_bytes(v_cache_f, LAYERS * KV_ELEMENTS * sizeof(float));
}
"""]
    if getattr(args, 'reference_arrays', None):
        out += [f'static const float max_atol = {args.reference_atol:.9e}f;',
                f'static const float mean_atol = {args.reference_mean_atol:.9e}f;',
                r'''
static int report_error(const char *name, unsigned position, float max, float sum,
                        unsigned count) {
  float mean = sum / (float)count;
  nr_puts("[compare] "); nr_puts(name); nr_puts(" position="); nr_hex32(position);
  nr_puts(" count="); nr_hex32(count);
  nr_puts(" max_abs_bits="); float_bits(max);
  nr_puts(" mean_abs_bits="); float_bits(mean);
  int ok = max <= max_atol && mean <= mean_atol;
  nr_puts(ok ? " PASS\r\n" : " FAIL\r\n");
  return ok ? 0 : -1;
}
static int compare_reference(const GraphResults *r, unsigned position,
                             const float *scores, int64_t stride) {
  (void)r;
''', f'  unsigned row = position - {args.prefill_len-1};',
                r'''
  if (row >= REF_LOGITS_ROWS) return -1;
  float max = 0, sum = 0;
  for (unsigned i = 0; i < VOCAB; ++i) {
    float diff = scores[i * stride] - model_reference_raw[row * VOCAB + i];
    if (diff < 0) diff = -diff;
    if (!(diff <= 3.402823466e38f)) return -1;
    if (diff > max) max = diff; sum += diff;
  }
  int status = report_error("logits", position, max, sum, VOCAB);
  for (unsigned which = 0; which < 2; ++which) {
    const float *gold = model_reference_raw + REF_LOGITS_ROWS * VOCAB +
                        which * LAYERS * 8 * REF_LENGTH * 128;
    const float *actual = which ? v_cache_f : k_cache_f;
    max = 0; sum = 0;
    for (unsigned l = 0; l < LAYERS; ++l)
      for (unsigned h = 0; h < 8; ++h)
        for (unsigned t = 0; t <= position; ++t)
          for (unsigned d = 0; d < 128; ++d) {
            float diff = actual[(l*8+h)*CAPACITY*128+t*128+d] -
                         gold[(l*8+h)*REF_LENGTH*128+t*128+d];
            if (diff < 0) diff = -diff;
            if (!(diff <= 3.402823466e38f)) return -1;
            if (diff > max) max = diff; sum += diff;
          }
    status |= report_error(which ? "value_cache" : "key_cache", position, max, sum,
                           LAYERS * 8 * (position+1) * 128);
  }
  return status;
}
''']
    for kind, g in all_graphs:
        length = args.prefill_len if kind == "prefill" else 1
        out += [f'static int run_{kind}(unsigned position, unsigned *token,',
                '                      float *score, int trace) {',
                f'  if (position + {length} > CAPACITY) return -1;',
                '  uint64_t preparation_begin = nr_cycles();',
                '  *cache_position = position;',
                '  uintptr_t mark = nr_heap_mark();',
                '  GraphResults result;',
                f'  MemRef2 ids = make_2(input_ids, 1, {length});',
                '  MemRef1 pos = make_1(cache_position, 1);',
                '  MemRef4 k[LAYERS], v[LAYERS];',
                '  for (unsigned l = 0; l < LAYERS; ++l) {',
                '    k[l] = make_4(k_cache_f + l * KV_ELEMENTS, 1, 8, CAPACITY, 128);',
                '    v[l] = make_4(v_cache_f + l * KV_ELEMENTS, 1, 8, CAPACITY, 128);',
                '  }']
        params = []
        for i, p in enumerate(g["parameters"]):
            rank = len(p['shape'])
            out.append(f'  MemRef{rank} p{i} = make_{rank}(weight_arena_raw + '
                       f'{offsets[p["name"]]}, ' + ', '.join(map(str,p['shape'])) + ');')
            params.append(f'&p{i}')
        params += ['&ids']
        for l in range(layers):
            params += ['&pos', f'&k[{l}]', f'&v[{l}]']
        entries, _ = ws[kind]
        for i, e in enumerate(g.get("workspace_inputs", [])):
            offset, size = entries[e['name']]
            rank = len(e['shape'])
            out.append(f'  MemRef{rank} w{i} = make_{rank}(ws_{kind}_raw + {offset}, '
                       + ', '.join(map(str,e['shape'])) + ');')
            if e['name'].endswith('_acc'):
                out.append(f'  zero_bytes(ws_{kind}_raw + {offset}, {size});')
            elif (e.get('role') == 'cache_positions' or
                  e['name'].endswith('_position')):
                if e['dtype'] != 'TensorDType.Int32' or e['shape'] != [length]:
                    raise ValueError(f"unexpected cache position workspace {e}")
                # The graph's scalar RoPE position is i64. Attention/KV Triton
                # kernels separately consume a caller-owned i32 position per
                # query. It must be regenerated at EVERY graph invocation.
                out.append(f'  for (unsigned j = 0; j < {length}; ++j) '
                           f'((int32_t *)(ws_{kind}_raw + {offset}))[j] = position + j;')
            params.append(f'&w{i}')
        out += ['  uint64_t preparation_cycles = nr_cycles() - preparation_begin;',
                f'  if (trace) {{ nr_puts("[model] {kind} begin position=");',
                '    nr_hex32(position); nr_puts(" input_token=");',
                '    nr_hex32((unsigned)input_ids[0]); nr_puts("\\r\\n"); }']
        if getattr(args, 'profile_kernels', False):
            out.append('  qwen_profile_reset();')
        if getattr(args, 'intermediate_arrays', None):
            out.append(f'  qwen_intermediate_begin(position, {length});')
        out += [
                '  uint64_t begin = nr_cycles();',
                f'  _mlir_ciface_forward_{kind}(&result, ' + ', '.join(params) + ');',
                '  ame_fence();',
                '  uint64_t compute = nr_cycles() - begin;',
                f'  int status = collect(&result, position + {length} - 1, token, score, trace);',
                '  uintptr_t peak = nr_heap_mark();',
                '  nr_heap_reset(mark);',
                f'  if (trace) {{ nr_puts("[model] {kind} position="); nr_hex32(position);',
                '    nr_puts(" token="); nr_hex32(*token);',
                '    nr_puts(" logit_bits="); float_bits(*score);',
                '    nr_puts(" compute_cycles="); nr_hex64(compute);',
                '    nr_puts(" preparation_cycles="); nr_hex64(preparation_cycles);',
                '    nr_puts(" selection_cycles="); nr_hex64(selection_cycles);',
                '    nr_puts(" cache_retention_cycles="); nr_hex64(cache_retention_cycles);',
                '    nr_puts(" model_cycles=");',
                '    nr_hex64(preparation_cycles + compute + selection_cycles + cache_retention_cycles);',
                '    nr_puts(" total_with_uart_cycles="); nr_hex64(nr_cycles() - begin);',
                '    nr_puts(" scratch_bytes="); nr_hex64(peak - mark);',
                '    nr_puts("\\r\\n"); }']
        if getattr(args, 'intermediate_arrays', None):
            out.append('  status |= qwen_intermediate_end();')
        if getattr(args, 'profile_kernels', False):
            out.append(f'  if (trace) qwen_profile_report(position + {length} - 1);')
        out += ['  return status;', '}']
    if prompt is not None:
        out += fixed_prompt_source(args, prompt, tokenizer_bytes)
    elif args.interactive:
        out += [r"""
static void emit_uart(void *ctx, const uint8_t *bytes, size_t n) {
  (void)ctx;
  nr_write(bytes, n);
}
static int interactive(void) {
  QwenTokenizerResource resource;
""", f'  if (qwen_tokenizer_open(&resource, tokenizer_blob_raw, {tokenizer_bytes})) return 1;',
                r"""
  static uint8_t line[2048], prompt[8192];
  static uint32_t ids[2048];
  int skip_lf = 0;
  nr_puts("[model] interactive single-turn; /quit exits\r\n");
  for (;;) {
    nr_puts("\r\nqwen> ");
    size_t length = 0; int overflow = 0;
    for (;;) {
      int c = nr_getchar();
      if (c < 0) continue;
      if (skip_lf && c == '\n') { skip_lf = 0; continue; }
      skip_lf = 0;
      if (c == '\r' || c == '\n') { skip_lf = c == '\r'; break; }
      if (c == 4) return 0;
      if (c == 8 || c == 127) {
        if (length) { --length; while (length && (line[length] & 0xc0) == 0x80) --length; }
        continue;
      }
      if (length < sizeof(line)) line[length++] = (uint8_t)c;
      else overflow = 1;
    }
    if (overflow) { nr_puts("input too long; rejected\r\n"); continue; }
    if (length == 5 && !memcmp(line, "/quit", 5)) return 0;
    if (!length) continue;
    size_t bytes = 0, count = 0;
    if (qwen_chat_single_turn(prompt, sizeof(prompt), &bytes, 0, 0, 0,
                              line, length, 0) ||
        qwen_encode(&resource, prompt, bytes, ids, 2048, &count)) {
      nr_puts("tokenizer rejected input\r\n"); continue;
    }
    if (!count || count > CAPACITY) {
      nr_puts("prompt exceeds context; rejected\r\n"); continue;
    }
    nr_puts("[model] prompt token IDs:");
    for (size_t i = 0; i < count; ++i) { nr_puts(" "); nr_hex32(ids[i]); }
    nr_puts("\r\n");
    reset_cache();
    unsigned next = 0; float score = 0;
    for (unsigned i = 0; i < count; ++i) {
      input_ids[0] = ids[i];
      if (run_decode(i, &next, &score, 0)) return 1;
    }
    /* next already predicts the first generated token. Do not replay prompt[-1]. */
    unsigned position = (unsigned)count;
    QwenUtf8Decoder utf8 = {{0,0,0,0},0,0};
""", f'    for (unsigned step = 0; step < {args.max_new_tokens}; ++step) {{',
                '      if (next == 151643 || next == 151645) break;',
                '      if (qwen_decode_token(&resource, &utf8, next, 1, emit_uart, 0)) return 1;',
                f'      if (step + 1 == {args.max_new_tokens} || position >= CAPACITY) break;',
                '      input_ids[0] = next;',
                '      if (run_decode(position++, &next, &score, 0)) return 1;',
                '    }', '    qwen_decode_finish(&utf8, emit_uart, 0);',
                '    nr_puts("\\r\\n");', '  }', '}',
                'int launch(void) { return interactive(); }']
    else:
        out += ['int launch(void) {', '  reset_cache();',
                '  static const int64_t prompt[] = {' + ', '.join(map(str,args.prompt_ids)) + '};',
                '  for (unsigned i = 0; i < sizeof(prompt)/sizeof(prompt[0]); ++i) input_ids[i] = prompt[i];',
                '  unsigned token = 0; float score = 0;',
                '  if (run_prefill(0, &token, &score, 1)) return 1;']
        if dec_graph:
            out += [f'  for (unsigned step = 0; step < {args.decode_steps}; ++step) {{',
                    '    input_ids[0] = token;',
                    f'    if (run_decode({args.prefill_len} + step, &token, &score, 1)) return 1;',
                    '  }']
        out += ['  return 0;', '}']
    return '\n'.join(out) + '\n', {
        "interactive": bool(args.interactive), "layers": layers,
        "text_mode": "fixed-validation" if prompt is not None else ("interactive-generation" if args.interactive else "numeric-only"),
        "fixed_prompt": ({"utf8_hex": prompt.hex(), "sha256": hashlib.sha256(prompt).hexdigest(),
                          "bytes": len(prompt), "thinking": False, "expected_ids": args.prompt_ids,
                          "model_ids_source": "on-board qwen_encode output; expected IDs only compare",
                          "prefill_calls": 1, "decode_calls": args.decode_steps,
                          "predictions_recorded": 1 + args.decode_steps,
                          "eos_policy": "record and continue fixed validation trajectory",
                          "output_capacity": getattr(args, 'text_output_bytes', 65536)} if prompt is not None else None),
        "tokenizer_bytes": tokenizer_bytes, "parameters": len(graph['parameters']),
        "workspace_bytes": {k: v[1] for k,v in ws.items()},
        "kv_bytes": layers * kv_elements * 8, "segment_bytes": segment['bytes'],
        "abi": contract, "scratch_lifetime": "one graph call, after retaining all cache results",
        "known_copy_cost": "copy returned KV only when graph output does not alias persistent cache",
        "profile_kernels": getattr(args, 'profile_kernels', False),
        "profile_progress": getattr(args, 'profile_progress', False),
        "profile_probe": getattr(args, 'profile_probe', None),
        "cycle_scope": {"compute_cycles": "compiled graph plus final AME fence (and optional profiler overhead)",
                        "model_cycles": "per-call descriptor/workspace preparation + graph + token selection + cache retention; excludes validation and UART",
                        "excluded": "session cache reset, tokenizer, input forwarding, validation, UART, scoped-heap reset",
                        "diagnostic_override": "--profile-progress adds UART inside graph timing; --profile-probe also adds returned/synced UART inside the selected kernel timing; --intermediate-arrays adds comparison inside graph timing. These diagnostic timings are not model throughput."},
    }


def build(args, output):
    """Assemble, compile, link and audit -- the same chain the operator examples
    use, so the graph object gets the AME encoding and the fence insertion that
    the FPGA ELF audit requires."""
    llvm = args.repo_root / "llvm/build-2d26/bin"
    qwen = args.repo_root / "examples/FPGA-BOSCAME/qwen3-0.6b"
    common = args.repo_root / "examples/FPGA-BOSCAME/common"
    nr = common / "nr"
    tools = args.repo_root / "examples/FPGA-BOSCAME/tools"
    steps = {}
    commands = []

    def run(command, **kwargs):
        commands.append([str(c) for c in command])
        redirected = any(k in kwargs for k in ("stdin", "stdout", "stderr"))
        if redirected:
            result = subprocess.run([str(c) for c in command], **kwargs)
            result.stdout = result.stdout or ""
            result.stderr = result.stderr or ""
            return result
        return subprocess.run([str(c) for c in command], capture_output=True,
                              text=True, **kwargs)

    flags = ["--target=riscv64-unknown-elf", "-march=rv64gc_zicbom",
             "-mabi=lp64d", "-mcmodel=medany", "-O2", "-nostdlib", "-ffreestanding",
             "-fno-builtin", "-fno-pie", "-fno-vectorize", "-fno-slp-vectorize",
             "-ffp-contract=off", f"-I{qwen}", f"-I{nr}", f"-I{common}/uart",
             f"-I{Path(__file__).resolve().parent.parent / 'text'}"]
    if args.uart_probe:
        # the NH-side UART sample in the runtime
        flags.append("-DNR_UART_DEBUG")

    # graph IR -> assembly -> AME encoding -> fence insertion -> object
    assembly = output / "forward_prefill.s"
    result = run([llvm / "llc", args.graph_ir, "-O2", "-filetype=asm",
                  "-mtriple=riscv64", "-target-abi=lp64d",
                  "-mattr=+m,+a,+f,+d,+c,-v,+xboscame", "-code-model=medium",
                  "-o", assembly])
    if result.returncode:
        steps["llc"] = result.stderr[-800:]
        return {"status": "FAILED compiling the graph IR", "steps": steps}
    for tool, source_suffix, target in (("ame_to_word.py", "s", "encoded.s"),
                                        ("restrict_fpga_assembly.py",
                                         "encoded.s", "nr.S")):
        handle_in = (output / f"forward_prefill.{source_suffix}").open("rb")
        handle_out = (output / f"forward_prefill.{target}").open("wb")
        result = run([sys.executable, tools / tool], stdin=handle_in,
                     stdout=handle_out)
        handle_in.close(); handle_out.close()
        if result.returncode:
            steps[tool] = result.stderr[-800:]
            return {"status": f"FAILED in {tool}", "steps": steps}
    graph_object = output / "forward_prefill.nr.o"
    result = run([llvm / "clang", *flags, "-c", output / "forward_prefill.nr.S",
                  "-o", graph_object])
    if result.returncode:
        steps["graph_assemble"] = result.stderr[-1500:]
        return {"status": "FAILED assembling the graph", "steps": steps}

    objects = [graph_object]
    profile_flags = []
    if getattr(args, 'intermediate_arrays', None):
        from intermediate_probe import generate_intermediate
        probe_source, probe_asm, probe_flags, _ = generate_intermediate(args, output)
        for source in (probe_source, probe_asm):
            obj = output / (source.stem + '.o')
            result = run([llvm / 'clang', *flags, '-c', source, '-o', obj])
            if result.returncode:
                return {'status': 'FAILED compiling intermediate probe', 'error': result.stderr}
            objects.append(obj)
        profile_flags += probe_flags
    if getattr(args, 'profile_kernels', False):
        from kernel_profile import generate_profile
        profile_source, kernel_profile_flags = generate_profile(
            args.adapters, output, progress=getattr(args, 'profile_progress', False),
            probe=getattr(args, 'profile_probe', None))
        profile_flags += kernel_profile_flags
        obj = output / 'kernel-profile.o'
        result = run([llvm / 'clang', *flags, '-c', profile_source, '-o', obj])
        if result.returncode:
            return {'status': 'FAILED compiling kernel profiling glue', 'error': result.stderr}
        objects.append(obj)
    if args.reference_arrays:
        obj = output / 'numeric-reference.o'
        result = run([llvm / 'clang', *flags, '-c', output / 'numeric-reference.S', '-o', obj])
        if result.returncode:
            return {"status": "FAILED assembling numeric oracle", "error": result.stderr}
        objects.append(obj)
    if args.interactive or fixed_prompt_bytes(args) is not None:
        # the tokenizer encoder, its resource reader and the generated tables
        text = Path(__file__).resolve().parent.parent / "text"
        for name in ("tokenizer_encode.c", "tokenizer_resource.c",
                     "unicode_tables.c"):
            obj = output / (Path(name).stem + ".o")
            result = run([llvm / "clang", *flags, "-c", text / name, "-o", obj])
            if result.returncode:
                steps[Path(name).stem] = result.stderr[-1500:]
                return {"status": f"FAILED compiling {name}", "steps": steps}
            objects.append(obj)
    if args.decode_ir is not None:
        dec_assembly = output / "forward_decode.s"
        result = run([llvm / "llc", args.decode_ir, "-O2", "-filetype=asm",
                      "-mtriple=riscv64", "-target-abi=lp64d",
                      "-mattr=+m,+a,+f,+d,+c,-v,+xboscame", "-code-model=medium",
                      "-o", dec_assembly])
        if result.returncode:
            steps["decode_llc"] = result.stderr[-800:]
            return {"status": "FAILED compiling the decode IR", "steps": steps}
        for tool, source_suffix, target in (("ame_to_word.py", "s", "encoded.s"),
                                            ("restrict_fpga_assembly.py",
                                             "encoded.s", "nr.S")):
            handle_in = (output / f"forward_decode.{source_suffix}").open("rb")
            handle_out = (output / f"forward_decode.{target}").open("wb")
            result = run([sys.executable, tools / tool], stdin=handle_in,
                         stdout=handle_out)
            handle_in.close(); handle_out.close()
            if result.returncode:
                steps[f"decode_{tool}"] = result.stderr[-800:]
                return {"status": f"FAILED in {tool} for decode", "steps": steps}
        raw_dec = output / "forward_decode.raw.o"
        result = run([llvm / "clang", *flags, "-c",
                      output / "forward_decode.nr.S", "-o", raw_dec])
        if result.returncode:
            steps["decode_assemble"] = result.stderr[-1500:]
            return {"status": "FAILED assembling the decode graph", "steps": steps}
        # Each graph lowerer emits its own dealloc_helper (from
        # bufferization-lower-deallocations), so two entry points compiled
        # separately collide at link time. Renaming the decode copy is enough:
        # the helper is module-local in purpose and objcopy rewrites the
        # references inside that object too.
        dec_object = output / "forward_decode.nr.o"
        result = run([llvm / "llvm-objcopy",
                      "--redefine-sym", "dealloc_helper=dealloc_helper_decode",
                      "--redefine-sym",
                      "_mlir_ciface_dealloc_helper=_mlir_ciface_dealloc_helper_decode",
                      raw_dec, dec_object])
        if result.returncode or not dec_object.is_file():
            steps["decode_rename"] = result.stderr[-1500:]
            return {"status": "FAILED renaming the decode helper", "steps": steps}
        objects.append(dec_object)
    for name, source, extra in (("model_main", output / "model_main.c", []),
                                ("adapters", args.adapters, []),
                                ("crt", nr / "crt.S", []),
                                ("nr_runtime", nr / "nr_runtime.c", []),
                                ("nr_math", nr / "nr_math.c", []),
                                ("ame_sync", nr / "ame_sync.c", []),
                                ("nr_copy", nr / "nr_copy.S",
                                 ["-march=rv64gcv_zicbom"])):
        source_flags = flags
        if extra:
            source_flags = [extra[0] if f.startswith("-march=") else f
                            for f in flags]
        obj = output / f"{name}.o"
        result = run([llvm / "clang", *source_flags, "-c", source, "-o", obj])
        if result.returncode:
            steps[name] = result.stderr[-1500:]
            return {"status": f"FAILED compiling {name}", "steps": steps}
        objects.append(obj)

    elf = output / "qwen_model.elf"
    linker = resolve_linker(getattr(args, "linker", None), llvm)
    result = run([linker["path"], "-m", "elf64lriscv", "--gc-sections",
                  "-T", nr / "nr.ld", f"-Map={output / 'qwen_model.map'}",
                  "-o", elf, *profile_flags, *objects, args.archive])
    if result.returncode or not elf.is_file():
        steps["link"] = result.stderr[-3000:]
        return {"status": "FAILED linking", "steps": steps}

    binary = output / "qwen_model.bin"
    converted = run([llvm / "llvm-objcopy", "-O", "binary", elf, binary])
    if converted.returncode or not binary.is_file():
        return {"status": "FAILED converting ELF to bin", "error": converted.stderr,
                "commands": commands}
    audit = run([sys.executable,
                 args.repo_root / "examples/FPGA-BOSCAME/tools/check_nr_elf.py",
                 elf, "--objdump", llvm / "llvm-objdump",
                 "--output", output / "elf-audit.json"])
    undefined = run([llvm / "llvm-nm", "--undefined-only", elf])
    sizes = run([llvm / "llvm-size", elf])
    report = {
        "status": "PASS" if audit.returncode == 0 else "built but the audit failed",
        "steps": steps,
        "elf": {"path": str(elf), "bytes": elf.stat().st_size},
        "bin": {"path": str(binary), "bytes": binary.stat().st_size},
        "undefined_symbols": sorted({l.split()[-1]
                                     for l in undefined.stdout.splitlines()
                                     if l.strip()}),
        "sizes": sizes.stdout,
        "audit_stdout": audit.stdout[-800:],
        "commands": commands,
        "linker": linker,
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [args.graph_ir, args.adapters, args.archive,
                                   Path(__file__), nr / "nr_runtime.c", nr / "nr.ld",
                                   nr / "ame_sync.c", nr / "nr_runtime.h",
                                   nr / "nr_math.c", nr / "nr_copy.S", nr / "crt.S"]
                         + ([args.decode_ir] if args.decode_ir else [])},
    }
    (output / "image.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("steps", "sizes")}, indent=2))
    return 0 if report["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--linker", type=Path, help="LLD >= 20 executable; resolved path/version/hash recorded")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--segment", type=Path, required=True)
    parser.add_argument("--graph-ir", type=Path, required=True)
    parser.add_argument("--decode-ir", type=Path, default=None,
                        help="forward_decode LLVM IR; adds 8 decode steps")
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--adapters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph", default="prefill")
    parser.add_argument("--layers", type=int, default=28)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--cache-len", type=int, default=128)
    parser.add_argument("--prefill-len", type=int, default=16)
    parser.add_argument("--prompt-ids", default="151644,872,198,3838,374,9625,30,"
                                               "151645,198,151644,77091,198,"
                                               "151667,271,151668,271")
    parser.add_argument("--debug-linear", action="store_true",
                        help="print one linear's activation scale, weight scale, "
                             "accumulator and output over UART, to locate a "
                             "bare-metal-only numeric defect")
    parser.add_argument("--interactive", action="store_true",
                        help="Stage E firmware: read a line from UART, tokenise "
                             "it on the board, feed it through the decode graph "
                             "one token at a time, generate, and print the text")
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument("--prompt-text", help="embed raw UTF-8 user text; board templates/tokenizes it, "
                        "checks --prompt-ids, then runs one prefill and exactly --decode-steps calls")
    prompt.add_argument("--prompt-file", type=Path, help="raw UTF-8 prompt file; bytes including trailing LF are preserved")
    parser.add_argument("--text-output-bytes", type=int, default=65536,
                        help="fixed-validation decoded UTF-8 output capacity; overflow is an error")
    parser.add_argument("--tokenizer-blob", type=Path, default=None,
                        help="packed tokenizer resource, loaded as its own DDR "
                             "segment")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--reference-arrays", type=Path,
                        help="optional independent numeric oracle NPZ, validation firmware only")
    parser.add_argument("--reference-metadata", type=Path,
                        help="reference metadata (default: quant-reference.json beside NPZ); "
                             "must match prompt, dimensions, trajectory and NR arithmetic")
    parser.add_argument("--reference-atol", type=float, default=1e-3)
    parser.add_argument("--reference-mean-atol", type=float, default=1e-4)
    parser.add_argument("--profile-kernels", action="store_true",
                        help="measure actual Triton kernel calls with optional linker wrappers; "
                             "adds fences/bookkeeping, reports counters after each graph call")
    parser.add_argument("--profile-progress", action="store_true",
                        help="diagnostic UART begin/end records for each kernel; requires --profile-kernels")
    parser.add_argument("--profile-probe", metavar="SYMBOL:CALL_INDEX",
                        help="selected kernel occurrence: descriptor, returned-before-fence and synced-after-fence UART; "
                             "requires --profile-kernels and --profile-progress; decimal or 0x index resets per graph")
    parser.add_argument('--intermediate-arrays', type=Path,
                        help='optional one-layer independent Stage B intermediate NPZ; instrumentation only')
    parser.add_argument('--intermediate-layout', type=Path,
                        help='value-matched weight-layout.json mapping graph parameters to checkpoint names')
    parser.add_argument('--intermediate-graph-dir', type=Path,
                        help='directory containing actual subgraph0_prefill/decode.triton.mlir')
    parser.add_argument('--intermediate-atol', type=float, default=1e-3)
    parser.add_argument('--intermediate-mean-atol', type=float, default=1e-4)
    parser.add_argument('--intermediate-progress', action='store_true',
                        help='diagnostic entry/rank/descriptor UART before each intermediate comparison')
    parser.add_argument("--uart-probe", action="store_true",
                        help="diagnostic firmware: print the UART register block "
                             "repeatedly so a sent line shows which offset carries "
                             "the received character")
    parser.add_argument("--dry-run", action="store_true",
                        help="generate and report sizes without compiling")
    args = parser.parse_args()
    args.prompt_ids = [int(v) for v in args.prompt_ids.split(",") if v.strip()]

    report = json.loads(args.report.read_text())
    segment = json.loads(args.segment.read_text())
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    reference = prepare_reference(args, output)
    source, summary = generate(report, segment, args)
    summary['numeric_reference'] = reference
    (output / "model_main.c").write_text(source)
    (output / "w8a8-image-plan.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return 0
    result = build(args, output)
    if isinstance(result, dict):
        (output / "image.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), file=sys.stderr)
        return 1
    return result


if __name__ == "__main__":
    sys.exit(main())
