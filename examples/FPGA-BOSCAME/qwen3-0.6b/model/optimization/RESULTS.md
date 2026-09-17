# AME v0.5 optimization validation

2026-09-17. Every board result below used FPGA5 through the reconnectable public
runner. DDR readback and exact uploaded image hashes are part of the archived
verification. All modes remain opt-in; original defaults and GEM5 are preserved.

Latest full-model result: capacity128 run `run-2c2dd9c210c1492a` passed complete
28-layer fixed16+8 numerical acceptance. See [CONTEXT128.md](CONTEXT128.md) and
`../validation/board/ame-v05/model-28l-cap128/verification.json`.
All 27 logits/KV checks passed; KV errors were zero, maximum logits error was
9.536743e-7 in decode position16 only. Prefill graph took 5,637,966,351 cycles;
mean decode graph took 1,129,291,836.25 cycles (76.59 s at configured14.7456MHz).
Mean full model step excluding validation/UART was80.01s; entire launch with
validation/text/output was1050.62s. No per-kernel profiler was used.
The earlier capacity512 profiled runs stalled at different boundaries; capacity
and instrumentation both changed, so this success does not identify their cause.
References below to outstanding28-layer acceptance describe the prior one-layer
optimization checkpoint, not this latest completed capacity128 run.

## Measured results

| Experiment | Baseline cycles | Optimized cycles | Speedup | Evidence |
|---|---:|---:|---:|---|
| Dequant 1×151936, scalar vs RVV | 37,566,389 | 4,762,838 | 7.887× | `../validation/board/ame-v05/dequant-1x151936/verification.json` |
| Dequant 16×1024, scalar vs RVV | 4,044,689 | 513,725 | 7.873× | `../validation/board/ame-v05/dequant-16x1024/verification.json` |
| INT8 lm_head 1×151936×1024, fixed vs direct GPR | 158,512,001.5 | 107,802,367 | 1.470× | `../validation/board/ame-v05/ame-ab-1x151936x1024/verification.json` |
| INT8 lm_head direct, N16 vs N32 | 107,846,633.0 | 78,223,503.5 | 1.379× | `../validation/board/ame-v05/ame-ab-n16-n32/verification.json` |
| INT8 lm_head direct, N32 vs N64 | 78,123,299.5 | 68,235,289.5 | 1.145× | `../validation/board/ame-v05/ame-ab-n32-n64/verification.json` |
| INT8 tail 3×19×70 direct, N16 vs N64 | 409,252.0 | 687,872.5 | 0.595× | `../validation/board/ame-v05/ame-ab-tail-n16-n64/verification.json` |
| INT8 prefill 16×1024×1024 direct, N16 vs N64 | 1,824,274.5 | 1,466,121.0 | 1.244× | `../validation/board/ame-v05/ame-ab-m16-n16-n64/verification.json` |
| INT8 lm_head direct N64, adjacent-fence sharing | 68,236,713.5 | 67,871,936.0 | 1.005× | `../validation/board/ame-v05/ame-ab-n64-coalesce/verification.json` |

Numbers are medians of alternating runs (3 rounds for dequant, 2 for full-vocab
AME, 16 for tail shape). N64 is slower on the small 3×19×70 tail because padded
Triton tiles do extra work; N64 is selected for aligned model shapes, not as a
universal default. Dequant timing covers adapter+kernel. AME covers adapter+kernel+existing
post-call `ame_fence`; pre-sync is recorded separately. Preparation and independent
oracle checks are excluded. AME tests use nonzero C and verify accumulation,
full outputs and guards; large B is sampled in every row, not exhaustively checked.

The high-GPR probe (`../validation/board/ame-v05/ame-gpr-probe/verification.json`) passed four changed-input
rounds: x18..x31, tile rd==rs1, padded byte strides, signed INT8, nonzero INT32 C,
and buffer/guard preservation. This does not claim arbitrary cache coherence.

## Model regression: dequantization only

Only lm_head dequantization was replaced for the first model regression. The
same Buddy graph, quantized weights, full vocabulary, 1 layer, fixed raw text,
16-token prefill and eight successive decode positions 16..23 were used.
`../validation/board/ame-v05/model-1l-dequant/verification.json` reports
MODEL_RUN_NUMERIC_PASS: 414 selected intermediate checks and 27 full logits/KV
checks, all zero max/mean absolute error. Board tokenizer and incremental text
decode matched the official resources. A 1-layer model's generated text is not
an acceptance result for the complete 28-layer language model.

With the same selected-intermediate instrumentation, mean decode graph cycles
changed from 310,473,305.875 to 278,973,556.125 (10.15% reduction). Total launch
cycles changed from 3,326,559,996 to 3,042,674,448. At the configured 14.7456 MHz
these mean decode figures correspond to 21.06 s vs 18.92 s, including diagnostic
work; the clock frequency has not been measured independently and these are
not uninstrumented throughput measurements. Exact per-stage records and UART
hashes are in `../validation/board/ame-v05/model-dequant-comparison.json`.

## Model regression: combined optimizations

The 18-kernel combined variant completed **MODEL_RUN_NUMERIC_PASS** on FPGA5,
run `run-7df735200fa54dab`. Exact image/weights/tokenizer readbacks passed.
`../validation/board/ame-v05/model-1l-combined/verification.json` establishes:

- Complete 1-layer prefill 16 + successive decode 8, full 151936 vocabulary.
- The same token trajectory as the independent quantized reference.
- 414 selected intermediate comparisons and 27 full-vocabulary logits/effective
  KV comparisons: all max/mean absolute errors zero.
- Board-side fixed-prompt tokenizer and incremental text decoding PASS.
- All 46 static-library kernels retain the expected symbols and object/IR
  evidence; exactly 11 AME linears and 7 dequantizers changed. Runtime and test
  launch/oracle objects are not library members.

| Diagnostic timing | Original | Combined | Speedup |
|---|---:|---:|---:|
| Prefill compute cycles | 660,295,789 | 480,831,987 | 1.373× |
| Mean decode compute cycles | 310,473,305.875 | 175,380,822 | 1.770× |
| Entire launch cycles | 3,326,559,996 | 2,066,624,765 | 1.610× |

At configured 14.7456 MHz, mean decode is 21.06 s → 11.89 s and entire launch is
225.60 s → 140.15 s. Both images retain the same selected intermediate instrumentation;
these are diagnostic comparisons, not uninstrumented throughput measurements.
Exact stage records are in `../validation/board/ame-v05/model-combined-comparison.json`.
The combined binary is `build/ame-v05/model-combined/prepared/image.bin`, hash
`44899af4532d9e051198e5301dc65d3bdb444e73341a124990a9eaf5b0da7e2c`.

This completes the four optimization experiments and the 1-layer numerical
regression. Complete 28-layer FPGA numerical acceptance remains outstanding;
this experiment makes no claim about its execution time or completion.

## Build switches and semantics

- `QWEN_TRITON_DEQUANT=rvv`: genuine row-based Triton kernel, elementwise fusion,
  Buddy `--vectorize-dequantize`, then RVV sitofp/multiply. FP32 order remains
  `(float(acc) * row_scale) * column_scale`, no additional fast-math.
- `AME_GPR_MODE=direct`: encode the allocated GPRs for tile/MLS. Keep the
  existing msettype preservation and all fences.
- `QWEN_TRITON_AME_N=32|64`: change Triton BN and the Buddy NR `nr-tile-n`
  together. M16/K64 hardware limits, physical B[N,K], partial tiles and the
  full-K accumulator remain explicit. Default N16 is unchanged.
- `NR_COALESCE_FENCES=1`: only share adjacent identical `fence rw,rw` in one
  basic block. Labels, directives and other instructions break sharing. Every
  AME and vector-memory instruction retains adjacent fences; ELF auditing is
  unchanged. No msettype, synthetic resync or cache-maintenance removal.

Use a fresh `QWEN_TRITON_BUILD_ROOT` for each variant and pass
`QWEN_MAKE_FLAGS='RISCV_LD=/usr/bin/ld.lld-20'`. Do not rebuild recorded inputs
in place. Static library members are only compiled kernel and ABI adapter
objects; test launch/oracles and runtime are linked separately.

## Compiler checks

The opt-in dequant pass rejects unsupported dtype/layout/reduction/partial
alias cases and includes exact-bit host execution for lengths 0..35 with
dynamic offset (630 values). N16/N32/N64 lowering tests cover tails, physical
and transposed/padded B, full K, invalid options, and unchanged GEM5 examples.
Encoder/ELF regression checks cover full five-bit GPR fields and fence sharing.

N32/N64, direct GPR and adjacent-fence sharing have independent board A/B
records. The 3×19×70 N64 sharing stress passed 32 alternating rounds with full
input and output checks (`ame-ab-tail-coalesce/verification.json`). The exact
18 model specializations (11 AME and 7 RVV dequant) also passed the production
launch oracles with zero reported errors; see
`../validation/board/ame-v05/optimized-model-kernels/verification.json`.
Combined 1-layer acceptance passed as recorded above. Modes stay opt-in;
complete 28-layer acceptance remains separate.
