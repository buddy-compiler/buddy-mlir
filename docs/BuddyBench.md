# buddy-bench

`buddy-bench` measures RAX model packages. It loads a `.rax` through the same
Buddy Runtime / `InferenceRunner` path as `buddy-cli` and reports results in a
`llama-bench` compatible layout.

```
buddy-bench --model <model.rax> [options]
```

## Benchmark modes

| Mode | CLI | Reports |
|------|-----|---------|
| Prompt processing (prefill) | `-p 512 -n 0` | prompt tokens, prefill time and tokens/s |
| Token generation (decode) | `-p 0 -n 128` | generated tokens, decode tokens/s |
| Combined prefill + decode | `-pg 512,128` | both, plus total time |
| Generic forward | `--mode forward` | forward latency, throughput |

`-p`, `-n` and `--pg` are independent selectors, exactly as in `llama-bench`:
`-p 128,256 -n 32,64` runs four rows. Defaults are `-p 512 -n 128`, and `--pg`
adds to them rather than replacing them. `--prompt` supplies the token content,
truncated or padded (with a benign newline token) to each requested length.
Decoding stops at EOS unless `--ignore-eos` is given. Only `--batch-size 1` is
accepted.

## Repetitions and statistics

`-r N` (default 5) repeats each configuration. A warm-up pass runs first on the
same loaded session and is excluded from the results; `--no-warmup` skips it.
The model is loaded once per sweep. Throughput is computed per repetition and
then averaged, so `n` and the seconds in tokens/s always come from the same run.

`pp` and `pg` rows report prefill + decode; a `tg` row reports decode alone and
its `prefill_tps` is `0`. Prefill throughput divides by the positions the model
actually computed: a tiered KV cache rounds `-p 100` up to its next tier, and
the reported tokens/s reflects that work, not the requested length.

## Parameter sweeps

Comma-separated lists and `llama-bench`-style ranges are supported:

```
buddy-bench --model m.rax -p 128,256,512 -n 32,64,128
buddy-bench --model m.rax -p 32..128:32 -n 16..64:16
buddy-bench --model m.rax -p 0 -n 0 -pg 128,32 -pg 512,128
```

## Output formats

`-o md` (default), `-o csv`, `-o json` and `-o jsonl`. `json` emits
`{"env": {...}, "results": [...]}`; `jsonl` emits one leading env line followed
by one self-contained JSON object per measurement. Every run also prints the
recorded runtime configuration to stderr: CPU model, available and used cores,
affinity, NUMA, ISA, precision, backend, device, batch size, buddy-mlir version
and the `.rax` path, size and mtime, so a result can be reproduced and traced
back to the package it came from.

## Affinity and NUMA

```
buddy-bench --model m.rax --cpus 0-47 --numa 0,1,2,3 -n 64
```

`--cpus` pins with `taskset`; `--numa` is shorthand for `--numa-cpubind` +
`--numa-interleave`. Malformed specs and bindings the OS rejects stop the run
instead of silently benchmarking a different placement.

## Examples

```
buddy-bench --model deepseek_r1.rax                      # pp512 and tg128
buddy-bench --model deepseek_r1.rax -p 0 -n 128 -o json  # decode only
buddy-bench --model deepseek_r1.rax -p 0 -n 0 -pg 512,128 -r 3 -o csv
buddy-bench --model deepseek_r1.rax --mode forward -p 128
```

## Adding a benchmark to a model

`buddy-bench` talks to models only through the `InferenceRunner` interface (see
`runtime/include/buddy/runtime/core/InferenceRunner.h`): it `dlopen`s the
runner plugin named by the manifest and calls `runBench(cfg, warmupReps, reps,
results)`. A non-LLM model implements that as its own forward pass, so
`--mode forward` extends to vision, audio, embedding and scientific models.

Every runner plugin exports `buddy_inference_runner_capabilities_v1()` with its
capability bits (`kRunnerCapBench` when it implements `runBench`, `0`
otherwise); `buddy-bench` rejects a plugin that predates the export. After
changing `InferenceRunner.h`, rebuild the runner and repack the `.rax`.
