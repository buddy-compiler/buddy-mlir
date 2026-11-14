# Methodology

## Profiling a model run

```bash
perf record -F 4000 -g -o perf.data <command>
perf report -i perf.data --no-children --sort sym -g none -n | head -60
```

- `-n` gives sample counts. Wall time of a phase ~= samples of its symbols /
  (frequency x busy threads). Drop the symbols of setup work (tokenizer and
  vocabulary hash tables, weight loading) before computing shares.
- Generated kernels show up by name (the kernel's `__tile` function, its
  `..omp_par` parallel body). Thread-pool symbols (`__kmpc_barrier`, `worker`) are idle
  threads: they mean imbalance or a serial section, not work.
- A spinning barrier is not always lost time. If the remaining threads
  already saturate DRAM, idle threads do not slow a bandwidth-bound kernel.

## Upper bounds

- Bandwidth-bound: bytes moved / measured bandwidth (measure the machine's
  read bandwidth with all cores; do not use the datasheet number).
  Example: LLM decode reads every weight once per token.
- Latency-bound loop: count the instructions on the critical path per
  iteration, times their measured latency.
- Effective rate of a kernel: bytes / time, compared to the machine peak,
  tells whether it is already at the bound.
- Turning a part's floor into a bound on the whole: see "Upper bound
  before work" in `SKILL.md` (saved fraction = share x (1 - floor / part)).

## Microbenchmarks

When the profile does not explain the time, measure the pieces:

1. **One kernel, isolated**: call the generated kernel from a small C driver
   with several copies of its inputs (larger than the last-level cache) so
   that each call streams from DRAM like in the model; report the best of N
   runs and bytes / time. Vary one parameter at a time (rows, block size,
   work split).
2. **Single instructions**: a loop of 16 independent instances (throughput)
   and 16 dependent ones (latency) per instruction, LMUL and element width.
   Assembly in a `.S` file so that the compiler cannot change it.
3. **Bit-exactness in the same driver**: write each variant's output to a
   file and `cmp` against the baseline.

Generators for these on the SpacemiT K3 are in
`hardware-targets/scripts/` and `llm-inference-optimization/scripts/`; copy
them as templates for another target.

## Reading the generated code

```bash
llvm-objdump -d --mattr=+v,+zfh,+zvfh kernels.o > k.asm
awk '/<my_kernel..omp_par>:/{p=1;next} p&&/>:$/&&!/Lpcrel/{p=0} p' k.asm \
  | awk '{print $3}' | sort | uniq -c | sort -rn | head -30
```

The instruction histogram of the hot function shows spills (whole-register
stores / reloads such as `vs4r.v` / `vl4r.v` against `sp`), unexpected
expensive instructions (`vrgather`, `vslide*` at high LMUL), and scalar
overhead. Compare it with what the source asked for.
