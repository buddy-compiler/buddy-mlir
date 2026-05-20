# Layer Partitioning

This document describes the layer partitioning workflow. It compiles smaller
MLIR fragments in parallel so the slowest model compile stages can use many
cores.

The current implementation and validation use DeepSeek R1 as the worked
example. Other models can use the same high-level flow, but they need their own
partition strategy, runtime ABI checks, and correctness tests before being
enabled by default.

## Environment

Run from the repository root and activate the Python environment used for model
import:

```bash
cd buddy-mlir
conda activate buddy
```

## Default Build Integration

Layer partitioning is controlled by the CMake option
`BUDDY_MODEL_LAYER_PARTITION`, which defaults to `ON` for supported models. The
normal README build therefore uses the validated partitioned path: fine-grained
prefill partitions and coarser Pow-boundary decode partitions. The finer
DeepSeek decode split can still be selected for experiments with
`BUDDY_DSR1_DECODE_SPLIT=fine`, but it is not the default because it changed
later greedy tokens during validation.

The default build command is:

```bash
python3 tools/buddy-codegen/build_model.py \
  --spec models/deepseek_r1/specs/f32.json \
  --build-dir build
```

To use the original whole-graph compile path, disable the option explicitly:

```bash
python3 tools/buddy-codegen/build_model.py \
  --spec models/deepseek_r1/specs/f32.json \
  --build-dir build \
  --cmake-args=-DBUDDY_MODEL_LAYER_PARTITION=OFF
```

The layer-partitioned path is automatically disabled for tiered KV cache builds,
RVV cross-compilation, and pre-generated MLIR directories that do not contain a
`layer_partitioned/partition_manifest.json` file.

## DeepSeek R1 Example

The commands below use the f32 DeepSeek R1 Distill Qwen 1.5B configuration as a
manual, decomposed version of the default flow. They are useful for measuring
baseline versus partitioned compile time or debugging the generated artifacts.

### Generate Config And MLIR

Generate a config from the f32 spec, then import the model while emitting the
extra `layer_partitioned` MLIR directory:

```bash
python3 tools/buddy-codegen/gen_config.py \
  --spec models/deepseek_r1/specs/f32.json \
  -o /tmp/deepseek_r1_f32_config.json

python3 tools/buddy-codegen/import_model.py \
  --config /tmp/deepseek_r1_f32_config.json \
  --output-dir /tmp/deepseek_r1_codegen_exp \
  --experimental-layer-partitioned \
  --skip-weights
```

The partitioned directory contains:

- `subgraph0_prefill<N>.mlir` and `subgraph0_decode<N>.mlir`: per-partition
  kernels. The validated DeepSeek default links both prefill and coarse decode
  partitions.
- `forward_prefill.mlir` and `forward_decode.mlir`: combined public entry
  points with the same runtime ABI as the original model.
- `forward_prefill<N>.mlir` and `forward_decode<N>.mlir`: debug wrappers for
  individual partitions.
- `partition_manifest.json`: a summary of partition counts and generated
  files.

### Compile Baseline And Partitioned MLIR

Compile the original full graphs as the baseline:

```bash
python3 tools/buddy-codegen/compile_pipeline.py \
  --config /tmp/deepseek_r1_f32_config.json \
  --compile-all \
  --mlir-dir /tmp/deepseek_r1_codegen_exp \
  --output-dir /tmp/deepseek_r1_codegen_exp/obj_baseline \
  --buddy-opt build/bin/buddy-opt \
  --llvm-tools-dir llvm/build/bin \
  -j 4
```

Compile the validated partitioned path in parallel:

```bash
python3 tools/buddy-codegen/compile_pipeline.py \
  --config /tmp/deepseek_r1_f32_config.json \
  --compile-partitioned \
  --mlir-dir /tmp/deepseek_r1_codegen_exp/layer_partitioned \
  --output-dir /tmp/deepseek_r1_codegen_exp/obj_partitioned \
  --buddy-opt build/bin/buddy-opt \
  --llvm-tools-dir llvm/build/bin \
  -j 48
```

The partitioned compile path rewrites very large all-zero tensor constants to
`tensor.empty` plus `linalg.fill` before lowering. This avoids embedding large
zero matrices into `.rodata`, which can otherwise create oversized object files
and linker relocation overflows.

### Verify Partition Structure

Before linking or running the model, verify that partitioning preserves graph
coverage and dependency order:

```bash
python3 tools/buddy-codegen/verify_layer_partition.py \
  --config /tmp/deepseek_r1_f32_config.json \
  --report /tmp/deepseek_r1_codegen_exp/layer_partition_verify.json
```

The verifier checks that every non-placeholder, non-output operation is present
exactly once, the concatenated partition order matches the original operation
order, and no producer partition appears after its consumer.

The command exits non-zero when a structural check fails. With `--report`, it
also writes a JSON summary containing the prefill/decode partition counts,
missing operations, duplicate operations, operation-order status, and
dependency-order violations. This is useful as a quick CI or local development
guard after changing partition strategies.

This check is structural only. It does not execute the compiled model or prove
that the generated tokens are unchanged. Runtime validation should still compare
greedy outputs or logits against the unpartitioned whole-graph model.

### Link Runtime Library

The CMake build links the partitioned runtime library automatically. For manual
runtime testing, link all `subgraph*.o` partition kernels plus the combined
public entry points. Do not link `forward_prefill<N>.o` or
`forward_decode<N>.o`; they are debug wrappers and are not needed by the
runtime path.

## Adapting To Other Models

To apply this flow to another model, add a model-specific split strategy under
`models/<model_family>/codegen/partition_strategy.py`. The generic importer
loads this module from `config["model_family"]` and calls
`layer_split_strategy(kind)` for `prefill` and `decode`. Validate the generated
public entry points against that model's runtime ABI. In practice this means
checking:

- partition boundaries match the model's layer structure;
- all cross-partition values are represented as explicit subgraph inputs and
  outputs;
- the combined `forward_*` functions keep the same signature and result order
  as the original runtime entry points;
- temporary tensors that cross partition boundaries have correct ownership and
  lifetime after bufferization;
- greedy inference or logits-level tests match the original unpartitioned
  model.

The `verify_layer_partition.py` script checks structural graph coverage and
dependency ordering, but it does not replace model-specific runtime validation.

## Observed DeepSeek R1 Results

On the f32 DeepSeek R1 Distill Qwen 1.5B experiment:

- Baseline MLIR compile time: about 157 seconds total on the test machine
  (`subgraph_prefill.o` about 134 seconds, `subgraph_decode.o` about 20
  seconds).
- Validated full partitioned compile time with 48 jobs: about 16 seconds total.
- Validated partition counts: 212 prefill partitions and 58 coarse decode
  partitions.
- Partition verifier result: no missing operations, no duplicate operations,
  and no dependency-order violations.
- Runtime validation: exact prefill logits and exact decode logits for 16
  greedy decode steps on the tested `Hi` and joke prompts.
- Fine-grained decode partitioning generated 169 decode partitions but changed
  later greedy tokens, so the default uses coarser Pow-boundary decode
  partitions.

These numbers are machine- and job-count-dependent, but they show the expected
benefit of compiling smaller graph fragments in parallel.
