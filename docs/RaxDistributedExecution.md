# RAX Distributed Execution

This document describes distributed execution support in RAX, including
collective communication, Transformer parallel planning and rank-local
materialization, and the currently validated DeepSeek R1 tensor-parallel
example.

The distributed path extends the existing RAX execution model rather than
introducing a model-specific communication runtime. The frontend derives
rank-local computation and communication from tensor layouts, RHAL represents
the resulting ordered schedule, and the RAX runtime delegates collective
operations to a communicator backend.

DeepSeek R1 with tensor parallelism of size 2 is the current end-to-end
integration used to validate this path.

## Overview

The distributed compilation and execution flow is:

```text
PyTorch / FX Graph
        │
        ▼
Transformer structure and template analysis
        │
        ▼
TransformerParallelPlan
  - parameter shards
  - value layouts
  - operation rewrites
  - collective boundaries
  - compute segments
        │
        ▼
ParallelTemplatePartitionedGraphDriver
  - rank-local subgraphs
  - rank-local wrappers
  - rank parameter packs
  - ordered runtime plans
        │
        ▼
rankN_forward_prefill.json
rankN_forward_decode.json
        │
        ▼
gen_manifest.py
        │
        ▼
rankN.rhal.mlir
  dispatch → collective → dispatch → ...
        │
        ▼
rax-pack
        │
        ▼
rankN.rax
        │
        ▼
RaxExecutor
        │
        ▼
Communicator
        │
        ▼
MpiCommunicator
        │
        ▼
MPI
```

Each distributed process executes one rank-local RAX package. Compute and
communication remain explicit in the RAX schedule, while the model runner
continues to provide the higher-level inference loop such as tokenization,
prefill, KV-cache management, and decode.

## RAX Communication Model

### Ordered RHAL Functions

The original RHAL function form describes a single dispatch through
attributes:

```mlir
rhal.func @forward {
  inputs = ["input"],
  outputs = ["output"],
  dispatch = "model_kernels",
  args = ["input", "output"]
}
```

This form remains supported.

For multi-step execution, `rhal.func` can instead contain an ordered body of
`rhal.dispatch` and `rhal.collective` operations:

```mlir
rhal.func @forward {
  inputs = ["input"],
  outputs = ["output"]
} body {
  rhal.dispatch @stage0 [@params, @input, @scratch]

  rhal.collective [@scratch] {
    kind = "all_reduce",
    reduction = "sum"
  }

  rhal.dispatch @stage1 [@params, @scratch, @output]
}
```

Operations in a body-form `rhal.func` form an ordered runtime schedule.
`rax-pack` serializes the schedule into the RAX FlatBuffer representation, and
`RaxExecutor` executes the operations in the same order.

The legacy attribute-only form and the ordered body form coexist so existing
single-dispatch RAX packages do not need to use the distributed execution
path.

### Supported Collectives

The runtime communication interface currently provides the following
collectives:

| Collective    | Current behavior                                    |
| ------------- | --------------------------------------------------- |
| Broadcast     | In-place broadcast from a specified root rank       |
| AllReduce     | F32 reduction with `sum`                            |
| AllGatherV    | Variable-size gather into an explicit output buffer |
| ReduceScatter | F32 `sum` reduction into an explicit output buffer  |

For example:

```mlir
rhal.collective [@input] {
  kind = "all_gatherv",
  output_buffers = [@output],
  recv_counts = array<i64: 2, 3>,
  displacements = array<i64: 0, 2>
}
```

and:

```mlir
rhal.collective [@input] {
  kind = "reduce_scatter",
  output_buffers = [@output],
  recv_counts = array<i64: 2, 2>,
  reduction = "sum"
}
```

The RHAL verifier checks properties that can be validated statically from the
IR. Communicator size, rank, and other execution-dependent properties are
validated by the runtime where required.

### Runtime Execution

Distributed execution is split between a generic executor and a communication
backend:

```text
RaxExecutionSession
        │
        ▼
RaxExecutor
  ├─ Dispatch
  │    └─ host shared library
  │
  └─ Collective
       └─ Communicator
            └─ MpiCommunicator
                 └─ MPI
```

`RaxExecutor` does not call MPI directly. It operates on the abstract
`Communicator` interface, which provides rank information and collective
operations.

`MpiCommunicator` is the MPI implementation of this interface. It maps RAX
collectives to the corresponding MPI operations.

MPI support is optional at build time and is controlled by:

```text
BUDDY_RUNTIME_ENABLE_MPI
```

When the option is disabled, the generic RAX execution library can still be
built without MPI.

## Frontend Parallel Planning

### Parallel Plan

Transformer parallelization is represented explicitly in the frontend before
rank-local MLIR is materialized.

The existing Transformer structure and template analysis first produces a
`TransformerPartitionPlan`. Parallel analysis then constructs a
`TransformerParallelPlan`.

The main parallel-plan objects are:

```text
ParameterShardSpec
ValueLayoutSpec
OpRewriteSpec
CollectiveBoundary
ComputeSegment
```

A `ParameterShardSpec` describes how a parameter is divided between ranks.
A `ValueLayoutSpec` describes the distributed layout of an intermediate value.
An `OpRewriteSpec` records rank-specific shape or operand rewrites.
A `CollectiveBoundary` represents communication required between compute
regions. A `ComputeSegment` groups consecutive operations that can execute
locally on a rank.

The current configuration is represented by `TransformerParallelConfig`.
Tensor parallelism of size 2 is the currently validated configuration used by
the DeepSeek integration.

### Layout-Driven Collectives

Parallel communication is derived from tensor-layout transitions rather than
from model layer indices.

The frontend tracks three layout kinds:

* `REPLICATED`: every rank holds the complete value;
* `SHARDED(axis)`: the value is partitioned along a tensor axis;
* `PARTIAL`: each rank holds a partial contribution that must be reduced before
  some consumers can use the value.

When a consumer requires a different layout, the planner inserts a collective
boundary. Important transitions include:

| Producer layout | Required layout | Collective    |
| --------------- | --------------- | ------------- |
| `PARTIAL`       | `REPLICATED`    | AllReduce     |
| `PARTIAL`       | `SHARDED(axis)` | ReduceScatter |
| `SHARDED(axis)` | `REPLICATED`    | AllGatherV    |

For example, a row-parallel projection can produce a partial result on each
rank. If the next operation requires the complete value on every rank, the
planner represents the transition as an AllReduce boundary.

This keeps the communication decision in the parallel data-layout analysis
instead of hard-coding communication after particular DeepSeek layers.

### Rank-Local Materialization

`ParallelTemplatePartitionedGraphDriver` consumes the parallel plan for one
rank and materializes the local program.

For each rank it produces:

* rank-local template subgraphs;
* rank-local compute-segment wrappers;
* rank-local tensor shapes;
* a rank-global parameter pack;
* an ordered runtime plan containing dispatch and collective operations.

A rank-global parameter pack is shared by the wrappers belonging to the same
rank. Individual wrappers use static offsets into that pack for the parameters
they consume.

The generated runtime plans are JSON files such as:

```text
layer_partitioned/runtime/
├── rank0_forward_prefill.json
├── rank0_forward_decode.json
├── rank1_forward_prefill.json
└── rank1_forward_decode.json
```

Each runtime plan describes the resources and ordered operations for one
function on one rank.

`gen_manifest.py` converts the prefill and decode runtime plans of a rank into
one body-form RHAL module. The module is then serialized by `rax-pack`, producing
one RAX package for each rank.

## DeepSeek R1 TP=2 Example

DeepSeek R1 is the currently validated end-to-end integration of the
distributed RAX path.

The TP=2 package is separate from the normal single-rank
`models/deepseek_r1` build. It generates two sets of parameters and runtime
plans, two RHAL schedules, and two rank-local RAX packages.

The example below uses the f32 DeepSeek R1 Distill Qwen 1.5B configuration.

### MPI / MPICH

The communication runtime uses the standard MPI C++ interface through CMake
`FindMPI`.

MPICH is the MPI implementation currently validated for the DeepSeek TP=2
flow. Make sure that the MPI compiler and launcher come from the same MPI
installation.

For a system MPICH installation, verify:

```bash
mpicxx --version
mpiexec --version
```

For a custom MPICH installation:

```bash
export MPI_HOME=/path/to/mpich-install
export PATH="$MPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$MPI_HOME/lib:$LD_LIBRARY_PATH"
```

If CMake does not automatically select the intended MPI compiler, it can be
specified through `build_model.py`:

```bash
--cmake-args=-DMPI_CXX_COMPILER="$MPI_HOME/bin/mpicxx"
```

The TP=2 build automatically enables `BUDDY_RUNTIME_ENABLE_MPI`.

### Build

Run the build from the repository root:

```bash
cd buddy-mlir
```

Build the f32 DeepSeek TP=2 package:

```bash
python3 tools/buddy-codegen/build_model.py \
  --spec models/deepseek_r1/specs/f32.json \
  --build-dir build \
  --tensor-parallel-size 2
```

For an existing local HuggingFace-format model directory:

```bash
python3 tools/buddy-codegen/build_model.py \
  --spec models/deepseek_r1/specs/f32.json \
  --build-dir build \
  --local-model /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
  --tensor-parallel-size 2
```

If a custom MPICH installation must be selected explicitly:

```bash
python3 tools/buddy-codegen/build_model.py \
  --spec models/deepseek_r1/specs/f32.json \
  --build-dir build \
  --local-model /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
  --tensor-parallel-size 2 \
  --cmake-args=-DMPI_CXX_COMPILER="$MPI_HOME/bin/mpicxx"
```

For `--tensor-parallel-size 2`, `build_model.py` automatically enables:

```text
BUDDY_BUILD_DEEPSEEK_R1_TP2_MODEL=ON
BUDDY_RUNTIME_ENABLE_MPI=ON
BUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON
```

and builds the `deepseek_r1_tp2_rax` target.

The TP=2 model has its own CMake build graph under:

```text
models/deepseek_r1_tp2/
```

It intentionally does not use the normal single-model `buddy_add_model()`
pipeline because distributed compilation produces rank-local runtime plans,
parameter packs, RHAL modules, and RAX packages instead of one ordinary model
artifact.

### Generated Artifacts

After a successful build, the main artifacts are under:

```text
build/models/deepseek_r1_tp2/
```

The important files include:

```text
build/models/deepseek_r1_tp2/
├── rank0.rax
├── rank1.rax
├── rank0.rhal.mlir
├── rank1.rhal.mlir
├── rank0_params_float32.data
├── rank1_params_float32.data
├── deepseek_r1_tp2_model.so
├── deepseek_r1_tp2_runner.so
├── vocab.txt
├── generated/
│   └── RaxShims.cpp
└── layer_partitioned/
    ├── rank0/
    ├── rank1/
    └── runtime/
        ├── rank0_forward_prefill.json
        ├── rank0_forward_decode.json
        ├── rank1_forward_prefill.json
        └── rank1_forward_decode.json
```

`rank0.rax` and `rank1.rax` are the two runtime entry packages. Each one
contains the schedule and resource metadata for its rank and refers to its
rank-local parameter pack.

`deepseek_r1_tp2_model.so` contains the compiled rank-local compute kernels and
the generated RAX ABI shims.

`deepseek_r1_tp2_runner.so` implements the DeepSeek TP=2 inference runner used
by `buddy-cli`.

If payload embedding is enabled in the RAX build, runtime libraries referenced
by the RAX package are embedded by `rax-pack` and extracted by the runtime when
the package is loaded.

### Run

Run two MPI processes and use the rank placeholder in the RAX path:

```bash
mpiexec -n 2 \
  ./build/bin/buddy-cli \
  --model "./build/models/deepseek_r1_tp2/rank{rank}.rax" \
  --prompt "Hello" \
  --temperature 0 \
  --max-tokens 2 \
  --no-stats
```

Under the validated MPICH launch path, each process receives `PMI_RANK`.
When the `--model` path contains `{rank}`, `buddy-cli` replaces the placeholder
with `PMI_RANK` before reading the RAX manifest.

The two MPI processes therefore resolve the same command line as:

```text
rank 0 → build/models/deepseek_r1_tp2/rank0.rax
rank 1 → build/models/deepseek_r1_tp2/rank1.rax
```

The `{rank}` handling is generic in `buddy-cli`; the CLI does not need a
DeepSeek-specific tensor-parallel option at runtime.

Each process loads its own RAX package and executes the same inference loop.
Collective operations inside the prefill and decode schedules synchronize the
rank-local computations through `MpiCommunicator`.

Only the designated output rank emits generated text, while all ranks
participate in the distributed computation.

## Current Limitations

The distributed RAX infrastructure is designed independently of DeepSeek, but
the current end-to-end integration has the following limitations:

* DeepSeek R1 with tensor parallelism of size 2 is the currently validated
  model configuration.
* The DeepSeek TP=2 package is currently validated for native builds and is not
  enabled for RVV cross-compilation.
* The current runtime collective data type is F32, and reduction collectives
  currently use `sum`.
* Each distributed process executes one rank-local RAX package.
* The current frontend entry point exposes the validated TP=2 configuration;
  broader distributed configurations require additional validation before
  being exposed through the model build interface.
