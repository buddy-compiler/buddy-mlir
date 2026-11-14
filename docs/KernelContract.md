# Kernel Contract v1: bufferized Linalg + neutral facts

This layer is the sole IR contract between triton-riscv (or another
frontend) and buddy-mlir. It is responsible for what the program reads and
writes, who depends on whom,which values are external, and which mask regions
 are valid.It is not responsible for bank placement, DMA channels,
 or packetization —that is `bosc-ame-plan`.

- Example: [kernel.contract.mlir](../tests/KernelContract/Inputs/kernel.contract.mlir)
  (196×768×768 kernel; parseable by `buddy-opt`)
- Namespace: `contract.*` (discardable attributes + verifier; no dialect)
- Version: `contract.version = 1`

## 1. Design principles

| # | Principle | Implication |
| --- | --- | --- |
| P1 | **Plan-sufficient, not plan-carrying** | Every contract fact must be usable by the planner; never encode target decisions or target constants |
| P2 | **SSA first** | Shape/stride/root/offset come from memref types and `reinterpret_cast`/`subview` chains; attributes only add what SSA cannot express (mask, other, origin, task graph, scopes, loop-carried links) |
| P3 | **Per-scope graphs; incompleteness allowed** | Graphs are organized by scope with per-scope completeness; unmodeled parts use `complete=false` + limitation |
| P4 | **Verified at both ends** | Frontend verifies after emit; planner verifies again at entry. The contract is an ABI |
| P5 | **Neutral naming** | No `bosc_ame.*` / `plan.*` in the contract; those appear only in Buddy planned/final IR |

**Settled schema choices:** discardable attributes + shared verifier (no
dialect); integer value IDs (not SSA refs); per-scope graphs with uniform-loop
summaries. Field semantics changes require bumping `contract.version`. Within
a major version, optional fields may be added; removals/redefinitions are
forbidden. Planners ignore unknown `contract.*` fields and report them. A
frontend that cannot supply a task graph may emit only
`access`/`buffer`/`compute`/`program`/`grid` with all scopes
`complete=false`; the planner must still plan from boundary facts alone.

## 2. Where attributes attach

MLIR cannot attach attributes to arbitrary SSA values:

| Fact | Carrier |
| --- | --- |
| Module | `builtin.module`: version / producer / ABI |
| Function | `func.func`: grid, compute-graph summary |
| Function args | arg attrs: program axes/roles, external `value_id` |
| Content versions written by an op | `contract.task.output_values` (loops: `scf.for` `contract.value_ids`) — versions are **not** SSA results |
| Memory boundary events | `contract.access` |
| Task nodes | `contract.task` |
| Compute / fill semantics | `contract.compute`, `contract.fill` |
| Loop-carried / body scope | `scf.for`: `loop_carried_slots` + `cf` + `scope` |
| Branch exclusivity / arm scopes | `scf.if`: `cf` + `scopes` |

Logical references use `int64` value IDs. The verifier enforces closure and
bidirectional consistency.

### Attribute index

| Attribute | Carrier | One-line meaning |
| --- | --- | --- |
| `contract.version` | module | Schema version |
| `contract.producer` | module | Frontend id (diagnostics) |
| `contract.abi` | module (driver) | Caller alignment / non-alias promise |
| `contract.compute_graph` | function | Root scope, aggregate completeness, scope count |
| `contract.grid` | function | Known program grid |
| `contract.program` | function arg | `program_id` / `num_programs` axis |
| `contract.value_id` | memref arg | External entry content version |
| `contract.buffer` | alloc/alloca | Tile role / logical shape |
| `contract.access` | load/store boundary op | Boundary, mask, other, origin |
| `contract.task` | compute/dataflow op | Task-graph node (incl. scope) |
| `contract.compute` | dot/elementwise/reduce | Compute semantics |
| `contract.fill` | `linalg.fill` | Padding-fill semantics |
| `contract.provenance` | op | Origin / lineage |
| `contract.program_axes` | op | Address/task varies with program axes |
| `contract.value_ids` / `contract.value_users` | `scf.for` / defining tasks | Loop iter/result versions; optional reverse users |
| `contract.loop_carried_slots` | `scf.for` | Carried-value mapping |
| `contract.cf` | `scf.if` / `scf.for` | Branch/loop control facts |
| `contract.scopes` / `contract.scope` | `scf.if` / `scf.for` | Arm / body scope records |
| `contract.math` | `func.func` | Explicit approximation authorization |

## 3. Attribute reference

### 3.1 Module

#### `contract.version` (i32, required)

Schema version. Currently `1`. Unknown major versions fail; unknown minor
fields are ignored.

#### `contract.producer` (string, required)

Frontend identifier (e.g. `"triton-riscv"`), diagnostics only.

#### `contract.abi` (dict, optional; injected by the pipeline driver)

```mlir
contract.abi = {args_aligned_bytes = 64 : i64, args_disjoint = true}
```

| Field | Meaning |
| --- | --- |
| `args_aligned_bytes` | Public entry memref aligned pointers are divisible by this positive byte count; for 64 B accesses it must be a positive multiple of 64 |
| `args_disjoint` | Distinct public entry memrefs point at non-overlapping DDR ranges |

Applies only to entry functions with **no** symbol uses in the module.
Helpers reached by `func.call`, `func.constant`, or other references do **not**
inherit these promises. Visibility (`public`/`private`) is not an entry proof.
Private alloc independence still comes from storage identity.

Alignment means divisibility, not “larger is always stronger”: 128 implies 64 B
alignment; 96 does not. Promises do not skip per-access address proofs with
offset, stride, tile origin, and tails.

Without this attribute the planner **must not** use direct DDR routes or merge
on “args do not alias”.

These fields do **not** provide allocation extent. For unranked DDR args the
caller must support the kernel’s legal accesses; the frontend must build valid
views from source shape, grid, and mask. The verifier may check transfers stay
inside that view; it cannot prove the view lies in a real external allocation
from ABI alone.

### 3.2 Function

#### `contract.compute_graph` (dict, required)

```mlir
contract.compute_graph = {version = 1 : i32, root_scope = 0 : i64,
                          complete = true, scope_count = 5 : i64,
                          source_lineage_complete = true}
```

| Field | Type | Meaning |
| --- | --- | --- |
| `version` | i32 | Graph schema version; must be 1 |
| `root_scope` | i64 | Function-body scope id; must be 0 |
| `complete` | bool | Aggregate completeness = AND of all scope `complete` |
| `scope_count` | i64 | Scope count including root (= 1 + recorded scopes) |
| `source_lineage_complete` | bool | Provenance completeness (reports / rematerialization only) |
| `limitation` | string, optional | Why incomplete |

Completeness is **per scope** (§3.10). Planner decisions look at the scopes of
relevant tasks, not only the function aggregate.

Graph completeness and source-lineage completeness are independent. Standard
Linalg without source origins may still have a complete memory graph.
Eliminating a physical load does not degrade lineage if source-event facts
remain. Constant/fill operands proved to have no load origin are not unknown
sources.

#### `contract.grid` (array<i32: 3>, optional)

Known program grid; `-1` means unknown on that axis.

```mlir
func.func @...(...) attributes {contract.grid = array<i32: 7, 6, 1>, ...}
```

### 3.3 Function arguments

#### `contract.program` (dict, arg attr)

```mlir
%arg6: i32 {contract.program = {role = "program_id", axis = 0 : i32}}
```

| role | Meaning |
| --- | --- |
| `"num_programs"` | Program count on that axis |
| `"program_id"` | Current program id on that axis |

#### `contract.value_id` (i64, memref args)

Marks the external memref’s entry content version. Only args with
`contract.value_id` may appear in task `input_values`. Ordinary scalars do not
produce logical values.

### 3.4 Values and buffers

#### `contract.buffer` (dict on `memref.alloc` / `memref.alloca`)

```mlir
%c_buf = memref.alloc() alignment = 64
  {contract.buffer = {role = "tile", kind = "accumulator",
                      logical_shape = array<i64: 32, 128>}} : memref<32x128xi32>
```

| Field | Values | Meaning |
| --- | --- | --- |
| `role` | `"tile"` \| `"scratch"` | Data tile vs private temporary |
| `kind` | `"accumulator"` \| `"input-a"` \| `"input-b"` \| `"temporary"` \| `"padding"` | Data role |
| `logical_shape` | array<i64> | Logical tile shape; memref type may be flattened |

No memory space / bank / offset / capacity — those are planner outputs.
Rank-0 reduction init/result uses empty `logical_shape` for one scalar
element. v1 supports ranks 0–2.

#### Content versions: `contract.task.output_values` / `contract.value_ids`

A logical value id is a **content version of storage**, not an SSA result.
Buffer-form `memref.copy` / `linalg.fill` / `linalg.matmul` have no SSA
results but write storage; each write creates a new version. Value id, buffer
root (via view chain, §3.11), and physical window (`plan.buffer`) stay
separate.

| Carrier | Meaning |
| --- | --- |
| Private root (`alloc`/`alloca` and views) | Whole-root snapshot; partial writes inherit uncovered bytes |
| External input (memref args) | Entry contents of the accessed region (`contract.value_id`); read-only |
| External output (store / direct external write) | **No version**; `output_values` must be empty |

Definition sites only:

1. Memref args: `contract.value_id`
2. Tasks: `contract.task.output_values` (new private-root versions; order =
   written-root operand order)
3. Fully modeled single-slot `scf.for`:
   `contract.value_ids = [iter_arg_value_id, result_value_id]` (§3.9)

Rules (V21–V26):

- Each value id is defined exactly once in static IR and is non-negative; all
  explicit refs resolve. A static definition inside a loop describes each
  iteration’s dynamic contents; cross-iteration links use loop-slot summaries
  only.
- Writers of private roots define a new version. `input_values` hold actual
  memory reads plus old versions needed for partial-write inheritance
  (§3.11). Pure WAW order uses `predecessors`, not fake inputs of fully
  killed contents.
- First write of a freshly allocated root must fully initialize it; alloc
  itself defines no version. Partial boundary loads must fill the whole root
  then copy the valid subview.
- Terminal external stores define no version.
- Unread versions are allowed; the verifier reports them.

Optional `contract.value_users` (array of arrays) lists reverse consumers and
must match reconstruction from `input_values` plus loop read/inherit edges.
Pure WAW slot consumers are not version readers. Do not treat every reverse
edge as a memory reader over the covered range (§3.11).

### 3.5 Memory boundaries

#### `contract.access` (dict on load/store boundary ops)

```mlir
memref.copy %c_view, %c_buf {contract.access = {
  kind = "load", boundary = "external_ddr", origin_id = 0 : i64,
  mask = {kind = "none"}, other = 0 : i32}}

memref.copy %c_src4, %c_dst4 {contract.access = {
  kind = "load", boundary = "external_ddr", origin_id = 0 : i64,
  mask = {kind = "rectangular", valid_shape = array<i64: 4, 128>},
  other = 0 : i32}}
```

| Field | Meaning |
| --- | --- |
| `kind` | `"load"` \| `"store"` |
| `boundary` | Externally visible DDR boundary (future: `"spm_view"`) |
| `origin_id` | Optional frontend source-event id; omit if none |
| `mask.kind` | `"none"` / `"rectangular"` (static) / `"opaque"` |
| `mask.valid_shape` | Static valid shape |
| `mask.dynamic` | Reserved for v2; **rejected in v1** (V16) |
| `other` | Fill for masked-out lanes (loads; optional if unmasked) |

### 3.6 Task graph

#### `contract.task` (dict on compute/dataflow ops)

```mlir
linalg.matmul {contract.task = {
  id = 3 : i64, kind = "dot", scope = 3 : i64, stage = 1 : i64,
  predecessors = array<i64: 0, 1, 2>,
  input_values = array<i64: 4, 5, 7>, output_values = array<i64: 6>,
  barrier_after = true, rematerializable = false, origin_id = 3 : i64}, ...}
```

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | i64 | Unique; strictly less than successors (topo order) |
| `kind` | string | `memory-load`, `memory-store`, `scalar-load`, `fill`, `dot`, `elementwise`, `reduce`, `copy` |
| `scope` | i64 | Innermost scope (§3.10) |
| `stage` | i64 | Sync stage: `max(pred.stage + pred.barrier_after)` |
| `predecessors` | array<i64> | Direct preds (same scope or legal cross-scope edges, V19) |
| `input_values` | array<i64> | Actual reads / partial-write inheritance (§3.11) |
| `output_values` | array<i64> | New private-root versions (empty for store/external) |
| `barrier_after` | bool | Sync point after the task |
| `rematerializable` | bool | Pure compute; safe to recompute |
| `origin_id` | i64, optional | Source-event id |

`stage` / `barrier_after` are frontend semantic facts for fuse legality and
window-reuse boundaries, not scheduling decisions. `scalar-load` defaults to
`barrier_after=false`, `rematerializable=false`: it preserves read timing, is
not a DMA, and must not rematerialize across writers that may overwrite its
source root.

### 3.7 Compute semantics

#### `contract.compute` (dict)

```mlir
linalg.matmul {contract.compute = {kind = "dot", signed = true,
                                   lhs = "i8", rhs = "i8", acc = "i32"}, ...}
```

Declares semantics only (“signed i8×i8→i32 dot”). AME vs other engines is a
planner/target decision. Keep linalg `cast = #linalg.type_fn<cast_signed>` as
linalg semantics.

For a serial reference matmul, declare order explicitly:

```mlir
linalg.matmul {contract.compute = {kind = "matmul", order = "mnk"}, ...}
```

Requires serial M/N/K loops, reading operands 0, 1, then old accumulator each
K step, then writing the original pure scalar arith payload immediately.
Constrains observable order and FP flags; does not name a target engine. v1
allows `order` only on `linalg.matmul` with `kind="matmul"` and value
`"mnk"`. Ordinary matmuls without order keep V25.

If A/B and the output cannot be proved disjoint, keep the op as a synchronous
but **unmodeled** memory boundary: scope stays incomplete, omit
`contract.task` and private output versions, invalidate possibly affected
roots, and do not claim complete slots for such loops. When independence is
proved, use normal task/version encoding (still not rematerializable). The
planner must use a mechanical reference recipe — not AME/RVV parallelization
or incomplete-proof forwarding/window reuse.

#### `contract.fill` (dict on `linalg.fill`)

`padding=true` means the fill initializes masked-load padding; it must precede
the corresponding prefix DMA (order via in-scope task order +
`predecessors`).

### 3.8 Provenance

#### `contract.provenance` (dict)

```mlir
linalg.matmul ... {contract.provenance = {
  origin_id = 3 : i64, source_load_origin_ids = array<i64: 0, 1, 2>,
  a_origin_id = 1 : i64, b_origin_id = 2 : i64,
  a_source_load_origin_ids = array<i64: 1>,
  b_source_load_origin_ids = array<i64: 2>,
  accumulator_source_load_origin_ids = array<i64: 0>,
  a_source_lineage_complete = true, b_source_lineage_complete = true,
  accumulator_source_lineage_complete = true}}
```

Used for reports, rematerialization, and recognizing the same logical value
across arms (shared `origin_id`). **Not** for alias analysis.

`source_load_origin_ids` and per-operand `*_source_load_origin_ids` list
source-load events (including merged/eliminated ones); they are not a numeric
dependence closure. Loop-carried accumulator origins follow the slot init
(e.g. initial C load), without flattening prior A/B arithmetic into that list.
Lineage completeness means the access tracking is complete — not numeric
independence. Optional `source_store_origin_ids` works the same way. All origin
ids are non-negative; `*_lineage_complete=true` requires a nonempty matching
list.

Constants / zero-fills proved to have no source load must not claim nonempty
operand lineage. Do not invent load origins for constants.

Origin IDs and task IDs are different namespaces: `task_map` covers pre-rewrite
task IDs; provenance keeps source-event IDs. Planner staging may omit origin or
use `contract.provenance={generated_by="planner-staging"}`; it must not invent
source origins. Fused loads/stores must retain their real source events.

### 3.9 Program axes, loops, control flow

#### `contract.program_axes` (dict)

```mlir
{contract.program_axes = {varying = array<i64: 0, 1>, invariant = array<i64: 2>}}
```

Whether addresses/tasks vary with program-id axes (reporting / future reuse).

#### `contract.loop_carried_slots` (array of dicts on `scf.for`)

```mlir
scf.for ... { } {contract.loop_carried_slots = [{
  slot = 0 : i64, init_value_id = 3 : i64, iter_arg_value_id = 7 : i64,
  yielded_value_id = 6 : i64, result_value_id = 8 : i64,
  init_producer_compute_ids = array<i64: 0>,
  iter_arg_consumer_compute_ids = array<i64: 3>,
  backedge_producer_compute_ids = array<i64: 3>,
  exit_consumer_compute_ids = array<i64: 4>}]}
```

Maps carried buffers to the value graph. `init_value_id` is produced outside
the loop; `iter`/`result` ids come from `contract.value_ids`.

**v1 limits:** exactly one buffer-carried slot with `slot=0`;
`value_ids = [iter, result]`; positive static trip count, positive step,
serial, no early exit; trip count matches bounds/step with overflow checks.
Zero-trip loops are not modeled as complete. Multi-slot / dynamic / other
unmodeled loops must set related scopes `complete=false` with limitation.

**Same-root latest-version machine** (verifier rebuilds from IR):

1. Derive unique private root R from the init version; init must be R’s latest
   version before the loop. Iter/result inherit that root; yielded’s writer
   must write the same R.
2. At body entry bind R’s current version to iter. Advance via real reads,
   partial-write inheritance, and writes (§3.11). First partial write inherits
   iter (or the latest body writer), never an older snapshot by skipping.
3. At normal body exit, yielded is exactly R’s last writer’s version in that
   body. After the loop, bind R to result for subsequent reads/partial writes
   until the next writer.
4. Dynamics: `iter[0]=init`, `iter[k+1]=yielded[k]`,
   `result=yielded[trip_count-1]`. Backedges are not topo-breaking
   `predecessors` edges; they live in the slot + serial-loop facts.

Producer/consumer lists are derived precisely (sorted unique IDs), not optional
hints. Iter/exit consumer lists include actual read/inherit tasks **and** first
full-kill WAW tasks that establish entry/exit order; those kill tasks must not
fake-read iter/result in `input_values` / `value_users`.

Incomplete loops must omit fake `value_ids`/slots; record the unmodeled
boundary in the scope limitation. Retained facts still obey schema. Incomplete
summaries prove no cross-iteration fuse/forward/reorder/reuse.

#### `contract.cf` (dict on `scf.if` / `scf.for`)

```mlir
scf.if ... { } else { } {contract.cf = {
  kind = "uniform_if", condition_axes = array<i64: 0>,
  arms_exclusive = true, arm_logical_tile = array<i64: 32, 128>}}

scf.for ... { } {contract.cf = {
  kind = "uniform_for", trip_count = 6 : i64, uniform = true, serial = true}}
```

| Field | Meaning |
| --- | --- |
| `kind` | `"uniform_if"` or `"uniform_for"` |
| `arms_exclusive` | Arms mutually exclusive per program → shared SPM windows allowed |
| `arm_logical_tile` | Both arms are variants of one logical tile |
| `condition_axes` | Program axes the condition depends on |
| `trip_count` / `uniform` / `serial` | Positive static trips, same across programs, serial, no early exit |

### 3.10 Scopes

The graph is a region tree under root scope 0:

```text
scope 0 (function)
├── scope 1 (if-arm then)   ← contract.scopes on scf.if
│   └── scope 3 (for-body)  ← contract.scope on scf.for
└── scope 2 (if-arm else)
    └── scope 4 (for-body)
```

#### `contract.scopes` (array on `scf.if`)

Each arm: `id`, `kind="if-arm"`, `arm` (`then`/`else`), `parent`,
`arms_exclusive=true`, `condition_axes`, `complete`.

#### `contract.scope` (dict on `scf.for`)

`id`, `kind="for-body"`, `parent`, `complete` (loop uniformity lives in
`contract.cf`).

#### Completeness (V18)

- **Leaf:** all tasks have full fields; value refs resolve (V7); predecessors
  are same-scope or legal V19 edges; no calls / unknown effects.
- **If:** all arm scopes complete; no private content-version edges between
  arms; sharing limited to external entry memrefs and ordinary scalar SSA;
  private roots defined per arm; `arms_exclusive`. v1 does not model parent
  private roots crossing into arms or merging private results — mark
  incomplete instead of forging edges.
- **For:** body complete + positive static trip / positive step /
  `uniform`/`serial` / no early exit + single buffer slot (§3.9) + no
  unmodeled effects in the loop.
- **Function:** `compute_graph.complete = AND(all scope.complete)`.

#### Legal cross-scope edges (V19)

1. Same-scope predecessors
2. Parent init producer → child loop iter consumers (reads/inherits or entry WAW)
3. Child backedge producer → parent exit consumers (reads/inherits or exit WAW)
4. Sibling if-arms: **no** predecessor or value edges

After an unmodeled arm touches a parent private root, that root’s version at
the join is unknown until a proved full initialization creates a new version.
External write effects union write regions into the parent; post-store reads of
the same external region remain unsupported (V24) even under arm exclusivity.
Recovery does not make the unmodeled scope complete or authorize reuse across
that boundary. The same invalidate/recover rules apply to incomplete loops and
unknown effects: only proved-unaffected roots keep versions; full pure writes
may recover; partial writes / reads of old init / unknown sources cannot.
Matmul/reduce that read old init cannot alone recover unknown.

### 3.11 Storage roots, ranges, and read/write derivation

**Storage root:** walk the view chain to `alloc`/`alloca`, a function arg, or a
call result. Intermediate `reinterpret_cast` / `subview` / `cast` /
`collapse_shape` / `expand_shape` / `transpose` change the view range only.
`contract.abi.args_disjoint` applies only to declared entry args. Distinct
views of one root **may** alias unless ranges are proved disjoint.

**Ranges:** byte address sets relative to one root (element offsets of
different types are not comparable). Compose the full SSA view chain,
including `memref.view` byte shifts; matching MemRefType offset/strides alone
is insufficient. Unproved base/offset/stride ⇒ possible overlap. Full root
coverage requires the write address set to cover the whole root with no
holes/duplicates; equal element counts or bounding boxes are not enough.

**Derivation table** (V22; no extra attributes):

| Op form | Read set (`input_values` order) | Write (new version) |
| --- | --- | --- |
| `memref.copy %src, %dst` | version of `%src` | new private root of `%dst`; external `%dst` → store, no version |
| `linalg.fill outs(%o)` | none (scalar SSA → predecessors) | new private root of `%o` |
| `linalg.matmul ins(%a,%b) outs(%c)` | `%a`, `%b`, old `%c` | new private `%c` (or direct external via store/route) |
| `linalg.generic` | only operands whose payload block args are used | one new version per private outs (v1: single output) |
| `linalg.reduce` | `%x` + old `%init` | new rank-0 `%init` |
| `memref.load` | version of `%src` at the load | none; `kind="scalar-load"` |
| `memref.store` | scalar SSA (predecessors); not a content version | v1: external only, no version |
| `scf.for` | — | `value_ids = [iter, result]` |

Partial private writes also append the prior same-root version for inheritance
(deduped). Full kills that do not read old output only create WAW
predecessors. Inheritance must use the **current** version before that write.
First private write must fully initialize.

**Masked-load chain:** fill(`other`) → version v1; copy valid subview → inherits
v1 → v2; consumers read v2. Full-root copies need no inheritance input.

**Scalar / index chains:** Pure scalar ops create no tasks/versions.
`memref.load` is a `scalar-load` task. Buffer tasks that consume scalars walk
use-def to those loads and add them as predecessors. RAW/WAR checks run at the
load; SSA scalars keep the loaded value after the buffer is overwritten. v1
models only same-scope scalar chains.

**In-place rules** (also V25): old versions must have no future **memory**
readers on the covered range (`scalar-load` counts; later pure scalar math does
not). Ops that read the old destination must list that old version.

| Op | Same-task alias condition |
| --- | --- |
| Pointwise generic | Overlapping read operands must match output address-wise; others disjoint; single arith, pure parallel, identity indexing |
| Matmul | Output overlaps only its own accumulator read; A/B disjoint from output |
| Reduce | Output overlaps only its init; reduction input disjoint; rank-0 init/result OK |
| Fill | No memref inputs; full cover or partial inherit |
| Copy | Source/dest disjoint or identical elementwise (self-copy); reject partial/unknown overlap |

Identity indexing alone is not enough
(`ins(root[0:768]) outs(root[1:769])` is illegal). When unproved, the frontend
allocates another root. Reference-ordered may-alias matmul (§3.7) stays an
incomplete boundary without a snapshot task and does not relax other in-place
rules.

**External write order (V24):** external writes follow all prior readers of that
region; v1 does not support read-after-write on external entry versions.
Compare all possibly intersecting external views, not only SSA root equality.

## 4. Mask contract

Frontends reduce predicates to one of:

| Class | Attribute | Planner |
| --- | --- | --- |
| None | `mask = {kind = "none"}` | Whole tile may DMA / direct |
| Static rectangular / prefix | `mask = {kind = "rectangular", valid_shape = [...]}` | Per-row prefix DMA; remainder from prior fill |
| Dynamic rectangular (**v2**) | `mask = {kind = "rectangular", dynamic = true}` + extent SSA in IR | **v1: verifier error (V16)**; v2: guarded prefix DMA if clamp+alignment proved, else opaque |
| Opaque | `mask = {kind = "opaque"}` | Vector / element copy only; no DMA bulk |

v1 accepts only `none` / static rectangular / `opaque`.

Static problem shapes are necessary but not sufficient: masks that still depend
on program id must be specialized into static arms (e.g. `pid0 < 6` → full vs
tail `valid_shape=[4,128]`). Leaving a runtime
`clamp(M - pid*tile, …)` extent fails v1 even when M is constant. Conversely,
an ABI that promises a runtime extent is a multiple of the tile may delete the
mask to `none`.

For v2, extent must appear as an IR operand (dynamic subview size, loop upper
bound, guard). Planner obligations for guarded prefix DMA: `0 ≤ extent ≤
static_max`, extent divisible by DMA packet elements, def dominates use, target
supports the form. Extent describes length only — not data-dependent lane sets.

## 5. Forbidden in the contract

| Forbidden | Why |
| --- | --- |
| `bank` / `offset` / `capacity` / `memory_space` | Target decisions |
| Packet lists / `channel` / `token` / wait policy | Physical DMA |
| Direct/staged/route choices | Planner |
| `bm/bn/bk`, `lower_to_ame`, `enable_*`, `placement` | Target/plan options |
| `bosc_ame.*`, `plan.*` | Backend coupling |
| Target constants | Break across targets |
| Redundant SSA facts (shape/stride/offset) | Already in types/views |
| Schedule / sync instructions | Plan and emit |

## 6. Verifier rules

Run at both ends. Violations are hard errors. `complete=false` marks an
explicit unmodeled boundary; it does **not** waive schema on declared facts.
Incomplete summaries never authorize fuse/forward/reuse/reorder proofs.

| # | Rule |
| --- | --- |
| V1 | Supported `contract.version`; unknown fields ignored |
| V2 | If `contract.abi` present: typed fields; `args_aligned_bytes` positive; for 64 B access must be a multiple of 64 (96 fails) |
| V3 | Memref-only memory interface: no tensor, unrealized casts, SPM spaces, or pointer types; ordinary scalar program args OK |
| V4 | Each `contract.access`: legal kind/boundary; mask.kind ∈ {none,rectangular,opaque}; rectangular `valid_shape` matches view rank, dims > 0; `other` only on loads with matching element type |
| V5 | Each `contract.buffer` on alloc/alloca; legal role/kind; `logical_shape` rank ∈ [0,2] and byte-consistent; rank-0 holds one element |
| V6 | Task ids unique, non-negative, **strictly increasing**; `scope_count` matches |
| V7 | Value defs only at arg `value_id` / task `output_values` / loop `value_ids=[iter,result]`; each id defined once; refs resolve |
| V8 | If `value_users` present: matches reconstruction; pure WAW slot consumers are not version readers |
| V9 | Each predecessor resolves and is `<` current id; cross-scope edges match V19 |
| V10 | `contract.cf` only on `scf.if`/`scf.for`; `arms_exclusive` only for `uniform_if`; condition axes exist |
| V11 | Loop slots: four value ids and task ids resolve; init producers in parent; iter/backedge consumers in body; exit consumers in parent |
| V12 | Provenance origin/source ids non-negative; complete lineage lists nonempty; function `source_lineage_complete` matches recomputed aggregate; do not mix task_map and origin namespaces |
| V13 | (v2) Dynamic mask: extent SSA locatable in IR |
| V14 | Recursive ban on `bosc_ame.*` / `plan.*` / SPM spaces / target decisions in all nested regions, attrs, and types; incomplete does not waive |
| V15 | Unmodeled boundaries → related scopes `complete=false` with located reason; declared parts still enforce schema; unknown old IDs illegal |
| V16 | **v1 rejects** `mask.dynamic = true` |
| V17 | Scope ids unique and ≠ 0; parents form a tree (root 0); if-arms are exactly then/else with `arms_exclusive=true` |
| V18 | Per-scope completeness per §3.10; function complete = AND(scopes) |
| V19 | Cross-scope edges: same scope / parent→child loop init / child→parent loop exit; no sibling-arm edges |
| V20 | Complete v1 loops: one `slot=0`, positive static trip/step, serial; `value_ids` match; same-root latest-version machine; unmodeled loops incomplete |
| V21 | Version-definition integrity (§3.4); store/external `output_values` empty |
| V22 | Read/inherit derivation (§3.11); full kills are WAW only; first private write fully initializes; entry versions read-only |
| V23 | RAW / latest-version at each actual read or inherit; `scalar-load` checked at the load |
| V24 | WAR/WAW total order on private roots; external entry reads before terminal writes; no v1 external RAW |
| V25 | In-place safety (§3.11); reference-ordered may-alias matmul only as incomplete no-task boundary |
| V26 | Scope/loop version constraints; unknown join versions after unmodeled parent-root arm access; arm external stores propagate to parent. Also rejects malformed `contract.math` |

## 7. Migration from legacy `bosc_ame.*`

| Legacy (bufferized IR) | Contract |
| --- | --- |
| `bosc_ame.program_axis` / `program_role` | `contract.program` |
| `bosc_ame.compute_graph_*` | `contract.compute_graph` + per-scope `scope(s)` |
| `bosc_ame.source_lineage_complete` | `compute_graph.source_lineage_complete` |
| `gpu.known_grid_size` | `contract.grid` |
| `bosc_ame.compute_id` / kind / stage / … | `contract.task` |
| `bosc_ame.*_program_axes` | `contract.program_axes` |
| `bosc_ame.logical_value_ids` / `users` | **Do not copy** — reallocate content versions per §3.4 |
| `bosc_ame.memory` | `contract.access` |
| `bosc_ame.padding` | `contract.fill {padding=true}` |
| `bosc_ame.lower_to_ame` | `contract.compute {kind="dot", …}` |
| origin / source_load fields | `contract.provenance` |
| `bosc_ame.logical_shape` | `contract.buffer.logical_shape` |
| `bosc_ame.loop_carried_slots` | `contract.loop_carried_slots` + `value_ids` |
| `bm/bn/bk/target/enable_*/placement/...` | **Remove** (ABI → `contract.abi`) |
| linalg `cast` | Keep |

Old “any control flow ⇒ incomplete” is replaced by the scope tree. Old logical
value ID spaces must not be imported; the new frontend bufferizes and assigns
unique producers.

## 8. Negative-test coverage

Beyond positive fixtures, conformance must reject at least:

- Loop yielded on the wrong root; init not parent-latest; yielded not
  body-latest; partial write inheriting init / skipping a prior writer; exit
  consumers bypassing result
- Zero-trip / non-positive step / multi-slot / dynamic / early-exit loops
  claiming v1 complete; incomplete loops forging slots/ids
- In-place “proofs” from matching root/type/shape alone (esp. ignoring
  `memref.view` byte shift)
- Hidden `bosc_ame.*` / `plan.*` / SPM spaces anywhere in the module
- Missing required cross-scope WAW edges, or fake old versions in
  `input_values`/`value_users` for pure kills
- 96 B ABI approving 64 B direct access; negative origins; empty complete
  lineage lists; lost folded load/store tasks or provenance origins
- Parent private roots read/written inside arms with forged completeness;
  parent reads after arm external stores of the same region; unrelated roots
  incorrectly invalidated
- Incomplete loop / unknown-effect roots left live without recovery; partial
  or old-init “recovery”; reuse across unknown boundaries after recovery
- Complete graphs with incomplete lineage (allowed) vs forged constant-load
  origins (rejected); degraded lineage after physical-load forwarding when
  source facts remain

## 9. Function-scoped mathematical accuracy

`contract.math` attaches to `func.func`. It is an explicit caller promise, not
inferred from a target or op. Absence never authorizes approximation, FP
fusion, reassociation, or a larger error budget. Source evaluation order stays
binding.

```mlir
func.func @kernel() attributes {
  contract.math = {
    exp = {domain = "all-f32", special_values = "ieee",
           max_abs_error = 1.0e-6 : f64, max_rel_error = 1.0e-5 : f64}
  }
} { return }
```

The outer dict is nonempty and accepts only `erf` and `exp`. Each entry has
exactly those four fields. `domain` must be `all-f32`; `special_values` must
be `ieee`. Error budgets are finite nonnegative f64. Nested, op-scoped,
unknown-field, or wrong-type declarations fail (V26).

For finite reference `r`, output `y` must satisfy
`abs(y - r) <= max_abs_error + max_rel_error * abs(r)`. `all-f32` includes
nonfinite inputs. `ieee` requires mathematical specials, signed zero where
defined, gradual underflow, and RN-ties-to-even. NaN payloads and exception
flags are out of scope. Target recipes and evidence live in Buddy, not this
neutral schema.

## 10. Implementation ownership

This document is the normative schema. Buddy implements it under
`midend/include/Analysis` and `midend/lib/Analysis`; triton-riscv owns its
strict producer under `include/triton-shared/Analysis` and `lib/Analysis`.
Repositories exchange standard MLIR text and a handoff manifest — no shared
library or SDK. Conformance runs in each repo’s normal suite.

Producers check existing `contract.version` and every
`contract.compute_graph.version` before normalization. Absent module version
allows first annotation; existing versions must be integer 1. Unknown versions
are never rewritten to v1.

Strict producers accept only standard buffer IR, logical memory space zero,
and neutral facts. `plan.*`, target dialects/types, and unrealized casts are
rejected. Buddy’s planned-IR adapter may admit `plan.*`, keep logical space
zero, and skip producer normalization. Synthetic transfers use a shared
callback; an `origin_id` on a synthetic transfer does not complete incomplete
retained lineage. Only Buddy interprets `generated_by="planner-staging"`.
Verified incomplete graphs keep their limitations; verification is not planner
acceptance.
