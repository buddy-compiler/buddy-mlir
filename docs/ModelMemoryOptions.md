# Model Memory Options

Two opt-in fields of the model spec of `tools/buddy-codegen` models
(`MODEL_KIND` `llm_prefill_decode`, e.g. DeepSeek R1) change how the model
library and the generated `ModelSession` handle memory:

```json
{
  "variant": "f32",
  "max_token_len": 1024,
  "prefill_chunk": 64,
  "arena": true,
  "hugepages": true
}
```

Both default to `false`; without them nothing changes.

## `hugepages`

The session asks for 2 MiB pages (`madvise(MADV_HUGEPAGE)`) for the weight
buffers before it reads the weights in. The buffers are allocated with
`malloc`, so when transparent huge pages are in `madvise` mode (a common
default, e.g. on Ubuntu and on the SpacemiT K3 kernel;
`/sys/kernel/mm/transparent_hugepage/enabled`) they otherwise get 4 KiB pages.
Decode reads all the weights for every token, and with 4 KiB pages it loses
bandwidth to TLB misses. It has no effect with THP in `always` mode (the
weights already get huge pages) or `never` mode.

## `arena`

Every buffer the model allocates during a forward call comes from a bump
arena in the model library, and nothing is freed during the call:

- `compile_pipeline.py` lowers the allocations to the generic MLIR allocation
  functions (`-finalize-memref-to-llvm=use-generic-functions=true`) and leaves
  out the buffer deallocation passes (`-ownership-based-buffer-deallocation`
  and the passes that simplify and lower its result);
- `runtime/arena/BuddyArena.c`, linked into the model library, implements
  those functions: an allocation bumps a pointer in one reserved address
  range, a free does nothing;
- the session calls `buddy_arena_reset()` (exported by the model library)
  before each forward call: by then it has copied the results of the previous
  call into its own buffers. It releases the results instead of freeing them.

This removes `malloc` / `free` from the forward calls, and the runtime
ownership checks of the deallocation passes. The price is memory: the arena
holds all the buffers of a call at once, so it needs as much as the call
allocates in total, not its peak. The touched pages stay mapped and the
following calls reuse them.

Requirements:

- `prefill_chunk` (docs/ChunkedPrefill.md). One prefill call over
  `max_token_len` positions allocates too much: 42 GiB for f32
  DeepSeek-R1-Distill-Qwen-1.5B (`gen_config.py` rejects `arena` without
  `prefill_chunk`);
- not with `tiered_kv_cache` or with layer partitioning.

Environment variables read by the model library:

| Variable | Default | Meaning |
| --- | --- | --- |
| `BUDDY_ARENA_RESERVE_MB` | 16384 | Address range reserved (only touched pages use memory). A call that needs more stops with a message. |
| `BUDDY_ARENA_PREFAULT_MB` | 512 | Pages faulted in when the library is loaded, instead of during the first call, whose threads would otherwise serialize on page faults (K3, int4 DeepSeek R1: the first 64-token prefill call 0.69 -> 0.57 s). `0` turns it off. |
| `BUDDY_ARENA_STATS` | unset | `1`: print the largest call's usage at exit. |

### When it pays off

The arena suits graphs whose forward calls allocate little, such as graphs
whose layers are calls to hand-written kernels. It does not suit the f32
graphs of `import_model.py`, whose prefill calls allocate several GiB.

DeepSeek-R1-Distill-Qwen-1.5B, `buddy-cli`, greedy, prompt of 458 tokens. The
generated text is the same in each pair.

| Build | Arena per call | Prefill | Decode | Peak RSS |
| --- | --- | --- | --- | --- |
| x86, f32, `prefill_chunk` 64 | – | 6.9 s | 12.1 tok/s | 7.1 GB |
| the same with `arena` and `hugepages` | 8.9 GiB | 10.2 s | 11.4 tok/s | 16.2 GB |
| SpacemiT K3, int4 kernels, `prefill_chunk` 64, `hugepages` | – | 1.93 s | 22.0 tok/s | |
| the same with `arena` | 289 MiB | 1.69 s | 23.3 tok/s | |

x86: Xeon Platinum 8575C, 48 threads, mean of two runs (decode varies by
about 10% from run to run on this shared machine). K3: the int4 build of
`tools/buddy-codegen/k3` on the 8 A100 cores (not yet upstream), two
runs each.

`hugepages` alone, x86, f32 without chunks, the same prompt and a short one:
decode 10.9 -> 11.5 tok/s and 9.0 -> 10.6 tok/s.
