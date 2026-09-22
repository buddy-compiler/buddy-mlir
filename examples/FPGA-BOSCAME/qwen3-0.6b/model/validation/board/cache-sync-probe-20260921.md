# Cache-Sync Probe (2026-09-21)

This probe separates the unverified RA CBO path from the validated NR runtime.
It used the same 1-layer graph, Triton archive, adapters, weights, tokenizer,
prompt, and FPGA5 platform for both variants.

## RA CBO diagnostic variant

Build output:

```text
model/build/cache-sync-probe-20260921/1l-workspace/
```

Options were `--layers 1 --cache-len 512 --prefill-len 16 --decode-steps 8
--ame-cache-sync workspace`, with all other synchronization options at their
defaults (`ame-startup=none`, graph-final `ame_fence()`). The ELF audit passed,
and the generated source contained range walks using `cbo.flush` and
`cbo.inval`.

FPGA run `run-8db71d9db7324543` used image SHA256
`4ac6dc1ef941f57a8f811a939c409e89736aec9f8e04b307aa7fda47f067f136`.
All three DDR readbacks matched. UART stopped at:

```text
[model] prefill begin position=00000000 input_token=0002505C
```

After 522 seconds there was no completion marker, trap, or model result. The
worker was stopped explicitly and reported `INTERRUPTED`; it did not emit
`mcause=2`. This is evidence that direct RA CBO range walking is unusable or
prohibitively slow on this platform, not evidence that CBO fixed the model hang.

## Default runtime regression

The same 1-layer inputs were rebuilt with the default `--ame-cache-sync none`.
The generated model source had no cache-sync calls; image SHA256 was
`f91dc984049aa43542ba457d15a0bbf52bed779a820944f6c89993e25c6616fc`.

FPGA run `run-51b9fa5154c0468b` loaded all three segments with matching
readbacks and completed all 8 decode steps. The UART log ended with:

```text
[text] output bytes=... internallyimizeimizeimizeimizeimizeimizeimizeimize
verify fixed text validation: PASS
[nr] RA returned: PASS
verify NR runtime: PASS
```

The 1-layer text is expected to be numerically different from the 28-layer
model; the relevant result is that the graph, decode loop, and completion
protocol all finished successfully.

## Runtime fix

`nr_getchar()` no longer emits RA-side `cbo.inval`; it uses ordered volatile
loads/stores and `fence rw,rw`, while NH remains the owner of UART MMIO and
cache maintenance. This matches the platform's validated transport contract.
The explicit `nr_ame_cache_clean/invalidate` hooks remain opt-in diagnostics and
are not used by production images.

