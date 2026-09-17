# Reading an incomplete model run

`summarize_progress.py` reads local snapshots from an image built with
`--profile-kernels --profile-progress`. It does not connect to the FPGA, restart
anything, or declare numerical success. Use the normal runner to recover the
existing session and obtain the UART log first.

```bash
MODEL=examples/FPGA-BOSCAME/qwen3-0.6b/model
python3 -B "$MODEL/optimization/summarize_progress.py" \
  --build "$MODEL/build/review-native-28l" --layers 28 \
  --profile <actual-image-directory>/kernel-profile.json \
  --uart-log <existing-run-directory>/uart.raw.log \
  --output <new-progress-snapshot.json>
```

The script validates unconditional, acyclic calls in the actual lowered LLVM
entry, proves call order using its control-flow graph, checks the RAW-to-CIFACE
bridges, and resolves the generated adapters to linked Triton kernel symbols.
For this model it requires one embedding, 28 identical 34-call decoder blocks,
then final norm and the three lm_head calls: **957 kernel calls per graph**.
It refuses a changed structure instead of inferring a layer from a symbol suffix.
Decoder block indices and the call index within each block are zero based.

`unpaired_begin` identifies the last entered kernel for which no matching end
has appeared in the snapshot. It does **not** establish a hang. A kernel end is
printed only after the kernel, the profiler's added `ame_fence()`, and counter
bookkeeping. A gap between an end and the next begin can be graph-internal
copies, allocation, RoPE, scalar work, or adapters. UART itself can also lag.
Kernel progress records do not contain timestamps; compare dated snapshots to
establish that output advances or has stayed unchanged during a measured window.

The existing profiler adds an AME completion operation after every Triton call
and UART before/after each call. This is the same instrumentation mode that
completed the accepted four-layer run `run-b8d3ed3ead854a24`; it differs from the
optimized one-layer image with selected intermediate comparisons. Do not use
their raw graph-cycle totals as an uninstrumented throughput comparison.

The parser was checked against all nine stages of that accepted four-layer UART,
partial logs ending in a kernel begin or an incomplete line, wrong call indices,
wrong end symbols, conditional LLVM calls, and the actual 28-layer LLVM/adapter
sequence. Final acceptance still requires `archive_model_run.py` and its
numerical, provenance, tokenizer, text, and profile validators.

## Separating kernel execution from completion synchronization

For a reproducible incomplete begin/end pair, build a **new** image with the
same graph, archive, weights and synchronization, adding for example:

```bash
--profile-kernels --profile-progress \
--profile-probe _mlir_ciface_kernel_dequantize_1x1024:12
```

The index is zero based, per symbol, and resets at each graph invocation.
Only the selected occurrence prints its actual descriptors before calling the
kernel, `returned` immediately after the real call, and `synced` after the
existing public `ame_fence()`. No fence or kernel instruction is removed.
Descriptor addresses, offsets, sizes and strides can be checked against the
image's workspace plan. The extra returned/synced UART is inside that selected
call's measured kernel cycles; do not use this probe as a throughput benchmark.
Additional logging and the relink can affect execution timing/layout, so a
successful probe run alone does not explain an earlier stall.
