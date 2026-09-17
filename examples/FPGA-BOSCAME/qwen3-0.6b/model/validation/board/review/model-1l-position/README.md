# Runtime-length attention: one-layer FPGA checkpoint

Run `run-9173a2481e114612` completed on FPGA5: one real Buddy graph prefill of
16 tokens and eight continuous decode steps at positions 16..23. All 27 complete
last-position vocabulary logits / effective K/V checks have max/mean error zero
against the independent `nr-fpga` arithmetic reference. The prefill prediction
is 33067; eight decode predictions are 11853. Reference data is validation only.

`numeric-verification.json` is `FULL_LOGITS_KV_PASS`, with reference metadata
verified. `kernel-profile-verification.json` is `KERNEL_PROFILE_PASS`: actual
LLVM entry calls and C adapters independently imply 40 kernel calls per graph
invocation. No test launch or oracle is linked into the Triton static archive.

`verification.json` ties the deployment/ELF hashes, uploaded boot hash, matching
DDR readbacks, and exact oracle bytes embedded in the executed ELF. Large
ELF/bin/reference arrays stay under `build/`, addressed by path and SHA256.
The original generated launch/profile sources, ABI adapter, replacement report,
link map, ELF audit, build command log and UART/worker result are saved here.

Graph-only cycles: prefill 949,799,911; decode 685,809,105..686,344,451; total
6,438,220,614. The preceding full-capacity baseline totaled 7,456,607,050.
The optimized run additionally uses profiling wrappers and completion fences,
so this is not a controlled uninstrumented throughput comparison. The K layout
kernel alone consumes approximately 395 million cycles per graph call. These
numbers do not include tokenizer, setup, validation, UART or cache retention.

The run resumed the existing remote worker after an SSH outage and exited
normally. This result does not establish four-/28-layer FPGA acceptance,
intermediate hidden-state validation, or physical UART input.
