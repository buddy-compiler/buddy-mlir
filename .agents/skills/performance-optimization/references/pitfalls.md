# Pitfalls

Mistakes that produced wrong measurements or wrong conclusions in this
repository. Check them before trusting a result.

## The build did not contain the change

- `tools/buddy-codegen/import_model.py` puts `<repo>/build/python_packages`
  first on `sys.path`, whatever `--build-dir` is. A cross build in
  `build-xc` with a stale or symlinked `build/` imports the old
  `k3_w4.py`. Keep `build/` the native build of the same checkout, rebuild it
  after Python changes, and grep the generated
  `build-*/models/<model>/*.mlir` for the change.
- The model import is cached: delete
  `<build-dir>/models/<model>/.buddy_import_done` to force it. It does not
  depend on the Python sources.
- A board run uses whatever `.rax` is on the board: name files after the
  commit they come from and copy them again after every rebuild.

## Unfair comparisons

- Align the baseline's formats with ours before comparing speed. Example:
  llama.cpp's `Q4_0` GGUF keeps the LM head in `Q6_K` (60 MB more per decoded
  token than an int4 head); with `--output-tensor-type q4_0` the decode
  speeds were equal, and the "6% faster" claim disappeared.
- Compare accuracy too (perplexity): a faster but less accurate format is not
  a like-for-like win.
- Run each framework the way it is meant to run on the platform (some need a
  special process mode, others are slower in it); see the platform's board
  setup in the `hardware-targets` skill.

## Wrong estimates

- Do not reuse a gain measured in an earlier experiment without checking
  whether a merged change already took it. A DMA weight prefetch was
  estimated from a benchmark whose gain the TCM-activation change had
  already captured; the profile showed 3% left instead of 15-25%.
- A microbenchmark must use the same parameters as the model (thread count,
  heads per item, spec fields). A variant that silently ignored an option
  ran a different configuration and looked 60% slower.

## Tooling

- `pre-commit run --all-files` reformats hundreds of unrelated files: run it
  on the changed files only.
- A FileCheck line matches a prefix: anchor names that are prefixes of
  others with `{{$}}`. A filter on generated code that matches a word can
  also match comments.
- An ssh command that launches a long build must not sit in a retry loop:
  each retry starts another build. Launch once, detached, and poll.
