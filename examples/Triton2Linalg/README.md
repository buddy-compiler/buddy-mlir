`Triton2Linalg` directory now owns the full pipeline end-to-end for all 4 examples:

1. Trigger RuyiAI-Stack/triton-riscv compilation and dump `tt.mlir`, `ttshared.mlir`, `ll.mlir`, `ll.ir`.
2. Generate `_ttshared_verify_out.mlir` from `ttshared.mlir` with `buddy-opt --verify-each`.
3. Generate `ttshared-main.mlir` by appending per-kernel test `func.func @main() -> i32` to `_ttshared_verify_out.mlir`.
4. Run `--empty-tensor-to-alloc-tensor` + `--one-shot-bufferize=allow-return-allocs-from-loops=true` pass to emit `linalg_bufferized_no_vec_no_loops.mlir`.
5. Collect `matmul`, `softmax`, `layernorm` and `vecadd` cases for now.
