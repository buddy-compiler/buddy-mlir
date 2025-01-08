# Converting Polybench to MLIR

This guide demonstrates how to convert [Polybench](https://www.cs.colostate.edu/~pouchet/software/polybench/), the C language polyhedral benchmark suite, to MLIR arith, affine and memref dialects using [Polygeist](https://github.com/llvm/Polygeist). Some manual modifications to the source code may be required.

## Requirements: Prepare Polygeist

1. Clone the Polygeist repository:
   ```bash
   $ git clone --recursive https://github.com/llvm/Polygeist
   $ cd Polygeist
   $ git checkout 77c04bb
   ```
2. Build the project according to the instructions in the `README` file.

## Example: Converting `mvt` in Polybench

Follow the steps below to convert the `mvt` kernel in Polybench to MLIR:

1. **Modify the Source Code.**
   Since some statements such as `printf` and `assert` will introduce Polygeist custom dialects like `polygeist.pointer2memref` in the convertion, these parts of the code should be manually removed.
   - Locate the file `./tools/cgeist/Test/polybench/linear-algebra/kernels/mvt/mvt.c`, delete or comment out the `polybench_prevent_dce` function.
   - Locate the file `./tools/cgeist/Test/polybench/utilities/polybench.c`, delete or comment out the `assert` statement (i.e., `assert (tmp <= 10.0);`) in the `polybench_flush_cache` function.

2. **Run the Conversion Command.**
   Run the following command from the root directory of the Polygeist repository:
   ```bash
   $ ./build/bin/cgeist ./tools/cgeist/Test/polybench/linear-algebra/kernels/mvt/mvt.c \
   ./tools/cgeist/Test/polybench/utilities/polybench.c \
   -resource-dir=./llvm-project/build/lib/clang/18 \
   -D POLYBENCH_NO_FLUSH_CACHE -D MINI_DATASET \
   -I ./tools/cgeist/Test/polybench/utilities -O3 -S -o mvt-polygeist.mlir
   ```
> Notes: Replace `MINI_DATASET` to `SMALL_DATASET`, `STANDARD_DATASET`, `LARGE_DATASET`, or `EXTRALARGE_DATASET` to set different dataset sizes.

3. **Verify the Generated MLIR.**
   After running the command, check the generated `mvt-polygeist.mlir` file. It should contain only the standard MLIR arith, affine and memref dialects, with no Polygeist custom dialects like `polygeist.pointer2memref`.
