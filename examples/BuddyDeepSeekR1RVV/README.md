## Running on RVV Target

To run the DeepSeekR1 example on a RISC-V Vector (RVV) machine, make sure you have first set up the RVV cross-compilation environment as described in [RVVEnvironment.md](https://github.com/buddy-compiler/buddy-mlir/blob/main/docs/RVVEnvironment.md).

Two additional executables are provided for RVV:

* `buddy-deepseek-r1-rvv-run` (f32)
* `buddy-deepseek-r1-f16-rvv-run` (f16)

Make sure the following `.so` libraries are available in your directory and linked correctly:

* `libmlir_c_runner_utils.so`
* `libmlir_float16_utils.so` (only for f16)
* `libomp.so`

Before running, set the `LD_LIBRARY_PATH` environment variable to include the directory containing these `.so` files. For example:

```bash
export LD_LIBRARY_PATH=/path-to-libs:$LD_LIBRARY_PATH
```

Also, place all required files in the same directory as the executables:

* `.data` files
* `vocab.txt`

Build and run the DEEPSEEKR1 example on RVV Target

```bash
$ cmake -G Ninja .. -DBUDDY_DEEPSEEKR1_RVV_EXAMPLES=ON

//f32
$ ninja buddy-deepseek-r1-rvv-run
$ cd bin
$ ./buddy-deepseek-r1-rvv-run

//f16
$ ninja buddy-deepseek-r1-f16-rvv-run
$ cd bin
$ ./buddy-deepseek-r1-f16-rvv-run
```
```
