# Buddy Compiler DeepSeekR1 Example

## Introduction

This example shows how to use Buddy Compiler to compile a DeepSeekR1 model to MLIR code then run it.

## How to run on non-RISC-V device

0. Enter Python virtual environment.

We recommend you to use anaconda3 to create python virtual environment. You should install python packages as buddy-mlir/requirements.

```
$ conda activate <your virtual environment name>
$ cd buddy-mlir
$ pip install -r requirements.txt
```

1. Build and check LLVM/MLIR

```
$ cd buddy-mlir
$ mkdir llvm/build
$ cd llvm/build
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=$(which python3)
$ ninja check-clang check-mlir omp
```

2. Build and check buddy-mlir

```
$ cd buddy-mlir
$ mkdir build
$ cd build
$ cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
    -DPython3_EXECUTABLE=$(which python3)
$ ninja
$ ninja check-buddy
```

Set the `PYTHONPATH` environment variable. Make sure that the `PYTHONPATH` variable includes the directory of LLVM/MLIR python bindings and the directory of Buddy MLIR python packages.

```bash
$ export PYTHONPATH=/path-to-buddy-mlir/llvm/build/tools/mlir/python_packages/mlir_core:/path-to-buddy-mlir/build/python_packages:${PYTHONPATH}

// For example:
// Navigate to your buddy-mlir/build directory
$ cd buddy-mlir/build
$ export BUDDY_MLIR_BUILD_DIR=$PWD
$ export LLVM_MLIR_BUILD_DIR=$PWD/../llvm/build
$ export PYTHONPATH=${LLVM_MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
```

3. Set model environment variable.

```bash
$ export DEEPSEEKR1_MODEL_PATH=/path-to-deepseek-r1-model/

// For example:
$ export DEEPSEEKR1_MODEL_PATH=/home/xxx/DeepSeek-R1-Distill-Qwen-1.5B
```
Alternatively, you can leave the path blank, and import-deepseek-r1.py will automatically download the model for you.

4. Build and run the DEEPSEEKR1 example

```bash
$ cmake -G Ninja .. -DBUDDY_DEEPSEEKR1_EXAMPLES=ON

//f32
$ ninja buddy-deepseek-r1-run
$ ./bin/buddy-deepseek-r1-run

//f32 tiered-kv-cache (dynamic prefill & decode optimization)
$ ninja buddy-deepseek-r1-tiered-kv-cache-run
$ ./bin/buddy-deepseek-r1-tiered-kv-cache-run

// NUMA node binding
numactl --cpunodebind=0,1,2,3 --interleave=0,1,2,3 taskset -c 0-47 ./bin/buddy-deepseek-r1-run

//f16
$ ninja buddy-deepseek-r1-f16-run
$ ./bin/buddy-deepseek-r1-f16-run

//bf16
$ ninja buddy-deepseek-r1-bf16-run
$ ./bin/buddy-deepseek-r1-bf16-run
```

5. Streaming inference with buddy-deepseek-r1-cli

`buddy-deepseek-r1-cli` reuses the inference flow from the example but focuses on a real-time streaming experience:

```bash
$ ninja buddy-deepseek-r1-cli
$ echo "Hello." | ./bin/buddy-deepseek-r1-cli --max-tokens=128

# Enter interactive conversation
$ ./bin/buddy-deepseek-r1-cli --interactive --no-stats
# Enter one user message at a time; the CLI streams a reply immediately. Type :exit or :quit to finish the session
```

- By default the CLI looks for `examples/BuddyDeepSeekR1/vocab.txt` and `build/examples/BuddyDeepSeekR1/arg0.data`. Override these with `--vocab` and `--model` if you use custom paths.
- Provide prompts via `--prompt`, `--prompt-file`, or standard input. Generated text goes to STDOUT while logs and performance metrics go to STDERR.
- Use options like `--max-tokens` and `--eos-id` to constrain the generation length/termination. Add `--no-stats` when you want pure text output without the performance summary.
- `--interactive` starts a REPL similar to `buddy-deepseek-r1-main.cpp`, handling one prompt and one response at a time. `--prompt` can act as a system prefix prepended to every user entry.

6. Enjoy it!

## How to run on RISC-V machine

We can't utilize the ecosystem of Python on RISC-V 'cause some critical packages like pytorch are still unavailable on it. So we have to generate some files on X86 or ARM, then transfer those files to RISC-V machine and build.

1. Build LLVM on RISC-V:

```sh
cd buddy-mlir/llvm
mkdir build && cd build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=RELEASE
ninja check-clang check-mlir omp
```

Some test errors may occur when running check-mlir on RISC-V platforms, please ignore them.

2. Build Buddy-mlir on RISC-V:

```sh
cd buddy-mlir
mkdir build && cd build
cmake -G Ninja .. \
  -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
  -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=RELEASE
ninja
```

3. Generate the required files by following the instructions in the previous section *on X86 or ARM device*. This will create the necessary files for building the DeepSeekR1 executable:

```text
Files generated in `buddy-mlir/build/examples/BuddyDeepSeekR1/`:
forward_prefill.mlir
forward_decode.mlir
subgraph0_prefill.mlir
subgraph0_decode.mlir
arg0.data
```

**Recommended approach**: Create a compressed package for easy transfer:

```sh
# In buddy-mlir/build
ninja buddy-deepseek-r1-package
```

This creates `buddy-deepseek-r1.tar.zst` containing all necessary files for cross-platform deployment.

**Alternative approach**: Transfer the individual files listed above directly to your RISC-V device if you prefer not to use the compressed package.

4. Transfer the files to your RISC-V device and prepare for building:

```sh
# On RISC-V device
cd buddy-mlir/build
cmake -G Ninja .. -DBUDDY_DEEPSEEKR1_EXAMPLES=ON
```

**If using the compressed package (recommended)**:

```sh
# Transfer the package
rsync -avP --progress /path/to/buddy-deepseek-r1.tar.zst user@risc-v-host:buddy-mlir/build/examples/BuddyDeepSeekR1
```

Then extract on the RISC-V device:

```sh
# On RISC-V device
cd buddy-mlir/build/examples/BuddyDeepSeekR1
tar -I zstd -xvf buddy-deepseek-r1.tar.zst --strip-components=1
```

5. Build and run the model:

```sh
# in buddy-mlir/build
cd buddy-mlir/build/
ninja buddy-deepseek-r1-run
./bin/buddy-deepseek-r1-run
```

## How to cross compile RISC-V target on x86_64 machine

0. Clone code and initialize
  Refer to steps 0. Prepare `buddy-mlir` and Submodules in the [docs/RVVEnvironment.md](https://github.com/buddy-compiler/buddy-mlir/blob/main/docs/RVVEnvironment.md).

1. Build and test LLVM/MLIR
  Refer to steps 1. Build Local LLVM/MLIR in the [docs/RVVEnvironment.md](https://github.com/buddy-compiler/buddy-mlir/blob/main/docs/RVVEnvironment.md).

2. Build buddy-mlir
  Refer to steps 2. Build Local `buddy-mlir` in the [docs/RVVEnvironment.md](https://github.com/buddy-compiler/buddy-mlir/blob/main/docs/RVVEnvironment.md) and add `export BUDDY_MLIR_BUILD_DIR=$PWD` `export PYTHONPATH=${BUILD_LOCAL_LLVM_DIR}/tools/mlir/python_packages/mlir_core:${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}` after execution.

3. Build Cross-Compiled MLIR
  Refer to steps 4. Build Cross-Compiled MLIR in the [docs/RVVEnvironment.md](https://github.com/buddy-compiler/buddy-mlir/blob/main/docs/RVVEnvironment.md).
  Note: Add the `-DMLIR_IRDL_TO_CPP_EXE=${BUILD_LOCAL_LLVM_DIR}/bin/mlir-irdl-to-cpp` parameter at the end of cmake.
       Add `export BUDDY_DEEPSEEKR1_LLVMOPT=${BUILD_CROSS_MLIR_DIR}/lib/libLLVMSupport.a` `export RISCV_MLIR_C_RUNNER_UTILS=${BUILD_CROSS_MLIR_DIR}/lib/libmlir_c_runner_utils.so.22.0git` parameter after execution.

4. Pull the OpenMP shared library of RISC-V
   Since the repository depends on OpenMP shared libraries, follow the steps below to set up the OpenMP dependency:
```bash
$ cd ${BUILD_LOCAL_LLVM_DIR}/../
$ wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1XEsAhOcMioN9gdufuyO9OrHIdR0UtHh2' -O build-omp-shared-rv.tar.gz
$ mkdir build-omp-shared-rv && tar -xzf build-omp-shared-rv.tar.gz -C build-omp-shared-rv && rm build-omp-shared-rv.tar.gz
$ export RISCV_OMP_SHARED=${BUILD_LOCAL_LLVM_DIR}/../build-omp-shared-rv/libomp.so
```

5. Build for the target platform

```bash
$ cd buddy-mlir/build
$ cmake -G Ninja .. \
-DRISCV_GNU_TOOLCHAIN=${BUILD_RISCV_GNU_TOOLCHAIN_DIR} \
-DDEEPSEEKR1_EXAMPLE_PATH=. \
-DDEEPSEEKR1_EXAMPLE_BUILD_PATH=. \
-DBUDDY_DEEPSEEKR1_EXAMPLES=ON \
-DIS_RVV_CROSSCOMPILING=ON \
-DRISCV_OMP_SHARED=${RISCV_OMP_SHARED} \
-DRISCV_MLIR_C_RUNNER_UTILS=${RISCV_MLIR_C_RUNNER_UTILS} \
-DBUILD_CROSS_MLIR_DIR=${BUILD_CROSS_MLIR_DIR} \
-DBUDDY_MLIR_BUILD_DIR=${BUDDY_MLIR_BUILD_DIR}
$ ninja <target> # For example: `ninja buddy-deepseek-r1-run`
```

Compile, package, and run
```bash
# f32
$ ninja buddy-deepseek-r1-run
$ ninja buddy-deepseek-r1-rvv-package
$ scp ./examples/BuddyDeepSeekR1/buddy-deepseek-r1-rvv-package.tgz user@risc-v-host:
# On risc-v-host
rv-user$ tar xzf buddy-deepseek-r1-rvv-package.tgz && cd buddy-deepseek-r1-rvv-package
rv-user$ export LD_LIBRARY_PATH=${PWD}:$LD_LIBRARY_PATH
rv-user$ ./buddy-deepseek-r1-run

# f16
$ ninja buddy-deepseek-r1-f16-run
$ ninja buddy-deepseek-r1-f16-rvv-package
$ scp ./examples/BuddyDeepSeekR1/buddy-deepseek-r1-f16-rvv-package.tgz user@risc-v-host:
# On risc-v-host
rv-user$ tar xzf buddy-deepseek-r1-f16-rvv-package.tgz && cd buddy-deepseek-r1-f16-rvv-package
rv-user$ export LD_LIBRARY_PATH=${PWD}:$LD_LIBRARY_PATH
rv-user$ ./buddy-deepseek-r1-f16-run

# bf16
$ ninja buddy-deepseek-r1-bf16-run
$ ninja buddy-deepseek-r1-bf16-rvv-package
$ scp ./examples/BuddyDeepSeekR1/buddy-deepseek-r1-bf16-rvv-package.tgz user@risc-v-host:
# On risc-v-host
rv-user$ tar xzf buddy-deepseek-r1-bf16-rvv-package.tgz && cd buddy-deepseek-r1-bf16-rvv-package
rv-user$ export LD_LIBRARY_PATH=${PWD}:$LD_LIBRARY_PATH
rv-user$ ./buddy-deepseek-r1-bf16-run

# cli
$ ninja buddy-deepseek-r1-cli
$ ninja buddy-deepseek-r1-cli-rvv-package
$ scp ./examples/BuddyDeepSeekR1/buddy-deepseek-r1-cli-rvv-package.tgz user@risc-v-host:
# On risc-v-host
rv-user:~$ tar xzf buddy-deepseek-r1-cli-rvv-package.tgz && cd buddy-deepseek-r1-cli-rvv-package
rv-user:~$ export LD_LIBRARY_PATH=${PWD}:$LD_LIBRARY_PATH
rv-user:~$ echo "Hello." | ./buddy-deepseek-r1-cli --no-stats 

# kv
$ ninja buddy-deepseek-r1-tiered-kv-cache-run
$ ninja buddy-deepseek-r1-tiered-kv-cache-run-rvv-package
$ scp /examples/BuddyDeepSeekR1/buddy-deepseek-r1-tiered-kv-cache-run-rvv-pkg.tgz user@risc-v-host:
# On risc-v-host
rv-user:~$ tar xzf buddy-deepseek-r1-tiered-kv-cache-run-rvv-pkg.tgz && cd buddy-deepseek-r1-tiered-kv-cache-run-rvv-pkg
rv-user:~$ export LD_LIBRARY_PATH=${PWD}:$LD_LIBRARY_PATH
rv-user:~$ ./buddy-deepseek-r1-tiered-kv-cache-run
```
