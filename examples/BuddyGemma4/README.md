# Buddy Compiler Gemma4 Example

## Introduction

This example shows how to use Buddy Compiler to compile a Gemma4-E2B model to MLIR code then run it.

Gemma4 is a multimodal model with hybrid attention (sliding window + full attention) and shared KV cache layers. This example extracts the text (CausalLM) portion and runs end-to-end inference.

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
$ export PYTHONPATH=/path-to-buddy-mlir/build/python_packages:${PYTHONPATH}

// For example:
// Navigate to your buddy-mlir/build directory
$ cd buddy-mlir/build
$ export BUDDY_MLIR_BUILD_DIR=$PWD
$ export LLVM_MLIR_BUILD_DIR=$PWD/../llvm/build
$ export PYTHONPATH=${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
```

3. Set model environment variable.

```bash
$ export GEMMA4_E2B_MODEL_PATH=/path-to-gemma4-model/

// For example:
$ export GEMMA4_E2B_MODEL_PATH=/home/xxx/gemma-4-E2B-it
```
Alternatively, you can leave the path blank, and import-gemma4.py will automatically download the model for you.

4. Build and run the Gemma4 example

```bash
$ cmake -G Ninja .. -DBUDDY_GEMMA4_EXAMPLES=ON

// f32
$ ninja buddy-gemma4-e2b-run
$ ./bin/buddy-gemma4-e2b-run

// NUMA node binding
numactl --cpunodebind=0,1,2,3 --interleave=0,1,2,3 taskset -c 0-47 ./bin/buddy-gemma4-e2b-run
```

5. Enjoy it!

## How to run on RISC-V machine

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

3. Generate the required files by following the instructions in the previous section *on X86 or ARM device*. This will create the necessary files for building the Gemma4 executable:

```text
Files generated in `buddy-mlir/build/examples/BuddyGemma4/`:
forward_prefill_e2b.mlir
forward_decode_e2b.mlir
subgraph0_prefill_e2b.mlir
subgraph0_decode_e2b.mlir
arg0_e2b.data
vocab.txt
```

**Recommended approach**: Create a compressed package for easy transfer:

```sh
# In buddy-mlir/build
ninja buddy-gemma4-e2b-package
```

This creates `buddy-gemma4-e2b.tar.zst` containing all necessary files for cross-platform deployment.

**Alternative approach**: Transfer the individual files listed above directly to your RISC-V device if you prefer not to use the compressed package.

4. Transfer the files to your RISC-V device and prepare for building:

```sh
# On RISC-V device
cd buddy-mlir/build
cmake -G Ninja .. -DBUDDY_GEMMA4_EXAMPLES=ON
```

**If using the compressed package (recommended)**:

```sh
# Transfer the package
rsync -avP --progress /path/to/buddy-gemma4-e2b.tar.zst user@risc-v-host:buddy-mlir/build/examples/BuddyGemma4
```

Then extract on the RISC-V device:

```sh
# On RISC-V device
cd buddy-mlir/build/examples/BuddyGemma4
tar -I zstd -xvf buddy-gemma4-e2b.tar.zst --strip-components=1
```

5. Build and run the model:

```sh
# In buddy-mlir/build
cd buddy-mlir/build
ninja buddy-gemma4-e2b-run
./bin/buddy-gemma4-e2b-run
```
