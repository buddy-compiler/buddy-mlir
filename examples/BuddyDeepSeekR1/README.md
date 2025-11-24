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

**If using the compressed package (recommended)**:

```sh
# Transfer the package
rsync -avP --progress /path/to/buddy-deepseek-r1.tar.zst user@risc-v-host:<path-to-buddy-mlir>/build/examples/BuddyDeepSeekR1
```

Then extract on the RISC-V device:

```sh
# On RISC-V device
cd buddy-mlir/build/examples/BuddyDeepSeekR1
tar -I zstd -xvf buddy-deepseek-r1.tar.zst --strip-components=1
```

5. Modify `CMakeLists.txt`:

  - For convenience, you could delete all unrelated compile options in the file first. Say, if you're going to build FP32, then delete all compile options about FP16 or BF16.
  - Add `-mattr=+m,+d,+v` for *all* `llc` command like this: `${LLVM_TOOLS_BINARY_DIR}/llc -mattr=+m,+d,+v -filetype=obj -relocation-model=pic -O3`.
  - Note: Since the files are extracted to the build directory, you don't need to change the dependent directory paths (they will use `${CMAKE_CURRENT_BINARY_DIR}` by default).
  - Add `arch` and `abi` related compile options for the main file. Just copy and paste these lines to CMakeLists:

    ```cmake
      target_compile_options(buddy-deepseek-r1-run PRIVATE
        -march=rv64gcv
        -mabi=lp64d
        -O3
        -Wall
      )

      target_link_options(buddy-deepseek-r1-run PRIVATE
        -march=rv64gcv
        -mabi=lp64d
      )
    ```

  The complete modified CMakeLists file is attached in appendix, you could copy and paste it directly.

6. Build and run the model:

```sh
# in buddy-mlir/build
cmake -G Ninja .. -DBUDDY_DEEPSEEKR1_EXAMPLES=ON
ninja buddy-deepseek-r1-run
cd bin
./buddy-deepseek-r1-run
```

## Appendix: The complete CMakeLists file for building on RISC-V

```cmake
set(BUFFERIZE_FULL_OPTS "unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries")
set(BUFFERIZE_SIMPLE_OPTS "bufferize-function-boundaries")
set(TOSA_PIPELINE "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))")

add_custom_command(
  OUTPUT forward_prefill.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_SOURCE_DIR}/forward_prefill.mlir
            -simplify-tosa-reshape |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -pass-pipeline ${TOSA_PIPELINE} |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -matmul-vectorization-blis
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp=num-threads=32
            -cse
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
        ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
        ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
        ${LLVM_TOOLS_BINARY_DIR}/llc -mattr=+m,+d,+v -filetype=obj -relocation-model=pic -O3
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_SOURCE_DIR}/forward_prefill.mlir
  COMMENT "Building forward_prefill.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph_prefill.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_SOURCE_DIR}/subgraph0_prefill.mlir
              -simplify-tosa-reshape |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
              -pass-pipeline ${TOSA_PIPELINE} |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -convert-elementwise-to-linalg
            -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -matmul-vectorization-blis
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp=num-threads=32
            -func-bufferize-dynamic-offset
            -cse
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -mattr=+m,+d,+v -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph_prefill.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_SOURCE_DIR}/subgraph0_prefill.mlir
    COMMENT "Building subgraph_prefill.o "
    VERBATIM)

add_custom_command(
  OUTPUT forward_decode.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_SOURCE_DIR}/forward_decode.mlir
            -simplify-tosa-reshape |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -pass-pipeline ${TOSA_PIPELINE} |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -matmul-vectorization-blis
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp=num-threads=32
            -cse
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
        ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
        ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
        ${LLVM_TOOLS_BINARY_DIR}/llc -mattr=+m,+d,+v -filetype=obj -relocation-model=pic -O3
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_decode.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_SOURCE_DIR}/forward_decode.mlir
  COMMENT "Building forward_decode.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph_decode.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_SOURCE_DIR}/subgraph0_decode.mlir
              -simplify-tosa-reshape |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
              -pass-pipeline ${TOSA_PIPELINE} |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -convert-elementwise-to-linalg
            -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -matmul-vectorization-blis
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp=num-threads=32
            -func-bufferize-dynamic-offset
            -cse
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -mattr=+m,+d,+v -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph_decode.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_SOURCE_DIR}/subgraph0_decode.mlir
    COMMENT "Building subgraph_decode.o "
    VERBATIM)

add_library(DEEPSEEKR1 STATIC forward_prefill.o subgraph_prefill.o forward_decode.o subgraph_decode.o)

SET_SOURCE_FILES_PROPERTIES(
  template.o
  PROPERTIES
  EXTERNAL_OBJECT true
  GENERATED true)

SET_TARGET_PROPERTIES(
  DEEPSEEKR1
  PROPERTIES
  LINKER_LANGUAGE C)

add_executable(buddy-deepseek-r1-run buddy-deepseek-r1-main.cpp)

target_compile_options(buddy-deepseek-r1-run PRIVATE
  -march=rv64gcv
  -mabi=lp64d
  -O3
  -Wall
)

target_link_options(buddy-deepseek-r1-run PRIVATE
  -march=rv64gcv
  -mabi=lp64d
)

set(DEEPSEEKR1_EXAMPLE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(DEEPSEEKR1_EXAMPLE_BUILD_PATH ${CMAKE_CURRENT_BINARY_DIR})

target_compile_definitions(buddy-deepseek-r1-run PRIVATE
  DEEPSEEKR1_EXAMPLE_PATH="${DEEPSEEKR1_EXAMPLE_PATH}/"
  DEEPSEEKR1_EXAMPLE_BUILD_PATH="${DEEPSEEKR1_EXAMPLE_BUILD_PATH}/"
)

target_link_directories(buddy-deepseek-r1-run PRIVATE ${LLVM_LIBRARY_DIR})

set(BUDDY_DEEPSEEKR1_LIBS
  DEEPSEEKR1
  mlir_c_runner_utils
  omp
)
if(BUDDY_MLIR_USE_MIMALLOC)
  list(APPEND BUDDY_DEEPSEEKR1_LIBS mimalloc)
endif()

target_link_libraries(buddy-deepseek-r1-run ${BUDDY_DEEPSEEKR1_LIBS})
```
