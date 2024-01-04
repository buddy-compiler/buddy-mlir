# Buddy Compiler LLaMA Example

1. Download LLaMA2 model

You should download llama model. You can get model from [meta ai](https://ai.meta.com/llama/).

2. Enter Python virtual environment

We recommend you to use anaconda3 to create python virtual environment. You should install python packages as buddy-mlir/requirements.

```
$ conda activate <your virtual environment name>
$ cd buddy-mlir
$ pip install -r requirements.txt
```

3. LLaMA2 model convert to HuggingFace format

You should convert LLaMA2 model which download from meta ai to HuggingFace format. Because we use HuggingFace api to get LLaMA2 model.

```
$ cd examples/BuddyLlama
$ python llama2-to-hf.py --input_dir path-to-llama2-model --model_size 7B --output_dir path-to-save-llama-hf-model
```

Such as you have a 7B LLaMA2 model, in your input_dir path-to-llama-model, you should have a tokenizer.model and a directory named "7B". You should put your 7B LLaMA2 model inside the "7B" directory.

In addition, set an environment variable for the generated LLaMA model.
```
$ export LLAMA_MODEL_PATH=/path-to-save-llama-hf-model/
```

4. Build and check LLVM/MLIR

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

5. Build and check buddy-mlir

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

```
$ export PYTHONPATH=/path-to-buddy-mlir/llvm/build/tools/mlir/python_packages/mlir_core:/path-to-buddy-mlir/build/python_packages:${PYTHONPATH}

// For example:
// Navigate to your buddy-mlir/build directory
$ cd buddy-mlir/build
$ export BUDDY_MLIR_BUILD_DIR=$PWD
$ export LLVM_MLIR_BUILD_DIR=$PWD/../llvm/build
$ export PYTHONPATH=${LLVM_MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
```

6. Build and run LLaMA example

```
$ cmake -G Ninja .. -DBUDDY_LLAMA_EXAMPLES=ON
$ ninja buddy-llama-run
$ cd bin
$ ./buddy-llama-run
```
This build will spend a few minutes. We recommend you to use better cpu such as server-level cpu to run buddy-llama-run.

If you wish to utilize `mimalloc` as a memory allocator, you need to set `BUDDY_MLIR_USE_MIMALLOC` and `MIMALLOC_BUILD_DIR`.
For more details, please see [here](../../thirdparty/README.md#the-mimalloc-allocator).
