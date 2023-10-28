# Buddy Compiler LLaMA Example

1. Download llama2 model

You should download llama model. You can get model from [meta ai](https://ai.meta.com/llama/).

2. Python virtual environment

We recommend you to use anaconda3 to create python virtual environment. You should install python packages as buddy-mlir/requirements.

3. llama2 model convert to hugging face format

You should convert llama2 model which download from meta ai to hugging face format. Because we use hugging face api to get llama2 model.

```
python llama2-to-hf.py --input_dir path-to-llama2-model --model_size 7B --output_dir path-to-save-llama-hf-model
```

Such as you have a 7B llama2 model, in your input_dir path-to-llama-model, you should have a tokenizer.model and a directory named "7B". You should put your 7B llama2 model inside the "7B" directory.

In addition, you should set your output_dir into test-llama2.py.
```
tokenizer = LlamaTokenizer.from_pretrained('path-to-llama2-hf-model')
model = LlamaForCausalLM.from_pretrained('path-to-llama2-hf-model', torchscript=True)
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
Set environment variable.
```
export LD_LIBRARY_PATH=$HOME/path-to-buddy-mlir/llvm/build/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$HOME/path-to-buddy-mlir/llvm/build/lib:$LIBRARY_PATH
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
Set environment variable.
```
export PYTHONPATH=/path-to-buddy-mlir/llvm/build/tools/mlir/python_packages/mlir_core:/path-to-buddy-mlir/build/python_packages:${PYTHONPATH}
```

6. Build and run LLaMA example

```
$ cmake -G Ninja .. -DBUDDY_LLAMA_EXAMPLES=ON
$ ninja buddy-llama-run
$ cd bin
$ ./buddy-llama-run
```
This build will spend a few minutes. We recommend you to use better cpu such as server-level cpu to run buddy-llama-run.