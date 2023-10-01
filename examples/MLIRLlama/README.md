# BUDDY MLIR

MLIR-Based Ideas Landing Project ([Project page](https://buddy-compiler.github.io/)).

## Getting Started

The default build system uses LLVM/MLIR as an external library. 
We also provide a [one-step build strategy](#one-step) for users who only want to use our tools.
Please make sure [the dependencies](https://llvm.org/docs/GettingStarted.html#requirements) are available on your machine.

### LLVM/MLIR Dependencies

Before building, please make sure [the dependencies](https://llvm.org/docs/GettingStarted.html#requirements) are available
on your machine.

### Clone and Initialize

```
$ git clone git@github.com:buddy-compiler/buddy-mlir.git
$ cd buddy-mlir
$ git submodule update --init
```

### Build and Test LLVM/MLIR/CLANG

```
$ cd buddy-mlir
$ mkdir llvm/build
$ cd llvm/build
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang;lld;openmp" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_RUNTIMES=all \
    -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
    -DMLIR_ENABLE_BINDINGS_PYTHON=True \
    -DPython3_EXECUTABLE=/Path/To/Python/Interpreter \
    -DCMAKE_BUILD_TYPE=RELEASE
$ ninja check-mlir check-clang
```

If your target machine includes a Nvidia GPU, you can use the following configuration:

```
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV;NVPTX" \
    -DMLIR_ENABLE_CUDA_RUNNER=ON \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE
```

If your target machine has lld installed, you can use the following configuration:

```
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_USE_LINKER=lld \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE
```

### Build buddy-mlir

If you have previously built the llvm-project, you can replace the $PWD with the path to the directory where you have successfully built the llvm-project.

```
$ cd buddy-mlir
$ mkdir build
$ cd build
$ cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE
$ ninja
$ ninja check-buddy
```

### Install python packages

You should install transformers and torch with same versions in requirements.txt as possible.
```
transformers == 4.33.1
torch == 2.2.0
```

### Set environment variable

If you want to use c++ to call mlir function and get llvm openmp support, you should set environment variable.

```
export LD_LIBRARY_PATH=$HOME/buddy-mlir/llvm/build/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$HOME/buddy-mlir/llvm/build/lib:$LIBRARY_PATH
```

If you want to lower llama model to mlir, you should set PYTHONPATH, because we implement llama lower by using mlir pybind.
```
export PYTHONPATH=$HOME/buddy-mlir/llvm/build/tools/mlir/python_packages/mlir_core:${PYTHONPATH}
``` 

### run llama model lower

First, you should download llama model. You can get model from [meta ai](https://ai.meta.com/llama/)
And then in file torch_mlir_llama_hf.py, you should change '/llama-2-7B-hf' to your model path in your server.
```
tokenizer = LlamaTokenizer.from_pretrained('/llama-2-7B-hf')
model = LlamaForCausalLM.from_pretrained('/llama-2-7B-hf', torchscript=True)
```
There are many params in llama model, we get them from model and store in your disk. In the step lower llama.mlir for inference, params will be read for inference. You can change buddy/global_var.py to config the store location for params.
```
"params-write-path":"/buddy-mlir-for-transformer/examples/MLIRLlama"
```
Run torch_mlir_llama_hf.py, and then you can get the llama mlir output.
```
python torch_mlir_llama_hf.py > buddy/llama.mlir
```
In your path for "params-write-path", you will get params_data directory, such as "/buddy-mlir-for-transformer/examples/MLIRLlama/params_data", which will store model params.
### lower llama.mlir for inference

```
cd buddy
make llama-ompopt
make llama-make
./llamaRun
```
When you run llamaRun, you should provide the model params data path and vocab file. In our repo, we have provided vocab file and you should provide params data path in your server. You should input these path in command line.
```
please input vocab file path
/buddy-mlir/examples/MLIRLlama/vocab.txt
please input params file directory
/buddy-mlir/examples/MLIRLlama/params_data
```
we recommand you choose llama-ompopt to make. This will use openmp to accelarate inference. We also provide other choice in makefile to run inference.
Such as, llama-lower, it's nearly no optimization, llama-batchmatmulopt, it provides vectorize optimization for batchmatmul op.
