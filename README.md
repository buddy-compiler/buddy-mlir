# BUDDY MLIR

An MLIR-based compiler framework designed for a co-design ecosystem from DSL (domain-specific languages) to DSA (domain-specific architectures). ([Project page](https://buddy-compiler.github.io/))

## Getting Started

### Dependencies

- **LLVM/MLIR dependencies**

Please make sure [the dependencies](https://llvm.org/docs/GettingStarted.html#requirements) are available
on your machine.

- **Other dependencies**

```
sudo apt install flatbuffers-compiler libflatbuffers-dev libnuma-dev
```

### Clone and Initialize

```
$ git clone git@github.com:buddy-compiler/buddy-mlir.git
$ cd buddy-mlir
$ git submodule update --init llvm
```

### Prepare Python Environment

```
$ conda activate <your virtual environment name>
$ cd buddy-mlir
$ pip install -r requirements.txt
```

### Build and Test LLVM/MLIR/CLANG

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

If your target machine includes an NVIDIA GPU, you can add the following configuration:

```
-DLLVM_TARGETS_TO_BUILD="host;RISCV;NVPTX" \
-DMLIR_ENABLE_CUDA_RUNNER=ON \
```

### Build buddy-mlir

```
$ cd buddy-mlir
$ mkdir build
$ cd build
$ cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
    -DPython3_EXECUTABLE=$(which python3)
$ ninja
$ ninja check-buddy
```

Set the `PYTHONPATH` environment variable to include both the LLVM/MLIR Python bindings and `buddy-mlir` Python packages:

```
$ export BUDDY_MLIR_BUILD_DIR=$PWD
$ export LLVM_MLIR_BUILD_DIR=$PWD/../llvm/build
$ export PYTHONPATH=${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
```

If you want to test your model end-to-end conversion and inference, you can add the following configuration

```
$ cmake -G Ninja .. -DBUDDY_ENABLE_E2E_TESTS=ON
$ ninja check-e2e
```

### Building and running the model

Use the following to build:

```bash
cd buddy-mlir
python3 tools/buddy-codegen/build_model.py \
  --spec models/deepseek_r1/specs/f32.json \
  --build-dir build
```

To import weights from a **local** HuggingFace style directory (offline or a custom path), pass `--local-model` to that directory (it must contain `config.json` and the weight files). If you omit `--hf-config`, `build_model.py` uses `<local-model>/config.json` for codegen when present:

```bash
python3 tools/buddy-codegen/build_model.py \
  --spec models/deepseek_r1/specs/f32.json \
  --build-dir build \
  --local-model /path/to/DeepSeek-R1-Distill-Qwen-1.5B
```

If CMake is configured with `-DBUDDY_BUILD_DEEPSEEK_R1_MODEL=ON`, you can build the model with:

```bash
ninja deepseek_r1_model_so deepseek_r1_rax buddy-cli
```

```bash
./build/bin/buddy-cli \
  --model ./build/models/deepseek_r1/deepseek_r1.rax \
  --prompt "Tell me a joke in 200 words."

# Equivalent to: numactl --cpunodebind=0,1,2,3 --interleave=0,1,2,3 taskset -c 0-47
./build/bin/buddy-cli \
  --numa 0,1,2,3 \
  --cpus 0-47 \
  --model ./build/models/deepseek_r1/deepseek_r1.rax \
  --prompt "Tell me a joke in 200 words."
```

## Build Python Package

We use `setuptools` to bundle CMake outputs (Python packages, `bin/`, and
`lib/`) into a single wheel.

Build x86_64 artifacts:

```bash
./scripts/release_wheel_manylinux.sh cp310-cp310 x86_64
```

Build riscv64 artifacts:

```bash
./scripts/release_wheel_manylinux.sh cp310-cp310 riscv64
```

This script calls `docker run` internally to enter the offical manylinux container,
builds LLVM and buddy_mlir, and writes artifacts to:

- `./build-docker/x86_64/<py_tag>/target`
- `./build-docker/riscv64/<py_tag>/target`

See [Manylinux release notes](./docs/ManylinuxReleaseNotes.md) for current
known build notes.

Install and test the wheel:

```bash
pip install buddy-*.whl --no-deps
python -c "import buddy; import buddy_mlir; print('ok')"
buddy-opt --help
```

## Examples

We provide examples to demonstrate how to use the passes and interfaces in `buddy-mlir`, including IR-level transformations, domain-specific applications, and testing demonstrations.

For more details, please see the [examples documentation](./examples/README.md).

## Contributions

We welcome contributions to our open-source project!

Before contributing, please read the [Contributor Guide](https://buddycompiler.com/Pages/ContributorGuide.html) and [Code Style](https://buddycompiler.com/Pages/Documentation/CodeStyle.html).

To maintain code quality, this project provides pre-commit checks:

```
$ pre-commit install
```

## How to Cite

If you find our project and research useful or refer to it in your own work, please cite the survey paper in which the Buddy Compiler design was first proposed:

```
@article{zhang2023compiler,
  title={Compiler Technologies in Deep Learning Co-Design: A Survey},
  author={Zhang, Hongbin and Xing, Mingjie and Wu, Yanjun and Zhao, Chen},
  journal={Intelligent Computing},
  year={2023},
  publisher={AAAS}
}
```

For direct access to the paper, please visit [Compiler Technologies in Deep Learning Co-Design: A Survey](https://spj.science.org/doi/10.34133/icomputing.0040).
