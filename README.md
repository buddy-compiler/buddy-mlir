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
git clone git@github.com:buddy-compiler/buddy-mlir.git
cd buddy-mlir
git submodule update --init llvm
```

### Prepare Python Environment

pip

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

uv

```
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

conda

```
conda activate <your virtual environment name>
cd buddy-mlir
pip install -r requirements.txt
```

### Build and Test LLVM/MLIR/CLANG

```
cd buddy-mlir
cmake -G Ninja -S llvm/llvm -B llvm/build \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_ENABLE_RUNTIMES="openmp" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE="$(which python)" \
    -DPython_EXECUTABLE="$(which python)"
ninja -C llvm/build check-clang check-mlir check-openmp
```

If your target machine includes an NVIDIA GPU, you can add the following configuration:

```
-DLLVM_TARGETS_TO_BUILD="host;RISCV;NVPTX" \
-DMLIR_ENABLE_CUDA_RUNNER=ON \
```

### Build buddy-mlir

```
cd buddy-mlir
cmake -G Ninja -S . -B build \
    -DMLIR_DIR=$PWD/llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
    -DPython3_EXECUTABLE="$(which python)" \
    -DPython_EXECUTABLE="$(which python)"
ninja -C build
ninja -C build check-buddy
```

### Prepare RISC-V cross-compilation dependencies

`make riscv` first builds the host tools (incrementally) and then prepares the
RISC-V GNU/Linux sysroot, cross-compiled MLIR runtime libraries, and shared
OpenMP runtime. The GNU toolchain is configured with `--disable-llvm` and
`--disable-gdb`; the RISC-V MLIR libraries are built with the host LLVM Clang.

```bash
make riscv
```

The target sets `MAKEINFO=:` for the GNU toolchain build: skipping optional
glibc Info manuals keeps builds working from non-ASCII source paths without
affecting the compiler, headers, or runtime libraries.

The generated files are laid out as follows:

```text
thirdparty/riscv-gnu-toolchain/          # source submodule
thirdparty/riscv-gnu-toolchain/build/    # GNU toolchain build tree
thirdparty/riscv/                        # installed RISC-V toolchain/sysroot
thirdparty/riscv/bin/                    # riscv64-unknown-linux-gnu-* tools
thirdparty/riscv/sysroot/                # glibc headers, crt files, libraries
thirdparty/riscv/lib/                    # libomp.so and libmlir_*.so
llvm/build-cross-mlir-rv/                # cross-compiled MLIR build tree
llvm/build-omp-shared-rv/                # cross-compiled OpenMP build tree
```

This target prepares compiler/runtime dependencies only; it does not import or
compile a model and does not build a RISC-V `buddy-cli`.

The Makefile provides scoped cleanup shortcuts:

```bash
# Remove generated model artifacts under build/models/.
make clean-models

# Remove host LLVM and Buddy build trees.
make clean-host

# Remove RISC-V toolchain/runtime build and install trees.
make clean-riscv

# Remove both host and RISC-V build trees.
make clean-all
```

`make clean` is an alias for `make clean-host`. The RISC-V cleanup removes only
generated directories and keeps `thirdparty/riscv-gnu-toolchain/` itself, which
is the source submodule. The equivalent manual command is:

```bash
rm -rf -- \
  thirdparty/riscv-gnu-toolchain/build \
  thirdparty/riscv \
  llvm/build-cross-mlir-rv \
  llvm/build-omp-shared-rv \
  build-riscv
```

Keep `thirdparty/riscv-gnu-toolchain/` itself: it is the source submodule.
Remove `llvm/build` and `build` as well when a full host rebuild is needed.

`make install` installs the already-built host tools and libraries with CMake;
it does not install operating-system packages or download model weights. The
default prefix is the ignored local directory `install/`; override it with
`PREFIX=/opt/buddy-mlir` (or another writable path). `make riscv-deps` is an
explicit alias for `make riscv`; `make install-riscv` is a more descriptive
spelling when the goal is to prepare the RISC-V prefix, which is controlled by
`RISCV_INSTALL`.

RISC-V model targets and their family aliases are generated from the available
model specs, so new `models/*/specs/*.json` files become Make targets
automatically:

```bash
make help
make list-riscv-models
make riscv-model-deepseek_r1-f32
make riscv-model-whisper       # defaults to the base spec
```

`make riscv-model` selects `RISCV_MODEL_DEFAULT` (default:
`deepseek_r1-f32`). A family alias selects `f32.json` when present, then
`base.json`, then the first checked-in spec. Select any other variant directly
with its generated Make target, such as `make riscv-model-deepseek_r1-w8a16`.

Every model target first runs `make riscv`, then invokes `build_model.py` in a
separate `build-riscv/<family>-<variant>/` tree. Set
`RISCV_MODEL_LOCAL=/path/to/model` for an offline/local HuggingFace snapshot,
`RISCV_MODEL_HF_CONFIG=/path/to/config.json` when a separate config is needed,
`PYTHON=/path/to/python` when using a Python environment other than the
repository's `.venv`, and `PARALLEL=8` to pass the corresponding model build
parallelism. The selected Python interpreter is also passed to CMake, so its
`nanobind` installation is used consistently.

For RVV model builds, the cross-compilation applies to the complete runtime
package: model kernels, runner plugin, and target runtime libraries are all
built for RISC-V and embedded in the `.rax` payload.
Qwen3-VL, BGE-M3, and ProteinGLM targets require their model-specific local
snapshot, for example:

```bash
make riscv-model-qwen3_vl-instruct_2b \
  RISCV_MODEL_LOCAL=/path/to/Qwen3-VL-2B-Instruct
```

Set the `PYTHONPATH` environment variable to include both the LLVM/MLIR Python bindings and `buddy-mlir` Python packages:

```
export BUDDY_MLIR_BUILD_DIR=$PWD/build
export LLVM_MLIR_BUILD_DIR=$PWD/llvm/build
export PYTHONPATH=${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
```

If you want to test your model end-to-end conversion and inference, you can add the following configuration

```
cmake -G Ninja -S . -B build -DBUDDY_ENABLE_E2E_TESTS=ON
ninja -C build check-e2e
```

### Building and running the model

Use the following to build:

```bash
cd buddy-mlir
python3 tools/buddy-codegen/build_model.py \
  --spec models/deepseek_r1/specs/f32.json \
  --build-dir build
```

To build the DeepSeek R1 model for RISC-V after `make riscv`, pass the
toolchain and the three target runtime libraries explicitly:

```bash
python3 tools/buddy-codegen/build_model.py \
  --spec models/deepseek_r1/specs/f32.json \
  --build-dir build-riscv/deepseek_r1-f32 \
  --is-rvv-crosscompile \
  --riscv-gnu-toolchain thirdparty/riscv \
  --riscv-omp-shared thirdparty/riscv/lib/libomp.so \
  --riscv-mlir-c-runner-utils thirdparty/riscv/lib/libmlir_c_runner_utils.so \
  --buddy-mlir-build-dir build \
  -j 8
```

The default spec downloads `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` from
Hugging Face. For an offline or local model, append
`--local-model /path/to/model` (the directory must contain `config.json` and
the weight files). The artifacts are written to
`build-riscv/deepseek_r1-f32/models/deepseek_r1/`, including
`deepseek_r1_model.so`,
`deepseek_r1.rax`, `deepseek_r1_runner.so`, and `deepseek_r1_serving.so`.
Only `deepseek_r1_model.so` is a RISC-V binary; the runner, serving plugin, and
`buddy-cli` are still built for the host.

The RAX manifest directly packages `libomp.so` and
`libmlir_c_runner_utils.so`. The latter also needs the installed
`libmlir_float16_utils.so` and `libmlir_apfloat_wrappers.so`; keep the RISC-V
library directory available on the target, for example:

```bash
export LD_LIBRARY_PATH="$PWD/thirdparty/riscv/lib:${LD_LIBRARY_PATH:-}"
```

For Whisper, use the same build entry point with the Whisper spec:

```bash
python3 tools/buddy-codegen/build_model.py \
  --spec models/whisper/specs/base.json \
  --build-dir build
```

DeepSeek R1, Whisper, and Qwen3-VL support template-based layer-partitioned compilation. It is disabled by default. Enable it explicitly by passing
`--cmake-args=-DBUDDY_MODEL_LAYER_PARTITION=ON` to `build_model.py`. DeepSeek R1 also retains the existing `PartitionedGraphDriver` workflow. See [Layer Partitioning](docs/LayerPartitioning.md) for details.

For example, enable template-based layer partitioning for Whisper with:

```bash
python3 tools/buddy-codegen/build_model.py \
  --spec models/whisper/specs/base.json \
  --build-dir build \
  --cmake-args=-DBUDDY_MODEL_LAYER_PARTITION=ON
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
ninja deepseek_r1_model_so deepseek_r1_rax
```

To build the DeepSeek R1 f32 tiered KV cache variant for use with `buddy-cli`,
use the dedicated spec:

```bash
python3 tools/buddy-codegen/build_model.py \
  --spec models/deepseek_r1/specs/f32_tiered_kv_cache.json \
  --build-dir build
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

Whisper uses the same `.rax` / `buddy-cli` deployment path, with an audio input:

```bash
./build/bin/buddy-cli \
  --model ./build/models/whisper/whisper.rax \
  --audio ./build/models/whisper/audio.wav
```

#### Qwen3-VL (vision-language OCR)

`models/qwen3_vl` is a self-contained vision-language model (ViT + DeepStack
encoder feeding a dense Qwen3 decoder) that runs end-to-end on buddy-compiled
kernels via `buddy-cli`. Use the same `tools/buddy-codegen/build_model.py` entry
point with the Qwen3-VL spec (a local HuggingFace snapshot is required):

```bash
python3 tools/buddy-codegen/build_model.py \
  --spec models/qwen3_vl/specs/instruct_2b.json \
  --build-dir build \
  --local-model /path/to/Qwen3-VL-2B-Instruct

./build/bin/buddy-cli \
  --model ./build/models/qwen3_vl/qwen3_vl.rax \
  --image ./models/qwen3_vl/test_text.png \
  --prompt "Read all the text in the image."
```

See [`models/qwen3_vl/README.md`](models/qwen3_vl/README.md) for prerequisites and
details.

## Build Python Package

We use `setuptools` to bundle CMake outputs (Python packages, `bin/`, and
`lib/`) into a single wheel.

Build x86_64 artifacts:

```bash
./scripts/release.sh cp312 0.0.0 x86_64
```

Build riscv64 artifacts:

```bash
./scripts/release.sh cp312 0.0.0 riscv64
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
pre-commit install
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
