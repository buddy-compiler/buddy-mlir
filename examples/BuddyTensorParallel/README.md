# Buddy Compiler DeepSeekR1 Distributed Tensor Parallel Example

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
$ export PYTHONPATH=${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
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
$ cmake -G Ninja .. -DBUDDY_TENSORPARALLEL_EXAMPLES=ON

$ ninja buddy-deepseek-r1-distributed
```
Run the distributed executable locally with MPICH Hydra:

```bash
$ ./examples/BuddyTensorParallel/mpich-install/bin/mpiexec.hydra \
    -n 3 \
    -outfile-pattern "output-%r.txt" \
    ./bin/buddy-deepseek-r1-distributed
```
This launches three MPI ranks on the same machine and simulates inter-process communication locally
This generates:

- `output-0.txt`
- `output-1.txt`
- `output-2.txt`

These files correspond to the outputs from rank 0, rank 1, and rank 2 respectively.

The final inference result is written to `output-0.txt`.

## How to cross compile RISC-V target on x86_64 machine


### 0. Prepare the RVV build environment

First, prepare the RVV cross-compilation environment by following the [RVV environment setup document](https://github.com/asdf1113/buddy-mlir/blob/split-ds/docs/RVVEnvironment.md). 

This example depends on the RISC-V OpenMP shared library. Download and extract it with:

```bash
$ cd buddy-mlir/llvm/
$ wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1XEsAhOcMioN9gdufuyO9OrHIdR0UtHh2' -O build-omp-shared-rv.tar.gz
$ mkdir build-omp-shared-rv
$ tar -xzf build-omp-shared-rv.tar.gz -C build-omp-shared-rv
$ rm build-omp-shared-rv.tar.gz
```

### 1. Set environment variables

export the required environment variables:

```bash
$ cd buddy-mlir/build #your local buddy-mlir
$ export BUDDY_MLIR_BUILD_DIR=$PWD
$ export LLVM_MLIR_BUILD_DIR=${BUDDY_MLIR_BUILD_DIR}/../llvm/build/
$ export PYTHONPATH=${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
$ export RISCV_GNU_TOOLCHAIN=${BUDDY_MLIR_BUILD_DIR}/thirdparty/riscv-gnu-toolchain
$ export RISCV_OMP_SHARED=${LLVM_MLIR_BUILD_DIR}/../build-omp-shared-rv/libomp.so
$ export RISCV_SYSROOT=${RISCV_GNU_TOOLCHAIN}/sysroot/
$ export BUDDY_MLIR_BUILD_CROSS_DIR=${BUDDY_MLIR_BUILD_DIR}/../build-cross-rv
```

### 2. Configure and build

Create a dedicated build directory for this distributed example and run CMake:

```bash
$ cd buddy-mlir
$ mkdir build-deepseek-distributed && cd build-deepseek-distributed
$ cmake -G Ninja .. \
    -DBUDDY_TENSORPARALLEL_EXAMPLES=ON \
    -DMLIR_DIR=${BUDDY_MLIR_BUILD_DIR}/../llvm/build/lib/cmake/mlir \
    -DCROSS_COMPILE_RVV=ON \
    -DBUDDY_MLIR_BUILD_DIR=${BUDDY_MLIR_BUILD_DIR} \
    -DLLVM_MLIR_BUILD_DIR=${LLVM_MLIR_BUILD_DIR} \
    -DBUDDY_MLIR_BUILD_CROSS_DIR=${BUDDY_MLIR_BUILD_CROSS_DIR} \
    -DRISCV_GNU_TOOLCHAIN=${RISCV_GNU_TOOLCHAIN} \
    -DRISCV_SYSROOT=${RISCV_SYSROOT} \
    -DRISCV_OMP_SHARED=${RISCV_OMP_SHARED} \
    -DCMAKE_C_COMPILER=${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++ \
    -DDSTP_EXAMPLE_PATH=. \
    -DDSTP_EXAMPLE_BUILD_PATH=.
```

Build the executable and generate the RVV runtime packages:

```bash
$ ninja buddy-deepseek-r1-distributed
$ ninja buddy-deepseek-r1-rvv-packages
```

This generates multiple compressed packages under: `./examples/BuddyTensorParallel/`

- `buddy-deepseek-r1-rvv-package-rank0.tgz`
- `buddy-deepseek-r1-rvv-package-rank1.tgz`

### 3. Copy packages to RVV hosts

```bash
$ scp ./examples/BuddyTensorParallel/buddy-deepseek-r1-rvv-package-rank0.tgz user@rvv-host0:
$ scp ./examples/BuddyTensorParallel/buddy-deepseek-r1-rvv-package-rank1.tgz user@rvv-host1:
```

 on **each** RVV host:

```
$ mkdir -p ~/buddy-deepseek-r1-dist
$ tar xzf buddy-deepseek-r1-rvv-package-rank0.tgz --strip-components=1 -C ~/buddy-deepseek-r1-dist
```


```
$ mkdir -p ~/buddy-deepseek-r1-dist
$ tar xzf buddy-deepseek-r1-rvv-package-rank1.tgz --strip-components=1 -C ~/buddy-deepseek-r1-dist
```

After extraction, all hosts should have the same runtime directory layout:

```
~/buddy-deepseek-r1-dist/
  buddy-deepseek-r1-distributed
  mpich-install/
  libomp.so
  ...
```

### 4. SSH & Hosts Configuration for MPICH

On the host that will launch rank 0, configure passwordless SSH access to the other machines:

```
$ ssh-keygen -t rsa
$ ssh-copy-id <user>@<target-ip-of-rvv-host1>
```

Verify that rank0 can SSH to the other hosts without a password prompt.

Create a hosts file on the rank0 machine.The file content should be the IP addresses or hostnames of all participating RVV hosts, one per line.

For example:

```bash
$ cd ~/buddy-deepseek-r1-dist

$ cat > hosts <<'EOF'
    target-ip-of-rvv-host0
    target-ip-of-rvv-host1
    EOF
```

### 5. Launch distributed inference

On `rvv-host0` (the machine acting as rank 0), run:

```
$ cd ~/buddy-deepseek-r1-dist
$ export LD_LIBRARY_PATH="$(pwd)/mpich-install/lib:$(pwd):${LD_LIBRARY_PATH}"

$ ./mpich-install/bin/mpiexec.hydra \
    -f hosts \
    -n 2 \
    -outfile-pattern "output-%r.txt" \
    ./buddy-deepseek-r1-distributed
```

This generates:

- `output-0.txt`
- `output-1.txt`

These files correspond to the outputs from rank 0 and rank 1 respectively.

The final inference result is written to `output-0.txt`.

