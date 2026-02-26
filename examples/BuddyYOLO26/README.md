# Buddy Compiler YOLO26n Example

1. Enter Python virtual environment

We recommend using anaconda3 to create a Python virtual environment.

```bash
$ conda activate <your virtual environment name>
$ cd buddy-mlir
$ pip install -r requirements.txt
```

2. Build LLVM/MLIR

```bash
$ cd buddy-mlir
$ mkdir -p llvm/build
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

3. Build buddy-mlir

```bash
$ cd buddy-mlir
$ mkdir -p build
$ cd build
$ cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
    -DBUDDY_MLIR_ENABLE_DIP_LIB=ON \
    -DBUDDY_ENABLE_PNG=ON \
    -DPython3_EXECUTABLE=$(which python3)
$ ninja
```

4. Set `PYTHONPATH`

```bash
$ cd buddy-mlir/build
$ export BUDDY_MLIR_BUILD_DIR=$PWD
$ export LLVM_MLIR_BUILD_DIR=$PWD/../llvm/build
$ export PYTHONPATH=${LLVM_MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
```

5. Set YOLO26n model path (optional)

```bash
$ export YOLO26N_MODEL_PATH=/path/to/yolo26n.pt
```

If `YOLO26N_MODEL_PATH` is not set, the importer tries `buddy-mlir/yolo26n.pt`.
If the file does not exist, Ultralytics will download `yolo26n.pt`
automatically. In the CMake flow above, it is downloaded to
`build/examples/BuddyYOLO26/yolo26n.pt`.

6. Build and run YOLO26n example

```bash
$ cd buddy-mlir/build
$ cmake -G Ninja .. -DBUDDY_YOLO26_EXAMPLES=ON
$ ninja buddy-yolo26n-run
$ ./bin/buddy-yolo26n-run ../examples/BuddyYOLO26/images/bus.bmp
```
