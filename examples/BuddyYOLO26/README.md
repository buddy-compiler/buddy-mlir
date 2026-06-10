# Buddy Compiler YOLO26n Example

1. Enter Python virtual environment

We recommend you to use anaconda3 to create python virtual environment. You should install python packages as `buddy-mlir/requirements.txt`.

```
$ conda activate <your virtual environment name>
$ cd buddy-mlir
$ pip install -r requirements.txt
```

To run YOLO26 example:

```
$ pip install ultralytics
```

---

2. Build and check LLVM/MLIR

```
$ cd buddy-mlir
$ mkdir -p llvm/build
$ cd llvm/build
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_ENABLE_RUNTIMES="openmp" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=$(which python3)
$ ninja check-clang check-mlir
```

---

3. Build and check buddy-mlir

```
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
    -DBUDDY_YOLO26_EXAMPLES=ON \
    -DPython3_EXECUTABLE=$(which python3)
$ ninja
```

---

4. Set environment variables

```
$ cd buddy-mlir/build
$ export BUDDY_MLIR_BUILD_DIR=$PWD
$ export LLVM_MLIR_BUILD_DIR=$PWD/../llvm/build
$ export PYTHONPATH=${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
```

Test Python bindings:

```
$ python3 -c "from buddy.compiler.frontend import DynamoCompiler"
```

---

5. Optional: Set YOLO26n model path

```
$ export YOLO26N_MODEL_PATH=/path/to/yolo26n.pt
```

If not set, the importer will automatically download `yolo26n.pt`.

---

6. Build and run YOLO26n example

```
$ cd buddy-mlir/build
$ cmake -G Ninja .. -DBUDDY_YOLO26_EXAMPLES=ON
$ ninja buddy-yolo26n-run
$ ./bin/buddy-yolo26n-run ../examples/BuddyYOLO26/images/bus_16bit.bmp
```

---

7. Expected output

```
YOLO26n Inference Powered by Buddy Compiler

[0] class_id=0 label=person ...
[1] class_id=0 label=person ...
[2] class_id=0 label=person ...
[3] class_id=5 label=bus ...
[4] class_id=0 label=person ...

Detections: 5
```

Runtime prints:

* class id
* class label
* confidence score
* bounding box coordinates