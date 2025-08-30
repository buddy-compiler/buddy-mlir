# Buddy Graph Representation Examples

## Run the Examples

0. Enter your Python Env
```
(base)$ conda activate buddy
(buddy)$ ...
```
1. Build Python Packages
2. Configure Python Path
```
(buddy)$ cd buddy-mlir/build
(buddy)$ export BUDDY_MLIR_BUILD_DIR=$PWD
(buddy)$ export LLVM_MLIR_BUILD_DIR=$PWD/../llvm/build
(buddy)$ export PYTHONPATH=${LLVM_MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}

```
3. Run the Examples
```
(buddy)$ cd examples/BuddyGraph
(buddy)$ python import-dynamo-break.py
```