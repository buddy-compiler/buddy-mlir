# BuddyTorq

`BuddyTorq` shows how buddy-mlir calls the external `TORQ-Tile` library to run
FP32 `matmul`.

The flow below was validated on SG2044.
The default matrix shape in this example is `M=64`, `K=64`, `N=64`.

## Prerequisites

- `torq-tile` repository: `https://github.com/RuyiAI-Stack/torq-tile/`
- TORQ-Tile built by following:
  `https://github.com/RuyiAI-Stack/torq-tile/blob/main/docs/BuildOnSG2044.md`

## Build buddy-mlir from scratch and run

```bash
ssh sg2044
source ~/.venv-buddy/bin/activate

cd buddy-mlir
mkdir build
cd build

export BUDDY_MLIR_BUILD_DIR=$PWD
export LLVM_MLIR_BUILD_DIR=$PWD/../llvm/build
export PYTHONPATH=${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
export CPATH=/usr/include

CC=/opt/gcc-native/bin/gcc \
CXX=/opt/gcc-native/bin/g++ \
cmake -G Ninja .. \
  -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
  -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
  -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
  -DCMAKE_C_FLAGS="-march=rv64gcv -mabi=lp64d" \
  -DCMAKE_CXX_FLAGS="-march=rv64gcv -mabi=lp64d" \
  -DBUDDY_TORQ_EXAMPLES=ON \
  -DBUDDY_RUNTIME_MATMUL_TORQ=ON \
  -DTorqTile_DIR=$PWD/../../torq-tile/build \
  -DBUDDY_TORQ_LLC_EXTRA="-mtriple=riscv64-unknown-linux-gnu -target-abi=lp64d -mattr=+m,+a,+f,+d,+c,+v"

ninja buddy-torq-run
./bin/buddy-torq-run
```

## Expected output

```text
=== BuddyTorq (TORQ-Tile matmul) ===
max abs error vs reference: 0.0000
Test PASSED!
```

## Notes

- `BUDDY_RUNTIME_MATMUL_TORQ=ON` builds `buddy_matmul_torq`
- `BUDDY_TORQ_LLC_EXTRA` ensures that `llc` generates object files with an ABI
  compatible with the SG2044 runtime environment
- `CPATH=/usr/include` is needed so `/opt/gcc-native/bin/g++` can find the
  system `pybind11` headers
