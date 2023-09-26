// RUN: buddy-opt %s \
// RUN:     -convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
// RUN:     -convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
// RUN:     -convert-func-to-llvm -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %slinalg-conv2d-tiling-run:

// BUDDY_OPT := ../../build/bin/buddy-opt
// MLIR_OPT := ../../llvm/build/bin/mlir-opt
// MLIR_TRANSLATE := ../../llvm/build/bin/mlir-translate
// MLIR_CPU_RUNNER := ../../llvm/build/bin/mlir-cpu-runner
// LLC := ../../llvm/build/bin/llc
// OPT_FLAG := -O0
// ifeq ($(shell uname),Linux)
// MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.so
// MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.so
// MLIR_ASYNC_RUNTIME := ../../llvm/build/lib/libmlir_async_runtime.so
// MTRIPLE := x86_64-unknown-linux-gnu
// else ifeq ($(shell uname),Darwin)
// MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.dylib
// MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.dylib
// MLIR_ASYNC_RUNTIME := ./../llvm/build/lib/libmlir_async_runtime.dylib
// MTRIPLE := x86_64-apple-darwin
// endif

// @${MLIR_OPT} linalg-conv2d.mlir ${MLIR_OPT_OPTIONS} \
// 	-test-transform-dialect-interpreter \
// 	-convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
// 	-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
// 	-convert-func-to-llvm -reconcile-unrealized-casts | \
// ${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

module{
    func.func private @printMemrefF32(memref<*xf32>)

    func.func @matmul(%a : memref<?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?xf32>) {
      linalg.matmul 
        ins(%a, %b: memref<?x?xf32>, memref<?x?xf32>)
       outs(%c:memref<?x?xf32>)
      return
    }

    func.func @main(){
      // Set up dims.
      %cM = arith.constant 1024 : index
      %cN = arith.constant 1024 : index
      %cK = arith.constant 1024 : index

      // Set Init Value.
      %cf1 = arith.constant 1.0 : f32

      %A = memref.alloc(%cM, %cK) : memref<?x?xf32>
      %B = memref.alloc(%cK, %cN) : memref<?x?xf32>
      %C = memref.alloc(%cM, %cN) : memref<?x?xf32>

      linalg.fill
      ins(%cf1 : f32)
      outs(%A:memref<?x?xf32>)

      linalg.fill
      ins(%cf1 : f32)
      outs(%B:memref<?x?xf32>)

      linalg.fill
      ins(%cf1 : f32)
      outs(%C:memref<?x?xf32>)

      call @matmul(%A, %B, %C) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()

      // Print output.
      // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [4, 4] strides = [4, 1] data =
      // CHECK-NEXT: [
      // CHECK-SAME:  [5, 5, 5, 5],
      // CHECK-NEXT:  [5, 5, 5, 5],
      // CHECK-NEXT:  [5, 5, 5, 5],
      // CHECK-NEXT:  [5, 5, 5, 5]
      // CHECK-SAME: ]
      // %print_C = memref.cast %C : memref<?x?xf32> to memref<*xf32>
      // call @printMemrefF32(%print_C) : (memref<*xf32>) -> ()

      memref.dealloc %C : memref<?x?xf32>
      memref.dealloc %B : memref<?x?xf32>
      memref.dealloc %A : memref<?x?xf32>
      return 
    }
}
