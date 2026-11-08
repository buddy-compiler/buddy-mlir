// RUN: buddy-opt %s -lower-linalg-to-vir -lower-vir-to-vector="vector-width=4" -cse --convert-vector-to-scf --lower-affine --expand-strided-metadata --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | mlir-runner -O0 -e main -entry-point-result=i32 -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext,%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext | FileCheck %s

// A linalg.map whose body reads linalg.index (an iota, as the upsampling of
// an image model builds): scalar loops, as for a linalg.generic.

func.func private @printMemrefI64(memref<*xi64>) attributes { llvm.emit_c_interface }

func.func @iota(%o: memref<7xi64>) {
  linalg.map outs(%o : memref<7xi64>)
    (%init: i64) {
      %i = linalg.index 0 : index
      %v = arith.index_cast %i : index to i64
      linalg.yield %v : i64
    }
  return
}

func.func @main() -> i32 {
  %o = memref.alloc() : memref<7xi64>
  call @iota(%o) : (memref<7xi64>) -> ()
  %p = memref.cast %o : memref<7xi64> to memref<*xi64>
  call @printMemrefI64(%p) : (memref<*xi64>) -> ()
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 1 offset = 0 sizes = \[7\] strides = \[1\] data =}}
  // CHECK-NEXT: [0, 1, 2, 3, 4, 5, 6]
  memref.dealloc %o : memref<7xi64>
  %ret = arith.constant 0 : i32
  return %ret : i32
}
