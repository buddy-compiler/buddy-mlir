// RUN: buddy-opt %s -lower-linalg-to-vir -lower-vir-to-vector="vector-width=4" -cse --convert-vector-to-scf --lower-affine --expand-strided-metadata --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | mlir-runner -O0 -e main -entry-point-result=i32 -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext,%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext | FileCheck %s

// A rank-0 input of an elementwise op (a scalar for every point, e.g. a
// constant of the graph): loaded once and broadcast.

memref.global "private" constant @a : memref<6xf32> = dense<[1.5, -2.0, 3.0, 0.25, 4.0, -1.0]>
memref.global "private" constant @s : memref<f32> = dense<2.0>

func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }

func.func @scale(%a: memref<6xf32>, %s: memref<f32>, %o: memref<6xf32>) {
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
                                   affine_map<(d0) -> ()>,
                                   affine_map<(d0) -> (d0)>],
                  iterator_types = ["parallel"]}
    ins(%a, %s : memref<6xf32>, memref<f32>) outs(%o : memref<6xf32>) {
  ^bb0(%x: f32, %y: f32, %z: f32):
    %m = arith.mulf %x, %y : f32
    linalg.yield %m : f32
  }
  return
}

func.func @main() -> i32 {
  %a = memref.get_global @a : memref<6xf32>
  %s = memref.get_global @s : memref<f32>
  %o = memref.alloc() : memref<6xf32>
  call @scale(%a, %s, %o) : (memref<6xf32>, memref<f32>, memref<6xf32>) -> ()
  %p = memref.cast %o : memref<6xf32> to memref<*xf32>
  call @printMemrefF32(%p) : (memref<*xf32>) -> ()
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 1 offset = 0 sizes = \[6\] strides = \[1\] data =}}
  // CHECK-NEXT: [3, -4, 6, 0.5, 8, -2]
  memref.dealloc %o : memref<6xf32>
  %ret = arith.constant 0 : i32
  return %ret : i32
}
