// RUN: buddy-opt -batchmatmul-optimize -verify-diagnostics -expand-strided-metadata -lower-affine -convert-vector-to-llvm \
// RUN:   -finalize-memref-to-llvm -convert-scf-to-cf -convert-linalg-to-loops -convert-scf-to-cf -llvm-request-c-wrappers \
// RUN:   -convert-func-to-llvm -reconcile-unrealized-casts %s \
// RUN: | mlir-cpu-runner -O0 -e buddy_batchmatmul_f32 -entry-point-result=void \
// RUN: -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext,%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

#map = affine_map<(d0) -> (d0 mod 64)>
#map1 = affine_map<(d0) -> (d0 floordiv 64)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0) -> (d0 * 64)>
#set = affine_set<(d0)[s0] : (d0 * -64 + s0 - 64 >= 0)>

func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }

func.func @conv_2d(%arg0: memref<1x1x1024xf32>, %arg1: memref<1x1024x1000xf32>, %arg2: memref<1x1x1000xf32>) {
  linalg.batch_matmul 
    ins(%arg0, %arg1 : memref<1x1x1024xf32>, memref<1x1024x1000xf32>) 
    outs(%arg2 : memref<1x1x1000xf32>)
  return
}

func.func @main(){

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c576 = arith.constant 576 : index
  %c1024 = arith.constant 1024 : index
  %c1000 = arith.constant 1000 : index
  %f0 = arith.constant 0.0 : f32
  %f2 = arith.constant 2.0 : f32
  %f3 = arith.constant 3.0 : f32

  %a = memref.alloc() : memref<1x1x1024xf32>
  scf.for %arg0 = %c0 to %c1 step %c1 {
    scf.for %arg1 = %c0 to %c1 step %c1 {
      scf.for %arg2 = %c0 to %c1024 step %c1 {
        memref.store %f3, %a[%arg0, %arg1, %arg2] : memref<1x1x1024xf32>
      }
    }
  }

  %b = memref.alloc() : memref<1x1024x1000xf32>
  scf.for %arg0 = %c0 to %c1 step %c1 {
    scf.for %arg1 = %c0 to %c1024 step %c1 {
      scf.for %arg2 = %c0 to %c1000 step %c1 {
        memref.store %f2, %b[%arg0, %arg1, %arg2] : memref<1x1024x1000xf32>
      }
    }
  }

  %c = memref.alloc() : memref<1x1x1000xf32>
  scf.for %arg0 = %c0 to %c1 step %c1 {
    scf.for %arg1 = %c0 to %c1 step %c1 {
      scf.for %arg2 = %c0 to %c1000 step %c1 {
        memref.store %f0, %c[%arg0, %arg1, %arg2] : memref<1x1x1000xf32>
      }
    }
  }
  
  linalg.batch_matmul 
    ins(%a, %b : memref<1x1x1024xf32>, memref<1x1024x1000xf32>) 
    outs(%c : memref<1x1x1000xf32>)

  %printed_c = memref.cast %c : memref<1x1x1000xf32> to memref<*xf32>
  call @printMemrefF32(%printed_c) : (memref<*xf32>) -> ()
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 3 offset = 0 sizes = \[2, 2, 4\] strides = \[8, 4, 1\] data =}}
  // CHECK{LITERAL}: [[[98,    226,    292,    164], 
  // CHECK{LITERAL}:   [12,    76,    96,    56]], 
  // CHECK{LITERAL}:  [[48,    162,    72,    156], 
  // CHECK{LITERAL}:   [16,    112,    0,    104]]]
  return
}
