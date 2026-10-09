// RUN: buddy-opt %s -linalg-transpose-tile="tile-size=8" | FileCheck %s --check-prefix=CHECK-IR
// RUN: buddy-opt %s -linalg-transpose-tile="tile-size=8" \
// RUN:     -scf-parallel-for-to-nested-fors -convert-scf-to-cf \
// RUN:     -convert-arith-to-llvm -finalize-memref-to-llvm \
// RUN:     -convert-cf-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s --check-prefix=CHECK-OUT

module {
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @transpose2d(%in: memref<8x16xf32>, %out: memref<16x8xf32>) {
    linalg.transpose ins(%in : memref<8x16xf32>)
                     outs(%out : memref<16x8xf32>)
                     permutation = [1, 0]
    return
  }

  func.func @transpose4d(%in: memref<2x4x8x16xf32>,
                         %out: memref<2x8x4x16xf32>) {
    linalg.transpose ins(%in : memref<2x4x8x16xf32>)
                     outs(%out : memref<2x8x4x16xf32>)
                     permutation = [0, 2, 1, 3]
    return
  }

  func.func @main() {
    %in2 = memref.alloc() : memref<8x16xf32>
    %out2 = memref.alloc() : memref<16x8xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        %vi = arith.index_cast %i : index to i32
        %vf = arith.sitofp %vi : i32 to f32
        %vj = arith.index_cast %j : index to i32
        %wj = arith.sitofp %vj : i32 to f32
        %val = arith.addf %vf, %wj : f32
        memref.store %val, %in2[%i, %j] : memref<8x16xf32>
      }
    }
    call @transpose2d(%in2, %out2) : (memref<8x16xf32>, memref<16x8xf32>) -> ()
    %p2 = memref.cast %out2 : memref<16x8xf32> to memref<*xf32>
    call @printMemrefF32(%p2) : (memref<*xf32>) -> ()

    %in4 = memref.alloc() : memref<2x4x8x16xf32>
    %out4 = memref.alloc() : memref<2x8x4x16xf32>
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    scf.for %a = %c0 to %c2 step %c1 {
      scf.for %b = %c0 to %c4 step %c1 {
        scf.for %d = %c0 to %c8 step %c1 {
          scf.for %e = %c0 to %c16 step %c1 {
            %vb = arith.index_cast %b : index to i32
            %vbf = arith.sitofp %vb : i32 to f32
            %vd = arith.index_cast %d : index to i32
            %vdf = arith.sitofp %vd : i32 to f32
            %val4 = arith.addf %vbf, %vdf : f32
            memref.store %val4, %in4[%a, %b, %d, %e] : memref<2x4x8x16xf32>
          }
        }
      }
    }
    call @transpose4d(%in4, %out4) : (memref<2x4x8x16xf32>, memref<2x8x4x16xf32>) -> ()
    %p4 = memref.cast %out4 : memref<2x8x4x16xf32> to memref<*xf32>
    call @printMemrefF32(%p4) : (memref<*xf32>) -> ()
    return
  }
}

// CHECK-IR-LABEL: func.func @transpose2d
// CHECK-IR-NOT:     linalg.transpose
// CHECK-IR:         scf.parallel
// CHECK-IR:         scf.for
// CHECK-IR:         memref.load
// CHECK-IR:         memref.store
// CHECK-IR-LABEL: func.func @transpose4d
// CHECK-IR-NOT:     linalg.transpose
// CHECK-IR:         scf.parallel
// CHECK-IR:         scf.for
// CHECK-IR:         memref.load
// CHECK-IR:         memref.store

// CHECK-OUT: sizes = [16, 8] strides = [8, 1]
// CHECK-OUT-NEXT: {{\[\[}}0,   1,   2,   3,   4,   5,   6,   7],
// CHECK-OUT-NEXT: [1,   2,   3,   4,   5,   6,   7,   8],
// CHECK-OUT-NEXT: [2,   3,   4,   5,   6,   7,   8,   9],
// CHECK-OUT: sizes = [2, 8, 4, 16]
// CHECK-OUT: [1,     1,     1,
