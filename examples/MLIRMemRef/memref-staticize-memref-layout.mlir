// RUN: buddy-opt %s -staticize-memref-layout | FileCheck %s
// RUN: buddy-opt %s -staticize-memref-layout \
// RUN:     -lower-affine -finalize-memref-to-llvm \
// RUN:     -convert-scf-to-cf -convert-cf-to-llvm -convert-func-to-llvm  \
// RUN:     -convert-arith-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s --check-prefix=RUNTIME

module {
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %f0 = arith.constant 0.0 : f32

    %base = memref.alloc() : memref<32xf32>
    %dst = memref.alloc() : memref<4x8xf32>

    // Initialize base to deterministic values (avoid printing uninitialized data).
    scf.for %i = %c0 to %c32 step %c1 {
      memref.store %f0, %base[%i] : memref<32xf32>
    }

    // Dynamic (unknown) leading stride + unknown offset in the result type.
    %src = memref.reinterpret_cast %base to
      offset: [%c0],
      sizes: [4, 8],
      strides: [%c8, 1]
      : memref<32xf32> to memref<4x8xf32, strided<[?, 1], offset: ?>>

    memref.copy %src, %dst
      : memref<4x8xf32, strided<[?, 1], offset: ?>> to memref<4x8xf32>

    %print_out = memref.cast %dst : memref<4x8xf32> to memref<*xf32>
    call @printMemrefF32(%print_out) : (memref<*xf32>) -> ()

    return
  }
}

// CHECK-LABEL: func.func @main
// CHECK: memref.reinterpret_cast {{.*}} : memref<32xf32> to memref<4x8xf32, strided<[8, 1]{{(, offset: 0)?}}>>
// CHECK-NOT: memref<4x8xf32, strided<[?, 1], offset: ?>>

// Runtime output check (memref print should show static stride [8, 1]).
// RUNTIME: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [4, 8] strides = [8, 1] data =


