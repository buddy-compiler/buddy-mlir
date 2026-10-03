// RUN: buddy-opt %s -lower-ime="target=k3" | FileCheck %s
// RUN: buddy-opt %s -lower-ime="target=k3" -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-func-to-llvm -finalize-memref-to-llvm -reconcile-unrealized-casts | FileCheck %s --check-prefix=LLVM

// Partial, rectangular tiles must be padded to A100 capacity. Strided source
// memrefs must be accessed through their descriptors rather than a flat pointer.
func.func @full_offset(%c: memref<8x8xi32, strided<[8, 1], offset: ?>>,
                      %a: memref<8x16xi8, strided<[16, 1], offset: ?>>) {
  // CHECK-LABEL: func.func @full_offset
  // CHECK-NOT: memref.alloca
  // CHECK: memref.extract_strided_metadata
  // CHECK: llvm.getelementptr
  // CHECK: "ime.intr.vmadot"
  // CHECK: llvm.store
  // CHECK-NOT: memref.alloca
  ime.vmadot %c, %a, %a : memref<8x8xi32, strided<[8, 1], offset: ?>>,
    memref<8x16xi8, strided<[16, 1], offset: ?>>,
    memref<8x16xi8, strided<[16, 1], offset: ?>>
  return
}

func.func @dot(%c: memref<3x5xi32, strided<[11, 2], offset: ?>>,
               %a: memref<3x7xi8, strided<[19, 2], offset: ?>>,
               %b: memref<5x7xi8, strided<[19, 2], offset: ?>>) {
  // CHECK-LABEL: func.func @dot
  // CHECK: memref.alloca_scope {
  // CHECK: memref.alloca() : memref<8x16xi8>
  // CHECK: memref.alloca() : memref<8x16xi8>
  // CHECK: memref.alloca() : memref<8x8xi32>
  // CHECK: memref.load {{.*}} : memref<3x7xi8, strided<[19, 2], offset: ?>>
  // CHECK: llvm.load {{.*}} : !llvm.ptr -> vector<[8]xi8>
  // CHECK: llvm.load {{.*}} : !llvm.ptr -> vector<[4]xi32>
  // CHECK: "ime.intr.vmadot"
  // CHECK: llvm.store
  // CHECK: memref.store {{.*}} : memref<3x5xi32, strided<[11, 2], offset: ?>>
  ime.vmadot %c, %a, %b : memref<3x5xi32, strided<[11, 2], offset: ?>>,
    memref<3x7xi8, strided<[19, 2], offset: ?>>,
    memref<5x7xi8, strided<[19, 2], offset: ?>>
  return
}

func.func @fp16(%c: memref<4x4xf16>, %a: memref<4x8xf16>, %b: memref<4x8xf16>) {
  // CHECK-LABEL: func.func @fp16
  // CHECK: memref.alloca() : memref<8x8xf32>
  // CHECK: arith.extf {{.*}} : f16 to f32
  // CHECK: "ime.intr.vfmadot"
  // CHECK: arith.truncf {{.*}} : f32 to f16
  ime.vfmadot %c, %a, %b : memref<4x4xf16>, memref<4x8xf16>, memref<4x8xf16>
  return
}

func.func @loop(%c: memref<1x1xi32>, %a: memref<1x1xi8>, %b: memref<1x1xi8>) {
  // CHECK-LABEL: func.func @loop
  // CHECK: scf.for
  // CHECK-NEXT: memref.alloca_scope {
  // CHECK: "ime.intr.vmadot"
  // LLVM-LABEL: llvm.func @loop
  // LLVM: llvm.intr.stacksave
  // LLVM: "ime.intr.vmadot"
  // LLVM: llvm.intr.stackrestore
  %lb = arith.constant 0 : index
  %ub = arith.constant 20 : index
  %step = arith.constant 1 : index
  scf.for %i = %lb to %ub step %step {
    ime.vmadot %c, %a, %b : memref<1x1xi32>, memref<1x1xi8>, memref<1x1xi8>
  }
  return
}
