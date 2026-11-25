// RUN: buddy-opt %s -lower-magia | FileCheck %s

// CHECK: func.func private @magia_gemm(i64, i64, i64, i64, i64, i64)
// CHECK-LABEL: func.func @product
// CHECK: memref.extract_aligned_pointer_as_index %{{.*}} : memref<32x768xf16>
// CHECK: call @magia_gemm(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i64, i64, i64, i64, i64, i64) -> ()
// CHECK-NOT: magia.gemm
func.func @product(%x: memref<32x768xf16>, %w: memref<768x64xf16>, %z: memref<32x64xf16>) {
  magia.gemm %x %w %z : memref<32x768xf16> memref<768x64xf16> memref<32x64xf16>
  return
}

// A matrix of a pack of parameters: its address is the pack's plus its offset,
// and its strides are checked when the program runs.
// CHECK-LABEL: func.func @pack
// CHECK: memref.extract_strided_metadata %{{.*}} : memref<768x64xf16, strided<[?, ?], offset: ?>>
// CHECK: cf.assert %{{.*}}, "the rows of a matrix of the mesh are not contiguous"
// CHECK: call @magia_gemm
// CHECK-NOT: magia.gemm
func.func @pack(%x: memref<32x768xf16>, %w: memref<768x64xf16, strided<[?, ?], offset: ?>>, %z: memref<32x64xf16>) {
  magia.gemm %x %w %z : memref<32x768xf16> memref<768x64xf16, strided<[?, ?], offset: ?>> memref<32x64xf16>
  return
}
