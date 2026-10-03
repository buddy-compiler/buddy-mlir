// RUN: buddy-opt %S/../../../examples/IMEDialect/k3_dot_test.mlir -lower-linalg-to-ime="target=k3" | FileCheck %s --check-prefix=TILES
// RUN: buddy-opt %s -lower-linalg-to-ime="target=k3" | FileCheck %s --check-prefix=KEEP

// Native tile sizes must reach the boundary-aware matmul and transposed batch
// paths. FP16 generics use that same path, rather than treating B[K,N] as B[N,K].
// TILES-LABEL: func.func @matmul
// TILES: memref.alloca() : memref<8x16xi8>
// TILES: memref.alloca() : memref<8x16xi8>
// TILES: memref.alloca() : memref<8x8xi32>
// TILES: ime.vmadot
// TILES-LABEL: func.func @generic_matmul
// TILES: memref.alloca() : memref<8x8xf16>
// TILES: ime.vfmadot
// TILES-LABEL: func.func @batch_matmul
// TILES: memref.alloca() : memref<8x16xi8>
// TILES: ime.vmadot

// BatchMatmulTransposeBOp is a C++ specialization of BatchMatmulOp in the new
// LLVM, not a distinct operation name. Ordinary indexing maps must not match
// the transpose-B rewrite or trigger an invalid cast.
func.func @ordinary_batch(%a: memref<2x3x4xi8>, %b: memref<2x4x5xi8>,
                          %c: memref<2x3x5xi32>) {
  linalg.batch_matmul ins(%a, %b : memref<2x3x4xi8>, memref<2x4x5xi8>)
                      outs(%c : memref<2x3x5xi32>)
  return
}
// KEEP-LABEL: func.func @ordinary_batch
// KEEP: linalg.batch_matmul
// KEEP-NOT: ime.vmadot

// Transposed named matmul likewise needs a dedicated packing path.
func.func @transposed_matmul(%a: memref<3x4xi8>, %b: memref<5x4xi8>,
                             %c: memref<3x5xi32>) {
  linalg.matmul indexing_maps = [affine_map<(m,n,k)->(m,k)>,
      affine_map<(m,n,k)->(n,k)>, affine_map<(m,n,k)->(m,n)>]
      ins(%a, %b : memref<3x4xi8>, memref<5x4xi8>)
      outs(%c : memref<3x5xi32>)
  return
}
// KEEP-LABEL: func.func @transposed_matmul
// KEEP: linalg.matmul
// KEEP-NOT: ime.vmadot
