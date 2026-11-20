// RUN: not buddy-opt %s -lower-linalg-to-xtame 2>&1 | FileCheck %s --check-prefix=ERROR

// The pass supports a narrow set of shapes. Anything else must fail
// legalisation rather than silently emit a kernel that computes something
// else, so this test pins the failure behaviour.

module {
  // The model only computes a tile with at least four rows.
  func.func @tile_rows_below_four(%a: memref<2x8xi8>,
                                  %b: memref<8x4xi8>,
                                  %c: memref<2x4xi32>) {
    linalg.matmul ins(%a, %b : memref<2x8xi8>, memref<8x4xi8>)
                  outs(%c : memref<2x4xi32>)
    return
  }

  // The same limit applies to every matmul of a batch.
  func.func @batch_rows_below_four(%a: memref<2x1x8xi8>,
                                   %b: memref<2x8x4xi8>,
                                   %c: memref<2x1x4xi32>) {
    linalg.batch_matmul ins(%a, %b : memref<2x1x8xi8>, memref<2x8x4xi8>)
                        outs(%c : memref<2x1x4xi32>)
    return
  }
}

// ERROR: failed to legalize operation 'linalg.matmul'
// ERROR: unsupported matrix multiply
// ERROR: unsupported matrix multiply
