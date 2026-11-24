// RUN: buddy-opt %s --fold-reduction-transposes | FileCheck %s --check-prefix=TRANSPOSE
// RUN: buddy-opt %s --crop-neutral-reductions | FileCheck %s --check-prefix=CROP
// RUN: buddy-opt %s --fuse-reduction-producers | FileCheck %s --check-prefix=FUSE

#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>

// TRANSPOSE-LABEL: func.func @transpose
// TRANSPOSE-NOT: linalg.transpose
// TRANSPOSE: linalg.generic
// TRANSPOSE-SAME: iterator_types = ["reduction", "parallel", "reduction"]
// TRANSPOSE: arith.subf
func.func @transpose(%input: tensor<2x?x4xf32>, %empty: tensor<4x2x?xf32>, %init: tensor<2xf32>) -> tensor<2xf32> {
  %permuted = linalg.transpose ins(%input : tensor<2x?x4xf32>) outs(%empty : tensor<4x2x?xf32>) permutation = [2, 0, 1]
  %reduced = linalg.reduce ins(%permuted : tensor<4x2x?xf32>) outs(%init : tensor<2xf32>) dimensions = [0, 2]
    (%x: f32, %acc: f32) {
      %next = arith.subf %acc, %x : f32
      linalg.yield %next : f32
    }
  return %reduced : tensor<2xf32>
}

// TRANSPOSE-LABEL: func.func @possibly_aliasing
// TRANSPOSE: linalg.transpose
// TRANSPOSE: linalg.reduce
// CROP-LABEL: func.func @possibly_aliasing
// CROP: linalg.reduce
// FUSE-LABEL: func.func @possibly_aliasing
// FUSE: linalg.reduce
func.func @possibly_aliasing(%input: memref<4x4xi32>, %destination: memref<4x4xi32>, %init: memref<4xi32>) {
  linalg.transpose ins(%input : memref<4x4xi32>) outs(%destination : memref<4x4xi32>) permutation = [1, 0]
  linalg.reduce ins(%destination : memref<4x4xi32>) outs(%init : memref<4xi32>) dimensions = [0]
    (%x: i32, %acc: i32) {
      %next = arith.addi %x, %acc : i32
      linalg.yield %next : i32
    }
  return
}

// CROP-LABEL: func.func @dynamic_rectangle
// CROP: tensor.extract_slice
// CROP: linalg.generic
// CROP-SAME: ins(%arg0 : tensor<?x?xi32>)
// CROP: arith.maxsi
// CROP: tensor.insert_slice
// CROP-NOT: linalg.reduce
func.func @dynamic_rectangle(%input: tensor<?x?xi32>, %valid_rows: index, %valid_cols: index) -> tensor<4xi32> {
  %identity = arith.constant -2147483648 : i32
  %tile = tensor.empty() : tensor<4x32xi32>
  %blank = linalg.fill ins(%identity : i32) outs(%tile : tensor<4x32xi32>) -> tensor<4x32xi32>
  %padded = tensor.insert_slice %input into %blank[0, 0] [%valid_rows, %valid_cols] [1, 1] : tensor<?x?xi32> into tensor<4x32xi32>
  %out = tensor.empty() : tensor<4xi32>
  %init = linalg.fill ins(%identity : i32) outs(%out : tensor<4xi32>) -> tensor<4xi32>
  %reduced = linalg.reduce ins(%padded : tensor<4x32xi32>) outs(%init : tensor<4xi32>) dimensions = [1]
    (%x: i32, %acc: i32) {
      %next = arith.maxsi %x, %acc : i32
      linalg.yield %next : i32
    }
  return %reduced : tensor<4xi32>
}

// CROP-LABEL: func.func @signed_zero
// CROP: tensor.insert_slice
// CROP: linalg.reduce
func.func @signed_zero(%input: tensor<?xf32>, %valid: index) -> tensor<f32> {
  %zero = arith.constant 0.0 : f32
  %tile = tensor.empty() : tensor<32xf32>
  %blank = linalg.fill ins(%zero : f32) outs(%tile : tensor<32xf32>) -> tensor<32xf32>
  %padded = tensor.insert_slice %input into %blank[0] [%valid] [1] : tensor<?xf32> into tensor<32xf32>
  %out = tensor.empty() : tensor<f32>
  %init = linalg.fill ins(%zero : f32) outs(%out : tensor<f32>) -> tensor<f32>
  %reduced = linalg.reduce ins(%padded : tensor<32xf32>) outs(%init : tensor<f32>) dimensions = [0]
    (%x: f32, %acc: f32) {
      %next = arith.addf %x, %acc : f32
      linalg.yield %next : f32
    }
  return %reduced : tensor<f32>
}

// FUSE-LABEL: func.func @fuse
// FUSE: linalg.generic
// FUSE-SAME: iterator_types = ["parallel", "reduction"]
// FUSE: arith.mulf
// FUSE-NEXT: arith.subf
// FUSE-NOT: linalg.reduce
// FUSE: return
func.func @fuse(%input: tensor<?x16xf32>, %empty: tensor<?x16xf32>, %init: tensor<?xf32>) -> tensor<?xf32> {
  %multiplied = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%input : tensor<?x16xf32>) outs(%empty : tensor<?x16xf32>) {
  ^bb0(%x: f32, %unused: f32):
    %square = arith.mulf %x, %x : f32
    linalg.yield %square : f32
  } -> tensor<?x16xf32>
  %reduced = linalg.reduce ins(%multiplied : tensor<?x16xf32>) outs(%init : tensor<?xf32>) dimensions = [1]
    (%x: f32, %acc: f32) {
      %next = arith.subf %acc, %x : f32
      linalg.yield %next : f32
    }
  return %reduced : tensor<?xf32>
}

// FUSE-LABEL: func.func @ordered_sum
// FUSE: linalg.generic
// FUSE-SAME: iterator_types = ["parallel", "parallel"]
// FUSE: arith.mulf
// FUSE: linalg.reduce
// FUSE: arith.addf
func.func @ordered_sum(%input: tensor<?x16xf32>, %empty: tensor<?x16xf32>, %init: tensor<?xf32>) -> tensor<?xf32> {
  %mapped = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%input : tensor<?x16xf32>) outs(%empty : tensor<?x16xf32>) {
  ^bb0(%x: f32, %unused: f32):
    %square = arith.mulf %x, %x : f32
    linalg.yield %square : f32
  } -> tensor<?x16xf32>
  %reduced = linalg.reduce ins(%mapped : tensor<?x16xf32>) outs(%init : tensor<?xf32>) dimensions = [1]
    (%x: f32, %acc: f32) {
      %next = arith.addf %acc, %x : f32
      linalg.yield %next : f32
    }
  return %reduced : tensor<?xf32>
}

// FUSE-LABEL: func.func @ordered_generic_sum
// FUSE: linalg.generic
// FUSE-SAME: iterator_types = ["parallel", "parallel"]
// FUSE: arith.mulf
// FUSE: linalg.generic
// FUSE-SAME: iterator_types = ["parallel", "reduction"]
// FUSE: arith.addf
func.func @ordered_generic_sum(%input: tensor<?x16xf32>, %empty: tensor<?x16xf32>, %init: tensor<?xf32>) -> tensor<?xf32> {
  %mapped = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%input : tensor<?x16xf32>) outs(%empty : tensor<?x16xf32>) {
  ^bb0(%x: f32, %unused: f32):
    %square = arith.mulf %x, %x : f32
    linalg.yield %square : f32
  } -> tensor<?x16xf32>
  %reduced = linalg.generic {indexing_maps = [#id, #row], iterator_types = ["parallel", "reduction"]} ins(%mapped : tensor<?x16xf32>) outs(%init : tensor<?xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %next = arith.addf %x, %acc : f32
    linalg.yield %next : f32
  } -> tensor<?xf32>
  return %reduced : tensor<?xf32>
}

// FUSE-LABEL: func.func @reassociable_sum
// FUSE: linalg.generic
// FUSE-SAME: iterator_types = ["parallel", "reduction"]
// FUSE: arith.mulf
// FUSE-NEXT: arith.addf {{.*}} fastmath<reassoc>
// FUSE-NOT: linalg.reduce
// FUSE: return
func.func @reassociable_sum(%input: tensor<?x16xf32>, %empty: tensor<?x16xf32>, %init: tensor<?xf32>) -> tensor<?xf32> {
  %mapped = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%input : tensor<?x16xf32>) outs(%empty : tensor<?x16xf32>) {
  ^bb0(%x: f32, %unused: f32):
    %square = arith.mulf %x, %x : f32
    linalg.yield %square : f32
  } -> tensor<?x16xf32>
  %reduced = linalg.reduce ins(%mapped : tensor<?x16xf32>) outs(%init : tensor<?xf32>) dimensions = [1]
    (%x: f32, %acc: f32) {
      %next = arith.addf %acc, %x fastmath<reassoc> : f32
      linalg.yield %next : f32
    }
  return %reduced : tensor<?xf32>
}
