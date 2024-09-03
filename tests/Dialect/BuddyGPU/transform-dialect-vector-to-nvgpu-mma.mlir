// RUN: buddy-opt --split-input-file --transform-interpreter %s | FileCheck %s


#matmat_accesses = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (m, n)>
]

#matmat_trait = {
  indexing_maps = #matmat_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

func.func @wmma(%a: memref<16x16xf32>, %b: memref<16x16xf32>, %c: memref<16x16xf32>) {
  %c0 = arith.constant 0: index
  %cst = arith.constant 0.0: f32
  %va = vector.transfer_read %a[%c0, %c0], %cst: memref<16x16xf32>, vector<16x16xf32>
  %vb = vector.transfer_read %b[%c0, %c0], %cst: memref<16x16xf32>, vector<16x16xf32>
  %vc = vector.transfer_read %c[%c0, %c0], %cst: memref<16x16xf32>, vector<16x16xf32>

  // CHECK-NOT: vector.contract
  //     CHECK:  gpu.subgroup_mma_compute
  %vres = vector.contract #matmat_trait %va, %vb, %vc
    : vector<16x16xf32>, vector<16x16xf32> into vector<16x16xf32>
  vector.transfer_write %vres, %c[%c0, %c0]: vector<16x16xf32>, memref<16x16xf32>
  return
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(
      %module: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %module
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.buddy.unroll_vectors_gpu_mma_sync
    } : !transform.any_op
    transform.buddy.vector.vector_to_mma_conversion %func { use_wmma } : (!transform.any_op) -> ()

    // Apply canonicalization post-hoc to trigger DCE and pass the test
    // (i.e. all vector.contract are dead).
    // TODO: consider having the vector_to_mma_conversion do the DCE automatically.
    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op

    transform.yield
  }
}

// -----

#matmat_accesses = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (m, n)>
]
#matmat_trait = {
  indexing_maps = #matmat_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

func.func @mma_sync(%a: memref<16x16xf32>, %b: memref<16x16xf32>, %c: memref<16x16xf32>) {
  %c0 = arith.constant 0: index
  %cst = arith.constant 0.0: f32
  %va = vector.transfer_read %a[%c0, %c0], %cst: memref<16x16xf32>, vector<16x16xf32>
  %vb = vector.transfer_read %b[%c0, %c0], %cst: memref<16x16xf32>, vector<16x16xf32>
  %vc = vector.transfer_read %c[%c0, %c0], %cst: memref<16x16xf32>, vector<16x16xf32>

  // CHECK-NOT: vector.contract
  //     CHECK: nvgpu.mma.sync{{.*}} tf32Enabled}
  %vres = vector.contract #matmat_trait %va, %vb, %vc
    : vector<16x16xf32>, vector<16x16xf32> into vector<16x16xf32>
  vector.transfer_write %vres, %c[%c0, %c0]: vector<16x16xf32>, memref<16x16xf32>
  return
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(
      %module: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %module
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.buddy.unroll_vectors_gpu_mma_sync
    } : !transform.any_op
    transform.buddy.vector.vector_to_mma_conversion %func { use_mma_sync } : (!transform.any_op) -> ()

    // Apply canonicalization post-hoc to trigger DCE and pass the test
    // (i.e. all vector.contract are dead).
    // TODO: consider having the vector_to_mma_conversion do the DCE automatically.
    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op

    transform.yield
  }
}
