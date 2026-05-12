// RUN: buddy-opt -verify-diagnostics %s

func.func @scatter_rejects_float_index_vector(
    %value: !vir.vec<?xf32>, %base: memref<?xf32>,
    %index: !vir.vec<?xf32>, %mask: !vir.vec<?xi1>) {
  // expected-error@+1 {{scatter index vector element type must be integer or index}}
  vir.scatter %value, %base[][%index], %mask : !vir.vec<?xf32>, !vir.vec<?xf32>, !vir.vec<?xi1> -> memref<?xf32>
  return
}

func.func @scatter_rejects_shape_mismatch(
    %value: !vir.vec<2x?xf32>, %base: memref<?xf32>,
    %index: !vir.vec<?xi32>, %mask: !vir.vec<2x?xi1>) {
  // expected-error@+1 {{scatter value and index vector shapes must match}}
  vir.scatter %value, %base[][%index], %mask : !vir.vec<2x?xf32>, !vir.vec<?xi32>, !vir.vec<2x?xi1> -> memref<?xf32>
  return
}

func.func @scatter_rejects_scaling_factor_mismatch(
    %value: !vir.vec<?xf32, m2>, %base: memref<?xf32>,
    %index: !vir.vec<?xi32, f2>, %mask: !vir.vec<?xi1, m2>) {
  // expected-error@+1 {{scatter value and index vector scaling factors must match}}
  vir.scatter %value, %base[][%index], %mask : !vir.vec<?xf32, m2>, !vir.vec<?xi32, f2>, !vir.vec<?xi1, m2> -> memref<?xf32>
  return
}

func.func @scatter_rejects_value_memref_element_type_mismatch(
    %value: !vir.vec<?xf32>, %base: memref<?xi32>,
    %index: !vir.vec<?xi32>, %mask: !vir.vec<?xi1>) {
  // expected-error@+1 {{scatter value element type must match memref element type}}
  vir.scatter %value, %base[][%index], %mask : !vir.vec<?xf32>, !vir.vec<?xi32>, !vir.vec<?xi1> -> memref<?xi32>
  return
}

func.func @scatter_rejects_mask_shape_mismatch(
    %value: !vir.vec<2x?xf32>, %base: memref<?xf32>,
    %index: !vir.vec<2x?xi32>, %mask: !vir.vec<?xi1>) {
  // expected-error@+1 {{scatter value and mask vector shapes must match}}
  vir.scatter %value, %base[][%index], %mask : !vir.vec<2x?xf32>, !vir.vec<2x?xi32>, !vir.vec<?xi1> -> memref<?xf32>
  return
}

func.func @scatter_rejects_non_i1_mask(
    %value: !vir.vec<?xf32>, %base: memref<?xf32>,
    %index: !vir.vec<?xi32>, %mask: !vir.vec<?xi32>) {
  // expected-error@+1 {{scatter mask vector element type must be i1}}
  vir.scatter %value, %base[][%index], %mask : !vir.vec<?xf32>, !vir.vec<?xi32>, !vir.vec<?xi32> -> memref<?xf32>
  return
}
