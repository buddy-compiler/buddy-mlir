// RUN: sed 's/STEP_PLACEHOLDER/4/g;s/SIZE_PLACEHOLDER/4096/g' %s | \
// RUN: buddy-opt \
// RUN:     -convert-vector-to-scf \
// RUN:     -lower-affine \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-cf-to-llvm \
// RUN:     -convert-vector-to-llvm \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s --check-prefix=CHECK-EXEC

func.func @saxpy_fixed(%size : index, %x: memref<?xf32>, %y: memref<?xf32>, %a: f32) {
  %c0 = arith.constant 0 : index
  // %size = memref.dim %x, %c0 : memref<?xf32>
  %step = arith.constant STEP_PLACEHOLDER : index
  %c1 = arith.constant 1 : index
  %i_bound_ = arith.subi %size, %step : index
  %i_bound = arith.addi %i_bound_, %c1 : index
  %a_vec = vector.broadcast %a : f32 to vector<STEP_PLACEHOLDERxf32>
  %iter_idx_init = arith.constant 0 : index
  // body
  %iter_idx = scf.for %i = %c0 to %i_bound step %step
      iter_args(%iter = %iter_idx_init) -> (index) {
    %xi = vector.load %x[%i] : memref<?xf32>, vector<STEP_PLACEHOLDERxf32>
    %yi = vector.load %y[%i] : memref<?xf32>, vector<STEP_PLACEHOLDERxf32>
    %updated_yi = vector.fma %a_vec, %xi, %yi : vector<STEP_PLACEHOLDERxf32>
    vector.store %updated_yi, %y[%i] : memref<?xf32>, vector<STEP_PLACEHOLDERxf32>
    scf.yield %i : index
  }
  // tail
  %iter_tail = arith.addi %iter_idx, %step : index
  affine.for %i = %iter_tail to %size {
    %xi = affine.load %x[%i] : memref<?xf32>
    %yi = affine.load %y[%i] : memref<?xf32>
    %axi = arith.mulf %a, %xi : f32
    %updated_yi = arith.addf %axi, %yi : f32
    affine.store %updated_yi, %y[%i] : memref<?xf32>
  }
  return
}

func.func @alloc_f32(%len: index, %val: f32) -> memref<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.alloc(%len) : memref<?xf32>
  scf.for %i = %c0 to %len step %c1 {
    memref.store %val, %0[%i] : memref<?xf32>
  }
  return %0 : memref<?xf32>
}

func.func @main() {
  %size = arith.constant SIZE_PLACEHOLDER : index
  %f2 = arith.constant 2.0 : f32
  %f3 = arith.constant 3.0 : f32
  %x = func.call @alloc_f32(%size, %f2) : (index, f32) -> memref<?xf32>
  %y = func.call @alloc_f32(%size, %f3) : (index, f32) -> memref<?xf32>
  %a = arith.constant 5.0 : f32

  %t_start = call @rtclock() : () -> f64
  func.call @saxpy_fixed(%size, %x, %y, %a) : (index, memref<?xf32>, memref<?xf32>, f32) -> ()
  %t_end = call @rtclock() : () -> f64

  %time = arith.subf %t_end, %t_start : f64
  vector.print %time : f64
  // CHECK-EXEC: {{[0-9]+\.[0-9]+}}

  return
}

func.func private @rtclock() -> f64
