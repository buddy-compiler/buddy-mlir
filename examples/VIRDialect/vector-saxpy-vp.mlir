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
// RUN: | FileCheck %s

func.func @saxpy_vp(%size : index, %x: memref<?xf32>, %y: memref<?xf32>, %a: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %vs = vector.vscale
  %factor = arith.constant STEP_PLACEHOLDER : index
  %step = arith.muli %vs, %factor : index

  %mask = arith.constant dense<1> : vector<[STEP_PLACEHOLDER]xi1>
  %a_vec = vector.broadcast %a : f32 to vector<[STEP_PLACEHOLDER]xf32>
  %vl_total_i32 = index.casts %size : index to i32
  %vl_step_i32 = index.casts %step : index to i32

  // body
  %iter_vl = scf.for %i = %c0 to %size step %step
      iter_args(%iter_vl_i32 = %vl_total_i32) -> (i32) {

    %xi = vector_exp.predication %mask, %iter_vl_i32 : vector<[STEP_PLACEHOLDER]xi1>, i32 {
      %ele = vector.load %x[%i] : memref<?xf32>, vector<[STEP_PLACEHOLDER]xf32>
      vector.yield %ele : vector<[STEP_PLACEHOLDER]xf32>
    } : vector<[STEP_PLACEHOLDER]xf32>

    %yi = vector_exp.predication %mask, %iter_vl_i32 : vector<[STEP_PLACEHOLDER]xi1>, i32 {
      %ele = vector.load %y[%i] : memref<?xf32>, vector<[STEP_PLACEHOLDER]xf32>
      vector.yield %ele : vector<[STEP_PLACEHOLDER]xf32>
    } : vector<[STEP_PLACEHOLDER]xf32>

    %updated_yi = "llvm.intr.vp.fma" (%a_vec, %xi, %yi, %mask, %iter_vl_i32) :
      (vector<[STEP_PLACEHOLDER]xf32>, vector<[STEP_PLACEHOLDER]xf32>, vector<[STEP_PLACEHOLDER]xf32>, vector<[STEP_PLACEHOLDER]xi1>, i32) -> vector<[STEP_PLACEHOLDER]xf32>

    vector_exp.predication %mask, %iter_vl_i32 : vector<[STEP_PLACEHOLDER]xi1>, i32 {
      vector.store %updated_yi, %y[%i] : memref<?xf32>, vector<[STEP_PLACEHOLDER]xf32>
      vector.yield
    } : () -> ()

    // Update dynamic vector length.
    %new_vl = arith.subi %iter_vl_i32, %vl_step_i32 : i32
    scf.yield %new_vl : i32
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
  func.call @saxpy_vp(%size, %x, %y, %a) : (index, memref<?xf32>, memref<?xf32>, f32) -> ()
  %t_end = call @rtclock() : () -> f64

  %time = arith.subf %t_end, %t_start : f64
  vector.print %time : f64
  // CHECK: print{{.*}}

  return
}

func.func private @rtclock() -> f64
