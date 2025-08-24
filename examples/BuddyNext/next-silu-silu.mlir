// RUN: buddy-opt %s \
// RUN:     -pass-pipeline="builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" \
// RUN: | buddy-opt \
// RUN:     -convert-linalg-to-loops \
// RUN:     -lower-affine \
// RUN:     -convert-vector-to-scf \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-cf-to-llvm \
// RUN:     -convert-vector-to-llvm \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts  \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s
#map = affine_map<(d0) -> (d0)>
  func.func private @rtclock() -> f64
  func.func private @printMemrefF32(%ptr: memref<*xf32>) attributes {llvm.emit_c_interface}

  func.func @kernel(%arg0: memref<1x40x8960xf32>) {
    %t_start = call @rtclock() : () -> f64

    %output = memref.alloc() : memref<1x40x8960xf32>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst_1f = arith.constant 1.0 : f32
    %vec_1f = vector.broadcast %cst_1f : f32 to vector<8xf32>
    %cst_0f = arith.constant 0.0 : f32 // for padding

    %d0 = memref.dim %arg0, %c0 : memref<1x40x8960xf32>
    %d1 = memref.dim %arg0, %c1 : memref<1x40x8960xf32>
    %d2 = memref.dim %arg0, %c2 : memref<1x40x8960xf32>

    affine.for %i = #map(%c0) to #map(%d0) {
      affine.for %j = #map(%c0) to #map(%d1) {
        affine.for %k = #map(%c0) to #map(%d2) step 8 {
          %x_vec = vector.transfer_read %arg0[%i, %j, %k], %cst_0f : memref<1x40x8960xf32>, vector<8xf32>
          %neg_x_vec = arith.negf %x_vec : vector<8xf32>
          %exp_neg_x_vec = math.exp %neg_x_vec : vector<8xf32>
          %one_plus_exp_vec = arith.addf %vec_1f, %exp_neg_x_vec : vector<8xf32>
          %sigmoid_x_vec = arith.divf %vec_1f, %one_plus_exp_vec : vector<8xf32>
          %silu_vec = arith.mulf %x_vec, %sigmoid_x_vec : vector<8xf32>
          vector.transfer_write %silu_vec, %output[%i, %j, %k] : vector<8xf32>, memref<1x40x8960xf32>
        }
      }
    }

    %t_end = call @rtclock() : () -> f64
    %unranked_result = memref.cast %output : memref<1x40x8960xf32> to memref<*xf32>
    // CHECK: Unranked Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [1, 40, 8960] strides = [358400, 8960, 1] data =
    // CHECK-NEXT: [
    // CHECK-SAME: [2.85772{{(, 2.85772)*}}],
    call @printMemrefF32(%unranked_result) : (memref<*xf32>) -> ()
    memref.dealloc %output : memref<1x40x8960xf32>

    %time = arith.subf %t_end, %t_start : f64
    vector.print %time : f64

    return
  }

  func.func @main() {
    %input = memref.alloc() : memref<1x40x8960xf32>
    %cst_neg_1_23 = arith.constant 3.0 : f32
    linalg.fill ins(%cst_neg_1_23 : f32) outs(%input : memref<1x40x8960xf32>)

    call @kernel(%input) : (memref<1x40x8960xf32>) -> ()

    memref.dealloc %input : memref<1x40x8960xf32>

    return
  }
