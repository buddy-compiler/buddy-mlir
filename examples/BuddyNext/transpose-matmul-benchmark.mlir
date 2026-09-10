// RUN: buddy-opt %s \
// RUN:   -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" \
// RUN: | buddy-opt \
// RUN:   -eliminate-empty-tensors \
// RUN:   -empty-tensor-to-alloc-tensor \
// RUN:   -convert-elementwise-to-linalg \
// RUN:   -convert-tensor-to-linalg \
// RUN:   -one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map unknown-type-conversion=identity-layout-map" \
// RUN:   -expand-strided-metadata \
// RUN:   -batchmatmul-optimize \
// RUN:   -convert-linalg-to-affine-loops \
// RUN:   -lower-affine \
// RUN:   -convert-vector-to-scf \
// RUN:   -convert-vector-to-llvm \
// RUN:   -convert-arith-to-llvm \
// RUN:   -finalize-memref-to-llvm \
// RUN:   -convert-scf-to-cf \
// RUN:   -convert-cf-to-llvm \
// RUN:   -convert-openmp-to-llvm \
// RUN:   -convert-math-to-llvm \
// RUN:   -convert-math-to-libm \
// RUN:   -convert-func-to-llvm \
// RUN:   -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:   -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext
//
// The main routine dispatches three representative transpose + (batch) matmul
// patterns distilled from DeepSeek R1 decode subgraphs. Each case captures the
// timing of (1) the transpose kernel, (2) the matmul kernel and (3) the overall
// fused pipeline, then prints the transpose time ratio. The order of prints per
// case is: case-id, transpose-time (s), matmul-time (s), total-time (s),
// transpose-ratio.

module {
  func.func private @rtclock() -> f64

  func.func @display_transpose_matmul_detail(
    %transpose_time: f64,
    %matmul_time: f64,
    %total_time: f64,
    %ratio: f64
  ) {
    vector.print str "Transpose time (s): "
    vector.print %transpose_time : f64

    vector.print str "Matmul time (s): "
    vector.print %matmul_time : f64

    vector.print str "Total time (s): "
    vector.print %total_time : f64

    vector.print str "Transpose ratio: "
    vector.print %ratio : f64

    vector.print str "\n"

    return
  }

  // Case 0 -------------------------------------------------------------------
  // Pattern: transpose -> matmul for MLP projection (pre-transposed weights).
  func.func @transpose_matmul_case0(%lhs : tensor<1x40x1536xf32>,
                                    %rhs : tensor<1x256x1536xf32>)
      -> tensor<1x40x256xf32> {
    vector.print str "Case 0 1x40x1536 @ 1x256x1536\n"

    %perm_lhs = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %perm_rhs = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>

    %t0 = call @rtclock() : () -> f64
    %lhs_t = tosa.transpose %lhs, %perm_lhs : (tensor<1x40x1536xf32>, tensor<3xi32>) -> tensor<1x1536x40xf32>
    %lhs_ready = tosa.transpose %lhs_t, %perm_lhs : (tensor<1x1536x40xf32>, tensor<3xi32>) -> tensor<1x40x1536xf32>
    %t1 = call @rtclock() : () -> f64

    %transpose_time = arith.subf %t1, %t0 : f64

    %rhs_ready = tosa.transpose %rhs, %perm_rhs : (tensor<1x256x1536xf32>, tensor<3xi32>) -> tensor<1x1536x256xf32>

    %t2 = call @rtclock() : () -> f64
    %mat = tosa.matmul %lhs_ready, %rhs_ready : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %t3 = call @rtclock() : () -> f64

    %mat_time = arith.subf %t3, %t2 : f64
    %total_time = arith.addf %transpose_time, %mat_time : f64
    %ratio = arith.divf %transpose_time, %total_time : f64

    call @display_transpose_matmul_detail(%transpose_time, %mat_time, %total_time, %ratio) : (f64, f64, f64, f64) -> ()

    return %mat : tensor<1x40x256xf32>
  }

  // Case 1 -------------------------------------------------------------------
  // Pattern: batch_matmul_transpose_b where only the RHS is transposed.
  func.func @transpose_matmul_case1(%lhs : tensor<1x12x1024x128xf32>,
                                    %rhs : tensor<1x12x96x128xf32>)
      -> tensor<1x12x1024x96xf32> {
    %c1 = arith.constant 1 : i32
  vector.print str "Case 1 1x12x1024x128 @ 1x12x96x128\n"

    %perm_rhs = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>

    %t0 = call @rtclock() : () -> f64
    %rhs_t = tosa.transpose %rhs, %perm_rhs : (tensor<1x12x96x128xf32>, tensor<4xi32>) -> tensor<1x12x128x96xf32>
    %t1 = call @rtclock() : () -> f64

    %transpose_time = arith.subf %t1, %t0 : f64

    %lhs_bc = tosa.reshape %lhs {new_shape = array<i64: 12, 1024, 128>} : (tensor<1x12x1024x128xf32>) -> tensor<12x1024x128xf32>
    %rhs_bc = tosa.reshape %rhs_t {new_shape = array<i64: 12, 128, 96>} : (tensor<1x12x128x96xf32>) -> tensor<12x128x96xf32>

    %t2 = call @rtclock() : () -> f64
    %mat = tosa.matmul %lhs_bc, %rhs_bc : (tensor<12x1024x128xf32>, tensor<12x128x96xf32>) -> tensor<12x1024x96xf32>
    %t3 = call @rtclock() : () -> f64

    %mat_time = arith.subf %t3, %t2 : f64
    %total_time = arith.addf %transpose_time, %mat_time : f64
    %ratio = arith.divf %transpose_time, %total_time : f64

    call @display_transpose_matmul_detail(%transpose_time, %mat_time, %total_time, %ratio) : (f64, f64, f64, f64) -> ()

    %res = tosa.reshape %mat {new_shape = array<i64: 1, 12, 1024, 96>} : (tensor<12x1024x96xf32>) -> tensor<1x12x1024x96xf32>
    return %res : tensor<1x12x1024x96xf32>
  }

  // Case 2 -------------------------------------------------------------------
  // Pattern: reshape -> batch_matmul_transpose_b canonical form.
  func.func @transpose_matmul_case2(%lhs : tensor<8x64x32xf32>,
                                    %rhs : tensor<8x24x32xf32>)
      -> tensor<8x64x24xf32> {
  vector.print str "Case 2 8x64x32 @ 8x24x32\n"

    %perm_rhs = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %t0 = call @rtclock() : () -> f64
    %rhs_t = tosa.transpose %rhs, %perm_rhs : (tensor<8x24x32xf32>, tensor<3xi32>) -> tensor<8x32x24xf32>
    %t1 = call @rtclock() : () -> f64

    %transpose_time = arith.subf %t1, %t0 : f64

    %t2 = call @rtclock() : () -> f64
    %mat = tosa.matmul %lhs, %rhs_t : (tensor<8x64x32xf32>, tensor<8x32x24xf32>) -> tensor<8x64x24xf32>
    %t3 = call @rtclock() : () -> f64

    %mat_time = arith.subf %t3, %t2 : f64
    %total_time = arith.addf %transpose_time, %mat_time : f64
    %ratio = arith.divf %transpose_time, %total_time : f64

    call @display_transpose_matmul_detail(%transpose_time, %mat_time, %total_time, %ratio) : (f64, f64, f64, f64) -> ()

    return %mat : tensor<8x64x24xf32>
  }

  func.func @main() {
    %lhs0 = arith.constant dense<2.0> : tensor<1x40x1536xf32>
    %rhs0 = arith.constant dense<1.0> : tensor<1x256x1536xf32>
    %0 = call @transpose_matmul_case0(%lhs0, %rhs0)
      : (tensor<1x40x1536xf32>, tensor<1x256x1536xf32>) -> tensor<1x40x256xf32>
    %lhs1 = arith.constant dense<3.0> : tensor<1x12x1024x128xf32>
    %rhs1 = arith.constant dense<4.0> : tensor<1x12x96x128xf32>
    %1 = call @transpose_matmul_case1(%lhs1, %rhs1)
      : (tensor<1x12x1024x128xf32>, tensor<1x12x96x128xf32>) -> tensor<1x12x1024x96xf32>
    %lhs2 = arith.constant dense<5.0> : tensor<8x64x32xf32>
    %rhs2 = arith.constant dense<6.0> : tensor<8x24x32xf32>
    %2 = call @transpose_matmul_case2(%lhs2, %rhs2)
      : (tensor<8x64x32xf32>, tensor<8x24x32xf32>) -> tensor<8x64x24xf32>

    return
  }
}
