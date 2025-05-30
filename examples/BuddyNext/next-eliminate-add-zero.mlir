// RUN: buddy-opt %s \
// RUN:     -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" \
// RUN: | buddy-opt \
// RUN:     -arith-expand \
// RUN:     -eliminate-empty-tensors \
// RUN:     -empty-tensor-to-alloc-tensor \
// RUN:     -one-shot-bufferize="bufferize-function-boundaries" \
// RUN:     -convert-linalg-to-affine-loops \
// RUN:     -affine-loop-fusion \
// RUN:     -lower-affine \
// RUN:     -convert-vector-to-scf \
// RUN:     -expand-strided-metadata \
// RUN:     -convert-vector-to-llvm \
// RUN:     -memref-expand \
// RUN:     -arith-expand \
// RUN:     -convert-arith-to-llvm \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-cf-to-llvm \
// RUN:     -convert-openmp-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-math-to-libm  \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s
module {
    func.func private @printMemrefF32(tensor<*xf32>)
    func.func private @rtclock() -> f64

    func.func @uvue_original() {
        %t0_original = call @rtclock() : () -> f64

        %84 = arith.constant dense<2.0> : tensor<1x32x40x128xf32>
        %92 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
        %93 = tosa.add %84, %92 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
        %94 = tosa.reshape %93 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>

        %t1_original = call @rtclock() : () -> f64
        %tensor_unranked = tensor.cast %94 : tensor<32x40x128xf32> to tensor<*xf32>

        // All the elements of the MemRef are the same,
        // only check the first line to verify the correctness.
        // CHECK: Unranked Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [32, 40, 128] strides = [5120, 128, 1] data =
        // CHECK-NEXT: [
        // CHECK-SAME: [
        // CHECK-SAME: [2{{(, 2)*}}],

        // Print results.
        call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
        // Print timings.

        %t_original = arith.subf %t1_original, %t0_original : f64
        vector.print str "original operation time: "
        vector.print %t_original : f64
        return
    }

    func.func @uve_optimized() {
        %t0_optimized = call @rtclock() : () -> f64

        %84 = arith.constant dense<2.0> : tensor<1x32x40x128xf32>
        %94 = tosa.reshape %84 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
        %t1_optimized = call @rtclock() : () -> f64

        %tensor_unranked = tensor.cast %94 : tensor<32x40x128xf32> to tensor<*xf32>



        // Print results.
        call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
        // Print timings.

        %t_optimized = arith.subf %t1_optimized, %t0_optimized : f64
        vector.print str "optimized operation time: "
        vector.print %t_optimized : f64
        return
    }


    func.func @main() {
        %84 = arith.constant dense<2.0> : tensor<1x32x40x128xf32>

        call @uvue_original() : () -> ()
        call @uve_optimized() : () -> ()

        return
    }
}
