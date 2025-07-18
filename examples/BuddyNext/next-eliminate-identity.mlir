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

    func.func @ie_original() {
        %t0_original = call @rtclock() : () -> f64

        %119 = arith.constant dense<1.0> : tensor<1x40x32x128xf32>
        %120 = tosa.identity %119 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
        %121 = tosa.reshape %120 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
        %t1_original = call @rtclock() : () -> f64

        %tensor_unranked = tensor.cast %121 : tensor<1x40x4096xf32> to tensor<*xf32>
        // All the elements of the MemRef are the same,
        // only check the first line to verify the correctness.
        // CHECK: Unranked Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [1, 40, 4096] strides = [163840, 4096, 1] data =
        // CHECK-NEXT: [
        // CHECK-SAME: [
        // CHECK-SAME: [1{{(, 1)*}}],

        // Print results.
        call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
        // Print timings.

        %t_original = arith.subf %t1_original, %t0_original : f64
        vector.print str "original operation time: "
        vector.print %t_original : f64
        return
    }

    func.func @ie_optimized() {
        %t0_optimized = call @rtclock() : () -> f64

        %119 = arith.constant dense<1.0> : tensor<1x40x32x128xf32>
        %121 = tosa.reshape %119 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
        %t1_optimized = call @rtclock() : () -> f64

        %tensor_unranked = tensor.cast %121 : tensor<1x40x4096xf32> to tensor<*xf32>
        // All the elements of the MemRef are the same,
        // only check the first line to verify the correctness.
        // CHECK: Unranked Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [1, 40, 4096] strides = [163840, 4096, 1] data =
        // CHECK-NEXT: [
        // CHECK-SAME: [
        // CHECK-SAME: [1{{(, 1)*}}],

        // Print results.
        call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
        // Print timings.

        %t_optimized = arith.subf %t1_optimized, %t0_optimized : f64
        vector.print str "optimized operation time: "
        vector.print %t_optimized : f64
        return
    }

    func.func @main() {

        call @ie_original() : () -> ()
        call @ie_optimized() : () -> ()

        return
    }
}
