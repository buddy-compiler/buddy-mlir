// RUN: buddy-opt %s \
// RUN:     --simplify-tosa-add-reshape \
// RUN: | buddy-opt \
// RUN:     -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" \
// RUN: | buddy-opt \
// RUN:     -arith-expand \
// RUN:     -eliminate-empty-tensors \
// RUN:     -empty-tensor-to-alloc-tensor \
// RUN:     -one-shot-bufferize \
// RUN:     -convert-linalg-to-affine-loops \
// RUN:     -affine-loop-fusion \
// RUN:     -lower-affine \
// RUN:     -func-bufferize \
// RUN:     -arith-bufferize \
// RUN:     -tensor-bufferize \
// RUN:     -buffer-deallocation \
// RUN:     -finalizing-bufferize \
// RUN:     -convert-vector-to-scf \
// RUN:     -expand-strided-metadata \
// RUN:     -convert-vector-to-llvm \
// RUN:     -memref-expand \
// RUN:     -arith-expand \
// RUN:     -convert-arith-to-llvm \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-openmp-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-math-to-libm  \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

module {
    func.func private @printMemrefF32(tensor<*xf32>)
    func.func private @rtclock() -> f64

    func.func @const_add_reshape() -> tensor<32x40x128xf32> {
        %0 = "tosa.const"() <{value = dense<3.5> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
        %1 = "tosa.const"() <{value = dense<3.5> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
        %2 = tosa.add %0, %1 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
        %3 = tosa.reshape %2 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>

        return %3 : tensor<32x40x128xf32>
    }

    func.func @main() {
        %t0_original = call @rtclock() : () -> f64 

        %res = call @const_add_reshape() : () -> tensor<32x40x128xf32>
        %t1_original = call @rtclock() : () -> f64
        %tensor_unranked = tensor.cast %res : tensor<32x40x128xf32> to tensor<*xf32>

        // All the elements of the MemRef are the same,
        // only check the first line to verify the correctness.
        // CHECK: Unranked Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [32, 40, 128] strides = [5120, 128, 1] data = 
        // CHECK-NEXT: [
        // CHECK-SAME: [
        // CHECK-SAME: [7{{(, 7)*}}],

        // Print results.
        call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
        // Print timings.

        %t_original = arith.subf %t1_original, %t0_original : f64
        vector.print %t_original : f64

        return 
    }
}
