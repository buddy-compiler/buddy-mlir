// RUN: buddy-opt %s \
// RUN:     -convert-vector-to-llvm \
// RUN:     -convert-cf-to-llvm \
// RUN:     -convert-func-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

// The example is equivalent to the following code.
// int main() {
//   int val = 0;
//   for (int i = 1; i < 5; i++) {
//     val += 5;
//     if (i == 3) {
//       std::cout << val << std::endl;
//       return 0;
//     }
//   }
//   return 0;
// }

module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c5 = arith.constant 5 : index
    %c1 = arith.constant 1 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_5 = arith.constant 5.000000e+00 : f32
    cf.br ^bb1(%c0, %cst_0 : index, f32)
  ^bb1(%0: index, %1: f32):
    %2 = arith.cmpi slt, %0, %c5 : index
    cf.cond_br %2, ^bb2, ^bb4(%1: f32)
  ^bb2:
    %3 = arith.addf %1, %cst_5 : f32
    %4 = arith.addi %0, %c1 : index
    cf.br ^bb3 (%4, %3 : index, f32)
  ^bb3(%iter_idx: index, %iter_var: f32):
    %eq = arith.cmpi eq, %iter_idx, %c3 : index
    cf.cond_br %eq, ^bb4(%iter_var: f32), ^bb1(%iter_idx, %iter_var: index, f32)
  ^bb4(%ret_var: f32):
    // CHECK: 15
    vector.print %ret_var : f32
    return
  }
}
