// RUN: buddy-opt %s \
// RUN:     -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" \
// RUN: | buddy-opt \
// RUN:     -eliminate-empty-tensors \
// RUN:     -empty-tensor-to-alloc-tensor \
// RUN:     -convert-elementwise-to-linalg \
// RUN:     -one-shot-bufferize="unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries" \
// RUN:     -expand-strided-metadata \
// RUN:     -ownership-based-buffer-deallocation \
// RUN:     -buffer-deallocation-simplification \
// RUN:     -bufferization-lower-deallocations \
// RUN:     -convert-bufferization-to-memref \
// RUN:     -matmul-parallel-vectorization-optimize \
// RUN:     -batchmatmul-optimize \
// RUN:     -convert-linalg-to-affine-loops \
// RUN:     -affine-loop-fusion \
// RUN:     -affine-parallelize \
// RUN:     -convert-vector-to-scf \
// RUN:     -lower-affine \
// RUN:     -convert-scf-to-openmp \
// RUN:     -func-bufferize-dynamic-offset \
// RUN:     -cse \
// RUN:     -memref-expand \
// RUN:     -arith-expand \
// RUN:     -convert-vector-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-cf-to-llvm \
// RUN:     -convert-openmp-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-math-to-libm \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libomp%shlibext \
// RUN: | FileCheck %s

func.func private @rtclock() -> f64
func.func private @printMemrefF32(%ptr : tensor<*xf32>)

func.func @kernel(%arg0: tensor<1x1024x1536xf32>, %arg1: tensor<1536xf32>, %arg2 : tensor<256xf32>, %arg3: tensor<1536x1536xf32>, %arg4: tensor<256x1536xf32>, %arg5: tensor<1x1024x128xf32>) -> (tensor<1x12x1024x128xf32>, tensor<1x12x1024x128xf32>, tensor<1x12x1024x128xf32>) {
  %t_start = call @rtclock() : () -> f64

  %67 = tosa.reshape %arg0 {new_shape = array<i64: 1024, 1536>} : (tensor<1x1024x1536xf32>) -> tensor<1024x1536xf32>
  %68 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %69 = tosa.transpose %arg3, %68 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
  %70 = tosa.reshape %67 {new_shape = array<i64: 1, 1024, 1536>} : (tensor<1024x1536xf32>) -> tensor<1x1024x1536xf32>
  %71 = tosa.reshape %69 {new_shape = array<i64: 1, 1536, 1536>} : (tensor<1536x1536xf32>) -> tensor<1x1536x1536xf32>
  %72 = tosa.matmul %70, %71 : (tensor<1x1024x1536xf32>, tensor<1x1536x1536xf32>) -> tensor<1x1024x1536xf32>
  %73 = tosa.reshape %72 {new_shape = array<i64: 1024, 1536>} : (tensor<1x1024x1536xf32>) -> tensor<1024x1536xf32>
  %74 = tosa.reshape %arg1 {new_shape = array<i64: 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
  %75 = tosa.add %74, %73 : (tensor<1x1536xf32>, tensor<1024x1536xf32>) -> tensor<1024x1536xf32>
  %76 = tosa.reshape %75 {new_shape = array<i64: 1, 1024, 1536>} : (tensor<1024x1536xf32>) -> tensor<1x1024x1536xf32>
  %77 = tosa.reshape %arg0 {new_shape = array<i64: 1024, 1536>} : (tensor<1x1024x1536xf32>) -> tensor<1024x1536xf32>
  %78 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %79 = tosa.transpose %arg4, %78 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
  %80 = tosa.reshape %77 {new_shape = array<i64: 1, 1024, 1536>} : (tensor<1024x1536xf32>) -> tensor<1x1024x1536xf32>
  %81 = tosa.reshape %79 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
  %82 = tosa.matmul %80, %81 : (tensor<1x1024x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x1024x256xf32>
  %83 = tosa.reshape %82 {new_shape = array<i64: 1024, 256>} : (tensor<1x1024x256xf32>) -> tensor<1024x256xf32>
  %84 = tosa.reshape %arg2 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
  %85 = tosa.add %84, %83 : (tensor<1x256xf32>, tensor<1024x256xf32>) -> tensor<1024x256xf32>
  %86 = tosa.reshape %85 {new_shape = array<i64: 1, 1024, 256>} : (tensor<1024x256xf32>) -> tensor<1x1024x256xf32>
  %87 = tosa.reshape %arg0 {new_shape = array<i64: 1024, 1536>} : (tensor<1x1024x1536xf32>) -> tensor<1024x1536xf32>
  %88 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %89 = tosa.transpose %arg4, %88 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
  %90 = tosa.reshape %87 {new_shape = array<i64: 1, 1024, 1536>} : (tensor<1024x1536xf32>) -> tensor<1x1024x1536xf32>
  %91 = tosa.reshape %89 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
  %92 = tosa.matmul %90, %91 : (tensor<1x1024x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x1024x256xf32>
  %93 = tosa.reshape %92 {new_shape = array<i64: 1024, 256>} : (tensor<1x1024x256xf32>) -> tensor<1024x256xf32>
  %94 = tosa.reshape %arg2 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
  %95 = tosa.add %94, %93 : (tensor<1x256xf32>, tensor<1024x256xf32>) -> tensor<1024x256xf32>
  %96 = tosa.reshape %95 {new_shape = array<i64: 1, 1024, 256>} : (tensor<1024x256xf32>) -> tensor<1x1024x256xf32>
  %97 = tosa.reshape %76 {new_shape = array<i64: 1, 1024, 12, 128>} : (tensor<1x1024x1536xf32>) -> tensor<1x1024x12x128xf32>
  %98 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %99 = tosa.transpose %97, %98 : (tensor<1x1024x12x128xf32>, tensor<4xi32>) -> tensor<1x12x1024x128xf32>
  %100 = tosa.reshape %86 {new_shape = array<i64: 1, 1024, 2, 128>} : (tensor<1x1024x256xf32>) -> tensor<1x1024x2x128xf32>
  %101 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %102 = tosa.transpose %100, %101 : (tensor<1x1024x2x128xf32>, tensor<4xi32>) -> tensor<1x2x1024x128xf32>
  %103 = tosa.reshape %96 {new_shape = array<i64: 1, 1024, 2, 128>} : (tensor<1x1024x256xf32>) -> tensor<1x1024x2x128xf32>
  %104 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %105 = tosa.transpose %103, %104 : (tensor<1x1024x2x128xf32>, tensor<4xi32>) -> tensor<1x2x1024x128xf32>
  %106 = tosa.reshape %arg5 {new_shape = array<i64: 1, 1, 1024, 128>} : (tensor<1x1024x128xf32>) -> tensor<1x1x1024x128xf32>
  %107 = tosa.reshape %arg5 {new_shape = array<i64: 1, 1, 1024, 128>} : (tensor<1x1024x128xf32>) -> tensor<1x1x1024x128xf32>
  %108 = tosa.mul %99, %106 : (tensor<1x12x1024x128xf32>, tensor<1x1x1024x128xf32>) -> tensor<1x12x1024x128xf32>
  %extracted_slice_27 = tensor.extract_slice %99[0, 0, 0, 0] [1, 12, 1024, 64] [1, 1, 1, 1] : tensor<1x12x1024x128xf32> to tensor<1x12x1024x64xf32>
  %extracted_slice_28 = tensor.extract_slice %99[0, 0, 0, 64] [1, 12, 1024, 64] [1, 1, 1, 1] : tensor<1x12x1024x128xf32> to tensor<1x12x1024x64xf32>
  %109 = tensor.empty() : tensor<1x12x1024x64xf32>
  %110 = linalg.negf ins(%extracted_slice_28 : tensor<1x12x1024x64xf32>) outs(%109 : tensor<1x12x1024x64xf32>) -> tensor<1x12x1024x64xf32>
  %111 = tensor.empty() : tensor<1x12x1024x128xf32>
  %inserted_slice_29 = tensor.insert_slice %110 into %111[0, 0, 0, 0] [1, 12, 1024, 64] [1, 1, 1, 1] : tensor<1x12x1024x64xf32> into tensor<1x12x1024x128xf32>
  %inserted_slice_30 = tensor.insert_slice %extracted_slice_27 into %inserted_slice_29[0, 0, 0, 64] [1, 12, 1024, 64] [1, 1, 1, 1] : tensor<1x12x1024x64xf32> into tensor<1x12x1024x128xf32>
  %112 = tosa.mul %inserted_slice_30, %107 : (tensor<1x12x1024x128xf32>, tensor<1x1x1024x128xf32>) -> tensor<1x12x1024x128xf32>
  %113 = tosa.add %108, %112 : (tensor<1x12x1024x128xf32>, tensor<1x12x1024x128xf32>) -> tensor<1x12x1024x128xf32>
  %114 = tosa.mul %102, %106 : (tensor<1x2x1024x128xf32>, tensor<1x1x1024x128xf32>) -> tensor<1x2x1024x128xf32>
  %extracted_slice_31 = tensor.extract_slice %102[0, 0, 0, 0] [1, 2, 1024, 64] [1, 1, 1, 1] : tensor<1x2x1024x128xf32> to tensor<1x2x1024x64xf32>
  %extracted_slice_32 = tensor.extract_slice %102[0, 0, 0, 64] [1, 2, 1024, 64] [1, 1, 1, 1] : tensor<1x2x1024x128xf32> to tensor<1x2x1024x64xf32>
  %115 = tensor.empty() : tensor<1x2x1024x64xf32>
  %116 = linalg.negf ins(%extracted_slice_32 : tensor<1x2x1024x64xf32>) outs(%115 : tensor<1x2x1024x64xf32>) -> tensor<1x2x1024x64xf32>
  %117 = tensor.empty() : tensor<1x2x1024x128xf32>
  %inserted_slice_33 = tensor.insert_slice %116 into %117[0, 0, 0, 0] [1, 2, 1024, 64] [1, 1, 1, 1] : tensor<1x2x1024x64xf32> into tensor<1x2x1024x128xf32>
  %inserted_slice_34 = tensor.insert_slice %extracted_slice_31 into %inserted_slice_33[0, 0, 0, 64] [1, 2, 1024, 64] [1, 1, 1, 1] : tensor<1x2x1024x64xf32> into tensor<1x2x1024x128xf32>
  %118 = tosa.mul %inserted_slice_34, %107 : (tensor<1x2x1024x128xf32>, tensor<1x1x1024x128xf32>) -> tensor<1x2x1024x128xf32>
  %119 = tosa.add %114, %118 : (tensor<1x2x1024x128xf32>, tensor<1x2x1024x128xf32>) -> tensor<1x2x1024x128xf32>
  %extracted_slice_35 = tensor.extract_slice %119[0, 0, 0, 0] [1, 2, 1024, 128] [1, 1, 1, 1] : tensor<1x2x1024x128xf32> to tensor<1x2x1024x128xf32>
  %extracted_slice_36 = tensor.extract_slice %extracted_slice_35[0, 0, 0, 0] [1, 2, 1024, 128] [1, 1, 1, 1] : tensor<1x2x1024x128xf32> to tensor<1x2x1024x128xf32>
  %120 = tosa.reshape %extracted_slice_36 {new_shape = array<i64: 1, 2, 1, 1024, 128>} : (tensor<1x2x1024x128xf32>) -> tensor<1x2x1x1024x128xf32>
  %extracted_slice_37 = tensor.extract_slice %120[0, 0, 0, 0, 0] [1, 2, 1, 1024, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x1024x128xf32> to tensor<1x2x1x1024x128xf32>
  %extracted_slice_38 = tensor.extract_slice %extracted_slice_37[0, 0, 0, 0, 0] [1, 2, 1, 1024, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x1024x128xf32> to tensor<1x2x1x1024x128xf32>
  %121 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x1024x128xf32>}> : () -> tensor<1x2x6x1024x128xf32>
  %122 = tosa.add %extracted_slice_38, %121 : (tensor<1x2x1x1024x128xf32>, tensor<1x2x6x1024x128xf32>) -> tensor<1x2x6x1024x128xf32>
  %123 = tosa.identity %122 : (tensor<1x2x6x1024x128xf32>) -> tensor<1x2x6x1024x128xf32>
  %124 = tosa.reshape %123 {new_shape = array<i64: 1, 12, 1024, 128>} : (tensor<1x2x6x1024x128xf32>) -> tensor<1x12x1024x128xf32>
  %extracted_slice_39 = tensor.extract_slice %105[0, 0, 0, 0] [1, 2, 1024, 128] [1, 1, 1, 1] : tensor<1x2x1024x128xf32> to tensor<1x2x1024x128xf32>
  %extracted_slice_40 = tensor.extract_slice %extracted_slice_39[0, 0, 0, 0] [1, 2, 1024, 128] [1, 1, 1, 1] : tensor<1x2x1024x128xf32> to tensor<1x2x1024x128xf32>
  %125 = tosa.reshape %extracted_slice_40 {new_shape = array<i64: 1, 2, 1, 1024, 128>} : (tensor<1x2x1024x128xf32>) -> tensor<1x2x1x1024x128xf32>
  %extracted_slice_41 = tensor.extract_slice %125[0, 0, 0, 0, 0] [1, 2, 1, 1024, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x1024x128xf32> to tensor<1x2x1x1024x128xf32>
  %extracted_slice_42 = tensor.extract_slice %extracted_slice_41[0, 0, 0, 0, 0] [1, 2, 1, 1024, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x1024x128xf32> to tensor<1x2x1x1024x128xf32>
  %126 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x1024x128xf32>}> : () -> tensor<1x2x6x1024x128xf32>
  %127 = tosa.add %extracted_slice_42, %126 : (tensor<1x2x1x1024x128xf32>, tensor<1x2x6x1024x128xf32>) -> tensor<1x2x6x1024x128xf32>
  %128 = tosa.identity %127 : (tensor<1x2x6x1024x128xf32>) -> tensor<1x2x6x1024x128xf32>
  %129 = tosa.reshape %128 {new_shape = array<i64: 1, 12, 1024, 128>} : (tensor<1x2x6x1024x128xf32>) -> tensor<1x12x1024x128xf32>

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  // Print timings.
  vector.print %time : f64
  // CHECK: {{[0-9]+\.[0-9]+}}

  return %113, %124, %129 : tensor<1x12x1024x128xf32>, tensor<1x12x1024x128xf32>, tensor<1x12x1024x128xf32>
}

func.func @main() {

  %cst_2 = arith.constant 2.0 : f32
  %empty_0 = tensor.empty() : tensor<1x1024x1536xf32>
  %c0 = linalg.fill ins(%cst_2 : f32) outs(%empty_0 : tensor<1x1024x1536xf32>) -> tensor<1x1024x1536xf32>

  %cst_3 = arith.constant 3.0 : f32
  %empty_1 = tensor.empty() : tensor<1536xf32>
  %c1 = linalg.fill ins(%cst_3 : f32) outs(%empty_1 : tensor<1536xf32>) -> tensor<1536xf32>

  %cst_4 = arith.constant 4.0 : f32
  %empty_2 = tensor.empty() : tensor<256xf32>
  %c2 = linalg.fill ins(%cst_4 : f32) outs(%empty_2 : tensor<256xf32>) -> tensor<256xf32>

  %cst_5 = arith.constant 5.0 : f32
  %empty_3 = tensor.empty() : tensor<1536x1536xf32>
  %c3 = linalg.fill ins(%cst_5 : f32) outs(%empty_3 : tensor<1536x1536xf32>) -> tensor<1536x1536xf32>

  %cst_6 = arith.constant 6.0 : f32
  %empty_4 = tensor.empty() : tensor<256x1536xf32>
  %c4 = linalg.fill ins(%cst_6 : f32) outs(%empty_4 : tensor<256x1536xf32>) -> tensor<256x1536xf32>

  %cst_7 = arith.constant 7.0 : f32
  %empty_5 = tensor.empty() : tensor<1x1024x128xf32>
  %c5 = linalg.fill ins(%cst_7 : f32) outs(%empty_5 : tensor<1x1024x128xf32>) -> tensor<1x1024x128xf32>

  %res:3 = call @kernel(%c0, %c1, %c2, %c3, %c4, %c5) : (tensor<1x1024x1536xf32>, tensor<1536xf32>, tensor<256xf32>, tensor<1536x1536xf32>, tensor<256x1536xf32>, tensor<1x1024x128xf32>) -> (tensor<1x12x1024x128xf32>, tensor<1x12x1024x128xf32>, tensor<1x12x1024x128xf32>)

  %tensor_unranked_q = tensor.cast %res#0 : tensor<1x12x1024x128xf32> to tensor<*xf32>
  %tensor_unranked_k = tensor.cast %res#1 : tensor<1x12x1024x128xf32> to tensor<*xf32>
  %tensor_unranked_v = tensor.cast %res#2 : tensor<1x12x1024x128xf32> to tensor<*xf32>
  // Print results.
  // call @printMemrefF32(%tensor_unranked_q) : (tensor<*xf32>) -> ()
  // call @printMemrefF32(%tensor_unranked_k) : (tensor<*xf32>) -> ()
  // call @printMemrefF32(%tensor_unranked_v) : (tensor<*xf32>) -> ()

  return
}
