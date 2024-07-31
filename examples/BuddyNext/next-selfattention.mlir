// RUN: buddy-opt %s \
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

#map = affine_map<(d0, d1, d2) -> (d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map6 = affine_map<(d0, d1, d2) -> (d0, 0, d1, d2)>
#map7 = affine_map<(d0, d1) -> (0, d0, d1)>
func.func private @rtclock() -> f64

func.func @kernel_self_attention(%arg0 : tensor<1x1x4096xf32>, %arg1 : tensor<1x40x4096xf32>, %arg2 : tensor<40xi64>, %arg3 : tensor<4096x4096xf32>, %arg4 : tensor<4096x4096xf32>, %arg5 : tensor<4096x4096xf32>, %arg6 : tensor<1x1x2048x128xf32>, %arg7 : tensor<1x1x2048x128xf32>, %arg8 : tensor<4096x4096xf32>, %arg9 : tensor<1x1x40x40xf32>) {
  %t_start = call @rtclock() : () -> f64

  // 计算 Query、Key 和 Value 矩阵
  %41 = tosa.mul %arg0, %arg1 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>

  %42 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %43 = tosa.transpose %arg3, %42 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
  %44 = tosa.reshape %41 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
  %cst_6 = arith.constant dense<0.0> : tensor<40x4096xf32>
  %45 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%44, %43 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_6 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
  %46 = tosa.reshape %45 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>

  %47 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %48 = tosa.transpose %arg4, %47 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
  %49 = tosa.reshape %41 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
  %cst_7 = arith.constant dense<0.0> : tensor<40x4096xf32>
  %50 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%49, %48 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_7 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
  %51 = tosa.reshape %50 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>

  %52 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %53 = tosa.transpose %arg5, %52 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
  %54 = tosa.reshape %41 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
  %cst_8 = arith.constant dense<0.0> : tensor<40x4096xf32>
  %55 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%54, %53 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_8 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
  %56 = tosa.reshape %55 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>

  // 对 Q、K 向量进行 RoPE (旋转式位置编码)
  %57 = tosa.reshape %46 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
  %58 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %59 = tosa.transpose %57, %58 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>

  %60 = tosa.reshape %51 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
  %61 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %62 = tosa.transpose %60, %61 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>

  %63 = tosa.reshape %56 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
  %64 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %65 = tosa.transpose %63, %64 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>

  // 计算 Softmax(Q,K) 以及Self-Attention的输出
  %extracted_slice_9 = tensor.extract_slice %arg6[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
  %extracted_slice_10 = tensor.extract_slice %extracted_slice_9[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
  %extracted_slice_11 = tensor.extract_slice %extracted_slice_10[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
  %extracted_slice_12 = tensor.extract_slice %arg7[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
  %extracted_slice_13 = tensor.extract_slice %extracted_slice_12[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
  %extracted_slice_14 = tensor.extract_slice %extracted_slice_13[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
  %66 = tensor.empty() : tensor<1x40x128xf32>
  %67 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_11 : tensor<1x1x40x128xf32>) outs(%66 : tensor<1x40x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x40x128xf32>
  %68 = tensor.empty() : tensor<40x128xf32>
  %69 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%67 : tensor<1x40x128xf32>) outs(%68 : tensor<40x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<40x128xf32>
  %70 = tensor.empty() : tensor<1x40x128xf32>
  %71 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_14 : tensor<1x1x40x128xf32>) outs(%70 : tensor<1x40x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x40x128xf32>
  %72 = tensor.empty() : tensor<40x128xf32>
  %73 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%71 : tensor<1x40x128xf32>) outs(%72 : tensor<40x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<40x128xf32>

  %74 = tensor.empty() : tensor<1x40x128xf32>
  %arg2_converted = tosa.reshape %arg2 {new_shape = array<i64: 40>} : (tensor<40xi64>) -> tensor<1x40xi64>
  %75 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg2_converted : tensor<1x40xi64>) outs(%74 : tensor<1x40x128xf32>) {
  ^bb0(%in: i64, %out: f32):
    %4175 = arith.index_cast %in : i64 to index
    %4176 = linalg.index 1 : index 
    %extracted = tensor.extract %69[%4175, %4176] : tensor<40x128xf32>
    linalg.yield %extracted : f32
  } -> tensor<1x40x128xf32>
  %76 = tosa.reshape %75 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>

  %77 = tensor.empty() : tensor<1x40x128xf32>
  %78 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg2_converted : tensor<1x40xi64>) outs(%77 : tensor<1x40x128xf32>) {
  ^bb0(%in: i64, %out: f32):
    %4175 = arith.index_cast %in : i64 to index
    %4176 = linalg.index 1 : index 
    %extracted = tensor.extract %73[%4175, %4176] : tensor<40x128xf32>
    linalg.yield %extracted : f32
  } -> tensor<1x40x128xf32>
  %79 = tosa.reshape %78 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>

  %80 = tosa.mul %59, %76 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
  %extracted_slice_15 = tensor.extract_slice %59[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
  %extracted_slice_16 = tensor.extract_slice %59[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
  %81 = tosa.negate %extracted_slice_16 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
  %82 = tensor.empty() : tensor<1x32x40x128xf32>
  %inserted_slice = tensor.insert_slice %81 into %82[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
  %inserted_slice_17 = tensor.insert_slice %extracted_slice_15 into %inserted_slice[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
  %83 = tosa.mul %inserted_slice_17, %79 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
  %84 = tosa.add %80, %83 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
  %85 = tosa.mul %62, %76 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
  %extracted_slice_18 = tensor.extract_slice %62[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
  %extracted_slice_19 = tensor.extract_slice %62[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
  %86 = tosa.negate %extracted_slice_19 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
  %87 = tensor.empty() : tensor<1x32x40x128xf32>
  %inserted_slice_20 = tensor.insert_slice %86 into %87[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
  %inserted_slice_21 = tensor.insert_slice %extracted_slice_18 into %inserted_slice_20[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>

  // 计算 Softmax(QK/sqrt(d_k))
  %88 = tosa.mul %inserted_slice_21, %79 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
  %89 = tosa.add %85, %88 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
  %90 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %91 = tosa.transpose %89, %90 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
  %92 = "tosa.const"() <{value = dense<0.0> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
  %93 = tosa.add %84, %92 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
  %94 = tosa.reshape %93 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
  %95 = "tosa.const"() <{value = dense<0.0> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
  %96 = tosa.add %91, %95 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
  %97 = tosa.reshape %96 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
  %98 = tosa.matmul %94, %97 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
  %99 = tosa.reshape %98 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
  %100 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
  %101 = tosa.reciprocal %100 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
  %102 = tosa.mul %99, %101 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
  %103 = tosa.add %102, %arg9 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
  %104 = tosa.reduce_max %103 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
  %105 = tosa.sub %103, %104 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
  %106 = tosa.exp %105 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
  %107 = tosa.reduce_sum %106 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
  %108 = tosa.reciprocal %107 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
  %109 = tosa.mul %106, %108 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>

  // 计算Self-Attention的输出
  %110 = "tosa.const"() <{value = dense<0.0> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
  %111 = tosa.add %109, %110 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
  %112 = tosa.reshape %111 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
  %113 = "tosa.const"() <{value = dense<0.0> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
  %114 = tosa.add %65, %113 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
  %115 = tosa.reshape %114 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
  %116 = tosa.matmul %112, %115 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>

  %117 = tosa.reshape %116 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
  %118 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %119 = tosa.transpose %117, %118 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
  %120 = tosa.identity %119 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
  %121 = tosa.reshape %120 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>

  %122 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %123 = tosa.transpose %arg8, %122 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
  %124 = tosa.reshape %121 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
  %cst_22 = arith.constant dense<0.0> : tensor<40x4096xf32>
  %125 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%124, %123 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_22 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
  %126 = tosa.reshape %125 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
  %127 = tosa.add %arg1, %126 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  %tensor_unranked = tensor.cast %127 : tensor<1x40x4096xf32> to tensor<*xf32>

  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
  vector.print %time : f64

  return
}

func.func @main() {
  %input_tensor_0 = arith.constant dense<3.0> : tensor<1x1x4096xf32>
  %input_tensor_1 = arith.constant dense<1.0> : tensor<1x40x4096xf32>
  %input_tensor_2 = arith.constant dense<2> : tensor<40xi64>
  %input_tensor_3 = arith.constant dense<1.0> : tensor<4096x4096xf32>
  %input_tensor_4 = arith.constant dense<1.0> : tensor<4096x4096xf32>
  %input_tensor_5 = arith.constant dense<1.0> : tensor<4096x4096xf32>
  %input_tensor_6 = arith.constant dense<1.0> : tensor<1x1x2048x128xf32>
  %input_tensor_7 = arith.constant dense<1.0> : tensor<1x1x2048x128xf32>
  %input_tensor_8 = arith.constant dense<2.0> : tensor<4096x4096xf32>
  %input_tensor_9 = arith.constant dense<0.0> : tensor<1x1x40x40xf32>

  call @kernel_self_attention(%input_tensor_0, %input_tensor_1, %input_tensor_2, %input_tensor_3, %input_tensor_4, %input_tensor_5, %input_tensor_6, %input_tensor_7, %input_tensor_8, %input_tensor_9) : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>, tensor<40xi64>, tensor<4096x4096xf32>, tensor<4096x4096xf32>, tensor<4096x4096xf32>, tensor<1x1x2048x128xf32>, tensor<1x1x2048x128xf32>, tensor<4096x4096xf32>, tensor<1x1x40x40xf32>) -> ()

  return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)