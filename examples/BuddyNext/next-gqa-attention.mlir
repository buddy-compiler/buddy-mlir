// RUN: buddy-opt %s \
// RUN:     -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" \
// RUN: | buddy-opt \
// RUN:     -arith-expand \
// RUN:     -eliminate-empty-tensors \
// RUN:     -empty-tensor-to-alloc-tensor \
// RUN:     -convert-elementwise-to-linalg \
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
// RUN:     -convert-math-to-libm  \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s



#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func private @printMemrefF32(%ptr : tensor<*xf32>)
func.func private @rtclock() -> f64
func.func @kernel(%q : tensor<1x12x1x128xf32>, %k_cache : tensor<1x2x1024x128xf32>, %v_cache : tensor<1x2x1024x128xf32>, %mask : tensor<1x1x1x1024xf32>) -> tensor<1x12x1x128xf32> {

  // ===========GQA cache reshape =====
  %136 = tosa.const_shape  {values = dense<[1, 2, 1, 1024, 128]> : tensor<5xindex>} : () -> !tosa.shape<5>
  %137 = tosa.reshape %k_cache, %136 : (tensor<1x2x1024x128xf32>, !tosa.shape<5>) -> tensor<1x2x1x1024x128xf32>
  %138 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1x2x6x1024x128xf32>}> : () -> tensor<1x2x6x1024x128xf32>
  %139 = tosa.add %137, %138 : (tensor<1x2x1x1024x128xf32>, tensor<1x2x6x1024x128xf32>) -> tensor<1x2x6x1024x128xf32>
  %140 = tosa.const_shape  {values = dense<[1, 12, 1024, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %141 = tosa.reshape %139, %140 : (tensor<1x2x6x1024x128xf32>, !tosa.shape<4>) -> tensor<1x12x1024x128xf32>

  %142 = tosa.const_shape  {values = dense<[1, 2, 1, 1024, 128]> : tensor<5xindex>} : () -> !tosa.shape<5>
  %143 = tosa.reshape %v_cache, %142 : (tensor<1x2x1024x128xf32>, !tosa.shape<5>) -> tensor<1x2x1x1024x128xf32>
  %144 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1x2x6x1024x128xf32>}> : () -> tensor<1x2x6x1024x128xf32>
  %145 = tosa.add %143, %144 : (tensor<1x2x1x1024x128xf32>, tensor<1x2x6x1024x128xf32>) -> tensor<1x2x6x1024x128xf32>
  %146 = tosa.const_shape  {values = dense<[1, 12, 1024, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %147 = tosa.reshape %145, %146 : (tensor<1x2x6x1024x128xf32>, !tosa.shape<4>) -> tensor<1x12x1024x128xf32>

  // ==============mask======
  %cst_21 = arith.constant 0.000000e+00 : f32
  %splat_22 = tensor.splat %cst_21 : tensor<1x1024xf32>
  %152 = tosa.const_shape  {values = dense<[1, 1024]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %153 = tosa.reshape %mask, %152 : (tensor<1x1x1x1024xf32>, !tosa.shape<2>) -> tensor<1x1024xf32>
  %154 = tosa.add %splat_22, %153 : (tensor<1x1024xf32>, tensor<1x1024xf32>) -> tensor<1x1024xf32>
  %155 = tosa.transpose %141 {perms = array<i32: 0, 1, 3, 2>} : (tensor<1x12x1024x128xf32>) -> tensor<1x12x128x1024xf32>
  %156 = tosa.const_shape  {values = dense<[12, 1, 128]> : tensor<3xindex>} : () -> !tosa.shape<3>

  // =============q reshape============
  %157 = tosa.reshape %q, %156 : (tensor<1x12x1x128xf32>, !tosa.shape<3>) -> tensor<12x1x128xf32>
  %158 = tosa.transpose %155 {perms = array<i32: 0, 1, 3, 2>} : (tensor<1x12x128x1024xf32>) -> tensor<1x12x1024x128xf32>
  %159 = tosa.const_shape  {values = dense<[12, 1024, 128]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %160 = tosa.reshape %158, %159 : (tensor<1x12x1024x128xf32>, !tosa.shape<3>) -> tensor<12x1024x128xf32>

    // ===== Attention QK^T =====

  %cst_23 = arith.constant dense<0.000000e+00> : tensor<12x1x1024xf32>
  %161 = linalg.batch_matmul_transpose_b ins(%157, %160 : tensor<12x1x128xf32>, tensor<12x1024x128xf32>) outs(%cst_23 : tensor<12x1x1024xf32>) -> tensor<12x1x1024xf32>
  %cst_24 = arith.constant 0.0883883461 : f32
  %splat_25 = tensor.splat %cst_24 : tensor<12x1x1024xf32>
  %162 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %163 = tosa.mul %161, %splat_25, %162 : (tensor<12x1x1024xf32>, tensor<12x1x1024xf32>, tensor<1xi8>) -> tensor<12x1x1024xf32>
  %164 = tosa.const_shape  {values = dense<[1, 1, 1024]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %165 = tosa.reshape %154, %164 : (tensor<1x1024xf32>, !tosa.shape<3>) -> tensor<1x1x1024xf32>
  %166 = tosa.add %163, %165 : (tensor<12x1x1024xf32>, tensor<1x1x1024xf32>) -> tensor<12x1x1024xf32>

  // ===== Attention Softmax =====

  %167 = tosa.reduce_max %166 {axis = 2 : i32} : (tensor<12x1x1024xf32>) -> tensor<12x1x1xf32>
  %168 = tosa.sub %166, %167 : (tensor<12x1x1024xf32>, tensor<12x1x1xf32>) -> tensor<12x1x1024xf32>
  %169 = math.exp %168 : tensor<12x1x1024xf32>
  %170 = tosa.reduce_sum %169 {axis = 2 : i32} : (tensor<12x1x1024xf32>) -> tensor<12x1x1xf32>
  %171 = tosa.log %170 : (tensor<12x1x1xf32>) -> tensor<12x1x1xf32>
  %172 = tosa.add %167, %171 : (tensor<12x1x1xf32>, tensor<12x1x1xf32>) -> tensor<12x1x1xf32>
  %173 = tosa.sub %166, %172 : (tensor<12x1x1024xf32>, tensor<12x1x1xf32>) -> tensor<12x1x1024xf32>
  %174 = math.exp %173 : tensor<12x1x1024xf32>

  // ===== Attention * V =====

  %175 = tosa.const_shape  {values = dense<[1, 12, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %176 = tosa.reshape %172, %175 : (tensor<12x1x1xf32>, !tosa.shape<3>) -> tensor<1x12x1xf32>
  %177 = tosa.const_shape  {values = dense<[12, 1024, 128]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %178 = tosa.reshape %147, %177 : (tensor<1x12x1024x128xf32>, !tosa.shape<3>) -> tensor<12x1024x128xf32>
  %179 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  %180 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  %181 = tosa.matmul %174, %178, %179, %180 : (tensor<12x1x1024xf32>, tensor<12x1024x128xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<12x1x128xf32>
  %182 = tosa.const_shape  {values = dense<[1, 12, 1, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %183 = tosa.reshape %181, %182 : (tensor<12x1x128xf32>, !tosa.shape<4>) -> tensor<1x12x1x128xf32>

  return %183 : tensor<1x12x1x128xf32>
}

func.func @main() {

  %Q = tensor.generate {
    ^bb0(%b: index, %h: index, %i: index, %k: index):
      %sum = arith.addi %b, %h : index
      %mix1 = arith.addi %sum, %i : index
      %mix2 = arith.addi %mix1, %k : index
      %c11 = arith.constant 11 : index
      %mod = arith.remui %mix2, %c11 : index
      %val = arith.index_cast %mod : index to i32
      %valf = arith.sitofp %val : i32 to f32
      tensor.yield %valf : f32
  } : tensor<1x12x1x128xf32>

  %K = tensor.generate {
    ^bb0(%b: index, %h: index, %k: index, %j: index):
      %sum = arith.addi %b, %h : index
      %mix1 = arith.addi %sum, %k : index
      %mix2 = arith.addi %mix1, %j : index
      %c17 = arith.constant 17 : index
      %mod = arith.remui %mix2, %c17 : index
      %val = arith.index_cast %mod : index to i32
      %valf = arith.sitofp %val : i32 to f32
      tensor.yield %valf : f32
  } : tensor<1x2x1024x128xf32>

  %V = tensor.generate {
    ^bb0(%b: index, %h: index, %i: index, %k: index):
      %sum = arith.addi %b, %h : index
      %mix1 = arith.addi %sum, %i : index
      %mix2 = arith.addi %mix1, %k : index
      %c13 = arith.constant 13 : index
      %mod = arith.remui %mix2, %c13 : index
      %val = arith.index_cast %mod : index to i32
      %valf = arith.sitofp %val : i32 to f32
      tensor.yield %valf : f32
  } : tensor<1x2x1024x128xf32>


    // Mask: only allow j <= i positions, simulate causal mask
    %zero = arith.constant 0.0 : f32
    %neg  = arith.constant -1.0E+9 : f32

    %mask = tensor.generate {
    ^bb0(%b: index, %h: index, %i: index, %j: index):
      %cond = arith.cmpi "slt", %i, %j : index
      %val = arith.select %cond, %neg, %zero : f32
      tensor.yield %val : f32
    } : tensor<1x1x1x1024xf32>

  %t_start = call @rtclock() : () -> f64

  %result_out = call @kernel(%Q, %K, %V, %mask) : (tensor<1x12x1x128xf32>, tensor<1x2x1024x128xf32>, tensor<1x2x1024x128xf32>, tensor<1x1x1x1024xf32>) -> tensor<1x12x1x128xf32>

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  vector.print %time : f64
  // CHECK: {{[0-9]+\.[0-9]+}}

  %tensor_unranked = tensor.cast %result_out : tensor<1x12x1x128xf32> to tensor<*xf32>
  // call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()

  return
}
