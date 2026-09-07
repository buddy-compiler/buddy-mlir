// RUN: buddy-opt %s --fuse-qwen-silu-mul-quantize --cse | FileCheck %s --check-prefix=FUSE
// RUN: buddy-opt %s --fuse-qwen-silu-mul-quantize --cse -o %t.fused
// RUN: buddy-opt %t.fused -pass-pipeline="builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" -o %t.linalg
// RUN: buddy-opt %t.linalg --eliminate-empty-tensors --empty-tensor-to-alloc-tensor --convert-elementwise-to-linalg --one-shot-bufferize="bufferize-function-boundaries" --lower-qwen-w8a8-to-boscame="profile-phases quantize-one-ahead=true" | FileCheck %s --check-prefix=LOWER

// The Qwen Down input is spelled exactly as emitted by the frontend.  The
// fusion keeps sigmoid and gate*sigmoid on their established lowering and
// combines only the final SiLU*Up product with amax.  The full-size product
// intermediate therefore becomes dead before bufferization.
func.func @qwen_down_t1(
    %gate: tensor<1x3072xf32>, %up: tensor<1x3072xf32>)
    -> (tensor<1x3072xi8>, tensor<1x6xf32>) {
  %shape3 = tosa.const_shape {values = dense<[1, 1, 3072]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %shape2 = tosa.const_shape {values = dense<[1, 3072]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %shift = arith.constant dense<0> : tensor<1xi8>
  %gate3 = tosa.reshape %gate, %shape3 : (tensor<1x3072xf32>, !tosa.shape<3>) -> tensor<1x1x3072xf32>
  %up3 = tosa.reshape %up, %shape3 : (tensor<1x3072xf32>, !tosa.shape<3>) -> tensor<1x1x3072xf32>
  %sigmoid = tosa.sigmoid %gate3 : (tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
  %silu = tosa.mul %gate3, %sigmoid, %shift : (tensor<1x1x3072xf32>, tensor<1x1x3072xf32>, tensor<1xi8>) -> tensor<1x1x3072xf32>
  %product = tosa.mul %silu, %up3, %shift : (tensor<1x1x3072xf32>, tensor<1x1x3072xf32>, tensor<1xi8>) -> tensor<1x1x3072xf32>
  %flat = tosa.reshape %product, %shape2 : (tensor<1x1x3072xf32>, !tosa.shape<2>) -> tensor<1x3072xf32>
  %q = tensor.empty() : tensor<1x3072xi8>
  %s = tensor.empty() : tensor<1x6xf32>
  %result:2 = "bosc_ame.quantize_per_group"(%flat, %q, %s) <{group_size = 512 : i64}> : (tensor<1x3072xf32>, tensor<1x3072xi8>, tensor<1x6xf32>) -> (tensor<1x3072xi8>, tensor<1x6xf32>)
  return %result#0, %result#1 : tensor<1x3072xi8>, tensor<1x6xf32>
}

// FUSE-LABEL: func.func @qwen_down_t1
// FUSE: %[[GATE3:.*]] = tosa.reshape %arg0
// FUSE: %[[SIGMOID:.*]] = tosa.sigmoid %[[GATE3]]
// FUSE: %[[SILU3:.*]] = tosa.mul %[[GATE3]], %[[SIGMOID]]
// FUSE: %[[SILU2:.*]] = tosa.reshape %[[SILU3]]
// FUSE: %[[RESULT:.*]]:2 = "bosc_ame.silu_mul_quantize_per_group"(%[[SILU2]], %arg1,
// FUSE-SAME: group_size = 512
// FUSE-NOT: "bosc_ame.quantize_per_group"
// FUSE-NOT: tosa.mul
// FUSE: return %[[RESULT]]#0, %[[RESULT]]#1

// LOWER-LABEL: func.func @qwen_down_t1
// LOWER: math.exp
// LOWER: arith.mulf
// LOWER: memref.get_global @__buddy_qwen_w8a8_silu_quant_scratch_f32
// LOWER: call @buddyTraceCycleStartPath
// LOWER: scf.for
// LOWER: scf.for
// LOWER: scf.for {{.*}} step %{{.*}} iter_args
// LOWER: arith.mulf
// LOWER: memref.store {{.*}}, %{{.*}}[%{{.*}}] : memref<1024xf32>
// LOWER: call @buddy_w8a8_quantize_write_one_ahead
// LOWER-NOT: bosc_ame.silu_mul_quantize_per_group

func.func @direct_t22(
    %silu: memref<22x3072xf32>, %up: memref<22x3072xf32>,
    %q: memref<22x3072xi8>, %scales: memref<22x6xf32>) {
  "bosc_ame.silu_mul_quantize_per_group"(
      %silu, %up, %q, %scales) {group_size = 512 : i64} :
      (memref<22x3072xf32>, memref<22x3072xf32>,
       memref<22x3072xi8>, memref<22x6xf32>) -> ()
  return
}

// LOWER-LABEL: func.func @direct_t22
// LOWER-DAG: %[[C22:.*]] = arith.constant 22 : index
// LOWER-DAG: %[[C6:.*]] = arith.constant 6 : index
// LOWER-DAG: %{{.*}} = arith.constant 512 : index
// LOWER: scf.for {{.*}} to %[[C22]] step
// LOWER: scf.for {{.*}} to %[[C6]] step
// LOWER: scf.for {{.*}} to %{{c512.*}} step
// LOWER-NOT: math.exp
// LOWER: arith.mulf
// LOWER: memref.store {{.*}}, %{{.*}}[{{.*}}] : memref<1024xf32>
// LOWER: call @buddy_w8a8_quantize_write_one_ahead
// LOWER-NOT: bosc_ame.silu_mul_quantize_per_group

// A strided int8 destination cannot use the contiguous one-ahead helper.  It
// must retain the exact inline quantize-write loop and honor the memref
// descriptor's inner stride.
func.func @direct_strided_q_fallback(
    %silu: memref<1x512xf32>, %up: memref<1x512xf32>,
    %q: memref<1x512xi8, strided<[1024, 2], offset: ?>>,
    %scales: memref<1x1xf32>) {
  "bosc_ame.silu_mul_quantize_per_group"(
      %silu, %up, %q, %scales) {group_size = 512 : i64} :
      (memref<1x512xf32>, memref<1x512xf32>,
       memref<1x512xi8, strided<[1024, 2], offset: ?>>,
       memref<1x1xf32>) -> ()
  return
}

// LOWER-LABEL: func.func @direct_strided_q_fallback
// LOWER-NOT: call @buddy_w8a8_quantize_write_one_ahead
// LOWER-NOT: math.exp
// LOWER: arith.mulf
// LOWER: arith.divf
// LOWER: arith.fptosi
// LOWER: memref.store {{.*}}, %arg2[{{.*}}]
// LOWER-NOT: call @buddy_w8a8_quantize_write_one_ahead
// LOWER: return

// A non-Down group size is a reliable fallback and must retain the original
// elementwise graph plus standalone quantizer.
func.func @non_qwen_group_fallback(
    %gate: tensor<1x256xf32>, %up: tensor<1x256xf32>)
    -> (tensor<1x256xi8>, tensor<1x1xf32>) {
  %shape3 = tosa.const_shape {values = dense<[1, 1, 256]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %shape2 = tosa.const_shape {values = dense<[1, 256]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %shift = arith.constant dense<0> : tensor<1xi8>
  %gate3 = tosa.reshape %gate, %shape3 : (tensor<1x256xf32>, !tosa.shape<3>) -> tensor<1x1x256xf32>
  %up3 = tosa.reshape %up, %shape3 : (tensor<1x256xf32>, !tosa.shape<3>) -> tensor<1x1x256xf32>
  %sigmoid = tosa.sigmoid %gate3 : (tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
  %silu = tosa.mul %gate3, %sigmoid, %shift : (tensor<1x1x256xf32>, tensor<1x1x256xf32>, tensor<1xi8>) -> tensor<1x1x256xf32>
  %product = tosa.mul %silu, %up3, %shift : (tensor<1x1x256xf32>, tensor<1x1x256xf32>, tensor<1xi8>) -> tensor<1x1x256xf32>
  %flat = tosa.reshape %product, %shape2 : (tensor<1x1x256xf32>, !tosa.shape<2>) -> tensor<1x256xf32>
  %q = tensor.empty() : tensor<1x256xi8>
  %s = tensor.empty() : tensor<1x1xf32>
  %result:2 = "bosc_ame.quantize_per_group"(%flat, %q, %s) <{group_size = 256 : i64}> : (tensor<1x256xf32>, tensor<1x256xi8>, tensor<1x1xf32>) -> (tensor<1x256xi8>, tensor<1x1xf32>)
  return %result#0, %result#1 : tensor<1x256xi8>, tensor<1x1xf32>
}

// FUSE-LABEL: func.func @non_qwen_group_fallback
// FUSE: tosa.sigmoid
// FUSE: tosa.mul
// FUSE: tosa.mul
// FUSE: "bosc_ame.quantize_per_group"
// FUSE-NOT: bosc_ame.silu_mul_quantize_per_group
