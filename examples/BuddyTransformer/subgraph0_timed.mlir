#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  // Declare timing functions
  func.func private @rtclock() -> f64
  func.func private @record_timing(!llvm.ptr, f64) -> ()
  
  // Operator name constants
  llvm.mlir.global private constant @op_name_input_layernorm("input_layernorm\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_q_proj("q_projection\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_k_proj("k_projection\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_v_proj("v_projection\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_attn_qk("attention_qk_matmul\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_attn_softmax("attention_softmax\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_attn_v("attention_v_matmul\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_o_proj("o_projection\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_post_attn_layernorm("post_attention_layernorm\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_ffn_gate("ffn_gate_projection\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_ffn_up("ffn_up_projection\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_ffn_down("ffn_down_projection\00") {addr_space = 0 : i32}
  
  func.func @subgraph0(%arg0: tensor<1x40x1536xf32>, %arg1: tensor<1x40x1536xf32>, %arg2: tensor<1536xf32>, %arg3: tensor<1536x1536xf32>, %arg4: tensor<256x1536xf32>, %arg5: tensor<256x1536xf32>, %arg6: tensor<1x40xi64>, %arg7: tensor<1536x1536xf32>, %arg8: tensor<1x40x1536xf32>, %arg9: tensor<1536xf32>, %arg10: tensor<8960x1536xf32>, %arg11: tensor<8960x1536xf32>, %arg12: tensor<1536x8960xf32>) -> tensor<1x40x1536xf32> {
    
    // ===== Input LayerNorm =====
    %t_start_0 = call @rtclock() : () -> f64
    
    %0 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32 = arith.constant 2 : i32
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8 : tensor<1x40x1536xf32>) outs(%0 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %111 = math.fpowi %in, %c2_i32 : f32, i32
      linalg.yield %111 : f32
    } -> tensor<1x40x1536xf32>
    %2 = tosa.reduce_sum %1 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %3 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4 = tosa.reciprocal %3 : (tensor<1xf32>) -> tensor<1xf32>
    %5 = tosa.reshape %4 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %6 = tosa.mul %5, %2 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %7 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %8 = tosa.add %6, %7 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %9 = tosa.rsqrt %8 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %10 = tosa.mul %arg8, %9 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %11 = tosa.reshape %arg2 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %12 = tosa.mul %11, %10 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    
    %t_end_0 = call @rtclock() : () -> f64
    %duration_0 = arith.subf %t_end_0, %t_start_0 : f64
    %name_ptr_0 = llvm.mlir.addressof @op_name_input_layernorm : !llvm.ptr
    call @record_timing(%name_ptr_0, %duration_0) : (!llvm.ptr, f64) -> ()
    
    // ===== Q Projection =====
    %t_start_1 = call @rtclock() : () -> f64
    
    %13 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %14 = tosa.transpose %arg3, %13 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %15 = tosa.reshape %12 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %16 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%15, %14 : tensor<40x1536xf32>, tensor<1536x1536xf32>) outs(%cst : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %17 = tosa.reshape %16 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    
    %t_end_1 = call @rtclock() : () -> f64
    %duration_1 = arith.subf %t_end_1, %t_start_1 : f64
    %name_ptr_1 = llvm.mlir.addressof @op_name_q_proj : !llvm.ptr
    call @record_timing(%name_ptr_1, %duration_1) : (!llvm.ptr, f64) -> ()
    
    // ===== K Projection =====
    %t_start_2 = call @rtclock() : () -> f64
    
    %18 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %19 = tosa.transpose %arg4, %18 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %20 = tosa.reshape %12 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<40x256xf32>
    %21 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%20, %19 : tensor<40x1536xf32>, tensor<1536x256xf32>) outs(%cst_0 : tensor<40x256xf32>) -> tensor<40x256xf32>
    %22 = tosa.reshape %21 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    
    %t_end_2 = call @rtclock() : () -> f64
    %duration_2 = arith.subf %t_end_2, %t_start_2 : f64
    %name_ptr_2 = llvm.mlir.addressof @op_name_k_proj : !llvm.ptr
    call @record_timing(%name_ptr_2, %duration_2) : (!llvm.ptr, f64) -> ()
    
    // ===== V Projection =====
    %t_start_3 = call @rtclock() : () -> f64
    
    %23 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %24 = tosa.transpose %arg5, %23 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %25 = tosa.reshape %12 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<40x256xf32>
    %26 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%25, %24 : tensor<40x1536xf32>, tensor<1536x256xf32>) outs(%cst_1 : tensor<40x256xf32>) -> tensor<40x256xf32>
    %27 = tosa.reshape %26 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    
    %t_end_3 = call @rtclock() : () -> f64
    %duration_3 = arith.subf %t_end_3, %t_start_3 : f64
    %name_ptr_3 = llvm.mlir.addressof @op_name_v_proj : !llvm.ptr
    call @record_timing(%name_ptr_3, %duration_3) : (!llvm.ptr, f64) -> ()
    
    // Reshape and transpose for attention
    %28 = tosa.reshape %17 {new_shape = array<i64: 1, 40, 12, 128>} : (tensor<1x40x1536xf32>) -> tensor<1x40x12x128xf32>
    %29 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %30 = tosa.transpose %28, %29 : (tensor<1x40x12x128xf32>, tensor<4xi32>) -> tensor<1x12x40x128xf32>
    %31 = tosa.reshape %22 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %32 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %33 = tosa.transpose %31, %32 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %34 = tosa.reshape %27 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %35 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %36 = tosa.transpose %34, %35 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %37 = tosa.reshape %33 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %38 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %39 = tosa.add %37, %38 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %40 = tosa.reshape %39 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %41 = tosa.reshape %36 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %42 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %43 = tosa.add %41, %42 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %44 = tosa.reshape %43 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %45 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %46 = tosa.transpose %40, %45 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x12x128x40xf32>
    
    // ===== Attention QK^T =====
    %t_start_4 = call @rtclock() : () -> f64
    
    %47 = tosa.reshape %30 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %48 = tosa.reshape %46 {new_shape = array<i64: 12, 128, 40>} : (tensor<1x12x128x40xf32>) -> tensor<12x128x40xf32>
    %49 = tosa.matmul %47, %48 : (tensor<12x40x128xf32>, tensor<12x128x40xf32>) -> tensor<12x40x40xf32>
    %50 = tosa.reshape %49 {new_shape = array<i64: 1, 12, 40, 40>} : (tensor<12x40x40xf32>) -> tensor<1x12x40x40xf32>
    %cst_2 = arith.constant dense<0.0883883461> : tensor<1xf32>
    %51 = tosa.reshape %cst_2 {new_shape = array<i64: 1, 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1x1xf32>
    %52 = tosa.mul %50, %51 : (tensor<1x12x40x40xf32>, tensor<1x1x1x1xf32>) -> tensor<1x12x40x40xf32>
    %53 = tosa.reshape %arg6 {new_shape = array<i64: 1, 1, 1, 40>} : (tensor<1x40xi64>) -> tensor<1x1x1x40xi64>
    %54 = tosa.cast %53 : (tensor<1x1x1x40xi64>) -> tensor<1x1x1x40xf32>
    %55 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1x1x40xf32>}> : () -> tensor<1x1x1x40xf32>
    %56 = tosa.sub %55, %54 : (tensor<1x1x1x40xf32>, tensor<1x1x1x40xf32>) -> tensor<1x1x1x40xf32>
    %cst_3 = arith.constant dense<-1.000000e+04> : tensor<1xf32>
    %57 = tosa.reshape %cst_3 {new_shape = array<i64: 1, 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1x1xf32>
    %58 = tosa.mul %56, %57 : (tensor<1x1x1x40xf32>, tensor<1x1x1x1xf32>) -> tensor<1x1x1x40xf32>
    %59 = tosa.add %52, %58 : (tensor<1x12x40x40xf32>, tensor<1x1x1x40xf32>) -> tensor<1x12x40x40xf32>
    
    %t_end_4 = call @rtclock() : () -> f64
    %duration_4 = arith.subf %t_end_4, %t_start_4 : f64
    %name_ptr_4 = llvm.mlir.addressof @op_name_attn_qk : !llvm.ptr
    call @record_timing(%name_ptr_4, %duration_4) : (!llvm.ptr, f64) -> ()
    
    // ===== Attention Softmax =====
    %t_start_5 = call @rtclock() : () -> f64
    
    %60 = tosa.reduce_max %59 {axis = 3 : i32} : (tensor<1x12x40x40xf32>) -> tensor<1x12x40x1xf32>
    %61 = tosa.sub %59, %60 : (tensor<1x12x40x40xf32>, tensor<1x12x40x1xf32>) -> tensor<1x12x40x40xf32>
    %62 = tosa.exp %61 : (tensor<1x12x40x40xf32>) -> tensor<1x12x40x40xf32>
    %63 = tosa.reduce_sum %62 {axis = 3 : i32} : (tensor<1x12x40x40xf32>) -> tensor<1x12x40x1xf32>
    %64 = tosa.reciprocal %63 : (tensor<1x12x40x1xf32>) -> tensor<1x12x40x1xf32>
    %65 = tosa.mul %62, %64 : (tensor<1x12x40x40xf32>, tensor<1x12x40x1xf32>) -> tensor<1x12x40x40xf32>
    
    %t_end_5 = call @rtclock() : () -> f64
    %duration_5 = arith.subf %t_end_5, %t_start_5 : f64
    %name_ptr_5 = llvm.mlir.addressof @op_name_attn_softmax : !llvm.ptr
    call @record_timing(%name_ptr_5, %duration_5) : (!llvm.ptr, f64) -> ()
    
    // ===== Attention * V =====
    %t_start_6 = call @rtclock() : () -> f64
    
    %66 = tosa.reshape %65 {new_shape = array<i64: 12, 40, 40>} : (tensor<1x12x40x40xf32>) -> tensor<12x40x40xf32>
    %67 = tosa.reshape %44 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %68 = tosa.matmul %66, %67 : (tensor<12x40x40xf32>, tensor<12x40x128xf32>) -> tensor<12x40x128xf32>
    %69 = tosa.reshape %68 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %70 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %71 = tosa.transpose %69, %70 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x40x12x128xf32>
    %72 = tosa.reshape %71 {new_shape = array<i64: 1, 40, 1536>} : (tensor<1x40x12x128xf32>) -> tensor<1x40x1536xf32>
    
    %t_end_6 = call @rtclock() : () -> f64
    %duration_6 = arith.subf %t_end_6, %t_start_6 : f64
    %name_ptr_6 = llvm.mlir.addressof @op_name_attn_v : !llvm.ptr
    call @record_timing(%name_ptr_6, %duration_6) : (!llvm.ptr, f64) -> ()
    
    // ===== O Projection =====
    %t_start_7 = call @rtclock() : () -> f64
    
    %73 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %74 = tosa.transpose %arg7, %73 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %75 = tosa.reshape %72 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %76 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%75, %74 : tensor<40x1536xf32>, tensor<1536x1536xf32>) outs(%cst_4 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %77 = tosa.reshape %76 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %78 = tosa.add %arg8, %77 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    
    %t_end_7 = call @rtclock() : () -> f64
    %duration_7 = arith.subf %t_end_7, %t_start_7 : f64
    %name_ptr_7 = llvm.mlir.addressof @op_name_o_proj : !llvm.ptr
    call @record_timing(%name_ptr_7, %duration_7) : (!llvm.ptr, f64) -> ()
    
    // ===== Post-Attention LayerNorm =====
    %t_start_8 = call @rtclock() : () -> f64
    
    %79 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_5 = arith.constant 2 : i32
    %80 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78 : tensor<1x40x1536xf32>) outs(%79 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %111 = math.fpowi %in, %c2_i32_5 : f32, i32
      linalg.yield %111 : f32
    } -> tensor<1x40x1536xf32>
    %81 = tosa.reduce_sum %80 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %82 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %83 = tosa.reciprocal %82 : (tensor<1xf32>) -> tensor<1xf32>
    %84 = tosa.reshape %83 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %85 = tosa.mul %84, %81 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %86 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %87 = tosa.add %85, %86 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %88 = tosa.rsqrt %87 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %89 = tosa.mul %78, %88 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %90 = tosa.reshape %arg9 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %91 = tosa.mul %90, %89 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    
    %t_end_8 = call @rtclock() : () -> f64
    %duration_8 = arith.subf %t_end_8, %t_start_8 : f64
    %name_ptr_8 = llvm.mlir.addressof @op_name_post_attn_layernorm : !llvm.ptr
    call @record_timing(%name_ptr_8, %duration_8) : (!llvm.ptr, f64) -> ()
    
    // ===== FFN Gate Projection =====
    %t_start_9 = call @rtclock() : () -> f64
    
    %92 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %93 = tosa.transpose %arg10, %92 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %94 = tosa.reshape %91 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %95 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%94, %93 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_6 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %96 = tosa.reshape %95 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %97 = tosa.sigmoid %96 : (tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %98 = tosa.mul %96, %97 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    
    %t_end_9 = call @rtclock() : () -> f64
    %duration_9 = arith.subf %t_end_9, %t_start_9 : f64
    %name_ptr_9 = llvm.mlir.addressof @op_name_ffn_gate : !llvm.ptr
    call @record_timing(%name_ptr_9, %duration_9) : (!llvm.ptr, f64) -> ()
    
    // ===== FFN Up Projection =====
    %t_start_10 = call @rtclock() : () -> f64
    
    %99 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %100 = tosa.transpose %arg11, %99 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %101 = tosa.reshape %91 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %102 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%101, %100 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_7 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %103 = tosa.reshape %102 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %104 = tosa.mul %98, %103 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    
    %t_end_10 = call @rtclock() : () -> f64
    %duration_10 = arith.subf %t_end_10, %t_start_10 : f64
    %name_ptr_10 = llvm.mlir.addressof @op_name_ffn_up : !llvm.ptr
    call @record_timing(%name_ptr_10, %duration_10) : (!llvm.ptr, f64) -> ()
    
    // ===== FFN Down Projection =====
    %t_start_11 = call @rtclock() : () -> f64
    
    %105 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %106 = tosa.transpose %arg12, %105 : (tensor<1536x8960xf32>, tensor<2xi32>) -> tensor<8960x1536xf32>
    %107 = tosa.reshape %104 {new_shape = array<i64: 40, 8960>} : (tensor<1x40x8960xf32>) -> tensor<40x8960xf32>
    %cst_8 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %108 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%107, %106 : tensor<40x8960xf32>, tensor<8960x1536xf32>) outs(%cst_8 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %109 = tosa.reshape %108 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %110 = tosa.add %78, %109 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    
    %t_end_11 = call @rtclock() : () -> f64
    %duration_11 = arith.subf %t_end_11, %t_start_11 : f64
    %name_ptr_11 = llvm.mlir.addressof @op_name_ffn_down : !llvm.ptr
    call @record_timing(%name_ptr_11, %duration_11) : (!llvm.ptr, f64) -> ()
    
    return %110 : tensor<1x40x1536xf32>
  }
}

