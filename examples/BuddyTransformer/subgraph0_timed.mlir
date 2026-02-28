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
      %149 = math.fpowi %in, %c2_i32 : f32, i32
      linalg.yield %149 : f32
    } -> tensor<1x40x1536xf32>
    %2 = tosa.reduce_sum %1 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %3 = "tosa.const"() <{values = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4 = tosa.reciprocal %3 : (tensor<1xf32>) -> tensor<1xf32>
    %5 = tosa.const_shape  {values = dense<1> : tensor<3xindex>} : () -> !tosa.shape<3>
    %6 = tosa.reshape %4, %5 : (tensor<1xf32>, !tosa.shape<3>) -> tensor<1x1x1xf32>
    %7 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %8 = tosa.mul %6, %2, %7 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>, tensor<1xi8>) -> tensor<1x40x1xf32>
    %9 = "tosa.const"() <{values = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %10 = tosa.add %8, %9 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %11 = tosa.rsqrt %10 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %12 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %13 = tosa.mul %arg8, %11, %12 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>, tensor<1xi8>) -> tensor<1x40x1536xf32>
    %14 = tosa.const_shape  {values = dense<[1, 1, 1536]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %15 = tosa.reshape %arg2, %14 : (tensor<1536xf32>, !tosa.shape<3>) -> tensor<1x1x1536xf32>
    %16 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %17 = tosa.mul %15, %13, %16 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>, tensor<1xi8>) -> tensor<1x40x1536xf32>

    %t_end_0 = call @rtclock() : () -> f64
    %duration_0 = arith.subf %t_end_0, %t_start_0 : f64
    %name_ptr_0 = llvm.mlir.addressof @op_name_input_layernorm : !llvm.ptr
    call @record_timing(%name_ptr_0, %duration_0) : (!llvm.ptr, f64) -> ()

    // ===== Q Projection =====
    %t_start_1 = call @rtclock() : () -> f64

    %18 = tosa.transpose %arg3 {perms = array<i32: 1, 0>} : (tensor<1536x1536xf32>) -> tensor<1536x1536xf32>
    %19 = tosa.const_shape  {values = dense<[40, 1536]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %20 = tosa.reshape %17, %19 : (tensor<1x40x1536xf32>, !tosa.shape<2>) -> tensor<40x1536xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %21 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%20, %18 : tensor<40x1536xf32>, tensor<1536x1536xf32>) outs(%cst : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %22 = tosa.const_shape  {values = dense<[1, 40, 1536]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %23 = tosa.reshape %21, %22 : (tensor<40x1536xf32>, !tosa.shape<3>) -> tensor<1x40x1536xf32>

    %t_end_1 = call @rtclock() : () -> f64
    %duration_1 = arith.subf %t_end_1, %t_start_1 : f64
    %name_ptr_1 = llvm.mlir.addressof @op_name_q_proj : !llvm.ptr
    call @record_timing(%name_ptr_1, %duration_1) : (!llvm.ptr, f64) -> ()

    // ===== K Projection =====
    %t_start_2 = call @rtclock() : () -> f64

    %24 = tosa.transpose %arg4 {perms = array<i32: 1, 0>} : (tensor<256x1536xf32>) -> tensor<1536x256xf32>
    %25 = tosa.const_shape  {values = dense<[40, 1536]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %26 = tosa.reshape %17, %25 : (tensor<1x40x1536xf32>, !tosa.shape<2>) -> tensor<40x1536xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<40x256xf32>
    %27 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%26, %24 : tensor<40x1536xf32>, tensor<1536x256xf32>) outs(%cst_0 : tensor<40x256xf32>) -> tensor<40x256xf32>
    %28 = tosa.const_shape  {values = dense<[1, 40, 256]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %29 = tosa.reshape %27, %28 : (tensor<40x256xf32>, !tosa.shape<3>) -> tensor<1x40x256xf32>


    %t_end_2 = call @rtclock() : () -> f64
    %duration_2 = arith.subf %t_end_2, %t_start_2 : f64
    %name_ptr_2 = llvm.mlir.addressof @op_name_k_proj : !llvm.ptr
    call @record_timing(%name_ptr_2, %duration_2) : (!llvm.ptr, f64) -> ()

    // ===== V Projection =====
    %t_start_3 = call @rtclock() : () -> f64

    %30 = tosa.transpose %arg5 {perms = array<i32: 1, 0>} : (tensor<256x1536xf32>) -> tensor<1536x256xf32>
    %31 = tosa.const_shape  {values = dense<[40, 1536]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %32 = tosa.reshape %17, %31 : (tensor<1x40x1536xf32>, !tosa.shape<2>) -> tensor<40x1536xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<40x256xf32>
    %33 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%32, %30 : tensor<40x1536xf32>, tensor<1536x256xf32>) outs(%cst_1 : tensor<40x256xf32>) -> tensor<40x256xf32>
    %34 = tosa.const_shape  {values = dense<[1, 40, 256]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %35 = tosa.reshape %33, %34 : (tensor<40x256xf32>, !tosa.shape<3>) -> tensor<1x40x256xf32>

    %t_end_3 = call @rtclock() : () -> f64
    %duration_3 = arith.subf %t_end_3, %t_start_3 : f64
    %name_ptr_3 = llvm.mlir.addressof @op_name_v_proj : !llvm.ptr
    call @record_timing(%name_ptr_3, %duration_3) : (!llvm.ptr, f64) -> ()

    // Reshape and transpose for attention
    %36 = tosa.const_shape  {values = dense<[1, 40, 12, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %37 = tosa.reshape %23, %36 : (tensor<1x40x1536xf32>, !tosa.shape<4>) -> tensor<1x40x12x128xf32>
    %38 = tosa.transpose %37 {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x40x12x128xf32>) -> tensor<1x12x40x128xf32>
    %39 = tosa.const_shape  {values = dense<[1, 40, 2, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %40 = tosa.reshape %29, %39 : (tensor<1x40x256xf32>, !tosa.shape<4>) -> tensor<1x40x2x128xf32>
    %41 = tosa.transpose %40 {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x40x2x128xf32>) -> tensor<1x2x40x128xf32>
    %42 = tosa.const_shape  {values = dense<[1, 40, 2, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %43 = tosa.reshape %35, %42 : (tensor<1x40x256xf32>, !tosa.shape<4>) -> tensor<1x40x2x128xf32>
    %44 = tosa.transpose %43 {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x40x2x128xf32>) -> tensor<1x2x40x128xf32>
    %45 = tosa.const_shape  {values = dense<[1, 2, 1, 40, 128]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %46 = tosa.reshape %41, %45 : (tensor<1x2x40x128xf32>, !tosa.shape<5>) -> tensor<1x2x1x40x128xf32>
    %47 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %48 = tosa.add %46, %47 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %49 = tosa.const_shape  {values = dense<[1, 12, 40, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %50 = tosa.reshape %48, %49 : (tensor<1x2x6x40x128xf32>, !tosa.shape<4>) -> tensor<1x12x40x128xf32>
    %51 = tosa.const_shape  {values = dense<[1, 2, 1, 40, 128]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %52 = tosa.reshape %44, %51 : (tensor<1x2x40x128xf32>, !tosa.shape<5>) -> tensor<1x2x1x40x128xf32>
    %53 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %54 = tosa.add %52, %53 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %55 = tosa.const_shape  {values = dense<[1, 12, 40, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %56 = tosa.reshape %54, %55 : (tensor<1x2x6x40x128xf32>, !tosa.shape<4>) -> tensor<1x12x40x128xf32>
    %57 = tosa.transpose %50 {perms = array<i32: 0, 1, 3, 2>} : (tensor<1x12x40x128xf32>) -> tensor<1x12x128x40xf32>

    // ===== Attention QK^T =====
    %t_start_4 = call @rtclock() : () -> f64

    %58 = tosa.const_shape  {values = dense<[12, 40, 128]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %59 = tosa.reshape %38, %58 : (tensor<1x12x40x128xf32>, !tosa.shape<3>) -> tensor<12x40x128xf32>
    %60 = tosa.const_shape  {values = dense<[12, 128, 40]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %61 = tosa.reshape %57, %60 : (tensor<1x12x128x40xf32>, !tosa.shape<3>) -> tensor<12x128x40xf32>
    %62 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %63 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %64 = tosa.matmul %59, %61, %62, %63 : (tensor<12x40x128xf32>, tensor<12x128x40xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<12x40x40xf32>
    %65 = tosa.const_shape  {values = dense<[1, 12, 40, 40]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %66 = tosa.reshape %64, %65 : (tensor<12x40x40xf32>, !tosa.shape<4>) -> tensor<1x12x40x40xf32>
    %cst_2 = arith.constant dense<0.0883883461> : tensor<1xf32>
    %67 = tosa.const_shape  {values = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
    %68 = tosa.reshape %cst_2, %67 : (tensor<1xf32>, !tosa.shape<4>) -> tensor<1x1x1x1xf32>
    %69 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %70 = tosa.mul %66, %68, %69 : (tensor<1x12x40x40xf32>, tensor<1x1x1x1xf32>, tensor<1xi8>) -> tensor<1x12x40x40xf32>
    %71 = tosa.const_shape  {values = dense<[1, 1, 1, 40]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %72 = tosa.reshape %arg6, %71 : (tensor<1x40xi64>, !tosa.shape<4>) -> tensor<1x1x1x40xi64>
    %73 = tosa.cast %72 : (tensor<1x1x1x40xi64>) -> tensor<1x1x1x40xf32>
    %74 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x1x1x40xf32>}> : () -> tensor<1x1x1x40xf32>
    %75 = tosa.sub %74, %73 : (tensor<1x1x1x40xf32>, tensor<1x1x1x40xf32>) -> tensor<1x1x1x40xf32>
    %cst_3 = arith.constant dense<-1.000000e+04> : tensor<1xf32>
    %76 = tosa.const_shape  {values = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
    %77 = tosa.reshape %cst_3, %76 : (tensor<1xf32>, !tosa.shape<4>) -> tensor<1x1x1x1xf32>
    %78 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %79 = tosa.mul %75, %77, %78 : (tensor<1x1x1x40xf32>, tensor<1x1x1x1xf32>, tensor<1xi8>) -> tensor<1x1x1x40xf32>
    %80 = tosa.add %70, %79 : (tensor<1x12x40x40xf32>, tensor<1x1x1x40xf32>) -> tensor<1x12x40x40xf32>

    %t_end_4 = call @rtclock() : () -> f64
    %duration_4 = arith.subf %t_end_4, %t_start_4 : f64
    %name_ptr_4 = llvm.mlir.addressof @op_name_attn_qk : !llvm.ptr
    call @record_timing(%name_ptr_4, %duration_4) : (!llvm.ptr, f64) -> ()

    // ===== Attention Softmax =====
    %t_start_5 = call @rtclock() : () -> f64

    %81 = tosa.reduce_max %80 {axis = 3 : i32} : (tensor<1x12x40x40xf32>) -> tensor<1x12x40x1xf32>
    %82 = tosa.sub %80, %81 : (tensor<1x12x40x40xf32>, tensor<1x12x40x1xf32>) -> tensor<1x12x40x40xf32>
    %83 = tosa.exp %82 : (tensor<1x12x40x40xf32>) -> tensor<1x12x40x40xf32>
    %84 = tosa.reduce_sum %83 {axis = 3 : i32} : (tensor<1x12x40x40xf32>) -> tensor<1x12x40x1xf32>
    %85 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %86 = tosa.reciprocal %84 : (tensor<1x12x40x1xf32>) -> tensor<1x12x40x1xf32>
    %87 = tosa.mul %83, %86, %85 : (tensor<1x12x40x40xf32>, tensor<1x12x40x1xf32>, tensor<1xi8>) -> tensor<1x12x40x40xf32>

    %t_end_5 = call @rtclock() : () -> f64
    %duration_5 = arith.subf %t_end_5, %t_start_5 : f64
    %name_ptr_5 = llvm.mlir.addressof @op_name_attn_softmax : !llvm.ptr
    call @record_timing(%name_ptr_5, %duration_5) : (!llvm.ptr, f64) -> ()

    // ===== Attention * V =====
    %t_start_6 = call @rtclock() : () -> f64

    %88 = tosa.const_shape  {values = dense<[12, 40, 40]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %89 = tosa.reshape %87, %88 : (tensor<1x12x40x40xf32>, !tosa.shape<3>) -> tensor<12x40x40xf32>
    %90 = tosa.const_shape  {values = dense<[12, 40, 128]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %91 = tosa.reshape %56, %90 : (tensor<1x12x40x128xf32>, !tosa.shape<3>) -> tensor<12x40x128xf32>
    %92 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %93 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %94 = tosa.matmul %89, %91, %92, %93 : (tensor<12x40x40xf32>, tensor<12x40x128xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<12x40x128xf32>
    %95 = tosa.const_shape  {values = dense<[1, 12, 40, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %96 = tosa.reshape %94, %95 : (tensor<12x40x128xf32>, !tosa.shape<4>) -> tensor<1x12x40x128xf32>

    %t_end_6 = call @rtclock() : () -> f64
    %duration_6 = arith.subf %t_end_6, %t_start_6 : f64
    %name_ptr_6 = llvm.mlir.addressof @op_name_attn_v : !llvm.ptr
    call @record_timing(%name_ptr_6, %duration_6) : (!llvm.ptr, f64) -> ()

    // ===== O Projection =====
    %t_start_7 = call @rtclock() : () -> f64

    %97 = tosa.transpose %96 {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x12x40x128xf32>) -> tensor<1x40x12x128xf32>
    %98 = tosa.const_shape  {values = dense<[1, 40, 1536]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %99 = tosa.reshape %97, %98 : (tensor<1x40x12x128xf32>, !tosa.shape<3>) -> tensor<1x40x1536xf32>
    %100 = tosa.transpose %arg7 {perms = array<i32: 1, 0>} : (tensor<1536x1536xf32>) -> tensor<1536x1536xf32>
    %101 = tosa.const_shape  {values = dense<[40, 1536]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %102 = tosa.reshape %99, %101 : (tensor<1x40x1536xf32>, !tosa.shape<2>) -> tensor<40x1536xf32>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %103 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%102, %100 : tensor<40x1536xf32>, tensor<1536x1536xf32>) outs(%cst_4 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %104 = tosa.const_shape  {values = dense<[1, 40, 1536]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %105 = tosa.reshape %103, %104 : (tensor<40x1536xf32>, !tosa.shape<3>) -> tensor<1x40x1536xf32>
    %106 = tosa.add %arg8, %105 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>

    %t_end_7 = call @rtclock() : () -> f64
    %duration_7 = arith.subf %t_end_7, %t_start_7 : f64
    %name_ptr_7 = llvm.mlir.addressof @op_name_o_proj : !llvm.ptr
    call @record_timing(%name_ptr_7, %duration_7) : (!llvm.ptr, f64) -> ()

    // ===== Post-Attention LayerNorm =====
    %t_start_8 = call @rtclock() : () -> f64

    %107 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_5 = arith.constant 2 : i32
    %108 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%106 : tensor<1x40x1536xf32>) outs(%107 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %149 = math.fpowi %in, %c2_i32_5 : f32, i32
      linalg.yield %149 : f32
    } -> tensor<1x40x1536xf32>
    %109 = tosa.reduce_sum %108 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %110 = "tosa.const"() <{values = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %111 = tosa.reciprocal %110 : (tensor<1xf32>) -> tensor<1xf32>
    %112 = tosa.const_shape  {values = dense<1> : tensor<3xindex>} : () -> !tosa.shape<3>
    %113 = tosa.reshape %111, %112 : (tensor<1xf32>, !tosa.shape<3>) -> tensor<1x1x1xf32>
    %114 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %115 = tosa.mul %113, %109, %114 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>, tensor<1xi8>) -> tensor<1x40x1xf32>
    %116 = "tosa.const"() <{values = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %117 = tosa.add %115, %116 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %118 = tosa.rsqrt %117 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %119 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %120 = tosa.mul %106, %118, %119 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>, tensor<1xi8>) -> tensor<1x40x1536xf32>
    %121 = tosa.const_shape  {values = dense<[1, 1, 1536]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %122 = tosa.reshape %arg9, %121 : (tensor<1536xf32>, !tosa.shape<3>) -> tensor<1x1x1536xf32>
    %123 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %124 = tosa.mul %122, %120, %123 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>, tensor<1xi8>) -> tensor<1x40x1536xf32>

    %t_end_8 = call @rtclock() : () -> f64
    %duration_8 = arith.subf %t_end_8, %t_start_8 : f64
    %name_ptr_8 = llvm.mlir.addressof @op_name_post_attn_layernorm : !llvm.ptr
    call @record_timing(%name_ptr_8, %duration_8) : (!llvm.ptr, f64) -> ()

    // ===== FFN Gate Projection =====
    %t_start_9 = call @rtclock() : () -> f64

    %125 = tosa.transpose %arg10 {perms = array<i32: 1, 0>} : (tensor<8960x1536xf32>) -> tensor<1536x8960xf32>
    %126 = tosa.const_shape  {values = dense<[40, 1536]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %127 = tosa.reshape %124, %126 : (tensor<1x40x1536xf32>, !tosa.shape<2>) -> tensor<40x1536xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %128 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%127, %125 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_6 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %129 = tosa.const_shape  {values = dense<[1, 40, 8960]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %130 = tosa.reshape %128, %129 : (tensor<40x8960xf32>, !tosa.shape<3>) -> tensor<1x40x8960xf32>
    %131 = tosa.sigmoid %130 : (tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %132 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %133 = tosa.mul %130, %131, %132 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>, tensor<1xi8>) -> tensor<1x40x8960xf32>

    %t_end_9 = call @rtclock() : () -> f64
    %duration_9 = arith.subf %t_end_9, %t_start_9 : f64
    %name_ptr_9 = llvm.mlir.addressof @op_name_ffn_gate : !llvm.ptr
    call @record_timing(%name_ptr_9, %duration_9) : (!llvm.ptr, f64) -> ()

    // ===== FFN Up Projection =====
    %t_start_10 = call @rtclock() : () -> f64

    %134 = tosa.transpose %arg11 {perms = array<i32: 1, 0>} : (tensor<8960x1536xf32>) -> tensor<1536x8960xf32>
    %135 = tosa.const_shape  {values = dense<[40, 1536]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %136 = tosa.reshape %124, %135 : (tensor<1x40x1536xf32>, !tosa.shape<2>) -> tensor<40x1536xf32>
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %137 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%136, %134 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_7 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %138 = tosa.const_shape  {values = dense<[1, 40, 8960]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %139 = tosa.reshape %137, %138 : (tensor<40x8960xf32>, !tosa.shape<3>) -> tensor<1x40x8960xf32>
    %140 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %141 = tosa.mul %133, %139, %140 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>, tensor<1xi8>) -> tensor<1x40x8960xf32>

    %t_end_10 = call @rtclock() : () -> f64
    %duration_10 = arith.subf %t_end_10, %t_start_10 : f64
    %name_ptr_10 = llvm.mlir.addressof @op_name_ffn_up : !llvm.ptr
    call @record_timing(%name_ptr_10, %duration_10) : (!llvm.ptr, f64) -> ()

    // ===== FFN Down Projection =====
    %t_start_11 = call @rtclock() : () -> f64

    %142 = tosa.transpose %arg12 {perms = array<i32: 1, 0>} : (tensor<1536x8960xf32>) -> tensor<8960x1536xf32>
    %143 = tosa.const_shape  {values = dense<[40, 8960]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %144 = tosa.reshape %141, %143 : (tensor<1x40x8960xf32>, !tosa.shape<2>) -> tensor<40x8960xf32>
    %cst_8 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %145 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%144, %142 : tensor<40x8960xf32>, tensor<8960x1536xf32>) outs(%cst_8 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %146 = tosa.const_shape  {values = dense<[1, 40, 1536]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %147 = tosa.reshape %145, %146 : (tensor<40x1536xf32>, !tosa.shape<3>) -> tensor<1x40x1536xf32>
    %148 = tosa.add %106, %147 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>

    %t_end_11 = call @rtclock() : () -> f64
    %duration_11 = arith.subf %t_end_11, %t_start_11 : f64
    %name_ptr_11 = llvm.mlir.addressof @op_name_ffn_down : !llvm.ptr
    call @record_timing(%name_ptr_11, %duration_11) : (!llvm.ptr, f64) -> ()

    return %148 : tensor<1x40x1536xf32>
  }
}

