#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1) -> (d0, 0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1, 0)>
module {
  // Timing function declarations
  func.func private @rtclock() -> f64
  llvm.mlir.global internal constant @str_layernorm1("LayerNorm1\00")
  llvm.mlir.global internal constant @str_q_proj("Q_Proj\00")
  llvm.mlir.global internal constant @str_k_proj("K_Proj\00")
  llvm.mlir.global internal constant @str_v_proj("V_Proj\00")
  llvm.mlir.global internal constant @str_attention("Attention\00")
  llvm.mlir.global internal constant @str_o_proj("O_Proj\00")
  llvm.mlir.global internal constant @str_layernorm2("LayerNorm2\00")
  llvm.mlir.global internal constant @str_gate_proj("Gate_Proj\00")
  llvm.mlir.global internal constant @str_up_proj("Up_Proj\00")
  llvm.mlir.global internal constant @str_down_proj("Down_Proj\00")
  func.func private @record_timing(!llvm.ptr, f64) -> ()

  func.func @subgraph0(%arg0: tensor<1x1x1536xf32>, %arg1: tensor<1x1x1536xf32>, %arg2: tensor<1536xf32>, %arg3: tensor<1536x1536xf32>, %arg4: tensor<1536x256xf32>, %arg5: tensor<1536x256xf32>, %arg6: tensor<1x1xi64>, %arg7: tensor<1536x1536xf32>, %arg8: tensor<1x1x1536xf32>, %arg9: tensor<1536xf32>, %arg10: tensor<1536x8960xf32>, %arg11: tensor<1536x8960xf32>, %arg12: tensor<8960x1536xf32>) -> tensor<1x1x1536xf32> {
    // ========== LayerNorm1 ==========
    %t0_start = call @rtclock() : () -> f64
    %0 = tensor.empty() : tensor<1x1x1536xf32>
    %c2_i32 = arith.constant 2 : i32
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8 : tensor<1x1x1536xf32>) outs(%0 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %95 = math.fpowi %in, %c2_i32 : f32, i32
      linalg.yield %95 : f32
    } -> tensor<1x1x1536xf32>
    %2 = tosa.reduce_sum %1 {axis = 2 : i32} : (tensor<1x1x1536xf32>) -> tensor<1x1x1xf32>
    %3 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4 = tosa.reciprocal %3 : (tensor<1xf32>) -> tensor<1xf32>
    %5 = tosa.reshape %4 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %6 = tosa.mul %5, %2 : (tensor<1x1x1xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
    %7 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
    %8 = tosa.add %6, %7 : (tensor<1x1x1xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
    %9 = tosa.rsqrt %8 : (tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
    %10 = tosa.mul %arg8, %9 : (tensor<1x1x1536xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1536xf32>
    %11 = tosa.reshape %arg2 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %12 = tosa.mul %11, %10 : (tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) -> tensor<1x1x1536xf32>
    %t0_end = call @rtclock() : () -> f64
    %t0_dur = arith.subf %t0_end, %t0_start : f64
    %str0 = llvm.mlir.addressof @str_layernorm1 : !llvm.ptr
    func.call @record_timing(%str0, %t0_dur) : (!llvm.ptr, f64) -> ()

    // ========== Q_Proj ==========
    %t1_start = call @rtclock() : () -> f64
    %13 = tosa.reshape %12 {new_shape = array<i64: 1, 1536>} : (tensor<1x1x1536xf32>) -> tensor<1x1536xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<1x1536xf32>
    %14 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%13, %arg3 : tensor<1x1536xf32>, tensor<1536x1536xf32>) outs(%cst : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %15 = tosa.reshape %14 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1x1536xf32>) -> tensor<1x1x1536xf32>
    %t1_end = call @rtclock() : () -> f64
    %t1_dur = arith.subf %t1_end, %t1_start : f64
    %str1 = llvm.mlir.addressof @str_q_proj : !llvm.ptr
    func.call @record_timing(%str1, %t1_dur) : (!llvm.ptr, f64) -> ()

    // ========== K_Proj ==========
    %t2_start = call @rtclock() : () -> f64
    %16 = tosa.reshape %12 {new_shape = array<i64: 1, 1536>} : (tensor<1x1x1536xf32>) -> tensor<1x1536xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x256xf32>
    %17 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%16, %arg4 : tensor<1x1536xf32>, tensor<1536x256xf32>) outs(%cst_0 : tensor<1x256xf32>) -> tensor<1x256xf32>
    %18 = tosa.reshape %17 {new_shape = array<i64: 1, 1, 256>} : (tensor<1x256xf32>) -> tensor<1x1x256xf32>
    %t2_end = call @rtclock() : () -> f64
    %t2_dur = arith.subf %t2_end, %t2_start : f64
    %str2 = llvm.mlir.addressof @str_k_proj : !llvm.ptr
    func.call @record_timing(%str2, %t2_dur) : (!llvm.ptr, f64) -> ()

    // ========== V_Proj ==========
    %t3_start = call @rtclock() : () -> f64
    %19 = tosa.reshape %12 {new_shape = array<i64: 1, 1536>} : (tensor<1x1x1536xf32>) -> tensor<1x1536xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<1x256xf32>
    %20 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%19, %arg5 : tensor<1x1536xf32>, tensor<1536x256xf32>) outs(%cst_1 : tensor<1x256xf32>) -> tensor<1x256xf32>
    %21 = tosa.reshape %20 {new_shape = array<i64: 1, 1, 256>} : (tensor<1x256xf32>) -> tensor<1x1x256xf32>
    %t3_end = call @rtclock() : () -> f64
    %t3_dur = arith.subf %t3_end, %t3_start : f64
    %str3 = llvm.mlir.addressof @str_v_proj : !llvm.ptr
    func.call @record_timing(%str3, %t3_dur) : (!llvm.ptr, f64) -> ()

    // ========== Attention ==========
    %t4_start = call @rtclock() : () -> f64
    %22 = tosa.reshape %15 {new_shape = array<i64: 1, 1, 12, 128>} : (tensor<1x1x1536xf32>) -> tensor<1x1x12x128xf32>
    %23 = tosa.reshape %22 {new_shape = array<i64: 1, 12, 1, 128>} : (tensor<1x1x12x128xf32>) -> tensor<1x12x1x128xf32>
    %24 = tosa.reshape %18 {new_shape = array<i64: 1, 1, 2, 128>} : (tensor<1x1x256xf32>) -> tensor<1x1x2x128xf32>
    %25 = tosa.reshape %21 {new_shape = array<i64: 1, 1, 2, 128>} : (tensor<1x1x256xf32>) -> tensor<1x1x2x128xf32>
    %26 = tosa.reshape %24 {new_shape = array<i64: 1, 2, 1, 1, 128>} : (tensor<1x1x2x128xf32>) -> tensor<1x2x1x1x128xf32>
    %27 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x1x128xf32>}> : () -> tensor<1x2x6x1x128xf32>
    %28 = tosa.add %26, %27 : (tensor<1x2x1x1x128xf32>, tensor<1x2x6x1x128xf32>) -> tensor<1x2x6x1x128xf32>
    %29 = tosa.reshape %28 {new_shape = array<i64: 1, 12, 1, 128>} : (tensor<1x2x6x1x128xf32>) -> tensor<1x12x1x128xf32>
    %30 = tosa.reshape %25 {new_shape = array<i64: 1, 2, 1, 1, 128>} : (tensor<1x1x2x128xf32>) -> tensor<1x2x1x1x128xf32>
    %31 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x1x128xf32>}> : () -> tensor<1x2x6x1x128xf32>
    %32 = tosa.add %30, %31 : (tensor<1x2x1x1x128xf32>, tensor<1x2x6x1x128xf32>) -> tensor<1x2x6x1x128xf32>
    %33 = tosa.reshape %32 {new_shape = array<i64: 1, 12, 1, 128>} : (tensor<1x2x6x1x128xf32>) -> tensor<1x12x1x128xf32>
    %34 = tosa.reshape %29 {new_shape = array<i64: 1, 12, 128, 1>} : (tensor<1x12x1x128xf32>) -> tensor<1x12x128x1xf32>
    %35 = tosa.reshape %23 {new_shape = array<i64: 12, 1, 128>} : (tensor<1x12x1x128xf32>) -> tensor<12x1x128xf32>
    %36 = tosa.reshape %34 {new_shape = array<i64: 12, 128, 1>} : (tensor<1x12x128x1xf32>) -> tensor<12x128x1xf32>
    %37 = tensor.empty() : tensor<12x128xf32>
    %38 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel"]} ins(%35 : tensor<12x1x128xf32>) outs(%37 : tensor<12x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<12x128xf32>
    %39 = tensor.empty() : tensor<12x128xf32>
    %40 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%36 : tensor<12x128x1xf32>) outs(%39 : tensor<12x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<12x128xf32>
    %41 = tosa.mul %38, %40 : (tensor<12x128xf32>, tensor<12x128xf32>) -> tensor<12x128xf32>
    %42 = tosa.reduce_sum %41 {axis = 1 : i32} : (tensor<12x128xf32>) -> tensor<12x1xf32>
    %43 = tosa.reshape %42 {new_shape = array<i64: 12, 1, 1>} : (tensor<12x1xf32>) -> tensor<12x1x1xf32>
    %44 = tosa.reshape %43 {new_shape = array<i64: 1, 12, 1, 1>} : (tensor<12x1x1xf32>) -> tensor<1x12x1x1xf32>
    %cst_2 = arith.constant dense<0.0883883461> : tensor<1xf32>
    %45 = tosa.reshape %cst_2 {new_shape = array<i64: 1, 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1x1xf32>
    %46 = tosa.mul %44, %45 : (tensor<1x12x1x1xf32>, tensor<1x1x1x1xf32>) -> tensor<1x12x1x1xf32>
    %47 = tosa.reshape %arg6 {new_shape = array<i64: 1, 1, 1, 1>} : (tensor<1x1xi64>) -> tensor<1x1x1x1xi64>
    %48 = tosa.cast %47 : (tensor<1x1x1x1xi64>) -> tensor<1x1x1x1xf32>
    %49 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
    %50 = tosa.sub %49, %48 : (tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
    %cst_3 = arith.constant dense<-1.000000e+04> : tensor<1xf32>
    %51 = tosa.reshape %cst_3 {new_shape = array<i64: 1, 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1x1xf32>
    %52 = tosa.mul %50, %51 : (tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
    %53 = tosa.add %46, %52 : (tensor<1x12x1x1xf32>, tensor<1x1x1x1xf32>) -> tensor<1x12x1x1xf32>
    %54 = tosa.reduce_max %53 {axis = 3 : i32} : (tensor<1x12x1x1xf32>) -> tensor<1x12x1x1xf32>
    %55 = tosa.sub %53, %54 : (tensor<1x12x1x1xf32>, tensor<1x12x1x1xf32>) -> tensor<1x12x1x1xf32>
    %56 = tosa.exp %55 : (tensor<1x12x1x1xf32>) -> tensor<1x12x1x1xf32>
    %57 = tosa.reduce_sum %56 {axis = 3 : i32} : (tensor<1x12x1x1xf32>) -> tensor<1x12x1x1xf32>
    %58 = tosa.reciprocal %57 : (tensor<1x12x1x1xf32>) -> tensor<1x12x1x1xf32>
    %59 = tosa.mul %56, %58 : (tensor<1x12x1x1xf32>, tensor<1x12x1x1xf32>) -> tensor<1x12x1x1xf32>
    %60 = tosa.reshape %59 {new_shape = array<i64: 12, 1, 1>} : (tensor<1x12x1x1xf32>) -> tensor<12x1x1xf32>
    %61 = tosa.reshape %33 {new_shape = array<i64: 12, 1, 128>} : (tensor<1x12x1x128xf32>) -> tensor<12x1x128xf32>
    %62 = tosa.matmul %60, %61 : (tensor<12x1x1xf32>, tensor<12x1x128xf32>) -> tensor<12x1x128xf32>
    %63 = tosa.reshape %62 {new_shape = array<i64: 1, 12, 1, 128>} : (tensor<12x1x128xf32>) -> tensor<1x12x1x128xf32>
    %64 = tosa.reshape %63 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1x12x1x128xf32>) -> tensor<1x1x1536xf32>
    %t4_end = call @rtclock() : () -> f64
    %t4_dur = arith.subf %t4_end, %t4_start : f64
    %str4 = llvm.mlir.addressof @str_attention : !llvm.ptr
    func.call @record_timing(%str4, %t4_dur) : (!llvm.ptr, f64) -> ()

    // ========== O_Proj ==========
    %t5_start = call @rtclock() : () -> f64
    %65 = tosa.reshape %64 {new_shape = array<i64: 1, 1536>} : (tensor<1x1x1536xf32>) -> tensor<1x1536xf32>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<1x1536xf32>
    %66 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%65, %arg7 : tensor<1x1536xf32>, tensor<1536x1536xf32>) outs(%cst_4 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %67 = tosa.reshape %66 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1x1536xf32>) -> tensor<1x1x1536xf32>
    %68 = tosa.add %arg8, %67 : (tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) -> tensor<1x1x1536xf32>
    %t5_end = call @rtclock() : () -> f64
    %t5_dur = arith.subf %t5_end, %t5_start : f64
    %str5 = llvm.mlir.addressof @str_o_proj : !llvm.ptr
    func.call @record_timing(%str5, %t5_dur) : (!llvm.ptr, f64) -> ()

    // ========== LayerNorm2 ==========
    %t6_start = call @rtclock() : () -> f64
    %69 = tensor.empty() : tensor<1x1x1536xf32>
    %c2_i32_5 = arith.constant 2 : i32
    %70 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%68 : tensor<1x1x1536xf32>) outs(%69 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %95 = math.fpowi %in, %c2_i32_5 : f32, i32
      linalg.yield %95 : f32
    } -> tensor<1x1x1536xf32>
    %71 = tosa.reduce_sum %70 {axis = 2 : i32} : (tensor<1x1x1536xf32>) -> tensor<1x1x1xf32>
    %72 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %73 = tosa.reciprocal %72 : (tensor<1xf32>) -> tensor<1xf32>
    %74 = tosa.reshape %73 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %75 = tosa.mul %74, %71 : (tensor<1x1x1xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
    %76 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
    %77 = tosa.add %75, %76 : (tensor<1x1x1xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
    %78 = tosa.rsqrt %77 : (tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
    %79 = tosa.mul %68, %78 : (tensor<1x1x1536xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1536xf32>
    %80 = tosa.reshape %arg9 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %81 = tosa.mul %80, %79 : (tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) -> tensor<1x1x1536xf32>
    %t6_end = call @rtclock() : () -> f64
    %t6_dur = arith.subf %t6_end, %t6_start : f64
    %str6 = llvm.mlir.addressof @str_layernorm2 : !llvm.ptr
    func.call @record_timing(%str6, %t6_dur) : (!llvm.ptr, f64) -> ()

    // ========== Gate_Proj ==========
    %t7_start = call @rtclock() : () -> f64
    %82 = tosa.reshape %81 {new_shape = array<i64: 1, 1536>} : (tensor<1x1x1536xf32>) -> tensor<1x1536xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<1x8960xf32>
    %83 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%82, %arg10 : tensor<1x1536xf32>, tensor<1536x8960xf32>) outs(%cst_6 : tensor<1x8960xf32>) -> tensor<1x8960xf32>
    %84 = tosa.reshape %83 {new_shape = array<i64: 1, 1, 8960>} : (tensor<1x8960xf32>) -> tensor<1x1x8960xf32>
    %85 = tosa.sigmoid %84 : (tensor<1x1x8960xf32>) -> tensor<1x1x8960xf32>
    %86 = tosa.mul %84, %85 : (tensor<1x1x8960xf32>, tensor<1x1x8960xf32>) -> tensor<1x1x8960xf32>
    %t7_end = call @rtclock() : () -> f64
    %t7_dur = arith.subf %t7_end, %t7_start : f64
    %str7 = llvm.mlir.addressof @str_gate_proj : !llvm.ptr
    func.call @record_timing(%str7, %t7_dur) : (!llvm.ptr, f64) -> ()

    // ========== Up_Proj ==========
    %t8_start = call @rtclock() : () -> f64
    %87 = tosa.reshape %81 {new_shape = array<i64: 1, 1536>} : (tensor<1x1x1536xf32>) -> tensor<1x1536xf32>
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<1x8960xf32>
    %88 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%87, %arg11 : tensor<1x1536xf32>, tensor<1536x8960xf32>) outs(%cst_7 : tensor<1x8960xf32>) -> tensor<1x8960xf32>
    %89 = tosa.reshape %88 {new_shape = array<i64: 1, 1, 8960>} : (tensor<1x8960xf32>) -> tensor<1x1x8960xf32>
    %90 = tosa.mul %86, %89 : (tensor<1x1x8960xf32>, tensor<1x1x8960xf32>) -> tensor<1x1x8960xf32>
    %t8_end = call @rtclock() : () -> f64
    %t8_dur = arith.subf %t8_end, %t8_start : f64
    %str8 = llvm.mlir.addressof @str_up_proj : !llvm.ptr
    func.call @record_timing(%str8, %t8_dur) : (!llvm.ptr, f64) -> ()

    // ========== Down_Proj ==========
    %t9_start = call @rtclock() : () -> f64
    %91 = tosa.reshape %90 {new_shape = array<i64: 1, 8960>} : (tensor<1x1x8960xf32>) -> tensor<1x8960xf32>
    %cst_8 = arith.constant dense<0.000000e+00> : tensor<1x1536xf32>
    %92 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%91, %arg12 : tensor<1x8960xf32>, tensor<8960x1536xf32>) outs(%cst_8 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %93 = tosa.reshape %92 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1x1536xf32>) -> tensor<1x1x1536xf32>
    %94 = tosa.add %68, %93 : (tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) -> tensor<1x1x1536xf32>
    %t9_end = call @rtclock() : () -> f64
    %t9_dur = arith.subf %t9_end, %t9_start : f64
    %str9 = llvm.mlir.addressof @str_down_proj : !llvm.ptr
    func.call @record_timing(%str9, %t9_dur) : (!llvm.ptr, f64) -> ()

    return %94 : tensor<1x1x1536xf32>
  }
}
