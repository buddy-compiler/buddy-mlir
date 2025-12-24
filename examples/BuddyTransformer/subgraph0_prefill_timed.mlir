#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
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

  func.func @subgraph0(%arg0: tensor<1x40x1536xf32>, %arg1: tensor<1x40x1536xf32>, %arg2: tensor<1536xf32>, %arg3: tensor<1536x1536xf32>, %arg4: tensor<1536x256xf32>, %arg5: tensor<1536x256xf32>, %arg6: tensor<1x40xi64>, %arg7: tensor<1536x1536xf32>, %arg8: tensor<1x40x1536xf32>, %arg9: tensor<1536xf32>, %arg10: tensor<1536x8960xf32>, %arg11: tensor<1536x8960xf32>, %arg12: tensor<8960x1536xf32>) -> tensor<1x40x1536xf32> {
    // ========== LayerNorm1 ==========
    %t0_start = call @rtclock() : () -> f64
    %0 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32 = arith.constant 2 : i32
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8 : tensor<1x40x1536xf32>) outs(%0 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %97 = math.fpowi %in, %c2_i32 : f32, i32
      linalg.yield %97 : f32
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
    %t0_end = call @rtclock() : () -> f64
    %t0_dur = arith.subf %t0_end, %t0_start : f64
    %str0 = llvm.mlir.addressof @str_layernorm1 : !llvm.ptr
    func.call @record_timing(%str0, %t0_dur) : (!llvm.ptr, f64) -> ()

    // ========== Q_Proj ==========
    %t1_start = call @rtclock() : () -> f64
    %13 = tosa.reshape %12 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %14 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%13, %arg3 : tensor<40x1536xf32>, tensor<1536x1536xf32>) outs(%cst : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %15 = tosa.reshape %14 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %t1_end = call @rtclock() : () -> f64
    %t1_dur = arith.subf %t1_end, %t1_start : f64
    %str1 = llvm.mlir.addressof @str_q_proj : !llvm.ptr
    func.call @record_timing(%str1, %t1_dur) : (!llvm.ptr, f64) -> ()

    // ========== K_Proj ==========
    %t2_start = call @rtclock() : () -> f64
    %16 = tosa.reshape %12 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<40x256xf32>
    %17 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%16, %arg4 : tensor<40x1536xf32>, tensor<1536x256xf32>) outs(%cst_0 : tensor<40x256xf32>) -> tensor<40x256xf32>
    %18 = tosa.reshape %17 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %t2_end = call @rtclock() : () -> f64
    %t2_dur = arith.subf %t2_end, %t2_start : f64
    %str2 = llvm.mlir.addressof @str_k_proj : !llvm.ptr
    func.call @record_timing(%str2, %t2_dur) : (!llvm.ptr, f64) -> ()

    // ========== V_Proj ==========
    %t3_start = call @rtclock() : () -> f64
    %19 = tosa.reshape %12 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<40x256xf32>
    %20 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%19, %arg5 : tensor<40x1536xf32>, tensor<1536x256xf32>) outs(%cst_1 : tensor<40x256xf32>) -> tensor<40x256xf32>
    %21 = tosa.reshape %20 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %t3_end = call @rtclock() : () -> f64
    %t3_dur = arith.subf %t3_end, %t3_start : f64
    %str3 = llvm.mlir.addressof @str_v_proj : !llvm.ptr
    func.call @record_timing(%str3, %t3_dur) : (!llvm.ptr, f64) -> ()

    // ========== Attention ==========
    %t4_start = call @rtclock() : () -> f64
    %22 = tosa.reshape %15 {new_shape = array<i64: 1, 40, 12, 128>} : (tensor<1x40x1536xf32>) -> tensor<1x40x12x128xf32>
    %23 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %24 = tosa.transpose %22, %23 : (tensor<1x40x12x128xf32>, tensor<4xi32>) -> tensor<1x12x40x128xf32>
    %25 = tosa.reshape %18 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %26 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %27 = tosa.transpose %25, %26 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %28 = tosa.reshape %21 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %29 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %30 = tosa.transpose %28, %29 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %31 = tosa.reshape %27 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %32 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %33 = tosa.add %31, %32 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %34 = tosa.reshape %33 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %35 = tosa.reshape %30 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %36 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %37 = tosa.add %35, %36 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %38 = tosa.reshape %37 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %39 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %40 = tosa.transpose %34, %39 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x12x128x40xf32>
    %41 = tosa.reshape %24 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %42 = tosa.reshape %40 {new_shape = array<i64: 12, 128, 40>} : (tensor<1x12x128x40xf32>) -> tensor<12x128x40xf32>
    %43 = tosa.matmul %41, %42 : (tensor<12x40x128xf32>, tensor<12x128x40xf32>) -> tensor<12x40x40xf32>
    %44 = tosa.reshape %43 {new_shape = array<i64: 1, 12, 40, 40>} : (tensor<12x40x40xf32>) -> tensor<1x12x40x40xf32>
    %cst_2 = arith.constant dense<0.0883883461> : tensor<1xf32>
    %45 = tosa.reshape %cst_2 {new_shape = array<i64: 1, 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1x1xf32>
    %46 = tosa.mul %44, %45 : (tensor<1x12x40x40xf32>, tensor<1x1x1x1xf32>) -> tensor<1x12x40x40xf32>
    %47 = tosa.reshape %arg6 {new_shape = array<i64: 1, 1, 1, 40>} : (tensor<1x40xi64>) -> tensor<1x1x1x40xi64>
    %48 = tosa.cast %47 : (tensor<1x1x1x40xi64>) -> tensor<1x1x1x40xf32>
    %49 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1x1x40xf32>}> : () -> tensor<1x1x1x40xf32>
    %50 = tosa.sub %49, %48 : (tensor<1x1x1x40xf32>, tensor<1x1x1x40xf32>) -> tensor<1x1x1x40xf32>
    %cst_3 = arith.constant dense<-1.000000e+04> : tensor<1xf32>
    %51 = tosa.reshape %cst_3 {new_shape = array<i64: 1, 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1x1xf32>
    %52 = tosa.mul %50, %51 : (tensor<1x1x1x40xf32>, tensor<1x1x1x1xf32>) -> tensor<1x1x1x40xf32>
    %53 = tosa.add %46, %52 : (tensor<1x12x40x40xf32>, tensor<1x1x1x40xf32>) -> tensor<1x12x40x40xf32>
    %54 = tosa.reduce_max %53 {axis = 3 : i32} : (tensor<1x12x40x40xf32>) -> tensor<1x12x40x1xf32>
    %55 = tosa.sub %53, %54 : (tensor<1x12x40x40xf32>, tensor<1x12x40x1xf32>) -> tensor<1x12x40x40xf32>
    %56 = tosa.exp %55 : (tensor<1x12x40x40xf32>) -> tensor<1x12x40x40xf32>
    %57 = tosa.reduce_sum %56 {axis = 3 : i32} : (tensor<1x12x40x40xf32>) -> tensor<1x12x40x1xf32>
    %58 = tosa.reciprocal %57 : (tensor<1x12x40x1xf32>) -> tensor<1x12x40x1xf32>
    %59 = tosa.mul %56, %58 : (tensor<1x12x40x40xf32>, tensor<1x12x40x1xf32>) -> tensor<1x12x40x40xf32>
    %60 = tosa.reshape %59 {new_shape = array<i64: 12, 40, 40>} : (tensor<1x12x40x40xf32>) -> tensor<12x40x40xf32>
    %61 = tosa.reshape %38 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %62 = tosa.matmul %60, %61 : (tensor<12x40x40xf32>, tensor<12x40x128xf32>) -> tensor<12x40x128xf32>
    %63 = tosa.reshape %62 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %64 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %65 = tosa.transpose %63, %64 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x40x12x128xf32>
    %66 = tosa.reshape %65 {new_shape = array<i64: 1, 40, 1536>} : (tensor<1x40x12x128xf32>) -> tensor<1x40x1536xf32>
    %t4_end = call @rtclock() : () -> f64
    %t4_dur = arith.subf %t4_end, %t4_start : f64
    %str4 = llvm.mlir.addressof @str_attention : !llvm.ptr
    func.call @record_timing(%str4, %t4_dur) : (!llvm.ptr, f64) -> ()

    // ========== O_Proj ==========
    %t5_start = call @rtclock() : () -> f64
    %67 = tosa.reshape %66 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %68 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%67, %arg7 : tensor<40x1536xf32>, tensor<1536x1536xf32>) outs(%cst_4 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %69 = tosa.reshape %68 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %70 = tosa.add %arg8, %69 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %t5_end = call @rtclock() : () -> f64
    %t5_dur = arith.subf %t5_end, %t5_start : f64
    %str5 = llvm.mlir.addressof @str_o_proj : !llvm.ptr
    func.call @record_timing(%str5, %t5_dur) : (!llvm.ptr, f64) -> ()

    // ========== LayerNorm2 ==========
    %t6_start = call @rtclock() : () -> f64
    %71 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_5 = arith.constant 2 : i32
    %72 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%70 : tensor<1x40x1536xf32>) outs(%71 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %97 = math.fpowi %in, %c2_i32_5 : f32, i32
      linalg.yield %97 : f32
    } -> tensor<1x40x1536xf32>
    %73 = tosa.reduce_sum %72 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %74 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %75 = tosa.reciprocal %74 : (tensor<1xf32>) -> tensor<1xf32>
    %76 = tosa.reshape %75 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %77 = tosa.mul %76, %73 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %78 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %79 = tosa.add %77, %78 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %80 = tosa.rsqrt %79 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %81 = tosa.mul %70, %80 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %82 = tosa.reshape %arg9 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %83 = tosa.mul %82, %81 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %t6_end = call @rtclock() : () -> f64
    %t6_dur = arith.subf %t6_end, %t6_start : f64
    %str6 = llvm.mlir.addressof @str_layernorm2 : !llvm.ptr
    func.call @record_timing(%str6, %t6_dur) : (!llvm.ptr, f64) -> ()

    // ========== Gate_Proj ==========
    %t7_start = call @rtclock() : () -> f64
    %84 = tosa.reshape %83 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %85 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%84, %arg10 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_6 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %86 = tosa.reshape %85 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %87 = tosa.sigmoid %86 : (tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %88 = tosa.mul %86, %87 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %t7_end = call @rtclock() : () -> f64
    %t7_dur = arith.subf %t7_end, %t7_start : f64
    %str7 = llvm.mlir.addressof @str_gate_proj : !llvm.ptr
    func.call @record_timing(%str7, %t7_dur) : (!llvm.ptr, f64) -> ()

    // ========== Up_Proj ==========
    %t8_start = call @rtclock() : () -> f64
    %89 = tosa.reshape %83 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %90 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%89, %arg11 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_7 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %91 = tosa.reshape %90 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %92 = tosa.mul %88, %91 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %t8_end = call @rtclock() : () -> f64
    %t8_dur = arith.subf %t8_end, %t8_start : f64
    %str8 = llvm.mlir.addressof @str_up_proj : !llvm.ptr
    func.call @record_timing(%str8, %t8_dur) : (!llvm.ptr, f64) -> ()

    // ========== Down_Proj ==========
    %t9_start = call @rtclock() : () -> f64
    %93 = tosa.reshape %92 {new_shape = array<i64: 40, 8960>} : (tensor<1x40x8960xf32>) -> tensor<40x8960xf32>
    %cst_8 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %94 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%93, %arg12 : tensor<40x8960xf32>, tensor<8960x1536xf32>) outs(%cst_8 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %95 = tosa.reshape %94 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %96 = tosa.add %70, %95 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %t9_end = call @rtclock() : () -> f64
    %t9_dur = arith.subf %t9_end, %t9_start : f64
    %str9 = llvm.mlir.addressof @str_down_proj : !llvm.ptr
    func.call @record_timing(%str9, %t9_dur) : (!llvm.ptr, f64) -> ()

    return %96 : tensor<1x40x1536xf32>
  }
}
