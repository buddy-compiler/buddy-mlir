module {
  func.func @forward(%arg0: tensor<30522x768xf32>, %arg1: tensor<2x768xf32>, %arg2: tensor<512x768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<768xf32>, %arg5: tensor<768x768xf32>, %arg6: tensor<768xf32>, %arg7: tensor<768x768xf32>, %arg8: tensor<768xf32>, %arg9: tensor<768x768xf32>, %arg10: tensor<768xf32>, %arg11: tensor<768x768xf32>, %arg12: tensor<768xf32>, %arg13: tensor<768xf32>, %arg14: tensor<768xf32>, %arg15: tensor<3072x768xf32>, %arg16: tensor<3072xf32>, %arg17: tensor<768x3072xf32>, %arg18: tensor<768xf32>, %arg19: tensor<768xf32>, %arg20: tensor<768xf32>, %arg21: tensor<768x768xf32>, %arg22: tensor<768xf32>, %arg23: tensor<768x768xf32>, %arg24: tensor<768xf32>, %arg25: tensor<768x768xf32>, %arg26: tensor<768xf32>, %arg27: tensor<768x768xf32>, %arg28: tensor<768xf32>, %arg29: tensor<768xf32>, %arg30: tensor<768xf32>, %arg31: tensor<3072x768xf32>, %arg32: tensor<3072xf32>, %arg33: tensor<768x3072xf32>, %arg34: tensor<768xf32>, %arg35: tensor<768xf32>, %arg36: tensor<768xf32>, %arg37: tensor<768x768xf32>, %arg38: tensor<768xf32>, %arg39: tensor<768x768xf32>, %arg40: tensor<768xf32>, %arg41: tensor<768x768xf32>, %arg42: tensor<768xf32>, %arg43: tensor<768x768xf32>, %arg44: tensor<768xf32>, %arg45: tensor<768xf32>, %arg46: tensor<768xf32>, %arg47: tensor<3072x768xf32>, %arg48: tensor<3072xf32>, %arg49: tensor<768x3072xf32>, %arg50: tensor<768xf32>, %arg51: tensor<768xf32>, %arg52: tensor<768xf32>, %arg53: tensor<768x768xf32>, %arg54: tensor<768xf32>, %arg55: tensor<768x768xf32>, %arg56: tensor<768xf32>, %arg57: tensor<768x768xf32>, %arg58: tensor<768xf32>, %arg59: tensor<768x768xf32>, %arg60: tensor<768xf32>, %arg61: tensor<768xf32>, %arg62: tensor<768xf32>, %arg63: tensor<3072x768xf32>, %arg64: tensor<3072xf32>, %arg65: tensor<768x3072xf32>, %arg66: tensor<768xf32>, %arg67: tensor<768xf32>, %arg68: tensor<768xf32>, %arg69: tensor<768x768xf32>, %arg70: tensor<768xf32>, %arg71: tensor<768x768xf32>, %arg72: tensor<768xf32>, %arg73: tensor<768x768xf32>, %arg74: tensor<768xf32>, %arg75: tensor<768x768xf32>, %arg76: tensor<768xf32>, %arg77: tensor<768xf32>, %arg78: tensor<768xf32>, %arg79: tensor<3072x768xf32>, %arg80: tensor<3072xf32>, %arg81: tensor<768x3072xf32>, %arg82: tensor<768xf32>, %arg83: tensor<768xf32>, %arg84: tensor<768xf32>, %arg85: tensor<768x768xf32>, %arg86: tensor<768xf32>, %arg87: tensor<768x768xf32>, %arg88: tensor<768xf32>, %arg89: tensor<768x768xf32>, %arg90: tensor<768xf32>, %arg91: tensor<768x768xf32>, %arg92: tensor<768xf32>, %arg93: tensor<768xf32>, %arg94: tensor<768xf32>, %arg95: tensor<3072x768xf32>, %arg96: tensor<3072xf32>, %arg97: tensor<768x3072xf32>, %arg98: tensor<768xf32>, %arg99: tensor<768xf32>, %arg100: tensor<768xf32>, %arg101: tensor<768x768xf32>, %arg102: tensor<768xf32>, %arg103: tensor<768x768xf32>, %arg104: tensor<768xf32>, %arg105: tensor<768x768xf32>, %arg106: tensor<768xf32>, %arg107: tensor<768x768xf32>, %arg108: tensor<768xf32>, %arg109: tensor<768xf32>, %arg110: tensor<768xf32>, %arg111: tensor<3072x768xf32>, %arg112: tensor<3072xf32>, %arg113: tensor<768x3072xf32>, %arg114: tensor<768xf32>, %arg115: tensor<768xf32>, %arg116: tensor<768xf32>, %arg117: tensor<768x768xf32>, %arg118: tensor<768xf32>, %arg119: tensor<768x768xf32>, %arg120: tensor<768xf32>, %arg121: tensor<768x768xf32>, %arg122: tensor<768xf32>, %arg123: tensor<768x768xf32>, %arg124: tensor<768xf32>, %arg125: tensor<768xf32>, %arg126: tensor<768xf32>, %arg127: tensor<3072x768xf32>, %arg128: tensor<3072xf32>, %arg129: tensor<768x3072xf32>, %arg130: tensor<768xf32>, %arg131: tensor<768xf32>, %arg132: tensor<768xf32>, %arg133: tensor<768x768xf32>, %arg134: tensor<768xf32>, %arg135: tensor<768x768xf32>, %arg136: tensor<768xf32>, %arg137: tensor<768x768xf32>, %arg138: tensor<768xf32>, %arg139: tensor<768x768xf32>, %arg140: tensor<768xf32>, %arg141: tensor<768xf32>, %arg142: tensor<768xf32>, %arg143: tensor<3072x768xf32>, %arg144: tensor<3072xf32>, %arg145: tensor<768x3072xf32>, %arg146: tensor<768xf32>, %arg147: tensor<768xf32>, %arg148: tensor<768xf32>, %arg149: tensor<768x768xf32>, %arg150: tensor<768xf32>, %arg151: tensor<768x768xf32>, %arg152: tensor<768xf32>, %arg153: tensor<768x768xf32>, %arg154: tensor<768xf32>, %arg155: tensor<768x768xf32>, %arg156: tensor<768xf32>, %arg157: tensor<768xf32>, %arg158: tensor<768xf32>, %arg159: tensor<3072x768xf32>, %arg160: tensor<3072xf32>, %arg161: tensor<768x3072xf32>, %arg162: tensor<768xf32>, %arg163: tensor<768xf32>, %arg164: tensor<768xf32>, %arg165: tensor<768x768xf32>, %arg166: tensor<768xf32>, %arg167: tensor<768x768xf32>, %arg168: tensor<768xf32>, %arg169: tensor<768x768xf32>, %arg170: tensor<768xf32>, %arg171: tensor<768x768xf32>, %arg172: tensor<768xf32>, %arg173: tensor<768xf32>, %arg174: tensor<768xf32>, %arg175: tensor<3072x768xf32>, %arg176: tensor<3072xf32>, %arg177: tensor<768x3072xf32>, %arg178: tensor<768xf32>, %arg179: tensor<768xf32>, %arg180: tensor<768xf32>, %arg181: tensor<768x768xf32>, %arg182: tensor<768xf32>, %arg183: tensor<768x768xf32>, %arg184: tensor<768xf32>, %arg185: tensor<768x768xf32>, %arg186: tensor<768xf32>, %arg187: tensor<768x768xf32>, %arg188: tensor<768xf32>, %arg189: tensor<768xf32>, %arg190: tensor<768xf32>, %arg191: tensor<3072x768xf32>, %arg192: tensor<3072xf32>, %arg193: tensor<768x3072xf32>, %arg194: tensor<768xf32>, %arg195: tensor<768xf32>, %arg196: tensor<768xf32>, %arg197: tensor<768x768xf32>, %arg198: tensor<768xf32>, %arg199: tensor<6x768xf32>, %arg200: tensor<6xf32>, %arg201: tensor<1x512xi64>, %arg202: tensor<1x5xi64>, %arg203: tensor<1x5xi64>, %arg204: tensor<1x5xi64>) -> tensor<1x6xf32> {
    %extracted_slice = tensor.extract_slice %arg203[0, 0] [1, 5] [1, 1] : tensor<1x5xi64> to tensor<1x5xi64>
    %0 = tosa.reshape %extracted_slice {new_shape = array<i64: 1, 1, 5>} : (tensor<1x5xi64>) -> tensor<1x1x5xi64>
    %1 = tosa.reshape %0 {new_shape = array<i64: 1, 1, 1, 5>} : (tensor<1x1x5xi64>) -> tensor<1x1x1x5xi64>
    %extracted_slice_0 = tensor.extract_slice %1[0, 0, 0, 0] [1, 1, 1, 5] [1, 1, 1, 1] : tensor<1x1x1x5xi64> to tensor<1x1x1x5xi64>
    %2 = tosa.cast %extracted_slice_0 : (tensor<1x1x1x5xi64>) -> tensor<1x1x1x5xf32>
    %3 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1x1x5xf32>}> : () -> tensor<1x1x1x5xf32>
    %4 = tosa.sub %3, %2 : (tensor<1x1x1x5xf32>, tensor<1x1x1x5xf32>) -> tensor<1x1x1x5xf32>
    %5 = "tosa.const"() <{value = dense<-3.40282347E+38> : tensor<1x1x1x5xf32>}> : () -> tensor<1x1x1x5xf32>
    %6 = tosa.mul %4, %5 {shift = 0 : i8} : (tensor<1x1x1x5xf32>, tensor<1x1x1x5xf32>) -> tensor<1x1x1x5xf32>
    %extracted_slice_1 = tensor.extract_slice %arg201[0, 0] [1, 512] [1, 1] : tensor<1x512xi64> to tensor<1x512xi64>
    %extracted_slice_2 = tensor.extract_slice %extracted_slice_1[0, 0] [1, 5] [1, 1] : tensor<1x512xi64> to tensor<1x5xi64>
    %7 = tosa.cast %arg202 : (tensor<1x5xi64>) -> tensor<1x5xi32>
    %8 = tosa.reshape %arg0 {new_shape = array<i64: 1, 30522, 768>} : (tensor<30522x768xf32>) -> tensor<1x30522x768xf32>
    %9 = tosa.gather %8, %7 : (tensor<1x30522x768xf32>, tensor<1x5xi32>) -> tensor<1x5x768xf32>
    %10 = tosa.reshape %9 {new_shape = array<i64: 1, 5, 768>} : (tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %11 = tosa.cast %arg204 : (tensor<1x5xi64>) -> tensor<1x5xi32>
    %12 = tosa.reshape %arg1 {new_shape = array<i64: 1, 2, 768>} : (tensor<2x768xf32>) -> tensor<1x2x768xf32>
    %13 = tosa.gather %12, %11 : (tensor<1x2x768xf32>, tensor<1x5xi32>) -> tensor<1x5x768xf32>
    %14 = tosa.reshape %13 {new_shape = array<i64: 1, 5, 768>} : (tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %15 = tosa.add %10, %14 : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %16 = tosa.cast %extracted_slice_2 : (tensor<1x5xi64>) -> tensor<1x5xi32>
    %17 = tosa.reshape %arg2 {new_shape = array<i64: 1, 512, 768>} : (tensor<512x768xf32>) -> tensor<1x512x768xf32>
    %18 = tosa.gather %17, %16 : (tensor<1x512x768xf32>, tensor<1x5xi32>) -> tensor<1x5x768xf32>
    %19 = tosa.reshape %18 {new_shape = array<i64: 1, 5, 768>} : (tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %20 = tosa.add %15, %19 : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %21 = tosa.reduce_sum %20 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %22 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %23 = tosa.reciprocal %22 : (tensor<1xf32>) -> tensor<1xf32>
    %24 = tosa.mul %23, %21 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %25 = tosa.sub %20, %24 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %26 = tosa.mul %25, %25 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %27 = tosa.reduce_sum %26 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %28 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %29 = tosa.reciprocal %28 : (tensor<1xf32>) -> tensor<1xf32>
    %30 = tosa.mul %29, %27 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %31 = "tosa.const"() <{value = dense<9.99999996E-13> : tensor<1x5x1xf32>}> : () -> tensor<1x5x1xf32>
    %32 = tosa.add %30, %31 : (tensor<1x5x1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %33 = tosa.rsqrt %32 : (tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %34 = tosa.sub %20, %24 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %35 = tosa.mul %34, %33 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %36 = tosa.reshape %arg3 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %37 = tosa.mul %35, %36 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %38 = tosa.reshape %arg4 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %39 = tosa.add %37, %38 : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %40 = tosa.identity %39 : (tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %41 = tosa.reshape %40 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %42 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %43 = tosa.transpose %arg5, %42 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %44 = tosa.reshape %41 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %45 = tosa.reshape %43 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %46 = tosa.matmul %44, %45 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %47 = tosa.reshape %46 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %48 = tosa.reshape %arg6 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %49 = tosa.add %48, %47 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %50 = tosa.reshape %49 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %51 = tosa.reshape %40 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %52 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %53 = tosa.transpose %arg7, %52 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %54 = tosa.reshape %51 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %55 = tosa.reshape %53 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %56 = tosa.matmul %54, %55 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %57 = tosa.reshape %56 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %58 = tosa.reshape %arg8 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %59 = tosa.add %58, %57 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %60 = tosa.reshape %59 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %61 = tosa.reshape %60 {new_shape = array<i64: 1, 5, 12, 64>} : (tensor<1x5x768xf32>) -> tensor<1x5x12x64xf32>
    %62 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %63 = tosa.transpose %61, %62 : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x12x5x64xf32>
    %64 = tosa.reshape %40 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %65 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %66 = tosa.transpose %arg9, %65 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %67 = tosa.reshape %64 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %68 = tosa.reshape %66 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %69 = tosa.matmul %67, %68 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %70 = tosa.reshape %69 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %71 = tosa.reshape %arg10 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %72 = tosa.add %71, %70 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %73 = tosa.reshape %72 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %74 = tosa.reshape %73 {new_shape = array<i64: 1, 5, 12, 64>} : (tensor<1x5x768xf32>) -> tensor<1x5x12x64xf32>
    %75 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %76 = tosa.transpose %74, %75 : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x12x5x64xf32>
    %77 = tosa.reshape %50 {new_shape = array<i64: 1, 5, 12, 64>} : (tensor<1x5x768xf32>) -> tensor<1x5x12x64xf32>
    %78 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %79 = tosa.transpose %77, %78 : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x12x5x64xf32>
    %80 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %81 = tosa.transpose %63, %80 : (tensor<1x12x5x64xf32>, tensor<4xi32>) -> tensor<1x12x64x5xf32>
    %82 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x5x64xf32>}> : () -> tensor<1x12x5x64xf32>
    %83 = tosa.add %79, %82 : (tensor<1x12x5x64xf32>, tensor<1x12x5x64xf32>) -> tensor<1x12x5x64xf32>
    %84 = tosa.reshape %83 {new_shape = array<i64: 12, 5, 64>} : (tensor<1x12x5x64xf32>) -> tensor<12x5x64xf32>
    %85 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x64x5xf32>}> : () -> tensor<1x12x64x5xf32>
    %86 = tosa.add %81, %85 : (tensor<1x12x64x5xf32>, tensor<1x12x64x5xf32>) -> tensor<1x12x64x5xf32>
    %87 = tosa.reshape %86 {new_shape = array<i64: 12, 64, 5>} : (tensor<1x12x64x5xf32>) -> tensor<12x64x5xf32>
    %88 = tosa.matmul %84, %87 : (tensor<12x5x64xf32>, tensor<12x64x5xf32>) -> tensor<12x5x5xf32>
    %89 = tosa.reshape %88 {new_shape = array<i64: 1, 12, 5, 5>} : (tensor<12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %90 = "tosa.const"() <{value = dense<8.000000e+00> : tensor<1x12x5x5xf32>}> : () -> tensor<1x12x5x5xf32>
    %91 = tosa.reciprocal %90 : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %92 = tosa.mul %89, %91 {shift = 0 : i8} : (tensor<1x12x5x5xf32>, tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %93 = tosa.add %92, %6 : (tensor<1x12x5x5xf32>, tensor<1x1x1x5xf32>) -> tensor<1x12x5x5xf32>
    %94 = tosa.reduce_max %93 {axis = 3 : i32} : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x1xf32>
    %95 = tosa.sub %93, %94 : (tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) -> tensor<1x12x5x5xf32>
    %96 = tosa.exp %95 : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %97 = tosa.reduce_sum %96 {axis = 3 : i32} : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x1xf32>
    %98 = tosa.reciprocal %97 : (tensor<1x12x5x1xf32>) -> tensor<1x12x5x1xf32>
    %99 = tosa.mul %96, %98 {shift = 0 : i8} : (tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) -> tensor<1x12x5x5xf32>
    %100 = tosa.identity %99 : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %101 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x5x5xf32>}> : () -> tensor<1x12x5x5xf32>
    %102 = tosa.add %100, %101 : (tensor<1x12x5x5xf32>, tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %103 = tosa.reshape %102 {new_shape = array<i64: 12, 5, 5>} : (tensor<1x12x5x5xf32>) -> tensor<12x5x5xf32>
    %104 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x5x64xf32>}> : () -> tensor<1x12x5x64xf32>
    %105 = tosa.add %76, %104 : (tensor<1x12x5x64xf32>, tensor<1x12x5x64xf32>) -> tensor<1x12x5x64xf32>
    %106 = tosa.reshape %105 {new_shape = array<i64: 12, 5, 64>} : (tensor<1x12x5x64xf32>) -> tensor<12x5x64xf32>
    %107 = tosa.matmul %103, %106 : (tensor<12x5x5xf32>, tensor<12x5x64xf32>) -> tensor<12x5x64xf32>
    %108 = tosa.reshape %107 {new_shape = array<i64: 1, 12, 5, 64>} : (tensor<12x5x64xf32>) -> tensor<1x12x5x64xf32>
    %109 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %110 = tosa.transpose %108, %109 : (tensor<1x12x5x64xf32>, tensor<4xi32>) -> tensor<1x5x12x64xf32>
    %111 = tosa.identity %110 : (tensor<1x5x12x64xf32>) -> tensor<1x5x12x64xf32>
    %112 = tosa.reshape %111 {new_shape = array<i64: 1, 5, 768>} : (tensor<1x5x12x64xf32>) -> tensor<1x5x768xf32>
    %113 = tosa.reshape %112 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %114 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %115 = tosa.transpose %arg11, %114 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %116 = tosa.reshape %113 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %117 = tosa.reshape %115 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %118 = tosa.matmul %116, %117 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %119 = tosa.reshape %118 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %120 = tosa.reshape %arg12 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %121 = tosa.add %120, %119 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %122 = tosa.reshape %121 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %123 = tosa.identity %122 : (tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %124 = tosa.add %123, %40 : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %125 = tosa.reduce_sum %124 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %126 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %127 = tosa.reciprocal %126 : (tensor<1xf32>) -> tensor<1xf32>
    %128 = tosa.mul %127, %125 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %129 = tosa.sub %124, %128 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %130 = tosa.mul %129, %129 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %131 = tosa.reduce_sum %130 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %132 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %133 = tosa.reciprocal %132 : (tensor<1xf32>) -> tensor<1xf32>
    %134 = tosa.mul %133, %131 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %135 = "tosa.const"() <{value = dense<9.99999996E-13> : tensor<1x5x1xf32>}> : () -> tensor<1x5x1xf32>
    %136 = tosa.add %134, %135 : (tensor<1x5x1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %137 = tosa.rsqrt %136 : (tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %138 = tosa.sub %124, %128 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %139 = tosa.mul %138, %137 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %140 = tosa.reshape %arg13 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %141 = tosa.mul %139, %140 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %142 = tosa.reshape %arg14 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %143 = tosa.add %141, %142 : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %144 = tosa.reshape %143 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %145 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %146 = tosa.transpose %arg15, %145 : (tensor<3072x768xf32>, tensor<2xi32>) -> tensor<768x3072xf32>
    %147 = tosa.reshape %144 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %148 = tosa.reshape %146 {new_shape = array<i64: 1, 768, 3072>} : (tensor<768x3072xf32>) -> tensor<1x768x3072xf32>
    %149 = tosa.matmul %147, %148 : (tensor<1x5x768xf32>, tensor<1x768x3072xf32>) -> tensor<1x5x3072xf32>
    %150 = tosa.reshape %149 {new_shape = array<i64: 5, 3072>} : (tensor<1x5x3072xf32>) -> tensor<5x3072xf32>
    %151 = tosa.reshape %arg16 {new_shape = array<i64: 1, 3072>} : (tensor<3072xf32>) -> tensor<1x3072xf32>
    %152 = tosa.add %151, %150 : (tensor<1x3072xf32>, tensor<5x3072xf32>) -> tensor<5x3072xf32>
    %153 = tosa.reshape %152 {new_shape = array<i64: 1, 5, 3072>} : (tensor<5x3072xf32>) -> tensor<1x5x3072xf32>
    %154 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x5x3072xf32>}> : () -> tensor<1x5x3072xf32>
    %155 = tosa.mul %153, %154 {shift = 0 : i8} : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %156 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x5x3072xf32>}> : () -> tensor<1x5x3072xf32>
    %157 = tosa.mul %153, %156 {shift = 0 : i8} : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %158 = math.erf %157 : tensor<1x5x3072xf32>
    %159 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x5x3072xf32>}> : () -> tensor<1x5x3072xf32>
    %160 = tosa.add %158, %159 : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %161 = tosa.mul %155, %160 {shift = 0 : i8} : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %162 = tosa.reshape %161 {new_shape = array<i64: 5, 3072>} : (tensor<1x5x3072xf32>) -> tensor<5x3072xf32>
    %163 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %164 = tosa.transpose %arg17, %163 : (tensor<768x3072xf32>, tensor<2xi32>) -> tensor<3072x768xf32>
    %165 = tosa.reshape %162 {new_shape = array<i64: 1, 5, 3072>} : (tensor<5x3072xf32>) -> tensor<1x5x3072xf32>
    %166 = tosa.reshape %164 {new_shape = array<i64: 1, 3072, 768>} : (tensor<3072x768xf32>) -> tensor<1x3072x768xf32>
    %167 = tosa.matmul %165, %166 : (tensor<1x5x3072xf32>, tensor<1x3072x768xf32>) -> tensor<1x5x768xf32>
    %168 = tosa.reshape %167 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %169 = tosa.reshape %arg18 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %170 = tosa.add %169, %168 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %171 = tosa.reshape %170 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %172 = tosa.identity %171 : (tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %173 = tosa.add %172, %143 : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %174 = tosa.reduce_sum %173 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %175 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %176 = tosa.reciprocal %175 : (tensor<1xf32>) -> tensor<1xf32>
    %177 = tosa.mul %176, %174 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %178 = tosa.sub %173, %177 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %179 = tosa.mul %178, %178 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %180 = tosa.reduce_sum %179 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %181 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %182 = tosa.reciprocal %181 : (tensor<1xf32>) -> tensor<1xf32>
    %183 = tosa.mul %182, %180 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %184 = "tosa.const"() <{value = dense<9.99999996E-13> : tensor<1x5x1xf32>}> : () -> tensor<1x5x1xf32>
    %185 = tosa.add %183, %184 : (tensor<1x5x1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %186 = tosa.rsqrt %185 : (tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %187 = tosa.sub %173, %177 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %188 = tosa.mul %187, %186 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %189 = tosa.reshape %arg19 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %190 = tosa.mul %188, %189 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %191 = tosa.reshape %arg20 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %192 = tosa.add %190, %191 : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %193 = tosa.reshape %192 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %194 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %195 = tosa.transpose %arg21, %194 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %196 = tosa.reshape %193 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %197 = tosa.reshape %195 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %198 = tosa.matmul %196, %197 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %199 = tosa.reshape %198 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %200 = tosa.reshape %arg22 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %201 = tosa.add %200, %199 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %202 = tosa.reshape %201 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %203 = tosa.reshape %192 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %204 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %205 = tosa.transpose %arg23, %204 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %206 = tosa.reshape %203 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %207 = tosa.reshape %205 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %208 = tosa.matmul %206, %207 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %209 = tosa.reshape %208 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %210 = tosa.reshape %arg24 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %211 = tosa.add %210, %209 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %212 = tosa.reshape %211 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %213 = tosa.reshape %212 {new_shape = array<i64: 1, 5, 12, 64>} : (tensor<1x5x768xf32>) -> tensor<1x5x12x64xf32>
    %214 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %215 = tosa.transpose %213, %214 : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x12x5x64xf32>
    %216 = tosa.reshape %192 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %217 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %218 = tosa.transpose %arg25, %217 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %219 = tosa.reshape %216 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %220 = tosa.reshape %218 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %221 = tosa.matmul %219, %220 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %222 = tosa.reshape %221 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %223 = tosa.reshape %arg26 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %224 = tosa.add %223, %222 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %225 = tosa.reshape %224 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %226 = tosa.reshape %225 {new_shape = array<i64: 1, 5, 12, 64>} : (tensor<1x5x768xf32>) -> tensor<1x5x12x64xf32>
    %227 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %228 = tosa.transpose %226, %227 : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x12x5x64xf32>
    %229 = tosa.reshape %202 {new_shape = array<i64: 1, 5, 12, 64>} : (tensor<1x5x768xf32>) -> tensor<1x5x12x64xf32>
    %230 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %231 = tosa.transpose %229, %230 : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x12x5x64xf32>
    %232 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %233 = tosa.transpose %215, %232 : (tensor<1x12x5x64xf32>, tensor<4xi32>) -> tensor<1x12x64x5xf32>
    %234 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x5x64xf32>}> : () -> tensor<1x12x5x64xf32>
    %235 = tosa.add %231, %234 : (tensor<1x12x5x64xf32>, tensor<1x12x5x64xf32>) -> tensor<1x12x5x64xf32>
    %236 = tosa.reshape %235 {new_shape = array<i64: 12, 5, 64>} : (tensor<1x12x5x64xf32>) -> tensor<12x5x64xf32>
    %237 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x64x5xf32>}> : () -> tensor<1x12x64x5xf32>
    %238 = tosa.add %233, %237 : (tensor<1x12x64x5xf32>, tensor<1x12x64x5xf32>) -> tensor<1x12x64x5xf32>
    %239 = tosa.reshape %238 {new_shape = array<i64: 12, 64, 5>} : (tensor<1x12x64x5xf32>) -> tensor<12x64x5xf32>
    %240 = tosa.matmul %236, %239 : (tensor<12x5x64xf32>, tensor<12x64x5xf32>) -> tensor<12x5x5xf32>
    %241 = tosa.reshape %240 {new_shape = array<i64: 1, 12, 5, 5>} : (tensor<12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %242 = "tosa.const"() <{value = dense<8.000000e+00> : tensor<1x12x5x5xf32>}> : () -> tensor<1x12x5x5xf32>
    %243 = tosa.reciprocal %242 : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %244 = tosa.mul %241, %243 {shift = 0 : i8} : (tensor<1x12x5x5xf32>, tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %245 = tosa.add %244, %6 : (tensor<1x12x5x5xf32>, tensor<1x1x1x5xf32>) -> tensor<1x12x5x5xf32>
    %246 = tosa.reduce_max %245 {axis = 3 : i32} : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x1xf32>
    %247 = tosa.sub %245, %246 : (tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) -> tensor<1x12x5x5xf32>
    %248 = tosa.exp %247 : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %249 = tosa.reduce_sum %248 {axis = 3 : i32} : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x1xf32>
    %250 = tosa.reciprocal %249 : (tensor<1x12x5x1xf32>) -> tensor<1x12x5x1xf32>
    %251 = tosa.mul %248, %250 {shift = 0 : i8} : (tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) -> tensor<1x12x5x5xf32>
    %252 = tosa.identity %251 : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %253 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x5x5xf32>}> : () -> tensor<1x12x5x5xf32>
    %254 = tosa.add %252, %253 : (tensor<1x12x5x5xf32>, tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %255 = tosa.reshape %254 {new_shape = array<i64: 12, 5, 5>} : (tensor<1x12x5x5xf32>) -> tensor<12x5x5xf32>
    %256 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x5x64xf32>}> : () -> tensor<1x12x5x64xf32>
    %257 = tosa.add %228, %256 : (tensor<1x12x5x64xf32>, tensor<1x12x5x64xf32>) -> tensor<1x12x5x64xf32>
    %258 = tosa.reshape %257 {new_shape = array<i64: 12, 5, 64>} : (tensor<1x12x5x64xf32>) -> tensor<12x5x64xf32>
    %259 = tosa.matmul %255, %258 : (tensor<12x5x5xf32>, tensor<12x5x64xf32>) -> tensor<12x5x64xf32>
    %260 = tosa.reshape %259 {new_shape = array<i64: 1, 12, 5, 64>} : (tensor<12x5x64xf32>) -> tensor<1x12x5x64xf32>
    %261 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %262 = tosa.transpose %260, %261 : (tensor<1x12x5x64xf32>, tensor<4xi32>) -> tensor<1x5x12x64xf32>
    %263 = tosa.identity %262 : (tensor<1x5x12x64xf32>) -> tensor<1x5x12x64xf32>
    %264 = tosa.reshape %263 {new_shape = array<i64: 1, 5, 768>} : (tensor<1x5x12x64xf32>) -> tensor<1x5x768xf32>
    %265 = tosa.reshape %264 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %266 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %267 = tosa.transpose %arg27, %266 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %268 = tosa.reshape %265 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %269 = tosa.reshape %267 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %270 = tosa.matmul %268, %269 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %271 = tosa.reshape %270 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %272 = tosa.reshape %arg28 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %273 = tosa.add %272, %271 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %274 = tosa.reshape %273 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %275 = tosa.identity %274 : (tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %276 = tosa.add %275, %192 : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %277 = tosa.reduce_sum %276 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %278 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %279 = tosa.reciprocal %278 : (tensor<1xf32>) -> tensor<1xf32>
    %280 = tosa.mul %279, %277 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %281 = tosa.sub %276, %280 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %282 = tosa.mul %281, %281 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %283 = tosa.reduce_sum %282 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %284 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %285 = tosa.reciprocal %284 : (tensor<1xf32>) -> tensor<1xf32>
    %286 = tosa.mul %285, %283 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %287 = "tosa.const"() <{value = dense<9.99999996E-13> : tensor<1x5x1xf32>}> : () -> tensor<1x5x1xf32>
    %288 = tosa.add %286, %287 : (tensor<1x5x1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %289 = tosa.rsqrt %288 : (tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %290 = tosa.sub %276, %280 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %291 = tosa.mul %290, %289 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %292 = tosa.reshape %arg29 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %293 = tosa.mul %291, %292 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %294 = tosa.reshape %arg30 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %295 = tosa.add %293, %294 : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %296 = tosa.reshape %295 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %297 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %298 = tosa.transpose %arg31, %297 : (tensor<3072x768xf32>, tensor<2xi32>) -> tensor<768x3072xf32>
    %299 = tosa.reshape %296 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %300 = tosa.reshape %298 {new_shape = array<i64: 1, 768, 3072>} : (tensor<768x3072xf32>) -> tensor<1x768x3072xf32>
    %301 = tosa.matmul %299, %300 : (tensor<1x5x768xf32>, tensor<1x768x3072xf32>) -> tensor<1x5x3072xf32>
    %302 = tosa.reshape %301 {new_shape = array<i64: 5, 3072>} : (tensor<1x5x3072xf32>) -> tensor<5x3072xf32>
    %303 = tosa.reshape %arg32 {new_shape = array<i64: 1, 3072>} : (tensor<3072xf32>) -> tensor<1x3072xf32>
    %304 = tosa.add %303, %302 : (tensor<1x3072xf32>, tensor<5x3072xf32>) -> tensor<5x3072xf32>
    %305 = tosa.reshape %304 {new_shape = array<i64: 1, 5, 3072>} : (tensor<5x3072xf32>) -> tensor<1x5x3072xf32>
    %306 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x5x3072xf32>}> : () -> tensor<1x5x3072xf32>
    %307 = tosa.mul %305, %306 {shift = 0 : i8} : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %308 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x5x3072xf32>}> : () -> tensor<1x5x3072xf32>
    %309 = tosa.mul %305, %308 {shift = 0 : i8} : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %310 = math.erf %309 : tensor<1x5x3072xf32>
    %311 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x5x3072xf32>}> : () -> tensor<1x5x3072xf32>
    %312 = tosa.add %310, %311 : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %313 = tosa.mul %307, %312 {shift = 0 : i8} : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %314 = tosa.reshape %313 {new_shape = array<i64: 5, 3072>} : (tensor<1x5x3072xf32>) -> tensor<5x3072xf32>
    %315 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %316 = tosa.transpose %arg33, %315 : (tensor<768x3072xf32>, tensor<2xi32>) -> tensor<3072x768xf32>
    %317 = tosa.reshape %314 {new_shape = array<i64: 1, 5, 3072>} : (tensor<5x3072xf32>) -> tensor<1x5x3072xf32>
    %318 = tosa.reshape %316 {new_shape = array<i64: 1, 3072, 768>} : (tensor<3072x768xf32>) -> tensor<1x3072x768xf32>
    %319 = tosa.matmul %317, %318 : (tensor<1x5x3072xf32>, tensor<1x3072x768xf32>) -> tensor<1x5x768xf32>
    %320 = tosa.reshape %319 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %321 = tosa.reshape %arg34 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %322 = tosa.add %321, %320 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %323 = tosa.reshape %322 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %324 = tosa.identity %323 : (tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %325 = tosa.add %324, %295 : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %326 = tosa.reduce_sum %325 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %327 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %328 = tosa.reciprocal %327 : (tensor<1xf32>) -> tensor<1xf32>
    %329 = tosa.mul %328, %326 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %330 = tosa.sub %325, %329 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %331 = tosa.mul %330, %330 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %332 = tosa.reduce_sum %331 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %333 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %334 = tosa.reciprocal %333 : (tensor<1xf32>) -> tensor<1xf32>
    %335 = tosa.mul %334, %332 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %336 = "tosa.const"() <{value = dense<9.99999996E-13> : tensor<1x5x1xf32>}> : () -> tensor<1x5x1xf32>
    %337 = tosa.add %335, %336 : (tensor<1x5x1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %338 = tosa.rsqrt %337 : (tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %339 = tosa.sub %325, %329 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %340 = tosa.mul %339, %338 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %341 = tosa.reshape %arg35 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %342 = tosa.mul %340, %341 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %343 = tosa.reshape %arg36 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %344 = tosa.add %342, %343 : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %345 = tosa.reshape %344 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %346 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %347 = tosa.transpose %arg37, %346 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %348 = tosa.reshape %345 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %349 = tosa.reshape %347 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %350 = tosa.matmul %348, %349 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %351 = tosa.reshape %350 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %352 = tosa.reshape %arg38 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %353 = tosa.add %352, %351 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %354 = tosa.reshape %353 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %355 = tosa.reshape %344 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %356 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %357 = tosa.transpose %arg39, %356 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %358 = tosa.reshape %355 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %359 = tosa.reshape %357 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %360 = tosa.matmul %358, %359 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %361 = tosa.reshape %360 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %362 = tosa.reshape %arg40 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %363 = tosa.add %362, %361 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %364 = tosa.reshape %363 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %365 = tosa.reshape %364 {new_shape = array<i64: 1, 5, 12, 64>} : (tensor<1x5x768xf32>) -> tensor<1x5x12x64xf32>
    %366 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %367 = tosa.transpose %365, %366 : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x12x5x64xf32>
    %368 = tosa.reshape %344 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %369 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %370 = tosa.transpose %arg41, %369 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %371 = tosa.reshape %368 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %372 = tosa.reshape %370 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %373 = tosa.matmul %371, %372 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %374 = tosa.reshape %373 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %375 = tosa.reshape %arg42 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %376 = tosa.add %375, %374 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %377 = tosa.reshape %376 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %378 = tosa.reshape %377 {new_shape = array<i64: 1, 5, 12, 64>} : (tensor<1x5x768xf32>) -> tensor<1x5x12x64xf32>
    %379 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %380 = tosa.transpose %378, %379 : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x12x5x64xf32>
    %381 = tosa.reshape %354 {new_shape = array<i64: 1, 5, 12, 64>} : (tensor<1x5x768xf32>) -> tensor<1x5x12x64xf32>
    %382 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %383 = tosa.transpose %381, %382 : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x12x5x64xf32>
    %384 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %385 = tosa.transpose %367, %384 : (tensor<1x12x5x64xf32>, tensor<4xi32>) -> tensor<1x12x64x5xf32>
    %386 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x5x64xf32>}> : () -> tensor<1x12x5x64xf32>
    %387 = tosa.add %383, %386 : (tensor<1x12x5x64xf32>, tensor<1x12x5x64xf32>) -> tensor<1x12x5x64xf32>
    %388 = tosa.reshape %387 {new_shape = array<i64: 12, 5, 64>} : (tensor<1x12x5x64xf32>) -> tensor<12x5x64xf32>
    %389 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x64x5xf32>}> : () -> tensor<1x12x64x5xf32>
    %390 = tosa.add %385, %389 : (tensor<1x12x64x5xf32>, tensor<1x12x64x5xf32>) -> tensor<1x12x64x5xf32>
    %391 = tosa.reshape %390 {new_shape = array<i64: 12, 64, 5>} : (tensor<1x12x64x5xf32>) -> tensor<12x64x5xf32>
    %392 = tosa.matmul %388, %391 : (tensor<12x5x64xf32>, tensor<12x64x5xf32>) -> tensor<12x5x5xf32>
    %393 = tosa.reshape %392 {new_shape = array<i64: 1, 12, 5, 5>} : (tensor<12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %394 = "tosa.const"() <{value = dense<8.000000e+00> : tensor<1x12x5x5xf32>}> : () -> tensor<1x12x5x5xf32>
    %395 = tosa.reciprocal %394 : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %396 = tosa.mul %393, %395 {shift = 0 : i8} : (tensor<1x12x5x5xf32>, tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %397 = tosa.add %396, %6 : (tensor<1x12x5x5xf32>, tensor<1x1x1x5xf32>) -> tensor<1x12x5x5xf32>
    %398 = tosa.reduce_max %397 {axis = 3 : i32} : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x1xf32>
    %399 = tosa.sub %397, %398 : (tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) -> tensor<1x12x5x5xf32>
    %400 = tosa.exp %399 : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %401 = tosa.reduce_sum %400 {axis = 3 : i32} : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x1xf32>
    %402 = tosa.reciprocal %401 : (tensor<1x12x5x1xf32>) -> tensor<1x12x5x1xf32>
    %403 = tosa.mul %400, %402 {shift = 0 : i8} : (tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) -> tensor<1x12x5x5xf32>
    %404 = tosa.identity %403 : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %405 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x5x5xf32>}> : () -> tensor<1x12x5x5xf32>
    %406 = tosa.add %404, %405 : (tensor<1x12x5x5xf32>, tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %407 = tosa.reshape %406 {new_shape = array<i64: 12, 5, 5>} : (tensor<1x12x5x5xf32>) -> tensor<12x5x5xf32>
    %408 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x5x64xf32>}> : () -> tensor<1x12x5x64xf32>
    %409 = tosa.add %380, %408 : (tensor<1x12x5x64xf32>, tensor<1x12x5x64xf32>) -> tensor<1x12x5x64xf32>
    %410 = tosa.reshape %409 {new_shape = array<i64: 12, 5, 64>} : (tensor<1x12x5x64xf32>) -> tensor<12x5x64xf32>
    %411 = tosa.matmul %407, %410 : (tensor<12x5x5xf32>, tensor<12x5x64xf32>) -> tensor<12x5x64xf32>
    %412 = tosa.reshape %411 {new_shape = array<i64: 1, 12, 5, 64>} : (tensor<12x5x64xf32>) -> tensor<1x12x5x64xf32>
    %413 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %414 = tosa.transpose %412, %413 : (tensor<1x12x5x64xf32>, tensor<4xi32>) -> tensor<1x5x12x64xf32>
    %415 = tosa.identity %414 : (tensor<1x5x12x64xf32>) -> tensor<1x5x12x64xf32>
    %416 = tosa.reshape %415 {new_shape = array<i64: 1, 5, 768>} : (tensor<1x5x12x64xf32>) -> tensor<1x5x768xf32>
    %417 = tosa.reshape %416 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %418 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %419 = tosa.transpose %arg43, %418 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %420 = tosa.reshape %417 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %421 = tosa.reshape %419 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %422 = tosa.matmul %420, %421 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %423 = tosa.reshape %422 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %424 = tosa.reshape %arg44 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %425 = tosa.add %424, %423 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %426 = tosa.reshape %425 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %427 = tosa.identity %426 : (tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %428 = tosa.add %427, %344 : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %429 = tosa.reduce_sum %428 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %430 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %431 = tosa.reciprocal %430 : (tensor<1xf32>) -> tensor<1xf32>
    %432 = tosa.mul %431, %429 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %433 = tosa.sub %428, %432 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %434 = tosa.mul %433, %433 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %435 = tosa.reduce_sum %434 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %436 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %437 = tosa.reciprocal %436 : (tensor<1xf32>) -> tensor<1xf32>
    %438 = tosa.mul %437, %435 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %439 = "tosa.const"() <{value = dense<9.99999996E-13> : tensor<1x5x1xf32>}> : () -> tensor<1x5x1xf32>
    %440 = tosa.add %438, %439 : (tensor<1x5x1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %441 = tosa.rsqrt %440 : (tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %442 = tosa.sub %428, %432 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %443 = tosa.mul %442, %441 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %444 = tosa.reshape %arg45 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %445 = tosa.mul %443, %444 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %446 = tosa.reshape %arg46 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %447 = tosa.add %445, %446 : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %448 = tosa.reshape %447 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %449 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %450 = tosa.transpose %arg47, %449 : (tensor<3072x768xf32>, tensor<2xi32>) -> tensor<768x3072xf32>
    %451 = tosa.reshape %448 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %452 = tosa.reshape %450 {new_shape = array<i64: 1, 768, 3072>} : (tensor<768x3072xf32>) -> tensor<1x768x3072xf32>
    %453 = tosa.matmul %451, %452 : (tensor<1x5x768xf32>, tensor<1x768x3072xf32>) -> tensor<1x5x3072xf32>
    %454 = tosa.reshape %453 {new_shape = array<i64: 5, 3072>} : (tensor<1x5x3072xf32>) -> tensor<5x3072xf32>
    %455 = tosa.reshape %arg48 {new_shape = array<i64: 1, 3072>} : (tensor<3072xf32>) -> tensor<1x3072xf32>
    %456 = tosa.add %455, %454 : (tensor<1x3072xf32>, tensor<5x3072xf32>) -> tensor<5x3072xf32>
    %457 = tosa.reshape %456 {new_shape = array<i64: 1, 5, 3072>} : (tensor<5x3072xf32>) -> tensor<1x5x3072xf32>
    %458 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x5x3072xf32>}> : () -> tensor<1x5x3072xf32>
    %459 = tosa.mul %457, %458 {shift = 0 : i8} : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %460 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x5x3072xf32>}> : () -> tensor<1x5x3072xf32>
    %461 = tosa.mul %457, %460 {shift = 0 : i8} : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %462 = math.erf %461 : tensor<1x5x3072xf32>
    %463 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x5x3072xf32>}> : () -> tensor<1x5x3072xf32>
    %464 = tosa.add %462, %463 : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %465 = tosa.mul %459, %464 {shift = 0 : i8} : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %466 = tosa.reshape %465 {new_shape = array<i64: 5, 3072>} : (tensor<1x5x3072xf32>) -> tensor<5x3072xf32>
    %467 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %468 = tosa.transpose %arg49, %467 : (tensor<768x3072xf32>, tensor<2xi32>) -> tensor<3072x768xf32>
    %469 = tosa.reshape %466 {new_shape = array<i64: 1, 5, 3072>} : (tensor<5x3072xf32>) -> tensor<1x5x3072xf32>
    %470 = tosa.reshape %468 {new_shape = array<i64: 1, 3072, 768>} : (tensor<3072x768xf32>) -> tensor<1x3072x768xf32>
    %471 = tosa.matmul %469, %470 : (tensor<1x5x3072xf32>, tensor<1x3072x768xf32>) -> tensor<1x5x768xf32>
    %472 = tosa.reshape %471 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %473 = tosa.reshape %arg50 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %474 = tosa.add %473, %472 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %475 = tosa.reshape %474 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %476 = tosa.identity %475 : (tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %477 = tosa.add %476, %447 : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %478 = tosa.reduce_sum %477 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %479 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %480 = tosa.reciprocal %479 : (tensor<1xf32>) -> tensor<1xf32>
    %481 = tosa.mul %480, %478 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %482 = tosa.sub %477, %481 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %483 = tosa.mul %482, %482 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %484 = tosa.reduce_sum %483 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %485 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %486 = tosa.reciprocal %485 : (tensor<1xf32>) -> tensor<1xf32>
    %487 = tosa.mul %486, %484 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %488 = "tosa.const"() <{value = dense<9.99999996E-13> : tensor<1x5x1xf32>}> : () -> tensor<1x5x1xf32>
    %489 = tosa.add %487, %488 : (tensor<1x5x1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %490 = tosa.rsqrt %489 : (tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %491 = tosa.sub %477, %481 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %492 = tosa.mul %491, %490 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %493 = tosa.reshape %arg51 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %494 = tosa.mul %492, %493 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %495 = tosa.reshape %arg52 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %496 = tosa.add %494, %495 : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %497 = tosa.reshape %496 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %498 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %499 = tosa.transpose %arg53, %498 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %500 = tosa.reshape %497 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %501 = tosa.reshape %499 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %502 = tosa.matmul %500, %501 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %503 = tosa.reshape %502 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %504 = tosa.reshape %arg54 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %505 = tosa.add %504, %503 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %506 = tosa.reshape %505 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %507 = tosa.reshape %496 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %508 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %509 = tosa.transpose %arg55, %508 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %510 = tosa.reshape %507 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %511 = tosa.reshape %509 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %512 = tosa.matmul %510, %511 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %513 = tosa.reshape %512 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %514 = tosa.reshape %arg56 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %515 = tosa.add %514, %513 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %516 = tosa.reshape %515 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %517 = tosa.reshape %516 {new_shape = array<i64: 1, 5, 12, 64>} : (tensor<1x5x768xf32>) -> tensor<1x5x12x64xf32>
    %518 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %519 = tosa.transpose %517, %518 : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x12x5x64xf32>
    %520 = tosa.reshape %496 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %521 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %522 = tosa.transpose %arg57, %521 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %523 = tosa.reshape %520 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %524 = tosa.reshape %522 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %525 = tosa.matmul %523, %524 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %526 = tosa.reshape %525 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %527 = tosa.reshape %arg58 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %528 = tosa.add %527, %526 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %529 = tosa.reshape %528 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %530 = tosa.reshape %529 {new_shape = array<i64: 1, 5, 12, 64>} : (tensor<1x5x768xf32>) -> tensor<1x5x12x64xf32>
    %531 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %532 = tosa.transpose %530, %531 : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x12x5x64xf32>
    %533 = tosa.reshape %506 {new_shape = array<i64: 1, 5, 12, 64>} : (tensor<1x5x768xf32>) -> tensor<1x5x12x64xf32>
    %534 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %535 = tosa.transpose %533, %534 : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x12x5x64xf32>
    %536 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %537 = tosa.transpose %519, %536 : (tensor<1x12x5x64xf32>, tensor<4xi32>) -> tensor<1x12x64x5xf32>
    %538 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x5x64xf32>}> : () -> tensor<1x12x5x64xf32>
    %539 = tosa.add %535, %538 : (tensor<1x12x5x64xf32>, tensor<1x12x5x64xf32>) -> tensor<1x12x5x64xf32>
    %540 = tosa.reshape %539 {new_shape = array<i64: 12, 5, 64>} : (tensor<1x12x5x64xf32>) -> tensor<12x5x64xf32>
    %541 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x64x5xf32>}> : () -> tensor<1x12x64x5xf32>
    %542 = tosa.add %537, %541 : (tensor<1x12x64x5xf32>, tensor<1x12x64x5xf32>) -> tensor<1x12x64x5xf32>
    %543 = tosa.reshape %542 {new_shape = array<i64: 12, 64, 5>} : (tensor<1x12x64x5xf32>) -> tensor<12x64x5xf32>
    %544 = tosa.matmul %540, %543 : (tensor<12x5x64xf32>, tensor<12x64x5xf32>) -> tensor<12x5x5xf32>
    %545 = tosa.reshape %544 {new_shape = array<i64: 1, 12, 5, 5>} : (tensor<12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %546 = "tosa.const"() <{value = dense<8.000000e+00> : tensor<1x12x5x5xf32>}> : () -> tensor<1x12x5x5xf32>
    %547 = tosa.reciprocal %546 : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %548 = tosa.mul %545, %547 {shift = 0 : i8} : (tensor<1x12x5x5xf32>, tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %549 = tosa.add %548, %6 : (tensor<1x12x5x5xf32>, tensor<1x1x1x5xf32>) -> tensor<1x12x5x5xf32>
    %550 = tosa.reduce_max %549 {axis = 3 : i32} : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x1xf32>
    %551 = tosa.sub %549, %550 : (tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) -> tensor<1x12x5x5xf32>
    %552 = tosa.exp %551 : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %553 = tosa.reduce_sum %552 {axis = 3 : i32} : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x1xf32>
    %554 = tosa.reciprocal %553 : (tensor<1x12x5x1xf32>) -> tensor<1x12x5x1xf32>
    %555 = tosa.mul %552, %554 {shift = 0 : i8} : (tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) -> tensor<1x12x5x5xf32>
    %556 = tosa.identity %555 : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %557 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x5x5xf32>}> : () -> tensor<1x12x5x5xf32>
    %558 = tosa.add %556, %557 : (tensor<1x12x5x5xf32>, tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %559 = tosa.reshape %558 {new_shape = array<i64: 12, 5, 5>} : (tensor<1x12x5x5xf32>) -> tensor<12x5x5xf32>
    %560 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x5x64xf32>}> : () -> tensor<1x12x5x64xf32>
    %561 = tosa.add %532, %560 : (tensor<1x12x5x64xf32>, tensor<1x12x5x64xf32>) -> tensor<1x12x5x64xf32>
    %562 = tosa.reshape %561 {new_shape = array<i64: 12, 5, 64>} : (tensor<1x12x5x64xf32>) -> tensor<12x5x64xf32>
    %563 = tosa.matmul %559, %562 : (tensor<12x5x5xf32>, tensor<12x5x64xf32>) -> tensor<12x5x64xf32>
    %564 = tosa.reshape %563 {new_shape = array<i64: 1, 12, 5, 64>} : (tensor<12x5x64xf32>) -> tensor<1x12x5x64xf32>
    %565 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %566 = tosa.transpose %564, %565 : (tensor<1x12x5x64xf32>, tensor<4xi32>) -> tensor<1x5x12x64xf32>
    %567 = tosa.identity %566 : (tensor<1x5x12x64xf32>) -> tensor<1x5x12x64xf32>
    %568 = tosa.reshape %567 {new_shape = array<i64: 1, 5, 768>} : (tensor<1x5x12x64xf32>) -> tensor<1x5x768xf32>
    %569 = tosa.reshape %568 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %570 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %571 = tosa.transpose %arg59, %570 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %572 = tosa.reshape %569 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %573 = tosa.reshape %571 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %574 = tosa.matmul %572, %573 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %575 = tosa.reshape %574 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %576 = tosa.reshape %arg60 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %577 = tosa.add %576, %575 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %578 = tosa.reshape %577 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %579 = tosa.identity %578 : (tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %580 = tosa.add %579, %496 : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %581 = tosa.reduce_sum %580 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %582 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %583 = tosa.reciprocal %582 : (tensor<1xf32>) -> tensor<1xf32>
    %584 = tosa.mul %583, %581 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %585 = tosa.sub %580, %584 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %586 = tosa.mul %585, %585 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %587 = tosa.reduce_sum %586 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %588 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %589 = tosa.reciprocal %588 : (tensor<1xf32>) -> tensor<1xf32>
    %590 = tosa.mul %589, %587 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %591 = "tosa.const"() <{value = dense<9.99999996E-13> : tensor<1x5x1xf32>}> : () -> tensor<1x5x1xf32>
    %592 = tosa.add %590, %591 : (tensor<1x5x1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %593 = tosa.rsqrt %592 : (tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %594 = tosa.sub %580, %584 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %595 = tosa.mul %594, %593 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %596 = tosa.reshape %arg61 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %597 = tosa.mul %595, %596 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %598 = tosa.reshape %arg62 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %599 = tosa.add %597, %598 : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %600 = tosa.reshape %599 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %601 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %602 = tosa.transpose %arg63, %601 : (tensor<3072x768xf32>, tensor<2xi32>) -> tensor<768x3072xf32>
    %603 = tosa.reshape %600 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %604 = tosa.reshape %602 {new_shape = array<i64: 1, 768, 3072>} : (tensor<768x3072xf32>) -> tensor<1x768x3072xf32>
    %605 = tosa.matmul %603, %604 : (tensor<1x5x768xf32>, tensor<1x768x3072xf32>) -> tensor<1x5x3072xf32>
    %606 = tosa.reshape %605 {new_shape = array<i64: 5, 3072>} : (tensor<1x5x3072xf32>) -> tensor<5x3072xf32>
    %607 = tosa.reshape %arg64 {new_shape = array<i64: 1, 3072>} : (tensor<3072xf32>) -> tensor<1x3072xf32>
    %608 = tosa.add %607, %606 : (tensor<1x3072xf32>, tensor<5x3072xf32>) -> tensor<5x3072xf32>
    %609 = tosa.reshape %608 {new_shape = array<i64: 1, 5, 3072>} : (tensor<5x3072xf32>) -> tensor<1x5x3072xf32>
    %610 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x5x3072xf32>}> : () -> tensor<1x5x3072xf32>
    %611 = tosa.mul %609, %610 {shift = 0 : i8} : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %612 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x5x3072xf32>}> : () -> tensor<1x5x3072xf32>
    %613 = tosa.mul %609, %612 {shift = 0 : i8} : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %614 = math.erf %613 : tensor<1x5x3072xf32>
    %615 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x5x3072xf32>}> : () -> tensor<1x5x3072xf32>
    %616 = tosa.add %614, %615 : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %617 = tosa.mul %611, %616 {shift = 0 : i8} : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %618 = tosa.reshape %617 {new_shape = array<i64: 5, 3072>} : (tensor<1x5x3072xf32>) -> tensor<5x3072xf32>
    %619 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %620 = tosa.transpose %arg65, %619 : (tensor<768x3072xf32>, tensor<2xi32>) -> tensor<3072x768xf32>
    %621 = tosa.reshape %618 {new_shape = array<i64: 1, 5, 3072>} : (tensor<5x3072xf32>) -> tensor<1x5x3072xf32>
    %622 = tosa.reshape %620 {new_shape = array<i64: 1, 3072, 768>} : (tensor<3072x768xf32>) -> tensor<1x3072x768xf32>
    %623 = tosa.matmul %621, %622 : (tensor<1x5x3072xf32>, tensor<1x3072x768xf32>) -> tensor<1x5x768xf32>
    %624 = tosa.reshape %623 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %625 = tosa.reshape %arg66 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %626 = tosa.add %625, %624 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %627 = tosa.reshape %626 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %628 = tosa.identity %627 : (tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %629 = tosa.add %628, %599 : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %630 = tosa.reduce_sum %629 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %631 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %632 = tosa.reciprocal %631 : (tensor<1xf32>) -> tensor<1xf32>
    %633 = tosa.mul %632, %630 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %634 = tosa.sub %629, %633 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %635 = tosa.mul %634, %634 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %636 = tosa.reduce_sum %635 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %637 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %638 = tosa.reciprocal %637 : (tensor<1xf32>) -> tensor<1xf32>
    %639 = tosa.mul %638, %636 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %640 = "tosa.const"() <{value = dense<9.99999996E-13> : tensor<1x5x1xf32>}> : () -> tensor<1x5x1xf32>
    %641 = tosa.add %639, %640 : (tensor<1x5x1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %642 = tosa.rsqrt %641 : (tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %643 = tosa.sub %629, %633 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %644 = tosa.mul %643, %642 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %645 = tosa.reshape %arg67 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %646 = tosa.mul %644, %645 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %647 = tosa.reshape %arg68 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %648 = tosa.add %646, %647 : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %649 = tosa.reshape %648 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %650 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %651 = tosa.transpose %arg69, %650 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %652 = tosa.reshape %649 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %653 = tosa.reshape %651 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %654 = tosa.matmul %652, %653 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %655 = tosa.reshape %654 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %656 = tosa.reshape %arg70 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %657 = tosa.add %656, %655 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %658 = tosa.reshape %657 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %659 = tosa.reshape %648 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %660 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %661 = tosa.transpose %arg71, %660 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %662 = tosa.reshape %659 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %663 = tosa.reshape %661 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %664 = tosa.matmul %662, %663 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %665 = tosa.reshape %664 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %666 = tosa.reshape %arg72 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %667 = tosa.add %666, %665 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %668 = tosa.reshape %667 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %669 = tosa.reshape %668 {new_shape = array<i64: 1, 5, 12, 64>} : (tensor<1x5x768xf32>) -> tensor<1x5x12x64xf32>
    %670 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %671 = tosa.transpose %669, %670 : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x12x5x64xf32>
    %672 = tosa.reshape %648 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %673 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %674 = tosa.transpose %arg73, %673 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %675 = tosa.reshape %672 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %676 = tosa.reshape %674 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %677 = tosa.matmul %675, %676 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %678 = tosa.reshape %677 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %679 = tosa.reshape %arg74 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %680 = tosa.add %679, %678 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %681 = tosa.reshape %680 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %682 = tosa.reshape %681 {new_shape = array<i64: 1, 5, 12, 64>} : (tensor<1x5x768xf32>) -> tensor<1x5x12x64xf32>
    %683 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %684 = tosa.transpose %682, %683 : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x12x5x64xf32>
    %685 = tosa.reshape %658 {new_shape = array<i64: 1, 5, 12, 64>} : (tensor<1x5x768xf32>) -> tensor<1x5x12x64xf32>
    %686 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %687 = tosa.transpose %685, %686 : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x12x5x64xf32>
    %688 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %689 = tosa.transpose %671, %688 : (tensor<1x12x5x64xf32>, tensor<4xi32>) -> tensor<1x12x64x5xf32>
    %690 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x5x64xf32>}> : () -> tensor<1x12x5x64xf32>
    %691 = tosa.add %687, %690 : (tensor<1x12x5x64xf32>, tensor<1x12x5x64xf32>) -> tensor<1x12x5x64xf32>
    %692 = tosa.reshape %691 {new_shape = array<i64: 12, 5, 64>} : (tensor<1x12x5x64xf32>) -> tensor<12x5x64xf32>
    %693 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x64x5xf32>}> : () -> tensor<1x12x64x5xf32>
    %694 = tosa.add %689, %693 : (tensor<1x12x64x5xf32>, tensor<1x12x64x5xf32>) -> tensor<1x12x64x5xf32>
    %695 = tosa.reshape %694 {new_shape = array<i64: 12, 64, 5>} : (tensor<1x12x64x5xf32>) -> tensor<12x64x5xf32>
    %696 = tosa.matmul %692, %695 : (tensor<12x5x64xf32>, tensor<12x64x5xf32>) -> tensor<12x5x5xf32>
    %697 = tosa.reshape %696 {new_shape = array<i64: 1, 12, 5, 5>} : (tensor<12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %698 = "tosa.const"() <{value = dense<8.000000e+00> : tensor<1x12x5x5xf32>}> : () -> tensor<1x12x5x5xf32>
    %699 = tosa.reciprocal %698 : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %700 = tosa.mul %697, %699 {shift = 0 : i8} : (tensor<1x12x5x5xf32>, tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %701 = tosa.add %700, %6 : (tensor<1x12x5x5xf32>, tensor<1x1x1x5xf32>) -> tensor<1x12x5x5xf32>
    %702 = tosa.reduce_max %701 {axis = 3 : i32} : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x1xf32>
    %703 = tosa.sub %701, %702 : (tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) -> tensor<1x12x5x5xf32>
    %704 = tosa.exp %703 : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %705 = tosa.reduce_sum %704 {axis = 3 : i32} : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x1xf32>
    %706 = tosa.reciprocal %705 : (tensor<1x12x5x1xf32>) -> tensor<1x12x5x1xf32>
    %707 = tosa.mul %704, %706 {shift = 0 : i8} : (tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) -> tensor<1x12x5x5xf32>
    %708 = tosa.identity %707 : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %709 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x5x5xf32>}> : () -> tensor<1x12x5x5xf32>
    %710 = tosa.add %708, %709 : (tensor<1x12x5x5xf32>, tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %711 = tosa.reshape %710 {new_shape = array<i64: 12, 5, 5>} : (tensor<1x12x5x5xf32>) -> tensor<12x5x5xf32>
    %712 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x5x64xf32>}> : () -> tensor<1x12x5x64xf32>
    %713 = tosa.add %684, %712 : (tensor<1x12x5x64xf32>, tensor<1x12x5x64xf32>) -> tensor<1x12x5x64xf32>
    %714 = tosa.reshape %713 {new_shape = array<i64: 12, 5, 64>} : (tensor<1x12x5x64xf32>) -> tensor<12x5x64xf32>
    %715 = tosa.matmul %711, %714 : (tensor<12x5x5xf32>, tensor<12x5x64xf32>) -> tensor<12x5x64xf32>
    %716 = tosa.reshape %715 {new_shape = array<i64: 1, 12, 5, 64>} : (tensor<12x5x64xf32>) -> tensor<1x12x5x64xf32>
    %717 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %718 = tosa.transpose %716, %717 : (tensor<1x12x5x64xf32>, tensor<4xi32>) -> tensor<1x5x12x64xf32>
    %719 = tosa.identity %718 : (tensor<1x5x12x64xf32>) -> tensor<1x5x12x64xf32>
    %720 = tosa.reshape %719 {new_shape = array<i64: 1, 5, 768>} : (tensor<1x5x12x64xf32>) -> tensor<1x5x768xf32>
    %721 = tosa.reshape %720 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %722 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %723 = tosa.transpose %arg75, %722 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %724 = tosa.reshape %721 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %725 = tosa.reshape %723 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %726 = tosa.matmul %724, %725 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %727 = tosa.reshape %726 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %728 = tosa.reshape %arg76 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %729 = tosa.add %728, %727 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %730 = tosa.reshape %729 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %731 = tosa.identity %730 : (tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %732 = tosa.add %731, %648 : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %733 = tosa.reduce_sum %732 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %734 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %735 = tosa.reciprocal %734 : (tensor<1xf32>) -> tensor<1xf32>
    %736 = tosa.mul %735, %733 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %737 = tosa.sub %732, %736 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %738 = tosa.mul %737, %737 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %739 = tosa.reduce_sum %738 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %740 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %741 = tosa.reciprocal %740 : (tensor<1xf32>) -> tensor<1xf32>
    %742 = tosa.mul %741, %739 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %743 = "tosa.const"() <{value = dense<9.99999996E-13> : tensor<1x5x1xf32>}> : () -> tensor<1x5x1xf32>
    %744 = tosa.add %742, %743 : (tensor<1x5x1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %745 = tosa.rsqrt %744 : (tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %746 = tosa.sub %732, %736 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %747 = tosa.mul %746, %745 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %748 = tosa.reshape %arg77 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %749 = tosa.mul %747, %748 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %750 = tosa.reshape %arg78 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %751 = tosa.add %749, %750 : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %752 = tosa.reshape %751 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %753 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %754 = tosa.transpose %arg79, %753 : (tensor<3072x768xf32>, tensor<2xi32>) -> tensor<768x3072xf32>
    %755 = tosa.reshape %752 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %756 = tosa.reshape %754 {new_shape = array<i64: 1, 768, 3072>} : (tensor<768x3072xf32>) -> tensor<1x768x3072xf32>
    %757 = tosa.matmul %755, %756 : (tensor<1x5x768xf32>, tensor<1x768x3072xf32>) -> tensor<1x5x3072xf32>
    %758 = tosa.reshape %757 {new_shape = array<i64: 5, 3072>} : (tensor<1x5x3072xf32>) -> tensor<5x3072xf32>
    %759 = tosa.reshape %arg80 {new_shape = array<i64: 1, 3072>} : (tensor<3072xf32>) -> tensor<1x3072xf32>
    %760 = tosa.add %759, %758 : (tensor<1x3072xf32>, tensor<5x3072xf32>) -> tensor<5x3072xf32>
    %761 = tosa.reshape %760 {new_shape = array<i64: 1, 5, 3072>} : (tensor<5x3072xf32>) -> tensor<1x5x3072xf32>
    %762 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x5x3072xf32>}> : () -> tensor<1x5x3072xf32>
    %763 = tosa.mul %761, %762 {shift = 0 : i8} : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %764 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x5x3072xf32>}> : () -> tensor<1x5x3072xf32>
    %765 = tosa.mul %761, %764 {shift = 0 : i8} : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %766 = math.erf %765 : tensor<1x5x3072xf32>
    %767 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x5x3072xf32>}> : () -> tensor<1x5x3072xf32>
    %768 = tosa.add %766, %767 : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %769 = tosa.mul %763, %768 {shift = 0 : i8} : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %770 = tosa.reshape %769 {new_shape = array<i64: 5, 3072>} : (tensor<1x5x3072xf32>) -> tensor<5x3072xf32>
    %771 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %772 = tosa.transpose %arg81, %771 : (tensor<768x3072xf32>, tensor<2xi32>) -> tensor<3072x768xf32>
    %773 = tosa.reshape %770 {new_shape = array<i64: 1, 5, 3072>} : (tensor<5x3072xf32>) -> tensor<1x5x3072xf32>
    %774 = tosa.reshape %772 {new_shape = array<i64: 1, 3072, 768>} : (tensor<3072x768xf32>) -> tensor<1x3072x768xf32>
    %775 = tosa.matmul %773, %774 : (tensor<1x5x3072xf32>, tensor<1x3072x768xf32>) -> tensor<1x5x768xf32>
    %776 = tosa.reshape %775 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %777 = tosa.reshape %arg82 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %778 = tosa.add %777, %776 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %779 = tosa.reshape %778 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %780 = tosa.identity %779 : (tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %781 = tosa.add %780, %751 : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %782 = tosa.reduce_sum %781 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %783 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %784 = tosa.reciprocal %783 : (tensor<1xf32>) -> tensor<1xf32>
    %785 = tosa.mul %784, %782 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %786 = tosa.sub %781, %785 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %787 = tosa.mul %786, %786 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %788 = tosa.reduce_sum %787 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %789 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %790 = tosa.reciprocal %789 : (tensor<1xf32>) -> tensor<1xf32>
    %791 = tosa.mul %790, %788 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %792 = "tosa.const"() <{value = dense<9.99999996E-13> : tensor<1x5x1xf32>}> : () -> tensor<1x5x1xf32>
    %793 = tosa.add %791, %792 : (tensor<1x5x1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %794 = tosa.rsqrt %793 : (tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %795 = tosa.sub %781, %785 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %796 = tosa.mul %795, %794 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %797 = tosa.reshape %arg83 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %798 = tosa.mul %796, %797 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %799 = tosa.reshape %arg84 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %800 = tosa.add %798, %799 : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %801 = tosa.reshape %800 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %802 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %803 = tosa.transpose %arg85, %802 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %804 = tosa.reshape %801 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %805 = tosa.reshape %803 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %806 = tosa.matmul %804, %805 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %807 = tosa.reshape %806 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %808 = tosa.reshape %arg86 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %809 = tosa.add %808, %807 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %810 = tosa.reshape %809 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %811 = tosa.reshape %800 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %812 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %813 = tosa.transpose %arg87, %812 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %814 = tosa.reshape %811 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %815 = tosa.reshape %813 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %816 = tosa.matmul %814, %815 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %817 = tosa.reshape %816 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %818 = tosa.reshape %arg88 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %819 = tosa.add %818, %817 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %820 = tosa.reshape %819 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %821 = tosa.reshape %820 {new_shape = array<i64: 1, 5, 12, 64>} : (tensor<1x5x768xf32>) -> tensor<1x5x12x64xf32>
    %822 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %823 = tosa.transpose %821, %822 : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x12x5x64xf32>
    %824 = tosa.reshape %800 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %825 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %826 = tosa.transpose %arg89, %825 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %827 = tosa.reshape %824 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %828 = tosa.reshape %826 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %829 = tosa.matmul %827, %828 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %830 = tosa.reshape %829 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %831 = tosa.reshape %arg90 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %832 = tosa.add %831, %830 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %833 = tosa.reshape %832 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %834 = tosa.reshape %833 {new_shape = array<i64: 1, 5, 12, 64>} : (tensor<1x5x768xf32>) -> tensor<1x5x12x64xf32>
    %835 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %836 = tosa.transpose %834, %835 : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x12x5x64xf32>
    %837 = tosa.reshape %810 {new_shape = array<i64: 1, 5, 12, 64>} : (tensor<1x5x768xf32>) -> tensor<1x5x12x64xf32>
    %838 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %839 = tosa.transpose %837, %838 : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x12x5x64xf32>
    %840 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %841 = tosa.transpose %823, %840 : (tensor<1x12x5x64xf32>, tensor<4xi32>) -> tensor<1x12x64x5xf32>
    %842 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x5x64xf32>}> : () -> tensor<1x12x5x64xf32>
    %843 = tosa.add %839, %842 : (tensor<1x12x5x64xf32>, tensor<1x12x5x64xf32>) -> tensor<1x12x5x64xf32>
    %844 = tosa.reshape %843 {new_shape = array<i64: 12, 5, 64>} : (tensor<1x12x5x64xf32>) -> tensor<12x5x64xf32>
    %845 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x64x5xf32>}> : () -> tensor<1x12x64x5xf32>
    %846 = tosa.add %841, %845 : (tensor<1x12x64x5xf32>, tensor<1x12x64x5xf32>) -> tensor<1x12x64x5xf32>
    %847 = tosa.reshape %846 {new_shape = array<i64: 12, 64, 5>} : (tensor<1x12x64x5xf32>) -> tensor<12x64x5xf32>
    %848 = tosa.matmul %844, %847 : (tensor<12x5x64xf32>, tensor<12x64x5xf32>) -> tensor<12x5x5xf32>
    %849 = tosa.reshape %848 {new_shape = array<i64: 1, 12, 5, 5>} : (tensor<12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %850 = "tosa.const"() <{value = dense<8.000000e+00> : tensor<1x12x5x5xf32>}> : () -> tensor<1x12x5x5xf32>
    %851 = tosa.reciprocal %850 : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %852 = tosa.mul %849, %851 {shift = 0 : i8} : (tensor<1x12x5x5xf32>, tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %853 = tosa.add %852, %6 : (tensor<1x12x5x5xf32>, tensor<1x1x1x5xf32>) -> tensor<1x12x5x5xf32>
    %854 = tosa.reduce_max %853 {axis = 3 : i32} : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x1xf32>
    %855 = tosa.sub %853, %854 : (tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) -> tensor<1x12x5x5xf32>
    %856 = tosa.exp %855 : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %857 = tosa.reduce_sum %856 {axis = 3 : i32} : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x1xf32>
    %858 = tosa.reciprocal %857 : (tensor<1x12x5x1xf32>) -> tensor<1x12x5x1xf32>
    %859 = tosa.mul %856, %858 {shift = 0 : i8} : (tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) -> tensor<1x12x5x5xf32>
    %860 = tosa.identity %859 : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %861 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x5x5xf32>}> : () -> tensor<1x12x5x5xf32>
    %862 = tosa.add %860, %861 : (tensor<1x12x5x5xf32>, tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %863 = tosa.reshape %862 {new_shape = array<i64: 12, 5, 5>} : (tensor<1x12x5x5xf32>) -> tensor<12x5x5xf32>
    %864 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x5x64xf32>}> : () -> tensor<1x12x5x64xf32>
    %865 = tosa.add %836, %864 : (tensor<1x12x5x64xf32>, tensor<1x12x5x64xf32>) -> tensor<1x12x5x64xf32>
    %866 = tosa.reshape %865 {new_shape = array<i64: 12, 5, 64>} : (tensor<1x12x5x64xf32>) -> tensor<12x5x64xf32>
    %867 = tosa.matmul %863, %866 : (tensor<12x5x5xf32>, tensor<12x5x64xf32>) -> tensor<12x5x64xf32>
    %868 = tosa.reshape %867 {new_shape = array<i64: 1, 12, 5, 64>} : (tensor<12x5x64xf32>) -> tensor<1x12x5x64xf32>
    %869 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %870 = tosa.transpose %868, %869 : (tensor<1x12x5x64xf32>, tensor<4xi32>) -> tensor<1x5x12x64xf32>
    %871 = tosa.identity %870 : (tensor<1x5x12x64xf32>) -> tensor<1x5x12x64xf32>
    %872 = tosa.reshape %871 {new_shape = array<i64: 1, 5, 768>} : (tensor<1x5x12x64xf32>) -> tensor<1x5x768xf32>
    %873 = tosa.reshape %872 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %874 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %875 = tosa.transpose %arg91, %874 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %876 = tosa.reshape %873 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %877 = tosa.reshape %875 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %878 = tosa.matmul %876, %877 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %879 = tosa.reshape %878 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %880 = tosa.reshape %arg92 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %881 = tosa.add %880, %879 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %882 = tosa.reshape %881 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %883 = tosa.identity %882 : (tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %884 = tosa.add %883, %800 : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %885 = tosa.reduce_sum %884 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %886 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %887 = tosa.reciprocal %886 : (tensor<1xf32>) -> tensor<1xf32>
    %888 = tosa.mul %887, %885 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %889 = tosa.sub %884, %888 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %890 = tosa.mul %889, %889 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %891 = tosa.reduce_sum %890 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %892 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %893 = tosa.reciprocal %892 : (tensor<1xf32>) -> tensor<1xf32>
    %894 = tosa.mul %893, %891 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %895 = "tosa.const"() <{value = dense<9.99999996E-13> : tensor<1x5x1xf32>}> : () -> tensor<1x5x1xf32>
    %896 = tosa.add %894, %895 : (tensor<1x5x1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %897 = tosa.rsqrt %896 : (tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %898 = tosa.sub %884, %888 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %899 = tosa.mul %898, %897 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %900 = tosa.reshape %arg93 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %901 = tosa.mul %899, %900 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %902 = tosa.reshape %arg94 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %903 = tosa.add %901, %902 : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %904 = tosa.reshape %903 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %905 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %906 = tosa.transpose %arg95, %905 : (tensor<3072x768xf32>, tensor<2xi32>) -> tensor<768x3072xf32>
    %907 = tosa.reshape %904 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %908 = tosa.reshape %906 {new_shape = array<i64: 1, 768, 3072>} : (tensor<768x3072xf32>) -> tensor<1x768x3072xf32>
    %909 = tosa.matmul %907, %908 : (tensor<1x5x768xf32>, tensor<1x768x3072xf32>) -> tensor<1x5x3072xf32>
    %910 = tosa.reshape %909 {new_shape = array<i64: 5, 3072>} : (tensor<1x5x3072xf32>) -> tensor<5x3072xf32>
    %911 = tosa.reshape %arg96 {new_shape = array<i64: 1, 3072>} : (tensor<3072xf32>) -> tensor<1x3072xf32>
    %912 = tosa.add %911, %910 : (tensor<1x3072xf32>, tensor<5x3072xf32>) -> tensor<5x3072xf32>
    %913 = tosa.reshape %912 {new_shape = array<i64: 1, 5, 3072>} : (tensor<5x3072xf32>) -> tensor<1x5x3072xf32>
    %914 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x5x3072xf32>}> : () -> tensor<1x5x3072xf32>
    %915 = tosa.mul %913, %914 {shift = 0 : i8} : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %916 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x5x3072xf32>}> : () -> tensor<1x5x3072xf32>
    %917 = tosa.mul %913, %916 {shift = 0 : i8} : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %918 = math.erf %917 : tensor<1x5x3072xf32>
    %919 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x5x3072xf32>}> : () -> tensor<1x5x3072xf32>
    %920 = tosa.add %918, %919 : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %921 = tosa.mul %915, %920 {shift = 0 : i8} : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %922 = tosa.reshape %921 {new_shape = array<i64: 5, 3072>} : (tensor<1x5x3072xf32>) -> tensor<5x3072xf32>
    %923 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %924 = tosa.transpose %arg97, %923 : (tensor<768x3072xf32>, tensor<2xi32>) -> tensor<3072x768xf32>
    %925 = tosa.reshape %922 {new_shape = array<i64: 1, 5, 3072>} : (tensor<5x3072xf32>) -> tensor<1x5x3072xf32>
    %926 = tosa.reshape %924 {new_shape = array<i64: 1, 3072, 768>} : (tensor<3072x768xf32>) -> tensor<1x3072x768xf32>
    %927 = tosa.matmul %925, %926 : (tensor<1x5x3072xf32>, tensor<1x3072x768xf32>) -> tensor<1x5x768xf32>
    %928 = tosa.reshape %927 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %929 = tosa.reshape %arg98 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %930 = tosa.add %929, %928 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %931 = tosa.reshape %930 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %932 = tosa.identity %931 : (tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %933 = tosa.add %932, %903 : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %934 = tosa.reduce_sum %933 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %935 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %936 = tosa.reciprocal %935 : (tensor<1xf32>) -> tensor<1xf32>
    %937 = tosa.mul %936, %934 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %938 = tosa.sub %933, %937 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %939 = tosa.mul %938, %938 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %940 = tosa.reduce_sum %939 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %941 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %942 = tosa.reciprocal %941 : (tensor<1xf32>) -> tensor<1xf32>
    %943 = tosa.mul %942, %940 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %944 = "tosa.const"() <{value = dense<9.99999996E-13> : tensor<1x5x1xf32>}> : () -> tensor<1x5x1xf32>
    %945 = tosa.add %943, %944 : (tensor<1x5x1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %946 = tosa.rsqrt %945 : (tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %947 = tosa.sub %933, %937 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %948 = tosa.mul %947, %946 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %949 = tosa.reshape %arg99 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %950 = tosa.mul %948, %949 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %951 = tosa.reshape %arg100 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %952 = tosa.add %950, %951 : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %953 = tosa.reshape %952 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %954 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %955 = tosa.transpose %arg101, %954 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %956 = tosa.reshape %953 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %957 = tosa.reshape %955 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %958 = tosa.matmul %956, %957 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %959 = tosa.reshape %958 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %960 = tosa.reshape %arg102 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %961 = tosa.add %960, %959 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %962 = tosa.reshape %961 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %963 = tosa.reshape %952 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %964 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %965 = tosa.transpose %arg103, %964 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %966 = tosa.reshape %963 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %967 = tosa.reshape %965 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %968 = tosa.matmul %966, %967 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %969 = tosa.reshape %968 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %970 = tosa.reshape %arg104 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %971 = tosa.add %970, %969 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %972 = tosa.reshape %971 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %973 = tosa.reshape %972 {new_shape = array<i64: 1, 5, 12, 64>} : (tensor<1x5x768xf32>) -> tensor<1x5x12x64xf32>
    %974 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %975 = tosa.transpose %973, %974 : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x12x5x64xf32>
    %976 = tosa.reshape %952 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %977 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %978 = tosa.transpose %arg105, %977 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %979 = tosa.reshape %976 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %980 = tosa.reshape %978 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %981 = tosa.matmul %979, %980 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %982 = tosa.reshape %981 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %983 = tosa.reshape %arg106 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %984 = tosa.add %983, %982 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %985 = tosa.reshape %984 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %986 = tosa.reshape %985 {new_shape = array<i64: 1, 5, 12, 64>} : (tensor<1x5x768xf32>) -> tensor<1x5x12x64xf32>
    %987 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %988 = tosa.transpose %986, %987 : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x12x5x64xf32>
    %989 = tosa.reshape %962 {new_shape = array<i64: 1, 5, 12, 64>} : (tensor<1x5x768xf32>) -> tensor<1x5x12x64xf32>
    %990 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %991 = tosa.transpose %989, %990 : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x12x5x64xf32>
    %992 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %993 = tosa.transpose %975, %992 : (tensor<1x12x5x64xf32>, tensor<4xi32>) -> tensor<1x12x64x5xf32>
    %994 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x5x64xf32>}> : () -> tensor<1x12x5x64xf32>
    %995 = tosa.add %991, %994 : (tensor<1x12x5x64xf32>, tensor<1x12x5x64xf32>) -> tensor<1x12x5x64xf32>
    %996 = tosa.reshape %995 {new_shape = array<i64: 12, 5, 64>} : (tensor<1x12x5x64xf32>) -> tensor<12x5x64xf32>
    %997 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x64x5xf32>}> : () -> tensor<1x12x64x5xf32>
    %998 = tosa.add %993, %997 : (tensor<1x12x64x5xf32>, tensor<1x12x64x5xf32>) -> tensor<1x12x64x5xf32>
    %999 = tosa.reshape %998 {new_shape = array<i64: 12, 64, 5>} : (tensor<1x12x64x5xf32>) -> tensor<12x64x5xf32>
    %1000 = tosa.matmul %996, %999 : (tensor<12x5x64xf32>, tensor<12x64x5xf32>) -> tensor<12x5x5xf32>
    %1001 = tosa.reshape %1000 {new_shape = array<i64: 1, 12, 5, 5>} : (tensor<12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %1002 = "tosa.const"() <{value = dense<8.000000e+00> : tensor<1x12x5x5xf32>}> : () -> tensor<1x12x5x5xf32>
    %1003 = tosa.reciprocal %1002 : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %1004 = tosa.mul %1001, %1003 {shift = 0 : i8} : (tensor<1x12x5x5xf32>, tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %1005 = tosa.add %1004, %6 : (tensor<1x12x5x5xf32>, tensor<1x1x1x5xf32>) -> tensor<1x12x5x5xf32>
    %1006 = tosa.reduce_max %1005 {axis = 3 : i32} : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x1xf32>
    %1007 = tosa.sub %1005, %1006 : (tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) -> tensor<1x12x5x5xf32>
    %1008 = tosa.exp %1007 : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %1009 = tosa.reduce_sum %1008 {axis = 3 : i32} : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x1xf32>
    %1010 = tosa.reciprocal %1009 : (tensor<1x12x5x1xf32>) -> tensor<1x12x5x1xf32>
    %1011 = tosa.mul %1008, %1010 {shift = 0 : i8} : (tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) -> tensor<1x12x5x5xf32>
    %1012 = tosa.identity %1011 : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %1013 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x5x5xf32>}> : () -> tensor<1x12x5x5xf32>
    %1014 = tosa.add %1012, %1013 : (tensor<1x12x5x5xf32>, tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %1015 = tosa.reshape %1014 {new_shape = array<i64: 12, 5, 5>} : (tensor<1x12x5x5xf32>) -> tensor<12x5x5xf32>
    %1016 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x5x64xf32>}> : () -> tensor<1x12x5x64xf32>
    %1017 = tosa.add %988, %1016 : (tensor<1x12x5x64xf32>, tensor<1x12x5x64xf32>) -> tensor<1x12x5x64xf32>
    %1018 = tosa.reshape %1017 {new_shape = array<i64: 12, 5, 64>} : (tensor<1x12x5x64xf32>) -> tensor<12x5x64xf32>
    %1019 = tosa.matmul %1015, %1018 : (tensor<12x5x5xf32>, tensor<12x5x64xf32>) -> tensor<12x5x64xf32>
    %1020 = tosa.reshape %1019 {new_shape = array<i64: 1, 12, 5, 64>} : (tensor<12x5x64xf32>) -> tensor<1x12x5x64xf32>
    %1021 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1022 = tosa.transpose %1020, %1021 : (tensor<1x12x5x64xf32>, tensor<4xi32>) -> tensor<1x5x12x64xf32>
    %1023 = tosa.identity %1022 : (tensor<1x5x12x64xf32>) -> tensor<1x5x12x64xf32>
    %1024 = tosa.reshape %1023 {new_shape = array<i64: 1, 5, 768>} : (tensor<1x5x12x64xf32>) -> tensor<1x5x768xf32>
    %1025 = tosa.reshape %1024 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1026 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1027 = tosa.transpose %arg107, %1026 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %1028 = tosa.reshape %1025 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1029 = tosa.reshape %1027 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %1030 = tosa.matmul %1028, %1029 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %1031 = tosa.reshape %1030 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1032 = tosa.reshape %arg108 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %1033 = tosa.add %1032, %1031 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %1034 = tosa.reshape %1033 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1035 = tosa.identity %1034 : (tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %1036 = tosa.add %1035, %952 : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %1037 = tosa.reduce_sum %1036 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %1038 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1039 = tosa.reciprocal %1038 : (tensor<1xf32>) -> tensor<1xf32>
    %1040 = tosa.mul %1039, %1037 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1041 = tosa.sub %1036, %1040 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %1042 = tosa.mul %1041, %1041 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %1043 = tosa.reduce_sum %1042 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %1044 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1045 = tosa.reciprocal %1044 : (tensor<1xf32>) -> tensor<1xf32>
    %1046 = tosa.mul %1045, %1043 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1047 = "tosa.const"() <{value = dense<9.99999996E-13> : tensor<1x5x1xf32>}> : () -> tensor<1x5x1xf32>
    %1048 = tosa.add %1046, %1047 : (tensor<1x5x1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1049 = tosa.rsqrt %1048 : (tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1050 = tosa.sub %1036, %1040 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %1051 = tosa.mul %1050, %1049 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %1052 = tosa.reshape %arg109 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1053 = tosa.mul %1051, %1052 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %1054 = tosa.reshape %arg110 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1055 = tosa.add %1053, %1054 : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %1056 = tosa.reshape %1055 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1057 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1058 = tosa.transpose %arg111, %1057 : (tensor<3072x768xf32>, tensor<2xi32>) -> tensor<768x3072xf32>
    %1059 = tosa.reshape %1056 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1060 = tosa.reshape %1058 {new_shape = array<i64: 1, 768, 3072>} : (tensor<768x3072xf32>) -> tensor<1x768x3072xf32>
    %1061 = tosa.matmul %1059, %1060 : (tensor<1x5x768xf32>, tensor<1x768x3072xf32>) -> tensor<1x5x3072xf32>
    %1062 = tosa.reshape %1061 {new_shape = array<i64: 5, 3072>} : (tensor<1x5x3072xf32>) -> tensor<5x3072xf32>
    %1063 = tosa.reshape %arg112 {new_shape = array<i64: 1, 3072>} : (tensor<3072xf32>) -> tensor<1x3072xf32>
    %1064 = tosa.add %1063, %1062 : (tensor<1x3072xf32>, tensor<5x3072xf32>) -> tensor<5x3072xf32>
    %1065 = tosa.reshape %1064 {new_shape = array<i64: 1, 5, 3072>} : (tensor<5x3072xf32>) -> tensor<1x5x3072xf32>
    %1066 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x5x3072xf32>}> : () -> tensor<1x5x3072xf32>
    %1067 = tosa.mul %1065, %1066 {shift = 0 : i8} : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %1068 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x5x3072xf32>}> : () -> tensor<1x5x3072xf32>
    %1069 = tosa.mul %1065, %1068 {shift = 0 : i8} : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %1070 = math.erf %1069 : tensor<1x5x3072xf32>
    %1071 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x5x3072xf32>}> : () -> tensor<1x5x3072xf32>
    %1072 = tosa.add %1070, %1071 : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %1073 = tosa.mul %1067, %1072 {shift = 0 : i8} : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %1074 = tosa.reshape %1073 {new_shape = array<i64: 5, 3072>} : (tensor<1x5x3072xf32>) -> tensor<5x3072xf32>
    %1075 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1076 = tosa.transpose %arg113, %1075 : (tensor<768x3072xf32>, tensor<2xi32>) -> tensor<3072x768xf32>
    %1077 = tosa.reshape %1074 {new_shape = array<i64: 1, 5, 3072>} : (tensor<5x3072xf32>) -> tensor<1x5x3072xf32>
    %1078 = tosa.reshape %1076 {new_shape = array<i64: 1, 3072, 768>} : (tensor<3072x768xf32>) -> tensor<1x3072x768xf32>
    %1079 = tosa.matmul %1077, %1078 : (tensor<1x5x3072xf32>, tensor<1x3072x768xf32>) -> tensor<1x5x768xf32>
    %1080 = tosa.reshape %1079 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1081 = tosa.reshape %arg114 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %1082 = tosa.add %1081, %1080 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %1083 = tosa.reshape %1082 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1084 = tosa.identity %1083 : (tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %1085 = tosa.add %1084, %1055 : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %1086 = tosa.reduce_sum %1085 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %1087 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1088 = tosa.reciprocal %1087 : (tensor<1xf32>) -> tensor<1xf32>
    %1089 = tosa.mul %1088, %1086 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1090 = tosa.sub %1085, %1089 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %1091 = tosa.mul %1090, %1090 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %1092 = tosa.reduce_sum %1091 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %1093 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1094 = tosa.reciprocal %1093 : (tensor<1xf32>) -> tensor<1xf32>
    %1095 = tosa.mul %1094, %1092 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1096 = "tosa.const"() <{value = dense<9.99999996E-13> : tensor<1x5x1xf32>}> : () -> tensor<1x5x1xf32>
    %1097 = tosa.add %1095, %1096 : (tensor<1x5x1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1098 = tosa.rsqrt %1097 : (tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1099 = tosa.sub %1085, %1089 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %1100 = tosa.mul %1099, %1098 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %1101 = tosa.reshape %arg115 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1102 = tosa.mul %1100, %1101 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %1103 = tosa.reshape %arg116 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1104 = tosa.add %1102, %1103 : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %1105 = tosa.reshape %1104 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1106 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1107 = tosa.transpose %arg117, %1106 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %1108 = tosa.reshape %1105 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1109 = tosa.reshape %1107 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %1110 = tosa.matmul %1108, %1109 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %1111 = tosa.reshape %1110 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1112 = tosa.reshape %arg118 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %1113 = tosa.add %1112, %1111 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %1114 = tosa.reshape %1113 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1115 = tosa.reshape %1104 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1116 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1117 = tosa.transpose %arg119, %1116 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %1118 = tosa.reshape %1115 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1119 = tosa.reshape %1117 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %1120 = tosa.matmul %1118, %1119 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %1121 = tosa.reshape %1120 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1122 = tosa.reshape %arg120 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %1123 = tosa.add %1122, %1121 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %1124 = tosa.reshape %1123 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1125 = tosa.reshape %1124 {new_shape = array<i64: 1, 5, 12, 64>} : (tensor<1x5x768xf32>) -> tensor<1x5x12x64xf32>
    %1126 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1127 = tosa.transpose %1125, %1126 : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x12x5x64xf32>
    %1128 = tosa.reshape %1104 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1129 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1130 = tosa.transpose %arg121, %1129 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %1131 = tosa.reshape %1128 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1132 = tosa.reshape %1130 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %1133 = tosa.matmul %1131, %1132 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %1134 = tosa.reshape %1133 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1135 = tosa.reshape %arg122 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %1136 = tosa.add %1135, %1134 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %1137 = tosa.reshape %1136 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1138 = tosa.reshape %1137 {new_shape = array<i64: 1, 5, 12, 64>} : (tensor<1x5x768xf32>) -> tensor<1x5x12x64xf32>
    %1139 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1140 = tosa.transpose %1138, %1139 : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x12x5x64xf32>
    %1141 = tosa.reshape %1114 {new_shape = array<i64: 1, 5, 12, 64>} : (tensor<1x5x768xf32>) -> tensor<1x5x12x64xf32>
    %1142 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1143 = tosa.transpose %1141, %1142 : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x12x5x64xf32>
    %1144 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1145 = tosa.transpose %1127, %1144 : (tensor<1x12x5x64xf32>, tensor<4xi32>) -> tensor<1x12x64x5xf32>
    %1146 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x5x64xf32>}> : () -> tensor<1x12x5x64xf32>
    %1147 = tosa.add %1143, %1146 : (tensor<1x12x5x64xf32>, tensor<1x12x5x64xf32>) -> tensor<1x12x5x64xf32>
    %1148 = tosa.reshape %1147 {new_shape = array<i64: 12, 5, 64>} : (tensor<1x12x5x64xf32>) -> tensor<12x5x64xf32>
    %1149 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x64x5xf32>}> : () -> tensor<1x12x64x5xf32>
    %1150 = tosa.add %1145, %1149 : (tensor<1x12x64x5xf32>, tensor<1x12x64x5xf32>) -> tensor<1x12x64x5xf32>
    %1151 = tosa.reshape %1150 {new_shape = array<i64: 12, 64, 5>} : (tensor<1x12x64x5xf32>) -> tensor<12x64x5xf32>
    %1152 = tosa.matmul %1148, %1151 : (tensor<12x5x64xf32>, tensor<12x64x5xf32>) -> tensor<12x5x5xf32>
    %1153 = tosa.reshape %1152 {new_shape = array<i64: 1, 12, 5, 5>} : (tensor<12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %1154 = "tosa.const"() <{value = dense<8.000000e+00> : tensor<1x12x5x5xf32>}> : () -> tensor<1x12x5x5xf32>
    %1155 = tosa.reciprocal %1154 : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %1156 = tosa.mul %1153, %1155 {shift = 0 : i8} : (tensor<1x12x5x5xf32>, tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %1157 = tosa.add %1156, %6 : (tensor<1x12x5x5xf32>, tensor<1x1x1x5xf32>) -> tensor<1x12x5x5xf32>
    %1158 = tosa.reduce_max %1157 {axis = 3 : i32} : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x1xf32>
    %1159 = tosa.sub %1157, %1158 : (tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) -> tensor<1x12x5x5xf32>
    %1160 = tosa.exp %1159 : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %1161 = tosa.reduce_sum %1160 {axis = 3 : i32} : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x1xf32>
    %1162 = tosa.reciprocal %1161 : (tensor<1x12x5x1xf32>) -> tensor<1x12x5x1xf32>
    %1163 = tosa.mul %1160, %1162 {shift = 0 : i8} : (tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) -> tensor<1x12x5x5xf32>
    %1164 = tosa.identity %1163 : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %1165 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x5x5xf32>}> : () -> tensor<1x12x5x5xf32>
    %1166 = tosa.add %1164, %1165 : (tensor<1x12x5x5xf32>, tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %1167 = tosa.reshape %1166 {new_shape = array<i64: 12, 5, 5>} : (tensor<1x12x5x5xf32>) -> tensor<12x5x5xf32>
    %1168 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x5x64xf32>}> : () -> tensor<1x12x5x64xf32>
    %1169 = tosa.add %1140, %1168 : (tensor<1x12x5x64xf32>, tensor<1x12x5x64xf32>) -> tensor<1x12x5x64xf32>
    %1170 = tosa.reshape %1169 {new_shape = array<i64: 12, 5, 64>} : (tensor<1x12x5x64xf32>) -> tensor<12x5x64xf32>
    %1171 = tosa.matmul %1167, %1170 : (tensor<12x5x5xf32>, tensor<12x5x64xf32>) -> tensor<12x5x64xf32>
    %1172 = tosa.reshape %1171 {new_shape = array<i64: 1, 12, 5, 64>} : (tensor<12x5x64xf32>) -> tensor<1x12x5x64xf32>
    %1173 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1174 = tosa.transpose %1172, %1173 : (tensor<1x12x5x64xf32>, tensor<4xi32>) -> tensor<1x5x12x64xf32>
    %1175 = tosa.identity %1174 : (tensor<1x5x12x64xf32>) -> tensor<1x5x12x64xf32>
    %1176 = tosa.reshape %1175 {new_shape = array<i64: 1, 5, 768>} : (tensor<1x5x12x64xf32>) -> tensor<1x5x768xf32>
    %1177 = tosa.reshape %1176 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1178 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1179 = tosa.transpose %arg123, %1178 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %1180 = tosa.reshape %1177 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1181 = tosa.reshape %1179 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %1182 = tosa.matmul %1180, %1181 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %1183 = tosa.reshape %1182 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1184 = tosa.reshape %arg124 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %1185 = tosa.add %1184, %1183 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %1186 = tosa.reshape %1185 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1187 = tosa.identity %1186 : (tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %1188 = tosa.add %1187, %1104 : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %1189 = tosa.reduce_sum %1188 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %1190 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1191 = tosa.reciprocal %1190 : (tensor<1xf32>) -> tensor<1xf32>
    %1192 = tosa.mul %1191, %1189 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1193 = tosa.sub %1188, %1192 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %1194 = tosa.mul %1193, %1193 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %1195 = tosa.reduce_sum %1194 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %1196 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1197 = tosa.reciprocal %1196 : (tensor<1xf32>) -> tensor<1xf32>
    %1198 = tosa.mul %1197, %1195 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1199 = "tosa.const"() <{value = dense<9.99999996E-13> : tensor<1x5x1xf32>}> : () -> tensor<1x5x1xf32>
    %1200 = tosa.add %1198, %1199 : (tensor<1x5x1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1201 = tosa.rsqrt %1200 : (tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1202 = tosa.sub %1188, %1192 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %1203 = tosa.mul %1202, %1201 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %1204 = tosa.reshape %arg125 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1205 = tosa.mul %1203, %1204 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %1206 = tosa.reshape %arg126 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1207 = tosa.add %1205, %1206 : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %1208 = tosa.reshape %1207 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1209 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1210 = tosa.transpose %arg127, %1209 : (tensor<3072x768xf32>, tensor<2xi32>) -> tensor<768x3072xf32>
    %1211 = tosa.reshape %1208 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1212 = tosa.reshape %1210 {new_shape = array<i64: 1, 768, 3072>} : (tensor<768x3072xf32>) -> tensor<1x768x3072xf32>
    %1213 = tosa.matmul %1211, %1212 : (tensor<1x5x768xf32>, tensor<1x768x3072xf32>) -> tensor<1x5x3072xf32>
    %1214 = tosa.reshape %1213 {new_shape = array<i64: 5, 3072>} : (tensor<1x5x3072xf32>) -> tensor<5x3072xf32>
    %1215 = tosa.reshape %arg128 {new_shape = array<i64: 1, 3072>} : (tensor<3072xf32>) -> tensor<1x3072xf32>
    %1216 = tosa.add %1215, %1214 : (tensor<1x3072xf32>, tensor<5x3072xf32>) -> tensor<5x3072xf32>
    %1217 = tosa.reshape %1216 {new_shape = array<i64: 1, 5, 3072>} : (tensor<5x3072xf32>) -> tensor<1x5x3072xf32>
    %1218 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x5x3072xf32>}> : () -> tensor<1x5x3072xf32>
    %1219 = tosa.mul %1217, %1218 {shift = 0 : i8} : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %1220 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x5x3072xf32>}> : () -> tensor<1x5x3072xf32>
    %1221 = tosa.mul %1217, %1220 {shift = 0 : i8} : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %1222 = math.erf %1221 : tensor<1x5x3072xf32>
    %1223 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x5x3072xf32>}> : () -> tensor<1x5x3072xf32>
    %1224 = tosa.add %1222, %1223 : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %1225 = tosa.mul %1219, %1224 {shift = 0 : i8} : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %1226 = tosa.reshape %1225 {new_shape = array<i64: 5, 3072>} : (tensor<1x5x3072xf32>) -> tensor<5x3072xf32>
    %1227 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1228 = tosa.transpose %arg129, %1227 : (tensor<768x3072xf32>, tensor<2xi32>) -> tensor<3072x768xf32>
    %1229 = tosa.reshape %1226 {new_shape = array<i64: 1, 5, 3072>} : (tensor<5x3072xf32>) -> tensor<1x5x3072xf32>
    %1230 = tosa.reshape %1228 {new_shape = array<i64: 1, 3072, 768>} : (tensor<3072x768xf32>) -> tensor<1x3072x768xf32>
    %1231 = tosa.matmul %1229, %1230 : (tensor<1x5x3072xf32>, tensor<1x3072x768xf32>) -> tensor<1x5x768xf32>
    %1232 = tosa.reshape %1231 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1233 = tosa.reshape %arg130 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %1234 = tosa.add %1233, %1232 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %1235 = tosa.reshape %1234 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1236 = tosa.identity %1235 : (tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %1237 = tosa.add %1236, %1207 : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %1238 = tosa.reduce_sum %1237 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %1239 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1240 = tosa.reciprocal %1239 : (tensor<1xf32>) -> tensor<1xf32>
    %1241 = tosa.mul %1240, %1238 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1242 = tosa.sub %1237, %1241 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %1243 = tosa.mul %1242, %1242 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %1244 = tosa.reduce_sum %1243 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %1245 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1246 = tosa.reciprocal %1245 : (tensor<1xf32>) -> tensor<1xf32>
    %1247 = tosa.mul %1246, %1244 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1248 = "tosa.const"() <{value = dense<9.99999996E-13> : tensor<1x5x1xf32>}> : () -> tensor<1x5x1xf32>
    %1249 = tosa.add %1247, %1248 : (tensor<1x5x1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1250 = tosa.rsqrt %1249 : (tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1251 = tosa.sub %1237, %1241 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %1252 = tosa.mul %1251, %1250 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %1253 = tosa.reshape %arg131 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1254 = tosa.mul %1252, %1253 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %1255 = tosa.reshape %arg132 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1256 = tosa.add %1254, %1255 : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %1257 = tosa.reshape %1256 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1258 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1259 = tosa.transpose %arg133, %1258 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %1260 = tosa.reshape %1257 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1261 = tosa.reshape %1259 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %1262 = tosa.matmul %1260, %1261 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %1263 = tosa.reshape %1262 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1264 = tosa.reshape %arg134 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %1265 = tosa.add %1264, %1263 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %1266 = tosa.reshape %1265 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1267 = tosa.reshape %1256 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1268 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1269 = tosa.transpose %arg135, %1268 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %1270 = tosa.reshape %1267 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1271 = tosa.reshape %1269 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %1272 = tosa.matmul %1270, %1271 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %1273 = tosa.reshape %1272 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1274 = tosa.reshape %arg136 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %1275 = tosa.add %1274, %1273 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %1276 = tosa.reshape %1275 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1277 = tosa.reshape %1276 {new_shape = array<i64: 1, 5, 12, 64>} : (tensor<1x5x768xf32>) -> tensor<1x5x12x64xf32>
    %1278 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1279 = tosa.transpose %1277, %1278 : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x12x5x64xf32>
    %1280 = tosa.reshape %1256 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1281 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1282 = tosa.transpose %arg137, %1281 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %1283 = tosa.reshape %1280 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1284 = tosa.reshape %1282 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %1285 = tosa.matmul %1283, %1284 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %1286 = tosa.reshape %1285 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1287 = tosa.reshape %arg138 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %1288 = tosa.add %1287, %1286 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %1289 = tosa.reshape %1288 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1290 = tosa.reshape %1289 {new_shape = array<i64: 1, 5, 12, 64>} : (tensor<1x5x768xf32>) -> tensor<1x5x12x64xf32>
    %1291 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1292 = tosa.transpose %1290, %1291 : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x12x5x64xf32>
    %1293 = tosa.reshape %1266 {new_shape = array<i64: 1, 5, 12, 64>} : (tensor<1x5x768xf32>) -> tensor<1x5x12x64xf32>
    %1294 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1295 = tosa.transpose %1293, %1294 : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x12x5x64xf32>
    %1296 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1297 = tosa.transpose %1279, %1296 : (tensor<1x12x5x64xf32>, tensor<4xi32>) -> tensor<1x12x64x5xf32>
    %1298 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x5x64xf32>}> : () -> tensor<1x12x5x64xf32>
    %1299 = tosa.add %1295, %1298 : (tensor<1x12x5x64xf32>, tensor<1x12x5x64xf32>) -> tensor<1x12x5x64xf32>
    %1300 = tosa.reshape %1299 {new_shape = array<i64: 12, 5, 64>} : (tensor<1x12x5x64xf32>) -> tensor<12x5x64xf32>
    %1301 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x64x5xf32>}> : () -> tensor<1x12x64x5xf32>
    %1302 = tosa.add %1297, %1301 : (tensor<1x12x64x5xf32>, tensor<1x12x64x5xf32>) -> tensor<1x12x64x5xf32>
    %1303 = tosa.reshape %1302 {new_shape = array<i64: 12, 64, 5>} : (tensor<1x12x64x5xf32>) -> tensor<12x64x5xf32>
    %1304 = tosa.matmul %1300, %1303 : (tensor<12x5x64xf32>, tensor<12x64x5xf32>) -> tensor<12x5x5xf32>
    %1305 = tosa.reshape %1304 {new_shape = array<i64: 1, 12, 5, 5>} : (tensor<12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %1306 = "tosa.const"() <{value = dense<8.000000e+00> : tensor<1x12x5x5xf32>}> : () -> tensor<1x12x5x5xf32>
    %1307 = tosa.reciprocal %1306 : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %1308 = tosa.mul %1305, %1307 {shift = 0 : i8} : (tensor<1x12x5x5xf32>, tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %1309 = tosa.add %1308, %6 : (tensor<1x12x5x5xf32>, tensor<1x1x1x5xf32>) -> tensor<1x12x5x5xf32>
    %1310 = tosa.reduce_max %1309 {axis = 3 : i32} : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x1xf32>
    %1311 = tosa.sub %1309, %1310 : (tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) -> tensor<1x12x5x5xf32>
    %1312 = tosa.exp %1311 : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %1313 = tosa.reduce_sum %1312 {axis = 3 : i32} : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x1xf32>
    %1314 = tosa.reciprocal %1313 : (tensor<1x12x5x1xf32>) -> tensor<1x12x5x1xf32>
    %1315 = tosa.mul %1312, %1314 {shift = 0 : i8} : (tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) -> tensor<1x12x5x5xf32>
    %1316 = tosa.identity %1315 : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %1317 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x5x5xf32>}> : () -> tensor<1x12x5x5xf32>
    %1318 = tosa.add %1316, %1317 : (tensor<1x12x5x5xf32>, tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %1319 = tosa.reshape %1318 {new_shape = array<i64: 12, 5, 5>} : (tensor<1x12x5x5xf32>) -> tensor<12x5x5xf32>
    %1320 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x5x64xf32>}> : () -> tensor<1x12x5x64xf32>
    %1321 = tosa.add %1292, %1320 : (tensor<1x12x5x64xf32>, tensor<1x12x5x64xf32>) -> tensor<1x12x5x64xf32>
    %1322 = tosa.reshape %1321 {new_shape = array<i64: 12, 5, 64>} : (tensor<1x12x5x64xf32>) -> tensor<12x5x64xf32>
    %1323 = tosa.matmul %1319, %1322 : (tensor<12x5x5xf32>, tensor<12x5x64xf32>) -> tensor<12x5x64xf32>
    %1324 = tosa.reshape %1323 {new_shape = array<i64: 1, 12, 5, 64>} : (tensor<12x5x64xf32>) -> tensor<1x12x5x64xf32>
    %1325 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1326 = tosa.transpose %1324, %1325 : (tensor<1x12x5x64xf32>, tensor<4xi32>) -> tensor<1x5x12x64xf32>
    %1327 = tosa.identity %1326 : (tensor<1x5x12x64xf32>) -> tensor<1x5x12x64xf32>
    %1328 = tosa.reshape %1327 {new_shape = array<i64: 1, 5, 768>} : (tensor<1x5x12x64xf32>) -> tensor<1x5x768xf32>
    %1329 = tosa.reshape %1328 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1330 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1331 = tosa.transpose %arg139, %1330 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %1332 = tosa.reshape %1329 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1333 = tosa.reshape %1331 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %1334 = tosa.matmul %1332, %1333 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %1335 = tosa.reshape %1334 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1336 = tosa.reshape %arg140 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %1337 = tosa.add %1336, %1335 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %1338 = tosa.reshape %1337 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1339 = tosa.identity %1338 : (tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %1340 = tosa.add %1339, %1256 : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %1341 = tosa.reduce_sum %1340 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %1342 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1343 = tosa.reciprocal %1342 : (tensor<1xf32>) -> tensor<1xf32>
    %1344 = tosa.mul %1343, %1341 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1345 = tosa.sub %1340, %1344 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %1346 = tosa.mul %1345, %1345 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %1347 = tosa.reduce_sum %1346 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %1348 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1349 = tosa.reciprocal %1348 : (tensor<1xf32>) -> tensor<1xf32>
    %1350 = tosa.mul %1349, %1347 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1351 = "tosa.const"() <{value = dense<9.99999996E-13> : tensor<1x5x1xf32>}> : () -> tensor<1x5x1xf32>
    %1352 = tosa.add %1350, %1351 : (tensor<1x5x1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1353 = tosa.rsqrt %1352 : (tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1354 = tosa.sub %1340, %1344 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %1355 = tosa.mul %1354, %1353 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %1356 = tosa.reshape %arg141 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1357 = tosa.mul %1355, %1356 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %1358 = tosa.reshape %arg142 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1359 = tosa.add %1357, %1358 : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %1360 = tosa.reshape %1359 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1361 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1362 = tosa.transpose %arg143, %1361 : (tensor<3072x768xf32>, tensor<2xi32>) -> tensor<768x3072xf32>
    %1363 = tosa.reshape %1360 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1364 = tosa.reshape %1362 {new_shape = array<i64: 1, 768, 3072>} : (tensor<768x3072xf32>) -> tensor<1x768x3072xf32>
    %1365 = tosa.matmul %1363, %1364 : (tensor<1x5x768xf32>, tensor<1x768x3072xf32>) -> tensor<1x5x3072xf32>
    %1366 = tosa.reshape %1365 {new_shape = array<i64: 5, 3072>} : (tensor<1x5x3072xf32>) -> tensor<5x3072xf32>
    %1367 = tosa.reshape %arg144 {new_shape = array<i64: 1, 3072>} : (tensor<3072xf32>) -> tensor<1x3072xf32>
    %1368 = tosa.add %1367, %1366 : (tensor<1x3072xf32>, tensor<5x3072xf32>) -> tensor<5x3072xf32>
    %1369 = tosa.reshape %1368 {new_shape = array<i64: 1, 5, 3072>} : (tensor<5x3072xf32>) -> tensor<1x5x3072xf32>
    %1370 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x5x3072xf32>}> : () -> tensor<1x5x3072xf32>
    %1371 = tosa.mul %1369, %1370 {shift = 0 : i8} : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %1372 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x5x3072xf32>}> : () -> tensor<1x5x3072xf32>
    %1373 = tosa.mul %1369, %1372 {shift = 0 : i8} : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %1374 = math.erf %1373 : tensor<1x5x3072xf32>
    %1375 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x5x3072xf32>}> : () -> tensor<1x5x3072xf32>
    %1376 = tosa.add %1374, %1375 : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %1377 = tosa.mul %1371, %1376 {shift = 0 : i8} : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %1378 = tosa.reshape %1377 {new_shape = array<i64: 5, 3072>} : (tensor<1x5x3072xf32>) -> tensor<5x3072xf32>
    %1379 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1380 = tosa.transpose %arg145, %1379 : (tensor<768x3072xf32>, tensor<2xi32>) -> tensor<3072x768xf32>
    %1381 = tosa.reshape %1378 {new_shape = array<i64: 1, 5, 3072>} : (tensor<5x3072xf32>) -> tensor<1x5x3072xf32>
    %1382 = tosa.reshape %1380 {new_shape = array<i64: 1, 3072, 768>} : (tensor<3072x768xf32>) -> tensor<1x3072x768xf32>
    %1383 = tosa.matmul %1381, %1382 : (tensor<1x5x3072xf32>, tensor<1x3072x768xf32>) -> tensor<1x5x768xf32>
    %1384 = tosa.reshape %1383 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1385 = tosa.reshape %arg146 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %1386 = tosa.add %1385, %1384 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %1387 = tosa.reshape %1386 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1388 = tosa.identity %1387 : (tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %1389 = tosa.add %1388, %1359 : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %1390 = tosa.reduce_sum %1389 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %1391 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1392 = tosa.reciprocal %1391 : (tensor<1xf32>) -> tensor<1xf32>
    %1393 = tosa.mul %1392, %1390 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1394 = tosa.sub %1389, %1393 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %1395 = tosa.mul %1394, %1394 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %1396 = tosa.reduce_sum %1395 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %1397 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1398 = tosa.reciprocal %1397 : (tensor<1xf32>) -> tensor<1xf32>
    %1399 = tosa.mul %1398, %1396 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1400 = "tosa.const"() <{value = dense<9.99999996E-13> : tensor<1x5x1xf32>}> : () -> tensor<1x5x1xf32>
    %1401 = tosa.add %1399, %1400 : (tensor<1x5x1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1402 = tosa.rsqrt %1401 : (tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1403 = tosa.sub %1389, %1393 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %1404 = tosa.mul %1403, %1402 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %1405 = tosa.reshape %arg147 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1406 = tosa.mul %1404, %1405 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %1407 = tosa.reshape %arg148 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1408 = tosa.add %1406, %1407 : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %1409 = tosa.reshape %1408 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1410 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1411 = tosa.transpose %arg149, %1410 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %1412 = tosa.reshape %1409 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1413 = tosa.reshape %1411 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %1414 = tosa.matmul %1412, %1413 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %1415 = tosa.reshape %1414 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1416 = tosa.reshape %arg150 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %1417 = tosa.add %1416, %1415 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %1418 = tosa.reshape %1417 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1419 = tosa.reshape %1408 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1420 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1421 = tosa.transpose %arg151, %1420 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %1422 = tosa.reshape %1419 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1423 = tosa.reshape %1421 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %1424 = tosa.matmul %1422, %1423 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %1425 = tosa.reshape %1424 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1426 = tosa.reshape %arg152 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %1427 = tosa.add %1426, %1425 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %1428 = tosa.reshape %1427 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1429 = tosa.reshape %1428 {new_shape = array<i64: 1, 5, 12, 64>} : (tensor<1x5x768xf32>) -> tensor<1x5x12x64xf32>
    %1430 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1431 = tosa.transpose %1429, %1430 : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x12x5x64xf32>
    %1432 = tosa.reshape %1408 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1433 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1434 = tosa.transpose %arg153, %1433 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %1435 = tosa.reshape %1432 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1436 = tosa.reshape %1434 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %1437 = tosa.matmul %1435, %1436 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %1438 = tosa.reshape %1437 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1439 = tosa.reshape %arg154 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %1440 = tosa.add %1439, %1438 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %1441 = tosa.reshape %1440 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1442 = tosa.reshape %1441 {new_shape = array<i64: 1, 5, 12, 64>} : (tensor<1x5x768xf32>) -> tensor<1x5x12x64xf32>
    %1443 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1444 = tosa.transpose %1442, %1443 : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x12x5x64xf32>
    %1445 = tosa.reshape %1418 {new_shape = array<i64: 1, 5, 12, 64>} : (tensor<1x5x768xf32>) -> tensor<1x5x12x64xf32>
    %1446 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1447 = tosa.transpose %1445, %1446 : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x12x5x64xf32>
    %1448 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1449 = tosa.transpose %1431, %1448 : (tensor<1x12x5x64xf32>, tensor<4xi32>) -> tensor<1x12x64x5xf32>
    %1450 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x5x64xf32>}> : () -> tensor<1x12x5x64xf32>
    %1451 = tosa.add %1447, %1450 : (tensor<1x12x5x64xf32>, tensor<1x12x5x64xf32>) -> tensor<1x12x5x64xf32>
    %1452 = tosa.reshape %1451 {new_shape = array<i64: 12, 5, 64>} : (tensor<1x12x5x64xf32>) -> tensor<12x5x64xf32>
    %1453 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x64x5xf32>}> : () -> tensor<1x12x64x5xf32>
    %1454 = tosa.add %1449, %1453 : (tensor<1x12x64x5xf32>, tensor<1x12x64x5xf32>) -> tensor<1x12x64x5xf32>
    %1455 = tosa.reshape %1454 {new_shape = array<i64: 12, 64, 5>} : (tensor<1x12x64x5xf32>) -> tensor<12x64x5xf32>
    %1456 = tosa.matmul %1452, %1455 : (tensor<12x5x64xf32>, tensor<12x64x5xf32>) -> tensor<12x5x5xf32>
    %1457 = tosa.reshape %1456 {new_shape = array<i64: 1, 12, 5, 5>} : (tensor<12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %1458 = "tosa.const"() <{value = dense<8.000000e+00> : tensor<1x12x5x5xf32>}> : () -> tensor<1x12x5x5xf32>
    %1459 = tosa.reciprocal %1458 : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %1460 = tosa.mul %1457, %1459 {shift = 0 : i8} : (tensor<1x12x5x5xf32>, tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %1461 = tosa.add %1460, %6 : (tensor<1x12x5x5xf32>, tensor<1x1x1x5xf32>) -> tensor<1x12x5x5xf32>
    %1462 = tosa.reduce_max %1461 {axis = 3 : i32} : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x1xf32>
    %1463 = tosa.sub %1461, %1462 : (tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) -> tensor<1x12x5x5xf32>
    %1464 = tosa.exp %1463 : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %1465 = tosa.reduce_sum %1464 {axis = 3 : i32} : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x1xf32>
    %1466 = tosa.reciprocal %1465 : (tensor<1x12x5x1xf32>) -> tensor<1x12x5x1xf32>
    %1467 = tosa.mul %1464, %1466 {shift = 0 : i8} : (tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) -> tensor<1x12x5x5xf32>
    %1468 = tosa.identity %1467 : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %1469 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x5x5xf32>}> : () -> tensor<1x12x5x5xf32>
    %1470 = tosa.add %1468, %1469 : (tensor<1x12x5x5xf32>, tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %1471 = tosa.reshape %1470 {new_shape = array<i64: 12, 5, 5>} : (tensor<1x12x5x5xf32>) -> tensor<12x5x5xf32>
    %1472 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x5x64xf32>}> : () -> tensor<1x12x5x64xf32>
    %1473 = tosa.add %1444, %1472 : (tensor<1x12x5x64xf32>, tensor<1x12x5x64xf32>) -> tensor<1x12x5x64xf32>
    %1474 = tosa.reshape %1473 {new_shape = array<i64: 12, 5, 64>} : (tensor<1x12x5x64xf32>) -> tensor<12x5x64xf32>
    %1475 = tosa.matmul %1471, %1474 : (tensor<12x5x5xf32>, tensor<12x5x64xf32>) -> tensor<12x5x64xf32>
    %1476 = tosa.reshape %1475 {new_shape = array<i64: 1, 12, 5, 64>} : (tensor<12x5x64xf32>) -> tensor<1x12x5x64xf32>
    %1477 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1478 = tosa.transpose %1476, %1477 : (tensor<1x12x5x64xf32>, tensor<4xi32>) -> tensor<1x5x12x64xf32>
    %1479 = tosa.identity %1478 : (tensor<1x5x12x64xf32>) -> tensor<1x5x12x64xf32>
    %1480 = tosa.reshape %1479 {new_shape = array<i64: 1, 5, 768>} : (tensor<1x5x12x64xf32>) -> tensor<1x5x768xf32>
    %1481 = tosa.reshape %1480 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1482 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1483 = tosa.transpose %arg155, %1482 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %1484 = tosa.reshape %1481 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1485 = tosa.reshape %1483 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %1486 = tosa.matmul %1484, %1485 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %1487 = tosa.reshape %1486 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1488 = tosa.reshape %arg156 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %1489 = tosa.add %1488, %1487 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %1490 = tosa.reshape %1489 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1491 = tosa.identity %1490 : (tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %1492 = tosa.add %1491, %1408 : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %1493 = tosa.reduce_sum %1492 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %1494 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1495 = tosa.reciprocal %1494 : (tensor<1xf32>) -> tensor<1xf32>
    %1496 = tosa.mul %1495, %1493 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1497 = tosa.sub %1492, %1496 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %1498 = tosa.mul %1497, %1497 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %1499 = tosa.reduce_sum %1498 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %1500 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1501 = tosa.reciprocal %1500 : (tensor<1xf32>) -> tensor<1xf32>
    %1502 = tosa.mul %1501, %1499 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1503 = "tosa.const"() <{value = dense<9.99999996E-13> : tensor<1x5x1xf32>}> : () -> tensor<1x5x1xf32>
    %1504 = tosa.add %1502, %1503 : (tensor<1x5x1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1505 = tosa.rsqrt %1504 : (tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1506 = tosa.sub %1492, %1496 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %1507 = tosa.mul %1506, %1505 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %1508 = tosa.reshape %arg157 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1509 = tosa.mul %1507, %1508 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %1510 = tosa.reshape %arg158 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1511 = tosa.add %1509, %1510 : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %1512 = tosa.reshape %1511 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1513 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1514 = tosa.transpose %arg159, %1513 : (tensor<3072x768xf32>, tensor<2xi32>) -> tensor<768x3072xf32>
    %1515 = tosa.reshape %1512 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1516 = tosa.reshape %1514 {new_shape = array<i64: 1, 768, 3072>} : (tensor<768x3072xf32>) -> tensor<1x768x3072xf32>
    %1517 = tosa.matmul %1515, %1516 : (tensor<1x5x768xf32>, tensor<1x768x3072xf32>) -> tensor<1x5x3072xf32>
    %1518 = tosa.reshape %1517 {new_shape = array<i64: 5, 3072>} : (tensor<1x5x3072xf32>) -> tensor<5x3072xf32>
    %1519 = tosa.reshape %arg160 {new_shape = array<i64: 1, 3072>} : (tensor<3072xf32>) -> tensor<1x3072xf32>
    %1520 = tosa.add %1519, %1518 : (tensor<1x3072xf32>, tensor<5x3072xf32>) -> tensor<5x3072xf32>
    %1521 = tosa.reshape %1520 {new_shape = array<i64: 1, 5, 3072>} : (tensor<5x3072xf32>) -> tensor<1x5x3072xf32>
    %1522 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x5x3072xf32>}> : () -> tensor<1x5x3072xf32>
    %1523 = tosa.mul %1521, %1522 {shift = 0 : i8} : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %1524 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x5x3072xf32>}> : () -> tensor<1x5x3072xf32>
    %1525 = tosa.mul %1521, %1524 {shift = 0 : i8} : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %1526 = math.erf %1525 : tensor<1x5x3072xf32>
    %1527 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x5x3072xf32>}> : () -> tensor<1x5x3072xf32>
    %1528 = tosa.add %1526, %1527 : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %1529 = tosa.mul %1523, %1528 {shift = 0 : i8} : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %1530 = tosa.reshape %1529 {new_shape = array<i64: 5, 3072>} : (tensor<1x5x3072xf32>) -> tensor<5x3072xf32>
    %1531 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1532 = tosa.transpose %arg161, %1531 : (tensor<768x3072xf32>, tensor<2xi32>) -> tensor<3072x768xf32>
    %1533 = tosa.reshape %1530 {new_shape = array<i64: 1, 5, 3072>} : (tensor<5x3072xf32>) -> tensor<1x5x3072xf32>
    %1534 = tosa.reshape %1532 {new_shape = array<i64: 1, 3072, 768>} : (tensor<3072x768xf32>) -> tensor<1x3072x768xf32>
    %1535 = tosa.matmul %1533, %1534 : (tensor<1x5x3072xf32>, tensor<1x3072x768xf32>) -> tensor<1x5x768xf32>
    %1536 = tosa.reshape %1535 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1537 = tosa.reshape %arg162 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %1538 = tosa.add %1537, %1536 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %1539 = tosa.reshape %1538 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1540 = tosa.identity %1539 : (tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %1541 = tosa.add %1540, %1511 : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %1542 = tosa.reduce_sum %1541 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %1543 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1544 = tosa.reciprocal %1543 : (tensor<1xf32>) -> tensor<1xf32>
    %1545 = tosa.mul %1544, %1542 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1546 = tosa.sub %1541, %1545 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %1547 = tosa.mul %1546, %1546 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %1548 = tosa.reduce_sum %1547 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %1549 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1550 = tosa.reciprocal %1549 : (tensor<1xf32>) -> tensor<1xf32>
    %1551 = tosa.mul %1550, %1548 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1552 = "tosa.const"() <{value = dense<9.99999996E-13> : tensor<1x5x1xf32>}> : () -> tensor<1x5x1xf32>
    %1553 = tosa.add %1551, %1552 : (tensor<1x5x1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1554 = tosa.rsqrt %1553 : (tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1555 = tosa.sub %1541, %1545 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %1556 = tosa.mul %1555, %1554 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %1557 = tosa.reshape %arg163 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1558 = tosa.mul %1556, %1557 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %1559 = tosa.reshape %arg164 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1560 = tosa.add %1558, %1559 : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %1561 = tosa.reshape %1560 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1562 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1563 = tosa.transpose %arg165, %1562 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %1564 = tosa.reshape %1561 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1565 = tosa.reshape %1563 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %1566 = tosa.matmul %1564, %1565 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %1567 = tosa.reshape %1566 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1568 = tosa.reshape %arg166 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %1569 = tosa.add %1568, %1567 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %1570 = tosa.reshape %1569 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1571 = tosa.reshape %1560 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1572 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1573 = tosa.transpose %arg167, %1572 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %1574 = tosa.reshape %1571 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1575 = tosa.reshape %1573 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %1576 = tosa.matmul %1574, %1575 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %1577 = tosa.reshape %1576 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1578 = tosa.reshape %arg168 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %1579 = tosa.add %1578, %1577 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %1580 = tosa.reshape %1579 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1581 = tosa.reshape %1580 {new_shape = array<i64: 1, 5, 12, 64>} : (tensor<1x5x768xf32>) -> tensor<1x5x12x64xf32>
    %1582 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1583 = tosa.transpose %1581, %1582 : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x12x5x64xf32>
    %1584 = tosa.reshape %1560 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1585 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1586 = tosa.transpose %arg169, %1585 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %1587 = tosa.reshape %1584 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1588 = tosa.reshape %1586 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %1589 = tosa.matmul %1587, %1588 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %1590 = tosa.reshape %1589 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1591 = tosa.reshape %arg170 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %1592 = tosa.add %1591, %1590 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %1593 = tosa.reshape %1592 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1594 = tosa.reshape %1593 {new_shape = array<i64: 1, 5, 12, 64>} : (tensor<1x5x768xf32>) -> tensor<1x5x12x64xf32>
    %1595 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1596 = tosa.transpose %1594, %1595 : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x12x5x64xf32>
    %1597 = tosa.reshape %1570 {new_shape = array<i64: 1, 5, 12, 64>} : (tensor<1x5x768xf32>) -> tensor<1x5x12x64xf32>
    %1598 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1599 = tosa.transpose %1597, %1598 : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x12x5x64xf32>
    %1600 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1601 = tosa.transpose %1583, %1600 : (tensor<1x12x5x64xf32>, tensor<4xi32>) -> tensor<1x12x64x5xf32>
    %1602 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x5x64xf32>}> : () -> tensor<1x12x5x64xf32>
    %1603 = tosa.add %1599, %1602 : (tensor<1x12x5x64xf32>, tensor<1x12x5x64xf32>) -> tensor<1x12x5x64xf32>
    %1604 = tosa.reshape %1603 {new_shape = array<i64: 12, 5, 64>} : (tensor<1x12x5x64xf32>) -> tensor<12x5x64xf32>
    %1605 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x64x5xf32>}> : () -> tensor<1x12x64x5xf32>
    %1606 = tosa.add %1601, %1605 : (tensor<1x12x64x5xf32>, tensor<1x12x64x5xf32>) -> tensor<1x12x64x5xf32>
    %1607 = tosa.reshape %1606 {new_shape = array<i64: 12, 64, 5>} : (tensor<1x12x64x5xf32>) -> tensor<12x64x5xf32>
    %1608 = tosa.matmul %1604, %1607 : (tensor<12x5x64xf32>, tensor<12x64x5xf32>) -> tensor<12x5x5xf32>
    %1609 = tosa.reshape %1608 {new_shape = array<i64: 1, 12, 5, 5>} : (tensor<12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %1610 = "tosa.const"() <{value = dense<8.000000e+00> : tensor<1x12x5x5xf32>}> : () -> tensor<1x12x5x5xf32>
    %1611 = tosa.reciprocal %1610 : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %1612 = tosa.mul %1609, %1611 {shift = 0 : i8} : (tensor<1x12x5x5xf32>, tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %1613 = tosa.add %1612, %6 : (tensor<1x12x5x5xf32>, tensor<1x1x1x5xf32>) -> tensor<1x12x5x5xf32>
    %1614 = tosa.reduce_max %1613 {axis = 3 : i32} : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x1xf32>
    %1615 = tosa.sub %1613, %1614 : (tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) -> tensor<1x12x5x5xf32>
    %1616 = tosa.exp %1615 : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %1617 = tosa.reduce_sum %1616 {axis = 3 : i32} : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x1xf32>
    %1618 = tosa.reciprocal %1617 : (tensor<1x12x5x1xf32>) -> tensor<1x12x5x1xf32>
    %1619 = tosa.mul %1616, %1618 {shift = 0 : i8} : (tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) -> tensor<1x12x5x5xf32>
    %1620 = tosa.identity %1619 : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %1621 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x5x5xf32>}> : () -> tensor<1x12x5x5xf32>
    %1622 = tosa.add %1620, %1621 : (tensor<1x12x5x5xf32>, tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %1623 = tosa.reshape %1622 {new_shape = array<i64: 12, 5, 5>} : (tensor<1x12x5x5xf32>) -> tensor<12x5x5xf32>
    %1624 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x5x64xf32>}> : () -> tensor<1x12x5x64xf32>
    %1625 = tosa.add %1596, %1624 : (tensor<1x12x5x64xf32>, tensor<1x12x5x64xf32>) -> tensor<1x12x5x64xf32>
    %1626 = tosa.reshape %1625 {new_shape = array<i64: 12, 5, 64>} : (tensor<1x12x5x64xf32>) -> tensor<12x5x64xf32>
    %1627 = tosa.matmul %1623, %1626 : (tensor<12x5x5xf32>, tensor<12x5x64xf32>) -> tensor<12x5x64xf32>
    %1628 = tosa.reshape %1627 {new_shape = array<i64: 1, 12, 5, 64>} : (tensor<12x5x64xf32>) -> tensor<1x12x5x64xf32>
    %1629 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1630 = tosa.transpose %1628, %1629 : (tensor<1x12x5x64xf32>, tensor<4xi32>) -> tensor<1x5x12x64xf32>
    %1631 = tosa.identity %1630 : (tensor<1x5x12x64xf32>) -> tensor<1x5x12x64xf32>
    %1632 = tosa.reshape %1631 {new_shape = array<i64: 1, 5, 768>} : (tensor<1x5x12x64xf32>) -> tensor<1x5x768xf32>
    %1633 = tosa.reshape %1632 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1634 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1635 = tosa.transpose %arg171, %1634 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %1636 = tosa.reshape %1633 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1637 = tosa.reshape %1635 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %1638 = tosa.matmul %1636, %1637 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %1639 = tosa.reshape %1638 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1640 = tosa.reshape %arg172 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %1641 = tosa.add %1640, %1639 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %1642 = tosa.reshape %1641 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1643 = tosa.identity %1642 : (tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %1644 = tosa.add %1643, %1560 : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %1645 = tosa.reduce_sum %1644 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %1646 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1647 = tosa.reciprocal %1646 : (tensor<1xf32>) -> tensor<1xf32>
    %1648 = tosa.mul %1647, %1645 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1649 = tosa.sub %1644, %1648 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %1650 = tosa.mul %1649, %1649 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %1651 = tosa.reduce_sum %1650 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %1652 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1653 = tosa.reciprocal %1652 : (tensor<1xf32>) -> tensor<1xf32>
    %1654 = tosa.mul %1653, %1651 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1655 = "tosa.const"() <{value = dense<9.99999996E-13> : tensor<1x5x1xf32>}> : () -> tensor<1x5x1xf32>
    %1656 = tosa.add %1654, %1655 : (tensor<1x5x1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1657 = tosa.rsqrt %1656 : (tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1658 = tosa.sub %1644, %1648 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %1659 = tosa.mul %1658, %1657 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %1660 = tosa.reshape %arg173 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1661 = tosa.mul %1659, %1660 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %1662 = tosa.reshape %arg174 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1663 = tosa.add %1661, %1662 : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %1664 = tosa.reshape %1663 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1665 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1666 = tosa.transpose %arg175, %1665 : (tensor<3072x768xf32>, tensor<2xi32>) -> tensor<768x3072xf32>
    %1667 = tosa.reshape %1664 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1668 = tosa.reshape %1666 {new_shape = array<i64: 1, 768, 3072>} : (tensor<768x3072xf32>) -> tensor<1x768x3072xf32>
    %1669 = tosa.matmul %1667, %1668 : (tensor<1x5x768xf32>, tensor<1x768x3072xf32>) -> tensor<1x5x3072xf32>
    %1670 = tosa.reshape %1669 {new_shape = array<i64: 5, 3072>} : (tensor<1x5x3072xf32>) -> tensor<5x3072xf32>
    %1671 = tosa.reshape %arg176 {new_shape = array<i64: 1, 3072>} : (tensor<3072xf32>) -> tensor<1x3072xf32>
    %1672 = tosa.add %1671, %1670 : (tensor<1x3072xf32>, tensor<5x3072xf32>) -> tensor<5x3072xf32>
    %1673 = tosa.reshape %1672 {new_shape = array<i64: 1, 5, 3072>} : (tensor<5x3072xf32>) -> tensor<1x5x3072xf32>
    %1674 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x5x3072xf32>}> : () -> tensor<1x5x3072xf32>
    %1675 = tosa.mul %1673, %1674 {shift = 0 : i8} : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %1676 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x5x3072xf32>}> : () -> tensor<1x5x3072xf32>
    %1677 = tosa.mul %1673, %1676 {shift = 0 : i8} : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %1678 = math.erf %1677 : tensor<1x5x3072xf32>
    %1679 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x5x3072xf32>}> : () -> tensor<1x5x3072xf32>
    %1680 = tosa.add %1678, %1679 : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %1681 = tosa.mul %1675, %1680 {shift = 0 : i8} : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %1682 = tosa.reshape %1681 {new_shape = array<i64: 5, 3072>} : (tensor<1x5x3072xf32>) -> tensor<5x3072xf32>
    %1683 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1684 = tosa.transpose %arg177, %1683 : (tensor<768x3072xf32>, tensor<2xi32>) -> tensor<3072x768xf32>
    %1685 = tosa.reshape %1682 {new_shape = array<i64: 1, 5, 3072>} : (tensor<5x3072xf32>) -> tensor<1x5x3072xf32>
    %1686 = tosa.reshape %1684 {new_shape = array<i64: 1, 3072, 768>} : (tensor<3072x768xf32>) -> tensor<1x3072x768xf32>
    %1687 = tosa.matmul %1685, %1686 : (tensor<1x5x3072xf32>, tensor<1x3072x768xf32>) -> tensor<1x5x768xf32>
    %1688 = tosa.reshape %1687 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1689 = tosa.reshape %arg178 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %1690 = tosa.add %1689, %1688 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %1691 = tosa.reshape %1690 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1692 = tosa.identity %1691 : (tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %1693 = tosa.add %1692, %1663 : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %1694 = tosa.reduce_sum %1693 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %1695 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1696 = tosa.reciprocal %1695 : (tensor<1xf32>) -> tensor<1xf32>
    %1697 = tosa.mul %1696, %1694 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1698 = tosa.sub %1693, %1697 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %1699 = tosa.mul %1698, %1698 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %1700 = tosa.reduce_sum %1699 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %1701 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1702 = tosa.reciprocal %1701 : (tensor<1xf32>) -> tensor<1xf32>
    %1703 = tosa.mul %1702, %1700 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1704 = "tosa.const"() <{value = dense<9.99999996E-13> : tensor<1x5x1xf32>}> : () -> tensor<1x5x1xf32>
    %1705 = tosa.add %1703, %1704 : (tensor<1x5x1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1706 = tosa.rsqrt %1705 : (tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1707 = tosa.sub %1693, %1697 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %1708 = tosa.mul %1707, %1706 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %1709 = tosa.reshape %arg179 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1710 = tosa.mul %1708, %1709 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %1711 = tosa.reshape %arg180 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1712 = tosa.add %1710, %1711 : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %1713 = tosa.reshape %1712 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1714 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1715 = tosa.transpose %arg181, %1714 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %1716 = tosa.reshape %1713 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1717 = tosa.reshape %1715 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %1718 = tosa.matmul %1716, %1717 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %1719 = tosa.reshape %1718 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1720 = tosa.reshape %arg182 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %1721 = tosa.add %1720, %1719 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %1722 = tosa.reshape %1721 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1723 = tosa.reshape %1712 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1724 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1725 = tosa.transpose %arg183, %1724 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %1726 = tosa.reshape %1723 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1727 = tosa.reshape %1725 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %1728 = tosa.matmul %1726, %1727 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %1729 = tosa.reshape %1728 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1730 = tosa.reshape %arg184 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %1731 = tosa.add %1730, %1729 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %1732 = tosa.reshape %1731 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1733 = tosa.reshape %1732 {new_shape = array<i64: 1, 5, 12, 64>} : (tensor<1x5x768xf32>) -> tensor<1x5x12x64xf32>
    %1734 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1735 = tosa.transpose %1733, %1734 : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x12x5x64xf32>
    %1736 = tosa.reshape %1712 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1737 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1738 = tosa.transpose %arg185, %1737 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %1739 = tosa.reshape %1736 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1740 = tosa.reshape %1738 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %1741 = tosa.matmul %1739, %1740 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %1742 = tosa.reshape %1741 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1743 = tosa.reshape %arg186 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %1744 = tosa.add %1743, %1742 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %1745 = tosa.reshape %1744 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1746 = tosa.reshape %1745 {new_shape = array<i64: 1, 5, 12, 64>} : (tensor<1x5x768xf32>) -> tensor<1x5x12x64xf32>
    %1747 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1748 = tosa.transpose %1746, %1747 : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x12x5x64xf32>
    %1749 = tosa.reshape %1722 {new_shape = array<i64: 1, 5, 12, 64>} : (tensor<1x5x768xf32>) -> tensor<1x5x12x64xf32>
    %1750 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1751 = tosa.transpose %1749, %1750 : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x12x5x64xf32>
    %1752 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1753 = tosa.transpose %1735, %1752 : (tensor<1x12x5x64xf32>, tensor<4xi32>) -> tensor<1x12x64x5xf32>
    %1754 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x5x64xf32>}> : () -> tensor<1x12x5x64xf32>
    %1755 = tosa.add %1751, %1754 : (tensor<1x12x5x64xf32>, tensor<1x12x5x64xf32>) -> tensor<1x12x5x64xf32>
    %1756 = tosa.reshape %1755 {new_shape = array<i64: 12, 5, 64>} : (tensor<1x12x5x64xf32>) -> tensor<12x5x64xf32>
    %1757 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x64x5xf32>}> : () -> tensor<1x12x64x5xf32>
    %1758 = tosa.add %1753, %1757 : (tensor<1x12x64x5xf32>, tensor<1x12x64x5xf32>) -> tensor<1x12x64x5xf32>
    %1759 = tosa.reshape %1758 {new_shape = array<i64: 12, 64, 5>} : (tensor<1x12x64x5xf32>) -> tensor<12x64x5xf32>
    %1760 = tosa.matmul %1756, %1759 : (tensor<12x5x64xf32>, tensor<12x64x5xf32>) -> tensor<12x5x5xf32>
    %1761 = tosa.reshape %1760 {new_shape = array<i64: 1, 12, 5, 5>} : (tensor<12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %1762 = "tosa.const"() <{value = dense<8.000000e+00> : tensor<1x12x5x5xf32>}> : () -> tensor<1x12x5x5xf32>
    %1763 = tosa.reciprocal %1762 : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %1764 = tosa.mul %1761, %1763 {shift = 0 : i8} : (tensor<1x12x5x5xf32>, tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %1765 = tosa.add %1764, %6 : (tensor<1x12x5x5xf32>, tensor<1x1x1x5xf32>) -> tensor<1x12x5x5xf32>
    %1766 = tosa.reduce_max %1765 {axis = 3 : i32} : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x1xf32>
    %1767 = tosa.sub %1765, %1766 : (tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) -> tensor<1x12x5x5xf32>
    %1768 = tosa.exp %1767 : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %1769 = tosa.reduce_sum %1768 {axis = 3 : i32} : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x1xf32>
    %1770 = tosa.reciprocal %1769 : (tensor<1x12x5x1xf32>) -> tensor<1x12x5x1xf32>
    %1771 = tosa.mul %1768, %1770 {shift = 0 : i8} : (tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) -> tensor<1x12x5x5xf32>
    %1772 = tosa.identity %1771 : (tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %1773 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x5x5xf32>}> : () -> tensor<1x12x5x5xf32>
    %1774 = tosa.add %1772, %1773 : (tensor<1x12x5x5xf32>, tensor<1x12x5x5xf32>) -> tensor<1x12x5x5xf32>
    %1775 = tosa.reshape %1774 {new_shape = array<i64: 12, 5, 5>} : (tensor<1x12x5x5xf32>) -> tensor<12x5x5xf32>
    %1776 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x12x5x64xf32>}> : () -> tensor<1x12x5x64xf32>
    %1777 = tosa.add %1748, %1776 : (tensor<1x12x5x64xf32>, tensor<1x12x5x64xf32>) -> tensor<1x12x5x64xf32>
    %1778 = tosa.reshape %1777 {new_shape = array<i64: 12, 5, 64>} : (tensor<1x12x5x64xf32>) -> tensor<12x5x64xf32>
    %1779 = tosa.matmul %1775, %1778 : (tensor<12x5x5xf32>, tensor<12x5x64xf32>) -> tensor<12x5x64xf32>
    %1780 = tosa.reshape %1779 {new_shape = array<i64: 1, 12, 5, 64>} : (tensor<12x5x64xf32>) -> tensor<1x12x5x64xf32>
    %1781 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1782 = tosa.transpose %1780, %1781 : (tensor<1x12x5x64xf32>, tensor<4xi32>) -> tensor<1x5x12x64xf32>
    %1783 = tosa.identity %1782 : (tensor<1x5x12x64xf32>) -> tensor<1x5x12x64xf32>
    %1784 = tosa.reshape %1783 {new_shape = array<i64: 1, 5, 768>} : (tensor<1x5x12x64xf32>) -> tensor<1x5x768xf32>
    %1785 = tosa.reshape %1784 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1786 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1787 = tosa.transpose %arg187, %1786 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %1788 = tosa.reshape %1785 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1789 = tosa.reshape %1787 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %1790 = tosa.matmul %1788, %1789 : (tensor<1x5x768xf32>, tensor<1x768x768xf32>) -> tensor<1x5x768xf32>
    %1791 = tosa.reshape %1790 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1792 = tosa.reshape %arg188 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %1793 = tosa.add %1792, %1791 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %1794 = tosa.reshape %1793 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1795 = tosa.identity %1794 : (tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %1796 = tosa.add %1795, %1712 : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %1797 = tosa.reduce_sum %1796 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %1798 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1799 = tosa.reciprocal %1798 : (tensor<1xf32>) -> tensor<1xf32>
    %1800 = tosa.mul %1799, %1797 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1801 = tosa.sub %1796, %1800 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %1802 = tosa.mul %1801, %1801 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %1803 = tosa.reduce_sum %1802 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %1804 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1805 = tosa.reciprocal %1804 : (tensor<1xf32>) -> tensor<1xf32>
    %1806 = tosa.mul %1805, %1803 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1807 = "tosa.const"() <{value = dense<9.99999996E-13> : tensor<1x5x1xf32>}> : () -> tensor<1x5x1xf32>
    %1808 = tosa.add %1806, %1807 : (tensor<1x5x1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1809 = tosa.rsqrt %1808 : (tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1810 = tosa.sub %1796, %1800 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %1811 = tosa.mul %1810, %1809 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %1812 = tosa.reshape %arg189 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1813 = tosa.mul %1811, %1812 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %1814 = tosa.reshape %arg190 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1815 = tosa.add %1813, %1814 : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %1816 = tosa.reshape %1815 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1817 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1818 = tosa.transpose %arg191, %1817 : (tensor<3072x768xf32>, tensor<2xi32>) -> tensor<768x3072xf32>
    %1819 = tosa.reshape %1816 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1820 = tosa.reshape %1818 {new_shape = array<i64: 1, 768, 3072>} : (tensor<768x3072xf32>) -> tensor<1x768x3072xf32>
    %1821 = tosa.matmul %1819, %1820 : (tensor<1x5x768xf32>, tensor<1x768x3072xf32>) -> tensor<1x5x3072xf32>
    %1822 = tosa.reshape %1821 {new_shape = array<i64: 5, 3072>} : (tensor<1x5x3072xf32>) -> tensor<5x3072xf32>
    %1823 = tosa.reshape %arg192 {new_shape = array<i64: 1, 3072>} : (tensor<3072xf32>) -> tensor<1x3072xf32>
    %1824 = tosa.add %1823, %1822 : (tensor<1x3072xf32>, tensor<5x3072xf32>) -> tensor<5x3072xf32>
    %1825 = tosa.reshape %1824 {new_shape = array<i64: 1, 5, 3072>} : (tensor<5x3072xf32>) -> tensor<1x5x3072xf32>
    %1826 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x5x3072xf32>}> : () -> tensor<1x5x3072xf32>
    %1827 = tosa.mul %1825, %1826 {shift = 0 : i8} : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %1828 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x5x3072xf32>}> : () -> tensor<1x5x3072xf32>
    %1829 = tosa.mul %1825, %1828 {shift = 0 : i8} : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %1830 = math.erf %1829 : tensor<1x5x3072xf32>
    %1831 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x5x3072xf32>}> : () -> tensor<1x5x3072xf32>
    %1832 = tosa.add %1830, %1831 : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %1833 = tosa.mul %1827, %1832 {shift = 0 : i8} : (tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) -> tensor<1x5x3072xf32>
    %1834 = tosa.reshape %1833 {new_shape = array<i64: 5, 3072>} : (tensor<1x5x3072xf32>) -> tensor<5x3072xf32>
    %1835 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1836 = tosa.transpose %arg193, %1835 : (tensor<768x3072xf32>, tensor<2xi32>) -> tensor<3072x768xf32>
    %1837 = tosa.reshape %1834 {new_shape = array<i64: 1, 5, 3072>} : (tensor<5x3072xf32>) -> tensor<1x5x3072xf32>
    %1838 = tosa.reshape %1836 {new_shape = array<i64: 1, 3072, 768>} : (tensor<3072x768xf32>) -> tensor<1x3072x768xf32>
    %1839 = tosa.matmul %1837, %1838 : (tensor<1x5x3072xf32>, tensor<1x3072x768xf32>) -> tensor<1x5x768xf32>
    %1840 = tosa.reshape %1839 {new_shape = array<i64: 5, 768>} : (tensor<1x5x768xf32>) -> tensor<5x768xf32>
    %1841 = tosa.reshape %arg194 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %1842 = tosa.add %1841, %1840 : (tensor<1x768xf32>, tensor<5x768xf32>) -> tensor<5x768xf32>
    %1843 = tosa.reshape %1842 {new_shape = array<i64: 1, 5, 768>} : (tensor<5x768xf32>) -> tensor<1x5x768xf32>
    %1844 = tosa.identity %1843 : (tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %1845 = tosa.add %1844, %1815 : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %1846 = tosa.reduce_sum %1845 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %1847 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1848 = tosa.reciprocal %1847 : (tensor<1xf32>) -> tensor<1xf32>
    %1849 = tosa.mul %1848, %1846 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1850 = tosa.sub %1845, %1849 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %1851 = tosa.mul %1850, %1850 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x768xf32>) -> tensor<1x5x768xf32>
    %1852 = tosa.reduce_sum %1851 {axis = 2 : i32} : (tensor<1x5x768xf32>) -> tensor<1x5x1xf32>
    %1853 = "tosa.const"() <{value = dense<7.680000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1854 = tosa.reciprocal %1853 : (tensor<1xf32>) -> tensor<1xf32>
    %1855 = tosa.mul %1854, %1852 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1856 = "tosa.const"() <{value = dense<9.99999996E-13> : tensor<1x5x1xf32>}> : () -> tensor<1x5x1xf32>
    %1857 = tosa.add %1855, %1856 : (tensor<1x5x1xf32>, tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1858 = tosa.rsqrt %1857 : (tensor<1x5x1xf32>) -> tensor<1x5x1xf32>
    %1859 = tosa.sub %1845, %1849 : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %1860 = tosa.mul %1859, %1858 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x5x1xf32>) -> tensor<1x5x768xf32>
    %1861 = tosa.reshape %arg195 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1862 = tosa.mul %1860, %1861 {shift = 0 : i8} : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %1863 = tosa.reshape %arg196 {new_shape = array<i64: 1, 1, 768>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1864 = tosa.add %1862, %1863 : (tensor<1x5x768xf32>, tensor<1x1x768xf32>) -> tensor<1x5x768xf32>
    %extracted_slice_3 = tensor.extract_slice %1864[0, 0, 0] [1, 5, 768] [1, 1, 1] : tensor<1x5x768xf32> to tensor<1x5x768xf32>
    %1865 = tosa.slice %extracted_slice_3 {size = array<i64: 1, 1, 768>, start = array<i64: 0, 0, 0>} : (tensor<1x5x768xf32>) -> tensor<1x1x768xf32>
    %1866 = tosa.reshape %1865 {new_shape = array<i64: 1, 768>} : (tensor<1x1x768xf32>) -> tensor<1x768xf32>
    %1867 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1868 = tosa.transpose %arg197, %1867 : (tensor<768x768xf32>, tensor<2xi32>) -> tensor<768x768xf32>
    %1869 = tosa.reshape %1866 {new_shape = array<i64: 1, 1, 768>} : (tensor<1x768xf32>) -> tensor<1x1x768xf32>
    %1870 = tosa.reshape %1868 {new_shape = array<i64: 1, 768, 768>} : (tensor<768x768xf32>) -> tensor<1x768x768xf32>
    %1871 = tosa.matmul %1869, %1870 : (tensor<1x1x768xf32>, tensor<1x768x768xf32>) -> tensor<1x1x768xf32>
    %1872 = tosa.reshape %1871 {new_shape = array<i64: 1, 768>} : (tensor<1x1x768xf32>) -> tensor<1x768xf32>
    %1873 = tosa.reshape %arg198 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %1874 = tosa.add %1873, %1872 : (tensor<1x768xf32>, tensor<1x768xf32>) -> tensor<1x768xf32>
    %1875 = tosa.tanh %1874 : (tensor<1x768xf32>) -> tensor<1x768xf32>
    %1876 = tosa.identity %1875 : (tensor<1x768xf32>) -> tensor<1x768xf32>
    %1877 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1878 = tosa.transpose %arg199, %1877 : (tensor<6x768xf32>, tensor<2xi32>) -> tensor<768x6xf32>
    %1879 = tosa.reshape %1876 {new_shape = array<i64: 1, 1, 768>} : (tensor<1x768xf32>) -> tensor<1x1x768xf32>
    %1880 = tosa.reshape %1878 {new_shape = array<i64: 1, 768, 6>} : (tensor<768x6xf32>) -> tensor<1x768x6xf32>
    %1881 = tosa.matmul %1879, %1880 : (tensor<1x1x768xf32>, tensor<1x768x6xf32>) -> tensor<1x1x6xf32>
    %1882 = tosa.reshape %1881 {new_shape = array<i64: 1, 6>} : (tensor<1x1x6xf32>) -> tensor<1x6xf32>
    %1883 = tosa.reshape %arg200 {new_shape = array<i64: 1, 6>} : (tensor<6xf32>) -> tensor<1x6xf32>
    %1884 = tosa.add %1883, %1882 : (tensor<1x6xf32>, tensor<1x6xf32>) -> tensor<1x6xf32>
    return %1884 : tensor<1x6xf32>
  }
}
