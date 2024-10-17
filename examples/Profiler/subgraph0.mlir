module {
  func.func @subgraph0(%arg0: tensor<1x3x224x224xf32>, %arg1: tensor<16x3x3x3xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16xf32>, %arg4: tensor<16xf32>, %arg5: tensor<16xf32>, %arg6: tensor<16x1x3x3xf32>, %arg7: tensor<16xf32>, %arg8: tensor<16xf32>, %arg9: tensor<16xf32>, %arg10: tensor<16xf32>, %arg11: tensor<8x16x1x1xf32>, %arg12: tensor<8xf32>, %arg13: tensor<16x8x1x1xf32>, %arg14: tensor<16xf32>, %arg15: tensor<16x16x1x1xf32>, %arg16: tensor<16xf32>, %arg17: tensor<16xf32>, %arg18: tensor<16xf32>, %arg19: tensor<16xf32>, %arg20: tensor<72x16x1x1xf32>, %arg21: tensor<72xf32>, %arg22: tensor<72xf32>, %arg23: tensor<72xf32>, %arg24: tensor<72xf32>, %arg25: tensor<72x1x3x3xf32>, %arg26: tensor<72xf32>, %arg27: tensor<72xf32>, %arg28: tensor<72xf32>, %arg29: tensor<72xf32>, %arg30: tensor<24x72x1x1xf32>, %arg31: tensor<24xf32>, %arg32: tensor<24xf32>, %arg33: tensor<24xf32>, %arg34: tensor<24xf32>, %arg35: tensor<88x24x1x1xf32>, %arg36: tensor<88xf32>, %arg37: tensor<88xf32>, %arg38: tensor<88xf32>, %arg39: tensor<88xf32>, %arg40: tensor<88x1x3x3xf32>, %arg41: tensor<88xf32>, %arg42: tensor<88xf32>, %arg43: tensor<88xf32>, %arg44: tensor<88xf32>, %arg45: tensor<24x88x1x1xf32>, %arg46: tensor<24xf32>, %arg47: tensor<24xf32>, %arg48: tensor<24xf32>, %arg49: tensor<24xf32>, %arg50: tensor<96x24x1x1xf32>, %arg51: tensor<96xf32>, %arg52: tensor<96xf32>, %arg53: tensor<96xf32>, %arg54: tensor<96xf32>, %arg55: tensor<96x1x5x5xf32>, %arg56: tensor<96xf32>, %arg57: tensor<96xf32>, %arg58: tensor<96xf32>, %arg59: tensor<96xf32>, %arg60: tensor<24x96x1x1xf32>, %arg61: tensor<24xf32>, %arg62: tensor<96x24x1x1xf32>, %arg63: tensor<96xf32>, %arg64: tensor<40x96x1x1xf32>, %arg65: tensor<40xf32>, %arg66: tensor<40xf32>, %arg67: tensor<40xf32>, %arg68: tensor<40xf32>, %arg69: tensor<240x40x1x1xf32>, %arg70: tensor<240xf32>, %arg71: tensor<240xf32>, %arg72: tensor<240xf32>, %arg73: tensor<240xf32>, %arg74: tensor<240x1x5x5xf32>, %arg75: tensor<240xf32>, %arg76: tensor<240xf32>, %arg77: tensor<240xf32>, %arg78: tensor<240xf32>, %arg79: tensor<64x240x1x1xf32>, %arg80: tensor<64xf32>, %arg81: tensor<240x64x1x1xf32>, %arg82: tensor<240xf32>, %arg83: tensor<40x240x1x1xf32>, %arg84: tensor<40xf32>, %arg85: tensor<40xf32>, %arg86: tensor<40xf32>, %arg87: tensor<40xf32>, %arg88: tensor<240x40x1x1xf32>, %arg89: tensor<240xf32>, %arg90: tensor<240xf32>, %arg91: tensor<240xf32>, %arg92: tensor<240xf32>, %arg93: tensor<240x1x5x5xf32>, %arg94: tensor<240xf32>, %arg95: tensor<240xf32>, %arg96: tensor<240xf32>, %arg97: tensor<240xf32>, %arg98: tensor<64x240x1x1xf32>, %arg99: tensor<64xf32>, %arg100: tensor<240x64x1x1xf32>, %arg101: tensor<240xf32>, %arg102: tensor<40x240x1x1xf32>, %arg103: tensor<40xf32>, %arg104: tensor<40xf32>, %arg105: tensor<40xf32>, %arg106: tensor<40xf32>, %arg107: tensor<120x40x1x1xf32>, %arg108: tensor<120xf32>, %arg109: tensor<120xf32>, %arg110: tensor<120xf32>, %arg111: tensor<120xf32>, %arg112: tensor<120x1x5x5xf32>, %arg113: tensor<120xf32>, %arg114: tensor<120xf32>, %arg115: tensor<120xf32>, %arg116: tensor<120xf32>, %arg117: tensor<32x120x1x1xf32>, %arg118: tensor<32xf32>, %arg119: tensor<120x32x1x1xf32>, %arg120: tensor<120xf32>, %arg121: tensor<48x120x1x1xf32>, %arg122: tensor<48xf32>, %arg123: tensor<48xf32>, %arg124: tensor<48xf32>, %arg125: tensor<48xf32>, %arg126: tensor<144x48x1x1xf32>, %arg127: tensor<144xf32>, %arg128: tensor<144xf32>, %arg129: tensor<144xf32>, %arg130: tensor<144xf32>, %arg131: tensor<144x1x5x5xf32>, %arg132: tensor<144xf32>, %arg133: tensor<144xf32>, %arg134: tensor<144xf32>, %arg135: tensor<144xf32>, %arg136: tensor<40x144x1x1xf32>, %arg137: tensor<40xf32>, %arg138: tensor<144x40x1x1xf32>, %arg139: tensor<144xf32>, %arg140: tensor<48x144x1x1xf32>, %arg141: tensor<48xf32>, %arg142: tensor<48xf32>, %arg143: tensor<48xf32>, %arg144: tensor<48xf32>, %arg145: tensor<288x48x1x1xf32>, %arg146: tensor<288xf32>, %arg147: tensor<288xf32>, %arg148: tensor<288xf32>, %arg149: tensor<288xf32>, %arg150: tensor<288x1x5x5xf32>, %arg151: tensor<288xf32>, %arg152: tensor<288xf32>, %arg153: tensor<288xf32>, %arg154: tensor<288xf32>, %arg155: tensor<72x288x1x1xf32>, %arg156: tensor<72xf32>, %arg157: tensor<288x72x1x1xf32>, %arg158: tensor<288xf32>, %arg159: tensor<96x288x1x1xf32>, %arg160: tensor<96xf32>, %arg161: tensor<96xf32>, %arg162: tensor<96xf32>, %arg163: tensor<96xf32>, %arg164: tensor<576x96x1x1xf32>, %arg165: tensor<576xf32>, %arg166: tensor<576xf32>, %arg167: tensor<576xf32>, %arg168: tensor<576xf32>, %arg169: tensor<576x1x5x5xf32>, %arg170: tensor<576xf32>, %arg171: tensor<576xf32>, %arg172: tensor<576xf32>, %arg173: tensor<576xf32>, %arg174: tensor<144x576x1x1xf32>, %arg175: tensor<144xf32>, %arg176: tensor<576x144x1x1xf32>, %arg177: tensor<576xf32>, %arg178: tensor<96x576x1x1xf32>, %arg179: tensor<96xf32>, %arg180: tensor<96xf32>, %arg181: tensor<96xf32>, %arg182: tensor<96xf32>, %arg183: tensor<576x96x1x1xf32>, %arg184: tensor<576xf32>, %arg185: tensor<576xf32>, %arg186: tensor<576xf32>, %arg187: tensor<576xf32>, %arg188: tensor<576x1x5x5xf32>, %arg189: tensor<576xf32>, %arg190: tensor<576xf32>, %arg191: tensor<576xf32>, %arg192: tensor<576xf32>, %arg193: tensor<144x576x1x1xf32>, %arg194: tensor<144xf32>, %arg195: tensor<576x144x1x1xf32>, %arg196: tensor<576xf32>, %arg197: tensor<96x576x1x1xf32>, %arg198: tensor<96xf32>, %arg199: tensor<96xf32>, %arg200: tensor<96xf32>, %arg201: tensor<96xf32>, %arg202: tensor<576x96x1x1xf32>, %arg203: tensor<576xf32>, %arg204: tensor<576xf32>, %arg205: tensor<576xf32>, %arg206: tensor<576xf32>, %arg207: tensor<1024x576xf32>, %arg208: tensor<1024xf32>, %arg209: tensor<1000x1024xf32>, %arg210: tensor<1000xf32>) -> tensor<1x1000xf32> {
    %0 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<16xf32>}> : () -> tensor<16xf32>
    %1 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2 = tosa.transpose %arg0, %1 : (tensor<1x3x224x224xf32>, tensor<4xi32>) -> tensor<1x224x224x3xf32>
    %3 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4 = tosa.transpose %arg1, %3 : (tensor<16x3x3x3xf32>, tensor<4xi32>) -> tensor<16x3x3x3xf32>
    %5 = tosa.conv2d %2, %4, %0 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>} : (tensor<1x224x224x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<1x112x112x16xf32>
    %6 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %7 = tosa.transpose %5, %6 : (tensor<1x112x112x16xf32>, tensor<4xi32>) -> tensor<1x16x112x112xf32>
    %8 = tosa.cast %arg2 : (tensor<16xf32>) -> tensor<16xf32>
    %9 = tosa.cast %arg3 : (tensor<16xf32>) -> tensor<16xf32>
    %10 = "tosa.const"() <{value = dense<1.000000e-03> : tensor<16xf32>}> : () -> tensor<16xf32>
    %11 = tosa.add %9, %10 : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    %12 = math.sqrt %11 : tensor<16xf32>
    %13 = tosa.reciprocal %12 : (tensor<16xf32>) -> tensor<16xf32>
    %14 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<16xf32>}> : () -> tensor<16xf32>
    %15 = tosa.mul %13, %14 {shift = 0 : i8} : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    %16 = tosa.reshape %8 {new_shape = array<i64: 16, 1>} : (tensor<16xf32>) -> tensor<16x1xf32>
    %17 = tosa.reshape %16 {new_shape = array<i64: 16, 1, 1>} : (tensor<16x1xf32>) -> tensor<16x1x1xf32>
    %18 = tosa.reshape %15 {new_shape = array<i64: 16, 1>} : (tensor<16xf32>) -> tensor<16x1xf32>
    %19 = tosa.reshape %18 {new_shape = array<i64: 16, 1, 1>} : (tensor<16x1xf32>) -> tensor<16x1x1xf32>
    %20 = tosa.reshape %17 {new_shape = array<i64: 1, 16, 1, 1>} : (tensor<16x1x1xf32>) -> tensor<1x16x1x1xf32>
    %21 = tosa.sub %7, %20 : (tensor<1x16x112x112xf32>, tensor<1x16x1x1xf32>) -> tensor<1x16x112x112xf32>
    %22 = tosa.reshape %19 {new_shape = array<i64: 1, 16, 1, 1>} : (tensor<16x1x1xf32>) -> tensor<1x16x1x1xf32>
    %23 = tosa.mul %21, %22 {shift = 0 : i8} : (tensor<1x16x112x112xf32>, tensor<1x16x1x1xf32>) -> tensor<1x16x112x112xf32>
    %24 = tosa.reshape %arg4 {new_shape = array<i64: 16, 1>} : (tensor<16xf32>) -> tensor<16x1xf32>
    %25 = tosa.reshape %24 {new_shape = array<i64: 16, 1, 1>} : (tensor<16x1xf32>) -> tensor<16x1x1xf32>
    %26 = tosa.reshape %25 {new_shape = array<i64: 1, 16, 1, 1>} : (tensor<16x1x1xf32>) -> tensor<1x16x1x1xf32>
    %27 = tosa.mul %23, %26 {shift = 0 : i8} : (tensor<1x16x112x112xf32>, tensor<1x16x1x1xf32>) -> tensor<1x16x112x112xf32>
    %28 = tosa.reshape %arg5 {new_shape = array<i64: 16, 1>} : (tensor<16xf32>) -> tensor<16x1xf32>
    %29 = tosa.reshape %28 {new_shape = array<i64: 16, 1, 1>} : (tensor<16x1xf32>) -> tensor<16x1x1xf32>
    %30 = tosa.reshape %29 {new_shape = array<i64: 1, 16, 1, 1>} : (tensor<16x1x1xf32>) -> tensor<1x16x1x1xf32>
    %31 = tosa.add %27, %30 : (tensor<1x16x112x112xf32>, tensor<1x16x1x1xf32>) -> tensor<1x16x112x112xf32>
    %32 = "tosa.const"() <{value = dense<3.000000e+00> : tensor<1x16x112x112xf32>}> : () -> tensor<1x16x112x112xf32>
    %33 = tosa.add %31, %32 : (tensor<1x16x112x112xf32>, tensor<1x16x112x112xf32>) -> tensor<1x16x112x112xf32>
    %34 = tosa.clamp %33 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x16x112x112xf32>) -> tensor<1x16x112x112xf32>
    %35 = tosa.clamp %34 {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0xFF800000 : f32, min_int = -9223372036854775807 : i64} : (tensor<1x16x112x112xf32>) -> tensor<1x16x112x112xf32>
    %36 = tosa.mul %31, %35 {shift = 0 : i8} : (tensor<1x16x112x112xf32>, tensor<1x16x112x112xf32>) -> tensor<1x16x112x112xf32>
    %37 = "tosa.const"() <{value = dense<6.000000e+00> : tensor<1x16x112x112xf32>}> : () -> tensor<1x16x112x112xf32>
    %38 = tosa.reciprocal %37 : (tensor<1x16x112x112xf32>) -> tensor<1x16x112x112xf32>
    %39 = tosa.mul %36, %38 {shift = 0 : i8} : (tensor<1x16x112x112xf32>, tensor<1x16x112x112xf32>) -> tensor<1x16x112x112xf32>
    %40 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<16xf32>}> : () -> tensor<16xf32>
    %41 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %42 = tosa.transpose %39, %41 : (tensor<1x16x112x112xf32>, tensor<4xi32>) -> tensor<1x112x112x16xf32>
    %43 = "tosa.const"() <{value = dense<[2, 3, 0, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %44 = tosa.transpose %arg6, %43 : (tensor<16x1x3x3xf32>, tensor<4xi32>) -> tensor<3x3x16x1xf32>
    %45 = tosa.depthwise_conv2d %42, %44, %40 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>} : (tensor<1x112x112x16xf32>, tensor<3x3x16x1xf32>, tensor<16xf32>) -> tensor<1x56x56x16xf32>
    %46 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %47 = tosa.transpose %45, %46 : (tensor<1x56x56x16xf32>, tensor<4xi32>) -> tensor<1x16x56x56xf32>
    %48 = tosa.cast %arg7 : (tensor<16xf32>) -> tensor<16xf32>
    %49 = tosa.cast %arg8 : (tensor<16xf32>) -> tensor<16xf32>
    %50 = "tosa.const"() <{value = dense<1.000000e-03> : tensor<16xf32>}> : () -> tensor<16xf32>
    %51 = tosa.add %49, %50 : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    %52 = math.sqrt %51 : tensor<16xf32>
    %53 = tosa.reciprocal %52 : (tensor<16xf32>) -> tensor<16xf32>
    %54 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<16xf32>}> : () -> tensor<16xf32>
    %55 = tosa.mul %53, %54 {shift = 0 : i8} : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    %56 = tosa.reshape %48 {new_shape = array<i64: 16, 1>} : (tensor<16xf32>) -> tensor<16x1xf32>
    %57 = tosa.reshape %56 {new_shape = array<i64: 16, 1, 1>} : (tensor<16x1xf32>) -> tensor<16x1x1xf32>
    %58 = tosa.reshape %55 {new_shape = array<i64: 16, 1>} : (tensor<16xf32>) -> tensor<16x1xf32>
    %59 = tosa.reshape %58 {new_shape = array<i64: 16, 1, 1>} : (tensor<16x1xf32>) -> tensor<16x1x1xf32>
    %60 = tosa.reshape %57 {new_shape = array<i64: 1, 16, 1, 1>} : (tensor<16x1x1xf32>) -> tensor<1x16x1x1xf32>
    %61 = tosa.sub %47, %60 : (tensor<1x16x56x56xf32>, tensor<1x16x1x1xf32>) -> tensor<1x16x56x56xf32>
    %62 = tosa.reshape %59 {new_shape = array<i64: 1, 16, 1, 1>} : (tensor<16x1x1xf32>) -> tensor<1x16x1x1xf32>
    %63 = tosa.mul %61, %62 {shift = 0 : i8} : (tensor<1x16x56x56xf32>, tensor<1x16x1x1xf32>) -> tensor<1x16x56x56xf32>
    %64 = tosa.reshape %arg9 {new_shape = array<i64: 16, 1>} : (tensor<16xf32>) -> tensor<16x1xf32>
    %65 = tosa.reshape %64 {new_shape = array<i64: 16, 1, 1>} : (tensor<16x1xf32>) -> tensor<16x1x1xf32>
    %66 = tosa.reshape %65 {new_shape = array<i64: 1, 16, 1, 1>} : (tensor<16x1x1xf32>) -> tensor<1x16x1x1xf32>
    %67 = tosa.mul %63, %66 {shift = 0 : i8} : (tensor<1x16x56x56xf32>, tensor<1x16x1x1xf32>) -> tensor<1x16x56x56xf32>
    %68 = tosa.reshape %arg10 {new_shape = array<i64: 16, 1>} : (tensor<16xf32>) -> tensor<16x1xf32>
    %69 = tosa.reshape %68 {new_shape = array<i64: 16, 1, 1>} : (tensor<16x1xf32>) -> tensor<16x1x1xf32>
    %70 = tosa.reshape %69 {new_shape = array<i64: 1, 16, 1, 1>} : (tensor<16x1x1xf32>) -> tensor<1x16x1x1xf32>
    %71 = tosa.add %67, %70 : (tensor<1x16x56x56xf32>, tensor<1x16x1x1xf32>) -> tensor<1x16x56x56xf32>
    %72 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x16x56x56xf32>}> : () -> tensor<1x16x56x56xf32>
    %73 = tosa.maximum %71, %72 : (tensor<1x16x56x56xf32>, tensor<1x16x56x56xf32>) -> tensor<1x16x56x56xf32>
    %74 = tosa.reduce_sum %73 {axis = 3 : i32} : (tensor<1x16x56x56xf32>) -> tensor<1x16x56x1xf32>
    %75 = tosa.reduce_sum %74 {axis = 2 : i32} : (tensor<1x16x56x1xf32>) -> tensor<1x16x1x1xf32>
    %76 = "tosa.const"() <{value = dense<3.136000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %77 = tosa.reciprocal %76 : (tensor<1xf32>) -> tensor<1xf32>
    %78 = tosa.mul %77, %75 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x16x1x1xf32>) -> tensor<1x16x1x1xf32>
    %79 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %80 = tosa.transpose %78, %79 : (tensor<1x16x1x1xf32>, tensor<4xi32>) -> tensor<1x1x1x16xf32>
    %81 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %82 = tosa.transpose %arg11, %81 : (tensor<8x16x1x1xf32>, tensor<4xi32>) -> tensor<8x1x1x16xf32>
    %83 = tosa.conv2d %80, %82, %arg12 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x1x1x16xf32>, tensor<8x1x1x16xf32>, tensor<8xf32>) -> tensor<1x1x1x8xf32>
    %84 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %85 = tosa.transpose %83, %84 : (tensor<1x1x1x8xf32>, tensor<4xi32>) -> tensor<1x8x1x1xf32>
    %86 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x8x1x1xf32>}> : () -> tensor<1x8x1x1xf32>
    %87 = tosa.maximum %85, %86 : (tensor<1x8x1x1xf32>, tensor<1x8x1x1xf32>) -> tensor<1x8x1x1xf32>
    %88 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %89 = tosa.transpose %87, %88 : (tensor<1x8x1x1xf32>, tensor<4xi32>) -> tensor<1x1x1x8xf32>
    %90 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %91 = tosa.transpose %arg13, %90 : (tensor<16x8x1x1xf32>, tensor<4xi32>) -> tensor<16x1x1x8xf32>
    %92 = tosa.conv2d %89, %91, %arg14 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x1x1x8xf32>, tensor<16x1x1x8xf32>, tensor<16xf32>) -> tensor<1x1x1x16xf32>
    %93 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %94 = tosa.transpose %92, %93 : (tensor<1x1x1x16xf32>, tensor<4xi32>) -> tensor<1x16x1x1xf32>
    %95 = "tosa.const"() <{value = dense<3.000000e+00> : tensor<1x16x1x1xf32>}> : () -> tensor<1x16x1x1xf32>
    %96 = tosa.add %94, %95 : (tensor<1x16x1x1xf32>, tensor<1x16x1x1xf32>) -> tensor<1x16x1x1xf32>
    %97 = tosa.clamp %96 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x16x1x1xf32>) -> tensor<1x16x1x1xf32>
    %98 = tosa.clamp %97 {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0xFF800000 : f32, min_int = -9223372036854775807 : i64} : (tensor<1x16x1x1xf32>) -> tensor<1x16x1x1xf32>
    %99 = "tosa.const"() <{value = dense<6.000000e+00> : tensor<1x16x1x1xf32>}> : () -> tensor<1x16x1x1xf32>
    %100 = tosa.reciprocal %99 : (tensor<1x16x1x1xf32>) -> tensor<1x16x1x1xf32>
    %101 = tosa.mul %98, %100 {shift = 0 : i8} : (tensor<1x16x1x1xf32>, tensor<1x16x1x1xf32>) -> tensor<1x16x1x1xf32>
    %102 = tosa.mul %101, %73 {shift = 0 : i8} : (tensor<1x16x1x1xf32>, tensor<1x16x56x56xf32>) -> tensor<1x16x56x56xf32>
    %103 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<16xf32>}> : () -> tensor<16xf32>
    %104 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %105 = tosa.transpose %102, %104 : (tensor<1x16x56x56xf32>, tensor<4xi32>) -> tensor<1x56x56x16xf32>
    %106 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %107 = tosa.transpose %arg15, %106 : (tensor<16x16x1x1xf32>, tensor<4xi32>) -> tensor<16x1x1x16xf32>
    %108 = tosa.conv2d %105, %107, %103 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x56x56x16xf32>, tensor<16x1x1x16xf32>, tensor<16xf32>) -> tensor<1x56x56x16xf32>
    %109 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %110 = tosa.transpose %108, %109 : (tensor<1x56x56x16xf32>, tensor<4xi32>) -> tensor<1x16x56x56xf32>
    %111 = tosa.cast %arg16 : (tensor<16xf32>) -> tensor<16xf32>
    %112 = tosa.cast %arg17 : (tensor<16xf32>) -> tensor<16xf32>
    %113 = "tosa.const"() <{value = dense<1.000000e-03> : tensor<16xf32>}> : () -> tensor<16xf32>
    %114 = tosa.add %112, %113 : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    %115 = math.sqrt %114 : tensor<16xf32>
    %116 = tosa.reciprocal %115 : (tensor<16xf32>) -> tensor<16xf32>
    %117 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<16xf32>}> : () -> tensor<16xf32>
    %118 = tosa.mul %116, %117 {shift = 0 : i8} : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    %119 = tosa.reshape %111 {new_shape = array<i64: 16, 1>} : (tensor<16xf32>) -> tensor<16x1xf32>
    %120 = tosa.reshape %119 {new_shape = array<i64: 16, 1, 1>} : (tensor<16x1xf32>) -> tensor<16x1x1xf32>
    %121 = tosa.reshape %118 {new_shape = array<i64: 16, 1>} : (tensor<16xf32>) -> tensor<16x1xf32>
    %122 = tosa.reshape %121 {new_shape = array<i64: 16, 1, 1>} : (tensor<16x1xf32>) -> tensor<16x1x1xf32>
    %123 = tosa.reshape %120 {new_shape = array<i64: 1, 16, 1, 1>} : (tensor<16x1x1xf32>) -> tensor<1x16x1x1xf32>
    %124 = tosa.sub %110, %123 : (tensor<1x16x56x56xf32>, tensor<1x16x1x1xf32>) -> tensor<1x16x56x56xf32>
    %125 = tosa.reshape %122 {new_shape = array<i64: 1, 16, 1, 1>} : (tensor<16x1x1xf32>) -> tensor<1x16x1x1xf32>
    %126 = tosa.mul %124, %125 {shift = 0 : i8} : (tensor<1x16x56x56xf32>, tensor<1x16x1x1xf32>) -> tensor<1x16x56x56xf32>
    %127 = tosa.reshape %arg18 {new_shape = array<i64: 16, 1>} : (tensor<16xf32>) -> tensor<16x1xf32>
    %128 = tosa.reshape %127 {new_shape = array<i64: 16, 1, 1>} : (tensor<16x1xf32>) -> tensor<16x1x1xf32>
    %129 = tosa.reshape %128 {new_shape = array<i64: 1, 16, 1, 1>} : (tensor<16x1x1xf32>) -> tensor<1x16x1x1xf32>
    %130 = tosa.mul %126, %129 {shift = 0 : i8} : (tensor<1x16x56x56xf32>, tensor<1x16x1x1xf32>) -> tensor<1x16x56x56xf32>
    %131 = tosa.reshape %arg19 {new_shape = array<i64: 16, 1>} : (tensor<16xf32>) -> tensor<16x1xf32>
    %132 = tosa.reshape %131 {new_shape = array<i64: 16, 1, 1>} : (tensor<16x1xf32>) -> tensor<16x1x1xf32>
    %133 = tosa.reshape %132 {new_shape = array<i64: 1, 16, 1, 1>} : (tensor<16x1x1xf32>) -> tensor<1x16x1x1xf32>
    %134 = tosa.add %130, %133 : (tensor<1x16x56x56xf32>, tensor<1x16x1x1xf32>) -> tensor<1x16x56x56xf32>
    %135 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<72xf32>}> : () -> tensor<72xf32>
    %136 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %137 = tosa.transpose %134, %136 : (tensor<1x16x56x56xf32>, tensor<4xi32>) -> tensor<1x56x56x16xf32>
    %138 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %139 = tosa.transpose %arg20, %138 : (tensor<72x16x1x1xf32>, tensor<4xi32>) -> tensor<72x1x1x16xf32>
    %140 = tosa.conv2d %137, %139, %135 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x56x56x16xf32>, tensor<72x1x1x16xf32>, tensor<72xf32>) -> tensor<1x56x56x72xf32>
    %141 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %142 = tosa.transpose %140, %141 : (tensor<1x56x56x72xf32>, tensor<4xi32>) -> tensor<1x72x56x56xf32>
    %143 = tosa.cast %arg21 : (tensor<72xf32>) -> tensor<72xf32>
    %144 = tosa.cast %arg22 : (tensor<72xf32>) -> tensor<72xf32>
    %145 = "tosa.const"() <{value = dense<1.000000e-03> : tensor<72xf32>}> : () -> tensor<72xf32>
    %146 = tosa.add %144, %145 : (tensor<72xf32>, tensor<72xf32>) -> tensor<72xf32>
    %147 = math.sqrt %146 : tensor<72xf32>
    %148 = tosa.reciprocal %147 : (tensor<72xf32>) -> tensor<72xf32>
    %149 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<72xf32>}> : () -> tensor<72xf32>
    %150 = tosa.mul %148, %149 {shift = 0 : i8} : (tensor<72xf32>, tensor<72xf32>) -> tensor<72xf32>
    %151 = tosa.reshape %143 {new_shape = array<i64: 72, 1>} : (tensor<72xf32>) -> tensor<72x1xf32>
    %152 = tosa.reshape %151 {new_shape = array<i64: 72, 1, 1>} : (tensor<72x1xf32>) -> tensor<72x1x1xf32>
    %153 = tosa.reshape %150 {new_shape = array<i64: 72, 1>} : (tensor<72xf32>) -> tensor<72x1xf32>
    %154 = tosa.reshape %153 {new_shape = array<i64: 72, 1, 1>} : (tensor<72x1xf32>) -> tensor<72x1x1xf32>
    %155 = tosa.reshape %152 {new_shape = array<i64: 1, 72, 1, 1>} : (tensor<72x1x1xf32>) -> tensor<1x72x1x1xf32>
    %156 = tosa.sub %142, %155 : (tensor<1x72x56x56xf32>, tensor<1x72x1x1xf32>) -> tensor<1x72x56x56xf32>
    %157 = tosa.reshape %154 {new_shape = array<i64: 1, 72, 1, 1>} : (tensor<72x1x1xf32>) -> tensor<1x72x1x1xf32>
    %158 = tosa.mul %156, %157 {shift = 0 : i8} : (tensor<1x72x56x56xf32>, tensor<1x72x1x1xf32>) -> tensor<1x72x56x56xf32>
    %159 = tosa.reshape %arg23 {new_shape = array<i64: 72, 1>} : (tensor<72xf32>) -> tensor<72x1xf32>
    %160 = tosa.reshape %159 {new_shape = array<i64: 72, 1, 1>} : (tensor<72x1xf32>) -> tensor<72x1x1xf32>
    %161 = tosa.reshape %160 {new_shape = array<i64: 1, 72, 1, 1>} : (tensor<72x1x1xf32>) -> tensor<1x72x1x1xf32>
    %162 = tosa.mul %158, %161 {shift = 0 : i8} : (tensor<1x72x56x56xf32>, tensor<1x72x1x1xf32>) -> tensor<1x72x56x56xf32>
    %163 = tosa.reshape %arg24 {new_shape = array<i64: 72, 1>} : (tensor<72xf32>) -> tensor<72x1xf32>
    %164 = tosa.reshape %163 {new_shape = array<i64: 72, 1, 1>} : (tensor<72x1xf32>) -> tensor<72x1x1xf32>
    %165 = tosa.reshape %164 {new_shape = array<i64: 1, 72, 1, 1>} : (tensor<72x1x1xf32>) -> tensor<1x72x1x1xf32>
    %166 = tosa.add %162, %165 : (tensor<1x72x56x56xf32>, tensor<1x72x1x1xf32>) -> tensor<1x72x56x56xf32>
    %167 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x72x56x56xf32>}> : () -> tensor<1x72x56x56xf32>
    %168 = tosa.maximum %166, %167 : (tensor<1x72x56x56xf32>, tensor<1x72x56x56xf32>) -> tensor<1x72x56x56xf32>
    %169 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<72xf32>}> : () -> tensor<72xf32>
    %170 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %171 = tosa.transpose %168, %170 : (tensor<1x72x56x56xf32>, tensor<4xi32>) -> tensor<1x56x56x72xf32>
    %172 = "tosa.const"() <{value = dense<[2, 3, 0, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %173 = tosa.transpose %arg25, %172 : (tensor<72x1x3x3xf32>, tensor<4xi32>) -> tensor<3x3x72x1xf32>
    %174 = tosa.depthwise_conv2d %171, %173, %169 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>} : (tensor<1x56x56x72xf32>, tensor<3x3x72x1xf32>, tensor<72xf32>) -> tensor<1x28x28x72xf32>
    %175 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %176 = tosa.transpose %174, %175 : (tensor<1x28x28x72xf32>, tensor<4xi32>) -> tensor<1x72x28x28xf32>
    %177 = tosa.cast %arg26 : (tensor<72xf32>) -> tensor<72xf32>
    %178 = tosa.cast %arg27 : (tensor<72xf32>) -> tensor<72xf32>
    %179 = "tosa.const"() <{value = dense<1.000000e-03> : tensor<72xf32>}> : () -> tensor<72xf32>
    %180 = tosa.add %178, %179 : (tensor<72xf32>, tensor<72xf32>) -> tensor<72xf32>
    %181 = math.sqrt %180 : tensor<72xf32>
    %182 = tosa.reciprocal %181 : (tensor<72xf32>) -> tensor<72xf32>
    %183 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<72xf32>}> : () -> tensor<72xf32>
    %184 = tosa.mul %182, %183 {shift = 0 : i8} : (tensor<72xf32>, tensor<72xf32>) -> tensor<72xf32>
    %185 = tosa.reshape %177 {new_shape = array<i64: 72, 1>} : (tensor<72xf32>) -> tensor<72x1xf32>
    %186 = tosa.reshape %185 {new_shape = array<i64: 72, 1, 1>} : (tensor<72x1xf32>) -> tensor<72x1x1xf32>
    %187 = tosa.reshape %184 {new_shape = array<i64: 72, 1>} : (tensor<72xf32>) -> tensor<72x1xf32>
    %188 = tosa.reshape %187 {new_shape = array<i64: 72, 1, 1>} : (tensor<72x1xf32>) -> tensor<72x1x1xf32>
    %189 = tosa.reshape %186 {new_shape = array<i64: 1, 72, 1, 1>} : (tensor<72x1x1xf32>) -> tensor<1x72x1x1xf32>
    %190 = tosa.sub %176, %189 : (tensor<1x72x28x28xf32>, tensor<1x72x1x1xf32>) -> tensor<1x72x28x28xf32>
    %191 = tosa.reshape %188 {new_shape = array<i64: 1, 72, 1, 1>} : (tensor<72x1x1xf32>) -> tensor<1x72x1x1xf32>
    %192 = tosa.mul %190, %191 {shift = 0 : i8} : (tensor<1x72x28x28xf32>, tensor<1x72x1x1xf32>) -> tensor<1x72x28x28xf32>
    %193 = tosa.reshape %arg28 {new_shape = array<i64: 72, 1>} : (tensor<72xf32>) -> tensor<72x1xf32>
    %194 = tosa.reshape %193 {new_shape = array<i64: 72, 1, 1>} : (tensor<72x1xf32>) -> tensor<72x1x1xf32>
    %195 = tosa.reshape %194 {new_shape = array<i64: 1, 72, 1, 1>} : (tensor<72x1x1xf32>) -> tensor<1x72x1x1xf32>
    %196 = tosa.mul %192, %195 {shift = 0 : i8} : (tensor<1x72x28x28xf32>, tensor<1x72x1x1xf32>) -> tensor<1x72x28x28xf32>
    %197 = tosa.reshape %arg29 {new_shape = array<i64: 72, 1>} : (tensor<72xf32>) -> tensor<72x1xf32>
    %198 = tosa.reshape %197 {new_shape = array<i64: 72, 1, 1>} : (tensor<72x1xf32>) -> tensor<72x1x1xf32>
    %199 = tosa.reshape %198 {new_shape = array<i64: 1, 72, 1, 1>} : (tensor<72x1x1xf32>) -> tensor<1x72x1x1xf32>
    %200 = tosa.add %196, %199 : (tensor<1x72x28x28xf32>, tensor<1x72x1x1xf32>) -> tensor<1x72x28x28xf32>
    %201 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x72x28x28xf32>}> : () -> tensor<1x72x28x28xf32>
    %202 = tosa.maximum %200, %201 : (tensor<1x72x28x28xf32>, tensor<1x72x28x28xf32>) -> tensor<1x72x28x28xf32>
    %203 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<24xf32>}> : () -> tensor<24xf32>
    %204 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %205 = tosa.transpose %202, %204 : (tensor<1x72x28x28xf32>, tensor<4xi32>) -> tensor<1x28x28x72xf32>
    %206 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %207 = tosa.transpose %arg30, %206 : (tensor<24x72x1x1xf32>, tensor<4xi32>) -> tensor<24x1x1x72xf32>
    %208 = tosa.conv2d %205, %207, %203 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x28x28x72xf32>, tensor<24x1x1x72xf32>, tensor<24xf32>) -> tensor<1x28x28x24xf32>
    %209 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %210 = tosa.transpose %208, %209 : (tensor<1x28x28x24xf32>, tensor<4xi32>) -> tensor<1x24x28x28xf32>
    %211 = tosa.cast %arg31 : (tensor<24xf32>) -> tensor<24xf32>
    %212 = tosa.cast %arg32 : (tensor<24xf32>) -> tensor<24xf32>
    %213 = "tosa.const"() <{value = dense<1.000000e-03> : tensor<24xf32>}> : () -> tensor<24xf32>
    %214 = tosa.add %212, %213 : (tensor<24xf32>, tensor<24xf32>) -> tensor<24xf32>
    %215 = math.sqrt %214 : tensor<24xf32>
    %216 = tosa.reciprocal %215 : (tensor<24xf32>) -> tensor<24xf32>
    %217 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<24xf32>}> : () -> tensor<24xf32>
    %218 = tosa.mul %216, %217 {shift = 0 : i8} : (tensor<24xf32>, tensor<24xf32>) -> tensor<24xf32>
    %219 = tosa.reshape %211 {new_shape = array<i64: 24, 1>} : (tensor<24xf32>) -> tensor<24x1xf32>
    %220 = tosa.reshape %219 {new_shape = array<i64: 24, 1, 1>} : (tensor<24x1xf32>) -> tensor<24x1x1xf32>
    %221 = tosa.reshape %218 {new_shape = array<i64: 24, 1>} : (tensor<24xf32>) -> tensor<24x1xf32>
    %222 = tosa.reshape %221 {new_shape = array<i64: 24, 1, 1>} : (tensor<24x1xf32>) -> tensor<24x1x1xf32>
    %223 = tosa.reshape %220 {new_shape = array<i64: 1, 24, 1, 1>} : (tensor<24x1x1xf32>) -> tensor<1x24x1x1xf32>
    %224 = tosa.sub %210, %223 : (tensor<1x24x28x28xf32>, tensor<1x24x1x1xf32>) -> tensor<1x24x28x28xf32>
    %225 = tosa.reshape %222 {new_shape = array<i64: 1, 24, 1, 1>} : (tensor<24x1x1xf32>) -> tensor<1x24x1x1xf32>
    %226 = tosa.mul %224, %225 {shift = 0 : i8} : (tensor<1x24x28x28xf32>, tensor<1x24x1x1xf32>) -> tensor<1x24x28x28xf32>
    %227 = tosa.reshape %arg33 {new_shape = array<i64: 24, 1>} : (tensor<24xf32>) -> tensor<24x1xf32>
    %228 = tosa.reshape %227 {new_shape = array<i64: 24, 1, 1>} : (tensor<24x1xf32>) -> tensor<24x1x1xf32>
    %229 = tosa.reshape %228 {new_shape = array<i64: 1, 24, 1, 1>} : (tensor<24x1x1xf32>) -> tensor<1x24x1x1xf32>
    %230 = tosa.mul %226, %229 {shift = 0 : i8} : (tensor<1x24x28x28xf32>, tensor<1x24x1x1xf32>) -> tensor<1x24x28x28xf32>
    %231 = tosa.reshape %arg34 {new_shape = array<i64: 24, 1>} : (tensor<24xf32>) -> tensor<24x1xf32>
    %232 = tosa.reshape %231 {new_shape = array<i64: 24, 1, 1>} : (tensor<24x1xf32>) -> tensor<24x1x1xf32>
    %233 = tosa.reshape %232 {new_shape = array<i64: 1, 24, 1, 1>} : (tensor<24x1x1xf32>) -> tensor<1x24x1x1xf32>
    %234 = tosa.add %230, %233 : (tensor<1x24x28x28xf32>, tensor<1x24x1x1xf32>) -> tensor<1x24x28x28xf32>
    %235 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<88xf32>}> : () -> tensor<88xf32>
    %236 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %237 = tosa.transpose %234, %236 : (tensor<1x24x28x28xf32>, tensor<4xi32>) -> tensor<1x28x28x24xf32>
    %238 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %239 = tosa.transpose %arg35, %238 : (tensor<88x24x1x1xf32>, tensor<4xi32>) -> tensor<88x1x1x24xf32>
    %240 = tosa.conv2d %237, %239, %235 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x28x28x24xf32>, tensor<88x1x1x24xf32>, tensor<88xf32>) -> tensor<1x28x28x88xf32>
    %241 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %242 = tosa.transpose %240, %241 : (tensor<1x28x28x88xf32>, tensor<4xi32>) -> tensor<1x88x28x28xf32>
    %243 = tosa.cast %arg36 : (tensor<88xf32>) -> tensor<88xf32>
    %244 = tosa.cast %arg37 : (tensor<88xf32>) -> tensor<88xf32>
    %245 = "tosa.const"() <{value = dense<1.000000e-03> : tensor<88xf32>}> : () -> tensor<88xf32>
    %246 = tosa.add %244, %245 : (tensor<88xf32>, tensor<88xf32>) -> tensor<88xf32>
    %247 = math.sqrt %246 : tensor<88xf32>
    %248 = tosa.reciprocal %247 : (tensor<88xf32>) -> tensor<88xf32>
    %249 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<88xf32>}> : () -> tensor<88xf32>
    %250 = tosa.mul %248, %249 {shift = 0 : i8} : (tensor<88xf32>, tensor<88xf32>) -> tensor<88xf32>
    %251 = tosa.reshape %243 {new_shape = array<i64: 88, 1>} : (tensor<88xf32>) -> tensor<88x1xf32>
    %252 = tosa.reshape %251 {new_shape = array<i64: 88, 1, 1>} : (tensor<88x1xf32>) -> tensor<88x1x1xf32>
    %253 = tosa.reshape %250 {new_shape = array<i64: 88, 1>} : (tensor<88xf32>) -> tensor<88x1xf32>
    %254 = tosa.reshape %253 {new_shape = array<i64: 88, 1, 1>} : (tensor<88x1xf32>) -> tensor<88x1x1xf32>
    %255 = tosa.reshape %252 {new_shape = array<i64: 1, 88, 1, 1>} : (tensor<88x1x1xf32>) -> tensor<1x88x1x1xf32>
    %256 = tosa.sub %242, %255 : (tensor<1x88x28x28xf32>, tensor<1x88x1x1xf32>) -> tensor<1x88x28x28xf32>
    %257 = tosa.reshape %254 {new_shape = array<i64: 1, 88, 1, 1>} : (tensor<88x1x1xf32>) -> tensor<1x88x1x1xf32>
    %258 = tosa.mul %256, %257 {shift = 0 : i8} : (tensor<1x88x28x28xf32>, tensor<1x88x1x1xf32>) -> tensor<1x88x28x28xf32>
    %259 = tosa.reshape %arg38 {new_shape = array<i64: 88, 1>} : (tensor<88xf32>) -> tensor<88x1xf32>
    %260 = tosa.reshape %259 {new_shape = array<i64: 88, 1, 1>} : (tensor<88x1xf32>) -> tensor<88x1x1xf32>
    %261 = tosa.reshape %260 {new_shape = array<i64: 1, 88, 1, 1>} : (tensor<88x1x1xf32>) -> tensor<1x88x1x1xf32>
    %262 = tosa.mul %258, %261 {shift = 0 : i8} : (tensor<1x88x28x28xf32>, tensor<1x88x1x1xf32>) -> tensor<1x88x28x28xf32>
    %263 = tosa.reshape %arg39 {new_shape = array<i64: 88, 1>} : (tensor<88xf32>) -> tensor<88x1xf32>
    %264 = tosa.reshape %263 {new_shape = array<i64: 88, 1, 1>} : (tensor<88x1xf32>) -> tensor<88x1x1xf32>
    %265 = tosa.reshape %264 {new_shape = array<i64: 1, 88, 1, 1>} : (tensor<88x1x1xf32>) -> tensor<1x88x1x1xf32>
    %266 = tosa.add %262, %265 : (tensor<1x88x28x28xf32>, tensor<1x88x1x1xf32>) -> tensor<1x88x28x28xf32>
    %267 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x88x28x28xf32>}> : () -> tensor<1x88x28x28xf32>
    %268 = tosa.maximum %266, %267 : (tensor<1x88x28x28xf32>, tensor<1x88x28x28xf32>) -> tensor<1x88x28x28xf32>
    %269 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<88xf32>}> : () -> tensor<88xf32>
    %270 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %271 = tosa.transpose %268, %270 : (tensor<1x88x28x28xf32>, tensor<4xi32>) -> tensor<1x28x28x88xf32>
    %272 = "tosa.const"() <{value = dense<[2, 3, 0, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %273 = tosa.transpose %arg40, %272 : (tensor<88x1x3x3xf32>, tensor<4xi32>) -> tensor<3x3x88x1xf32>
    %274 = tosa.depthwise_conv2d %271, %273, %269 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x28x28x88xf32>, tensor<3x3x88x1xf32>, tensor<88xf32>) -> tensor<1x28x28x88xf32>
    %275 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %276 = tosa.transpose %274, %275 : (tensor<1x28x28x88xf32>, tensor<4xi32>) -> tensor<1x88x28x28xf32>
    %277 = tosa.cast %arg41 : (tensor<88xf32>) -> tensor<88xf32>
    %278 = tosa.cast %arg42 : (tensor<88xf32>) -> tensor<88xf32>
    %279 = "tosa.const"() <{value = dense<1.000000e-03> : tensor<88xf32>}> : () -> tensor<88xf32>
    %280 = tosa.add %278, %279 : (tensor<88xf32>, tensor<88xf32>) -> tensor<88xf32>
    %281 = math.sqrt %280 : tensor<88xf32>
    %282 = tosa.reciprocal %281 : (tensor<88xf32>) -> tensor<88xf32>
    %283 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<88xf32>}> : () -> tensor<88xf32>
    %284 = tosa.mul %282, %283 {shift = 0 : i8} : (tensor<88xf32>, tensor<88xf32>) -> tensor<88xf32>
    %285 = tosa.reshape %277 {new_shape = array<i64: 88, 1>} : (tensor<88xf32>) -> tensor<88x1xf32>
    %286 = tosa.reshape %285 {new_shape = array<i64: 88, 1, 1>} : (tensor<88x1xf32>) -> tensor<88x1x1xf32>
    %287 = tosa.reshape %284 {new_shape = array<i64: 88, 1>} : (tensor<88xf32>) -> tensor<88x1xf32>
    %288 = tosa.reshape %287 {new_shape = array<i64: 88, 1, 1>} : (tensor<88x1xf32>) -> tensor<88x1x1xf32>
    %289 = tosa.reshape %286 {new_shape = array<i64: 1, 88, 1, 1>} : (tensor<88x1x1xf32>) -> tensor<1x88x1x1xf32>
    %290 = tosa.sub %276, %289 : (tensor<1x88x28x28xf32>, tensor<1x88x1x1xf32>) -> tensor<1x88x28x28xf32>
    %291 = tosa.reshape %288 {new_shape = array<i64: 1, 88, 1, 1>} : (tensor<88x1x1xf32>) -> tensor<1x88x1x1xf32>
    %292 = tosa.mul %290, %291 {shift = 0 : i8} : (tensor<1x88x28x28xf32>, tensor<1x88x1x1xf32>) -> tensor<1x88x28x28xf32>
    %293 = tosa.reshape %arg43 {new_shape = array<i64: 88, 1>} : (tensor<88xf32>) -> tensor<88x1xf32>
    %294 = tosa.reshape %293 {new_shape = array<i64: 88, 1, 1>} : (tensor<88x1xf32>) -> tensor<88x1x1xf32>
    %295 = tosa.reshape %294 {new_shape = array<i64: 1, 88, 1, 1>} : (tensor<88x1x1xf32>) -> tensor<1x88x1x1xf32>
    %296 = tosa.mul %292, %295 {shift = 0 : i8} : (tensor<1x88x28x28xf32>, tensor<1x88x1x1xf32>) -> tensor<1x88x28x28xf32>
    %297 = tosa.reshape %arg44 {new_shape = array<i64: 88, 1>} : (tensor<88xf32>) -> tensor<88x1xf32>
    %298 = tosa.reshape %297 {new_shape = array<i64: 88, 1, 1>} : (tensor<88x1xf32>) -> tensor<88x1x1xf32>
    %299 = tosa.reshape %298 {new_shape = array<i64: 1, 88, 1, 1>} : (tensor<88x1x1xf32>) -> tensor<1x88x1x1xf32>
    %300 = tosa.add %296, %299 : (tensor<1x88x28x28xf32>, tensor<1x88x1x1xf32>) -> tensor<1x88x28x28xf32>
    %301 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x88x28x28xf32>}> : () -> tensor<1x88x28x28xf32>
    %302 = tosa.maximum %300, %301 : (tensor<1x88x28x28xf32>, tensor<1x88x28x28xf32>) -> tensor<1x88x28x28xf32>
    %303 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<24xf32>}> : () -> tensor<24xf32>
    %304 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %305 = tosa.transpose %302, %304 : (tensor<1x88x28x28xf32>, tensor<4xi32>) -> tensor<1x28x28x88xf32>
    %306 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %307 = tosa.transpose %arg45, %306 : (tensor<24x88x1x1xf32>, tensor<4xi32>) -> tensor<24x1x1x88xf32>
    %308 = tosa.conv2d %305, %307, %303 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x28x28x88xf32>, tensor<24x1x1x88xf32>, tensor<24xf32>) -> tensor<1x28x28x24xf32>
    %309 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %310 = tosa.transpose %308, %309 : (tensor<1x28x28x24xf32>, tensor<4xi32>) -> tensor<1x24x28x28xf32>
    %311 = tosa.cast %arg46 : (tensor<24xf32>) -> tensor<24xf32>
    %312 = tosa.cast %arg47 : (tensor<24xf32>) -> tensor<24xf32>
    %313 = "tosa.const"() <{value = dense<1.000000e-03> : tensor<24xf32>}> : () -> tensor<24xf32>
    %314 = tosa.add %312, %313 : (tensor<24xf32>, tensor<24xf32>) -> tensor<24xf32>
    %315 = math.sqrt %314 : tensor<24xf32>
    %316 = tosa.reciprocal %315 : (tensor<24xf32>) -> tensor<24xf32>
    %317 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<24xf32>}> : () -> tensor<24xf32>
    %318 = tosa.mul %316, %317 {shift = 0 : i8} : (tensor<24xf32>, tensor<24xf32>) -> tensor<24xf32>
    %319 = tosa.reshape %311 {new_shape = array<i64: 24, 1>} : (tensor<24xf32>) -> tensor<24x1xf32>
    %320 = tosa.reshape %319 {new_shape = array<i64: 24, 1, 1>} : (tensor<24x1xf32>) -> tensor<24x1x1xf32>
    %321 = tosa.reshape %318 {new_shape = array<i64: 24, 1>} : (tensor<24xf32>) -> tensor<24x1xf32>
    %322 = tosa.reshape %321 {new_shape = array<i64: 24, 1, 1>} : (tensor<24x1xf32>) -> tensor<24x1x1xf32>
    %323 = tosa.reshape %320 {new_shape = array<i64: 1, 24, 1, 1>} : (tensor<24x1x1xf32>) -> tensor<1x24x1x1xf32>
    %324 = tosa.sub %310, %323 : (tensor<1x24x28x28xf32>, tensor<1x24x1x1xf32>) -> tensor<1x24x28x28xf32>
    %325 = tosa.reshape %322 {new_shape = array<i64: 1, 24, 1, 1>} : (tensor<24x1x1xf32>) -> tensor<1x24x1x1xf32>
    %326 = tosa.mul %324, %325 {shift = 0 : i8} : (tensor<1x24x28x28xf32>, tensor<1x24x1x1xf32>) -> tensor<1x24x28x28xf32>
    %327 = tosa.reshape %arg48 {new_shape = array<i64: 24, 1>} : (tensor<24xf32>) -> tensor<24x1xf32>
    %328 = tosa.reshape %327 {new_shape = array<i64: 24, 1, 1>} : (tensor<24x1xf32>) -> tensor<24x1x1xf32>
    %329 = tosa.reshape %328 {new_shape = array<i64: 1, 24, 1, 1>} : (tensor<24x1x1xf32>) -> tensor<1x24x1x1xf32>
    %330 = tosa.mul %326, %329 {shift = 0 : i8} : (tensor<1x24x28x28xf32>, tensor<1x24x1x1xf32>) -> tensor<1x24x28x28xf32>
    %331 = tosa.reshape %arg49 {new_shape = array<i64: 24, 1>} : (tensor<24xf32>) -> tensor<24x1xf32>
    %332 = tosa.reshape %331 {new_shape = array<i64: 24, 1, 1>} : (tensor<24x1xf32>) -> tensor<24x1x1xf32>
    %333 = tosa.reshape %332 {new_shape = array<i64: 1, 24, 1, 1>} : (tensor<24x1x1xf32>) -> tensor<1x24x1x1xf32>
    %334 = tosa.add %330, %333 : (tensor<1x24x28x28xf32>, tensor<1x24x1x1xf32>) -> tensor<1x24x28x28xf32>
    %335 = tosa.add %334, %234 : (tensor<1x24x28x28xf32>, tensor<1x24x28x28xf32>) -> tensor<1x24x28x28xf32>
    %336 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<96xf32>}> : () -> tensor<96xf32>
    %337 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %338 = tosa.transpose %335, %337 : (tensor<1x24x28x28xf32>, tensor<4xi32>) -> tensor<1x28x28x24xf32>
    %339 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %340 = tosa.transpose %arg50, %339 : (tensor<96x24x1x1xf32>, tensor<4xi32>) -> tensor<96x1x1x24xf32>
    %341 = tosa.conv2d %338, %340, %336 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x28x28x24xf32>, tensor<96x1x1x24xf32>, tensor<96xf32>) -> tensor<1x28x28x96xf32>
    %342 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %343 = tosa.transpose %341, %342 : (tensor<1x28x28x96xf32>, tensor<4xi32>) -> tensor<1x96x28x28xf32>
    %344 = tosa.cast %arg51 : (tensor<96xf32>) -> tensor<96xf32>
    %345 = tosa.cast %arg52 : (tensor<96xf32>) -> tensor<96xf32>
    %346 = "tosa.const"() <{value = dense<1.000000e-03> : tensor<96xf32>}> : () -> tensor<96xf32>
    %347 = tosa.add %345, %346 : (tensor<96xf32>, tensor<96xf32>) -> tensor<96xf32>
    %348 = math.sqrt %347 : tensor<96xf32>
    %349 = tosa.reciprocal %348 : (tensor<96xf32>) -> tensor<96xf32>
    %350 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<96xf32>}> : () -> tensor<96xf32>
    %351 = tosa.mul %349, %350 {shift = 0 : i8} : (tensor<96xf32>, tensor<96xf32>) -> tensor<96xf32>
    %352 = tosa.reshape %344 {new_shape = array<i64: 96, 1>} : (tensor<96xf32>) -> tensor<96x1xf32>
    %353 = tosa.reshape %352 {new_shape = array<i64: 96, 1, 1>} : (tensor<96x1xf32>) -> tensor<96x1x1xf32>
    %354 = tosa.reshape %351 {new_shape = array<i64: 96, 1>} : (tensor<96xf32>) -> tensor<96x1xf32>
    %355 = tosa.reshape %354 {new_shape = array<i64: 96, 1, 1>} : (tensor<96x1xf32>) -> tensor<96x1x1xf32>
    %356 = tosa.reshape %353 {new_shape = array<i64: 1, 96, 1, 1>} : (tensor<96x1x1xf32>) -> tensor<1x96x1x1xf32>
    %357 = tosa.sub %343, %356 : (tensor<1x96x28x28xf32>, tensor<1x96x1x1xf32>) -> tensor<1x96x28x28xf32>
    %358 = tosa.reshape %355 {new_shape = array<i64: 1, 96, 1, 1>} : (tensor<96x1x1xf32>) -> tensor<1x96x1x1xf32>
    %359 = tosa.mul %357, %358 {shift = 0 : i8} : (tensor<1x96x28x28xf32>, tensor<1x96x1x1xf32>) -> tensor<1x96x28x28xf32>
    %360 = tosa.reshape %arg53 {new_shape = array<i64: 96, 1>} : (tensor<96xf32>) -> tensor<96x1xf32>
    %361 = tosa.reshape %360 {new_shape = array<i64: 96, 1, 1>} : (tensor<96x1xf32>) -> tensor<96x1x1xf32>
    %362 = tosa.reshape %361 {new_shape = array<i64: 1, 96, 1, 1>} : (tensor<96x1x1xf32>) -> tensor<1x96x1x1xf32>
    %363 = tosa.mul %359, %362 {shift = 0 : i8} : (tensor<1x96x28x28xf32>, tensor<1x96x1x1xf32>) -> tensor<1x96x28x28xf32>
    %364 = tosa.reshape %arg54 {new_shape = array<i64: 96, 1>} : (tensor<96xf32>) -> tensor<96x1xf32>
    %365 = tosa.reshape %364 {new_shape = array<i64: 96, 1, 1>} : (tensor<96x1xf32>) -> tensor<96x1x1xf32>
    %366 = tosa.reshape %365 {new_shape = array<i64: 1, 96, 1, 1>} : (tensor<96x1x1xf32>) -> tensor<1x96x1x1xf32>
    %367 = tosa.add %363, %366 : (tensor<1x96x28x28xf32>, tensor<1x96x1x1xf32>) -> tensor<1x96x28x28xf32>
    %368 = "tosa.const"() <{value = dense<3.000000e+00> : tensor<1x96x28x28xf32>}> : () -> tensor<1x96x28x28xf32>
    %369 = tosa.add %367, %368 : (tensor<1x96x28x28xf32>, tensor<1x96x28x28xf32>) -> tensor<1x96x28x28xf32>
    %370 = tosa.clamp %369 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x96x28x28xf32>) -> tensor<1x96x28x28xf32>
    %371 = tosa.clamp %370 {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0xFF800000 : f32, min_int = -9223372036854775807 : i64} : (tensor<1x96x28x28xf32>) -> tensor<1x96x28x28xf32>
    %372 = tosa.mul %367, %371 {shift = 0 : i8} : (tensor<1x96x28x28xf32>, tensor<1x96x28x28xf32>) -> tensor<1x96x28x28xf32>
    %373 = "tosa.const"() <{value = dense<6.000000e+00> : tensor<1x96x28x28xf32>}> : () -> tensor<1x96x28x28xf32>
    %374 = tosa.reciprocal %373 : (tensor<1x96x28x28xf32>) -> tensor<1x96x28x28xf32>
    %375 = tosa.mul %372, %374 {shift = 0 : i8} : (tensor<1x96x28x28xf32>, tensor<1x96x28x28xf32>) -> tensor<1x96x28x28xf32>
    %376 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<96xf32>}> : () -> tensor<96xf32>
    %377 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %378 = tosa.transpose %375, %377 : (tensor<1x96x28x28xf32>, tensor<4xi32>) -> tensor<1x28x28x96xf32>
    %379 = "tosa.const"() <{value = dense<[2, 3, 0, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %380 = tosa.transpose %arg55, %379 : (tensor<96x1x5x5xf32>, tensor<4xi32>) -> tensor<5x5x96x1xf32>
    %381 = tosa.depthwise_conv2d %378, %380, %376 {dilation = array<i64: 1, 1>, pad = array<i64: 2, 2, 2, 2>, stride = array<i64: 2, 2>} : (tensor<1x28x28x96xf32>, tensor<5x5x96x1xf32>, tensor<96xf32>) -> tensor<1x14x14x96xf32>
    %382 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %383 = tosa.transpose %381, %382 : (tensor<1x14x14x96xf32>, tensor<4xi32>) -> tensor<1x96x14x14xf32>
    %384 = tosa.cast %arg56 : (tensor<96xf32>) -> tensor<96xf32>
    %385 = tosa.cast %arg57 : (tensor<96xf32>) -> tensor<96xf32>
    %386 = "tosa.const"() <{value = dense<1.000000e-03> : tensor<96xf32>}> : () -> tensor<96xf32>
    %387 = tosa.add %385, %386 : (tensor<96xf32>, tensor<96xf32>) -> tensor<96xf32>
    %388 = math.sqrt %387 : tensor<96xf32>
    %389 = tosa.reciprocal %388 : (tensor<96xf32>) -> tensor<96xf32>
    %390 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<96xf32>}> : () -> tensor<96xf32>
    %391 = tosa.mul %389, %390 {shift = 0 : i8} : (tensor<96xf32>, tensor<96xf32>) -> tensor<96xf32>
    %392 = tosa.reshape %384 {new_shape = array<i64: 96, 1>} : (tensor<96xf32>) -> tensor<96x1xf32>
    %393 = tosa.reshape %392 {new_shape = array<i64: 96, 1, 1>} : (tensor<96x1xf32>) -> tensor<96x1x1xf32>
    %394 = tosa.reshape %391 {new_shape = array<i64: 96, 1>} : (tensor<96xf32>) -> tensor<96x1xf32>
    %395 = tosa.reshape %394 {new_shape = array<i64: 96, 1, 1>} : (tensor<96x1xf32>) -> tensor<96x1x1xf32>
    %396 = tosa.reshape %393 {new_shape = array<i64: 1, 96, 1, 1>} : (tensor<96x1x1xf32>) -> tensor<1x96x1x1xf32>
    %397 = tosa.sub %383, %396 : (tensor<1x96x14x14xf32>, tensor<1x96x1x1xf32>) -> tensor<1x96x14x14xf32>
    %398 = tosa.reshape %395 {new_shape = array<i64: 1, 96, 1, 1>} : (tensor<96x1x1xf32>) -> tensor<1x96x1x1xf32>
    %399 = tosa.mul %397, %398 {shift = 0 : i8} : (tensor<1x96x14x14xf32>, tensor<1x96x1x1xf32>) -> tensor<1x96x14x14xf32>
    %400 = tosa.reshape %arg58 {new_shape = array<i64: 96, 1>} : (tensor<96xf32>) -> tensor<96x1xf32>
    %401 = tosa.reshape %400 {new_shape = array<i64: 96, 1, 1>} : (tensor<96x1xf32>) -> tensor<96x1x1xf32>
    %402 = tosa.reshape %401 {new_shape = array<i64: 1, 96, 1, 1>} : (tensor<96x1x1xf32>) -> tensor<1x96x1x1xf32>
    %403 = tosa.mul %399, %402 {shift = 0 : i8} : (tensor<1x96x14x14xf32>, tensor<1x96x1x1xf32>) -> tensor<1x96x14x14xf32>
    %404 = tosa.reshape %arg59 {new_shape = array<i64: 96, 1>} : (tensor<96xf32>) -> tensor<96x1xf32>
    %405 = tosa.reshape %404 {new_shape = array<i64: 96, 1, 1>} : (tensor<96x1xf32>) -> tensor<96x1x1xf32>
    %406 = tosa.reshape %405 {new_shape = array<i64: 1, 96, 1, 1>} : (tensor<96x1x1xf32>) -> tensor<1x96x1x1xf32>
    %407 = tosa.add %403, %406 : (tensor<1x96x14x14xf32>, tensor<1x96x1x1xf32>) -> tensor<1x96x14x14xf32>
    %408 = "tosa.const"() <{value = dense<3.000000e+00> : tensor<1x96x14x14xf32>}> : () -> tensor<1x96x14x14xf32>
    %409 = tosa.add %407, %408 : (tensor<1x96x14x14xf32>, tensor<1x96x14x14xf32>) -> tensor<1x96x14x14xf32>
    %410 = tosa.clamp %409 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x96x14x14xf32>) -> tensor<1x96x14x14xf32>
    %411 = tosa.clamp %410 {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0xFF800000 : f32, min_int = -9223372036854775807 : i64} : (tensor<1x96x14x14xf32>) -> tensor<1x96x14x14xf32>
    %412 = tosa.mul %407, %411 {shift = 0 : i8} : (tensor<1x96x14x14xf32>, tensor<1x96x14x14xf32>) -> tensor<1x96x14x14xf32>
    %413 = "tosa.const"() <{value = dense<6.000000e+00> : tensor<1x96x14x14xf32>}> : () -> tensor<1x96x14x14xf32>
    %414 = tosa.reciprocal %413 : (tensor<1x96x14x14xf32>) -> tensor<1x96x14x14xf32>
    %415 = tosa.mul %412, %414 {shift = 0 : i8} : (tensor<1x96x14x14xf32>, tensor<1x96x14x14xf32>) -> tensor<1x96x14x14xf32>
    %416 = tosa.reduce_sum %415 {axis = 3 : i32} : (tensor<1x96x14x14xf32>) -> tensor<1x96x14x1xf32>
    %417 = tosa.reduce_sum %416 {axis = 2 : i32} : (tensor<1x96x14x1xf32>) -> tensor<1x96x1x1xf32>
    %418 = "tosa.const"() <{value = dense<1.960000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %419 = tosa.reciprocal %418 : (tensor<1xf32>) -> tensor<1xf32>
    %420 = tosa.mul %419, %417 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x96x1x1xf32>) -> tensor<1x96x1x1xf32>
    %421 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %422 = tosa.transpose %420, %421 : (tensor<1x96x1x1xf32>, tensor<4xi32>) -> tensor<1x1x1x96xf32>
    %423 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %424 = tosa.transpose %arg60, %423 : (tensor<24x96x1x1xf32>, tensor<4xi32>) -> tensor<24x1x1x96xf32>
    %425 = tosa.conv2d %422, %424, %arg61 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x1x1x96xf32>, tensor<24x1x1x96xf32>, tensor<24xf32>) -> tensor<1x1x1x24xf32>
    %426 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %427 = tosa.transpose %425, %426 : (tensor<1x1x1x24xf32>, tensor<4xi32>) -> tensor<1x24x1x1xf32>
    %428 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x24x1x1xf32>}> : () -> tensor<1x24x1x1xf32>
    %429 = tosa.maximum %427, %428 : (tensor<1x24x1x1xf32>, tensor<1x24x1x1xf32>) -> tensor<1x24x1x1xf32>
    %430 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %431 = tosa.transpose %429, %430 : (tensor<1x24x1x1xf32>, tensor<4xi32>) -> tensor<1x1x1x24xf32>
    %432 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %433 = tosa.transpose %arg62, %432 : (tensor<96x24x1x1xf32>, tensor<4xi32>) -> tensor<96x1x1x24xf32>
    %434 = tosa.conv2d %431, %433, %arg63 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x1x1x24xf32>, tensor<96x1x1x24xf32>, tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %435 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %436 = tosa.transpose %434, %435 : (tensor<1x1x1x96xf32>, tensor<4xi32>) -> tensor<1x96x1x1xf32>
    %437 = "tosa.const"() <{value = dense<3.000000e+00> : tensor<1x96x1x1xf32>}> : () -> tensor<1x96x1x1xf32>
    %438 = tosa.add %436, %437 : (tensor<1x96x1x1xf32>, tensor<1x96x1x1xf32>) -> tensor<1x96x1x1xf32>
    %439 = tosa.clamp %438 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x96x1x1xf32>) -> tensor<1x96x1x1xf32>
    %440 = tosa.clamp %439 {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0xFF800000 : f32, min_int = -9223372036854775807 : i64} : (tensor<1x96x1x1xf32>) -> tensor<1x96x1x1xf32>
    %441 = "tosa.const"() <{value = dense<6.000000e+00> : tensor<1x96x1x1xf32>}> : () -> tensor<1x96x1x1xf32>
    %442 = tosa.reciprocal %441 : (tensor<1x96x1x1xf32>) -> tensor<1x96x1x1xf32>
    %443 = tosa.mul %440, %442 {shift = 0 : i8} : (tensor<1x96x1x1xf32>, tensor<1x96x1x1xf32>) -> tensor<1x96x1x1xf32>
    %444 = tosa.mul %443, %415 {shift = 0 : i8} : (tensor<1x96x1x1xf32>, tensor<1x96x14x14xf32>) -> tensor<1x96x14x14xf32>
    %445 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<40xf32>}> : () -> tensor<40xf32>
    %446 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %447 = tosa.transpose %444, %446 : (tensor<1x96x14x14xf32>, tensor<4xi32>) -> tensor<1x14x14x96xf32>
    %448 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %449 = tosa.transpose %arg64, %448 : (tensor<40x96x1x1xf32>, tensor<4xi32>) -> tensor<40x1x1x96xf32>
    %450 = tosa.conv2d %447, %449, %445 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x14x14x96xf32>, tensor<40x1x1x96xf32>, tensor<40xf32>) -> tensor<1x14x14x40xf32>
    %451 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %452 = tosa.transpose %450, %451 : (tensor<1x14x14x40xf32>, tensor<4xi32>) -> tensor<1x40x14x14xf32>
    %453 = tosa.cast %arg65 : (tensor<40xf32>) -> tensor<40xf32>
    %454 = tosa.cast %arg66 : (tensor<40xf32>) -> tensor<40xf32>
    %455 = "tosa.const"() <{value = dense<1.000000e-03> : tensor<40xf32>}> : () -> tensor<40xf32>
    %456 = tosa.add %454, %455 : (tensor<40xf32>, tensor<40xf32>) -> tensor<40xf32>
    %457 = math.sqrt %456 : tensor<40xf32>
    %458 = tosa.reciprocal %457 : (tensor<40xf32>) -> tensor<40xf32>
    %459 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<40xf32>}> : () -> tensor<40xf32>
    %460 = tosa.mul %458, %459 {shift = 0 : i8} : (tensor<40xf32>, tensor<40xf32>) -> tensor<40xf32>
    %461 = tosa.reshape %453 {new_shape = array<i64: 40, 1>} : (tensor<40xf32>) -> tensor<40x1xf32>
    %462 = tosa.reshape %461 {new_shape = array<i64: 40, 1, 1>} : (tensor<40x1xf32>) -> tensor<40x1x1xf32>
    %463 = tosa.reshape %460 {new_shape = array<i64: 40, 1>} : (tensor<40xf32>) -> tensor<40x1xf32>
    %464 = tosa.reshape %463 {new_shape = array<i64: 40, 1, 1>} : (tensor<40x1xf32>) -> tensor<40x1x1xf32>
    %465 = tosa.reshape %462 {new_shape = array<i64: 1, 40, 1, 1>} : (tensor<40x1x1xf32>) -> tensor<1x40x1x1xf32>
    %466 = tosa.sub %452, %465 : (tensor<1x40x14x14xf32>, tensor<1x40x1x1xf32>) -> tensor<1x40x14x14xf32>
    %467 = tosa.reshape %464 {new_shape = array<i64: 1, 40, 1, 1>} : (tensor<40x1x1xf32>) -> tensor<1x40x1x1xf32>
    %468 = tosa.mul %466, %467 {shift = 0 : i8} : (tensor<1x40x14x14xf32>, tensor<1x40x1x1xf32>) -> tensor<1x40x14x14xf32>
    %469 = tosa.reshape %arg67 {new_shape = array<i64: 40, 1>} : (tensor<40xf32>) -> tensor<40x1xf32>
    %470 = tosa.reshape %469 {new_shape = array<i64: 40, 1, 1>} : (tensor<40x1xf32>) -> tensor<40x1x1xf32>
    %471 = tosa.reshape %470 {new_shape = array<i64: 1, 40, 1, 1>} : (tensor<40x1x1xf32>) -> tensor<1x40x1x1xf32>
    %472 = tosa.mul %468, %471 {shift = 0 : i8} : (tensor<1x40x14x14xf32>, tensor<1x40x1x1xf32>) -> tensor<1x40x14x14xf32>
    %473 = tosa.reshape %arg68 {new_shape = array<i64: 40, 1>} : (tensor<40xf32>) -> tensor<40x1xf32>
    %474 = tosa.reshape %473 {new_shape = array<i64: 40, 1, 1>} : (tensor<40x1xf32>) -> tensor<40x1x1xf32>
    %475 = tosa.reshape %474 {new_shape = array<i64: 1, 40, 1, 1>} : (tensor<40x1x1xf32>) -> tensor<1x40x1x1xf32>
    %476 = tosa.add %472, %475 : (tensor<1x40x14x14xf32>, tensor<1x40x1x1xf32>) -> tensor<1x40x14x14xf32>
    %477 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<240xf32>}> : () -> tensor<240xf32>
    %478 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %479 = tosa.transpose %476, %478 : (tensor<1x40x14x14xf32>, tensor<4xi32>) -> tensor<1x14x14x40xf32>
    %480 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %481 = tosa.transpose %arg69, %480 : (tensor<240x40x1x1xf32>, tensor<4xi32>) -> tensor<240x1x1x40xf32>
    %482 = tosa.conv2d %479, %481, %477 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x14x14x40xf32>, tensor<240x1x1x40xf32>, tensor<240xf32>) -> tensor<1x14x14x240xf32>
    %483 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %484 = tosa.transpose %482, %483 : (tensor<1x14x14x240xf32>, tensor<4xi32>) -> tensor<1x240x14x14xf32>
    %485 = tosa.cast %arg70 : (tensor<240xf32>) -> tensor<240xf32>
    %486 = tosa.cast %arg71 : (tensor<240xf32>) -> tensor<240xf32>
    %487 = "tosa.const"() <{value = dense<1.000000e-03> : tensor<240xf32>}> : () -> tensor<240xf32>
    %488 = tosa.add %486, %487 : (tensor<240xf32>, tensor<240xf32>) -> tensor<240xf32>
    %489 = math.sqrt %488 : tensor<240xf32>
    %490 = tosa.reciprocal %489 : (tensor<240xf32>) -> tensor<240xf32>
    %491 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<240xf32>}> : () -> tensor<240xf32>
    %492 = tosa.mul %490, %491 {shift = 0 : i8} : (tensor<240xf32>, tensor<240xf32>) -> tensor<240xf32>
    %493 = tosa.reshape %485 {new_shape = array<i64: 240, 1>} : (tensor<240xf32>) -> tensor<240x1xf32>
    %494 = tosa.reshape %493 {new_shape = array<i64: 240, 1, 1>} : (tensor<240x1xf32>) -> tensor<240x1x1xf32>
    %495 = tosa.reshape %492 {new_shape = array<i64: 240, 1>} : (tensor<240xf32>) -> tensor<240x1xf32>
    %496 = tosa.reshape %495 {new_shape = array<i64: 240, 1, 1>} : (tensor<240x1xf32>) -> tensor<240x1x1xf32>
    %497 = tosa.reshape %494 {new_shape = array<i64: 1, 240, 1, 1>} : (tensor<240x1x1xf32>) -> tensor<1x240x1x1xf32>
    %498 = tosa.sub %484, %497 : (tensor<1x240x14x14xf32>, tensor<1x240x1x1xf32>) -> tensor<1x240x14x14xf32>
    %499 = tosa.reshape %496 {new_shape = array<i64: 1, 240, 1, 1>} : (tensor<240x1x1xf32>) -> tensor<1x240x1x1xf32>
    %500 = tosa.mul %498, %499 {shift = 0 : i8} : (tensor<1x240x14x14xf32>, tensor<1x240x1x1xf32>) -> tensor<1x240x14x14xf32>
    %501 = tosa.reshape %arg72 {new_shape = array<i64: 240, 1>} : (tensor<240xf32>) -> tensor<240x1xf32>
    %502 = tosa.reshape %501 {new_shape = array<i64: 240, 1, 1>} : (tensor<240x1xf32>) -> tensor<240x1x1xf32>
    %503 = tosa.reshape %502 {new_shape = array<i64: 1, 240, 1, 1>} : (tensor<240x1x1xf32>) -> tensor<1x240x1x1xf32>
    %504 = tosa.mul %500, %503 {shift = 0 : i8} : (tensor<1x240x14x14xf32>, tensor<1x240x1x1xf32>) -> tensor<1x240x14x14xf32>
    %505 = tosa.reshape %arg73 {new_shape = array<i64: 240, 1>} : (tensor<240xf32>) -> tensor<240x1xf32>
    %506 = tosa.reshape %505 {new_shape = array<i64: 240, 1, 1>} : (tensor<240x1xf32>) -> tensor<240x1x1xf32>
    %507 = tosa.reshape %506 {new_shape = array<i64: 1, 240, 1, 1>} : (tensor<240x1x1xf32>) -> tensor<1x240x1x1xf32>
    %508 = tosa.add %504, %507 : (tensor<1x240x14x14xf32>, tensor<1x240x1x1xf32>) -> tensor<1x240x14x14xf32>
    %509 = "tosa.const"() <{value = dense<3.000000e+00> : tensor<1x240x14x14xf32>}> : () -> tensor<1x240x14x14xf32>
    %510 = tosa.add %508, %509 : (tensor<1x240x14x14xf32>, tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %511 = tosa.clamp %510 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %512 = tosa.clamp %511 {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0xFF800000 : f32, min_int = -9223372036854775807 : i64} : (tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %513 = tosa.mul %508, %512 {shift = 0 : i8} : (tensor<1x240x14x14xf32>, tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %514 = "tosa.const"() <{value = dense<6.000000e+00> : tensor<1x240x14x14xf32>}> : () -> tensor<1x240x14x14xf32>
    %515 = tosa.reciprocal %514 : (tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %516 = tosa.mul %513, %515 {shift = 0 : i8} : (tensor<1x240x14x14xf32>, tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %517 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<240xf32>}> : () -> tensor<240xf32>
    %518 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %519 = tosa.transpose %516, %518 : (tensor<1x240x14x14xf32>, tensor<4xi32>) -> tensor<1x14x14x240xf32>
    %520 = "tosa.const"() <{value = dense<[2, 3, 0, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %521 = tosa.transpose %arg74, %520 : (tensor<240x1x5x5xf32>, tensor<4xi32>) -> tensor<5x5x240x1xf32>
    %522 = tosa.depthwise_conv2d %519, %521, %517 {dilation = array<i64: 1, 1>, pad = array<i64: 2, 2, 2, 2>, stride = array<i64: 1, 1>} : (tensor<1x14x14x240xf32>, tensor<5x5x240x1xf32>, tensor<240xf32>) -> tensor<1x14x14x240xf32>
    %523 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %524 = tosa.transpose %522, %523 : (tensor<1x14x14x240xf32>, tensor<4xi32>) -> tensor<1x240x14x14xf32>
    %525 = tosa.cast %arg75 : (tensor<240xf32>) -> tensor<240xf32>
    %526 = tosa.cast %arg76 : (tensor<240xf32>) -> tensor<240xf32>
    %527 = "tosa.const"() <{value = dense<1.000000e-03> : tensor<240xf32>}> : () -> tensor<240xf32>
    %528 = tosa.add %526, %527 : (tensor<240xf32>, tensor<240xf32>) -> tensor<240xf32>
    %529 = math.sqrt %528 : tensor<240xf32>
    %530 = tosa.reciprocal %529 : (tensor<240xf32>) -> tensor<240xf32>
    %531 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<240xf32>}> : () -> tensor<240xf32>
    %532 = tosa.mul %530, %531 {shift = 0 : i8} : (tensor<240xf32>, tensor<240xf32>) -> tensor<240xf32>
    %533 = tosa.reshape %525 {new_shape = array<i64: 240, 1>} : (tensor<240xf32>) -> tensor<240x1xf32>
    %534 = tosa.reshape %533 {new_shape = array<i64: 240, 1, 1>} : (tensor<240x1xf32>) -> tensor<240x1x1xf32>
    %535 = tosa.reshape %532 {new_shape = array<i64: 240, 1>} : (tensor<240xf32>) -> tensor<240x1xf32>
    %536 = tosa.reshape %535 {new_shape = array<i64: 240, 1, 1>} : (tensor<240x1xf32>) -> tensor<240x1x1xf32>
    %537 = tosa.reshape %534 {new_shape = array<i64: 1, 240, 1, 1>} : (tensor<240x1x1xf32>) -> tensor<1x240x1x1xf32>
    %538 = tosa.sub %524, %537 : (tensor<1x240x14x14xf32>, tensor<1x240x1x1xf32>) -> tensor<1x240x14x14xf32>
    %539 = tosa.reshape %536 {new_shape = array<i64: 1, 240, 1, 1>} : (tensor<240x1x1xf32>) -> tensor<1x240x1x1xf32>
    %540 = tosa.mul %538, %539 {shift = 0 : i8} : (tensor<1x240x14x14xf32>, tensor<1x240x1x1xf32>) -> tensor<1x240x14x14xf32>
    %541 = tosa.reshape %arg77 {new_shape = array<i64: 240, 1>} : (tensor<240xf32>) -> tensor<240x1xf32>
    %542 = tosa.reshape %541 {new_shape = array<i64: 240, 1, 1>} : (tensor<240x1xf32>) -> tensor<240x1x1xf32>
    %543 = tosa.reshape %542 {new_shape = array<i64: 1, 240, 1, 1>} : (tensor<240x1x1xf32>) -> tensor<1x240x1x1xf32>
    %544 = tosa.mul %540, %543 {shift = 0 : i8} : (tensor<1x240x14x14xf32>, tensor<1x240x1x1xf32>) -> tensor<1x240x14x14xf32>
    %545 = tosa.reshape %arg78 {new_shape = array<i64: 240, 1>} : (tensor<240xf32>) -> tensor<240x1xf32>
    %546 = tosa.reshape %545 {new_shape = array<i64: 240, 1, 1>} : (tensor<240x1xf32>) -> tensor<240x1x1xf32>
    %547 = tosa.reshape %546 {new_shape = array<i64: 1, 240, 1, 1>} : (tensor<240x1x1xf32>) -> tensor<1x240x1x1xf32>
    %548 = tosa.add %544, %547 : (tensor<1x240x14x14xf32>, tensor<1x240x1x1xf32>) -> tensor<1x240x14x14xf32>
    %549 = "tosa.const"() <{value = dense<3.000000e+00> : tensor<1x240x14x14xf32>}> : () -> tensor<1x240x14x14xf32>
    %550 = tosa.add %548, %549 : (tensor<1x240x14x14xf32>, tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %551 = tosa.clamp %550 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %552 = tosa.clamp %551 {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0xFF800000 : f32, min_int = -9223372036854775807 : i64} : (tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %553 = tosa.mul %548, %552 {shift = 0 : i8} : (tensor<1x240x14x14xf32>, tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %554 = "tosa.const"() <{value = dense<6.000000e+00> : tensor<1x240x14x14xf32>}> : () -> tensor<1x240x14x14xf32>
    %555 = tosa.reciprocal %554 : (tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %556 = tosa.mul %553, %555 {shift = 0 : i8} : (tensor<1x240x14x14xf32>, tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %557 = tosa.reduce_sum %556 {axis = 3 : i32} : (tensor<1x240x14x14xf32>) -> tensor<1x240x14x1xf32>
    %558 = tosa.reduce_sum %557 {axis = 2 : i32} : (tensor<1x240x14x1xf32>) -> tensor<1x240x1x1xf32>
    %559 = "tosa.const"() <{value = dense<1.960000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %560 = tosa.reciprocal %559 : (tensor<1xf32>) -> tensor<1xf32>
    %561 = tosa.mul %560, %558 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x240x1x1xf32>) -> tensor<1x240x1x1xf32>
    %562 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %563 = tosa.transpose %561, %562 : (tensor<1x240x1x1xf32>, tensor<4xi32>) -> tensor<1x1x1x240xf32>
    %564 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %565 = tosa.transpose %arg79, %564 : (tensor<64x240x1x1xf32>, tensor<4xi32>) -> tensor<64x1x1x240xf32>
    %566 = tosa.conv2d %563, %565, %arg80 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x1x1x240xf32>, tensor<64x1x1x240xf32>, tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %567 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %568 = tosa.transpose %566, %567 : (tensor<1x1x1x64xf32>, tensor<4xi32>) -> tensor<1x64x1x1xf32>
    %569 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x64x1x1xf32>}> : () -> tensor<1x64x1x1xf32>
    %570 = tosa.maximum %568, %569 : (tensor<1x64x1x1xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %571 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %572 = tosa.transpose %570, %571 : (tensor<1x64x1x1xf32>, tensor<4xi32>) -> tensor<1x1x1x64xf32>
    %573 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %574 = tosa.transpose %arg81, %573 : (tensor<240x64x1x1xf32>, tensor<4xi32>) -> tensor<240x1x1x64xf32>
    %575 = tosa.conv2d %572, %574, %arg82 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x1x1x64xf32>, tensor<240x1x1x64xf32>, tensor<240xf32>) -> tensor<1x1x1x240xf32>
    %576 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %577 = tosa.transpose %575, %576 : (tensor<1x1x1x240xf32>, tensor<4xi32>) -> tensor<1x240x1x1xf32>
    %578 = "tosa.const"() <{value = dense<3.000000e+00> : tensor<1x240x1x1xf32>}> : () -> tensor<1x240x1x1xf32>
    %579 = tosa.add %577, %578 : (tensor<1x240x1x1xf32>, tensor<1x240x1x1xf32>) -> tensor<1x240x1x1xf32>
    %580 = tosa.clamp %579 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x240x1x1xf32>) -> tensor<1x240x1x1xf32>
    %581 = tosa.clamp %580 {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0xFF800000 : f32, min_int = -9223372036854775807 : i64} : (tensor<1x240x1x1xf32>) -> tensor<1x240x1x1xf32>
    %582 = "tosa.const"() <{value = dense<6.000000e+00> : tensor<1x240x1x1xf32>}> : () -> tensor<1x240x1x1xf32>
    %583 = tosa.reciprocal %582 : (tensor<1x240x1x1xf32>) -> tensor<1x240x1x1xf32>
    %584 = tosa.mul %581, %583 {shift = 0 : i8} : (tensor<1x240x1x1xf32>, tensor<1x240x1x1xf32>) -> tensor<1x240x1x1xf32>
    %585 = tosa.mul %584, %556 {shift = 0 : i8} : (tensor<1x240x1x1xf32>, tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %586 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<40xf32>}> : () -> tensor<40xf32>
    %587 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %588 = tosa.transpose %585, %587 : (tensor<1x240x14x14xf32>, tensor<4xi32>) -> tensor<1x14x14x240xf32>
    %589 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %590 = tosa.transpose %arg83, %589 : (tensor<40x240x1x1xf32>, tensor<4xi32>) -> tensor<40x1x1x240xf32>
    %591 = tosa.conv2d %588, %590, %586 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x14x14x240xf32>, tensor<40x1x1x240xf32>, tensor<40xf32>) -> tensor<1x14x14x40xf32>
    %592 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %593 = tosa.transpose %591, %592 : (tensor<1x14x14x40xf32>, tensor<4xi32>) -> tensor<1x40x14x14xf32>
    %594 = tosa.cast %arg84 : (tensor<40xf32>) -> tensor<40xf32>
    %595 = tosa.cast %arg85 : (tensor<40xf32>) -> tensor<40xf32>
    %596 = "tosa.const"() <{value = dense<1.000000e-03> : tensor<40xf32>}> : () -> tensor<40xf32>
    %597 = tosa.add %595, %596 : (tensor<40xf32>, tensor<40xf32>) -> tensor<40xf32>
    %598 = math.sqrt %597 : tensor<40xf32>
    %599 = tosa.reciprocal %598 : (tensor<40xf32>) -> tensor<40xf32>
    %600 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<40xf32>}> : () -> tensor<40xf32>
    %601 = tosa.mul %599, %600 {shift = 0 : i8} : (tensor<40xf32>, tensor<40xf32>) -> tensor<40xf32>
    %602 = tosa.reshape %594 {new_shape = array<i64: 40, 1>} : (tensor<40xf32>) -> tensor<40x1xf32>
    %603 = tosa.reshape %602 {new_shape = array<i64: 40, 1, 1>} : (tensor<40x1xf32>) -> tensor<40x1x1xf32>
    %604 = tosa.reshape %601 {new_shape = array<i64: 40, 1>} : (tensor<40xf32>) -> tensor<40x1xf32>
    %605 = tosa.reshape %604 {new_shape = array<i64: 40, 1, 1>} : (tensor<40x1xf32>) -> tensor<40x1x1xf32>
    %606 = tosa.reshape %603 {new_shape = array<i64: 1, 40, 1, 1>} : (tensor<40x1x1xf32>) -> tensor<1x40x1x1xf32>
    %607 = tosa.sub %593, %606 : (tensor<1x40x14x14xf32>, tensor<1x40x1x1xf32>) -> tensor<1x40x14x14xf32>
    %608 = tosa.reshape %605 {new_shape = array<i64: 1, 40, 1, 1>} : (tensor<40x1x1xf32>) -> tensor<1x40x1x1xf32>
    %609 = tosa.mul %607, %608 {shift = 0 : i8} : (tensor<1x40x14x14xf32>, tensor<1x40x1x1xf32>) -> tensor<1x40x14x14xf32>
    %610 = tosa.reshape %arg86 {new_shape = array<i64: 40, 1>} : (tensor<40xf32>) -> tensor<40x1xf32>
    %611 = tosa.reshape %610 {new_shape = array<i64: 40, 1, 1>} : (tensor<40x1xf32>) -> tensor<40x1x1xf32>
    %612 = tosa.reshape %611 {new_shape = array<i64: 1, 40, 1, 1>} : (tensor<40x1x1xf32>) -> tensor<1x40x1x1xf32>
    %613 = tosa.mul %609, %612 {shift = 0 : i8} : (tensor<1x40x14x14xf32>, tensor<1x40x1x1xf32>) -> tensor<1x40x14x14xf32>
    %614 = tosa.reshape %arg87 {new_shape = array<i64: 40, 1>} : (tensor<40xf32>) -> tensor<40x1xf32>
    %615 = tosa.reshape %614 {new_shape = array<i64: 40, 1, 1>} : (tensor<40x1xf32>) -> tensor<40x1x1xf32>
    %616 = tosa.reshape %615 {new_shape = array<i64: 1, 40, 1, 1>} : (tensor<40x1x1xf32>) -> tensor<1x40x1x1xf32>
    %617 = tosa.add %613, %616 : (tensor<1x40x14x14xf32>, tensor<1x40x1x1xf32>) -> tensor<1x40x14x14xf32>
    %618 = tosa.add %617, %476 : (tensor<1x40x14x14xf32>, tensor<1x40x14x14xf32>) -> tensor<1x40x14x14xf32>
    %619 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<240xf32>}> : () -> tensor<240xf32>
    %620 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %621 = tosa.transpose %618, %620 : (tensor<1x40x14x14xf32>, tensor<4xi32>) -> tensor<1x14x14x40xf32>
    %622 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %623 = tosa.transpose %arg88, %622 : (tensor<240x40x1x1xf32>, tensor<4xi32>) -> tensor<240x1x1x40xf32>
    %624 = tosa.conv2d %621, %623, %619 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x14x14x40xf32>, tensor<240x1x1x40xf32>, tensor<240xf32>) -> tensor<1x14x14x240xf32>
    %625 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %626 = tosa.transpose %624, %625 : (tensor<1x14x14x240xf32>, tensor<4xi32>) -> tensor<1x240x14x14xf32>
    %627 = tosa.cast %arg89 : (tensor<240xf32>) -> tensor<240xf32>
    %628 = tosa.cast %arg90 : (tensor<240xf32>) -> tensor<240xf32>
    %629 = "tosa.const"() <{value = dense<1.000000e-03> : tensor<240xf32>}> : () -> tensor<240xf32>
    %630 = tosa.add %628, %629 : (tensor<240xf32>, tensor<240xf32>) -> tensor<240xf32>
    %631 = math.sqrt %630 : tensor<240xf32>
    %632 = tosa.reciprocal %631 : (tensor<240xf32>) -> tensor<240xf32>
    %633 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<240xf32>}> : () -> tensor<240xf32>
    %634 = tosa.mul %632, %633 {shift = 0 : i8} : (tensor<240xf32>, tensor<240xf32>) -> tensor<240xf32>
    %635 = tosa.reshape %627 {new_shape = array<i64: 240, 1>} : (tensor<240xf32>) -> tensor<240x1xf32>
    %636 = tosa.reshape %635 {new_shape = array<i64: 240, 1, 1>} : (tensor<240x1xf32>) -> tensor<240x1x1xf32>
    %637 = tosa.reshape %634 {new_shape = array<i64: 240, 1>} : (tensor<240xf32>) -> tensor<240x1xf32>
    %638 = tosa.reshape %637 {new_shape = array<i64: 240, 1, 1>} : (tensor<240x1xf32>) -> tensor<240x1x1xf32>
    %639 = tosa.reshape %636 {new_shape = array<i64: 1, 240, 1, 1>} : (tensor<240x1x1xf32>) -> tensor<1x240x1x1xf32>
    %640 = tosa.sub %626, %639 : (tensor<1x240x14x14xf32>, tensor<1x240x1x1xf32>) -> tensor<1x240x14x14xf32>
    %641 = tosa.reshape %638 {new_shape = array<i64: 1, 240, 1, 1>} : (tensor<240x1x1xf32>) -> tensor<1x240x1x1xf32>
    %642 = tosa.mul %640, %641 {shift = 0 : i8} : (tensor<1x240x14x14xf32>, tensor<1x240x1x1xf32>) -> tensor<1x240x14x14xf32>
    %643 = tosa.reshape %arg91 {new_shape = array<i64: 240, 1>} : (tensor<240xf32>) -> tensor<240x1xf32>
    %644 = tosa.reshape %643 {new_shape = array<i64: 240, 1, 1>} : (tensor<240x1xf32>) -> tensor<240x1x1xf32>
    %645 = tosa.reshape %644 {new_shape = array<i64: 1, 240, 1, 1>} : (tensor<240x1x1xf32>) -> tensor<1x240x1x1xf32>
    %646 = tosa.mul %642, %645 {shift = 0 : i8} : (tensor<1x240x14x14xf32>, tensor<1x240x1x1xf32>) -> tensor<1x240x14x14xf32>
    %647 = tosa.reshape %arg92 {new_shape = array<i64: 240, 1>} : (tensor<240xf32>) -> tensor<240x1xf32>
    %648 = tosa.reshape %647 {new_shape = array<i64: 240, 1, 1>} : (tensor<240x1xf32>) -> tensor<240x1x1xf32>
    %649 = tosa.reshape %648 {new_shape = array<i64: 1, 240, 1, 1>} : (tensor<240x1x1xf32>) -> tensor<1x240x1x1xf32>
    %650 = tosa.add %646, %649 : (tensor<1x240x14x14xf32>, tensor<1x240x1x1xf32>) -> tensor<1x240x14x14xf32>
    %651 = "tosa.const"() <{value = dense<3.000000e+00> : tensor<1x240x14x14xf32>}> : () -> tensor<1x240x14x14xf32>
    %652 = tosa.add %650, %651 : (tensor<1x240x14x14xf32>, tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %653 = tosa.clamp %652 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %654 = tosa.clamp %653 {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0xFF800000 : f32, min_int = -9223372036854775807 : i64} : (tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %655 = tosa.mul %650, %654 {shift = 0 : i8} : (tensor<1x240x14x14xf32>, tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %656 = "tosa.const"() <{value = dense<6.000000e+00> : tensor<1x240x14x14xf32>}> : () -> tensor<1x240x14x14xf32>
    %657 = tosa.reciprocal %656 : (tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %658 = tosa.mul %655, %657 {shift = 0 : i8} : (tensor<1x240x14x14xf32>, tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %659 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<240xf32>}> : () -> tensor<240xf32>
    %660 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %661 = tosa.transpose %658, %660 : (tensor<1x240x14x14xf32>, tensor<4xi32>) -> tensor<1x14x14x240xf32>
    %662 = "tosa.const"() <{value = dense<[2, 3, 0, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %663 = tosa.transpose %arg93, %662 : (tensor<240x1x5x5xf32>, tensor<4xi32>) -> tensor<5x5x240x1xf32>
    %664 = tosa.depthwise_conv2d %661, %663, %659 {dilation = array<i64: 1, 1>, pad = array<i64: 2, 2, 2, 2>, stride = array<i64: 1, 1>} : (tensor<1x14x14x240xf32>, tensor<5x5x240x1xf32>, tensor<240xf32>) -> tensor<1x14x14x240xf32>
    %665 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %666 = tosa.transpose %664, %665 : (tensor<1x14x14x240xf32>, tensor<4xi32>) -> tensor<1x240x14x14xf32>
    %667 = tosa.cast %arg94 : (tensor<240xf32>) -> tensor<240xf32>
    %668 = tosa.cast %arg95 : (tensor<240xf32>) -> tensor<240xf32>
    %669 = "tosa.const"() <{value = dense<1.000000e-03> : tensor<240xf32>}> : () -> tensor<240xf32>
    %670 = tosa.add %668, %669 : (tensor<240xf32>, tensor<240xf32>) -> tensor<240xf32>
    %671 = math.sqrt %670 : tensor<240xf32>
    %672 = tosa.reciprocal %671 : (tensor<240xf32>) -> tensor<240xf32>
    %673 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<240xf32>}> : () -> tensor<240xf32>
    %674 = tosa.mul %672, %673 {shift = 0 : i8} : (tensor<240xf32>, tensor<240xf32>) -> tensor<240xf32>
    %675 = tosa.reshape %667 {new_shape = array<i64: 240, 1>} : (tensor<240xf32>) -> tensor<240x1xf32>
    %676 = tosa.reshape %675 {new_shape = array<i64: 240, 1, 1>} : (tensor<240x1xf32>) -> tensor<240x1x1xf32>
    %677 = tosa.reshape %674 {new_shape = array<i64: 240, 1>} : (tensor<240xf32>) -> tensor<240x1xf32>
    %678 = tosa.reshape %677 {new_shape = array<i64: 240, 1, 1>} : (tensor<240x1xf32>) -> tensor<240x1x1xf32>
    %679 = tosa.reshape %676 {new_shape = array<i64: 1, 240, 1, 1>} : (tensor<240x1x1xf32>) -> tensor<1x240x1x1xf32>
    %680 = tosa.sub %666, %679 : (tensor<1x240x14x14xf32>, tensor<1x240x1x1xf32>) -> tensor<1x240x14x14xf32>
    %681 = tosa.reshape %678 {new_shape = array<i64: 1, 240, 1, 1>} : (tensor<240x1x1xf32>) -> tensor<1x240x1x1xf32>
    %682 = tosa.mul %680, %681 {shift = 0 : i8} : (tensor<1x240x14x14xf32>, tensor<1x240x1x1xf32>) -> tensor<1x240x14x14xf32>
    %683 = tosa.reshape %arg96 {new_shape = array<i64: 240, 1>} : (tensor<240xf32>) -> tensor<240x1xf32>
    %684 = tosa.reshape %683 {new_shape = array<i64: 240, 1, 1>} : (tensor<240x1xf32>) -> tensor<240x1x1xf32>
    %685 = tosa.reshape %684 {new_shape = array<i64: 1, 240, 1, 1>} : (tensor<240x1x1xf32>) -> tensor<1x240x1x1xf32>
    %686 = tosa.mul %682, %685 {shift = 0 : i8} : (tensor<1x240x14x14xf32>, tensor<1x240x1x1xf32>) -> tensor<1x240x14x14xf32>
    %687 = tosa.reshape %arg97 {new_shape = array<i64: 240, 1>} : (tensor<240xf32>) -> tensor<240x1xf32>
    %688 = tosa.reshape %687 {new_shape = array<i64: 240, 1, 1>} : (tensor<240x1xf32>) -> tensor<240x1x1xf32>
    %689 = tosa.reshape %688 {new_shape = array<i64: 1, 240, 1, 1>} : (tensor<240x1x1xf32>) -> tensor<1x240x1x1xf32>
    %690 = tosa.add %686, %689 : (tensor<1x240x14x14xf32>, tensor<1x240x1x1xf32>) -> tensor<1x240x14x14xf32>
    %691 = "tosa.const"() <{value = dense<3.000000e+00> : tensor<1x240x14x14xf32>}> : () -> tensor<1x240x14x14xf32>
    %692 = tosa.add %690, %691 : (tensor<1x240x14x14xf32>, tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %693 = tosa.clamp %692 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %694 = tosa.clamp %693 {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0xFF800000 : f32, min_int = -9223372036854775807 : i64} : (tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %695 = tosa.mul %690, %694 {shift = 0 : i8} : (tensor<1x240x14x14xf32>, tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %696 = "tosa.const"() <{value = dense<6.000000e+00> : tensor<1x240x14x14xf32>}> : () -> tensor<1x240x14x14xf32>
    %697 = tosa.reciprocal %696 : (tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %698 = tosa.mul %695, %697 {shift = 0 : i8} : (tensor<1x240x14x14xf32>, tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %699 = tosa.reduce_sum %698 {axis = 3 : i32} : (tensor<1x240x14x14xf32>) -> tensor<1x240x14x1xf32>
    %700 = tosa.reduce_sum %699 {axis = 2 : i32} : (tensor<1x240x14x1xf32>) -> tensor<1x240x1x1xf32>
    %701 = "tosa.const"() <{value = dense<1.960000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %702 = tosa.reciprocal %701 : (tensor<1xf32>) -> tensor<1xf32>
    %703 = tosa.mul %702, %700 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x240x1x1xf32>) -> tensor<1x240x1x1xf32>
    %704 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %705 = tosa.transpose %703, %704 : (tensor<1x240x1x1xf32>, tensor<4xi32>) -> tensor<1x1x1x240xf32>
    %706 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %707 = tosa.transpose %arg98, %706 : (tensor<64x240x1x1xf32>, tensor<4xi32>) -> tensor<64x1x1x240xf32>
    %708 = tosa.conv2d %705, %707, %arg99 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x1x1x240xf32>, tensor<64x1x1x240xf32>, tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %709 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %710 = tosa.transpose %708, %709 : (tensor<1x1x1x64xf32>, tensor<4xi32>) -> tensor<1x64x1x1xf32>
    %711 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x64x1x1xf32>}> : () -> tensor<1x64x1x1xf32>
    %712 = tosa.maximum %710, %711 : (tensor<1x64x1x1xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %713 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %714 = tosa.transpose %712, %713 : (tensor<1x64x1x1xf32>, tensor<4xi32>) -> tensor<1x1x1x64xf32>
    %715 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %716 = tosa.transpose %arg100, %715 : (tensor<240x64x1x1xf32>, tensor<4xi32>) -> tensor<240x1x1x64xf32>
    %717 = tosa.conv2d %714, %716, %arg101 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x1x1x64xf32>, tensor<240x1x1x64xf32>, tensor<240xf32>) -> tensor<1x1x1x240xf32>
    %718 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %719 = tosa.transpose %717, %718 : (tensor<1x1x1x240xf32>, tensor<4xi32>) -> tensor<1x240x1x1xf32>
    %720 = "tosa.const"() <{value = dense<3.000000e+00> : tensor<1x240x1x1xf32>}> : () -> tensor<1x240x1x1xf32>
    %721 = tosa.add %719, %720 : (tensor<1x240x1x1xf32>, tensor<1x240x1x1xf32>) -> tensor<1x240x1x1xf32>
    %722 = tosa.clamp %721 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x240x1x1xf32>) -> tensor<1x240x1x1xf32>
    %723 = tosa.clamp %722 {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0xFF800000 : f32, min_int = -9223372036854775807 : i64} : (tensor<1x240x1x1xf32>) -> tensor<1x240x1x1xf32>
    %724 = "tosa.const"() <{value = dense<6.000000e+00> : tensor<1x240x1x1xf32>}> : () -> tensor<1x240x1x1xf32>
    %725 = tosa.reciprocal %724 : (tensor<1x240x1x1xf32>) -> tensor<1x240x1x1xf32>
    %726 = tosa.mul %723, %725 {shift = 0 : i8} : (tensor<1x240x1x1xf32>, tensor<1x240x1x1xf32>) -> tensor<1x240x1x1xf32>
    %727 = tosa.mul %726, %698 {shift = 0 : i8} : (tensor<1x240x1x1xf32>, tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %728 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<40xf32>}> : () -> tensor<40xf32>
    %729 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %730 = tosa.transpose %727, %729 : (tensor<1x240x14x14xf32>, tensor<4xi32>) -> tensor<1x14x14x240xf32>
    %731 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %732 = tosa.transpose %arg102, %731 : (tensor<40x240x1x1xf32>, tensor<4xi32>) -> tensor<40x1x1x240xf32>
    %733 = tosa.conv2d %730, %732, %728 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x14x14x240xf32>, tensor<40x1x1x240xf32>, tensor<40xf32>) -> tensor<1x14x14x40xf32>
    %734 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %735 = tosa.transpose %733, %734 : (tensor<1x14x14x40xf32>, tensor<4xi32>) -> tensor<1x40x14x14xf32>
    %736 = tosa.cast %arg103 : (tensor<40xf32>) -> tensor<40xf32>
    %737 = tosa.cast %arg104 : (tensor<40xf32>) -> tensor<40xf32>
    %738 = "tosa.const"() <{value = dense<1.000000e-03> : tensor<40xf32>}> : () -> tensor<40xf32>
    %739 = tosa.add %737, %738 : (tensor<40xf32>, tensor<40xf32>) -> tensor<40xf32>
    %740 = math.sqrt %739 : tensor<40xf32>
    %741 = tosa.reciprocal %740 : (tensor<40xf32>) -> tensor<40xf32>
    %742 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<40xf32>}> : () -> tensor<40xf32>
    %743 = tosa.mul %741, %742 {shift = 0 : i8} : (tensor<40xf32>, tensor<40xf32>) -> tensor<40xf32>
    %744 = tosa.reshape %736 {new_shape = array<i64: 40, 1>} : (tensor<40xf32>) -> tensor<40x1xf32>
    %745 = tosa.reshape %744 {new_shape = array<i64: 40, 1, 1>} : (tensor<40x1xf32>) -> tensor<40x1x1xf32>
    %746 = tosa.reshape %743 {new_shape = array<i64: 40, 1>} : (tensor<40xf32>) -> tensor<40x1xf32>
    %747 = tosa.reshape %746 {new_shape = array<i64: 40, 1, 1>} : (tensor<40x1xf32>) -> tensor<40x1x1xf32>
    %748 = tosa.reshape %745 {new_shape = array<i64: 1, 40, 1, 1>} : (tensor<40x1x1xf32>) -> tensor<1x40x1x1xf32>
    %749 = tosa.sub %735, %748 : (tensor<1x40x14x14xf32>, tensor<1x40x1x1xf32>) -> tensor<1x40x14x14xf32>
    %750 = tosa.reshape %747 {new_shape = array<i64: 1, 40, 1, 1>} : (tensor<40x1x1xf32>) -> tensor<1x40x1x1xf32>
    %751 = tosa.mul %749, %750 {shift = 0 : i8} : (tensor<1x40x14x14xf32>, tensor<1x40x1x1xf32>) -> tensor<1x40x14x14xf32>
    %752 = tosa.reshape %arg105 {new_shape = array<i64: 40, 1>} : (tensor<40xf32>) -> tensor<40x1xf32>
    %753 = tosa.reshape %752 {new_shape = array<i64: 40, 1, 1>} : (tensor<40x1xf32>) -> tensor<40x1x1xf32>
    %754 = tosa.reshape %753 {new_shape = array<i64: 1, 40, 1, 1>} : (tensor<40x1x1xf32>) -> tensor<1x40x1x1xf32>
    %755 = tosa.mul %751, %754 {shift = 0 : i8} : (tensor<1x40x14x14xf32>, tensor<1x40x1x1xf32>) -> tensor<1x40x14x14xf32>
    %756 = tosa.reshape %arg106 {new_shape = array<i64: 40, 1>} : (tensor<40xf32>) -> tensor<40x1xf32>
    %757 = tosa.reshape %756 {new_shape = array<i64: 40, 1, 1>} : (tensor<40x1xf32>) -> tensor<40x1x1xf32>
    %758 = tosa.reshape %757 {new_shape = array<i64: 1, 40, 1, 1>} : (tensor<40x1x1xf32>) -> tensor<1x40x1x1xf32>
    %759 = tosa.add %755, %758 : (tensor<1x40x14x14xf32>, tensor<1x40x1x1xf32>) -> tensor<1x40x14x14xf32>
    %760 = tosa.add %759, %618 : (tensor<1x40x14x14xf32>, tensor<1x40x14x14xf32>) -> tensor<1x40x14x14xf32>
    %761 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<120xf32>}> : () -> tensor<120xf32>
    %762 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %763 = tosa.transpose %760, %762 : (tensor<1x40x14x14xf32>, tensor<4xi32>) -> tensor<1x14x14x40xf32>
    %764 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %765 = tosa.transpose %arg107, %764 : (tensor<120x40x1x1xf32>, tensor<4xi32>) -> tensor<120x1x1x40xf32>
    %766 = tosa.conv2d %763, %765, %761 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x14x14x40xf32>, tensor<120x1x1x40xf32>, tensor<120xf32>) -> tensor<1x14x14x120xf32>
    %767 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %768 = tosa.transpose %766, %767 : (tensor<1x14x14x120xf32>, tensor<4xi32>) -> tensor<1x120x14x14xf32>
    %769 = tosa.cast %arg108 : (tensor<120xf32>) -> tensor<120xf32>
    %770 = tosa.cast %arg109 : (tensor<120xf32>) -> tensor<120xf32>
    %771 = "tosa.const"() <{value = dense<1.000000e-03> : tensor<120xf32>}> : () -> tensor<120xf32>
    %772 = tosa.add %770, %771 : (tensor<120xf32>, tensor<120xf32>) -> tensor<120xf32>
    %773 = math.sqrt %772 : tensor<120xf32>
    %774 = tosa.reciprocal %773 : (tensor<120xf32>) -> tensor<120xf32>
    %775 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<120xf32>}> : () -> tensor<120xf32>
    %776 = tosa.mul %774, %775 {shift = 0 : i8} : (tensor<120xf32>, tensor<120xf32>) -> tensor<120xf32>
    %777 = tosa.reshape %769 {new_shape = array<i64: 120, 1>} : (tensor<120xf32>) -> tensor<120x1xf32>
    %778 = tosa.reshape %777 {new_shape = array<i64: 120, 1, 1>} : (tensor<120x1xf32>) -> tensor<120x1x1xf32>
    %779 = tosa.reshape %776 {new_shape = array<i64: 120, 1>} : (tensor<120xf32>) -> tensor<120x1xf32>
    %780 = tosa.reshape %779 {new_shape = array<i64: 120, 1, 1>} : (tensor<120x1xf32>) -> tensor<120x1x1xf32>
    %781 = tosa.reshape %778 {new_shape = array<i64: 1, 120, 1, 1>} : (tensor<120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %782 = tosa.sub %768, %781 : (tensor<1x120x14x14xf32>, tensor<1x120x1x1xf32>) -> tensor<1x120x14x14xf32>
    %783 = tosa.reshape %780 {new_shape = array<i64: 1, 120, 1, 1>} : (tensor<120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %784 = tosa.mul %782, %783 {shift = 0 : i8} : (tensor<1x120x14x14xf32>, tensor<1x120x1x1xf32>) -> tensor<1x120x14x14xf32>
    %785 = tosa.reshape %arg110 {new_shape = array<i64: 120, 1>} : (tensor<120xf32>) -> tensor<120x1xf32>
    %786 = tosa.reshape %785 {new_shape = array<i64: 120, 1, 1>} : (tensor<120x1xf32>) -> tensor<120x1x1xf32>
    %787 = tosa.reshape %786 {new_shape = array<i64: 1, 120, 1, 1>} : (tensor<120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %788 = tosa.mul %784, %787 {shift = 0 : i8} : (tensor<1x120x14x14xf32>, tensor<1x120x1x1xf32>) -> tensor<1x120x14x14xf32>
    %789 = tosa.reshape %arg111 {new_shape = array<i64: 120, 1>} : (tensor<120xf32>) -> tensor<120x1xf32>
    %790 = tosa.reshape %789 {new_shape = array<i64: 120, 1, 1>} : (tensor<120x1xf32>) -> tensor<120x1x1xf32>
    %791 = tosa.reshape %790 {new_shape = array<i64: 1, 120, 1, 1>} : (tensor<120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %792 = tosa.add %788, %791 : (tensor<1x120x14x14xf32>, tensor<1x120x1x1xf32>) -> tensor<1x120x14x14xf32>
    %793 = "tosa.const"() <{value = dense<3.000000e+00> : tensor<1x120x14x14xf32>}> : () -> tensor<1x120x14x14xf32>
    %794 = tosa.add %792, %793 : (tensor<1x120x14x14xf32>, tensor<1x120x14x14xf32>) -> tensor<1x120x14x14xf32>
    %795 = tosa.clamp %794 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x120x14x14xf32>) -> tensor<1x120x14x14xf32>
    %796 = tosa.clamp %795 {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0xFF800000 : f32, min_int = -9223372036854775807 : i64} : (tensor<1x120x14x14xf32>) -> tensor<1x120x14x14xf32>
    %797 = tosa.mul %792, %796 {shift = 0 : i8} : (tensor<1x120x14x14xf32>, tensor<1x120x14x14xf32>) -> tensor<1x120x14x14xf32>
    %798 = "tosa.const"() <{value = dense<6.000000e+00> : tensor<1x120x14x14xf32>}> : () -> tensor<1x120x14x14xf32>
    %799 = tosa.reciprocal %798 : (tensor<1x120x14x14xf32>) -> tensor<1x120x14x14xf32>
    %800 = tosa.mul %797, %799 {shift = 0 : i8} : (tensor<1x120x14x14xf32>, tensor<1x120x14x14xf32>) -> tensor<1x120x14x14xf32>
    %801 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<120xf32>}> : () -> tensor<120xf32>
    %802 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %803 = tosa.transpose %800, %802 : (tensor<1x120x14x14xf32>, tensor<4xi32>) -> tensor<1x14x14x120xf32>
    %804 = "tosa.const"() <{value = dense<[2, 3, 0, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %805 = tosa.transpose %arg112, %804 : (tensor<120x1x5x5xf32>, tensor<4xi32>) -> tensor<5x5x120x1xf32>
    %806 = tosa.depthwise_conv2d %803, %805, %801 {dilation = array<i64: 1, 1>, pad = array<i64: 2, 2, 2, 2>, stride = array<i64: 1, 1>} : (tensor<1x14x14x120xf32>, tensor<5x5x120x1xf32>, tensor<120xf32>) -> tensor<1x14x14x120xf32>
    %807 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %808 = tosa.transpose %806, %807 : (tensor<1x14x14x120xf32>, tensor<4xi32>) -> tensor<1x120x14x14xf32>
    %809 = tosa.cast %arg113 : (tensor<120xf32>) -> tensor<120xf32>
    %810 = tosa.cast %arg114 : (tensor<120xf32>) -> tensor<120xf32>
    %811 = "tosa.const"() <{value = dense<1.000000e-03> : tensor<120xf32>}> : () -> tensor<120xf32>
    %812 = tosa.add %810, %811 : (tensor<120xf32>, tensor<120xf32>) -> tensor<120xf32>
    %813 = math.sqrt %812 : tensor<120xf32>
    %814 = tosa.reciprocal %813 : (tensor<120xf32>) -> tensor<120xf32>
    %815 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<120xf32>}> : () -> tensor<120xf32>
    %816 = tosa.mul %814, %815 {shift = 0 : i8} : (tensor<120xf32>, tensor<120xf32>) -> tensor<120xf32>
    %817 = tosa.reshape %809 {new_shape = array<i64: 120, 1>} : (tensor<120xf32>) -> tensor<120x1xf32>
    %818 = tosa.reshape %817 {new_shape = array<i64: 120, 1, 1>} : (tensor<120x1xf32>) -> tensor<120x1x1xf32>
    %819 = tosa.reshape %816 {new_shape = array<i64: 120, 1>} : (tensor<120xf32>) -> tensor<120x1xf32>
    %820 = tosa.reshape %819 {new_shape = array<i64: 120, 1, 1>} : (tensor<120x1xf32>) -> tensor<120x1x1xf32>
    %821 = tosa.reshape %818 {new_shape = array<i64: 1, 120, 1, 1>} : (tensor<120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %822 = tosa.sub %808, %821 : (tensor<1x120x14x14xf32>, tensor<1x120x1x1xf32>) -> tensor<1x120x14x14xf32>
    %823 = tosa.reshape %820 {new_shape = array<i64: 1, 120, 1, 1>} : (tensor<120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %824 = tosa.mul %822, %823 {shift = 0 : i8} : (tensor<1x120x14x14xf32>, tensor<1x120x1x1xf32>) -> tensor<1x120x14x14xf32>
    %825 = tosa.reshape %arg115 {new_shape = array<i64: 120, 1>} : (tensor<120xf32>) -> tensor<120x1xf32>
    %826 = tosa.reshape %825 {new_shape = array<i64: 120, 1, 1>} : (tensor<120x1xf32>) -> tensor<120x1x1xf32>
    %827 = tosa.reshape %826 {new_shape = array<i64: 1, 120, 1, 1>} : (tensor<120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %828 = tosa.mul %824, %827 {shift = 0 : i8} : (tensor<1x120x14x14xf32>, tensor<1x120x1x1xf32>) -> tensor<1x120x14x14xf32>
    %829 = tosa.reshape %arg116 {new_shape = array<i64: 120, 1>} : (tensor<120xf32>) -> tensor<120x1xf32>
    %830 = tosa.reshape %829 {new_shape = array<i64: 120, 1, 1>} : (tensor<120x1xf32>) -> tensor<120x1x1xf32>
    %831 = tosa.reshape %830 {new_shape = array<i64: 1, 120, 1, 1>} : (tensor<120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %832 = tosa.add %828, %831 : (tensor<1x120x14x14xf32>, tensor<1x120x1x1xf32>) -> tensor<1x120x14x14xf32>
    %833 = "tosa.const"() <{value = dense<3.000000e+00> : tensor<1x120x14x14xf32>}> : () -> tensor<1x120x14x14xf32>
    %834 = tosa.add %832, %833 : (tensor<1x120x14x14xf32>, tensor<1x120x14x14xf32>) -> tensor<1x120x14x14xf32>
    %835 = tosa.clamp %834 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x120x14x14xf32>) -> tensor<1x120x14x14xf32>
    %836 = tosa.clamp %835 {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0xFF800000 : f32, min_int = -9223372036854775807 : i64} : (tensor<1x120x14x14xf32>) -> tensor<1x120x14x14xf32>
    %837 = tosa.mul %832, %836 {shift = 0 : i8} : (tensor<1x120x14x14xf32>, tensor<1x120x14x14xf32>) -> tensor<1x120x14x14xf32>
    %838 = "tosa.const"() <{value = dense<6.000000e+00> : tensor<1x120x14x14xf32>}> : () -> tensor<1x120x14x14xf32>
    %839 = tosa.reciprocal %838 : (tensor<1x120x14x14xf32>) -> tensor<1x120x14x14xf32>
    %840 = tosa.mul %837, %839 {shift = 0 : i8} : (tensor<1x120x14x14xf32>, tensor<1x120x14x14xf32>) -> tensor<1x120x14x14xf32>
    %841 = tosa.reduce_sum %840 {axis = 3 : i32} : (tensor<1x120x14x14xf32>) -> tensor<1x120x14x1xf32>
    %842 = tosa.reduce_sum %841 {axis = 2 : i32} : (tensor<1x120x14x1xf32>) -> tensor<1x120x1x1xf32>
    %843 = "tosa.const"() <{value = dense<1.960000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %844 = tosa.reciprocal %843 : (tensor<1xf32>) -> tensor<1xf32>
    %845 = tosa.mul %844, %842 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %846 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %847 = tosa.transpose %845, %846 : (tensor<1x120x1x1xf32>, tensor<4xi32>) -> tensor<1x1x1x120xf32>
    %848 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %849 = tosa.transpose %arg117, %848 : (tensor<32x120x1x1xf32>, tensor<4xi32>) -> tensor<32x1x1x120xf32>
    %850 = tosa.conv2d %847, %849, %arg118 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x1x1x120xf32>, tensor<32x1x1x120xf32>, tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %851 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %852 = tosa.transpose %850, %851 : (tensor<1x1x1x32xf32>, tensor<4xi32>) -> tensor<1x32x1x1xf32>
    %853 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %854 = tosa.maximum %852, %853 : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %855 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %856 = tosa.transpose %854, %855 : (tensor<1x32x1x1xf32>, tensor<4xi32>) -> tensor<1x1x1x32xf32>
    %857 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %858 = tosa.transpose %arg119, %857 : (tensor<120x32x1x1xf32>, tensor<4xi32>) -> tensor<120x1x1x32xf32>
    %859 = tosa.conv2d %856, %858, %arg120 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x1x1x32xf32>, tensor<120x1x1x32xf32>, tensor<120xf32>) -> tensor<1x1x1x120xf32>
    %860 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %861 = tosa.transpose %859, %860 : (tensor<1x1x1x120xf32>, tensor<4xi32>) -> tensor<1x120x1x1xf32>
    %862 = "tosa.const"() <{value = dense<3.000000e+00> : tensor<1x120x1x1xf32>}> : () -> tensor<1x120x1x1xf32>
    %863 = tosa.add %861, %862 : (tensor<1x120x1x1xf32>, tensor<1x120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %864 = tosa.clamp %863 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %865 = tosa.clamp %864 {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0xFF800000 : f32, min_int = -9223372036854775807 : i64} : (tensor<1x120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %866 = "tosa.const"() <{value = dense<6.000000e+00> : tensor<1x120x1x1xf32>}> : () -> tensor<1x120x1x1xf32>
    %867 = tosa.reciprocal %866 : (tensor<1x120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %868 = tosa.mul %865, %867 {shift = 0 : i8} : (tensor<1x120x1x1xf32>, tensor<1x120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %869 = tosa.mul %868, %840 {shift = 0 : i8} : (tensor<1x120x1x1xf32>, tensor<1x120x14x14xf32>) -> tensor<1x120x14x14xf32>
    %870 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<48xf32>}> : () -> tensor<48xf32>
    %871 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %872 = tosa.transpose %869, %871 : (tensor<1x120x14x14xf32>, tensor<4xi32>) -> tensor<1x14x14x120xf32>
    %873 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %874 = tosa.transpose %arg121, %873 : (tensor<48x120x1x1xf32>, tensor<4xi32>) -> tensor<48x1x1x120xf32>
    %875 = tosa.conv2d %872, %874, %870 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x14x14x120xf32>, tensor<48x1x1x120xf32>, tensor<48xf32>) -> tensor<1x14x14x48xf32>
    %876 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %877 = tosa.transpose %875, %876 : (tensor<1x14x14x48xf32>, tensor<4xi32>) -> tensor<1x48x14x14xf32>
    %878 = tosa.cast %arg122 : (tensor<48xf32>) -> tensor<48xf32>
    %879 = tosa.cast %arg123 : (tensor<48xf32>) -> tensor<48xf32>
    %880 = "tosa.const"() <{value = dense<1.000000e-03> : tensor<48xf32>}> : () -> tensor<48xf32>
    %881 = tosa.add %879, %880 : (tensor<48xf32>, tensor<48xf32>) -> tensor<48xf32>
    %882 = math.sqrt %881 : tensor<48xf32>
    %883 = tosa.reciprocal %882 : (tensor<48xf32>) -> tensor<48xf32>
    %884 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<48xf32>}> : () -> tensor<48xf32>
    %885 = tosa.mul %883, %884 {shift = 0 : i8} : (tensor<48xf32>, tensor<48xf32>) -> tensor<48xf32>
    %886 = tosa.reshape %878 {new_shape = array<i64: 48, 1>} : (tensor<48xf32>) -> tensor<48x1xf32>
    %887 = tosa.reshape %886 {new_shape = array<i64: 48, 1, 1>} : (tensor<48x1xf32>) -> tensor<48x1x1xf32>
    %888 = tosa.reshape %885 {new_shape = array<i64: 48, 1>} : (tensor<48xf32>) -> tensor<48x1xf32>
    %889 = tosa.reshape %888 {new_shape = array<i64: 48, 1, 1>} : (tensor<48x1xf32>) -> tensor<48x1x1xf32>
    %890 = tosa.reshape %887 {new_shape = array<i64: 1, 48, 1, 1>} : (tensor<48x1x1xf32>) -> tensor<1x48x1x1xf32>
    %891 = tosa.sub %877, %890 : (tensor<1x48x14x14xf32>, tensor<1x48x1x1xf32>) -> tensor<1x48x14x14xf32>
    %892 = tosa.reshape %889 {new_shape = array<i64: 1, 48, 1, 1>} : (tensor<48x1x1xf32>) -> tensor<1x48x1x1xf32>
    %893 = tosa.mul %891, %892 {shift = 0 : i8} : (tensor<1x48x14x14xf32>, tensor<1x48x1x1xf32>) -> tensor<1x48x14x14xf32>
    %894 = tosa.reshape %arg124 {new_shape = array<i64: 48, 1>} : (tensor<48xf32>) -> tensor<48x1xf32>
    %895 = tosa.reshape %894 {new_shape = array<i64: 48, 1, 1>} : (tensor<48x1xf32>) -> tensor<48x1x1xf32>
    %896 = tosa.reshape %895 {new_shape = array<i64: 1, 48, 1, 1>} : (tensor<48x1x1xf32>) -> tensor<1x48x1x1xf32>
    %897 = tosa.mul %893, %896 {shift = 0 : i8} : (tensor<1x48x14x14xf32>, tensor<1x48x1x1xf32>) -> tensor<1x48x14x14xf32>
    %898 = tosa.reshape %arg125 {new_shape = array<i64: 48, 1>} : (tensor<48xf32>) -> tensor<48x1xf32>
    %899 = tosa.reshape %898 {new_shape = array<i64: 48, 1, 1>} : (tensor<48x1xf32>) -> tensor<48x1x1xf32>
    %900 = tosa.reshape %899 {new_shape = array<i64: 1, 48, 1, 1>} : (tensor<48x1x1xf32>) -> tensor<1x48x1x1xf32>
    %901 = tosa.add %897, %900 : (tensor<1x48x14x14xf32>, tensor<1x48x1x1xf32>) -> tensor<1x48x14x14xf32>
    %902 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<144xf32>}> : () -> tensor<144xf32>
    %903 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %904 = tosa.transpose %901, %903 : (tensor<1x48x14x14xf32>, tensor<4xi32>) -> tensor<1x14x14x48xf32>
    %905 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %906 = tosa.transpose %arg126, %905 : (tensor<144x48x1x1xf32>, tensor<4xi32>) -> tensor<144x1x1x48xf32>
    %907 = tosa.conv2d %904, %906, %902 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x14x14x48xf32>, tensor<144x1x1x48xf32>, tensor<144xf32>) -> tensor<1x14x14x144xf32>
    %908 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %909 = tosa.transpose %907, %908 : (tensor<1x14x14x144xf32>, tensor<4xi32>) -> tensor<1x144x14x14xf32>
    %910 = tosa.cast %arg127 : (tensor<144xf32>) -> tensor<144xf32>
    %911 = tosa.cast %arg128 : (tensor<144xf32>) -> tensor<144xf32>
    %912 = "tosa.const"() <{value = dense<1.000000e-03> : tensor<144xf32>}> : () -> tensor<144xf32>
    %913 = tosa.add %911, %912 : (tensor<144xf32>, tensor<144xf32>) -> tensor<144xf32>
    %914 = math.sqrt %913 : tensor<144xf32>
    %915 = tosa.reciprocal %914 : (tensor<144xf32>) -> tensor<144xf32>
    %916 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<144xf32>}> : () -> tensor<144xf32>
    %917 = tosa.mul %915, %916 {shift = 0 : i8} : (tensor<144xf32>, tensor<144xf32>) -> tensor<144xf32>
    %918 = tosa.reshape %910 {new_shape = array<i64: 144, 1>} : (tensor<144xf32>) -> tensor<144x1xf32>
    %919 = tosa.reshape %918 {new_shape = array<i64: 144, 1, 1>} : (tensor<144x1xf32>) -> tensor<144x1x1xf32>
    %920 = tosa.reshape %917 {new_shape = array<i64: 144, 1>} : (tensor<144xf32>) -> tensor<144x1xf32>
    %921 = tosa.reshape %920 {new_shape = array<i64: 144, 1, 1>} : (tensor<144x1xf32>) -> tensor<144x1x1xf32>
    %922 = tosa.reshape %919 {new_shape = array<i64: 1, 144, 1, 1>} : (tensor<144x1x1xf32>) -> tensor<1x144x1x1xf32>
    %923 = tosa.sub %909, %922 : (tensor<1x144x14x14xf32>, tensor<1x144x1x1xf32>) -> tensor<1x144x14x14xf32>
    %924 = tosa.reshape %921 {new_shape = array<i64: 1, 144, 1, 1>} : (tensor<144x1x1xf32>) -> tensor<1x144x1x1xf32>
    %925 = tosa.mul %923, %924 {shift = 0 : i8} : (tensor<1x144x14x14xf32>, tensor<1x144x1x1xf32>) -> tensor<1x144x14x14xf32>
    %926 = tosa.reshape %arg129 {new_shape = array<i64: 144, 1>} : (tensor<144xf32>) -> tensor<144x1xf32>
    %927 = tosa.reshape %926 {new_shape = array<i64: 144, 1, 1>} : (tensor<144x1xf32>) -> tensor<144x1x1xf32>
    %928 = tosa.reshape %927 {new_shape = array<i64: 1, 144, 1, 1>} : (tensor<144x1x1xf32>) -> tensor<1x144x1x1xf32>
    %929 = tosa.mul %925, %928 {shift = 0 : i8} : (tensor<1x144x14x14xf32>, tensor<1x144x1x1xf32>) -> tensor<1x144x14x14xf32>
    %930 = tosa.reshape %arg130 {new_shape = array<i64: 144, 1>} : (tensor<144xf32>) -> tensor<144x1xf32>
    %931 = tosa.reshape %930 {new_shape = array<i64: 144, 1, 1>} : (tensor<144x1xf32>) -> tensor<144x1x1xf32>
    %932 = tosa.reshape %931 {new_shape = array<i64: 1, 144, 1, 1>} : (tensor<144x1x1xf32>) -> tensor<1x144x1x1xf32>
    %933 = tosa.add %929, %932 : (tensor<1x144x14x14xf32>, tensor<1x144x1x1xf32>) -> tensor<1x144x14x14xf32>
    %934 = "tosa.const"() <{value = dense<3.000000e+00> : tensor<1x144x14x14xf32>}> : () -> tensor<1x144x14x14xf32>
    %935 = tosa.add %933, %934 : (tensor<1x144x14x14xf32>, tensor<1x144x14x14xf32>) -> tensor<1x144x14x14xf32>
    %936 = tosa.clamp %935 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x144x14x14xf32>) -> tensor<1x144x14x14xf32>
    %937 = tosa.clamp %936 {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0xFF800000 : f32, min_int = -9223372036854775807 : i64} : (tensor<1x144x14x14xf32>) -> tensor<1x144x14x14xf32>
    %938 = tosa.mul %933, %937 {shift = 0 : i8} : (tensor<1x144x14x14xf32>, tensor<1x144x14x14xf32>) -> tensor<1x144x14x14xf32>
    %939 = "tosa.const"() <{value = dense<6.000000e+00> : tensor<1x144x14x14xf32>}> : () -> tensor<1x144x14x14xf32>
    %940 = tosa.reciprocal %939 : (tensor<1x144x14x14xf32>) -> tensor<1x144x14x14xf32>
    %941 = tosa.mul %938, %940 {shift = 0 : i8} : (tensor<1x144x14x14xf32>, tensor<1x144x14x14xf32>) -> tensor<1x144x14x14xf32>
    %942 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<144xf32>}> : () -> tensor<144xf32>
    %943 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %944 = tosa.transpose %941, %943 : (tensor<1x144x14x14xf32>, tensor<4xi32>) -> tensor<1x14x14x144xf32>
    %945 = "tosa.const"() <{value = dense<[2, 3, 0, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %946 = tosa.transpose %arg131, %945 : (tensor<144x1x5x5xf32>, tensor<4xi32>) -> tensor<5x5x144x1xf32>
    %947 = tosa.depthwise_conv2d %944, %946, %942 {dilation = array<i64: 1, 1>, pad = array<i64: 2, 2, 2, 2>, stride = array<i64: 1, 1>} : (tensor<1x14x14x144xf32>, tensor<5x5x144x1xf32>, tensor<144xf32>) -> tensor<1x14x14x144xf32>
    %948 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %949 = tosa.transpose %947, %948 : (tensor<1x14x14x144xf32>, tensor<4xi32>) -> tensor<1x144x14x14xf32>
    %950 = tosa.cast %arg132 : (tensor<144xf32>) -> tensor<144xf32>
    %951 = tosa.cast %arg133 : (tensor<144xf32>) -> tensor<144xf32>
    %952 = "tosa.const"() <{value = dense<1.000000e-03> : tensor<144xf32>}> : () -> tensor<144xf32>
    %953 = tosa.add %951, %952 : (tensor<144xf32>, tensor<144xf32>) -> tensor<144xf32>
    %954 = math.sqrt %953 : tensor<144xf32>
    %955 = tosa.reciprocal %954 : (tensor<144xf32>) -> tensor<144xf32>
    %956 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<144xf32>}> : () -> tensor<144xf32>
    %957 = tosa.mul %955, %956 {shift = 0 : i8} : (tensor<144xf32>, tensor<144xf32>) -> tensor<144xf32>
    %958 = tosa.reshape %950 {new_shape = array<i64: 144, 1>} : (tensor<144xf32>) -> tensor<144x1xf32>
    %959 = tosa.reshape %958 {new_shape = array<i64: 144, 1, 1>} : (tensor<144x1xf32>) -> tensor<144x1x1xf32>
    %960 = tosa.reshape %957 {new_shape = array<i64: 144, 1>} : (tensor<144xf32>) -> tensor<144x1xf32>
    %961 = tosa.reshape %960 {new_shape = array<i64: 144, 1, 1>} : (tensor<144x1xf32>) -> tensor<144x1x1xf32>
    %962 = tosa.reshape %959 {new_shape = array<i64: 1, 144, 1, 1>} : (tensor<144x1x1xf32>) -> tensor<1x144x1x1xf32>
    %963 = tosa.sub %949, %962 : (tensor<1x144x14x14xf32>, tensor<1x144x1x1xf32>) -> tensor<1x144x14x14xf32>
    %964 = tosa.reshape %961 {new_shape = array<i64: 1, 144, 1, 1>} : (tensor<144x1x1xf32>) -> tensor<1x144x1x1xf32>
    %965 = tosa.mul %963, %964 {shift = 0 : i8} : (tensor<1x144x14x14xf32>, tensor<1x144x1x1xf32>) -> tensor<1x144x14x14xf32>
    %966 = tosa.reshape %arg134 {new_shape = array<i64: 144, 1>} : (tensor<144xf32>) -> tensor<144x1xf32>
    %967 = tosa.reshape %966 {new_shape = array<i64: 144, 1, 1>} : (tensor<144x1xf32>) -> tensor<144x1x1xf32>
    %968 = tosa.reshape %967 {new_shape = array<i64: 1, 144, 1, 1>} : (tensor<144x1x1xf32>) -> tensor<1x144x1x1xf32>
    %969 = tosa.mul %965, %968 {shift = 0 : i8} : (tensor<1x144x14x14xf32>, tensor<1x144x1x1xf32>) -> tensor<1x144x14x14xf32>
    %970 = tosa.reshape %arg135 {new_shape = array<i64: 144, 1>} : (tensor<144xf32>) -> tensor<144x1xf32>
    %971 = tosa.reshape %970 {new_shape = array<i64: 144, 1, 1>} : (tensor<144x1xf32>) -> tensor<144x1x1xf32>
    %972 = tosa.reshape %971 {new_shape = array<i64: 1, 144, 1, 1>} : (tensor<144x1x1xf32>) -> tensor<1x144x1x1xf32>
    %973 = tosa.add %969, %972 : (tensor<1x144x14x14xf32>, tensor<1x144x1x1xf32>) -> tensor<1x144x14x14xf32>
    %974 = "tosa.const"() <{value = dense<3.000000e+00> : tensor<1x144x14x14xf32>}> : () -> tensor<1x144x14x14xf32>
    %975 = tosa.add %973, %974 : (tensor<1x144x14x14xf32>, tensor<1x144x14x14xf32>) -> tensor<1x144x14x14xf32>
    %976 = tosa.clamp %975 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x144x14x14xf32>) -> tensor<1x144x14x14xf32>
    %977 = tosa.clamp %976 {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0xFF800000 : f32, min_int = -9223372036854775807 : i64} : (tensor<1x144x14x14xf32>) -> tensor<1x144x14x14xf32>
    %978 = tosa.mul %973, %977 {shift = 0 : i8} : (tensor<1x144x14x14xf32>, tensor<1x144x14x14xf32>) -> tensor<1x144x14x14xf32>
    %979 = "tosa.const"() <{value = dense<6.000000e+00> : tensor<1x144x14x14xf32>}> : () -> tensor<1x144x14x14xf32>
    %980 = tosa.reciprocal %979 : (tensor<1x144x14x14xf32>) -> tensor<1x144x14x14xf32>
    %981 = tosa.mul %978, %980 {shift = 0 : i8} : (tensor<1x144x14x14xf32>, tensor<1x144x14x14xf32>) -> tensor<1x144x14x14xf32>
    %982 = tosa.reduce_sum %981 {axis = 3 : i32} : (tensor<1x144x14x14xf32>) -> tensor<1x144x14x1xf32>
    %983 = tosa.reduce_sum %982 {axis = 2 : i32} : (tensor<1x144x14x1xf32>) -> tensor<1x144x1x1xf32>
    %984 = "tosa.const"() <{value = dense<1.960000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %985 = tosa.reciprocal %984 : (tensor<1xf32>) -> tensor<1xf32>
    %986 = tosa.mul %985, %983 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x144x1x1xf32>) -> tensor<1x144x1x1xf32>
    %987 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %988 = tosa.transpose %986, %987 : (tensor<1x144x1x1xf32>, tensor<4xi32>) -> tensor<1x1x1x144xf32>
    %989 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %990 = tosa.transpose %arg136, %989 : (tensor<40x144x1x1xf32>, tensor<4xi32>) -> tensor<40x1x1x144xf32>
    %991 = tosa.conv2d %988, %990, %arg137 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x1x1x144xf32>, tensor<40x1x1x144xf32>, tensor<40xf32>) -> tensor<1x1x1x40xf32>
    %992 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %993 = tosa.transpose %991, %992 : (tensor<1x1x1x40xf32>, tensor<4xi32>) -> tensor<1x40x1x1xf32>
    %994 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x40x1x1xf32>}> : () -> tensor<1x40x1x1xf32>
    %995 = tosa.maximum %993, %994 : (tensor<1x40x1x1xf32>, tensor<1x40x1x1xf32>) -> tensor<1x40x1x1xf32>
    %996 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %997 = tosa.transpose %995, %996 : (tensor<1x40x1x1xf32>, tensor<4xi32>) -> tensor<1x1x1x40xf32>
    %998 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %999 = tosa.transpose %arg138, %998 : (tensor<144x40x1x1xf32>, tensor<4xi32>) -> tensor<144x1x1x40xf32>
    %1000 = tosa.conv2d %997, %999, %arg139 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x1x1x40xf32>, tensor<144x1x1x40xf32>, tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %1001 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1002 = tosa.transpose %1000, %1001 : (tensor<1x1x1x144xf32>, tensor<4xi32>) -> tensor<1x144x1x1xf32>
    %1003 = "tosa.const"() <{value = dense<3.000000e+00> : tensor<1x144x1x1xf32>}> : () -> tensor<1x144x1x1xf32>
    %1004 = tosa.add %1002, %1003 : (tensor<1x144x1x1xf32>, tensor<1x144x1x1xf32>) -> tensor<1x144x1x1xf32>
    %1005 = tosa.clamp %1004 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x144x1x1xf32>) -> tensor<1x144x1x1xf32>
    %1006 = tosa.clamp %1005 {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0xFF800000 : f32, min_int = -9223372036854775807 : i64} : (tensor<1x144x1x1xf32>) -> tensor<1x144x1x1xf32>
    %1007 = "tosa.const"() <{value = dense<6.000000e+00> : tensor<1x144x1x1xf32>}> : () -> tensor<1x144x1x1xf32>
    %1008 = tosa.reciprocal %1007 : (tensor<1x144x1x1xf32>) -> tensor<1x144x1x1xf32>
    %1009 = tosa.mul %1006, %1008 {shift = 0 : i8} : (tensor<1x144x1x1xf32>, tensor<1x144x1x1xf32>) -> tensor<1x144x1x1xf32>
    %1010 = tosa.mul %1009, %981 {shift = 0 : i8} : (tensor<1x144x1x1xf32>, tensor<1x144x14x14xf32>) -> tensor<1x144x14x14xf32>
    %1011 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<48xf32>}> : () -> tensor<48xf32>
    %1012 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1013 = tosa.transpose %1010, %1012 : (tensor<1x144x14x14xf32>, tensor<4xi32>) -> tensor<1x14x14x144xf32>
    %1014 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1015 = tosa.transpose %arg140, %1014 : (tensor<48x144x1x1xf32>, tensor<4xi32>) -> tensor<48x1x1x144xf32>
    %1016 = tosa.conv2d %1013, %1015, %1011 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x14x14x144xf32>, tensor<48x1x1x144xf32>, tensor<48xf32>) -> tensor<1x14x14x48xf32>
    %1017 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1018 = tosa.transpose %1016, %1017 : (tensor<1x14x14x48xf32>, tensor<4xi32>) -> tensor<1x48x14x14xf32>
    %1019 = tosa.cast %arg141 : (tensor<48xf32>) -> tensor<48xf32>
    %1020 = tosa.cast %arg142 : (tensor<48xf32>) -> tensor<48xf32>
    %1021 = "tosa.const"() <{value = dense<1.000000e-03> : tensor<48xf32>}> : () -> tensor<48xf32>
    %1022 = tosa.add %1020, %1021 : (tensor<48xf32>, tensor<48xf32>) -> tensor<48xf32>
    %1023 = math.sqrt %1022 : tensor<48xf32>
    %1024 = tosa.reciprocal %1023 : (tensor<48xf32>) -> tensor<48xf32>
    %1025 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<48xf32>}> : () -> tensor<48xf32>
    %1026 = tosa.mul %1024, %1025 {shift = 0 : i8} : (tensor<48xf32>, tensor<48xf32>) -> tensor<48xf32>
    %1027 = tosa.reshape %1019 {new_shape = array<i64: 48, 1>} : (tensor<48xf32>) -> tensor<48x1xf32>
    %1028 = tosa.reshape %1027 {new_shape = array<i64: 48, 1, 1>} : (tensor<48x1xf32>) -> tensor<48x1x1xf32>
    %1029 = tosa.reshape %1026 {new_shape = array<i64: 48, 1>} : (tensor<48xf32>) -> tensor<48x1xf32>
    %1030 = tosa.reshape %1029 {new_shape = array<i64: 48, 1, 1>} : (tensor<48x1xf32>) -> tensor<48x1x1xf32>
    %1031 = tosa.reshape %1028 {new_shape = array<i64: 1, 48, 1, 1>} : (tensor<48x1x1xf32>) -> tensor<1x48x1x1xf32>
    %1032 = tosa.sub %1018, %1031 : (tensor<1x48x14x14xf32>, tensor<1x48x1x1xf32>) -> tensor<1x48x14x14xf32>
    %1033 = tosa.reshape %1030 {new_shape = array<i64: 1, 48, 1, 1>} : (tensor<48x1x1xf32>) -> tensor<1x48x1x1xf32>
    %1034 = tosa.mul %1032, %1033 {shift = 0 : i8} : (tensor<1x48x14x14xf32>, tensor<1x48x1x1xf32>) -> tensor<1x48x14x14xf32>
    %1035 = tosa.reshape %arg143 {new_shape = array<i64: 48, 1>} : (tensor<48xf32>) -> tensor<48x1xf32>
    %1036 = tosa.reshape %1035 {new_shape = array<i64: 48, 1, 1>} : (tensor<48x1xf32>) -> tensor<48x1x1xf32>
    %1037 = tosa.reshape %1036 {new_shape = array<i64: 1, 48, 1, 1>} : (tensor<48x1x1xf32>) -> tensor<1x48x1x1xf32>
    %1038 = tosa.mul %1034, %1037 {shift = 0 : i8} : (tensor<1x48x14x14xf32>, tensor<1x48x1x1xf32>) -> tensor<1x48x14x14xf32>
    %1039 = tosa.reshape %arg144 {new_shape = array<i64: 48, 1>} : (tensor<48xf32>) -> tensor<48x1xf32>
    %1040 = tosa.reshape %1039 {new_shape = array<i64: 48, 1, 1>} : (tensor<48x1xf32>) -> tensor<48x1x1xf32>
    %1041 = tosa.reshape %1040 {new_shape = array<i64: 1, 48, 1, 1>} : (tensor<48x1x1xf32>) -> tensor<1x48x1x1xf32>
    %1042 = tosa.add %1038, %1041 : (tensor<1x48x14x14xf32>, tensor<1x48x1x1xf32>) -> tensor<1x48x14x14xf32>
    %1043 = tosa.add %1042, %901 : (tensor<1x48x14x14xf32>, tensor<1x48x14x14xf32>) -> tensor<1x48x14x14xf32>
    %1044 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<288xf32>}> : () -> tensor<288xf32>
    %1045 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1046 = tosa.transpose %1043, %1045 : (tensor<1x48x14x14xf32>, tensor<4xi32>) -> tensor<1x14x14x48xf32>
    %1047 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1048 = tosa.transpose %arg145, %1047 : (tensor<288x48x1x1xf32>, tensor<4xi32>) -> tensor<288x1x1x48xf32>
    %1049 = tosa.conv2d %1046, %1048, %1044 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x14x14x48xf32>, tensor<288x1x1x48xf32>, tensor<288xf32>) -> tensor<1x14x14x288xf32>
    %1050 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1051 = tosa.transpose %1049, %1050 : (tensor<1x14x14x288xf32>, tensor<4xi32>) -> tensor<1x288x14x14xf32>
    %1052 = tosa.cast %arg146 : (tensor<288xf32>) -> tensor<288xf32>
    %1053 = tosa.cast %arg147 : (tensor<288xf32>) -> tensor<288xf32>
    %1054 = "tosa.const"() <{value = dense<1.000000e-03> : tensor<288xf32>}> : () -> tensor<288xf32>
    %1055 = tosa.add %1053, %1054 : (tensor<288xf32>, tensor<288xf32>) -> tensor<288xf32>
    %1056 = math.sqrt %1055 : tensor<288xf32>
    %1057 = tosa.reciprocal %1056 : (tensor<288xf32>) -> tensor<288xf32>
    %1058 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<288xf32>}> : () -> tensor<288xf32>
    %1059 = tosa.mul %1057, %1058 {shift = 0 : i8} : (tensor<288xf32>, tensor<288xf32>) -> tensor<288xf32>
    %1060 = tosa.reshape %1052 {new_shape = array<i64: 288, 1>} : (tensor<288xf32>) -> tensor<288x1xf32>
    %1061 = tosa.reshape %1060 {new_shape = array<i64: 288, 1, 1>} : (tensor<288x1xf32>) -> tensor<288x1x1xf32>
    %1062 = tosa.reshape %1059 {new_shape = array<i64: 288, 1>} : (tensor<288xf32>) -> tensor<288x1xf32>
    %1063 = tosa.reshape %1062 {new_shape = array<i64: 288, 1, 1>} : (tensor<288x1xf32>) -> tensor<288x1x1xf32>
    %1064 = tosa.reshape %1061 {new_shape = array<i64: 1, 288, 1, 1>} : (tensor<288x1x1xf32>) -> tensor<1x288x1x1xf32>
    %1065 = tosa.sub %1051, %1064 : (tensor<1x288x14x14xf32>, tensor<1x288x1x1xf32>) -> tensor<1x288x14x14xf32>
    %1066 = tosa.reshape %1063 {new_shape = array<i64: 1, 288, 1, 1>} : (tensor<288x1x1xf32>) -> tensor<1x288x1x1xf32>
    %1067 = tosa.mul %1065, %1066 {shift = 0 : i8} : (tensor<1x288x14x14xf32>, tensor<1x288x1x1xf32>) -> tensor<1x288x14x14xf32>
    %1068 = tosa.reshape %arg148 {new_shape = array<i64: 288, 1>} : (tensor<288xf32>) -> tensor<288x1xf32>
    %1069 = tosa.reshape %1068 {new_shape = array<i64: 288, 1, 1>} : (tensor<288x1xf32>) -> tensor<288x1x1xf32>
    %1070 = tosa.reshape %1069 {new_shape = array<i64: 1, 288, 1, 1>} : (tensor<288x1x1xf32>) -> tensor<1x288x1x1xf32>
    %1071 = tosa.mul %1067, %1070 {shift = 0 : i8} : (tensor<1x288x14x14xf32>, tensor<1x288x1x1xf32>) -> tensor<1x288x14x14xf32>
    %1072 = tosa.reshape %arg149 {new_shape = array<i64: 288, 1>} : (tensor<288xf32>) -> tensor<288x1xf32>
    %1073 = tosa.reshape %1072 {new_shape = array<i64: 288, 1, 1>} : (tensor<288x1xf32>) -> tensor<288x1x1xf32>
    %1074 = tosa.reshape %1073 {new_shape = array<i64: 1, 288, 1, 1>} : (tensor<288x1x1xf32>) -> tensor<1x288x1x1xf32>
    %1075 = tosa.add %1071, %1074 : (tensor<1x288x14x14xf32>, tensor<1x288x1x1xf32>) -> tensor<1x288x14x14xf32>
    %1076 = "tosa.const"() <{value = dense<3.000000e+00> : tensor<1x288x14x14xf32>}> : () -> tensor<1x288x14x14xf32>
    %1077 = tosa.add %1075, %1076 : (tensor<1x288x14x14xf32>, tensor<1x288x14x14xf32>) -> tensor<1x288x14x14xf32>
    %1078 = tosa.clamp %1077 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x288x14x14xf32>) -> tensor<1x288x14x14xf32>
    %1079 = tosa.clamp %1078 {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0xFF800000 : f32, min_int = -9223372036854775807 : i64} : (tensor<1x288x14x14xf32>) -> tensor<1x288x14x14xf32>
    %1080 = tosa.mul %1075, %1079 {shift = 0 : i8} : (tensor<1x288x14x14xf32>, tensor<1x288x14x14xf32>) -> tensor<1x288x14x14xf32>
    %1081 = "tosa.const"() <{value = dense<6.000000e+00> : tensor<1x288x14x14xf32>}> : () -> tensor<1x288x14x14xf32>
    %1082 = tosa.reciprocal %1081 : (tensor<1x288x14x14xf32>) -> tensor<1x288x14x14xf32>
    %1083 = tosa.mul %1080, %1082 {shift = 0 : i8} : (tensor<1x288x14x14xf32>, tensor<1x288x14x14xf32>) -> tensor<1x288x14x14xf32>
    %1084 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<288xf32>}> : () -> tensor<288xf32>
    %1085 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1086 = tosa.transpose %1083, %1085 : (tensor<1x288x14x14xf32>, tensor<4xi32>) -> tensor<1x14x14x288xf32>
    %1087 = "tosa.const"() <{value = dense<[2, 3, 0, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1088 = tosa.transpose %arg150, %1087 : (tensor<288x1x5x5xf32>, tensor<4xi32>) -> tensor<5x5x288x1xf32>
    %1089 = tosa.depthwise_conv2d %1086, %1088, %1084 {dilation = array<i64: 1, 1>, pad = array<i64: 2, 2, 2, 2>, stride = array<i64: 2, 2>} : (tensor<1x14x14x288xf32>, tensor<5x5x288x1xf32>, tensor<288xf32>) -> tensor<1x7x7x288xf32>
    %1090 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1091 = tosa.transpose %1089, %1090 : (tensor<1x7x7x288xf32>, tensor<4xi32>) -> tensor<1x288x7x7xf32>
    %1092 = tosa.cast %arg151 : (tensor<288xf32>) -> tensor<288xf32>
    %1093 = tosa.cast %arg152 : (tensor<288xf32>) -> tensor<288xf32>
    %1094 = "tosa.const"() <{value = dense<1.000000e-03> : tensor<288xf32>}> : () -> tensor<288xf32>
    %1095 = tosa.add %1093, %1094 : (tensor<288xf32>, tensor<288xf32>) -> tensor<288xf32>
    %1096 = math.sqrt %1095 : tensor<288xf32>
    %1097 = tosa.reciprocal %1096 : (tensor<288xf32>) -> tensor<288xf32>
    %1098 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<288xf32>}> : () -> tensor<288xf32>
    %1099 = tosa.mul %1097, %1098 {shift = 0 : i8} : (tensor<288xf32>, tensor<288xf32>) -> tensor<288xf32>
    %1100 = tosa.reshape %1092 {new_shape = array<i64: 288, 1>} : (tensor<288xf32>) -> tensor<288x1xf32>
    %1101 = tosa.reshape %1100 {new_shape = array<i64: 288, 1, 1>} : (tensor<288x1xf32>) -> tensor<288x1x1xf32>
    %1102 = tosa.reshape %1099 {new_shape = array<i64: 288, 1>} : (tensor<288xf32>) -> tensor<288x1xf32>
    %1103 = tosa.reshape %1102 {new_shape = array<i64: 288, 1, 1>} : (tensor<288x1xf32>) -> tensor<288x1x1xf32>
    %1104 = tosa.reshape %1101 {new_shape = array<i64: 1, 288, 1, 1>} : (tensor<288x1x1xf32>) -> tensor<1x288x1x1xf32>
    %1105 = tosa.sub %1091, %1104 : (tensor<1x288x7x7xf32>, tensor<1x288x1x1xf32>) -> tensor<1x288x7x7xf32>
    %1106 = tosa.reshape %1103 {new_shape = array<i64: 1, 288, 1, 1>} : (tensor<288x1x1xf32>) -> tensor<1x288x1x1xf32>
    %1107 = tosa.mul %1105, %1106 {shift = 0 : i8} : (tensor<1x288x7x7xf32>, tensor<1x288x1x1xf32>) -> tensor<1x288x7x7xf32>
    %1108 = tosa.reshape %arg153 {new_shape = array<i64: 288, 1>} : (tensor<288xf32>) -> tensor<288x1xf32>
    %1109 = tosa.reshape %1108 {new_shape = array<i64: 288, 1, 1>} : (tensor<288x1xf32>) -> tensor<288x1x1xf32>
    %1110 = tosa.reshape %1109 {new_shape = array<i64: 1, 288, 1, 1>} : (tensor<288x1x1xf32>) -> tensor<1x288x1x1xf32>
    %1111 = tosa.mul %1107, %1110 {shift = 0 : i8} : (tensor<1x288x7x7xf32>, tensor<1x288x1x1xf32>) -> tensor<1x288x7x7xf32>
    %1112 = tosa.reshape %arg154 {new_shape = array<i64: 288, 1>} : (tensor<288xf32>) -> tensor<288x1xf32>
    %1113 = tosa.reshape %1112 {new_shape = array<i64: 288, 1, 1>} : (tensor<288x1xf32>) -> tensor<288x1x1xf32>
    %1114 = tosa.reshape %1113 {new_shape = array<i64: 1, 288, 1, 1>} : (tensor<288x1x1xf32>) -> tensor<1x288x1x1xf32>
    %1115 = tosa.add %1111, %1114 : (tensor<1x288x7x7xf32>, tensor<1x288x1x1xf32>) -> tensor<1x288x7x7xf32>
    %1116 = "tosa.const"() <{value = dense<3.000000e+00> : tensor<1x288x7x7xf32>}> : () -> tensor<1x288x7x7xf32>
    %1117 = tosa.add %1115, %1116 : (tensor<1x288x7x7xf32>, tensor<1x288x7x7xf32>) -> tensor<1x288x7x7xf32>
    %1118 = tosa.clamp %1117 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x288x7x7xf32>) -> tensor<1x288x7x7xf32>
    %1119 = tosa.clamp %1118 {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0xFF800000 : f32, min_int = -9223372036854775807 : i64} : (tensor<1x288x7x7xf32>) -> tensor<1x288x7x7xf32>
    %1120 = tosa.mul %1115, %1119 {shift = 0 : i8} : (tensor<1x288x7x7xf32>, tensor<1x288x7x7xf32>) -> tensor<1x288x7x7xf32>
    %1121 = "tosa.const"() <{value = dense<6.000000e+00> : tensor<1x288x7x7xf32>}> : () -> tensor<1x288x7x7xf32>
    %1122 = tosa.reciprocal %1121 : (tensor<1x288x7x7xf32>) -> tensor<1x288x7x7xf32>
    %1123 = tosa.mul %1120, %1122 {shift = 0 : i8} : (tensor<1x288x7x7xf32>, tensor<1x288x7x7xf32>) -> tensor<1x288x7x7xf32>
    %1124 = tosa.reduce_sum %1123 {axis = 3 : i32} : (tensor<1x288x7x7xf32>) -> tensor<1x288x7x1xf32>
    %1125 = tosa.reduce_sum %1124 {axis = 2 : i32} : (tensor<1x288x7x1xf32>) -> tensor<1x288x1x1xf32>
    %1126 = "tosa.const"() <{value = dense<4.900000e+01> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1127 = tosa.reciprocal %1126 : (tensor<1xf32>) -> tensor<1xf32>
    %1128 = tosa.mul %1127, %1125 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x288x1x1xf32>) -> tensor<1x288x1x1xf32>
    %1129 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1130 = tosa.transpose %1128, %1129 : (tensor<1x288x1x1xf32>, tensor<4xi32>) -> tensor<1x1x1x288xf32>
    %1131 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1132 = tosa.transpose %arg155, %1131 : (tensor<72x288x1x1xf32>, tensor<4xi32>) -> tensor<72x1x1x288xf32>
    %1133 = tosa.conv2d %1130, %1132, %arg156 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x1x1x288xf32>, tensor<72x1x1x288xf32>, tensor<72xf32>) -> tensor<1x1x1x72xf32>
    %1134 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1135 = tosa.transpose %1133, %1134 : (tensor<1x1x1x72xf32>, tensor<4xi32>) -> tensor<1x72x1x1xf32>
    %1136 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x72x1x1xf32>}> : () -> tensor<1x72x1x1xf32>
    %1137 = tosa.maximum %1135, %1136 : (tensor<1x72x1x1xf32>, tensor<1x72x1x1xf32>) -> tensor<1x72x1x1xf32>
    %1138 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1139 = tosa.transpose %1137, %1138 : (tensor<1x72x1x1xf32>, tensor<4xi32>) -> tensor<1x1x1x72xf32>
    %1140 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1141 = tosa.transpose %arg157, %1140 : (tensor<288x72x1x1xf32>, tensor<4xi32>) -> tensor<288x1x1x72xf32>
    %1142 = tosa.conv2d %1139, %1141, %arg158 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x1x1x72xf32>, tensor<288x1x1x72xf32>, tensor<288xf32>) -> tensor<1x1x1x288xf32>
    %1143 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1144 = tosa.transpose %1142, %1143 : (tensor<1x1x1x288xf32>, tensor<4xi32>) -> tensor<1x288x1x1xf32>
    %1145 = "tosa.const"() <{value = dense<3.000000e+00> : tensor<1x288x1x1xf32>}> : () -> tensor<1x288x1x1xf32>
    %1146 = tosa.add %1144, %1145 : (tensor<1x288x1x1xf32>, tensor<1x288x1x1xf32>) -> tensor<1x288x1x1xf32>
    %1147 = tosa.clamp %1146 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x288x1x1xf32>) -> tensor<1x288x1x1xf32>
    %1148 = tosa.clamp %1147 {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0xFF800000 : f32, min_int = -9223372036854775807 : i64} : (tensor<1x288x1x1xf32>) -> tensor<1x288x1x1xf32>
    %1149 = "tosa.const"() <{value = dense<6.000000e+00> : tensor<1x288x1x1xf32>}> : () -> tensor<1x288x1x1xf32>
    %1150 = tosa.reciprocal %1149 : (tensor<1x288x1x1xf32>) -> tensor<1x288x1x1xf32>
    %1151 = tosa.mul %1148, %1150 {shift = 0 : i8} : (tensor<1x288x1x1xf32>, tensor<1x288x1x1xf32>) -> tensor<1x288x1x1xf32>
    %1152 = tosa.mul %1151, %1123 {shift = 0 : i8} : (tensor<1x288x1x1xf32>, tensor<1x288x7x7xf32>) -> tensor<1x288x7x7xf32>
    %1153 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<96xf32>}> : () -> tensor<96xf32>
    %1154 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1155 = tosa.transpose %1152, %1154 : (tensor<1x288x7x7xf32>, tensor<4xi32>) -> tensor<1x7x7x288xf32>
    %1156 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1157 = tosa.transpose %arg159, %1156 : (tensor<96x288x1x1xf32>, tensor<4xi32>) -> tensor<96x1x1x288xf32>
    %1158 = tosa.conv2d %1155, %1157, %1153 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x7x7x288xf32>, tensor<96x1x1x288xf32>, tensor<96xf32>) -> tensor<1x7x7x96xf32>
    %1159 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1160 = tosa.transpose %1158, %1159 : (tensor<1x7x7x96xf32>, tensor<4xi32>) -> tensor<1x96x7x7xf32>
    %1161 = tosa.cast %arg160 : (tensor<96xf32>) -> tensor<96xf32>
    %1162 = tosa.cast %arg161 : (tensor<96xf32>) -> tensor<96xf32>
    %1163 = "tosa.const"() <{value = dense<1.000000e-03> : tensor<96xf32>}> : () -> tensor<96xf32>
    %1164 = tosa.add %1162, %1163 : (tensor<96xf32>, tensor<96xf32>) -> tensor<96xf32>
    %1165 = math.sqrt %1164 : tensor<96xf32>
    %1166 = tosa.reciprocal %1165 : (tensor<96xf32>) -> tensor<96xf32>
    %1167 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<96xf32>}> : () -> tensor<96xf32>
    %1168 = tosa.mul %1166, %1167 {shift = 0 : i8} : (tensor<96xf32>, tensor<96xf32>) -> tensor<96xf32>
    %1169 = tosa.reshape %1161 {new_shape = array<i64: 96, 1>} : (tensor<96xf32>) -> tensor<96x1xf32>
    %1170 = tosa.reshape %1169 {new_shape = array<i64: 96, 1, 1>} : (tensor<96x1xf32>) -> tensor<96x1x1xf32>
    %1171 = tosa.reshape %1168 {new_shape = array<i64: 96, 1>} : (tensor<96xf32>) -> tensor<96x1xf32>
    %1172 = tosa.reshape %1171 {new_shape = array<i64: 96, 1, 1>} : (tensor<96x1xf32>) -> tensor<96x1x1xf32>
    %1173 = tosa.reshape %1170 {new_shape = array<i64: 1, 96, 1, 1>} : (tensor<96x1x1xf32>) -> tensor<1x96x1x1xf32>
    %1174 = tosa.sub %1160, %1173 : (tensor<1x96x7x7xf32>, tensor<1x96x1x1xf32>) -> tensor<1x96x7x7xf32>
    %1175 = tosa.reshape %1172 {new_shape = array<i64: 1, 96, 1, 1>} : (tensor<96x1x1xf32>) -> tensor<1x96x1x1xf32>
    %1176 = tosa.mul %1174, %1175 {shift = 0 : i8} : (tensor<1x96x7x7xf32>, tensor<1x96x1x1xf32>) -> tensor<1x96x7x7xf32>
    %1177 = tosa.reshape %arg162 {new_shape = array<i64: 96, 1>} : (tensor<96xf32>) -> tensor<96x1xf32>
    %1178 = tosa.reshape %1177 {new_shape = array<i64: 96, 1, 1>} : (tensor<96x1xf32>) -> tensor<96x1x1xf32>
    %1179 = tosa.reshape %1178 {new_shape = array<i64: 1, 96, 1, 1>} : (tensor<96x1x1xf32>) -> tensor<1x96x1x1xf32>
    %1180 = tosa.mul %1176, %1179 {shift = 0 : i8} : (tensor<1x96x7x7xf32>, tensor<1x96x1x1xf32>) -> tensor<1x96x7x7xf32>
    %1181 = tosa.reshape %arg163 {new_shape = array<i64: 96, 1>} : (tensor<96xf32>) -> tensor<96x1xf32>
    %1182 = tosa.reshape %1181 {new_shape = array<i64: 96, 1, 1>} : (tensor<96x1xf32>) -> tensor<96x1x1xf32>
    %1183 = tosa.reshape %1182 {new_shape = array<i64: 1, 96, 1, 1>} : (tensor<96x1x1xf32>) -> tensor<1x96x1x1xf32>
    %1184 = tosa.add %1180, %1183 : (tensor<1x96x7x7xf32>, tensor<1x96x1x1xf32>) -> tensor<1x96x7x7xf32>
    %1185 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<576xf32>}> : () -> tensor<576xf32>
    %1186 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1187 = tosa.transpose %1184, %1186 : (tensor<1x96x7x7xf32>, tensor<4xi32>) -> tensor<1x7x7x96xf32>
    %1188 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1189 = tosa.transpose %arg164, %1188 : (tensor<576x96x1x1xf32>, tensor<4xi32>) -> tensor<576x1x1x96xf32>
    %1190 = tosa.conv2d %1187, %1189, %1185 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x7x7x96xf32>, tensor<576x1x1x96xf32>, tensor<576xf32>) -> tensor<1x7x7x576xf32>
    %1191 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1192 = tosa.transpose %1190, %1191 : (tensor<1x7x7x576xf32>, tensor<4xi32>) -> tensor<1x576x7x7xf32>
    %1193 = tosa.cast %arg165 : (tensor<576xf32>) -> tensor<576xf32>
    %1194 = tosa.cast %arg166 : (tensor<576xf32>) -> tensor<576xf32>
    %1195 = "tosa.const"() <{value = dense<1.000000e-03> : tensor<576xf32>}> : () -> tensor<576xf32>
    %1196 = tosa.add %1194, %1195 : (tensor<576xf32>, tensor<576xf32>) -> tensor<576xf32>
    %1197 = math.sqrt %1196 : tensor<576xf32>
    %1198 = tosa.reciprocal %1197 : (tensor<576xf32>) -> tensor<576xf32>
    %1199 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<576xf32>}> : () -> tensor<576xf32>
    %1200 = tosa.mul %1198, %1199 {shift = 0 : i8} : (tensor<576xf32>, tensor<576xf32>) -> tensor<576xf32>
    %1201 = tosa.reshape %1193 {new_shape = array<i64: 576, 1>} : (tensor<576xf32>) -> tensor<576x1xf32>
    %1202 = tosa.reshape %1201 {new_shape = array<i64: 576, 1, 1>} : (tensor<576x1xf32>) -> tensor<576x1x1xf32>
    %1203 = tosa.reshape %1200 {new_shape = array<i64: 576, 1>} : (tensor<576xf32>) -> tensor<576x1xf32>
    %1204 = tosa.reshape %1203 {new_shape = array<i64: 576, 1, 1>} : (tensor<576x1xf32>) -> tensor<576x1x1xf32>
    %1205 = tosa.reshape %1202 {new_shape = array<i64: 1, 576, 1, 1>} : (tensor<576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %1206 = tosa.sub %1192, %1205 : (tensor<1x576x7x7xf32>, tensor<1x576x1x1xf32>) -> tensor<1x576x7x7xf32>
    %1207 = tosa.reshape %1204 {new_shape = array<i64: 1, 576, 1, 1>} : (tensor<576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %1208 = tosa.mul %1206, %1207 {shift = 0 : i8} : (tensor<1x576x7x7xf32>, tensor<1x576x1x1xf32>) -> tensor<1x576x7x7xf32>
    %1209 = tosa.reshape %arg167 {new_shape = array<i64: 576, 1>} : (tensor<576xf32>) -> tensor<576x1xf32>
    %1210 = tosa.reshape %1209 {new_shape = array<i64: 576, 1, 1>} : (tensor<576x1xf32>) -> tensor<576x1x1xf32>
    %1211 = tosa.reshape %1210 {new_shape = array<i64: 1, 576, 1, 1>} : (tensor<576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %1212 = tosa.mul %1208, %1211 {shift = 0 : i8} : (tensor<1x576x7x7xf32>, tensor<1x576x1x1xf32>) -> tensor<1x576x7x7xf32>
    %1213 = tosa.reshape %arg168 {new_shape = array<i64: 576, 1>} : (tensor<576xf32>) -> tensor<576x1xf32>
    %1214 = tosa.reshape %1213 {new_shape = array<i64: 576, 1, 1>} : (tensor<576x1xf32>) -> tensor<576x1x1xf32>
    %1215 = tosa.reshape %1214 {new_shape = array<i64: 1, 576, 1, 1>} : (tensor<576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %1216 = tosa.add %1212, %1215 : (tensor<1x576x7x7xf32>, tensor<1x576x1x1xf32>) -> tensor<1x576x7x7xf32>
    %1217 = "tosa.const"() <{value = dense<3.000000e+00> : tensor<1x576x7x7xf32>}> : () -> tensor<1x576x7x7xf32>
    %1218 = tosa.add %1216, %1217 : (tensor<1x576x7x7xf32>, tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %1219 = tosa.clamp %1218 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %1220 = tosa.clamp %1219 {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0xFF800000 : f32, min_int = -9223372036854775807 : i64} : (tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %1221 = tosa.mul %1216, %1220 {shift = 0 : i8} : (tensor<1x576x7x7xf32>, tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %1222 = "tosa.const"() <{value = dense<6.000000e+00> : tensor<1x576x7x7xf32>}> : () -> tensor<1x576x7x7xf32>
    %1223 = tosa.reciprocal %1222 : (tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %1224 = tosa.mul %1221, %1223 {shift = 0 : i8} : (tensor<1x576x7x7xf32>, tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %1225 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<576xf32>}> : () -> tensor<576xf32>
    %1226 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1227 = tosa.transpose %1224, %1226 : (tensor<1x576x7x7xf32>, tensor<4xi32>) -> tensor<1x7x7x576xf32>
    %1228 = "tosa.const"() <{value = dense<[2, 3, 0, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1229 = tosa.transpose %arg169, %1228 : (tensor<576x1x5x5xf32>, tensor<4xi32>) -> tensor<5x5x576x1xf32>
    %1230 = tosa.depthwise_conv2d %1227, %1229, %1225 {dilation = array<i64: 1, 1>, pad = array<i64: 2, 2, 2, 2>, stride = array<i64: 1, 1>} : (tensor<1x7x7x576xf32>, tensor<5x5x576x1xf32>, tensor<576xf32>) -> tensor<1x7x7x576xf32>
    %1231 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1232 = tosa.transpose %1230, %1231 : (tensor<1x7x7x576xf32>, tensor<4xi32>) -> tensor<1x576x7x7xf32>
    %1233 = tosa.cast %arg170 : (tensor<576xf32>) -> tensor<576xf32>
    %1234 = tosa.cast %arg171 : (tensor<576xf32>) -> tensor<576xf32>
    %1235 = "tosa.const"() <{value = dense<1.000000e-03> : tensor<576xf32>}> : () -> tensor<576xf32>
    %1236 = tosa.add %1234, %1235 : (tensor<576xf32>, tensor<576xf32>) -> tensor<576xf32>
    %1237 = math.sqrt %1236 : tensor<576xf32>
    %1238 = tosa.reciprocal %1237 : (tensor<576xf32>) -> tensor<576xf32>
    %1239 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<576xf32>}> : () -> tensor<576xf32>
    %1240 = tosa.mul %1238, %1239 {shift = 0 : i8} : (tensor<576xf32>, tensor<576xf32>) -> tensor<576xf32>
    %1241 = tosa.reshape %1233 {new_shape = array<i64: 576, 1>} : (tensor<576xf32>) -> tensor<576x1xf32>
    %1242 = tosa.reshape %1241 {new_shape = array<i64: 576, 1, 1>} : (tensor<576x1xf32>) -> tensor<576x1x1xf32>
    %1243 = tosa.reshape %1240 {new_shape = array<i64: 576, 1>} : (tensor<576xf32>) -> tensor<576x1xf32>
    %1244 = tosa.reshape %1243 {new_shape = array<i64: 576, 1, 1>} : (tensor<576x1xf32>) -> tensor<576x1x1xf32>
    %1245 = tosa.reshape %1242 {new_shape = array<i64: 1, 576, 1, 1>} : (tensor<576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %1246 = tosa.sub %1232, %1245 : (tensor<1x576x7x7xf32>, tensor<1x576x1x1xf32>) -> tensor<1x576x7x7xf32>
    %1247 = tosa.reshape %1244 {new_shape = array<i64: 1, 576, 1, 1>} : (tensor<576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %1248 = tosa.mul %1246, %1247 {shift = 0 : i8} : (tensor<1x576x7x7xf32>, tensor<1x576x1x1xf32>) -> tensor<1x576x7x7xf32>
    %1249 = tosa.reshape %arg172 {new_shape = array<i64: 576, 1>} : (tensor<576xf32>) -> tensor<576x1xf32>
    %1250 = tosa.reshape %1249 {new_shape = array<i64: 576, 1, 1>} : (tensor<576x1xf32>) -> tensor<576x1x1xf32>
    %1251 = tosa.reshape %1250 {new_shape = array<i64: 1, 576, 1, 1>} : (tensor<576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %1252 = tosa.mul %1248, %1251 {shift = 0 : i8} : (tensor<1x576x7x7xf32>, tensor<1x576x1x1xf32>) -> tensor<1x576x7x7xf32>
    %1253 = tosa.reshape %arg173 {new_shape = array<i64: 576, 1>} : (tensor<576xf32>) -> tensor<576x1xf32>
    %1254 = tosa.reshape %1253 {new_shape = array<i64: 576, 1, 1>} : (tensor<576x1xf32>) -> tensor<576x1x1xf32>
    %1255 = tosa.reshape %1254 {new_shape = array<i64: 1, 576, 1, 1>} : (tensor<576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %1256 = tosa.add %1252, %1255 : (tensor<1x576x7x7xf32>, tensor<1x576x1x1xf32>) -> tensor<1x576x7x7xf32>
    %1257 = "tosa.const"() <{value = dense<3.000000e+00> : tensor<1x576x7x7xf32>}> : () -> tensor<1x576x7x7xf32>
    %1258 = tosa.add %1256, %1257 : (tensor<1x576x7x7xf32>, tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %1259 = tosa.clamp %1258 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %1260 = tosa.clamp %1259 {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0xFF800000 : f32, min_int = -9223372036854775807 : i64} : (tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %1261 = tosa.mul %1256, %1260 {shift = 0 : i8} : (tensor<1x576x7x7xf32>, tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %1262 = "tosa.const"() <{value = dense<6.000000e+00> : tensor<1x576x7x7xf32>}> : () -> tensor<1x576x7x7xf32>
    %1263 = tosa.reciprocal %1262 : (tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %1264 = tosa.mul %1261, %1263 {shift = 0 : i8} : (tensor<1x576x7x7xf32>, tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %1265 = tosa.reduce_sum %1264 {axis = 3 : i32} : (tensor<1x576x7x7xf32>) -> tensor<1x576x7x1xf32>
    %1266 = tosa.reduce_sum %1265 {axis = 2 : i32} : (tensor<1x576x7x1xf32>) -> tensor<1x576x1x1xf32>
    %1267 = "tosa.const"() <{value = dense<4.900000e+01> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1268 = tosa.reciprocal %1267 : (tensor<1xf32>) -> tensor<1xf32>
    %1269 = tosa.mul %1268, %1266 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %1270 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1271 = tosa.transpose %1269, %1270 : (tensor<1x576x1x1xf32>, tensor<4xi32>) -> tensor<1x1x1x576xf32>
    %1272 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1273 = tosa.transpose %arg174, %1272 : (tensor<144x576x1x1xf32>, tensor<4xi32>) -> tensor<144x1x1x576xf32>
    %1274 = tosa.conv2d %1271, %1273, %arg175 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x1x1x576xf32>, tensor<144x1x1x576xf32>, tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %1275 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1276 = tosa.transpose %1274, %1275 : (tensor<1x1x1x144xf32>, tensor<4xi32>) -> tensor<1x144x1x1xf32>
    %1277 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x144x1x1xf32>}> : () -> tensor<1x144x1x1xf32>
    %1278 = tosa.maximum %1276, %1277 : (tensor<1x144x1x1xf32>, tensor<1x144x1x1xf32>) -> tensor<1x144x1x1xf32>
    %1279 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1280 = tosa.transpose %1278, %1279 : (tensor<1x144x1x1xf32>, tensor<4xi32>) -> tensor<1x1x1x144xf32>
    %1281 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1282 = tosa.transpose %arg176, %1281 : (tensor<576x144x1x1xf32>, tensor<4xi32>) -> tensor<576x1x1x144xf32>
    %1283 = tosa.conv2d %1280, %1282, %arg177 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x1x1x144xf32>, tensor<576x1x1x144xf32>, tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %1284 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1285 = tosa.transpose %1283, %1284 : (tensor<1x1x1x576xf32>, tensor<4xi32>) -> tensor<1x576x1x1xf32>
    %1286 = "tosa.const"() <{value = dense<3.000000e+00> : tensor<1x576x1x1xf32>}> : () -> tensor<1x576x1x1xf32>
    %1287 = tosa.add %1285, %1286 : (tensor<1x576x1x1xf32>, tensor<1x576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %1288 = tosa.clamp %1287 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %1289 = tosa.clamp %1288 {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0xFF800000 : f32, min_int = -9223372036854775807 : i64} : (tensor<1x576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %1290 = "tosa.const"() <{value = dense<6.000000e+00> : tensor<1x576x1x1xf32>}> : () -> tensor<1x576x1x1xf32>
    %1291 = tosa.reciprocal %1290 : (tensor<1x576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %1292 = tosa.mul %1289, %1291 {shift = 0 : i8} : (tensor<1x576x1x1xf32>, tensor<1x576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %1293 = tosa.mul %1292, %1264 {shift = 0 : i8} : (tensor<1x576x1x1xf32>, tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %1294 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<96xf32>}> : () -> tensor<96xf32>
    %1295 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1296 = tosa.transpose %1293, %1295 : (tensor<1x576x7x7xf32>, tensor<4xi32>) -> tensor<1x7x7x576xf32>
    %1297 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1298 = tosa.transpose %arg178, %1297 : (tensor<96x576x1x1xf32>, tensor<4xi32>) -> tensor<96x1x1x576xf32>
    %1299 = tosa.conv2d %1296, %1298, %1294 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x7x7x576xf32>, tensor<96x1x1x576xf32>, tensor<96xf32>) -> tensor<1x7x7x96xf32>
    %1300 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1301 = tosa.transpose %1299, %1300 : (tensor<1x7x7x96xf32>, tensor<4xi32>) -> tensor<1x96x7x7xf32>
    %1302 = tosa.cast %arg179 : (tensor<96xf32>) -> tensor<96xf32>
    %1303 = tosa.cast %arg180 : (tensor<96xf32>) -> tensor<96xf32>
    %1304 = "tosa.const"() <{value = dense<1.000000e-03> : tensor<96xf32>}> : () -> tensor<96xf32>
    %1305 = tosa.add %1303, %1304 : (tensor<96xf32>, tensor<96xf32>) -> tensor<96xf32>
    %1306 = math.sqrt %1305 : tensor<96xf32>
    %1307 = tosa.reciprocal %1306 : (tensor<96xf32>) -> tensor<96xf32>
    %1308 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<96xf32>}> : () -> tensor<96xf32>
    %1309 = tosa.mul %1307, %1308 {shift = 0 : i8} : (tensor<96xf32>, tensor<96xf32>) -> tensor<96xf32>
    %1310 = tosa.reshape %1302 {new_shape = array<i64: 96, 1>} : (tensor<96xf32>) -> tensor<96x1xf32>
    %1311 = tosa.reshape %1310 {new_shape = array<i64: 96, 1, 1>} : (tensor<96x1xf32>) -> tensor<96x1x1xf32>
    %1312 = tosa.reshape %1309 {new_shape = array<i64: 96, 1>} : (tensor<96xf32>) -> tensor<96x1xf32>
    %1313 = tosa.reshape %1312 {new_shape = array<i64: 96, 1, 1>} : (tensor<96x1xf32>) -> tensor<96x1x1xf32>
    %1314 = tosa.reshape %1311 {new_shape = array<i64: 1, 96, 1, 1>} : (tensor<96x1x1xf32>) -> tensor<1x96x1x1xf32>
    %1315 = tosa.sub %1301, %1314 : (tensor<1x96x7x7xf32>, tensor<1x96x1x1xf32>) -> tensor<1x96x7x7xf32>
    %1316 = tosa.reshape %1313 {new_shape = array<i64: 1, 96, 1, 1>} : (tensor<96x1x1xf32>) -> tensor<1x96x1x1xf32>
    %1317 = tosa.mul %1315, %1316 {shift = 0 : i8} : (tensor<1x96x7x7xf32>, tensor<1x96x1x1xf32>) -> tensor<1x96x7x7xf32>
    %1318 = tosa.reshape %arg181 {new_shape = array<i64: 96, 1>} : (tensor<96xf32>) -> tensor<96x1xf32>
    %1319 = tosa.reshape %1318 {new_shape = array<i64: 96, 1, 1>} : (tensor<96x1xf32>) -> tensor<96x1x1xf32>
    %1320 = tosa.reshape %1319 {new_shape = array<i64: 1, 96, 1, 1>} : (tensor<96x1x1xf32>) -> tensor<1x96x1x1xf32>
    %1321 = tosa.mul %1317, %1320 {shift = 0 : i8} : (tensor<1x96x7x7xf32>, tensor<1x96x1x1xf32>) -> tensor<1x96x7x7xf32>
    %1322 = tosa.reshape %arg182 {new_shape = array<i64: 96, 1>} : (tensor<96xf32>) -> tensor<96x1xf32>
    %1323 = tosa.reshape %1322 {new_shape = array<i64: 96, 1, 1>} : (tensor<96x1xf32>) -> tensor<96x1x1xf32>
    %1324 = tosa.reshape %1323 {new_shape = array<i64: 1, 96, 1, 1>} : (tensor<96x1x1xf32>) -> tensor<1x96x1x1xf32>
    %1325 = tosa.add %1321, %1324 : (tensor<1x96x7x7xf32>, tensor<1x96x1x1xf32>) -> tensor<1x96x7x7xf32>
    %1326 = tosa.add %1325, %1184 : (tensor<1x96x7x7xf32>, tensor<1x96x7x7xf32>) -> tensor<1x96x7x7xf32>
    %1327 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<576xf32>}> : () -> tensor<576xf32>
    %1328 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1329 = tosa.transpose %1326, %1328 : (tensor<1x96x7x7xf32>, tensor<4xi32>) -> tensor<1x7x7x96xf32>
    %1330 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1331 = tosa.transpose %arg183, %1330 : (tensor<576x96x1x1xf32>, tensor<4xi32>) -> tensor<576x1x1x96xf32>
    %1332 = tosa.conv2d %1329, %1331, %1327 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x7x7x96xf32>, tensor<576x1x1x96xf32>, tensor<576xf32>) -> tensor<1x7x7x576xf32>
    %1333 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1334 = tosa.transpose %1332, %1333 : (tensor<1x7x7x576xf32>, tensor<4xi32>) -> tensor<1x576x7x7xf32>
    %1335 = tosa.cast %arg184 : (tensor<576xf32>) -> tensor<576xf32>
    %1336 = tosa.cast %arg185 : (tensor<576xf32>) -> tensor<576xf32>
    %1337 = "tosa.const"() <{value = dense<1.000000e-03> : tensor<576xf32>}> : () -> tensor<576xf32>
    %1338 = tosa.add %1336, %1337 : (tensor<576xf32>, tensor<576xf32>) -> tensor<576xf32>
    %1339 = math.sqrt %1338 : tensor<576xf32>
    %1340 = tosa.reciprocal %1339 : (tensor<576xf32>) -> tensor<576xf32>
    %1341 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<576xf32>}> : () -> tensor<576xf32>
    %1342 = tosa.mul %1340, %1341 {shift = 0 : i8} : (tensor<576xf32>, tensor<576xf32>) -> tensor<576xf32>
    %1343 = tosa.reshape %1335 {new_shape = array<i64: 576, 1>} : (tensor<576xf32>) -> tensor<576x1xf32>
    %1344 = tosa.reshape %1343 {new_shape = array<i64: 576, 1, 1>} : (tensor<576x1xf32>) -> tensor<576x1x1xf32>
    %1345 = tosa.reshape %1342 {new_shape = array<i64: 576, 1>} : (tensor<576xf32>) -> tensor<576x1xf32>
    %1346 = tosa.reshape %1345 {new_shape = array<i64: 576, 1, 1>} : (tensor<576x1xf32>) -> tensor<576x1x1xf32>
    %1347 = tosa.reshape %1344 {new_shape = array<i64: 1, 576, 1, 1>} : (tensor<576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %1348 = tosa.sub %1334, %1347 : (tensor<1x576x7x7xf32>, tensor<1x576x1x1xf32>) -> tensor<1x576x7x7xf32>
    %1349 = tosa.reshape %1346 {new_shape = array<i64: 1, 576, 1, 1>} : (tensor<576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %1350 = tosa.mul %1348, %1349 {shift = 0 : i8} : (tensor<1x576x7x7xf32>, tensor<1x576x1x1xf32>) -> tensor<1x576x7x7xf32>
    %1351 = tosa.reshape %arg186 {new_shape = array<i64: 576, 1>} : (tensor<576xf32>) -> tensor<576x1xf32>
    %1352 = tosa.reshape %1351 {new_shape = array<i64: 576, 1, 1>} : (tensor<576x1xf32>) -> tensor<576x1x1xf32>
    %1353 = tosa.reshape %1352 {new_shape = array<i64: 1, 576, 1, 1>} : (tensor<576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %1354 = tosa.mul %1350, %1353 {shift = 0 : i8} : (tensor<1x576x7x7xf32>, tensor<1x576x1x1xf32>) -> tensor<1x576x7x7xf32>
    %1355 = tosa.reshape %arg187 {new_shape = array<i64: 576, 1>} : (tensor<576xf32>) -> tensor<576x1xf32>
    %1356 = tosa.reshape %1355 {new_shape = array<i64: 576, 1, 1>} : (tensor<576x1xf32>) -> tensor<576x1x1xf32>
    %1357 = tosa.reshape %1356 {new_shape = array<i64: 1, 576, 1, 1>} : (tensor<576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %1358 = tosa.add %1354, %1357 : (tensor<1x576x7x7xf32>, tensor<1x576x1x1xf32>) -> tensor<1x576x7x7xf32>
    %1359 = "tosa.const"() <{value = dense<3.000000e+00> : tensor<1x576x7x7xf32>}> : () -> tensor<1x576x7x7xf32>
    %1360 = tosa.add %1358, %1359 : (tensor<1x576x7x7xf32>, tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %1361 = tosa.clamp %1360 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %1362 = tosa.clamp %1361 {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0xFF800000 : f32, min_int = -9223372036854775807 : i64} : (tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %1363 = tosa.mul %1358, %1362 {shift = 0 : i8} : (tensor<1x576x7x7xf32>, tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %1364 = "tosa.const"() <{value = dense<6.000000e+00> : tensor<1x576x7x7xf32>}> : () -> tensor<1x576x7x7xf32>
    %1365 = tosa.reciprocal %1364 : (tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %1366 = tosa.mul %1363, %1365 {shift = 0 : i8} : (tensor<1x576x7x7xf32>, tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %1367 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<576xf32>}> : () -> tensor<576xf32>
    %1368 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1369 = tosa.transpose %1366, %1368 : (tensor<1x576x7x7xf32>, tensor<4xi32>) -> tensor<1x7x7x576xf32>
    %1370 = "tosa.const"() <{value = dense<[2, 3, 0, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1371 = tosa.transpose %arg188, %1370 : (tensor<576x1x5x5xf32>, tensor<4xi32>) -> tensor<5x5x576x1xf32>
    %1372 = tosa.depthwise_conv2d %1369, %1371, %1367 {dilation = array<i64: 1, 1>, pad = array<i64: 2, 2, 2, 2>, stride = array<i64: 1, 1>} : (tensor<1x7x7x576xf32>, tensor<5x5x576x1xf32>, tensor<576xf32>) -> tensor<1x7x7x576xf32>
    %1373 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1374 = tosa.transpose %1372, %1373 : (tensor<1x7x7x576xf32>, tensor<4xi32>) -> tensor<1x576x7x7xf32>
    %1375 = tosa.cast %arg189 : (tensor<576xf32>) -> tensor<576xf32>
    %1376 = tosa.cast %arg190 : (tensor<576xf32>) -> tensor<576xf32>
    %1377 = "tosa.const"() <{value = dense<1.000000e-03> : tensor<576xf32>}> : () -> tensor<576xf32>
    %1378 = tosa.add %1376, %1377 : (tensor<576xf32>, tensor<576xf32>) -> tensor<576xf32>
    %1379 = math.sqrt %1378 : tensor<576xf32>
    %1380 = tosa.reciprocal %1379 : (tensor<576xf32>) -> tensor<576xf32>
    %1381 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<576xf32>}> : () -> tensor<576xf32>
    %1382 = tosa.mul %1380, %1381 {shift = 0 : i8} : (tensor<576xf32>, tensor<576xf32>) -> tensor<576xf32>
    %1383 = tosa.reshape %1375 {new_shape = array<i64: 576, 1>} : (tensor<576xf32>) -> tensor<576x1xf32>
    %1384 = tosa.reshape %1383 {new_shape = array<i64: 576, 1, 1>} : (tensor<576x1xf32>) -> tensor<576x1x1xf32>
    %1385 = tosa.reshape %1382 {new_shape = array<i64: 576, 1>} : (tensor<576xf32>) -> tensor<576x1xf32>
    %1386 = tosa.reshape %1385 {new_shape = array<i64: 576, 1, 1>} : (tensor<576x1xf32>) -> tensor<576x1x1xf32>
    %1387 = tosa.reshape %1384 {new_shape = array<i64: 1, 576, 1, 1>} : (tensor<576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %1388 = tosa.sub %1374, %1387 : (tensor<1x576x7x7xf32>, tensor<1x576x1x1xf32>) -> tensor<1x576x7x7xf32>
    %1389 = tosa.reshape %1386 {new_shape = array<i64: 1, 576, 1, 1>} : (tensor<576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %1390 = tosa.mul %1388, %1389 {shift = 0 : i8} : (tensor<1x576x7x7xf32>, tensor<1x576x1x1xf32>) -> tensor<1x576x7x7xf32>
    %1391 = tosa.reshape %arg191 {new_shape = array<i64: 576, 1>} : (tensor<576xf32>) -> tensor<576x1xf32>
    %1392 = tosa.reshape %1391 {new_shape = array<i64: 576, 1, 1>} : (tensor<576x1xf32>) -> tensor<576x1x1xf32>
    %1393 = tosa.reshape %1392 {new_shape = array<i64: 1, 576, 1, 1>} : (tensor<576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %1394 = tosa.mul %1390, %1393 {shift = 0 : i8} : (tensor<1x576x7x7xf32>, tensor<1x576x1x1xf32>) -> tensor<1x576x7x7xf32>
    %1395 = tosa.reshape %arg192 {new_shape = array<i64: 576, 1>} : (tensor<576xf32>) -> tensor<576x1xf32>
    %1396 = tosa.reshape %1395 {new_shape = array<i64: 576, 1, 1>} : (tensor<576x1xf32>) -> tensor<576x1x1xf32>
    %1397 = tosa.reshape %1396 {new_shape = array<i64: 1, 576, 1, 1>} : (tensor<576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %1398 = tosa.add %1394, %1397 : (tensor<1x576x7x7xf32>, tensor<1x576x1x1xf32>) -> tensor<1x576x7x7xf32>
    %1399 = "tosa.const"() <{value = dense<3.000000e+00> : tensor<1x576x7x7xf32>}> : () -> tensor<1x576x7x7xf32>
    %1400 = tosa.add %1398, %1399 : (tensor<1x576x7x7xf32>, tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %1401 = tosa.clamp %1400 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %1402 = tosa.clamp %1401 {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0xFF800000 : f32, min_int = -9223372036854775807 : i64} : (tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %1403 = tosa.mul %1398, %1402 {shift = 0 : i8} : (tensor<1x576x7x7xf32>, tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %1404 = "tosa.const"() <{value = dense<6.000000e+00> : tensor<1x576x7x7xf32>}> : () -> tensor<1x576x7x7xf32>
    %1405 = tosa.reciprocal %1404 : (tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %1406 = tosa.mul %1403, %1405 {shift = 0 : i8} : (tensor<1x576x7x7xf32>, tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %1407 = tosa.reduce_sum %1406 {axis = 3 : i32} : (tensor<1x576x7x7xf32>) -> tensor<1x576x7x1xf32>
    %1408 = tosa.reduce_sum %1407 {axis = 2 : i32} : (tensor<1x576x7x1xf32>) -> tensor<1x576x1x1xf32>
    %1409 = "tosa.const"() <{value = dense<4.900000e+01> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1410 = tosa.reciprocal %1409 : (tensor<1xf32>) -> tensor<1xf32>
    %1411 = tosa.mul %1410, %1408 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %1412 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1413 = tosa.transpose %1411, %1412 : (tensor<1x576x1x1xf32>, tensor<4xi32>) -> tensor<1x1x1x576xf32>
    %1414 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1415 = tosa.transpose %arg193, %1414 : (tensor<144x576x1x1xf32>, tensor<4xi32>) -> tensor<144x1x1x576xf32>
    %1416 = tosa.conv2d %1413, %1415, %arg194 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x1x1x576xf32>, tensor<144x1x1x576xf32>, tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %1417 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1418 = tosa.transpose %1416, %1417 : (tensor<1x1x1x144xf32>, tensor<4xi32>) -> tensor<1x144x1x1xf32>
    %1419 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x144x1x1xf32>}> : () -> tensor<1x144x1x1xf32>
    %1420 = tosa.maximum %1418, %1419 : (tensor<1x144x1x1xf32>, tensor<1x144x1x1xf32>) -> tensor<1x144x1x1xf32>
    %1421 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1422 = tosa.transpose %1420, %1421 : (tensor<1x144x1x1xf32>, tensor<4xi32>) -> tensor<1x1x1x144xf32>
    %1423 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1424 = tosa.transpose %arg195, %1423 : (tensor<576x144x1x1xf32>, tensor<4xi32>) -> tensor<576x1x1x144xf32>
    %1425 = tosa.conv2d %1422, %1424, %arg196 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x1x1x144xf32>, tensor<576x1x1x144xf32>, tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %1426 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1427 = tosa.transpose %1425, %1426 : (tensor<1x1x1x576xf32>, tensor<4xi32>) -> tensor<1x576x1x1xf32>
    %1428 = "tosa.const"() <{value = dense<3.000000e+00> : tensor<1x576x1x1xf32>}> : () -> tensor<1x576x1x1xf32>
    %1429 = tosa.add %1427, %1428 : (tensor<1x576x1x1xf32>, tensor<1x576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %1430 = tosa.clamp %1429 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %1431 = tosa.clamp %1430 {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0xFF800000 : f32, min_int = -9223372036854775807 : i64} : (tensor<1x576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %1432 = "tosa.const"() <{value = dense<6.000000e+00> : tensor<1x576x1x1xf32>}> : () -> tensor<1x576x1x1xf32>
    %1433 = tosa.reciprocal %1432 : (tensor<1x576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %1434 = tosa.mul %1431, %1433 {shift = 0 : i8} : (tensor<1x576x1x1xf32>, tensor<1x576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %1435 = tosa.mul %1434, %1406 {shift = 0 : i8} : (tensor<1x576x1x1xf32>, tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %1436 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<96xf32>}> : () -> tensor<96xf32>
    %1437 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1438 = tosa.transpose %1435, %1437 : (tensor<1x576x7x7xf32>, tensor<4xi32>) -> tensor<1x7x7x576xf32>
    %1439 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1440 = tosa.transpose %arg197, %1439 : (tensor<96x576x1x1xf32>, tensor<4xi32>) -> tensor<96x1x1x576xf32>
    %1441 = tosa.conv2d %1438, %1440, %1436 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x7x7x576xf32>, tensor<96x1x1x576xf32>, tensor<96xf32>) -> tensor<1x7x7x96xf32>
    %1442 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1443 = tosa.transpose %1441, %1442 : (tensor<1x7x7x96xf32>, tensor<4xi32>) -> tensor<1x96x7x7xf32>
    %1444 = tosa.cast %arg198 : (tensor<96xf32>) -> tensor<96xf32>
    %1445 = tosa.cast %arg199 : (tensor<96xf32>) -> tensor<96xf32>
    %1446 = "tosa.const"() <{value = dense<1.000000e-03> : tensor<96xf32>}> : () -> tensor<96xf32>
    %1447 = tosa.add %1445, %1446 : (tensor<96xf32>, tensor<96xf32>) -> tensor<96xf32>
    %1448 = math.sqrt %1447 : tensor<96xf32>
    %1449 = tosa.reciprocal %1448 : (tensor<96xf32>) -> tensor<96xf32>
    %1450 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<96xf32>}> : () -> tensor<96xf32>
    %1451 = tosa.mul %1449, %1450 {shift = 0 : i8} : (tensor<96xf32>, tensor<96xf32>) -> tensor<96xf32>
    %1452 = tosa.reshape %1444 {new_shape = array<i64: 96, 1>} : (tensor<96xf32>) -> tensor<96x1xf32>
    %1453 = tosa.reshape %1452 {new_shape = array<i64: 96, 1, 1>} : (tensor<96x1xf32>) -> tensor<96x1x1xf32>
    %1454 = tosa.reshape %1451 {new_shape = array<i64: 96, 1>} : (tensor<96xf32>) -> tensor<96x1xf32>
    %1455 = tosa.reshape %1454 {new_shape = array<i64: 96, 1, 1>} : (tensor<96x1xf32>) -> tensor<96x1x1xf32>
    %1456 = tosa.reshape %1453 {new_shape = array<i64: 1, 96, 1, 1>} : (tensor<96x1x1xf32>) -> tensor<1x96x1x1xf32>
    %1457 = tosa.sub %1443, %1456 : (tensor<1x96x7x7xf32>, tensor<1x96x1x1xf32>) -> tensor<1x96x7x7xf32>
    %1458 = tosa.reshape %1455 {new_shape = array<i64: 1, 96, 1, 1>} : (tensor<96x1x1xf32>) -> tensor<1x96x1x1xf32>
    %1459 = tosa.mul %1457, %1458 {shift = 0 : i8} : (tensor<1x96x7x7xf32>, tensor<1x96x1x1xf32>) -> tensor<1x96x7x7xf32>
    %1460 = tosa.reshape %arg200 {new_shape = array<i64: 96, 1>} : (tensor<96xf32>) -> tensor<96x1xf32>
    %1461 = tosa.reshape %1460 {new_shape = array<i64: 96, 1, 1>} : (tensor<96x1xf32>) -> tensor<96x1x1xf32>
    %1462 = tosa.reshape %1461 {new_shape = array<i64: 1, 96, 1, 1>} : (tensor<96x1x1xf32>) -> tensor<1x96x1x1xf32>
    %1463 = tosa.mul %1459, %1462 {shift = 0 : i8} : (tensor<1x96x7x7xf32>, tensor<1x96x1x1xf32>) -> tensor<1x96x7x7xf32>
    %1464 = tosa.reshape %arg201 {new_shape = array<i64: 96, 1>} : (tensor<96xf32>) -> tensor<96x1xf32>
    %1465 = tosa.reshape %1464 {new_shape = array<i64: 96, 1, 1>} : (tensor<96x1xf32>) -> tensor<96x1x1xf32>
    %1466 = tosa.reshape %1465 {new_shape = array<i64: 1, 96, 1, 1>} : (tensor<96x1x1xf32>) -> tensor<1x96x1x1xf32>
    %1467 = tosa.add %1463, %1466 : (tensor<1x96x7x7xf32>, tensor<1x96x1x1xf32>) -> tensor<1x96x7x7xf32>
    %1468 = tosa.add %1467, %1326 : (tensor<1x96x7x7xf32>, tensor<1x96x7x7xf32>) -> tensor<1x96x7x7xf32>
    %1469 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<576xf32>}> : () -> tensor<576xf32>
    %1470 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1471 = tosa.transpose %1468, %1470 : (tensor<1x96x7x7xf32>, tensor<4xi32>) -> tensor<1x7x7x96xf32>
    %1472 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1473 = tosa.transpose %arg202, %1472 : (tensor<576x96x1x1xf32>, tensor<4xi32>) -> tensor<576x1x1x96xf32>
    %1474 = tosa.conv2d %1471, %1473, %1469 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x7x7x96xf32>, tensor<576x1x1x96xf32>, tensor<576xf32>) -> tensor<1x7x7x576xf32>
    %1475 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1476 = tosa.transpose %1474, %1475 : (tensor<1x7x7x576xf32>, tensor<4xi32>) -> tensor<1x576x7x7xf32>
    %1477 = tosa.cast %arg203 : (tensor<576xf32>) -> tensor<576xf32>
    %1478 = tosa.cast %arg204 : (tensor<576xf32>) -> tensor<576xf32>
    %1479 = "tosa.const"() <{value = dense<1.000000e-03> : tensor<576xf32>}> : () -> tensor<576xf32>
    %1480 = tosa.add %1478, %1479 : (tensor<576xf32>, tensor<576xf32>) -> tensor<576xf32>
    %1481 = math.sqrt %1480 : tensor<576xf32>
    %1482 = tosa.reciprocal %1481 : (tensor<576xf32>) -> tensor<576xf32>
    %1483 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<576xf32>}> : () -> tensor<576xf32>
    %1484 = tosa.mul %1482, %1483 {shift = 0 : i8} : (tensor<576xf32>, tensor<576xf32>) -> tensor<576xf32>
    %1485 = tosa.reshape %1477 {new_shape = array<i64: 576, 1>} : (tensor<576xf32>) -> tensor<576x1xf32>
    %1486 = tosa.reshape %1485 {new_shape = array<i64: 576, 1, 1>} : (tensor<576x1xf32>) -> tensor<576x1x1xf32>
    %1487 = tosa.reshape %1484 {new_shape = array<i64: 576, 1>} : (tensor<576xf32>) -> tensor<576x1xf32>
    %1488 = tosa.reshape %1487 {new_shape = array<i64: 576, 1, 1>} : (tensor<576x1xf32>) -> tensor<576x1x1xf32>
    %1489 = tosa.reshape %1486 {new_shape = array<i64: 1, 576, 1, 1>} : (tensor<576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %1490 = tosa.sub %1476, %1489 : (tensor<1x576x7x7xf32>, tensor<1x576x1x1xf32>) -> tensor<1x576x7x7xf32>
    %1491 = tosa.reshape %1488 {new_shape = array<i64: 1, 576, 1, 1>} : (tensor<576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %1492 = tosa.mul %1490, %1491 {shift = 0 : i8} : (tensor<1x576x7x7xf32>, tensor<1x576x1x1xf32>) -> tensor<1x576x7x7xf32>
    %1493 = tosa.reshape %arg205 {new_shape = array<i64: 576, 1>} : (tensor<576xf32>) -> tensor<576x1xf32>
    %1494 = tosa.reshape %1493 {new_shape = array<i64: 576, 1, 1>} : (tensor<576x1xf32>) -> tensor<576x1x1xf32>
    %1495 = tosa.reshape %1494 {new_shape = array<i64: 1, 576, 1, 1>} : (tensor<576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %1496 = tosa.mul %1492, %1495 {shift = 0 : i8} : (tensor<1x576x7x7xf32>, tensor<1x576x1x1xf32>) -> tensor<1x576x7x7xf32>
    %1497 = tosa.reshape %arg206 {new_shape = array<i64: 576, 1>} : (tensor<576xf32>) -> tensor<576x1xf32>
    %1498 = tosa.reshape %1497 {new_shape = array<i64: 576, 1, 1>} : (tensor<576x1xf32>) -> tensor<576x1x1xf32>
    %1499 = tosa.reshape %1498 {new_shape = array<i64: 1, 576, 1, 1>} : (tensor<576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %1500 = tosa.add %1496, %1499 : (tensor<1x576x7x7xf32>, tensor<1x576x1x1xf32>) -> tensor<1x576x7x7xf32>
    %1501 = "tosa.const"() <{value = dense<3.000000e+00> : tensor<1x576x7x7xf32>}> : () -> tensor<1x576x7x7xf32>
    %1502 = tosa.add %1500, %1501 : (tensor<1x576x7x7xf32>, tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %1503 = tosa.clamp %1502 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %1504 = tosa.clamp %1503 {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0xFF800000 : f32, min_int = -9223372036854775807 : i64} : (tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %1505 = tosa.mul %1500, %1504 {shift = 0 : i8} : (tensor<1x576x7x7xf32>, tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %1506 = "tosa.const"() <{value = dense<6.000000e+00> : tensor<1x576x7x7xf32>}> : () -> tensor<1x576x7x7xf32>
    %1507 = tosa.reciprocal %1506 : (tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %1508 = tosa.mul %1505, %1507 {shift = 0 : i8} : (tensor<1x576x7x7xf32>, tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %1509 = tosa.reduce_sum %1508 {axis = 3 : i32} : (tensor<1x576x7x7xf32>) -> tensor<1x576x7x1xf32>
    %1510 = tosa.reduce_sum %1509 {axis = 2 : i32} : (tensor<1x576x7x1xf32>) -> tensor<1x576x1x1xf32>
    %1511 = "tosa.const"() <{value = dense<4.900000e+01> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1512 = tosa.reciprocal %1511 : (tensor<1xf32>) -> tensor<1xf32>
    %1513 = tosa.mul %1512, %1510 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %1514 = tosa.reshape %1513 {new_shape = array<i64: 1, 576>} : (tensor<1x576x1x1xf32>) -> tensor<1x576xf32>
    %1515 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1516 = tosa.transpose %arg207, %1515 : (tensor<1024x576xf32>, tensor<2xi32>) -> tensor<576x1024xf32>
    %1517 = tosa.reshape %1514 {new_shape = array<i64: 1, 1, 576>} : (tensor<1x576xf32>) -> tensor<1x1x576xf32>
    %1518 = tosa.reshape %1516 {new_shape = array<i64: 1, 576, 1024>} : (tensor<576x1024xf32>) -> tensor<1x576x1024xf32>
    %1519 = tosa.matmul %1517, %1518 : (tensor<1x1x576xf32>, tensor<1x576x1024xf32>) -> tensor<1x1x1024xf32>
    %1520 = tosa.reshape %1519 {new_shape = array<i64: 1, 1024>} : (tensor<1x1x1024xf32>) -> tensor<1x1024xf32>
    %1521 = tosa.reshape %arg208 {new_shape = array<i64: 1, 1024>} : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %1522 = tosa.add %1521, %1520 : (tensor<1x1024xf32>, tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %1523 = "tosa.const"() <{value = dense<3.000000e+00> : tensor<1x1024xf32>}> : () -> tensor<1x1024xf32>
    %1524 = tosa.add %1522, %1523 : (tensor<1x1024xf32>, tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %1525 = tosa.clamp %1524 {max_fp = 0x7F800000 : f32, max_int = 9223372036854775807 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %1526 = tosa.clamp %1525 {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0xFF800000 : f32, min_int = -9223372036854775807 : i64} : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %1527 = tosa.mul %1522, %1526 {shift = 0 : i8} : (tensor<1x1024xf32>, tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %1528 = "tosa.const"() <{value = dense<6.000000e+00> : tensor<1x1024xf32>}> : () -> tensor<1x1024xf32>
    %1529 = tosa.reciprocal %1528 : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %1530 = tosa.mul %1527, %1529 {shift = 0 : i8} : (tensor<1x1024xf32>, tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %1531 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1532 = tosa.transpose %arg209, %1531 : (tensor<1000x1024xf32>, tensor<2xi32>) -> tensor<1024x1000xf32>
    %1533 = tosa.reshape %1530 {new_shape = array<i64: 1, 1, 1024>} : (tensor<1x1024xf32>) -> tensor<1x1x1024xf32>
    %1534 = tosa.reshape %1532 {new_shape = array<i64: 1, 1024, 1000>} : (tensor<1024x1000xf32>) -> tensor<1x1024x1000xf32>
    %1535 = tosa.matmul %1533, %1534 : (tensor<1x1x1024xf32>, tensor<1x1024x1000xf32>) -> tensor<1x1x1000xf32>
    %1536 = tosa.reshape %1535 {new_shape = array<i64: 1, 1000>} : (tensor<1x1x1000xf32>) -> tensor<1x1000xf32>
    %1537 = tosa.reshape %arg210 {new_shape = array<i64: 1, 1000>} : (tensor<1000xf32>) -> tensor<1x1000xf32>
    %1538 = tosa.add %1537, %1536 : (tensor<1x1000xf32>, tensor<1x1000xf32>) -> tensor<1x1000xf32>
    return %1538 : tensor<1x1000xf32>
  }
}

