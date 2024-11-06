module {
  func.func @subgraph0(%arg0: tensor<1x3x224x224xf32>, %arg1: tensor<64x3x7x7xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64x64x3x3xf32>, %arg7: tensor<64xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64xf32>, %arg10: tensor<64xf32>, %arg11: tensor<64x64x3x3xf32>, %arg12: tensor<64xf32>, %arg13: tensor<64xf32>, %arg14: tensor<64xf32>, %arg15: tensor<64xf32>, %arg16: tensor<64x64x3x3xf32>, %arg17: tensor<64xf32>, %arg18: tensor<64xf32>, %arg19: tensor<64xf32>, %arg20: tensor<64xf32>, %arg21: tensor<64x64x3x3xf32>, %arg22: tensor<64xf32>, %arg23: tensor<64xf32>, %arg24: tensor<64xf32>, %arg25: tensor<64xf32>, %arg26: tensor<128x64x3x3xf32>, %arg27: tensor<128xf32>, %arg28: tensor<128xf32>, %arg29: tensor<128xf32>, %arg30: tensor<128xf32>, %arg31: tensor<128x128x3x3xf32>, %arg32: tensor<128xf32>, %arg33: tensor<128xf32>, %arg34: tensor<128xf32>, %arg35: tensor<128xf32>, %arg36: tensor<128x64x1x1xf32>, %arg37: tensor<128xf32>, %arg38: tensor<128xf32>, %arg39: tensor<128xf32>, %arg40: tensor<128xf32>, %arg41: tensor<128x128x3x3xf32>, %arg42: tensor<128xf32>, %arg43: tensor<128xf32>, %arg44: tensor<128xf32>, %arg45: tensor<128xf32>, %arg46: tensor<128x128x3x3xf32>, %arg47: tensor<128xf32>, %arg48: tensor<128xf32>, %arg49: tensor<128xf32>, %arg50: tensor<128xf32>, %arg51: tensor<256x128x3x3xf32>, %arg52: tensor<256xf32>, %arg53: tensor<256xf32>, %arg54: tensor<256xf32>, %arg55: tensor<256xf32>, %arg56: tensor<256x256x3x3xf32>, %arg57: tensor<256xf32>, %arg58: tensor<256xf32>, %arg59: tensor<256xf32>, %arg60: tensor<256xf32>, %arg61: tensor<256x128x1x1xf32>, %arg62: tensor<256xf32>, %arg63: tensor<256xf32>, %arg64: tensor<256xf32>, %arg65: tensor<256xf32>, %arg66: tensor<256x256x3x3xf32>, %arg67: tensor<256xf32>, %arg68: tensor<256xf32>, %arg69: tensor<256xf32>, %arg70: tensor<256xf32>, %arg71: tensor<256x256x3x3xf32>, %arg72: tensor<256xf32>, %arg73: tensor<256xf32>, %arg74: tensor<256xf32>, %arg75: tensor<256xf32>, %arg76: tensor<512x256x3x3xf32>, %arg77: tensor<512xf32>, %arg78: tensor<512xf32>, %arg79: tensor<512xf32>, %arg80: tensor<512xf32>, %arg81: tensor<512x512x3x3xf32>, %arg82: tensor<512xf32>, %arg83: tensor<512xf32>, %arg84: tensor<512xf32>, %arg85: tensor<512xf32>, %arg86: tensor<512x256x1x1xf32>, %arg87: tensor<512xf32>, %arg88: tensor<512xf32>, %arg89: tensor<512xf32>, %arg90: tensor<512xf32>, %arg91: tensor<512x512x3x3xf32>, %arg92: tensor<512xf32>, %arg93: tensor<512xf32>, %arg94: tensor<512xf32>, %arg95: tensor<512xf32>, %arg96: tensor<512x512x3x3xf32>, %arg97: tensor<512xf32>, %arg98: tensor<512xf32>, %arg99: tensor<512xf32>, %arg100: tensor<512xf32>, %arg101: tensor<10x512xf32>, %arg102: tensor<10xf32>) -> tensor<1x10xf32> {
    %0 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %1 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2 = tosa.transpose %arg0, %1 : (tensor<1x3x224x224xf32>, tensor<4xi32>) -> tensor<1x224x224x3xf32>
    %3 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4 = tosa.transpose %arg1, %3 : (tensor<64x3x7x7xf32>, tensor<4xi32>) -> tensor<64x7x7x3xf32>
    %5 = tosa.conv2d %2, %4, %0 {dilation = array<i64: 1, 1>, pad = array<i64: 3, 3, 3, 3>, stride = array<i64: 2, 2>} : (tensor<1x224x224x3xf32>, tensor<64x7x7x3xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
    %6 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %7 = tosa.transpose %5, %6 : (tensor<1x112x112x64xf32>, tensor<4xi32>) -> tensor<1x64x112x112xf32>
    %8 = tosa.cast %arg2 : (tensor<64xf32>) -> tensor<64xf32>
    %9 = tosa.cast %arg3 : (tensor<64xf32>) -> tensor<64xf32>
    %10 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<64xf32>}> : () -> tensor<64xf32>
    %11 = tosa.add %9, %10 : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %12 = math.sqrt %11 : tensor<64xf32>
    %13 = tosa.reciprocal %12 : (tensor<64xf32>) -> tensor<64xf32>
    %14 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %15 = tosa.mul %13, %14 {shift = 0 : i8} : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %16 = tosa.reshape %8 {new_shape = array<i64: 64, 1>} : (tensor<64xf32>) -> tensor<64x1xf32>
    %17 = tosa.reshape %16 {new_shape = array<i64: 64, 1, 1>} : (tensor<64x1xf32>) -> tensor<64x1x1xf32>
    %18 = tosa.reshape %15 {new_shape = array<i64: 64, 1>} : (tensor<64xf32>) -> tensor<64x1xf32>
    %19 = tosa.reshape %18 {new_shape = array<i64: 64, 1, 1>} : (tensor<64x1xf32>) -> tensor<64x1x1xf32>
    %20 = tosa.reshape %17 {new_shape = array<i64: 1, 64, 1, 1>} : (tensor<64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %21 = tosa.sub %7, %20 : (tensor<1x64x112x112xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x112x112xf32>
    %22 = tosa.reshape %19 {new_shape = array<i64: 1, 64, 1, 1>} : (tensor<64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %23 = tosa.mul %21, %22 {shift = 0 : i8} : (tensor<1x64x112x112xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x112x112xf32>
    %24 = tosa.reshape %arg4 {new_shape = array<i64: 64, 1>} : (tensor<64xf32>) -> tensor<64x1xf32>
    %25 = tosa.reshape %24 {new_shape = array<i64: 64, 1, 1>} : (tensor<64x1xf32>) -> tensor<64x1x1xf32>
    %26 = tosa.reshape %25 {new_shape = array<i64: 1, 64, 1, 1>} : (tensor<64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %27 = tosa.mul %23, %26 {shift = 0 : i8} : (tensor<1x64x112x112xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x112x112xf32>
    %28 = tosa.reshape %arg5 {new_shape = array<i64: 64, 1>} : (tensor<64xf32>) -> tensor<64x1xf32>
    %29 = tosa.reshape %28 {new_shape = array<i64: 64, 1, 1>} : (tensor<64x1xf32>) -> tensor<64x1x1xf32>
    %30 = tosa.reshape %29 {new_shape = array<i64: 1, 64, 1, 1>} : (tensor<64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %31 = tosa.add %27, %30 : (tensor<1x64x112x112xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x112x112xf32>
    %32 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x64x112x112xf32>}> : () -> tensor<1x64x112x112xf32>
    %33 = tosa.maximum %31, %32 : (tensor<1x64x112x112xf32>, tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %34 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %35 = tosa.transpose %33, %34 : (tensor<1x64x112x112xf32>, tensor<4xi32>) -> tensor<1x112x112x64xf32>
    %36 = tosa.max_pool2d %35 {kernel = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>} : (tensor<1x112x112x64xf32>) -> tensor<1x56x56x64xf32>
    %37 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %38 = tosa.transpose %36, %37 : (tensor<1x56x56x64xf32>, tensor<4xi32>) -> tensor<1x64x56x56xf32>
    %39 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %40 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %41 = tosa.transpose %38, %40 : (tensor<1x64x56x56xf32>, tensor<4xi32>) -> tensor<1x56x56x64xf32>
    %42 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %43 = tosa.transpose %arg6, %42 : (tensor<64x64x3x3xf32>, tensor<4xi32>) -> tensor<64x3x3x64xf32>
    %44 = tosa.conv2d %41, %43, %39 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x56x56x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
    %45 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %46 = tosa.transpose %44, %45 : (tensor<1x56x56x64xf32>, tensor<4xi32>) -> tensor<1x64x56x56xf32>
    %47 = tosa.cast %arg7 : (tensor<64xf32>) -> tensor<64xf32>
    %48 = tosa.cast %arg8 : (tensor<64xf32>) -> tensor<64xf32>
    %49 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<64xf32>}> : () -> tensor<64xf32>
    %50 = tosa.add %48, %49 : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %51 = math.sqrt %50 : tensor<64xf32>
    %52 = tosa.reciprocal %51 : (tensor<64xf32>) -> tensor<64xf32>
    %53 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %54 = tosa.mul %52, %53 {shift = 0 : i8} : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %55 = tosa.reshape %47 {new_shape = array<i64: 64, 1>} : (tensor<64xf32>) -> tensor<64x1xf32>
    %56 = tosa.reshape %55 {new_shape = array<i64: 64, 1, 1>} : (tensor<64x1xf32>) -> tensor<64x1x1xf32>
    %57 = tosa.reshape %54 {new_shape = array<i64: 64, 1>} : (tensor<64xf32>) -> tensor<64x1xf32>
    %58 = tosa.reshape %57 {new_shape = array<i64: 64, 1, 1>} : (tensor<64x1xf32>) -> tensor<64x1x1xf32>
    %59 = tosa.reshape %56 {new_shape = array<i64: 1, 64, 1, 1>} : (tensor<64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %60 = tosa.sub %46, %59 : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %61 = tosa.reshape %58 {new_shape = array<i64: 1, 64, 1, 1>} : (tensor<64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %62 = tosa.mul %60, %61 {shift = 0 : i8} : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %63 = tosa.reshape %arg9 {new_shape = array<i64: 64, 1>} : (tensor<64xf32>) -> tensor<64x1xf32>
    %64 = tosa.reshape %63 {new_shape = array<i64: 64, 1, 1>} : (tensor<64x1xf32>) -> tensor<64x1x1xf32>
    %65 = tosa.reshape %64 {new_shape = array<i64: 1, 64, 1, 1>} : (tensor<64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %66 = tosa.mul %62, %65 {shift = 0 : i8} : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %67 = tosa.reshape %arg10 {new_shape = array<i64: 64, 1>} : (tensor<64xf32>) -> tensor<64x1xf32>
    %68 = tosa.reshape %67 {new_shape = array<i64: 64, 1, 1>} : (tensor<64x1xf32>) -> tensor<64x1x1xf32>
    %69 = tosa.reshape %68 {new_shape = array<i64: 1, 64, 1, 1>} : (tensor<64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %70 = tosa.add %66, %69 : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %71 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x64x56x56xf32>}> : () -> tensor<1x64x56x56xf32>
    %72 = tosa.maximum %70, %71 : (tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %73 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %74 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %75 = tosa.transpose %72, %74 : (tensor<1x64x56x56xf32>, tensor<4xi32>) -> tensor<1x56x56x64xf32>
    %76 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %77 = tosa.transpose %arg11, %76 : (tensor<64x64x3x3xf32>, tensor<4xi32>) -> tensor<64x3x3x64xf32>
    %78 = tosa.conv2d %75, %77, %73 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x56x56x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
    %79 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %80 = tosa.transpose %78, %79 : (tensor<1x56x56x64xf32>, tensor<4xi32>) -> tensor<1x64x56x56xf32>
    %81 = tosa.cast %arg12 : (tensor<64xf32>) -> tensor<64xf32>
    %82 = tosa.cast %arg13 : (tensor<64xf32>) -> tensor<64xf32>
    %83 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<64xf32>}> : () -> tensor<64xf32>
    %84 = tosa.add %82, %83 : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %85 = math.sqrt %84 : tensor<64xf32>
    %86 = tosa.reciprocal %85 : (tensor<64xf32>) -> tensor<64xf32>
    %87 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %88 = tosa.mul %86, %87 {shift = 0 : i8} : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %89 = tosa.reshape %81 {new_shape = array<i64: 64, 1>} : (tensor<64xf32>) -> tensor<64x1xf32>
    %90 = tosa.reshape %89 {new_shape = array<i64: 64, 1, 1>} : (tensor<64x1xf32>) -> tensor<64x1x1xf32>
    %91 = tosa.reshape %88 {new_shape = array<i64: 64, 1>} : (tensor<64xf32>) -> tensor<64x1xf32>
    %92 = tosa.reshape %91 {new_shape = array<i64: 64, 1, 1>} : (tensor<64x1xf32>) -> tensor<64x1x1xf32>
    %93 = tosa.reshape %90 {new_shape = array<i64: 1, 64, 1, 1>} : (tensor<64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %94 = tosa.sub %80, %93 : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %95 = tosa.reshape %92 {new_shape = array<i64: 1, 64, 1, 1>} : (tensor<64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %96 = tosa.mul %94, %95 {shift = 0 : i8} : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %97 = tosa.reshape %arg14 {new_shape = array<i64: 64, 1>} : (tensor<64xf32>) -> tensor<64x1xf32>
    %98 = tosa.reshape %97 {new_shape = array<i64: 64, 1, 1>} : (tensor<64x1xf32>) -> tensor<64x1x1xf32>
    %99 = tosa.reshape %98 {new_shape = array<i64: 1, 64, 1, 1>} : (tensor<64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %100 = tosa.mul %96, %99 {shift = 0 : i8} : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %101 = tosa.reshape %arg15 {new_shape = array<i64: 64, 1>} : (tensor<64xf32>) -> tensor<64x1xf32>
    %102 = tosa.reshape %101 {new_shape = array<i64: 64, 1, 1>} : (tensor<64x1xf32>) -> tensor<64x1x1xf32>
    %103 = tosa.reshape %102 {new_shape = array<i64: 1, 64, 1, 1>} : (tensor<64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %104 = tosa.add %100, %103 : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %105 = tosa.add %104, %38 : (tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %106 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x64x56x56xf32>}> : () -> tensor<1x64x56x56xf32>
    %107 = tosa.maximum %105, %106 : (tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %108 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %109 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %110 = tosa.transpose %107, %109 : (tensor<1x64x56x56xf32>, tensor<4xi32>) -> tensor<1x56x56x64xf32>
    %111 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %112 = tosa.transpose %arg16, %111 : (tensor<64x64x3x3xf32>, tensor<4xi32>) -> tensor<64x3x3x64xf32>
    %113 = tosa.conv2d %110, %112, %108 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x56x56x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
    %114 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %115 = tosa.transpose %113, %114 : (tensor<1x56x56x64xf32>, tensor<4xi32>) -> tensor<1x64x56x56xf32>
    %116 = tosa.cast %arg17 : (tensor<64xf32>) -> tensor<64xf32>
    %117 = tosa.cast %arg18 : (tensor<64xf32>) -> tensor<64xf32>
    %118 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<64xf32>}> : () -> tensor<64xf32>
    %119 = tosa.add %117, %118 : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %120 = math.sqrt %119 : tensor<64xf32>
    %121 = tosa.reciprocal %120 : (tensor<64xf32>) -> tensor<64xf32>
    %122 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %123 = tosa.mul %121, %122 {shift = 0 : i8} : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %124 = tosa.reshape %116 {new_shape = array<i64: 64, 1>} : (tensor<64xf32>) -> tensor<64x1xf32>
    %125 = tosa.reshape %124 {new_shape = array<i64: 64, 1, 1>} : (tensor<64x1xf32>) -> tensor<64x1x1xf32>
    %126 = tosa.reshape %123 {new_shape = array<i64: 64, 1>} : (tensor<64xf32>) -> tensor<64x1xf32>
    %127 = tosa.reshape %126 {new_shape = array<i64: 64, 1, 1>} : (tensor<64x1xf32>) -> tensor<64x1x1xf32>
    %128 = tosa.reshape %125 {new_shape = array<i64: 1, 64, 1, 1>} : (tensor<64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %129 = tosa.sub %115, %128 : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %130 = tosa.reshape %127 {new_shape = array<i64: 1, 64, 1, 1>} : (tensor<64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %131 = tosa.mul %129, %130 {shift = 0 : i8} : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %132 = tosa.reshape %arg19 {new_shape = array<i64: 64, 1>} : (tensor<64xf32>) -> tensor<64x1xf32>
    %133 = tosa.reshape %132 {new_shape = array<i64: 64, 1, 1>} : (tensor<64x1xf32>) -> tensor<64x1x1xf32>
    %134 = tosa.reshape %133 {new_shape = array<i64: 1, 64, 1, 1>} : (tensor<64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %135 = tosa.mul %131, %134 {shift = 0 : i8} : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %136 = tosa.reshape %arg20 {new_shape = array<i64: 64, 1>} : (tensor<64xf32>) -> tensor<64x1xf32>
    %137 = tosa.reshape %136 {new_shape = array<i64: 64, 1, 1>} : (tensor<64x1xf32>) -> tensor<64x1x1xf32>
    %138 = tosa.reshape %137 {new_shape = array<i64: 1, 64, 1, 1>} : (tensor<64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %139 = tosa.add %135, %138 : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %140 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x64x56x56xf32>}> : () -> tensor<1x64x56x56xf32>
    %141 = tosa.maximum %139, %140 : (tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %142 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %143 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %144 = tosa.transpose %141, %143 : (tensor<1x64x56x56xf32>, tensor<4xi32>) -> tensor<1x56x56x64xf32>
    %145 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %146 = tosa.transpose %arg21, %145 : (tensor<64x64x3x3xf32>, tensor<4xi32>) -> tensor<64x3x3x64xf32>
    %147 = tosa.conv2d %144, %146, %142 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x56x56x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
    %148 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %149 = tosa.transpose %147, %148 : (tensor<1x56x56x64xf32>, tensor<4xi32>) -> tensor<1x64x56x56xf32>
    %150 = tosa.cast %arg22 : (tensor<64xf32>) -> tensor<64xf32>
    %151 = tosa.cast %arg23 : (tensor<64xf32>) -> tensor<64xf32>
    %152 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<64xf32>}> : () -> tensor<64xf32>
    %153 = tosa.add %151, %152 : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %154 = math.sqrt %153 : tensor<64xf32>
    %155 = tosa.reciprocal %154 : (tensor<64xf32>) -> tensor<64xf32>
    %156 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %157 = tosa.mul %155, %156 {shift = 0 : i8} : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %158 = tosa.reshape %150 {new_shape = array<i64: 64, 1>} : (tensor<64xf32>) -> tensor<64x1xf32>
    %159 = tosa.reshape %158 {new_shape = array<i64: 64, 1, 1>} : (tensor<64x1xf32>) -> tensor<64x1x1xf32>
    %160 = tosa.reshape %157 {new_shape = array<i64: 64, 1>} : (tensor<64xf32>) -> tensor<64x1xf32>
    %161 = tosa.reshape %160 {new_shape = array<i64: 64, 1, 1>} : (tensor<64x1xf32>) -> tensor<64x1x1xf32>
    %162 = tosa.reshape %159 {new_shape = array<i64: 1, 64, 1, 1>} : (tensor<64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %163 = tosa.sub %149, %162 : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %164 = tosa.reshape %161 {new_shape = array<i64: 1, 64, 1, 1>} : (tensor<64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %165 = tosa.mul %163, %164 {shift = 0 : i8} : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %166 = tosa.reshape %arg24 {new_shape = array<i64: 64, 1>} : (tensor<64xf32>) -> tensor<64x1xf32>
    %167 = tosa.reshape %166 {new_shape = array<i64: 64, 1, 1>} : (tensor<64x1xf32>) -> tensor<64x1x1xf32>
    %168 = tosa.reshape %167 {new_shape = array<i64: 1, 64, 1, 1>} : (tensor<64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %169 = tosa.mul %165, %168 {shift = 0 : i8} : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %170 = tosa.reshape %arg25 {new_shape = array<i64: 64, 1>} : (tensor<64xf32>) -> tensor<64x1xf32>
    %171 = tosa.reshape %170 {new_shape = array<i64: 64, 1, 1>} : (tensor<64x1xf32>) -> tensor<64x1x1xf32>
    %172 = tosa.reshape %171 {new_shape = array<i64: 1, 64, 1, 1>} : (tensor<64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %173 = tosa.add %169, %172 : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %174 = tosa.add %173, %107 : (tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %175 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x64x56x56xf32>}> : () -> tensor<1x64x56x56xf32>
    %176 = tosa.maximum %174, %175 : (tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %177 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<128xf32>}> : () -> tensor<128xf32>
    %178 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %179 = tosa.transpose %176, %178 : (tensor<1x64x56x56xf32>, tensor<4xi32>) -> tensor<1x56x56x64xf32>
    %180 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %181 = tosa.transpose %arg26, %180 : (tensor<128x64x3x3xf32>, tensor<4xi32>) -> tensor<128x3x3x64xf32>
    %182 = tosa.conv2d %179, %181, %177 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>} : (tensor<1x56x56x64xf32>, tensor<128x3x3x64xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
    %183 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %184 = tosa.transpose %182, %183 : (tensor<1x28x28x128xf32>, tensor<4xi32>) -> tensor<1x128x28x28xf32>
    %185 = tosa.cast %arg27 : (tensor<128xf32>) -> tensor<128xf32>
    %186 = tosa.cast %arg28 : (tensor<128xf32>) -> tensor<128xf32>
    %187 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<128xf32>}> : () -> tensor<128xf32>
    %188 = tosa.add %186, %187 : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %189 = math.sqrt %188 : tensor<128xf32>
    %190 = tosa.reciprocal %189 : (tensor<128xf32>) -> tensor<128xf32>
    %191 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<128xf32>}> : () -> tensor<128xf32>
    %192 = tosa.mul %190, %191 {shift = 0 : i8} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %193 = tosa.reshape %185 {new_shape = array<i64: 128, 1>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %194 = tosa.reshape %193 {new_shape = array<i64: 128, 1, 1>} : (tensor<128x1xf32>) -> tensor<128x1x1xf32>
    %195 = tosa.reshape %192 {new_shape = array<i64: 128, 1>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %196 = tosa.reshape %195 {new_shape = array<i64: 128, 1, 1>} : (tensor<128x1xf32>) -> tensor<128x1x1xf32>
    %197 = tosa.reshape %194 {new_shape = array<i64: 1, 128, 1, 1>} : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %198 = tosa.sub %184, %197 : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %199 = tosa.reshape %196 {new_shape = array<i64: 1, 128, 1, 1>} : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %200 = tosa.mul %198, %199 {shift = 0 : i8} : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %201 = tosa.reshape %arg29 {new_shape = array<i64: 128, 1>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %202 = tosa.reshape %201 {new_shape = array<i64: 128, 1, 1>} : (tensor<128x1xf32>) -> tensor<128x1x1xf32>
    %203 = tosa.reshape %202 {new_shape = array<i64: 1, 128, 1, 1>} : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %204 = tosa.mul %200, %203 {shift = 0 : i8} : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %205 = tosa.reshape %arg30 {new_shape = array<i64: 128, 1>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %206 = tosa.reshape %205 {new_shape = array<i64: 128, 1, 1>} : (tensor<128x1xf32>) -> tensor<128x1x1xf32>
    %207 = tosa.reshape %206 {new_shape = array<i64: 1, 128, 1, 1>} : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %208 = tosa.add %204, %207 : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %209 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x128x28x28xf32>}> : () -> tensor<1x128x28x28xf32>
    %210 = tosa.maximum %208, %209 : (tensor<1x128x28x28xf32>, tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %211 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<128xf32>}> : () -> tensor<128xf32>
    %212 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %213 = tosa.transpose %210, %212 : (tensor<1x128x28x28xf32>, tensor<4xi32>) -> tensor<1x28x28x128xf32>
    %214 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %215 = tosa.transpose %arg31, %214 : (tensor<128x128x3x3xf32>, tensor<4xi32>) -> tensor<128x3x3x128xf32>
    %216 = tosa.conv2d %213, %215, %211 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x28x28x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
    %217 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %218 = tosa.transpose %216, %217 : (tensor<1x28x28x128xf32>, tensor<4xi32>) -> tensor<1x128x28x28xf32>
    %219 = tosa.cast %arg32 : (tensor<128xf32>) -> tensor<128xf32>
    %220 = tosa.cast %arg33 : (tensor<128xf32>) -> tensor<128xf32>
    %221 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<128xf32>}> : () -> tensor<128xf32>
    %222 = tosa.add %220, %221 : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %223 = math.sqrt %222 : tensor<128xf32>
    %224 = tosa.reciprocal %223 : (tensor<128xf32>) -> tensor<128xf32>
    %225 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<128xf32>}> : () -> tensor<128xf32>
    %226 = tosa.mul %224, %225 {shift = 0 : i8} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %227 = tosa.reshape %219 {new_shape = array<i64: 128, 1>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %228 = tosa.reshape %227 {new_shape = array<i64: 128, 1, 1>} : (tensor<128x1xf32>) -> tensor<128x1x1xf32>
    %229 = tosa.reshape %226 {new_shape = array<i64: 128, 1>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %230 = tosa.reshape %229 {new_shape = array<i64: 128, 1, 1>} : (tensor<128x1xf32>) -> tensor<128x1x1xf32>
    %231 = tosa.reshape %228 {new_shape = array<i64: 1, 128, 1, 1>} : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %232 = tosa.sub %218, %231 : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %233 = tosa.reshape %230 {new_shape = array<i64: 1, 128, 1, 1>} : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %234 = tosa.mul %232, %233 {shift = 0 : i8} : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %235 = tosa.reshape %arg34 {new_shape = array<i64: 128, 1>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %236 = tosa.reshape %235 {new_shape = array<i64: 128, 1, 1>} : (tensor<128x1xf32>) -> tensor<128x1x1xf32>
    %237 = tosa.reshape %236 {new_shape = array<i64: 1, 128, 1, 1>} : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %238 = tosa.mul %234, %237 {shift = 0 : i8} : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %239 = tosa.reshape %arg35 {new_shape = array<i64: 128, 1>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %240 = tosa.reshape %239 {new_shape = array<i64: 128, 1, 1>} : (tensor<128x1xf32>) -> tensor<128x1x1xf32>
    %241 = tosa.reshape %240 {new_shape = array<i64: 1, 128, 1, 1>} : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %242 = tosa.add %238, %241 : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %243 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<128xf32>}> : () -> tensor<128xf32>
    %244 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %245 = tosa.transpose %176, %244 : (tensor<1x64x56x56xf32>, tensor<4xi32>) -> tensor<1x56x56x64xf32>
    %246 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %247 = tosa.transpose %arg36, %246 : (tensor<128x64x1x1xf32>, tensor<4xi32>) -> tensor<128x1x1x64xf32>
    %248 = tosa.conv2d %245, %247, %243 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x56x56x64xf32>, tensor<128x1x1x64xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
    %249 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %250 = tosa.transpose %248, %249 : (tensor<1x28x28x128xf32>, tensor<4xi32>) -> tensor<1x128x28x28xf32>
    %251 = tosa.cast %arg37 : (tensor<128xf32>) -> tensor<128xf32>
    %252 = tosa.cast %arg38 : (tensor<128xf32>) -> tensor<128xf32>
    %253 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<128xf32>}> : () -> tensor<128xf32>
    %254 = tosa.add %252, %253 : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %255 = math.sqrt %254 : tensor<128xf32>
    %256 = tosa.reciprocal %255 : (tensor<128xf32>) -> tensor<128xf32>
    %257 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<128xf32>}> : () -> tensor<128xf32>
    %258 = tosa.mul %256, %257 {shift = 0 : i8} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %259 = tosa.reshape %251 {new_shape = array<i64: 128, 1>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %260 = tosa.reshape %259 {new_shape = array<i64: 128, 1, 1>} : (tensor<128x1xf32>) -> tensor<128x1x1xf32>
    %261 = tosa.reshape %258 {new_shape = array<i64: 128, 1>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %262 = tosa.reshape %261 {new_shape = array<i64: 128, 1, 1>} : (tensor<128x1xf32>) -> tensor<128x1x1xf32>
    %263 = tosa.reshape %260 {new_shape = array<i64: 1, 128, 1, 1>} : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %264 = tosa.sub %250, %263 : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %265 = tosa.reshape %262 {new_shape = array<i64: 1, 128, 1, 1>} : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %266 = tosa.mul %264, %265 {shift = 0 : i8} : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %267 = tosa.reshape %arg39 {new_shape = array<i64: 128, 1>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %268 = tosa.reshape %267 {new_shape = array<i64: 128, 1, 1>} : (tensor<128x1xf32>) -> tensor<128x1x1xf32>
    %269 = tosa.reshape %268 {new_shape = array<i64: 1, 128, 1, 1>} : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %270 = tosa.mul %266, %269 {shift = 0 : i8} : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %271 = tosa.reshape %arg40 {new_shape = array<i64: 128, 1>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %272 = tosa.reshape %271 {new_shape = array<i64: 128, 1, 1>} : (tensor<128x1xf32>) -> tensor<128x1x1xf32>
    %273 = tosa.reshape %272 {new_shape = array<i64: 1, 128, 1, 1>} : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %274 = tosa.add %270, %273 : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %275 = tosa.add %242, %274 : (tensor<1x128x28x28xf32>, tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %276 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x128x28x28xf32>}> : () -> tensor<1x128x28x28xf32>
    %277 = tosa.maximum %275, %276 : (tensor<1x128x28x28xf32>, tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %278 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<128xf32>}> : () -> tensor<128xf32>
    %279 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %280 = tosa.transpose %277, %279 : (tensor<1x128x28x28xf32>, tensor<4xi32>) -> tensor<1x28x28x128xf32>
    %281 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %282 = tosa.transpose %arg41, %281 : (tensor<128x128x3x3xf32>, tensor<4xi32>) -> tensor<128x3x3x128xf32>
    %283 = tosa.conv2d %280, %282, %278 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x28x28x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
    %284 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %285 = tosa.transpose %283, %284 : (tensor<1x28x28x128xf32>, tensor<4xi32>) -> tensor<1x128x28x28xf32>
    %286 = tosa.cast %arg42 : (tensor<128xf32>) -> tensor<128xf32>
    %287 = tosa.cast %arg43 : (tensor<128xf32>) -> tensor<128xf32>
    %288 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<128xf32>}> : () -> tensor<128xf32>
    %289 = tosa.add %287, %288 : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %290 = math.sqrt %289 : tensor<128xf32>
    %291 = tosa.reciprocal %290 : (tensor<128xf32>) -> tensor<128xf32>
    %292 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<128xf32>}> : () -> tensor<128xf32>
    %293 = tosa.mul %291, %292 {shift = 0 : i8} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %294 = tosa.reshape %286 {new_shape = array<i64: 128, 1>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %295 = tosa.reshape %294 {new_shape = array<i64: 128, 1, 1>} : (tensor<128x1xf32>) -> tensor<128x1x1xf32>
    %296 = tosa.reshape %293 {new_shape = array<i64: 128, 1>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %297 = tosa.reshape %296 {new_shape = array<i64: 128, 1, 1>} : (tensor<128x1xf32>) -> tensor<128x1x1xf32>
    %298 = tosa.reshape %295 {new_shape = array<i64: 1, 128, 1, 1>} : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %299 = tosa.sub %285, %298 : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %300 = tosa.reshape %297 {new_shape = array<i64: 1, 128, 1, 1>} : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %301 = tosa.mul %299, %300 {shift = 0 : i8} : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %302 = tosa.reshape %arg44 {new_shape = array<i64: 128, 1>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %303 = tosa.reshape %302 {new_shape = array<i64: 128, 1, 1>} : (tensor<128x1xf32>) -> tensor<128x1x1xf32>
    %304 = tosa.reshape %303 {new_shape = array<i64: 1, 128, 1, 1>} : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %305 = tosa.mul %301, %304 {shift = 0 : i8} : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %306 = tosa.reshape %arg45 {new_shape = array<i64: 128, 1>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %307 = tosa.reshape %306 {new_shape = array<i64: 128, 1, 1>} : (tensor<128x1xf32>) -> tensor<128x1x1xf32>
    %308 = tosa.reshape %307 {new_shape = array<i64: 1, 128, 1, 1>} : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %309 = tosa.add %305, %308 : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %310 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x128x28x28xf32>}> : () -> tensor<1x128x28x28xf32>
    %311 = tosa.maximum %309, %310 : (tensor<1x128x28x28xf32>, tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %312 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<128xf32>}> : () -> tensor<128xf32>
    %313 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %314 = tosa.transpose %311, %313 : (tensor<1x128x28x28xf32>, tensor<4xi32>) -> tensor<1x28x28x128xf32>
    %315 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %316 = tosa.transpose %arg46, %315 : (tensor<128x128x3x3xf32>, tensor<4xi32>) -> tensor<128x3x3x128xf32>
    %317 = tosa.conv2d %314, %316, %312 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x28x28x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
    %318 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %319 = tosa.transpose %317, %318 : (tensor<1x28x28x128xf32>, tensor<4xi32>) -> tensor<1x128x28x28xf32>
    %320 = tosa.cast %arg47 : (tensor<128xf32>) -> tensor<128xf32>
    %321 = tosa.cast %arg48 : (tensor<128xf32>) -> tensor<128xf32>
    %322 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<128xf32>}> : () -> tensor<128xf32>
    %323 = tosa.add %321, %322 : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %324 = math.sqrt %323 : tensor<128xf32>
    %325 = tosa.reciprocal %324 : (tensor<128xf32>) -> tensor<128xf32>
    %326 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<128xf32>}> : () -> tensor<128xf32>
    %327 = tosa.mul %325, %326 {shift = 0 : i8} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %328 = tosa.reshape %320 {new_shape = array<i64: 128, 1>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %329 = tosa.reshape %328 {new_shape = array<i64: 128, 1, 1>} : (tensor<128x1xf32>) -> tensor<128x1x1xf32>
    %330 = tosa.reshape %327 {new_shape = array<i64: 128, 1>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %331 = tosa.reshape %330 {new_shape = array<i64: 128, 1, 1>} : (tensor<128x1xf32>) -> tensor<128x1x1xf32>
    %332 = tosa.reshape %329 {new_shape = array<i64: 1, 128, 1, 1>} : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %333 = tosa.sub %319, %332 : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %334 = tosa.reshape %331 {new_shape = array<i64: 1, 128, 1, 1>} : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %335 = tosa.mul %333, %334 {shift = 0 : i8} : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %336 = tosa.reshape %arg49 {new_shape = array<i64: 128, 1>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %337 = tosa.reshape %336 {new_shape = array<i64: 128, 1, 1>} : (tensor<128x1xf32>) -> tensor<128x1x1xf32>
    %338 = tosa.reshape %337 {new_shape = array<i64: 1, 128, 1, 1>} : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %339 = tosa.mul %335, %338 {shift = 0 : i8} : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %340 = tosa.reshape %arg50 {new_shape = array<i64: 128, 1>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %341 = tosa.reshape %340 {new_shape = array<i64: 128, 1, 1>} : (tensor<128x1xf32>) -> tensor<128x1x1xf32>
    %342 = tosa.reshape %341 {new_shape = array<i64: 1, 128, 1, 1>} : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %343 = tosa.add %339, %342 : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %344 = tosa.add %343, %277 : (tensor<1x128x28x28xf32>, tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %345 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x128x28x28xf32>}> : () -> tensor<1x128x28x28xf32>
    %346 = tosa.maximum %344, %345 : (tensor<1x128x28x28xf32>, tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %347 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<256xf32>}> : () -> tensor<256xf32>
    %348 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %349 = tosa.transpose %346, %348 : (tensor<1x128x28x28xf32>, tensor<4xi32>) -> tensor<1x28x28x128xf32>
    %350 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %351 = tosa.transpose %arg51, %350 : (tensor<256x128x3x3xf32>, tensor<4xi32>) -> tensor<256x3x3x128xf32>
    %352 = tosa.conv2d %349, %351, %347 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>} : (tensor<1x28x28x128xf32>, tensor<256x3x3x128xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %353 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %354 = tosa.transpose %352, %353 : (tensor<1x14x14x256xf32>, tensor<4xi32>) -> tensor<1x256x14x14xf32>
    %355 = tosa.cast %arg52 : (tensor<256xf32>) -> tensor<256xf32>
    %356 = tosa.cast %arg53 : (tensor<256xf32>) -> tensor<256xf32>
    %357 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<256xf32>}> : () -> tensor<256xf32>
    %358 = tosa.add %356, %357 : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %359 = math.sqrt %358 : tensor<256xf32>
    %360 = tosa.reciprocal %359 : (tensor<256xf32>) -> tensor<256xf32>
    %361 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<256xf32>}> : () -> tensor<256xf32>
    %362 = tosa.mul %360, %361 {shift = 0 : i8} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %363 = tosa.reshape %355 {new_shape = array<i64: 256, 1>} : (tensor<256xf32>) -> tensor<256x1xf32>
    %364 = tosa.reshape %363 {new_shape = array<i64: 256, 1, 1>} : (tensor<256x1xf32>) -> tensor<256x1x1xf32>
    %365 = tosa.reshape %362 {new_shape = array<i64: 256, 1>} : (tensor<256xf32>) -> tensor<256x1xf32>
    %366 = tosa.reshape %365 {new_shape = array<i64: 256, 1, 1>} : (tensor<256x1xf32>) -> tensor<256x1x1xf32>
    %367 = tosa.reshape %364 {new_shape = array<i64: 1, 256, 1, 1>} : (tensor<256x1x1xf32>) -> tensor<1x256x1x1xf32>
    %368 = tosa.sub %354, %367 : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %369 = tosa.reshape %366 {new_shape = array<i64: 1, 256, 1, 1>} : (tensor<256x1x1xf32>) -> tensor<1x256x1x1xf32>
    %370 = tosa.mul %368, %369 {shift = 0 : i8} : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %371 = tosa.reshape %arg54 {new_shape = array<i64: 256, 1>} : (tensor<256xf32>) -> tensor<256x1xf32>
    %372 = tosa.reshape %371 {new_shape = array<i64: 256, 1, 1>} : (tensor<256x1xf32>) -> tensor<256x1x1xf32>
    %373 = tosa.reshape %372 {new_shape = array<i64: 1, 256, 1, 1>} : (tensor<256x1x1xf32>) -> tensor<1x256x1x1xf32>
    %374 = tosa.mul %370, %373 {shift = 0 : i8} : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %375 = tosa.reshape %arg55 {new_shape = array<i64: 256, 1>} : (tensor<256xf32>) -> tensor<256x1xf32>
    %376 = tosa.reshape %375 {new_shape = array<i64: 256, 1, 1>} : (tensor<256x1xf32>) -> tensor<256x1x1xf32>
    %377 = tosa.reshape %376 {new_shape = array<i64: 1, 256, 1, 1>} : (tensor<256x1x1xf32>) -> tensor<1x256x1x1xf32>
    %378 = tosa.add %374, %377 : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %379 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x256x14x14xf32>}> : () -> tensor<1x256x14x14xf32>
    %380 = tosa.maximum %378, %379 : (tensor<1x256x14x14xf32>, tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %381 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<256xf32>}> : () -> tensor<256xf32>
    %382 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %383 = tosa.transpose %380, %382 : (tensor<1x256x14x14xf32>, tensor<4xi32>) -> tensor<1x14x14x256xf32>
    %384 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %385 = tosa.transpose %arg56, %384 : (tensor<256x256x3x3xf32>, tensor<4xi32>) -> tensor<256x3x3x256xf32>
    %386 = tosa.conv2d %383, %385, %381 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x14x14x256xf32>, tensor<256x3x3x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %387 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %388 = tosa.transpose %386, %387 : (tensor<1x14x14x256xf32>, tensor<4xi32>) -> tensor<1x256x14x14xf32>
    %389 = tosa.cast %arg57 : (tensor<256xf32>) -> tensor<256xf32>
    %390 = tosa.cast %arg58 : (tensor<256xf32>) -> tensor<256xf32>
    %391 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<256xf32>}> : () -> tensor<256xf32>
    %392 = tosa.add %390, %391 : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %393 = math.sqrt %392 : tensor<256xf32>
    %394 = tosa.reciprocal %393 : (tensor<256xf32>) -> tensor<256xf32>
    %395 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<256xf32>}> : () -> tensor<256xf32>
    %396 = tosa.mul %394, %395 {shift = 0 : i8} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %397 = tosa.reshape %389 {new_shape = array<i64: 256, 1>} : (tensor<256xf32>) -> tensor<256x1xf32>
    %398 = tosa.reshape %397 {new_shape = array<i64: 256, 1, 1>} : (tensor<256x1xf32>) -> tensor<256x1x1xf32>
    %399 = tosa.reshape %396 {new_shape = array<i64: 256, 1>} : (tensor<256xf32>) -> tensor<256x1xf32>
    %400 = tosa.reshape %399 {new_shape = array<i64: 256, 1, 1>} : (tensor<256x1xf32>) -> tensor<256x1x1xf32>
    %401 = tosa.reshape %398 {new_shape = array<i64: 1, 256, 1, 1>} : (tensor<256x1x1xf32>) -> tensor<1x256x1x1xf32>
    %402 = tosa.sub %388, %401 : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %403 = tosa.reshape %400 {new_shape = array<i64: 1, 256, 1, 1>} : (tensor<256x1x1xf32>) -> tensor<1x256x1x1xf32>
    %404 = tosa.mul %402, %403 {shift = 0 : i8} : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %405 = tosa.reshape %arg59 {new_shape = array<i64: 256, 1>} : (tensor<256xf32>) -> tensor<256x1xf32>
    %406 = tosa.reshape %405 {new_shape = array<i64: 256, 1, 1>} : (tensor<256x1xf32>) -> tensor<256x1x1xf32>
    %407 = tosa.reshape %406 {new_shape = array<i64: 1, 256, 1, 1>} : (tensor<256x1x1xf32>) -> tensor<1x256x1x1xf32>
    %408 = tosa.mul %404, %407 {shift = 0 : i8} : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %409 = tosa.reshape %arg60 {new_shape = array<i64: 256, 1>} : (tensor<256xf32>) -> tensor<256x1xf32>
    %410 = tosa.reshape %409 {new_shape = array<i64: 256, 1, 1>} : (tensor<256x1xf32>) -> tensor<256x1x1xf32>
    %411 = tosa.reshape %410 {new_shape = array<i64: 1, 256, 1, 1>} : (tensor<256x1x1xf32>) -> tensor<1x256x1x1xf32>
    %412 = tosa.add %408, %411 : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %413 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<256xf32>}> : () -> tensor<256xf32>
    %414 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %415 = tosa.transpose %346, %414 : (tensor<1x128x28x28xf32>, tensor<4xi32>) -> tensor<1x28x28x128xf32>
    %416 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %417 = tosa.transpose %arg61, %416 : (tensor<256x128x1x1xf32>, tensor<4xi32>) -> tensor<256x1x1x128xf32>
    %418 = tosa.conv2d %415, %417, %413 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x28x28x128xf32>, tensor<256x1x1x128xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %419 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %420 = tosa.transpose %418, %419 : (tensor<1x14x14x256xf32>, tensor<4xi32>) -> tensor<1x256x14x14xf32>
    %421 = tosa.cast %arg62 : (tensor<256xf32>) -> tensor<256xf32>
    %422 = tosa.cast %arg63 : (tensor<256xf32>) -> tensor<256xf32>
    %423 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<256xf32>}> : () -> tensor<256xf32>
    %424 = tosa.add %422, %423 : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %425 = math.sqrt %424 : tensor<256xf32>
    %426 = tosa.reciprocal %425 : (tensor<256xf32>) -> tensor<256xf32>
    %427 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<256xf32>}> : () -> tensor<256xf32>
    %428 = tosa.mul %426, %427 {shift = 0 : i8} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %429 = tosa.reshape %421 {new_shape = array<i64: 256, 1>} : (tensor<256xf32>) -> tensor<256x1xf32>
    %430 = tosa.reshape %429 {new_shape = array<i64: 256, 1, 1>} : (tensor<256x1xf32>) -> tensor<256x1x1xf32>
    %431 = tosa.reshape %428 {new_shape = array<i64: 256, 1>} : (tensor<256xf32>) -> tensor<256x1xf32>
    %432 = tosa.reshape %431 {new_shape = array<i64: 256, 1, 1>} : (tensor<256x1xf32>) -> tensor<256x1x1xf32>
    %433 = tosa.reshape %430 {new_shape = array<i64: 1, 256, 1, 1>} : (tensor<256x1x1xf32>) -> tensor<1x256x1x1xf32>
    %434 = tosa.sub %420, %433 : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %435 = tosa.reshape %432 {new_shape = array<i64: 1, 256, 1, 1>} : (tensor<256x1x1xf32>) -> tensor<1x256x1x1xf32>
    %436 = tosa.mul %434, %435 {shift = 0 : i8} : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %437 = tosa.reshape %arg64 {new_shape = array<i64: 256, 1>} : (tensor<256xf32>) -> tensor<256x1xf32>
    %438 = tosa.reshape %437 {new_shape = array<i64: 256, 1, 1>} : (tensor<256x1xf32>) -> tensor<256x1x1xf32>
    %439 = tosa.reshape %438 {new_shape = array<i64: 1, 256, 1, 1>} : (tensor<256x1x1xf32>) -> tensor<1x256x1x1xf32>
    %440 = tosa.mul %436, %439 {shift = 0 : i8} : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %441 = tosa.reshape %arg65 {new_shape = array<i64: 256, 1>} : (tensor<256xf32>) -> tensor<256x1xf32>
    %442 = tosa.reshape %441 {new_shape = array<i64: 256, 1, 1>} : (tensor<256x1xf32>) -> tensor<256x1x1xf32>
    %443 = tosa.reshape %442 {new_shape = array<i64: 1, 256, 1, 1>} : (tensor<256x1x1xf32>) -> tensor<1x256x1x1xf32>
    %444 = tosa.add %440, %443 : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %445 = tosa.add %412, %444 : (tensor<1x256x14x14xf32>, tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %446 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x256x14x14xf32>}> : () -> tensor<1x256x14x14xf32>
    %447 = tosa.maximum %445, %446 : (tensor<1x256x14x14xf32>, tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %448 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<256xf32>}> : () -> tensor<256xf32>
    %449 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %450 = tosa.transpose %447, %449 : (tensor<1x256x14x14xf32>, tensor<4xi32>) -> tensor<1x14x14x256xf32>
    %451 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %452 = tosa.transpose %arg66, %451 : (tensor<256x256x3x3xf32>, tensor<4xi32>) -> tensor<256x3x3x256xf32>
    %453 = tosa.conv2d %450, %452, %448 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x14x14x256xf32>, tensor<256x3x3x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %454 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %455 = tosa.transpose %453, %454 : (tensor<1x14x14x256xf32>, tensor<4xi32>) -> tensor<1x256x14x14xf32>
    %456 = tosa.cast %arg67 : (tensor<256xf32>) -> tensor<256xf32>
    %457 = tosa.cast %arg68 : (tensor<256xf32>) -> tensor<256xf32>
    %458 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<256xf32>}> : () -> tensor<256xf32>
    %459 = tosa.add %457, %458 : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %460 = math.sqrt %459 : tensor<256xf32>
    %461 = tosa.reciprocal %460 : (tensor<256xf32>) -> tensor<256xf32>
    %462 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<256xf32>}> : () -> tensor<256xf32>
    %463 = tosa.mul %461, %462 {shift = 0 : i8} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %464 = tosa.reshape %456 {new_shape = array<i64: 256, 1>} : (tensor<256xf32>) -> tensor<256x1xf32>
    %465 = tosa.reshape %464 {new_shape = array<i64: 256, 1, 1>} : (tensor<256x1xf32>) -> tensor<256x1x1xf32>
    %466 = tosa.reshape %463 {new_shape = array<i64: 256, 1>} : (tensor<256xf32>) -> tensor<256x1xf32>
    %467 = tosa.reshape %466 {new_shape = array<i64: 256, 1, 1>} : (tensor<256x1xf32>) -> tensor<256x1x1xf32>
    %468 = tosa.reshape %465 {new_shape = array<i64: 1, 256, 1, 1>} : (tensor<256x1x1xf32>) -> tensor<1x256x1x1xf32>
    %469 = tosa.sub %455, %468 : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %470 = tosa.reshape %467 {new_shape = array<i64: 1, 256, 1, 1>} : (tensor<256x1x1xf32>) -> tensor<1x256x1x1xf32>
    %471 = tosa.mul %469, %470 {shift = 0 : i8} : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %472 = tosa.reshape %arg69 {new_shape = array<i64: 256, 1>} : (tensor<256xf32>) -> tensor<256x1xf32>
    %473 = tosa.reshape %472 {new_shape = array<i64: 256, 1, 1>} : (tensor<256x1xf32>) -> tensor<256x1x1xf32>
    %474 = tosa.reshape %473 {new_shape = array<i64: 1, 256, 1, 1>} : (tensor<256x1x1xf32>) -> tensor<1x256x1x1xf32>
    %475 = tosa.mul %471, %474 {shift = 0 : i8} : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %476 = tosa.reshape %arg70 {new_shape = array<i64: 256, 1>} : (tensor<256xf32>) -> tensor<256x1xf32>
    %477 = tosa.reshape %476 {new_shape = array<i64: 256, 1, 1>} : (tensor<256x1xf32>) -> tensor<256x1x1xf32>
    %478 = tosa.reshape %477 {new_shape = array<i64: 1, 256, 1, 1>} : (tensor<256x1x1xf32>) -> tensor<1x256x1x1xf32>
    %479 = tosa.add %475, %478 : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %480 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x256x14x14xf32>}> : () -> tensor<1x256x14x14xf32>
    %481 = tosa.maximum %479, %480 : (tensor<1x256x14x14xf32>, tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %482 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<256xf32>}> : () -> tensor<256xf32>
    %483 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %484 = tosa.transpose %481, %483 : (tensor<1x256x14x14xf32>, tensor<4xi32>) -> tensor<1x14x14x256xf32>
    %485 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %486 = tosa.transpose %arg71, %485 : (tensor<256x256x3x3xf32>, tensor<4xi32>) -> tensor<256x3x3x256xf32>
    %487 = tosa.conv2d %484, %486, %482 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x14x14x256xf32>, tensor<256x3x3x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %488 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %489 = tosa.transpose %487, %488 : (tensor<1x14x14x256xf32>, tensor<4xi32>) -> tensor<1x256x14x14xf32>
    %490 = tosa.cast %arg72 : (tensor<256xf32>) -> tensor<256xf32>
    %491 = tosa.cast %arg73 : (tensor<256xf32>) -> tensor<256xf32>
    %492 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<256xf32>}> : () -> tensor<256xf32>
    %493 = tosa.add %491, %492 : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %494 = math.sqrt %493 : tensor<256xf32>
    %495 = tosa.reciprocal %494 : (tensor<256xf32>) -> tensor<256xf32>
    %496 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<256xf32>}> : () -> tensor<256xf32>
    %497 = tosa.mul %495, %496 {shift = 0 : i8} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %498 = tosa.reshape %490 {new_shape = array<i64: 256, 1>} : (tensor<256xf32>) -> tensor<256x1xf32>
    %499 = tosa.reshape %498 {new_shape = array<i64: 256, 1, 1>} : (tensor<256x1xf32>) -> tensor<256x1x1xf32>
    %500 = tosa.reshape %497 {new_shape = array<i64: 256, 1>} : (tensor<256xf32>) -> tensor<256x1xf32>
    %501 = tosa.reshape %500 {new_shape = array<i64: 256, 1, 1>} : (tensor<256x1xf32>) -> tensor<256x1x1xf32>
    %502 = tosa.reshape %499 {new_shape = array<i64: 1, 256, 1, 1>} : (tensor<256x1x1xf32>) -> tensor<1x256x1x1xf32>
    %503 = tosa.sub %489, %502 : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %504 = tosa.reshape %501 {new_shape = array<i64: 1, 256, 1, 1>} : (tensor<256x1x1xf32>) -> tensor<1x256x1x1xf32>
    %505 = tosa.mul %503, %504 {shift = 0 : i8} : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %506 = tosa.reshape %arg74 {new_shape = array<i64: 256, 1>} : (tensor<256xf32>) -> tensor<256x1xf32>
    %507 = tosa.reshape %506 {new_shape = array<i64: 256, 1, 1>} : (tensor<256x1xf32>) -> tensor<256x1x1xf32>
    %508 = tosa.reshape %507 {new_shape = array<i64: 1, 256, 1, 1>} : (tensor<256x1x1xf32>) -> tensor<1x256x1x1xf32>
    %509 = tosa.mul %505, %508 {shift = 0 : i8} : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %510 = tosa.reshape %arg75 {new_shape = array<i64: 256, 1>} : (tensor<256xf32>) -> tensor<256x1xf32>
    %511 = tosa.reshape %510 {new_shape = array<i64: 256, 1, 1>} : (tensor<256x1xf32>) -> tensor<256x1x1xf32>
    %512 = tosa.reshape %511 {new_shape = array<i64: 1, 256, 1, 1>} : (tensor<256x1x1xf32>) -> tensor<1x256x1x1xf32>
    %513 = tosa.add %509, %512 : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %514 = tosa.add %513, %447 : (tensor<1x256x14x14xf32>, tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %515 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x256x14x14xf32>}> : () -> tensor<1x256x14x14xf32>
    %516 = tosa.maximum %514, %515 : (tensor<1x256x14x14xf32>, tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %517 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
    %518 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %519 = tosa.transpose %516, %518 : (tensor<1x256x14x14xf32>, tensor<4xi32>) -> tensor<1x14x14x256xf32>
    %520 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %521 = tosa.transpose %arg76, %520 : (tensor<512x256x3x3xf32>, tensor<4xi32>) -> tensor<512x3x3x256xf32>
    %522 = tosa.conv2d %519, %521, %517 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>} : (tensor<1x14x14x256xf32>, tensor<512x3x3x256xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
    %523 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %524 = tosa.transpose %522, %523 : (tensor<1x7x7x512xf32>, tensor<4xi32>) -> tensor<1x512x7x7xf32>
    %525 = tosa.cast %arg77 : (tensor<512xf32>) -> tensor<512xf32>
    %526 = tosa.cast %arg78 : (tensor<512xf32>) -> tensor<512xf32>
    %527 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<512xf32>}> : () -> tensor<512xf32>
    %528 = tosa.add %526, %527 : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %529 = math.sqrt %528 : tensor<512xf32>
    %530 = tosa.reciprocal %529 : (tensor<512xf32>) -> tensor<512xf32>
    %531 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
    %532 = tosa.mul %530, %531 {shift = 0 : i8} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %533 = tosa.reshape %525 {new_shape = array<i64: 512, 1>} : (tensor<512xf32>) -> tensor<512x1xf32>
    %534 = tosa.reshape %533 {new_shape = array<i64: 512, 1, 1>} : (tensor<512x1xf32>) -> tensor<512x1x1xf32>
    %535 = tosa.reshape %532 {new_shape = array<i64: 512, 1>} : (tensor<512xf32>) -> tensor<512x1xf32>
    %536 = tosa.reshape %535 {new_shape = array<i64: 512, 1, 1>} : (tensor<512x1xf32>) -> tensor<512x1x1xf32>
    %537 = tosa.reshape %534 {new_shape = array<i64: 1, 512, 1, 1>} : (tensor<512x1x1xf32>) -> tensor<1x512x1x1xf32>
    %538 = tosa.sub %524, %537 : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %539 = tosa.reshape %536 {new_shape = array<i64: 1, 512, 1, 1>} : (tensor<512x1x1xf32>) -> tensor<1x512x1x1xf32>
    %540 = tosa.mul %538, %539 {shift = 0 : i8} : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %541 = tosa.reshape %arg79 {new_shape = array<i64: 512, 1>} : (tensor<512xf32>) -> tensor<512x1xf32>
    %542 = tosa.reshape %541 {new_shape = array<i64: 512, 1, 1>} : (tensor<512x1xf32>) -> tensor<512x1x1xf32>
    %543 = tosa.reshape %542 {new_shape = array<i64: 1, 512, 1, 1>} : (tensor<512x1x1xf32>) -> tensor<1x512x1x1xf32>
    %544 = tosa.mul %540, %543 {shift = 0 : i8} : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %545 = tosa.reshape %arg80 {new_shape = array<i64: 512, 1>} : (tensor<512xf32>) -> tensor<512x1xf32>
    %546 = tosa.reshape %545 {new_shape = array<i64: 512, 1, 1>} : (tensor<512x1xf32>) -> tensor<512x1x1xf32>
    %547 = tosa.reshape %546 {new_shape = array<i64: 1, 512, 1, 1>} : (tensor<512x1x1xf32>) -> tensor<1x512x1x1xf32>
    %548 = tosa.add %544, %547 : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %549 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x512x7x7xf32>}> : () -> tensor<1x512x7x7xf32>
    %550 = tosa.maximum %548, %549 : (tensor<1x512x7x7xf32>, tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %551 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
    %552 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %553 = tosa.transpose %550, %552 : (tensor<1x512x7x7xf32>, tensor<4xi32>) -> tensor<1x7x7x512xf32>
    %554 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %555 = tosa.transpose %arg81, %554 : (tensor<512x512x3x3xf32>, tensor<4xi32>) -> tensor<512x3x3x512xf32>
    %556 = tosa.conv2d %553, %555, %551 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x7x7x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
    %557 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %558 = tosa.transpose %556, %557 : (tensor<1x7x7x512xf32>, tensor<4xi32>) -> tensor<1x512x7x7xf32>
    %559 = tosa.cast %arg82 : (tensor<512xf32>) -> tensor<512xf32>
    %560 = tosa.cast %arg83 : (tensor<512xf32>) -> tensor<512xf32>
    %561 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<512xf32>}> : () -> tensor<512xf32>
    %562 = tosa.add %560, %561 : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %563 = math.sqrt %562 : tensor<512xf32>
    %564 = tosa.reciprocal %563 : (tensor<512xf32>) -> tensor<512xf32>
    %565 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
    %566 = tosa.mul %564, %565 {shift = 0 : i8} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %567 = tosa.reshape %559 {new_shape = array<i64: 512, 1>} : (tensor<512xf32>) -> tensor<512x1xf32>
    %568 = tosa.reshape %567 {new_shape = array<i64: 512, 1, 1>} : (tensor<512x1xf32>) -> tensor<512x1x1xf32>
    %569 = tosa.reshape %566 {new_shape = array<i64: 512, 1>} : (tensor<512xf32>) -> tensor<512x1xf32>
    %570 = tosa.reshape %569 {new_shape = array<i64: 512, 1, 1>} : (tensor<512x1xf32>) -> tensor<512x1x1xf32>
    %571 = tosa.reshape %568 {new_shape = array<i64: 1, 512, 1, 1>} : (tensor<512x1x1xf32>) -> tensor<1x512x1x1xf32>
    %572 = tosa.sub %558, %571 : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %573 = tosa.reshape %570 {new_shape = array<i64: 1, 512, 1, 1>} : (tensor<512x1x1xf32>) -> tensor<1x512x1x1xf32>
    %574 = tosa.mul %572, %573 {shift = 0 : i8} : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %575 = tosa.reshape %arg84 {new_shape = array<i64: 512, 1>} : (tensor<512xf32>) -> tensor<512x1xf32>
    %576 = tosa.reshape %575 {new_shape = array<i64: 512, 1, 1>} : (tensor<512x1xf32>) -> tensor<512x1x1xf32>
    %577 = tosa.reshape %576 {new_shape = array<i64: 1, 512, 1, 1>} : (tensor<512x1x1xf32>) -> tensor<1x512x1x1xf32>
    %578 = tosa.mul %574, %577 {shift = 0 : i8} : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %579 = tosa.reshape %arg85 {new_shape = array<i64: 512, 1>} : (tensor<512xf32>) -> tensor<512x1xf32>
    %580 = tosa.reshape %579 {new_shape = array<i64: 512, 1, 1>} : (tensor<512x1xf32>) -> tensor<512x1x1xf32>
    %581 = tosa.reshape %580 {new_shape = array<i64: 1, 512, 1, 1>} : (tensor<512x1x1xf32>) -> tensor<1x512x1x1xf32>
    %582 = tosa.add %578, %581 : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %583 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
    %584 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %585 = tosa.transpose %516, %584 : (tensor<1x256x14x14xf32>, tensor<4xi32>) -> tensor<1x14x14x256xf32>
    %586 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %587 = tosa.transpose %arg86, %586 : (tensor<512x256x1x1xf32>, tensor<4xi32>) -> tensor<512x1x1x256xf32>
    %588 = tosa.conv2d %585, %587, %583 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x14x14x256xf32>, tensor<512x1x1x256xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
    %589 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %590 = tosa.transpose %588, %589 : (tensor<1x7x7x512xf32>, tensor<4xi32>) -> tensor<1x512x7x7xf32>
    %591 = tosa.cast %arg87 : (tensor<512xf32>) -> tensor<512xf32>
    %592 = tosa.cast %arg88 : (tensor<512xf32>) -> tensor<512xf32>
    %593 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<512xf32>}> : () -> tensor<512xf32>
    %594 = tosa.add %592, %593 : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %595 = math.sqrt %594 : tensor<512xf32>
    %596 = tosa.reciprocal %595 : (tensor<512xf32>) -> tensor<512xf32>
    %597 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
    %598 = tosa.mul %596, %597 {shift = 0 : i8} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %599 = tosa.reshape %591 {new_shape = array<i64: 512, 1>} : (tensor<512xf32>) -> tensor<512x1xf32>
    %600 = tosa.reshape %599 {new_shape = array<i64: 512, 1, 1>} : (tensor<512x1xf32>) -> tensor<512x1x1xf32>
    %601 = tosa.reshape %598 {new_shape = array<i64: 512, 1>} : (tensor<512xf32>) -> tensor<512x1xf32>
    %602 = tosa.reshape %601 {new_shape = array<i64: 512, 1, 1>} : (tensor<512x1xf32>) -> tensor<512x1x1xf32>
    %603 = tosa.reshape %600 {new_shape = array<i64: 1, 512, 1, 1>} : (tensor<512x1x1xf32>) -> tensor<1x512x1x1xf32>
    %604 = tosa.sub %590, %603 : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %605 = tosa.reshape %602 {new_shape = array<i64: 1, 512, 1, 1>} : (tensor<512x1x1xf32>) -> tensor<1x512x1x1xf32>
    %606 = tosa.mul %604, %605 {shift = 0 : i8} : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %607 = tosa.reshape %arg89 {new_shape = array<i64: 512, 1>} : (tensor<512xf32>) -> tensor<512x1xf32>
    %608 = tosa.reshape %607 {new_shape = array<i64: 512, 1, 1>} : (tensor<512x1xf32>) -> tensor<512x1x1xf32>
    %609 = tosa.reshape %608 {new_shape = array<i64: 1, 512, 1, 1>} : (tensor<512x1x1xf32>) -> tensor<1x512x1x1xf32>
    %610 = tosa.mul %606, %609 {shift = 0 : i8} : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %611 = tosa.reshape %arg90 {new_shape = array<i64: 512, 1>} : (tensor<512xf32>) -> tensor<512x1xf32>
    %612 = tosa.reshape %611 {new_shape = array<i64: 512, 1, 1>} : (tensor<512x1xf32>) -> tensor<512x1x1xf32>
    %613 = tosa.reshape %612 {new_shape = array<i64: 1, 512, 1, 1>} : (tensor<512x1x1xf32>) -> tensor<1x512x1x1xf32>
    %614 = tosa.add %610, %613 : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %615 = tosa.add %582, %614 : (tensor<1x512x7x7xf32>, tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %616 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x512x7x7xf32>}> : () -> tensor<1x512x7x7xf32>
    %617 = tosa.maximum %615, %616 : (tensor<1x512x7x7xf32>, tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %618 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
    %619 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %620 = tosa.transpose %617, %619 : (tensor<1x512x7x7xf32>, tensor<4xi32>) -> tensor<1x7x7x512xf32>
    %621 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %622 = tosa.transpose %arg91, %621 : (tensor<512x512x3x3xf32>, tensor<4xi32>) -> tensor<512x3x3x512xf32>
    %623 = tosa.conv2d %620, %622, %618 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x7x7x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
    %624 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %625 = tosa.transpose %623, %624 : (tensor<1x7x7x512xf32>, tensor<4xi32>) -> tensor<1x512x7x7xf32>
    %626 = tosa.cast %arg92 : (tensor<512xf32>) -> tensor<512xf32>
    %627 = tosa.cast %arg93 : (tensor<512xf32>) -> tensor<512xf32>
    %628 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<512xf32>}> : () -> tensor<512xf32>
    %629 = tosa.add %627, %628 : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %630 = math.sqrt %629 : tensor<512xf32>
    %631 = tosa.reciprocal %630 : (tensor<512xf32>) -> tensor<512xf32>
    %632 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
    %633 = tosa.mul %631, %632 {shift = 0 : i8} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %634 = tosa.reshape %626 {new_shape = array<i64: 512, 1>} : (tensor<512xf32>) -> tensor<512x1xf32>
    %635 = tosa.reshape %634 {new_shape = array<i64: 512, 1, 1>} : (tensor<512x1xf32>) -> tensor<512x1x1xf32>
    %636 = tosa.reshape %633 {new_shape = array<i64: 512, 1>} : (tensor<512xf32>) -> tensor<512x1xf32>
    %637 = tosa.reshape %636 {new_shape = array<i64: 512, 1, 1>} : (tensor<512x1xf32>) -> tensor<512x1x1xf32>
    %638 = tosa.reshape %635 {new_shape = array<i64: 1, 512, 1, 1>} : (tensor<512x1x1xf32>) -> tensor<1x512x1x1xf32>
    %639 = tosa.sub %625, %638 : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %640 = tosa.reshape %637 {new_shape = array<i64: 1, 512, 1, 1>} : (tensor<512x1x1xf32>) -> tensor<1x512x1x1xf32>
    %641 = tosa.mul %639, %640 {shift = 0 : i8} : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %642 = tosa.reshape %arg94 {new_shape = array<i64: 512, 1>} : (tensor<512xf32>) -> tensor<512x1xf32>
    %643 = tosa.reshape %642 {new_shape = array<i64: 512, 1, 1>} : (tensor<512x1xf32>) -> tensor<512x1x1xf32>
    %644 = tosa.reshape %643 {new_shape = array<i64: 1, 512, 1, 1>} : (tensor<512x1x1xf32>) -> tensor<1x512x1x1xf32>
    %645 = tosa.mul %641, %644 {shift = 0 : i8} : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %646 = tosa.reshape %arg95 {new_shape = array<i64: 512, 1>} : (tensor<512xf32>) -> tensor<512x1xf32>
    %647 = tosa.reshape %646 {new_shape = array<i64: 512, 1, 1>} : (tensor<512x1xf32>) -> tensor<512x1x1xf32>
    %648 = tosa.reshape %647 {new_shape = array<i64: 1, 512, 1, 1>} : (tensor<512x1x1xf32>) -> tensor<1x512x1x1xf32>
    %649 = tosa.add %645, %648 : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %650 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x512x7x7xf32>}> : () -> tensor<1x512x7x7xf32>
    %651 = tosa.maximum %649, %650 : (tensor<1x512x7x7xf32>, tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %652 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
    %653 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %654 = tosa.transpose %651, %653 : (tensor<1x512x7x7xf32>, tensor<4xi32>) -> tensor<1x7x7x512xf32>
    %655 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %656 = tosa.transpose %arg96, %655 : (tensor<512x512x3x3xf32>, tensor<4xi32>) -> tensor<512x3x3x512xf32>
    %657 = tosa.conv2d %654, %656, %652 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x7x7x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
    %658 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %659 = tosa.transpose %657, %658 : (tensor<1x7x7x512xf32>, tensor<4xi32>) -> tensor<1x512x7x7xf32>
    %660 = tosa.cast %arg97 : (tensor<512xf32>) -> tensor<512xf32>
    %661 = tosa.cast %arg98 : (tensor<512xf32>) -> tensor<512xf32>
    %662 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<512xf32>}> : () -> tensor<512xf32>
    %663 = tosa.add %661, %662 : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %664 = math.sqrt %663 : tensor<512xf32>
    %665 = tosa.reciprocal %664 : (tensor<512xf32>) -> tensor<512xf32>
    %666 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
    %667 = tosa.mul %665, %666 {shift = 0 : i8} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %668 = tosa.reshape %660 {new_shape = array<i64: 512, 1>} : (tensor<512xf32>) -> tensor<512x1xf32>
    %669 = tosa.reshape %668 {new_shape = array<i64: 512, 1, 1>} : (tensor<512x1xf32>) -> tensor<512x1x1xf32>
    %670 = tosa.reshape %667 {new_shape = array<i64: 512, 1>} : (tensor<512xf32>) -> tensor<512x1xf32>
    %671 = tosa.reshape %670 {new_shape = array<i64: 512, 1, 1>} : (tensor<512x1xf32>) -> tensor<512x1x1xf32>
    %672 = tosa.reshape %669 {new_shape = array<i64: 1, 512, 1, 1>} : (tensor<512x1x1xf32>) -> tensor<1x512x1x1xf32>
    %673 = tosa.sub %659, %672 : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %674 = tosa.reshape %671 {new_shape = array<i64: 1, 512, 1, 1>} : (tensor<512x1x1xf32>) -> tensor<1x512x1x1xf32>
    %675 = tosa.mul %673, %674 {shift = 0 : i8} : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %676 = tosa.reshape %arg99 {new_shape = array<i64: 512, 1>} : (tensor<512xf32>) -> tensor<512x1xf32>
    %677 = tosa.reshape %676 {new_shape = array<i64: 512, 1, 1>} : (tensor<512x1xf32>) -> tensor<512x1x1xf32>
    %678 = tosa.reshape %677 {new_shape = array<i64: 1, 512, 1, 1>} : (tensor<512x1x1xf32>) -> tensor<1x512x1x1xf32>
    %679 = tosa.mul %675, %678 {shift = 0 : i8} : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %680 = tosa.reshape %arg100 {new_shape = array<i64: 512, 1>} : (tensor<512xf32>) -> tensor<512x1xf32>
    %681 = tosa.reshape %680 {new_shape = array<i64: 512, 1, 1>} : (tensor<512x1xf32>) -> tensor<512x1x1xf32>
    %682 = tosa.reshape %681 {new_shape = array<i64: 1, 512, 1, 1>} : (tensor<512x1x1xf32>) -> tensor<1x512x1x1xf32>
    %683 = tosa.add %679, %682 : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %684 = tosa.add %683, %617 : (tensor<1x512x7x7xf32>, tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %685 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x512x7x7xf32>}> : () -> tensor<1x512x7x7xf32>
    %686 = tosa.maximum %684, %685 : (tensor<1x512x7x7xf32>, tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %687 = tosa.reduce_sum %686 {axis = 3 : i32} : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x1xf32>
    %688 = tosa.reduce_sum %687 {axis = 2 : i32} : (tensor<1x512x7x1xf32>) -> tensor<1x512x1x1xf32>
    %689 = "tosa.const"() <{value = dense<4.900000e+01> : tensor<1xf32>}> : () -> tensor<1xf32>
    %690 = tosa.reciprocal %689 : (tensor<1xf32>) -> tensor<1xf32>
    %691 = tosa.mul %690, %688 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x1x1xf32>
    %692 = tosa.reshape %691 {new_shape = array<i64: 1, 512>} : (tensor<1x512x1x1xf32>) -> tensor<1x512xf32>
    %693 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %694 = tosa.transpose %arg101, %693 : (tensor<10x512xf32>, tensor<2xi32>) -> tensor<512x10xf32>
    %695 = tosa.reshape %692 {new_shape = array<i64: 1, 1, 512>} : (tensor<1x512xf32>) -> tensor<1x1x512xf32>
    %696 = tosa.reshape %694 {new_shape = array<i64: 1, 512, 10>} : (tensor<512x10xf32>) -> tensor<1x512x10xf32>
    %697 = tosa.matmul %695, %696 : (tensor<1x1x512xf32>, tensor<1x512x10xf32>) -> tensor<1x1x10xf32>
    %698 = tosa.reshape %697 {new_shape = array<i64: 1, 10>} : (tensor<1x1x10xf32>) -> tensor<1x10xf32>
    %699 = tosa.reshape %arg102 {new_shape = array<i64: 1, 10>} : (tensor<10xf32>) -> tensor<1x10xf32>
    %700 = tosa.add %699, %698 : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    return %700 : tensor<1x10xf32>
  }
}

