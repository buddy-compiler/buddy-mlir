#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
"builtin.module"() ({
  "func.func"() <{function_type = (tensor<1x4x64x64xf32>, tensor<4x4x1x1xf32>, tensor<4xf32>, tensor<512x4x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x128x3x3xf32>, tensor<3xf32>) -> tensor<1x3x512x512xf32>, sym_name = "subgraph0"}> ({
  ^bb0(%arg0: tensor<1x4x64x64xf32>, %arg1: tensor<4x4x1x1xf32>, %arg2: tensor<4xf32>, %arg3: tensor<512x4x3x3xf32>, %arg4: tensor<512xf32>, %arg5: tensor<512xf32>, %arg6: tensor<512xf32>, %arg7: tensor<512x512x3x3xf32>, %arg8: tensor<512xf32>, %arg9: tensor<512xf32>, %arg10: tensor<512xf32>, %arg11: tensor<512x512x3x3xf32>, %arg12: tensor<512xf32>, %arg13: tensor<512xf32>, %arg14: tensor<512xf32>, %arg15: tensor<512x512xf32>, %arg16: tensor<512xf32>, %arg17: tensor<512x512xf32>, %arg18: tensor<512xf32>, %arg19: tensor<512x512xf32>, %arg20: tensor<512xf32>, %arg21: tensor<512x512xf32>, %arg22: tensor<512xf32>, %arg23: tensor<512xf32>, %arg24: tensor<512xf32>, %arg25: tensor<512x512x3x3xf32>, %arg26: tensor<512xf32>, %arg27: tensor<512xf32>, %arg28: tensor<512xf32>, %arg29: tensor<512x512x3x3xf32>, %arg30: tensor<512xf32>, %arg31: tensor<512xf32>, %arg32: tensor<512xf32>, %arg33: tensor<512x512x3x3xf32>, %arg34: tensor<512xf32>, %arg35: tensor<512xf32>, %arg36: tensor<512xf32>, %arg37: tensor<512x512x3x3xf32>, %arg38: tensor<512xf32>, %arg39: tensor<512xf32>, %arg40: tensor<512xf32>, %arg41: tensor<512x512x3x3xf32>, %arg42: tensor<512xf32>, %arg43: tensor<512xf32>, %arg44: tensor<512xf32>, %arg45: tensor<512x512x3x3xf32>, %arg46: tensor<512xf32>, %arg47: tensor<512xf32>, %arg48: tensor<512xf32>, %arg49: tensor<512x512x3x3xf32>, %arg50: tensor<512xf32>, %arg51: tensor<512xf32>, %arg52: tensor<512xf32>, %arg53: tensor<512x512x3x3xf32>, %arg54: tensor<512xf32>, %arg55: tensor<512x512x3x3xf32>, %arg56: tensor<512xf32>, %arg57: tensor<512xf32>, %arg58: tensor<512xf32>, %arg59: tensor<512x512x3x3xf32>, %arg60: tensor<512xf32>, %arg61: tensor<512xf32>, %arg62: tensor<512xf32>, %arg63: tensor<512x512x3x3xf32>, %arg64: tensor<512xf32>, %arg65: tensor<512xf32>, %arg66: tensor<512xf32>, %arg67: tensor<512x512x3x3xf32>, %arg68: tensor<512xf32>, %arg69: tensor<512xf32>, %arg70: tensor<512xf32>, %arg71: tensor<512x512x3x3xf32>, %arg72: tensor<512xf32>, %arg73: tensor<512xf32>, %arg74: tensor<512xf32>, %arg75: tensor<512x512x3x3xf32>, %arg76: tensor<512xf32>, %arg77: tensor<512xf32>, %arg78: tensor<512xf32>, %arg79: tensor<512x512x3x3xf32>, %arg80: tensor<512xf32>, %arg81: tensor<512x512x3x3xf32>, %arg82: tensor<512xf32>, %arg83: tensor<512xf32>, %arg84: tensor<512xf32>, %arg85: tensor<256x512x3x3xf32>, %arg86: tensor<256xf32>, %arg87: tensor<256xf32>, %arg88: tensor<256xf32>, %arg89: tensor<256x256x3x3xf32>, %arg90: tensor<256xf32>, %arg91: tensor<256x512x1x1xf32>, %arg92: tensor<256xf32>, %arg93: tensor<256xf32>, %arg94: tensor<256xf32>, %arg95: tensor<256x256x3x3xf32>, %arg96: tensor<256xf32>, %arg97: tensor<256xf32>, %arg98: tensor<256xf32>, %arg99: tensor<256x256x3x3xf32>, %arg100: tensor<256xf32>, %arg101: tensor<256xf32>, %arg102: tensor<256xf32>, %arg103: tensor<256x256x3x3xf32>, %arg104: tensor<256xf32>, %arg105: tensor<256xf32>, %arg106: tensor<256xf32>, %arg107: tensor<256x256x3x3xf32>, %arg108: tensor<256xf32>, %arg109: tensor<256x256x3x3xf32>, %arg110: tensor<256xf32>, %arg111: tensor<256xf32>, %arg112: tensor<256xf32>, %arg113: tensor<128x256x3x3xf32>, %arg114: tensor<128xf32>, %arg115: tensor<128xf32>, %arg116: tensor<128xf32>, %arg117: tensor<128x128x3x3xf32>, %arg118: tensor<128xf32>, %arg119: tensor<128x256x1x1xf32>, %arg120: tensor<128xf32>, %arg121: tensor<128xf32>, %arg122: tensor<128xf32>, %arg123: tensor<128x128x3x3xf32>, %arg124: tensor<128xf32>, %arg125: tensor<128xf32>, %arg126: tensor<128xf32>, %arg127: tensor<128x128x3x3xf32>, %arg128: tensor<128xf32>, %arg129: tensor<128xf32>, %arg130: tensor<128xf32>, %arg131: tensor<128x128x3x3xf32>, %arg132: tensor<128xf32>, %arg133: tensor<128xf32>, %arg134: tensor<128xf32>, %arg135: tensor<128x128x3x3xf32>, %arg136: tensor<128xf32>, %arg137: tensor<128xf32>, %arg138: tensor<128xf32>, %arg139: tensor<3x128x3x3xf32>, %arg140: tensor<3xf32>):
    %0 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1 = "tosa.transpose"(%arg0, %0) : (tensor<1x4x64x64xf32>, tensor<4xi32>) -> tensor<1x64x64x4xf32>
    %2 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3 = "tosa.transpose"(%arg1, %2) : (tensor<4x4x1x1xf32>, tensor<4xi32>) -> tensor<4x1x1x4xf32>
    %4 = "tosa.conv2d"(%1, %3, %arg2) <{dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}> : (tensor<1x64x64x4xf32>, tensor<4x1x1x4xf32>, tensor<4xf32>) -> tensor<1x64x64x4xf32>
    %5 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6 = "tosa.transpose"(%4, %5) : (tensor<1x64x64x4xf32>, tensor<4xi32>) -> tensor<1x4x64x64xf32>
    %7 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %8 = "tosa.transpose"(%6, %7) : (tensor<1x4x64x64xf32>, tensor<4xi32>) -> tensor<1x64x64x4xf32>
    %9 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %10 = "tosa.transpose"(%arg3, %9) : (tensor<512x4x3x3xf32>, tensor<4xi32>) -> tensor<512x3x3x4xf32>
    %11 = "tosa.conv2d"(%8, %10, %arg4) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x64x64x4xf32>, tensor<512x3x3x4xf32>, tensor<512xf32>) -> tensor<1x64x64x512xf32>
    %12 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %13 = "tosa.transpose"(%11, %12) : (tensor<1x64x64x512xf32>, tensor<4xi32>) -> tensor<1x512x64x64xf32>
    %14 = "tosa.reshape"(%13) <{new_shape = array<i64: 1, 32, 16, 4096>}> : (tensor<1x512x64x64xf32>) -> tensor<1x32x16x4096xf32>
    %15 = "tosa.reduce_sum"(%14) <{axis = 2 : i32}> : (tensor<1x32x16x4096xf32>) -> tensor<1x32x1x4096xf32>
    %16 = "tosa.reduce_sum"(%15) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %17 = "tosa.const"() <{value = dense<6.553600e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %18 = "tosa.reciprocal"(%17) : (tensor<1xf32>) -> tensor<1xf32>
    %19 = "tosa.mul"(%18, %16) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %20 = "tosa.sub"(%14, %19) : (tensor<1x32x16x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x4096xf32>
    %21 = "tosa.mul"(%20, %20) <{shift = 0 : i8}> : (tensor<1x32x16x4096xf32>, tensor<1x32x16x4096xf32>) -> tensor<1x32x16x4096xf32>
    %22 = "tosa.reduce_sum"(%21) <{axis = 2 : i32}> : (tensor<1x32x16x4096xf32>) -> tensor<1x32x1x4096xf32>
    %23 = "tosa.reduce_sum"(%22) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %24 = "tosa.const"() <{value = dense<6.553600e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %25 = "tosa.reciprocal"(%24) : (tensor<1xf32>) -> tensor<1xf32>
    %26 = "tosa.mul"(%25, %23) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %27 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %28 = "tosa.add"(%26, %27) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %29 = "tosa.rsqrt"(%28) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %30 = "tosa.sub"(%14, %19) : (tensor<1x32x16x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x4096xf32>
    %31 = "tosa.mul"(%30, %29) <{shift = 0 : i8}> : (tensor<1x32x16x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x4096xf32>
    %32 = "tosa.reshape"(%31) <{new_shape = array<i64: 1, 512, 64, 64>}> : (tensor<1x32x16x4096xf32>) -> tensor<1x512x64x64xf32>
    %33 = "tosa.reshape"(%arg5) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %34 = "tosa.reshape"(%33) <{new_shape = array<i64: 1, 512, 1>}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
    %35 = "tosa.reshape"(%34) <{new_shape = array<i64: 1, 512, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1x1xf32>
    %36 = "tosa.reshape"(%arg6) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %37 = "tosa.reshape"(%36) <{new_shape = array<i64: 1, 512, 1>}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
    %38 = "tosa.reshape"(%37) <{new_shape = array<i64: 1, 512, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1x1xf32>
    %39 = "tosa.mul"(%32, %38) <{shift = 0 : i8}> : (tensor<1x512x64x64xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x64x64xf32>
    %40 = "tosa.add"(%39, %35) : (tensor<1x512x64x64xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x64x64xf32>
    %41 = "tosa.sigmoid"(%40) : (tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %42 = "tosa.mul"(%40, %41) <{shift = 0 : i8}> : (tensor<1x512x64x64xf32>, tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %43 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %44 = "tosa.transpose"(%42, %43) : (tensor<1x512x64x64xf32>, tensor<4xi32>) -> tensor<1x64x64x512xf32>
    %45 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %46 = "tosa.transpose"(%arg7, %45) : (tensor<512x512x3x3xf32>, tensor<4xi32>) -> tensor<512x3x3x512xf32>
    %47 = "tosa.conv2d"(%44, %46, %arg8) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x64x64x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>) -> tensor<1x64x64x512xf32>
    %48 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %49 = "tosa.transpose"(%47, %48) : (tensor<1x64x64x512xf32>, tensor<4xi32>) -> tensor<1x512x64x64xf32>
    %50 = "tosa.reshape"(%49) <{new_shape = array<i64: 1, 32, 16, 4096>}> : (tensor<1x512x64x64xf32>) -> tensor<1x32x16x4096xf32>
    %51 = "tosa.reduce_sum"(%50) <{axis = 2 : i32}> : (tensor<1x32x16x4096xf32>) -> tensor<1x32x1x4096xf32>
    %52 = "tosa.reduce_sum"(%51) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %53 = "tosa.const"() <{value = dense<6.553600e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %54 = "tosa.reciprocal"(%53) : (tensor<1xf32>) -> tensor<1xf32>
    %55 = "tosa.mul"(%54, %52) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %56 = "tosa.sub"(%50, %55) : (tensor<1x32x16x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x4096xf32>
    %57 = "tosa.mul"(%56, %56) <{shift = 0 : i8}> : (tensor<1x32x16x4096xf32>, tensor<1x32x16x4096xf32>) -> tensor<1x32x16x4096xf32>
    %58 = "tosa.reduce_sum"(%57) <{axis = 2 : i32}> : (tensor<1x32x16x4096xf32>) -> tensor<1x32x1x4096xf32>
    %59 = "tosa.reduce_sum"(%58) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %60 = "tosa.const"() <{value = dense<6.553600e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %61 = "tosa.reciprocal"(%60) : (tensor<1xf32>) -> tensor<1xf32>
    %62 = "tosa.mul"(%61, %59) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %63 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %64 = "tosa.add"(%62, %63) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %65 = "tosa.rsqrt"(%64) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %66 = "tosa.sub"(%50, %55) : (tensor<1x32x16x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x4096xf32>
    %67 = "tosa.mul"(%66, %65) <{shift = 0 : i8}> : (tensor<1x32x16x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x4096xf32>
    %68 = "tosa.reshape"(%67) <{new_shape = array<i64: 1, 512, 64, 64>}> : (tensor<1x32x16x4096xf32>) -> tensor<1x512x64x64xf32>
    %69 = "tosa.reshape"(%arg9) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %70 = "tosa.reshape"(%69) <{new_shape = array<i64: 1, 512, 1>}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
    %71 = "tosa.reshape"(%70) <{new_shape = array<i64: 1, 512, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1x1xf32>
    %72 = "tosa.reshape"(%arg10) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %73 = "tosa.reshape"(%72) <{new_shape = array<i64: 1, 512, 1>}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
    %74 = "tosa.reshape"(%73) <{new_shape = array<i64: 1, 512, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1x1xf32>
    %75 = "tosa.mul"(%68, %74) <{shift = 0 : i8}> : (tensor<1x512x64x64xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x64x64xf32>
    %76 = "tosa.add"(%75, %71) : (tensor<1x512x64x64xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x64x64xf32>
    %77 = "tosa.sigmoid"(%76) : (tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %78 = "tosa.mul"(%76, %77) <{shift = 0 : i8}> : (tensor<1x512x64x64xf32>, tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %79 = "tosa.identity"(%78) : (tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %80 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %81 = "tosa.transpose"(%79, %80) : (tensor<1x512x64x64xf32>, tensor<4xi32>) -> tensor<1x64x64x512xf32>
    %82 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %83 = "tosa.transpose"(%arg11, %82) : (tensor<512x512x3x3xf32>, tensor<4xi32>) -> tensor<512x3x3x512xf32>
    %84 = "tosa.conv2d"(%81, %83, %arg12) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x64x64x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>) -> tensor<1x64x64x512xf32>
    %85 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %86 = "tosa.transpose"(%84, %85) : (tensor<1x64x64x512xf32>, tensor<4xi32>) -> tensor<1x512x64x64xf32>
    %87 = "tosa.add"(%13, %86) : (tensor<1x512x64x64xf32>, tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %88 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x512x64x64xf32>}> : () -> tensor<1x512x64x64xf32>
    %89 = "tosa.reciprocal"(%88) : (tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %90 = "tosa.mul"(%87, %89) <{shift = 0 : i8}> : (tensor<1x512x64x64xf32>, tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %91 = "tosa.reshape"(%90) <{new_shape = array<i64: 1, 512, 4096>}> : (tensor<1x512x64x64xf32>) -> tensor<1x512x4096xf32>
    %92 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %93 = "tosa.transpose"(%91, %92) : (tensor<1x512x4096xf32>, tensor<3xi32>) -> tensor<1x4096x512xf32>
    %94 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %95 = "tosa.transpose"(%93, %94) : (tensor<1x4096x512xf32>, tensor<3xi32>) -> tensor<1x512x4096xf32>
    %96 = "tosa.reshape"(%95) <{new_shape = array<i64: 1, 32, 16, 4096>}> : (tensor<1x512x4096xf32>) -> tensor<1x32x16x4096xf32>
    %97 = "tosa.reduce_sum"(%96) <{axis = 2 : i32}> : (tensor<1x32x16x4096xf32>) -> tensor<1x32x1x4096xf32>
    %98 = "tosa.reduce_sum"(%97) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %99 = "tosa.const"() <{value = dense<6.553600e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %100 = "tosa.reciprocal"(%99) : (tensor<1xf32>) -> tensor<1xf32>
    %101 = "tosa.mul"(%100, %98) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %102 = "tosa.sub"(%96, %101) : (tensor<1x32x16x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x4096xf32>
    %103 = "tosa.mul"(%102, %102) <{shift = 0 : i8}> : (tensor<1x32x16x4096xf32>, tensor<1x32x16x4096xf32>) -> tensor<1x32x16x4096xf32>
    %104 = "tosa.reduce_sum"(%103) <{axis = 2 : i32}> : (tensor<1x32x16x4096xf32>) -> tensor<1x32x1x4096xf32>
    %105 = "tosa.reduce_sum"(%104) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %106 = "tosa.const"() <{value = dense<6.553600e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %107 = "tosa.reciprocal"(%106) : (tensor<1xf32>) -> tensor<1xf32>
    %108 = "tosa.mul"(%107, %105) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %109 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %110 = "tosa.add"(%108, %109) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %111 = "tosa.rsqrt"(%110) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %112 = "tosa.sub"(%96, %101) : (tensor<1x32x16x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x4096xf32>
    %113 = "tosa.mul"(%112, %111) <{shift = 0 : i8}> : (tensor<1x32x16x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x4096xf32>
    %114 = "tosa.reshape"(%113) <{new_shape = array<i64: 1, 512, 4096>}> : (tensor<1x32x16x4096xf32>) -> tensor<1x512x4096xf32>
    %115 = "tosa.reshape"(%arg13) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %116 = "tosa.reshape"(%115) <{new_shape = array<i64: 1, 512, 1>}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
    %117 = "tosa.reshape"(%arg14) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %118 = "tosa.reshape"(%117) <{new_shape = array<i64: 1, 512, 1>}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
    %119 = "tosa.mul"(%114, %118) <{shift = 0 : i8}> : (tensor<1x512x4096xf32>, tensor<1x512x1xf32>) -> tensor<1x512x4096xf32>
    %120 = "tosa.add"(%119, %116) : (tensor<1x512x4096xf32>, tensor<1x512x1xf32>) -> tensor<1x512x4096xf32>
    %121 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %122 = "tosa.transpose"(%120, %121) : (tensor<1x512x4096xf32>, tensor<3xi32>) -> tensor<1x4096x512xf32>
    %123 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %124 = "tosa.transpose"(%arg15, %123) : (tensor<512x512xf32>, tensor<2xi32>) -> tensor<512x512xf32>
    %125 = "tosa.reshape"(%122) <{new_shape = array<i64: 4096, 512>}> : (tensor<1x4096x512xf32>) -> tensor<4096x512xf32>
    %126 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x512xf32>}> : () -> tensor<4096x512xf32>
    %127 = "linalg.matmul"(%125, %124, %126) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg141: f32, %arg142: f32, %arg143: f32):
      %1346 = "arith.mulf"(%arg141, %arg142) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %1347 = "arith.addf"(%arg143, %1346) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1347) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map, #map1, #map2]} : (tensor<4096x512xf32>, tensor<512x512xf32>, tensor<4096x512xf32>) -> tensor<4096x512xf32>
    %128 = "tosa.reshape"(%127) <{new_shape = array<i64: 1, 4096, 512>}> : (tensor<4096x512xf32>) -> tensor<1x4096x512xf32>
    %129 = "tosa.reshape"(%arg16) <{new_shape = array<i64: 1, 1, 512>}> : (tensor<512xf32>) -> tensor<1x1x512xf32>
    %130 = "tosa.add"(%128, %129) : (tensor<1x4096x512xf32>, tensor<1x1x512xf32>) -> tensor<1x4096x512xf32>
    %131 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %132 = "tosa.transpose"(%arg17, %131) : (tensor<512x512xf32>, tensor<2xi32>) -> tensor<512x512xf32>
    %133 = "tosa.reshape"(%122) <{new_shape = array<i64: 4096, 512>}> : (tensor<1x4096x512xf32>) -> tensor<4096x512xf32>
    %134 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x512xf32>}> : () -> tensor<4096x512xf32>
    %135 = "linalg.matmul"(%133, %132, %134) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg141: f32, %arg142: f32, %arg143: f32):
      %1346 = "arith.mulf"(%arg141, %arg142) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %1347 = "arith.addf"(%arg143, %1346) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1347) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map, #map1, #map2]} : (tensor<4096x512xf32>, tensor<512x512xf32>, tensor<4096x512xf32>) -> tensor<4096x512xf32>
    %136 = "tosa.reshape"(%135) <{new_shape = array<i64: 1, 4096, 512>}> : (tensor<4096x512xf32>) -> tensor<1x4096x512xf32>
    %137 = "tosa.reshape"(%arg18) <{new_shape = array<i64: 1, 1, 512>}> : (tensor<512xf32>) -> tensor<1x1x512xf32>
    %138 = "tosa.add"(%136, %137) : (tensor<1x4096x512xf32>, tensor<1x1x512xf32>) -> tensor<1x4096x512xf32>
    %139 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %140 = "tosa.transpose"(%arg19, %139) : (tensor<512x512xf32>, tensor<2xi32>) -> tensor<512x512xf32>
    %141 = "tosa.reshape"(%122) <{new_shape = array<i64: 4096, 512>}> : (tensor<1x4096x512xf32>) -> tensor<4096x512xf32>
    %142 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x512xf32>}> : () -> tensor<4096x512xf32>
    %143 = "linalg.matmul"(%141, %140, %142) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg141: f32, %arg142: f32, %arg143: f32):
      %1346 = "arith.mulf"(%arg141, %arg142) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %1347 = "arith.addf"(%arg143, %1346) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1347) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map, #map1, #map2]} : (tensor<4096x512xf32>, tensor<512x512xf32>, tensor<4096x512xf32>) -> tensor<4096x512xf32>
    %144 = "tosa.reshape"(%143) <{new_shape = array<i64: 1, 4096, 512>}> : (tensor<4096x512xf32>) -> tensor<1x4096x512xf32>
    %145 = "tosa.reshape"(%arg20) <{new_shape = array<i64: 1, 1, 512>}> : (tensor<512xf32>) -> tensor<1x1x512xf32>
    %146 = "tosa.add"(%144, %145) : (tensor<1x4096x512xf32>, tensor<1x1x512xf32>) -> tensor<1x4096x512xf32>
    %147 = "tosa.reshape"(%130) <{new_shape = array<i64: 1, 4096, 1, 512>}> : (tensor<1x4096x512xf32>) -> tensor<1x4096x1x512xf32>
    %148 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %149 = "tosa.transpose"(%147, %148) : (tensor<1x4096x1x512xf32>, tensor<4xi32>) -> tensor<1x1x4096x512xf32>
    %150 = "tosa.reshape"(%138) <{new_shape = array<i64: 1, 4096, 1, 512>}> : (tensor<1x4096x512xf32>) -> tensor<1x4096x1x512xf32>
    %151 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %152 = "tosa.transpose"(%150, %151) : (tensor<1x4096x1x512xf32>, tensor<4xi32>) -> tensor<1x1x4096x512xf32>
    %153 = "tosa.reshape"(%146) <{new_shape = array<i64: 1, 4096, 1, 512>}> : (tensor<1x4096x512xf32>) -> tensor<1x4096x1x512xf32>
    %154 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %155 = "tosa.transpose"(%153, %154) : (tensor<1x4096x1x512xf32>, tensor<4xi32>) -> tensor<1x1x4096x512xf32>
    %156 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x4096xf32>}> : () -> tensor<4096x4096xf32>
    %157 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %158 = "tosa.transpose"(%152, %157) : (tensor<1x1x4096x512xf32>, tensor<4xi32>) -> tensor<1x1x512x4096xf32>
    %159 = "tosa.reshape"(%149) <{new_shape = array<i64: 1, 4096, 512>}> : (tensor<1x1x4096x512xf32>) -> tensor<1x4096x512xf32>
    %160 = "tosa.reshape"(%158) <{new_shape = array<i64: 1, 512, 4096>}> : (tensor<1x1x512x4096xf32>) -> tensor<1x512x4096xf32>
    %161 = "tosa.matmul"(%159, %160) : (tensor<1x4096x512xf32>, tensor<1x512x4096xf32>) -> tensor<1x4096x4096xf32>
    %162 = "arith.constant"() <{value = dense<0.0441941731> : tensor<1x4096x4096xf32>}> : () -> tensor<1x4096x4096xf32>
    %163 = "tosa.mul"(%161, %162) <{shift = 0 : i8}> : (tensor<1x4096x4096xf32>, tensor<1x4096x4096xf32>) -> tensor<1x4096x4096xf32>
    %164 = "tosa.add"(%163, %156) : (tensor<1x4096x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096x4096xf32>
    %165 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1x4096x4096xf32>}> : () -> tensor<1x4096x4096xf32>
    %166 = "linalg.softmax"(%164, %165) <{dimension = 3 : i64}> : (tensor<1x4096x4096xf32>, tensor<1x4096x4096xf32>) -> tensor<1x4096x4096xf32>
    %167 = "tosa.reshape"(%155) <{new_shape = array<i64: 1, 4096, 512>}> : (tensor<1x1x4096x512xf32>) -> tensor<1x4096x512xf32>
    %168 = "tosa.matmul"(%166, %167) : (tensor<1x4096x4096xf32>, tensor<1x4096x512xf32>) -> tensor<1x4096x512xf32>
    %169 = "tosa.reshape"(%168) <{new_shape = array<i64: 1, 1, 4096, 512>}> : (tensor<1x4096x512xf32>) -> tensor<1x1x4096x512xf32>
    %170 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %171 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %172 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %173 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %174 = "tosa.transpose"(%169, %173) : (tensor<1x1x4096x512xf32>, tensor<4xi32>) -> tensor<1x4096x1x512xf32>
    %175 = "tosa.reshape"(%174) <{new_shape = array<i64: 1, 4096, 512>}> : (tensor<1x4096x1x512xf32>) -> tensor<1x4096x512xf32>
    %176 = "tosa.reshape"(%175) <{new_shape = array<i64: 4096, 512>}> : (tensor<1x4096x512xf32>) -> tensor<4096x512xf32>
    %177 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %178 = "tosa.transpose"(%arg21, %177) : (tensor<512x512xf32>, tensor<2xi32>) -> tensor<512x512xf32>
    %179 = "tosa.reshape"(%176) <{new_shape = array<i64: 1, 4096, 512>}> : (tensor<4096x512xf32>) -> tensor<1x4096x512xf32>
    %180 = "tosa.reshape"(%178) <{new_shape = array<i64: 1, 512, 512>}> : (tensor<512x512xf32>) -> tensor<1x512x512xf32>
    %181 = "tosa.matmul"(%179, %180) : (tensor<1x4096x512xf32>, tensor<1x512x512xf32>) -> tensor<1x4096x512xf32>
    %182 = "tosa.reshape"(%181) <{new_shape = array<i64: 4096, 512>}> : (tensor<1x4096x512xf32>) -> tensor<4096x512xf32>
    %183 = "tosa.reshape"(%arg22) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %184 = "tosa.add"(%183, %182) : (tensor<1x512xf32>, tensor<4096x512xf32>) -> tensor<4096x512xf32>
    %185 = "tosa.reshape"(%184) <{new_shape = array<i64: 1, 4096, 512>}> : (tensor<4096x512xf32>) -> tensor<1x4096x512xf32>
    %186 = "tosa.identity"(%185) : (tensor<1x4096x512xf32>) -> tensor<1x4096x512xf32>
    %187 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %188 = "tosa.transpose"(%186, %187) : (tensor<1x4096x512xf32>, tensor<3xi32>) -> tensor<1x512x4096xf32>
    %189 = "tosa.reshape"(%188) <{new_shape = array<i64: 1, 512, 64, 64>}> : (tensor<1x512x4096xf32>) -> tensor<1x512x64x64xf32>
    %190 = "tosa.add"(%189, %90) : (tensor<1x512x64x64xf32>, tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %191 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x512x64x64xf32>}> : () -> tensor<1x512x64x64xf32>
    %192 = "tosa.reciprocal"(%191) : (tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %193 = "tosa.mul"(%190, %192) <{shift = 0 : i8}> : (tensor<1x512x64x64xf32>, tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %194 = "tosa.identity"(%193) : (tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %195 = "tosa.reshape"(%194) <{new_shape = array<i64: 1, 32, 16, 4096>}> : (tensor<1x512x64x64xf32>) -> tensor<1x32x16x4096xf32>
    %196 = "tosa.reduce_sum"(%195) <{axis = 2 : i32}> : (tensor<1x32x16x4096xf32>) -> tensor<1x32x1x4096xf32>
    %197 = "tosa.reduce_sum"(%196) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %198 = "tosa.const"() <{value = dense<6.553600e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %199 = "tosa.reciprocal"(%198) : (tensor<1xf32>) -> tensor<1xf32>
    %200 = "tosa.mul"(%199, %197) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %201 = "tosa.sub"(%195, %200) : (tensor<1x32x16x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x4096xf32>
    %202 = "tosa.mul"(%201, %201) <{shift = 0 : i8}> : (tensor<1x32x16x4096xf32>, tensor<1x32x16x4096xf32>) -> tensor<1x32x16x4096xf32>
    %203 = "tosa.reduce_sum"(%202) <{axis = 2 : i32}> : (tensor<1x32x16x4096xf32>) -> tensor<1x32x1x4096xf32>
    %204 = "tosa.reduce_sum"(%203) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %205 = "tosa.const"() <{value = dense<6.553600e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %206 = "tosa.reciprocal"(%205) : (tensor<1xf32>) -> tensor<1xf32>
    %207 = "tosa.mul"(%206, %204) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %208 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %209 = "tosa.add"(%207, %208) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %210 = "tosa.rsqrt"(%209) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %211 = "tosa.sub"(%195, %200) : (tensor<1x32x16x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x4096xf32>
    %212 = "tosa.mul"(%211, %210) <{shift = 0 : i8}> : (tensor<1x32x16x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x4096xf32>
    %213 = "tosa.reshape"(%212) <{new_shape = array<i64: 1, 512, 64, 64>}> : (tensor<1x32x16x4096xf32>) -> tensor<1x512x64x64xf32>
    %214 = "tosa.reshape"(%arg23) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %215 = "tosa.reshape"(%214) <{new_shape = array<i64: 1, 512, 1>}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
    %216 = "tosa.reshape"(%215) <{new_shape = array<i64: 1, 512, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1x1xf32>
    %217 = "tosa.reshape"(%arg24) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %218 = "tosa.reshape"(%217) <{new_shape = array<i64: 1, 512, 1>}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
    %219 = "tosa.reshape"(%218) <{new_shape = array<i64: 1, 512, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1x1xf32>
    %220 = "tosa.mul"(%213, %219) <{shift = 0 : i8}> : (tensor<1x512x64x64xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x64x64xf32>
    %221 = "tosa.add"(%220, %216) : (tensor<1x512x64x64xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x64x64xf32>
    %222 = "tosa.sigmoid"(%221) : (tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %223 = "tosa.mul"(%221, %222) <{shift = 0 : i8}> : (tensor<1x512x64x64xf32>, tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %224 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %225 = "tosa.transpose"(%223, %224) : (tensor<1x512x64x64xf32>, tensor<4xi32>) -> tensor<1x64x64x512xf32>
    %226 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %227 = "tosa.transpose"(%arg25, %226) : (tensor<512x512x3x3xf32>, tensor<4xi32>) -> tensor<512x3x3x512xf32>
    %228 = "tosa.conv2d"(%225, %227, %arg26) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x64x64x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>) -> tensor<1x64x64x512xf32>
    %229 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %230 = "tosa.transpose"(%228, %229) : (tensor<1x64x64x512xf32>, tensor<4xi32>) -> tensor<1x512x64x64xf32>
    %231 = "tosa.reshape"(%230) <{new_shape = array<i64: 1, 32, 16, 4096>}> : (tensor<1x512x64x64xf32>) -> tensor<1x32x16x4096xf32>
    %232 = "tosa.reduce_sum"(%231) <{axis = 2 : i32}> : (tensor<1x32x16x4096xf32>) -> tensor<1x32x1x4096xf32>
    %233 = "tosa.reduce_sum"(%232) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %234 = "tosa.const"() <{value = dense<6.553600e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %235 = "tosa.reciprocal"(%234) : (tensor<1xf32>) -> tensor<1xf32>
    %236 = "tosa.mul"(%235, %233) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %237 = "tosa.sub"(%231, %236) : (tensor<1x32x16x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x4096xf32>
    %238 = "tosa.mul"(%237, %237) <{shift = 0 : i8}> : (tensor<1x32x16x4096xf32>, tensor<1x32x16x4096xf32>) -> tensor<1x32x16x4096xf32>
    %239 = "tosa.reduce_sum"(%238) <{axis = 2 : i32}> : (tensor<1x32x16x4096xf32>) -> tensor<1x32x1x4096xf32>
    %240 = "tosa.reduce_sum"(%239) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %241 = "tosa.const"() <{value = dense<6.553600e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %242 = "tosa.reciprocal"(%241) : (tensor<1xf32>) -> tensor<1xf32>
    %243 = "tosa.mul"(%242, %240) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %244 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %245 = "tosa.add"(%243, %244) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %246 = "tosa.rsqrt"(%245) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %247 = "tosa.sub"(%231, %236) : (tensor<1x32x16x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x4096xf32>
    %248 = "tosa.mul"(%247, %246) <{shift = 0 : i8}> : (tensor<1x32x16x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x4096xf32>
    %249 = "tosa.reshape"(%248) <{new_shape = array<i64: 1, 512, 64, 64>}> : (tensor<1x32x16x4096xf32>) -> tensor<1x512x64x64xf32>
    %250 = "tosa.reshape"(%arg27) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %251 = "tosa.reshape"(%250) <{new_shape = array<i64: 1, 512, 1>}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
    %252 = "tosa.reshape"(%251) <{new_shape = array<i64: 1, 512, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1x1xf32>
    %253 = "tosa.reshape"(%arg28) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %254 = "tosa.reshape"(%253) <{new_shape = array<i64: 1, 512, 1>}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
    %255 = "tosa.reshape"(%254) <{new_shape = array<i64: 1, 512, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1x1xf32>
    %256 = "tosa.mul"(%249, %255) <{shift = 0 : i8}> : (tensor<1x512x64x64xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x64x64xf32>
    %257 = "tosa.add"(%256, %252) : (tensor<1x512x64x64xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x64x64xf32>
    %258 = "tosa.sigmoid"(%257) : (tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %259 = "tosa.mul"(%257, %258) <{shift = 0 : i8}> : (tensor<1x512x64x64xf32>, tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %260 = "tosa.identity"(%259) : (tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %261 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %262 = "tosa.transpose"(%260, %261) : (tensor<1x512x64x64xf32>, tensor<4xi32>) -> tensor<1x64x64x512xf32>
    %263 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %264 = "tosa.transpose"(%arg29, %263) : (tensor<512x512x3x3xf32>, tensor<4xi32>) -> tensor<512x3x3x512xf32>
    %265 = "tosa.conv2d"(%262, %264, %arg30) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x64x64x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>) -> tensor<1x64x64x512xf32>
    %266 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %267 = "tosa.transpose"(%265, %266) : (tensor<1x64x64x512xf32>, tensor<4xi32>) -> tensor<1x512x64x64xf32>
    %268 = "tosa.add"(%193, %267) : (tensor<1x512x64x64xf32>, tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %269 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x512x64x64xf32>}> : () -> tensor<1x512x64x64xf32>
    %270 = "tosa.reciprocal"(%269) : (tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %271 = "tosa.mul"(%268, %270) <{shift = 0 : i8}> : (tensor<1x512x64x64xf32>, tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %272 = "tosa.identity"(%271) : (tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %273 = "tosa.reshape"(%272) <{new_shape = array<i64: 1, 32, 16, 4096>}> : (tensor<1x512x64x64xf32>) -> tensor<1x32x16x4096xf32>
    %274 = "tosa.reduce_sum"(%273) <{axis = 2 : i32}> : (tensor<1x32x16x4096xf32>) -> tensor<1x32x1x4096xf32>
    %275 = "tosa.reduce_sum"(%274) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %276 = "tosa.const"() <{value = dense<6.553600e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %277 = "tosa.reciprocal"(%276) : (tensor<1xf32>) -> tensor<1xf32>
    %278 = "tosa.mul"(%277, %275) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %279 = "tosa.sub"(%273, %278) : (tensor<1x32x16x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x4096xf32>
    %280 = "tosa.mul"(%279, %279) <{shift = 0 : i8}> : (tensor<1x32x16x4096xf32>, tensor<1x32x16x4096xf32>) -> tensor<1x32x16x4096xf32>
    %281 = "tosa.reduce_sum"(%280) <{axis = 2 : i32}> : (tensor<1x32x16x4096xf32>) -> tensor<1x32x1x4096xf32>
    %282 = "tosa.reduce_sum"(%281) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %283 = "tosa.const"() <{value = dense<6.553600e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %284 = "tosa.reciprocal"(%283) : (tensor<1xf32>) -> tensor<1xf32>
    %285 = "tosa.mul"(%284, %282) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %286 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %287 = "tosa.add"(%285, %286) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %288 = "tosa.rsqrt"(%287) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %289 = "tosa.sub"(%273, %278) : (tensor<1x32x16x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x4096xf32>
    %290 = "tosa.mul"(%289, %288) <{shift = 0 : i8}> : (tensor<1x32x16x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x4096xf32>
    %291 = "tosa.reshape"(%290) <{new_shape = array<i64: 1, 512, 64, 64>}> : (tensor<1x32x16x4096xf32>) -> tensor<1x512x64x64xf32>
    %292 = "tosa.reshape"(%arg31) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %293 = "tosa.reshape"(%292) <{new_shape = array<i64: 1, 512, 1>}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
    %294 = "tosa.reshape"(%293) <{new_shape = array<i64: 1, 512, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1x1xf32>
    %295 = "tosa.reshape"(%arg32) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %296 = "tosa.reshape"(%295) <{new_shape = array<i64: 1, 512, 1>}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
    %297 = "tosa.reshape"(%296) <{new_shape = array<i64: 1, 512, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1x1xf32>
    %298 = "tosa.mul"(%291, %297) <{shift = 0 : i8}> : (tensor<1x512x64x64xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x64x64xf32>
    %299 = "tosa.add"(%298, %294) : (tensor<1x512x64x64xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x64x64xf32>
    %300 = "tosa.sigmoid"(%299) : (tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %301 = "tosa.mul"(%299, %300) <{shift = 0 : i8}> : (tensor<1x512x64x64xf32>, tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %302 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %303 = "tosa.transpose"(%301, %302) : (tensor<1x512x64x64xf32>, tensor<4xi32>) -> tensor<1x64x64x512xf32>
    %304 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %305 = "tosa.transpose"(%arg33, %304) : (tensor<512x512x3x3xf32>, tensor<4xi32>) -> tensor<512x3x3x512xf32>
    %306 = "tosa.conv2d"(%303, %305, %arg34) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x64x64x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>) -> tensor<1x64x64x512xf32>
    %307 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %308 = "tosa.transpose"(%306, %307) : (tensor<1x64x64x512xf32>, tensor<4xi32>) -> tensor<1x512x64x64xf32>
    %309 = "tosa.reshape"(%308) <{new_shape = array<i64: 1, 32, 16, 4096>}> : (tensor<1x512x64x64xf32>) -> tensor<1x32x16x4096xf32>
    %310 = "tosa.reduce_sum"(%309) <{axis = 2 : i32}> : (tensor<1x32x16x4096xf32>) -> tensor<1x32x1x4096xf32>
    %311 = "tosa.reduce_sum"(%310) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %312 = "tosa.const"() <{value = dense<6.553600e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %313 = "tosa.reciprocal"(%312) : (tensor<1xf32>) -> tensor<1xf32>
    %314 = "tosa.mul"(%313, %311) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %315 = "tosa.sub"(%309, %314) : (tensor<1x32x16x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x4096xf32>
    %316 = "tosa.mul"(%315, %315) <{shift = 0 : i8}> : (tensor<1x32x16x4096xf32>, tensor<1x32x16x4096xf32>) -> tensor<1x32x16x4096xf32>
    %317 = "tosa.reduce_sum"(%316) <{axis = 2 : i32}> : (tensor<1x32x16x4096xf32>) -> tensor<1x32x1x4096xf32>
    %318 = "tosa.reduce_sum"(%317) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %319 = "tosa.const"() <{value = dense<6.553600e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %320 = "tosa.reciprocal"(%319) : (tensor<1xf32>) -> tensor<1xf32>
    %321 = "tosa.mul"(%320, %318) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %322 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %323 = "tosa.add"(%321, %322) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %324 = "tosa.rsqrt"(%323) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %325 = "tosa.sub"(%309, %314) : (tensor<1x32x16x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x4096xf32>
    %326 = "tosa.mul"(%325, %324) <{shift = 0 : i8}> : (tensor<1x32x16x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x4096xf32>
    %327 = "tosa.reshape"(%326) <{new_shape = array<i64: 1, 512, 64, 64>}> : (tensor<1x32x16x4096xf32>) -> tensor<1x512x64x64xf32>
    %328 = "tosa.reshape"(%arg35) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %329 = "tosa.reshape"(%328) <{new_shape = array<i64: 1, 512, 1>}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
    %330 = "tosa.reshape"(%329) <{new_shape = array<i64: 1, 512, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1x1xf32>
    %331 = "tosa.reshape"(%arg36) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %332 = "tosa.reshape"(%331) <{new_shape = array<i64: 1, 512, 1>}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
    %333 = "tosa.reshape"(%332) <{new_shape = array<i64: 1, 512, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1x1xf32>
    %334 = "tosa.mul"(%327, %333) <{shift = 0 : i8}> : (tensor<1x512x64x64xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x64x64xf32>
    %335 = "tosa.add"(%334, %330) : (tensor<1x512x64x64xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x64x64xf32>
    %336 = "tosa.sigmoid"(%335) : (tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %337 = "tosa.mul"(%335, %336) <{shift = 0 : i8}> : (tensor<1x512x64x64xf32>, tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %338 = "tosa.identity"(%337) : (tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %339 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %340 = "tosa.transpose"(%338, %339) : (tensor<1x512x64x64xf32>, tensor<4xi32>) -> tensor<1x64x64x512xf32>
    %341 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %342 = "tosa.transpose"(%arg37, %341) : (tensor<512x512x3x3xf32>, tensor<4xi32>) -> tensor<512x3x3x512xf32>
    %343 = "tosa.conv2d"(%340, %342, %arg38) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x64x64x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>) -> tensor<1x64x64x512xf32>
    %344 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %345 = "tosa.transpose"(%343, %344) : (tensor<1x64x64x512xf32>, tensor<4xi32>) -> tensor<1x512x64x64xf32>
    %346 = "tosa.add"(%271, %345) : (tensor<1x512x64x64xf32>, tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %347 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x512x64x64xf32>}> : () -> tensor<1x512x64x64xf32>
    %348 = "tosa.reciprocal"(%347) : (tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %349 = "tosa.mul"(%346, %348) <{shift = 0 : i8}> : (tensor<1x512x64x64xf32>, tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %350 = "tosa.identity"(%349) : (tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %351 = "tosa.reshape"(%350) <{new_shape = array<i64: 1, 32, 16, 4096>}> : (tensor<1x512x64x64xf32>) -> tensor<1x32x16x4096xf32>
    %352 = "tosa.reduce_sum"(%351) <{axis = 2 : i32}> : (tensor<1x32x16x4096xf32>) -> tensor<1x32x1x4096xf32>
    %353 = "tosa.reduce_sum"(%352) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %354 = "tosa.const"() <{value = dense<6.553600e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %355 = "tosa.reciprocal"(%354) : (tensor<1xf32>) -> tensor<1xf32>
    %356 = "tosa.mul"(%355, %353) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %357 = "tosa.sub"(%351, %356) : (tensor<1x32x16x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x4096xf32>
    %358 = "tosa.mul"(%357, %357) <{shift = 0 : i8}> : (tensor<1x32x16x4096xf32>, tensor<1x32x16x4096xf32>) -> tensor<1x32x16x4096xf32>
    %359 = "tosa.reduce_sum"(%358) <{axis = 2 : i32}> : (tensor<1x32x16x4096xf32>) -> tensor<1x32x1x4096xf32>
    %360 = "tosa.reduce_sum"(%359) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %361 = "tosa.const"() <{value = dense<6.553600e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %362 = "tosa.reciprocal"(%361) : (tensor<1xf32>) -> tensor<1xf32>
    %363 = "tosa.mul"(%362, %360) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %364 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %365 = "tosa.add"(%363, %364) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %366 = "tosa.rsqrt"(%365) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %367 = "tosa.sub"(%351, %356) : (tensor<1x32x16x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x4096xf32>
    %368 = "tosa.mul"(%367, %366) <{shift = 0 : i8}> : (tensor<1x32x16x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x4096xf32>
    %369 = "tosa.reshape"(%368) <{new_shape = array<i64: 1, 512, 64, 64>}> : (tensor<1x32x16x4096xf32>) -> tensor<1x512x64x64xf32>
    %370 = "tosa.reshape"(%arg39) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %371 = "tosa.reshape"(%370) <{new_shape = array<i64: 1, 512, 1>}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
    %372 = "tosa.reshape"(%371) <{new_shape = array<i64: 1, 512, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1x1xf32>
    %373 = "tosa.reshape"(%arg40) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %374 = "tosa.reshape"(%373) <{new_shape = array<i64: 1, 512, 1>}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
    %375 = "tosa.reshape"(%374) <{new_shape = array<i64: 1, 512, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1x1xf32>
    %376 = "tosa.mul"(%369, %375) <{shift = 0 : i8}> : (tensor<1x512x64x64xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x64x64xf32>
    %377 = "tosa.add"(%376, %372) : (tensor<1x512x64x64xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x64x64xf32>
    %378 = "tosa.sigmoid"(%377) : (tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %379 = "tosa.mul"(%377, %378) <{shift = 0 : i8}> : (tensor<1x512x64x64xf32>, tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %380 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %381 = "tosa.transpose"(%379, %380) : (tensor<1x512x64x64xf32>, tensor<4xi32>) -> tensor<1x64x64x512xf32>
    %382 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %383 = "tosa.transpose"(%arg41, %382) : (tensor<512x512x3x3xf32>, tensor<4xi32>) -> tensor<512x3x3x512xf32>
    %384 = "tosa.conv2d"(%381, %383, %arg42) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x64x64x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>) -> tensor<1x64x64x512xf32>
    %385 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %386 = "tosa.transpose"(%384, %385) : (tensor<1x64x64x512xf32>, tensor<4xi32>) -> tensor<1x512x64x64xf32>
    %387 = "tosa.reshape"(%386) <{new_shape = array<i64: 1, 32, 16, 4096>}> : (tensor<1x512x64x64xf32>) -> tensor<1x32x16x4096xf32>
    %388 = "tosa.reduce_sum"(%387) <{axis = 2 : i32}> : (tensor<1x32x16x4096xf32>) -> tensor<1x32x1x4096xf32>
    %389 = "tosa.reduce_sum"(%388) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %390 = "tosa.const"() <{value = dense<6.553600e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %391 = "tosa.reciprocal"(%390) : (tensor<1xf32>) -> tensor<1xf32>
    %392 = "tosa.mul"(%391, %389) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %393 = "tosa.sub"(%387, %392) : (tensor<1x32x16x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x4096xf32>
    %394 = "tosa.mul"(%393, %393) <{shift = 0 : i8}> : (tensor<1x32x16x4096xf32>, tensor<1x32x16x4096xf32>) -> tensor<1x32x16x4096xf32>
    %395 = "tosa.reduce_sum"(%394) <{axis = 2 : i32}> : (tensor<1x32x16x4096xf32>) -> tensor<1x32x1x4096xf32>
    %396 = "tosa.reduce_sum"(%395) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %397 = "tosa.const"() <{value = dense<6.553600e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %398 = "tosa.reciprocal"(%397) : (tensor<1xf32>) -> tensor<1xf32>
    %399 = "tosa.mul"(%398, %396) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %400 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %401 = "tosa.add"(%399, %400) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %402 = "tosa.rsqrt"(%401) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %403 = "tosa.sub"(%387, %392) : (tensor<1x32x16x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x4096xf32>
    %404 = "tosa.mul"(%403, %402) <{shift = 0 : i8}> : (tensor<1x32x16x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x4096xf32>
    %405 = "tosa.reshape"(%404) <{new_shape = array<i64: 1, 512, 64, 64>}> : (tensor<1x32x16x4096xf32>) -> tensor<1x512x64x64xf32>
    %406 = "tosa.reshape"(%arg43) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %407 = "tosa.reshape"(%406) <{new_shape = array<i64: 1, 512, 1>}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
    %408 = "tosa.reshape"(%407) <{new_shape = array<i64: 1, 512, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1x1xf32>
    %409 = "tosa.reshape"(%arg44) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %410 = "tosa.reshape"(%409) <{new_shape = array<i64: 1, 512, 1>}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
    %411 = "tosa.reshape"(%410) <{new_shape = array<i64: 1, 512, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1x1xf32>
    %412 = "tosa.mul"(%405, %411) <{shift = 0 : i8}> : (tensor<1x512x64x64xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x64x64xf32>
    %413 = "tosa.add"(%412, %408) : (tensor<1x512x64x64xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x64x64xf32>
    %414 = "tosa.sigmoid"(%413) : (tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %415 = "tosa.mul"(%413, %414) <{shift = 0 : i8}> : (tensor<1x512x64x64xf32>, tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %416 = "tosa.identity"(%415) : (tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %417 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %418 = "tosa.transpose"(%416, %417) : (tensor<1x512x64x64xf32>, tensor<4xi32>) -> tensor<1x64x64x512xf32>
    %419 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %420 = "tosa.transpose"(%arg45, %419) : (tensor<512x512x3x3xf32>, tensor<4xi32>) -> tensor<512x3x3x512xf32>
    %421 = "tosa.conv2d"(%418, %420, %arg46) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x64x64x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>) -> tensor<1x64x64x512xf32>
    %422 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %423 = "tosa.transpose"(%421, %422) : (tensor<1x64x64x512xf32>, tensor<4xi32>) -> tensor<1x512x64x64xf32>
    %424 = "tosa.add"(%349, %423) : (tensor<1x512x64x64xf32>, tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %425 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x512x64x64xf32>}> : () -> tensor<1x512x64x64xf32>
    %426 = "tosa.reciprocal"(%425) : (tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %427 = "tosa.mul"(%424, %426) <{shift = 0 : i8}> : (tensor<1x512x64x64xf32>, tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %428 = "tosa.identity"(%427) : (tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %429 = "tosa.reshape"(%428) <{new_shape = array<i64: 1, 32, 16, 4096>}> : (tensor<1x512x64x64xf32>) -> tensor<1x32x16x4096xf32>
    %430 = "tosa.reduce_sum"(%429) <{axis = 2 : i32}> : (tensor<1x32x16x4096xf32>) -> tensor<1x32x1x4096xf32>
    %431 = "tosa.reduce_sum"(%430) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %432 = "tosa.const"() <{value = dense<6.553600e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %433 = "tosa.reciprocal"(%432) : (tensor<1xf32>) -> tensor<1xf32>
    %434 = "tosa.mul"(%433, %431) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %435 = "tosa.sub"(%429, %434) : (tensor<1x32x16x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x4096xf32>
    %436 = "tosa.mul"(%435, %435) <{shift = 0 : i8}> : (tensor<1x32x16x4096xf32>, tensor<1x32x16x4096xf32>) -> tensor<1x32x16x4096xf32>
    %437 = "tosa.reduce_sum"(%436) <{axis = 2 : i32}> : (tensor<1x32x16x4096xf32>) -> tensor<1x32x1x4096xf32>
    %438 = "tosa.reduce_sum"(%437) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %439 = "tosa.const"() <{value = dense<6.553600e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %440 = "tosa.reciprocal"(%439) : (tensor<1xf32>) -> tensor<1xf32>
    %441 = "tosa.mul"(%440, %438) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %442 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %443 = "tosa.add"(%441, %442) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %444 = "tosa.rsqrt"(%443) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %445 = "tosa.sub"(%429, %434) : (tensor<1x32x16x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x4096xf32>
    %446 = "tosa.mul"(%445, %444) <{shift = 0 : i8}> : (tensor<1x32x16x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x4096xf32>
    %447 = "tosa.reshape"(%446) <{new_shape = array<i64: 1, 512, 64, 64>}> : (tensor<1x32x16x4096xf32>) -> tensor<1x512x64x64xf32>
    %448 = "tosa.reshape"(%arg47) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %449 = "tosa.reshape"(%448) <{new_shape = array<i64: 1, 512, 1>}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
    %450 = "tosa.reshape"(%449) <{new_shape = array<i64: 1, 512, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1x1xf32>
    %451 = "tosa.reshape"(%arg48) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %452 = "tosa.reshape"(%451) <{new_shape = array<i64: 1, 512, 1>}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
    %453 = "tosa.reshape"(%452) <{new_shape = array<i64: 1, 512, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1x1xf32>
    %454 = "tosa.mul"(%447, %453) <{shift = 0 : i8}> : (tensor<1x512x64x64xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x64x64xf32>
    %455 = "tosa.add"(%454, %450) : (tensor<1x512x64x64xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x64x64xf32>
    %456 = "tosa.sigmoid"(%455) : (tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %457 = "tosa.mul"(%455, %456) <{shift = 0 : i8}> : (tensor<1x512x64x64xf32>, tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %458 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %459 = "tosa.transpose"(%457, %458) : (tensor<1x512x64x64xf32>, tensor<4xi32>) -> tensor<1x64x64x512xf32>
    %460 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %461 = "tosa.transpose"(%arg49, %460) : (tensor<512x512x3x3xf32>, tensor<4xi32>) -> tensor<512x3x3x512xf32>
    %462 = "tosa.conv2d"(%459, %461, %arg50) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x64x64x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>) -> tensor<1x64x64x512xf32>
    %463 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %464 = "tosa.transpose"(%462, %463) : (tensor<1x64x64x512xf32>, tensor<4xi32>) -> tensor<1x512x64x64xf32>
    %465 = "tosa.reshape"(%464) <{new_shape = array<i64: 1, 32, 16, 4096>}> : (tensor<1x512x64x64xf32>) -> tensor<1x32x16x4096xf32>
    %466 = "tosa.reduce_sum"(%465) <{axis = 2 : i32}> : (tensor<1x32x16x4096xf32>) -> tensor<1x32x1x4096xf32>
    %467 = "tosa.reduce_sum"(%466) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %468 = "tosa.const"() <{value = dense<6.553600e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %469 = "tosa.reciprocal"(%468) : (tensor<1xf32>) -> tensor<1xf32>
    %470 = "tosa.mul"(%469, %467) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %471 = "tosa.sub"(%465, %470) : (tensor<1x32x16x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x4096xf32>
    %472 = "tosa.mul"(%471, %471) <{shift = 0 : i8}> : (tensor<1x32x16x4096xf32>, tensor<1x32x16x4096xf32>) -> tensor<1x32x16x4096xf32>
    %473 = "tosa.reduce_sum"(%472) <{axis = 2 : i32}> : (tensor<1x32x16x4096xf32>) -> tensor<1x32x1x4096xf32>
    %474 = "tosa.reduce_sum"(%473) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %475 = "tosa.const"() <{value = dense<6.553600e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %476 = "tosa.reciprocal"(%475) : (tensor<1xf32>) -> tensor<1xf32>
    %477 = "tosa.mul"(%476, %474) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %478 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %479 = "tosa.add"(%477, %478) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %480 = "tosa.rsqrt"(%479) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %481 = "tosa.sub"(%465, %470) : (tensor<1x32x16x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x4096xf32>
    %482 = "tosa.mul"(%481, %480) <{shift = 0 : i8}> : (tensor<1x32x16x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x4096xf32>
    %483 = "tosa.reshape"(%482) <{new_shape = array<i64: 1, 512, 64, 64>}> : (tensor<1x32x16x4096xf32>) -> tensor<1x512x64x64xf32>
    %484 = "tosa.reshape"(%arg51) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %485 = "tosa.reshape"(%484) <{new_shape = array<i64: 1, 512, 1>}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
    %486 = "tosa.reshape"(%485) <{new_shape = array<i64: 1, 512, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1x1xf32>
    %487 = "tosa.reshape"(%arg52) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %488 = "tosa.reshape"(%487) <{new_shape = array<i64: 1, 512, 1>}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
    %489 = "tosa.reshape"(%488) <{new_shape = array<i64: 1, 512, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1x1xf32>
    %490 = "tosa.mul"(%483, %489) <{shift = 0 : i8}> : (tensor<1x512x64x64xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x64x64xf32>
    %491 = "tosa.add"(%490, %486) : (tensor<1x512x64x64xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x64x64xf32>
    %492 = "tosa.sigmoid"(%491) : (tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %493 = "tosa.mul"(%491, %492) <{shift = 0 : i8}> : (tensor<1x512x64x64xf32>, tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %494 = "tosa.identity"(%493) : (tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %495 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %496 = "tosa.transpose"(%494, %495) : (tensor<1x512x64x64xf32>, tensor<4xi32>) -> tensor<1x64x64x512xf32>
    %497 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %498 = "tosa.transpose"(%arg53, %497) : (tensor<512x512x3x3xf32>, tensor<4xi32>) -> tensor<512x3x3x512xf32>
    %499 = "tosa.conv2d"(%496, %498, %arg54) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x64x64x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>) -> tensor<1x64x64x512xf32>
    %500 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %501 = "tosa.transpose"(%499, %500) : (tensor<1x64x64x512xf32>, tensor<4xi32>) -> tensor<1x512x64x64xf32>
    %502 = "tosa.add"(%427, %501) : (tensor<1x512x64x64xf32>, tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %503 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x512x64x64xf32>}> : () -> tensor<1x512x64x64xf32>
    %504 = "tosa.reciprocal"(%503) : (tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %505 = "tosa.mul"(%502, %504) <{shift = 0 : i8}> : (tensor<1x512x64x64xf32>, tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    %506 = "tosa.const"() <{value = dense<"0x00000000000000000100000000000000020000000000000003000000000000000400000000000000050000000000000006000000000000000700000000000000080000000000000009000000000000000A000000000000000B000000000000000C000000000000000D000000000000000E000000000000000F0000000000000010000000000000001100000000000000120000000000000013000000000000001400000000000000150000000000000016000000000000001700000000000000180000000000000019000000000000001A000000000000001B000000000000001C000000000000001D000000000000001E000000000000001F0000000000000020000000000000002100000000000000220000000000000023000000000000002400000000000000250000000000000026000000000000002700000000000000280000000000000029000000000000002A000000000000002B000000000000002C000000000000002D000000000000002E000000000000002F0000000000000030000000000000003100000000000000320000000000000033000000000000003400000000000000350000000000000036000000000000003700000000000000380000000000000039000000000000003A000000000000003B000000000000003C000000000000003D000000000000003E000000000000003F0000000000000040000000000000004100000000000000420000000000000043000000000000004400000000000000450000000000000046000000000000004700000000000000480000000000000049000000000000004A000000000000004B000000000000004C000000000000004D000000000000004E000000000000004F0000000000000050000000000000005100000000000000520000000000000053000000000000005400000000000000550000000000000056000000000000005700000000000000580000000000000059000000000000005A000000000000005B000000000000005C000000000000005D000000000000005E000000000000005F0000000000000060000000000000006100000000000000620000000000000063000000000000006400000000000000650000000000000066000000000000006700000000000000680000000000000069000000000000006A000000000000006B000000000000006C000000000000006D000000000000006E000000000000006F0000000000000070000000000000007100000000000000720000000000000073000000000000007400000000000000750000000000000076000000000000007700000000000000780000000000000079000000000000007A000000000000007B000000000000007C000000000000007D000000000000007E000000000000007F00000000000000"> : tensor<128xi64>}> : () -> tensor<128xi64>
    %507 = "tosa.cast"(%506) : (tensor<128xi64>) -> tensor<128xf32>
    %508 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<128xf32>}> : () -> tensor<128xf32>
    %509 = "tosa.mul"(%507, %508) <{shift = 0 : i8}> : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %510 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<128xf32>}> : () -> tensor<128xf32>
    %511 = "tosa.add"(%509, %510) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %512 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<128xf32>}> : () -> tensor<128xf32>
    %513 = "tosa.mul"(%511, %512) <{shift = 0 : i8}> : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %514 = "tosa.cast"(%513) : (tensor<128xf32>) -> tensor<128xi64>
    %515 = "tosa.reshape"(%514) <{new_shape = array<i64: 128, 1>}> : (tensor<128xi64>) -> tensor<128x1xi64>
    %516 = "tosa.const"() <{value = dense<"0x00000000000000000100000000000000020000000000000003000000000000000400000000000000050000000000000006000000000000000700000000000000080000000000000009000000000000000A000000000000000B000000000000000C000000000000000D000000000000000E000000000000000F0000000000000010000000000000001100000000000000120000000000000013000000000000001400000000000000150000000000000016000000000000001700000000000000180000000000000019000000000000001A000000000000001B000000000000001C000000000000001D000000000000001E000000000000001F0000000000000020000000000000002100000000000000220000000000000023000000000000002400000000000000250000000000000026000000000000002700000000000000280000000000000029000000000000002A000000000000002B000000000000002C000000000000002D000000000000002E000000000000002F0000000000000030000000000000003100000000000000320000000000000033000000000000003400000000000000350000000000000036000000000000003700000000000000380000000000000039000000000000003A000000000000003B000000000000003C000000000000003D000000000000003E000000000000003F0000000000000040000000000000004100000000000000420000000000000043000000000000004400000000000000450000000000000046000000000000004700000000000000480000000000000049000000000000004A000000000000004B000000000000004C000000000000004D000000000000004E000000000000004F0000000000000050000000000000005100000000000000520000000000000053000000000000005400000000000000550000000000000056000000000000005700000000000000580000000000000059000000000000005A000000000000005B000000000000005C000000000000005D000000000000005E000000000000005F0000000000000060000000000000006100000000000000620000000000000063000000000000006400000000000000650000000000000066000000000000006700000000000000680000000000000069000000000000006A000000000000006B000000000000006C000000000000006D000000000000006E000000000000006F0000000000000070000000000000007100000000000000720000000000000073000000000000007400000000000000750000000000000076000000000000007700000000000000780000000000000079000000000000007A000000000000007B000000000000007C000000000000007D000000000000007E000000000000007F00000000000000"> : tensor<128xi64>}> : () -> tensor<128xi64>
    %517 = "tosa.cast"(%516) : (tensor<128xi64>) -> tensor<128xf32>
    %518 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<128xf32>}> : () -> tensor<128xf32>
    %519 = "tosa.mul"(%517, %518) <{shift = 0 : i8}> : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %520 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<128xf32>}> : () -> tensor<128xf32>
    %521 = "tosa.add"(%519, %520) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %522 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<128xf32>}> : () -> tensor<128xf32>
    %523 = "tosa.mul"(%521, %522) <{shift = 0 : i8}> : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %524 = "tosa.cast"(%523) : (tensor<128xf32>) -> tensor<128xi64>
    %525 = "tensor.empty"() : () -> tensor<1x512x128x128xf32>
    %526 = "linalg.generic"(%515, %524, %525) <{indexing_maps = [#map3, #map3, #map4], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg141: i64, %arg142: i64, %arg143: f32):
      %1346 = "arith.index_cast"(%arg141) : (i64) -> index
      %1347 = "arith.index_cast"(%arg142) : (i64) -> index
      %1348 = "linalg.index"() <{dim = 2 : i64}> : () -> index
      %1349 = "tensor.extract"(%505, %1346, %1347, %1348) : (tensor<1x512x64x64xf32>, index, index, index) -> f32
      "linalg.yield"(%1349) : (f32) -> ()
    }) : (tensor<128x1xi64>, tensor<128xi64>, tensor<1x512x128x128xf32>) -> tensor<1x512x128x128xf32>
    %527 = "tosa.identity"(%526) : (tensor<1x512x128x128xf32>) -> tensor<1x512x128x128xf32>
    %528 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %529 = "tosa.transpose"(%527, %528) : (tensor<1x512x128x128xf32>, tensor<4xi32>) -> tensor<1x128x128x512xf32>
    %530 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %531 = "tosa.transpose"(%arg55, %530) : (tensor<512x512x3x3xf32>, tensor<4xi32>) -> tensor<512x3x3x512xf32>
    %532 = "tosa.conv2d"(%529, %531, %arg56) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x128x128x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>) -> tensor<1x128x128x512xf32>
    %533 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %534 = "tosa.transpose"(%532, %533) : (tensor<1x128x128x512xf32>, tensor<4xi32>) -> tensor<1x512x128x128xf32>
    %535 = "tosa.identity"(%534) : (tensor<1x512x128x128xf32>) -> tensor<1x512x128x128xf32>
    %536 = "tosa.reshape"(%535) <{new_shape = array<i64: 1, 32, 16, 16384>}> : (tensor<1x512x128x128xf32>) -> tensor<1x32x16x16384xf32>
    %537 = "tosa.reduce_sum"(%536) <{axis = 2 : i32}> : (tensor<1x32x16x16384xf32>) -> tensor<1x32x1x16384xf32>
    %538 = "tosa.reduce_sum"(%537) <{axis = 3 : i32}> : (tensor<1x32x1x16384xf32>) -> tensor<1x32x1x1xf32>
    %539 = "tosa.const"() <{value = dense<2.621440e+05> : tensor<1xf32>}> : () -> tensor<1xf32>
    %540 = "tosa.reciprocal"(%539) : (tensor<1xf32>) -> tensor<1xf32>
    %541 = "tosa.mul"(%540, %538) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %542 = "tosa.sub"(%536, %541) : (tensor<1x32x16x16384xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x16384xf32>
    %543 = "tosa.mul"(%542, %542) <{shift = 0 : i8}> : (tensor<1x32x16x16384xf32>, tensor<1x32x16x16384xf32>) -> tensor<1x32x16x16384xf32>
    %544 = "tosa.reduce_sum"(%543) <{axis = 2 : i32}> : (tensor<1x32x16x16384xf32>) -> tensor<1x32x1x16384xf32>
    %545 = "tosa.reduce_sum"(%544) <{axis = 3 : i32}> : (tensor<1x32x1x16384xf32>) -> tensor<1x32x1x1xf32>
    %546 = "tosa.const"() <{value = dense<2.621440e+05> : tensor<1xf32>}> : () -> tensor<1xf32>
    %547 = "tosa.reciprocal"(%546) : (tensor<1xf32>) -> tensor<1xf32>
    %548 = "tosa.mul"(%547, %545) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %549 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %550 = "tosa.add"(%548, %549) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %551 = "tosa.rsqrt"(%550) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %552 = "tosa.sub"(%536, %541) : (tensor<1x32x16x16384xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x16384xf32>
    %553 = "tosa.mul"(%552, %551) <{shift = 0 : i8}> : (tensor<1x32x16x16384xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x16384xf32>
    %554 = "tosa.reshape"(%553) <{new_shape = array<i64: 1, 512, 128, 128>}> : (tensor<1x32x16x16384xf32>) -> tensor<1x512x128x128xf32>
    %555 = "tosa.reshape"(%arg57) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %556 = "tosa.reshape"(%555) <{new_shape = array<i64: 1, 512, 1>}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
    %557 = "tosa.reshape"(%556) <{new_shape = array<i64: 1, 512, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1x1xf32>
    %558 = "tosa.reshape"(%arg58) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %559 = "tosa.reshape"(%558) <{new_shape = array<i64: 1, 512, 1>}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
    %560 = "tosa.reshape"(%559) <{new_shape = array<i64: 1, 512, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1x1xf32>
    %561 = "tosa.mul"(%554, %560) <{shift = 0 : i8}> : (tensor<1x512x128x128xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x128x128xf32>
    %562 = "tosa.add"(%561, %557) : (tensor<1x512x128x128xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x128x128xf32>
    %563 = "tosa.sigmoid"(%562) : (tensor<1x512x128x128xf32>) -> tensor<1x512x128x128xf32>
    %564 = "tosa.mul"(%562, %563) <{shift = 0 : i8}> : (tensor<1x512x128x128xf32>, tensor<1x512x128x128xf32>) -> tensor<1x512x128x128xf32>
    %565 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %566 = "tosa.transpose"(%564, %565) : (tensor<1x512x128x128xf32>, tensor<4xi32>) -> tensor<1x128x128x512xf32>
    %567 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %568 = "tosa.transpose"(%arg59, %567) : (tensor<512x512x3x3xf32>, tensor<4xi32>) -> tensor<512x3x3x512xf32>
    %569 = "tosa.conv2d"(%566, %568, %arg60) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x128x128x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>) -> tensor<1x128x128x512xf32>
    %570 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %571 = "tosa.transpose"(%569, %570) : (tensor<1x128x128x512xf32>, tensor<4xi32>) -> tensor<1x512x128x128xf32>
    %572 = "tosa.reshape"(%571) <{new_shape = array<i64: 1, 32, 16, 16384>}> : (tensor<1x512x128x128xf32>) -> tensor<1x32x16x16384xf32>
    %573 = "tosa.reduce_sum"(%572) <{axis = 2 : i32}> : (tensor<1x32x16x16384xf32>) -> tensor<1x32x1x16384xf32>
    %574 = "tosa.reduce_sum"(%573) <{axis = 3 : i32}> : (tensor<1x32x1x16384xf32>) -> tensor<1x32x1x1xf32>
    %575 = "tosa.const"() <{value = dense<2.621440e+05> : tensor<1xf32>}> : () -> tensor<1xf32>
    %576 = "tosa.reciprocal"(%575) : (tensor<1xf32>) -> tensor<1xf32>
    %577 = "tosa.mul"(%576, %574) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %578 = "tosa.sub"(%572, %577) : (tensor<1x32x16x16384xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x16384xf32>
    %579 = "tosa.mul"(%578, %578) <{shift = 0 : i8}> : (tensor<1x32x16x16384xf32>, tensor<1x32x16x16384xf32>) -> tensor<1x32x16x16384xf32>
    %580 = "tosa.reduce_sum"(%579) <{axis = 2 : i32}> : (tensor<1x32x16x16384xf32>) -> tensor<1x32x1x16384xf32>
    %581 = "tosa.reduce_sum"(%580) <{axis = 3 : i32}> : (tensor<1x32x1x16384xf32>) -> tensor<1x32x1x1xf32>
    %582 = "tosa.const"() <{value = dense<2.621440e+05> : tensor<1xf32>}> : () -> tensor<1xf32>
    %583 = "tosa.reciprocal"(%582) : (tensor<1xf32>) -> tensor<1xf32>
    %584 = "tosa.mul"(%583, %581) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %585 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %586 = "tosa.add"(%584, %585) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %587 = "tosa.rsqrt"(%586) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %588 = "tosa.sub"(%572, %577) : (tensor<1x32x16x16384xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x16384xf32>
    %589 = "tosa.mul"(%588, %587) <{shift = 0 : i8}> : (tensor<1x32x16x16384xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x16384xf32>
    %590 = "tosa.reshape"(%589) <{new_shape = array<i64: 1, 512, 128, 128>}> : (tensor<1x32x16x16384xf32>) -> tensor<1x512x128x128xf32>
    %591 = "tosa.reshape"(%arg61) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %592 = "tosa.reshape"(%591) <{new_shape = array<i64: 1, 512, 1>}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
    %593 = "tosa.reshape"(%592) <{new_shape = array<i64: 1, 512, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1x1xf32>
    %594 = "tosa.reshape"(%arg62) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %595 = "tosa.reshape"(%594) <{new_shape = array<i64: 1, 512, 1>}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
    %596 = "tosa.reshape"(%595) <{new_shape = array<i64: 1, 512, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1x1xf32>
    %597 = "tosa.mul"(%590, %596) <{shift = 0 : i8}> : (tensor<1x512x128x128xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x128x128xf32>
    %598 = "tosa.add"(%597, %593) : (tensor<1x512x128x128xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x128x128xf32>
    %599 = "tosa.sigmoid"(%598) : (tensor<1x512x128x128xf32>) -> tensor<1x512x128x128xf32>
    %600 = "tosa.mul"(%598, %599) <{shift = 0 : i8}> : (tensor<1x512x128x128xf32>, tensor<1x512x128x128xf32>) -> tensor<1x512x128x128xf32>
    %601 = "tosa.identity"(%600) : (tensor<1x512x128x128xf32>) -> tensor<1x512x128x128xf32>
    %602 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %603 = "tosa.transpose"(%601, %602) : (tensor<1x512x128x128xf32>, tensor<4xi32>) -> tensor<1x128x128x512xf32>
    %604 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %605 = "tosa.transpose"(%arg63, %604) : (tensor<512x512x3x3xf32>, tensor<4xi32>) -> tensor<512x3x3x512xf32>
    %606 = "tosa.conv2d"(%603, %605, %arg64) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x128x128x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>) -> tensor<1x128x128x512xf32>
    %607 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %608 = "tosa.transpose"(%606, %607) : (tensor<1x128x128x512xf32>, tensor<4xi32>) -> tensor<1x512x128x128xf32>
    %609 = "tosa.add"(%534, %608) : (tensor<1x512x128x128xf32>, tensor<1x512x128x128xf32>) -> tensor<1x512x128x128xf32>
    %610 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x512x128x128xf32>}> : () -> tensor<1x512x128x128xf32>
    %611 = "tosa.reciprocal"(%610) : (tensor<1x512x128x128xf32>) -> tensor<1x512x128x128xf32>
    %612 = "tosa.mul"(%609, %611) <{shift = 0 : i8}> : (tensor<1x512x128x128xf32>, tensor<1x512x128x128xf32>) -> tensor<1x512x128x128xf32>
    %613 = "tosa.identity"(%612) : (tensor<1x512x128x128xf32>) -> tensor<1x512x128x128xf32>
    %614 = "tosa.reshape"(%613) <{new_shape = array<i64: 1, 32, 16, 16384>}> : (tensor<1x512x128x128xf32>) -> tensor<1x32x16x16384xf32>
    %615 = "tosa.reduce_sum"(%614) <{axis = 2 : i32}> : (tensor<1x32x16x16384xf32>) -> tensor<1x32x1x16384xf32>
    %616 = "tosa.reduce_sum"(%615) <{axis = 3 : i32}> : (tensor<1x32x1x16384xf32>) -> tensor<1x32x1x1xf32>
    %617 = "tosa.const"() <{value = dense<2.621440e+05> : tensor<1xf32>}> : () -> tensor<1xf32>
    %618 = "tosa.reciprocal"(%617) : (tensor<1xf32>) -> tensor<1xf32>
    %619 = "tosa.mul"(%618, %616) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %620 = "tosa.sub"(%614, %619) : (tensor<1x32x16x16384xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x16384xf32>
    %621 = "tosa.mul"(%620, %620) <{shift = 0 : i8}> : (tensor<1x32x16x16384xf32>, tensor<1x32x16x16384xf32>) -> tensor<1x32x16x16384xf32>
    %622 = "tosa.reduce_sum"(%621) <{axis = 2 : i32}> : (tensor<1x32x16x16384xf32>) -> tensor<1x32x1x16384xf32>
    %623 = "tosa.reduce_sum"(%622) <{axis = 3 : i32}> : (tensor<1x32x1x16384xf32>) -> tensor<1x32x1x1xf32>
    %624 = "tosa.const"() <{value = dense<2.621440e+05> : tensor<1xf32>}> : () -> tensor<1xf32>
    %625 = "tosa.reciprocal"(%624) : (tensor<1xf32>) -> tensor<1xf32>
    %626 = "tosa.mul"(%625, %623) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %627 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %628 = "tosa.add"(%626, %627) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %629 = "tosa.rsqrt"(%628) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %630 = "tosa.sub"(%614, %619) : (tensor<1x32x16x16384xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x16384xf32>
    %631 = "tosa.mul"(%630, %629) <{shift = 0 : i8}> : (tensor<1x32x16x16384xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x16384xf32>
    %632 = "tosa.reshape"(%631) <{new_shape = array<i64: 1, 512, 128, 128>}> : (tensor<1x32x16x16384xf32>) -> tensor<1x512x128x128xf32>
    %633 = "tosa.reshape"(%arg65) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %634 = "tosa.reshape"(%633) <{new_shape = array<i64: 1, 512, 1>}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
    %635 = "tosa.reshape"(%634) <{new_shape = array<i64: 1, 512, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1x1xf32>
    %636 = "tosa.reshape"(%arg66) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %637 = "tosa.reshape"(%636) <{new_shape = array<i64: 1, 512, 1>}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
    %638 = "tosa.reshape"(%637) <{new_shape = array<i64: 1, 512, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1x1xf32>
    %639 = "tosa.mul"(%632, %638) <{shift = 0 : i8}> : (tensor<1x512x128x128xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x128x128xf32>
    %640 = "tosa.add"(%639, %635) : (tensor<1x512x128x128xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x128x128xf32>
    %641 = "tosa.sigmoid"(%640) : (tensor<1x512x128x128xf32>) -> tensor<1x512x128x128xf32>
    %642 = "tosa.mul"(%640, %641) <{shift = 0 : i8}> : (tensor<1x512x128x128xf32>, tensor<1x512x128x128xf32>) -> tensor<1x512x128x128xf32>
    %643 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %644 = "tosa.transpose"(%642, %643) : (tensor<1x512x128x128xf32>, tensor<4xi32>) -> tensor<1x128x128x512xf32>
    %645 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %646 = "tosa.transpose"(%arg67, %645) : (tensor<512x512x3x3xf32>, tensor<4xi32>) -> tensor<512x3x3x512xf32>
    %647 = "tosa.conv2d"(%644, %646, %arg68) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x128x128x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>) -> tensor<1x128x128x512xf32>
    %648 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %649 = "tosa.transpose"(%647, %648) : (tensor<1x128x128x512xf32>, tensor<4xi32>) -> tensor<1x512x128x128xf32>
    %650 = "tosa.reshape"(%649) <{new_shape = array<i64: 1, 32, 16, 16384>}> : (tensor<1x512x128x128xf32>) -> tensor<1x32x16x16384xf32>
    %651 = "tosa.reduce_sum"(%650) <{axis = 2 : i32}> : (tensor<1x32x16x16384xf32>) -> tensor<1x32x1x16384xf32>
    %652 = "tosa.reduce_sum"(%651) <{axis = 3 : i32}> : (tensor<1x32x1x16384xf32>) -> tensor<1x32x1x1xf32>
    %653 = "tosa.const"() <{value = dense<2.621440e+05> : tensor<1xf32>}> : () -> tensor<1xf32>
    %654 = "tosa.reciprocal"(%653) : (tensor<1xf32>) -> tensor<1xf32>
    %655 = "tosa.mul"(%654, %652) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %656 = "tosa.sub"(%650, %655) : (tensor<1x32x16x16384xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x16384xf32>
    %657 = "tosa.mul"(%656, %656) <{shift = 0 : i8}> : (tensor<1x32x16x16384xf32>, tensor<1x32x16x16384xf32>) -> tensor<1x32x16x16384xf32>
    %658 = "tosa.reduce_sum"(%657) <{axis = 2 : i32}> : (tensor<1x32x16x16384xf32>) -> tensor<1x32x1x16384xf32>
    %659 = "tosa.reduce_sum"(%658) <{axis = 3 : i32}> : (tensor<1x32x1x16384xf32>) -> tensor<1x32x1x1xf32>
    %660 = "tosa.const"() <{value = dense<2.621440e+05> : tensor<1xf32>}> : () -> tensor<1xf32>
    %661 = "tosa.reciprocal"(%660) : (tensor<1xf32>) -> tensor<1xf32>
    %662 = "tosa.mul"(%661, %659) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %663 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %664 = "tosa.add"(%662, %663) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %665 = "tosa.rsqrt"(%664) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %666 = "tosa.sub"(%650, %655) : (tensor<1x32x16x16384xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x16384xf32>
    %667 = "tosa.mul"(%666, %665) <{shift = 0 : i8}> : (tensor<1x32x16x16384xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x16384xf32>
    %668 = "tosa.reshape"(%667) <{new_shape = array<i64: 1, 512, 128, 128>}> : (tensor<1x32x16x16384xf32>) -> tensor<1x512x128x128xf32>
    %669 = "tosa.reshape"(%arg69) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %670 = "tosa.reshape"(%669) <{new_shape = array<i64: 1, 512, 1>}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
    %671 = "tosa.reshape"(%670) <{new_shape = array<i64: 1, 512, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1x1xf32>
    %672 = "tosa.reshape"(%arg70) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %673 = "tosa.reshape"(%672) <{new_shape = array<i64: 1, 512, 1>}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
    %674 = "tosa.reshape"(%673) <{new_shape = array<i64: 1, 512, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1x1xf32>
    %675 = "tosa.mul"(%668, %674) <{shift = 0 : i8}> : (tensor<1x512x128x128xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x128x128xf32>
    %676 = "tosa.add"(%675, %671) : (tensor<1x512x128x128xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x128x128xf32>
    %677 = "tosa.sigmoid"(%676) : (tensor<1x512x128x128xf32>) -> tensor<1x512x128x128xf32>
    %678 = "tosa.mul"(%676, %677) <{shift = 0 : i8}> : (tensor<1x512x128x128xf32>, tensor<1x512x128x128xf32>) -> tensor<1x512x128x128xf32>
    %679 = "tosa.identity"(%678) : (tensor<1x512x128x128xf32>) -> tensor<1x512x128x128xf32>
    %680 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %681 = "tosa.transpose"(%679, %680) : (tensor<1x512x128x128xf32>, tensor<4xi32>) -> tensor<1x128x128x512xf32>
    %682 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %683 = "tosa.transpose"(%arg71, %682) : (tensor<512x512x3x3xf32>, tensor<4xi32>) -> tensor<512x3x3x512xf32>
    %684 = "tosa.conv2d"(%681, %683, %arg72) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x128x128x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>) -> tensor<1x128x128x512xf32>
    %685 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %686 = "tosa.transpose"(%684, %685) : (tensor<1x128x128x512xf32>, tensor<4xi32>) -> tensor<1x512x128x128xf32>
    %687 = "tosa.add"(%612, %686) : (tensor<1x512x128x128xf32>, tensor<1x512x128x128xf32>) -> tensor<1x512x128x128xf32>
    %688 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x512x128x128xf32>}> : () -> tensor<1x512x128x128xf32>
    %689 = "tosa.reciprocal"(%688) : (tensor<1x512x128x128xf32>) -> tensor<1x512x128x128xf32>
    %690 = "tosa.mul"(%687, %689) <{shift = 0 : i8}> : (tensor<1x512x128x128xf32>, tensor<1x512x128x128xf32>) -> tensor<1x512x128x128xf32>
    %691 = "tosa.identity"(%690) : (tensor<1x512x128x128xf32>) -> tensor<1x512x128x128xf32>
    %692 = "tosa.reshape"(%691) <{new_shape = array<i64: 1, 32, 16, 16384>}> : (tensor<1x512x128x128xf32>) -> tensor<1x32x16x16384xf32>
    %693 = "tosa.reduce_sum"(%692) <{axis = 2 : i32}> : (tensor<1x32x16x16384xf32>) -> tensor<1x32x1x16384xf32>
    %694 = "tosa.reduce_sum"(%693) <{axis = 3 : i32}> : (tensor<1x32x1x16384xf32>) -> tensor<1x32x1x1xf32>
    %695 = "tosa.const"() <{value = dense<2.621440e+05> : tensor<1xf32>}> : () -> tensor<1xf32>
    %696 = "tosa.reciprocal"(%695) : (tensor<1xf32>) -> tensor<1xf32>
    %697 = "tosa.mul"(%696, %694) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %698 = "tosa.sub"(%692, %697) : (tensor<1x32x16x16384xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x16384xf32>
    %699 = "tosa.mul"(%698, %698) <{shift = 0 : i8}> : (tensor<1x32x16x16384xf32>, tensor<1x32x16x16384xf32>) -> tensor<1x32x16x16384xf32>
    %700 = "tosa.reduce_sum"(%699) <{axis = 2 : i32}> : (tensor<1x32x16x16384xf32>) -> tensor<1x32x1x16384xf32>
    %701 = "tosa.reduce_sum"(%700) <{axis = 3 : i32}> : (tensor<1x32x1x16384xf32>) -> tensor<1x32x1x1xf32>
    %702 = "tosa.const"() <{value = dense<2.621440e+05> : tensor<1xf32>}> : () -> tensor<1xf32>
    %703 = "tosa.reciprocal"(%702) : (tensor<1xf32>) -> tensor<1xf32>
    %704 = "tosa.mul"(%703, %701) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %705 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %706 = "tosa.add"(%704, %705) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %707 = "tosa.rsqrt"(%706) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %708 = "tosa.sub"(%692, %697) : (tensor<1x32x16x16384xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x16384xf32>
    %709 = "tosa.mul"(%708, %707) <{shift = 0 : i8}> : (tensor<1x32x16x16384xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x16384xf32>
    %710 = "tosa.reshape"(%709) <{new_shape = array<i64: 1, 512, 128, 128>}> : (tensor<1x32x16x16384xf32>) -> tensor<1x512x128x128xf32>
    %711 = "tosa.reshape"(%arg73) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %712 = "tosa.reshape"(%711) <{new_shape = array<i64: 1, 512, 1>}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
    %713 = "tosa.reshape"(%712) <{new_shape = array<i64: 1, 512, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1x1xf32>
    %714 = "tosa.reshape"(%arg74) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %715 = "tosa.reshape"(%714) <{new_shape = array<i64: 1, 512, 1>}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
    %716 = "tosa.reshape"(%715) <{new_shape = array<i64: 1, 512, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1x1xf32>
    %717 = "tosa.mul"(%710, %716) <{shift = 0 : i8}> : (tensor<1x512x128x128xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x128x128xf32>
    %718 = "tosa.add"(%717, %713) : (tensor<1x512x128x128xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x128x128xf32>
    %719 = "tosa.sigmoid"(%718) : (tensor<1x512x128x128xf32>) -> tensor<1x512x128x128xf32>
    %720 = "tosa.mul"(%718, %719) <{shift = 0 : i8}> : (tensor<1x512x128x128xf32>, tensor<1x512x128x128xf32>) -> tensor<1x512x128x128xf32>
    %721 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %722 = "tosa.transpose"(%720, %721) : (tensor<1x512x128x128xf32>, tensor<4xi32>) -> tensor<1x128x128x512xf32>
    %723 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %724 = "tosa.transpose"(%arg75, %723) : (tensor<512x512x3x3xf32>, tensor<4xi32>) -> tensor<512x3x3x512xf32>
    %725 = "tosa.conv2d"(%722, %724, %arg76) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x128x128x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>) -> tensor<1x128x128x512xf32>
    %726 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %727 = "tosa.transpose"(%725, %726) : (tensor<1x128x128x512xf32>, tensor<4xi32>) -> tensor<1x512x128x128xf32>
    %728 = "tosa.reshape"(%727) <{new_shape = array<i64: 1, 32, 16, 16384>}> : (tensor<1x512x128x128xf32>) -> tensor<1x32x16x16384xf32>
    %729 = "tosa.reduce_sum"(%728) <{axis = 2 : i32}> : (tensor<1x32x16x16384xf32>) -> tensor<1x32x1x16384xf32>
    %730 = "tosa.reduce_sum"(%729) <{axis = 3 : i32}> : (tensor<1x32x1x16384xf32>) -> tensor<1x32x1x1xf32>
    %731 = "tosa.const"() <{value = dense<2.621440e+05> : tensor<1xf32>}> : () -> tensor<1xf32>
    %732 = "tosa.reciprocal"(%731) : (tensor<1xf32>) -> tensor<1xf32>
    %733 = "tosa.mul"(%732, %730) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %734 = "tosa.sub"(%728, %733) : (tensor<1x32x16x16384xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x16384xf32>
    %735 = "tosa.mul"(%734, %734) <{shift = 0 : i8}> : (tensor<1x32x16x16384xf32>, tensor<1x32x16x16384xf32>) -> tensor<1x32x16x16384xf32>
    %736 = "tosa.reduce_sum"(%735) <{axis = 2 : i32}> : (tensor<1x32x16x16384xf32>) -> tensor<1x32x1x16384xf32>
    %737 = "tosa.reduce_sum"(%736) <{axis = 3 : i32}> : (tensor<1x32x1x16384xf32>) -> tensor<1x32x1x1xf32>
    %738 = "tosa.const"() <{value = dense<2.621440e+05> : tensor<1xf32>}> : () -> tensor<1xf32>
    %739 = "tosa.reciprocal"(%738) : (tensor<1xf32>) -> tensor<1xf32>
    %740 = "tosa.mul"(%739, %737) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %741 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %742 = "tosa.add"(%740, %741) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %743 = "tosa.rsqrt"(%742) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %744 = "tosa.sub"(%728, %733) : (tensor<1x32x16x16384xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x16384xf32>
    %745 = "tosa.mul"(%744, %743) <{shift = 0 : i8}> : (tensor<1x32x16x16384xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x16384xf32>
    %746 = "tosa.reshape"(%745) <{new_shape = array<i64: 1, 512, 128, 128>}> : (tensor<1x32x16x16384xf32>) -> tensor<1x512x128x128xf32>
    %747 = "tosa.reshape"(%arg77) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %748 = "tosa.reshape"(%747) <{new_shape = array<i64: 1, 512, 1>}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
    %749 = "tosa.reshape"(%748) <{new_shape = array<i64: 1, 512, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1x1xf32>
    %750 = "tosa.reshape"(%arg78) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %751 = "tosa.reshape"(%750) <{new_shape = array<i64: 1, 512, 1>}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
    %752 = "tosa.reshape"(%751) <{new_shape = array<i64: 1, 512, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1x1xf32>
    %753 = "tosa.mul"(%746, %752) <{shift = 0 : i8}> : (tensor<1x512x128x128xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x128x128xf32>
    %754 = "tosa.add"(%753, %749) : (tensor<1x512x128x128xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x128x128xf32>
    %755 = "tosa.sigmoid"(%754) : (tensor<1x512x128x128xf32>) -> tensor<1x512x128x128xf32>
    %756 = "tosa.mul"(%754, %755) <{shift = 0 : i8}> : (tensor<1x512x128x128xf32>, tensor<1x512x128x128xf32>) -> tensor<1x512x128x128xf32>
    %757 = "tosa.identity"(%756) : (tensor<1x512x128x128xf32>) -> tensor<1x512x128x128xf32>
    %758 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %759 = "tosa.transpose"(%757, %758) : (tensor<1x512x128x128xf32>, tensor<4xi32>) -> tensor<1x128x128x512xf32>
    %760 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %761 = "tosa.transpose"(%arg79, %760) : (tensor<512x512x3x3xf32>, tensor<4xi32>) -> tensor<512x3x3x512xf32>
    %762 = "tosa.conv2d"(%759, %761, %arg80) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x128x128x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>) -> tensor<1x128x128x512xf32>
    %763 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %764 = "tosa.transpose"(%762, %763) : (tensor<1x128x128x512xf32>, tensor<4xi32>) -> tensor<1x512x128x128xf32>
    %765 = "tosa.add"(%690, %764) : (tensor<1x512x128x128xf32>, tensor<1x512x128x128xf32>) -> tensor<1x512x128x128xf32>
    %766 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x512x128x128xf32>}> : () -> tensor<1x512x128x128xf32>
    %767 = "tosa.reciprocal"(%766) : (tensor<1x512x128x128xf32>) -> tensor<1x512x128x128xf32>
    %768 = "tosa.mul"(%765, %767) <{shift = 0 : i8}> : (tensor<1x512x128x128xf32>, tensor<1x512x128x128xf32>) -> tensor<1x512x128x128xf32>
    %769 = "tosa.const"() <{value = dense<"0x00000000000000000100000000000000020000000000000003000000000000000400000000000000050000000000000006000000000000000700000000000000080000000000000009000000000000000A000000000000000B000000000000000C000000000000000D000000000000000E000000000000000F0000000000000010000000000000001100000000000000120000000000000013000000000000001400000000000000150000000000000016000000000000001700000000000000180000000000000019000000000000001A000000000000001B000000000000001C000000000000001D000000000000001E000000000000001F0000000000000020000000000000002100000000000000220000000000000023000000000000002400000000000000250000000000000026000000000000002700000000000000280000000000000029000000000000002A000000000000002B000000000000002C000000000000002D000000000000002E000000000000002F0000000000000030000000000000003100000000000000320000000000000033000000000000003400000000000000350000000000000036000000000000003700000000000000380000000000000039000000000000003A000000000000003B000000000000003C000000000000003D000000000000003E000000000000003F0000000000000040000000000000004100000000000000420000000000000043000000000000004400000000000000450000000000000046000000000000004700000000000000480000000000000049000000000000004A000000000000004B000000000000004C000000000000004D000000000000004E000000000000004F0000000000000050000000000000005100000000000000520000000000000053000000000000005400000000000000550000000000000056000000000000005700000000000000580000000000000059000000000000005A000000000000005B000000000000005C000000000000005D000000000000005E000000000000005F0000000000000060000000000000006100000000000000620000000000000063000000000000006400000000000000650000000000000066000000000000006700000000000000680000000000000069000000000000006A000000000000006B000000000000006C000000000000006D000000000000006E000000000000006F0000000000000070000000000000007100000000000000720000000000000073000000000000007400000000000000750000000000000076000000000000007700000000000000780000000000000079000000000000007A000000000000007B000000000000007C000000000000007D000000000000007E000000000000007F0000000000000080000000000000008100000000000000820000000000000083000000000000008400000000000000850000000000000086000000000000008700000000000000880000000000000089000000000000008A000000000000008B000000000000008C000000000000008D000000000000008E000000000000008F0000000000000090000000000000009100000000000000920000000000000093000000000000009400000000000000950000000000000096000000000000009700000000000000980000000000000099000000000000009A000000000000009B000000000000009C000000000000009D000000000000009E000000000000009F00000000000000A000000000000000A100000000000000A200000000000000A300000000000000A400000000000000A500000000000000A600000000000000A700000000000000A800000000000000A900000000000000AA00000000000000AB00000000000000AC00000000000000AD00000000000000AE00000000000000AF00000000000000B000000000000000B100000000000000B200000000000000B300000000000000B400000000000000B500000000000000B600000000000000B700000000000000B800000000000000B900000000000000BA00000000000000BB00000000000000BC00000000000000BD00000000000000BE00000000000000BF00000000000000C000000000000000C100000000000000C200000000000000C300000000000000C400000000000000C500000000000000C600000000000000C700000000000000C800000000000000C900000000000000CA00000000000000CB00000000000000CC00000000000000CD00000000000000CE00000000000000CF00000000000000D000000000000000D100000000000000D200000000000000D300000000000000D400000000000000D500000000000000D600000000000000D700000000000000D800000000000000D900000000000000DA00000000000000DB00000000000000DC00000000000000DD00000000000000DE00000000000000DF00000000000000E000000000000000E100000000000000E200000000000000E300000000000000E400000000000000E500000000000000E600000000000000E700000000000000E800000000000000E900000000000000EA00000000000000EB00000000000000EC00000000000000ED00000000000000EE00000000000000EF00000000000000F000000000000000F100000000000000F200000000000000F300000000000000F400000000000000F500000000000000F600000000000000F700000000000000F800000000000000F900000000000000FA00000000000000FB00000000000000FC00000000000000FD00000000000000FE00000000000000FF00000000000000"> : tensor<256xi64>}> : () -> tensor<256xi64>
    %770 = "tosa.cast"(%769) : (tensor<256xi64>) -> tensor<256xf32>
    %771 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<256xf32>}> : () -> tensor<256xf32>
    %772 = "tosa.mul"(%770, %771) <{shift = 0 : i8}> : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %773 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<256xf32>}> : () -> tensor<256xf32>
    %774 = "tosa.add"(%772, %773) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %775 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<256xf32>}> : () -> tensor<256xf32>
    %776 = "tosa.mul"(%774, %775) <{shift = 0 : i8}> : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %777 = "tosa.cast"(%776) : (tensor<256xf32>) -> tensor<256xi64>
    %778 = "tosa.reshape"(%777) <{new_shape = array<i64: 256, 1>}> : (tensor<256xi64>) -> tensor<256x1xi64>
    %779 = "tosa.const"() <{value = dense<"0x00000000000000000100000000000000020000000000000003000000000000000400000000000000050000000000000006000000000000000700000000000000080000000000000009000000000000000A000000000000000B000000000000000C000000000000000D000000000000000E000000000000000F0000000000000010000000000000001100000000000000120000000000000013000000000000001400000000000000150000000000000016000000000000001700000000000000180000000000000019000000000000001A000000000000001B000000000000001C000000000000001D000000000000001E000000000000001F0000000000000020000000000000002100000000000000220000000000000023000000000000002400000000000000250000000000000026000000000000002700000000000000280000000000000029000000000000002A000000000000002B000000000000002C000000000000002D000000000000002E000000000000002F0000000000000030000000000000003100000000000000320000000000000033000000000000003400000000000000350000000000000036000000000000003700000000000000380000000000000039000000000000003A000000000000003B000000000000003C000000000000003D000000000000003E000000000000003F0000000000000040000000000000004100000000000000420000000000000043000000000000004400000000000000450000000000000046000000000000004700000000000000480000000000000049000000000000004A000000000000004B000000000000004C000000000000004D000000000000004E000000000000004F0000000000000050000000000000005100000000000000520000000000000053000000000000005400000000000000550000000000000056000000000000005700000000000000580000000000000059000000000000005A000000000000005B000000000000005C000000000000005D000000000000005E000000000000005F0000000000000060000000000000006100000000000000620000000000000063000000000000006400000000000000650000000000000066000000000000006700000000000000680000000000000069000000000000006A000000000000006B000000000000006C000000000000006D000000000000006E000000000000006F0000000000000070000000000000007100000000000000720000000000000073000000000000007400000000000000750000000000000076000000000000007700000000000000780000000000000079000000000000007A000000000000007B000000000000007C000000000000007D000000000000007E000000000000007F0000000000000080000000000000008100000000000000820000000000000083000000000000008400000000000000850000000000000086000000000000008700000000000000880000000000000089000000000000008A000000000000008B000000000000008C000000000000008D000000000000008E000000000000008F0000000000000090000000000000009100000000000000920000000000000093000000000000009400000000000000950000000000000096000000000000009700000000000000980000000000000099000000000000009A000000000000009B000000000000009C000000000000009D000000000000009E000000000000009F00000000000000A000000000000000A100000000000000A200000000000000A300000000000000A400000000000000A500000000000000A600000000000000A700000000000000A800000000000000A900000000000000AA00000000000000AB00000000000000AC00000000000000AD00000000000000AE00000000000000AF00000000000000B000000000000000B100000000000000B200000000000000B300000000000000B400000000000000B500000000000000B600000000000000B700000000000000B800000000000000B900000000000000BA00000000000000BB00000000000000BC00000000000000BD00000000000000BE00000000000000BF00000000000000C000000000000000C100000000000000C200000000000000C300000000000000C400000000000000C500000000000000C600000000000000C700000000000000C800000000000000C900000000000000CA00000000000000CB00000000000000CC00000000000000CD00000000000000CE00000000000000CF00000000000000D000000000000000D100000000000000D200000000000000D300000000000000D400000000000000D500000000000000D600000000000000D700000000000000D800000000000000D900000000000000DA00000000000000DB00000000000000DC00000000000000DD00000000000000DE00000000000000DF00000000000000E000000000000000E100000000000000E200000000000000E300000000000000E400000000000000E500000000000000E600000000000000E700000000000000E800000000000000E900000000000000EA00000000000000EB00000000000000EC00000000000000ED00000000000000EE00000000000000EF00000000000000F000000000000000F100000000000000F200000000000000F300000000000000F400000000000000F500000000000000F600000000000000F700000000000000F800000000000000F900000000000000FA00000000000000FB00000000000000FC00000000000000FD00000000000000FE00000000000000FF00000000000000"> : tensor<256xi64>}> : () -> tensor<256xi64>
    %780 = "tosa.cast"(%779) : (tensor<256xi64>) -> tensor<256xf32>
    %781 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<256xf32>}> : () -> tensor<256xf32>
    %782 = "tosa.mul"(%780, %781) <{shift = 0 : i8}> : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %783 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<256xf32>}> : () -> tensor<256xf32>
    %784 = "tosa.add"(%782, %783) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %785 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<256xf32>}> : () -> tensor<256xf32>
    %786 = "tosa.mul"(%784, %785) <{shift = 0 : i8}> : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %787 = "tosa.cast"(%786) : (tensor<256xf32>) -> tensor<256xi64>
    %788 = "tensor.empty"() : () -> tensor<1x512x256x256xf32>
    %789 = "linalg.generic"(%778, %787, %788) <{indexing_maps = [#map3, #map3, #map4], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg141: i64, %arg142: i64, %arg143: f32):
      %1346 = "arith.index_cast"(%arg141) : (i64) -> index
      %1347 = "arith.index_cast"(%arg142) : (i64) -> index
      %1348 = "linalg.index"() <{dim = 2 : i64}> : () -> index
      %1349 = "tensor.extract"(%768, %1346, %1347, %1348) : (tensor<1x512x128x128xf32>, index, index, index) -> f32
      "linalg.yield"(%1349) : (f32) -> ()
    }) : (tensor<256x1xi64>, tensor<256xi64>, tensor<1x512x256x256xf32>) -> tensor<1x512x256x256xf32>
    %790 = "tosa.identity"(%789) : (tensor<1x512x256x256xf32>) -> tensor<1x512x256x256xf32>
    %791 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %792 = "tosa.transpose"(%790, %791) : (tensor<1x512x256x256xf32>, tensor<4xi32>) -> tensor<1x256x256x512xf32>
    %793 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %794 = "tosa.transpose"(%arg81, %793) : (tensor<512x512x3x3xf32>, tensor<4xi32>) -> tensor<512x3x3x512xf32>
    %795 = "tosa.conv2d"(%792, %794, %arg82) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x256x256x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>) -> tensor<1x256x256x512xf32>
    %796 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %797 = "tosa.transpose"(%795, %796) : (tensor<1x256x256x512xf32>, tensor<4xi32>) -> tensor<1x512x256x256xf32>
    %798 = "tosa.identity"(%797) : (tensor<1x512x256x256xf32>) -> tensor<1x512x256x256xf32>
    %799 = "tosa.reshape"(%798) <{new_shape = array<i64: 1, 32, 16, 65536>}> : (tensor<1x512x256x256xf32>) -> tensor<1x32x16x65536xf32>
    %800 = "tosa.reduce_sum"(%799) <{axis = 2 : i32}> : (tensor<1x32x16x65536xf32>) -> tensor<1x32x1x65536xf32>
    %801 = "tosa.reduce_sum"(%800) <{axis = 3 : i32}> : (tensor<1x32x1x65536xf32>) -> tensor<1x32x1x1xf32>
    %802 = "tosa.const"() <{value = dense<0x49800000> : tensor<1xf32>}> : () -> tensor<1xf32>
    %803 = "tosa.reciprocal"(%802) : (tensor<1xf32>) -> tensor<1xf32>
    %804 = "tosa.mul"(%803, %801) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %805 = "tosa.sub"(%799, %804) : (tensor<1x32x16x65536xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x65536xf32>
    %806 = "tosa.mul"(%805, %805) <{shift = 0 : i8}> : (tensor<1x32x16x65536xf32>, tensor<1x32x16x65536xf32>) -> tensor<1x32x16x65536xf32>
    %807 = "tosa.reduce_sum"(%806) <{axis = 2 : i32}> : (tensor<1x32x16x65536xf32>) -> tensor<1x32x1x65536xf32>
    %808 = "tosa.reduce_sum"(%807) <{axis = 3 : i32}> : (tensor<1x32x1x65536xf32>) -> tensor<1x32x1x1xf32>
    %809 = "tosa.const"() <{value = dense<0x49800000> : tensor<1xf32>}> : () -> tensor<1xf32>
    %810 = "tosa.reciprocal"(%809) : (tensor<1xf32>) -> tensor<1xf32>
    %811 = "tosa.mul"(%810, %808) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %812 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %813 = "tosa.add"(%811, %812) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %814 = "tosa.rsqrt"(%813) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %815 = "tosa.sub"(%799, %804) : (tensor<1x32x16x65536xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x65536xf32>
    %816 = "tosa.mul"(%815, %814) <{shift = 0 : i8}> : (tensor<1x32x16x65536xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x16x65536xf32>
    %817 = "tosa.reshape"(%816) <{new_shape = array<i64: 1, 512, 256, 256>}> : (tensor<1x32x16x65536xf32>) -> tensor<1x512x256x256xf32>
    %818 = "tosa.reshape"(%arg83) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %819 = "tosa.reshape"(%818) <{new_shape = array<i64: 1, 512, 1>}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
    %820 = "tosa.reshape"(%819) <{new_shape = array<i64: 1, 512, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1x1xf32>
    %821 = "tosa.reshape"(%arg84) <{new_shape = array<i64: 1, 512>}> : (tensor<512xf32>) -> tensor<1x512xf32>
    %822 = "tosa.reshape"(%821) <{new_shape = array<i64: 1, 512, 1>}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
    %823 = "tosa.reshape"(%822) <{new_shape = array<i64: 1, 512, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1x1xf32>
    %824 = "tosa.mul"(%817, %823) <{shift = 0 : i8}> : (tensor<1x512x256x256xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x256x256xf32>
    %825 = "tosa.add"(%824, %820) : (tensor<1x512x256x256xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x256x256xf32>
    %826 = "tosa.sigmoid"(%825) : (tensor<1x512x256x256xf32>) -> tensor<1x512x256x256xf32>
    %827 = "tosa.mul"(%825, %826) <{shift = 0 : i8}> : (tensor<1x512x256x256xf32>, tensor<1x512x256x256xf32>) -> tensor<1x512x256x256xf32>
    %828 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %829 = "tosa.transpose"(%827, %828) : (tensor<1x512x256x256xf32>, tensor<4xi32>) -> tensor<1x256x256x512xf32>
    %830 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %831 = "tosa.transpose"(%arg85, %830) : (tensor<256x512x3x3xf32>, tensor<4xi32>) -> tensor<256x3x3x512xf32>
    %832 = "tosa.conv2d"(%829, %831, %arg86) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x256x256x512xf32>, tensor<256x3x3x512xf32>, tensor<256xf32>) -> tensor<1x256x256x256xf32>
    %833 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %834 = "tosa.transpose"(%832, %833) : (tensor<1x256x256x256xf32>, tensor<4xi32>) -> tensor<1x256x256x256xf32>
    %835 = "tosa.reshape"(%834) <{new_shape = array<i64: 1, 32, 8, 65536>}> : (tensor<1x256x256x256xf32>) -> tensor<1x32x8x65536xf32>
    %836 = "tosa.reduce_sum"(%835) <{axis = 2 : i32}> : (tensor<1x32x8x65536xf32>) -> tensor<1x32x1x65536xf32>
    %837 = "tosa.reduce_sum"(%836) <{axis = 3 : i32}> : (tensor<1x32x1x65536xf32>) -> tensor<1x32x1x1xf32>
    %838 = "tosa.const"() <{value = dense<5.242880e+05> : tensor<1xf32>}> : () -> tensor<1xf32>
    %839 = "tosa.reciprocal"(%838) : (tensor<1xf32>) -> tensor<1xf32>
    %840 = "tosa.mul"(%839, %837) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %841 = "tosa.sub"(%835, %840) : (tensor<1x32x8x65536xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x8x65536xf32>
    %842 = "tosa.mul"(%841, %841) <{shift = 0 : i8}> : (tensor<1x32x8x65536xf32>, tensor<1x32x8x65536xf32>) -> tensor<1x32x8x65536xf32>
    %843 = "tosa.reduce_sum"(%842) <{axis = 2 : i32}> : (tensor<1x32x8x65536xf32>) -> tensor<1x32x1x65536xf32>
    %844 = "tosa.reduce_sum"(%843) <{axis = 3 : i32}> : (tensor<1x32x1x65536xf32>) -> tensor<1x32x1x1xf32>
    %845 = "tosa.const"() <{value = dense<5.242880e+05> : tensor<1xf32>}> : () -> tensor<1xf32>
    %846 = "tosa.reciprocal"(%845) : (tensor<1xf32>) -> tensor<1xf32>
    %847 = "tosa.mul"(%846, %844) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %848 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %849 = "tosa.add"(%847, %848) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %850 = "tosa.rsqrt"(%849) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %851 = "tosa.sub"(%835, %840) : (tensor<1x32x8x65536xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x8x65536xf32>
    %852 = "tosa.mul"(%851, %850) <{shift = 0 : i8}> : (tensor<1x32x8x65536xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x8x65536xf32>
    %853 = "tosa.reshape"(%852) <{new_shape = array<i64: 1, 256, 256, 256>}> : (tensor<1x32x8x65536xf32>) -> tensor<1x256x256x256xf32>
    %854 = "tosa.reshape"(%arg87) <{new_shape = array<i64: 1, 256>}> : (tensor<256xf32>) -> tensor<1x256xf32>
    %855 = "tosa.reshape"(%854) <{new_shape = array<i64: 1, 256, 1>}> : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %856 = "tosa.reshape"(%855) <{new_shape = array<i64: 1, 256, 1, 1>}> : (tensor<1x256x1xf32>) -> tensor<1x256x1x1xf32>
    %857 = "tosa.reshape"(%arg88) <{new_shape = array<i64: 1, 256>}> : (tensor<256xf32>) -> tensor<1x256xf32>
    %858 = "tosa.reshape"(%857) <{new_shape = array<i64: 1, 256, 1>}> : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %859 = "tosa.reshape"(%858) <{new_shape = array<i64: 1, 256, 1, 1>}> : (tensor<1x256x1xf32>) -> tensor<1x256x1x1xf32>
    %860 = "tosa.mul"(%853, %859) <{shift = 0 : i8}> : (tensor<1x256x256x256xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x256x256xf32>
    %861 = "tosa.add"(%860, %856) : (tensor<1x256x256x256xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x256x256xf32>
    %862 = "tosa.sigmoid"(%861) : (tensor<1x256x256x256xf32>) -> tensor<1x256x256x256xf32>
    %863 = "tosa.mul"(%861, %862) <{shift = 0 : i8}> : (tensor<1x256x256x256xf32>, tensor<1x256x256x256xf32>) -> tensor<1x256x256x256xf32>
    %864 = "tosa.identity"(%863) : (tensor<1x256x256x256xf32>) -> tensor<1x256x256x256xf32>
    %865 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %866 = "tosa.transpose"(%864, %865) : (tensor<1x256x256x256xf32>, tensor<4xi32>) -> tensor<1x256x256x256xf32>
    %867 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %868 = "tosa.transpose"(%arg89, %867) : (tensor<256x256x3x3xf32>, tensor<4xi32>) -> tensor<256x3x3x256xf32>
    %869 = "tosa.conv2d"(%866, %868, %arg90) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x256x256x256xf32>, tensor<256x3x3x256xf32>, tensor<256xf32>) -> tensor<1x256x256x256xf32>
    %870 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %871 = "tosa.transpose"(%869, %870) : (tensor<1x256x256x256xf32>, tensor<4xi32>) -> tensor<1x256x256x256xf32>
    %872 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %873 = "tosa.transpose"(%797, %872) : (tensor<1x512x256x256xf32>, tensor<4xi32>) -> tensor<1x256x256x512xf32>
    %874 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %875 = "tosa.transpose"(%arg91, %874) : (tensor<256x512x1x1xf32>, tensor<4xi32>) -> tensor<256x1x1x512xf32>
    %876 = "tosa.conv2d"(%873, %875, %arg92) <{dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}> : (tensor<1x256x256x512xf32>, tensor<256x1x1x512xf32>, tensor<256xf32>) -> tensor<1x256x256x256xf32>
    %877 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %878 = "tosa.transpose"(%876, %877) : (tensor<1x256x256x256xf32>, tensor<4xi32>) -> tensor<1x256x256x256xf32>
    %879 = "tosa.add"(%878, %871) : (tensor<1x256x256x256xf32>, tensor<1x256x256x256xf32>) -> tensor<1x256x256x256xf32>
    %880 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x256x256x256xf32>}> : () -> tensor<1x256x256x256xf32>
    %881 = "tosa.reciprocal"(%880) : (tensor<1x256x256x256xf32>) -> tensor<1x256x256x256xf32>
    %882 = "tosa.mul"(%879, %881) <{shift = 0 : i8}> : (tensor<1x256x256x256xf32>, tensor<1x256x256x256xf32>) -> tensor<1x256x256x256xf32>
    %883 = "tosa.identity"(%882) : (tensor<1x256x256x256xf32>) -> tensor<1x256x256x256xf32>
    %884 = "tosa.reshape"(%883) <{new_shape = array<i64: 1, 32, 8, 65536>}> : (tensor<1x256x256x256xf32>) -> tensor<1x32x8x65536xf32>
    %885 = "tosa.reduce_sum"(%884) <{axis = 2 : i32}> : (tensor<1x32x8x65536xf32>) -> tensor<1x32x1x65536xf32>
    %886 = "tosa.reduce_sum"(%885) <{axis = 3 : i32}> : (tensor<1x32x1x65536xf32>) -> tensor<1x32x1x1xf32>
    %887 = "tosa.const"() <{value = dense<5.242880e+05> : tensor<1xf32>}> : () -> tensor<1xf32>
    %888 = "tosa.reciprocal"(%887) : (tensor<1xf32>) -> tensor<1xf32>
    %889 = "tosa.mul"(%888, %886) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %890 = "tosa.sub"(%884, %889) : (tensor<1x32x8x65536xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x8x65536xf32>
    %891 = "tosa.mul"(%890, %890) <{shift = 0 : i8}> : (tensor<1x32x8x65536xf32>, tensor<1x32x8x65536xf32>) -> tensor<1x32x8x65536xf32>
    %892 = "tosa.reduce_sum"(%891) <{axis = 2 : i32}> : (tensor<1x32x8x65536xf32>) -> tensor<1x32x1x65536xf32>
    %893 = "tosa.reduce_sum"(%892) <{axis = 3 : i32}> : (tensor<1x32x1x65536xf32>) -> tensor<1x32x1x1xf32>
    %894 = "tosa.const"() <{value = dense<5.242880e+05> : tensor<1xf32>}> : () -> tensor<1xf32>
    %895 = "tosa.reciprocal"(%894) : (tensor<1xf32>) -> tensor<1xf32>
    %896 = "tosa.mul"(%895, %893) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %897 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %898 = "tosa.add"(%896, %897) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %899 = "tosa.rsqrt"(%898) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %900 = "tosa.sub"(%884, %889) : (tensor<1x32x8x65536xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x8x65536xf32>
    %901 = "tosa.mul"(%900, %899) <{shift = 0 : i8}> : (tensor<1x32x8x65536xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x8x65536xf32>
    %902 = "tosa.reshape"(%901) <{new_shape = array<i64: 1, 256, 256, 256>}> : (tensor<1x32x8x65536xf32>) -> tensor<1x256x256x256xf32>
    %903 = "tosa.reshape"(%arg93) <{new_shape = array<i64: 1, 256>}> : (tensor<256xf32>) -> tensor<1x256xf32>
    %904 = "tosa.reshape"(%903) <{new_shape = array<i64: 1, 256, 1>}> : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %905 = "tosa.reshape"(%904) <{new_shape = array<i64: 1, 256, 1, 1>}> : (tensor<1x256x1xf32>) -> tensor<1x256x1x1xf32>
    %906 = "tosa.reshape"(%arg94) <{new_shape = array<i64: 1, 256>}> : (tensor<256xf32>) -> tensor<1x256xf32>
    %907 = "tosa.reshape"(%906) <{new_shape = array<i64: 1, 256, 1>}> : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %908 = "tosa.reshape"(%907) <{new_shape = array<i64: 1, 256, 1, 1>}> : (tensor<1x256x1xf32>) -> tensor<1x256x1x1xf32>
    %909 = "tosa.mul"(%902, %908) <{shift = 0 : i8}> : (tensor<1x256x256x256xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x256x256xf32>
    %910 = "tosa.add"(%909, %905) : (tensor<1x256x256x256xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x256x256xf32>
    %911 = "tosa.sigmoid"(%910) : (tensor<1x256x256x256xf32>) -> tensor<1x256x256x256xf32>
    %912 = "tosa.mul"(%910, %911) <{shift = 0 : i8}> : (tensor<1x256x256x256xf32>, tensor<1x256x256x256xf32>) -> tensor<1x256x256x256xf32>
    %913 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %914 = "tosa.transpose"(%912, %913) : (tensor<1x256x256x256xf32>, tensor<4xi32>) -> tensor<1x256x256x256xf32>
    %915 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %916 = "tosa.transpose"(%arg95, %915) : (tensor<256x256x3x3xf32>, tensor<4xi32>) -> tensor<256x3x3x256xf32>
    %917 = "tosa.conv2d"(%914, %916, %arg96) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x256x256x256xf32>, tensor<256x3x3x256xf32>, tensor<256xf32>) -> tensor<1x256x256x256xf32>
    %918 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %919 = "tosa.transpose"(%917, %918) : (tensor<1x256x256x256xf32>, tensor<4xi32>) -> tensor<1x256x256x256xf32>
    %920 = "tosa.reshape"(%919) <{new_shape = array<i64: 1, 32, 8, 65536>}> : (tensor<1x256x256x256xf32>) -> tensor<1x32x8x65536xf32>
    %921 = "tosa.reduce_sum"(%920) <{axis = 2 : i32}> : (tensor<1x32x8x65536xf32>) -> tensor<1x32x1x65536xf32>
    %922 = "tosa.reduce_sum"(%921) <{axis = 3 : i32}> : (tensor<1x32x1x65536xf32>) -> tensor<1x32x1x1xf32>
    %923 = "tosa.const"() <{value = dense<5.242880e+05> : tensor<1xf32>}> : () -> tensor<1xf32>
    %924 = "tosa.reciprocal"(%923) : (tensor<1xf32>) -> tensor<1xf32>
    %925 = "tosa.mul"(%924, %922) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %926 = "tosa.sub"(%920, %925) : (tensor<1x32x8x65536xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x8x65536xf32>
    %927 = "tosa.mul"(%926, %926) <{shift = 0 : i8}> : (tensor<1x32x8x65536xf32>, tensor<1x32x8x65536xf32>) -> tensor<1x32x8x65536xf32>
    %928 = "tosa.reduce_sum"(%927) <{axis = 2 : i32}> : (tensor<1x32x8x65536xf32>) -> tensor<1x32x1x65536xf32>
    %929 = "tosa.reduce_sum"(%928) <{axis = 3 : i32}> : (tensor<1x32x1x65536xf32>) -> tensor<1x32x1x1xf32>
    %930 = "tosa.const"() <{value = dense<5.242880e+05> : tensor<1xf32>}> : () -> tensor<1xf32>
    %931 = "tosa.reciprocal"(%930) : (tensor<1xf32>) -> tensor<1xf32>
    %932 = "tosa.mul"(%931, %929) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %933 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %934 = "tosa.add"(%932, %933) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %935 = "tosa.rsqrt"(%934) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %936 = "tosa.sub"(%920, %925) : (tensor<1x32x8x65536xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x8x65536xf32>
    %937 = "tosa.mul"(%936, %935) <{shift = 0 : i8}> : (tensor<1x32x8x65536xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x8x65536xf32>
    %938 = "tosa.reshape"(%937) <{new_shape = array<i64: 1, 256, 256, 256>}> : (tensor<1x32x8x65536xf32>) -> tensor<1x256x256x256xf32>
    %939 = "tosa.reshape"(%arg97) <{new_shape = array<i64: 1, 256>}> : (tensor<256xf32>) -> tensor<1x256xf32>
    %940 = "tosa.reshape"(%939) <{new_shape = array<i64: 1, 256, 1>}> : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %941 = "tosa.reshape"(%940) <{new_shape = array<i64: 1, 256, 1, 1>}> : (tensor<1x256x1xf32>) -> tensor<1x256x1x1xf32>
    %942 = "tosa.reshape"(%arg98) <{new_shape = array<i64: 1, 256>}> : (tensor<256xf32>) -> tensor<1x256xf32>
    %943 = "tosa.reshape"(%942) <{new_shape = array<i64: 1, 256, 1>}> : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %944 = "tosa.reshape"(%943) <{new_shape = array<i64: 1, 256, 1, 1>}> : (tensor<1x256x1xf32>) -> tensor<1x256x1x1xf32>
    %945 = "tosa.mul"(%938, %944) <{shift = 0 : i8}> : (tensor<1x256x256x256xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x256x256xf32>
    %946 = "tosa.add"(%945, %941) : (tensor<1x256x256x256xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x256x256xf32>
    %947 = "tosa.sigmoid"(%946) : (tensor<1x256x256x256xf32>) -> tensor<1x256x256x256xf32>
    %948 = "tosa.mul"(%946, %947) <{shift = 0 : i8}> : (tensor<1x256x256x256xf32>, tensor<1x256x256x256xf32>) -> tensor<1x256x256x256xf32>
    %949 = "tosa.identity"(%948) : (tensor<1x256x256x256xf32>) -> tensor<1x256x256x256xf32>
    %950 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %951 = "tosa.transpose"(%949, %950) : (tensor<1x256x256x256xf32>, tensor<4xi32>) -> tensor<1x256x256x256xf32>
    %952 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %953 = "tosa.transpose"(%arg99, %952) : (tensor<256x256x3x3xf32>, tensor<4xi32>) -> tensor<256x3x3x256xf32>
    %954 = "tosa.conv2d"(%951, %953, %arg100) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x256x256x256xf32>, tensor<256x3x3x256xf32>, tensor<256xf32>) -> tensor<1x256x256x256xf32>
    %955 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %956 = "tosa.transpose"(%954, %955) : (tensor<1x256x256x256xf32>, tensor<4xi32>) -> tensor<1x256x256x256xf32>
    %957 = "tosa.add"(%882, %956) : (tensor<1x256x256x256xf32>, tensor<1x256x256x256xf32>) -> tensor<1x256x256x256xf32>
    %958 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x256x256x256xf32>}> : () -> tensor<1x256x256x256xf32>
    %959 = "tosa.reciprocal"(%958) : (tensor<1x256x256x256xf32>) -> tensor<1x256x256x256xf32>
    %960 = "tosa.mul"(%957, %959) <{shift = 0 : i8}> : (tensor<1x256x256x256xf32>, tensor<1x256x256x256xf32>) -> tensor<1x256x256x256xf32>
    %961 = "tosa.identity"(%960) : (tensor<1x256x256x256xf32>) -> tensor<1x256x256x256xf32>
    %962 = "tosa.reshape"(%961) <{new_shape = array<i64: 1, 32, 8, 65536>}> : (tensor<1x256x256x256xf32>) -> tensor<1x32x8x65536xf32>
    %963 = "tosa.reduce_sum"(%962) <{axis = 2 : i32}> : (tensor<1x32x8x65536xf32>) -> tensor<1x32x1x65536xf32>
    %964 = "tosa.reduce_sum"(%963) <{axis = 3 : i32}> : (tensor<1x32x1x65536xf32>) -> tensor<1x32x1x1xf32>
    %965 = "tosa.const"() <{value = dense<5.242880e+05> : tensor<1xf32>}> : () -> tensor<1xf32>
    %966 = "tosa.reciprocal"(%965) : (tensor<1xf32>) -> tensor<1xf32>
    %967 = "tosa.mul"(%966, %964) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %968 = "tosa.sub"(%962, %967) : (tensor<1x32x8x65536xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x8x65536xf32>
    %969 = "tosa.mul"(%968, %968) <{shift = 0 : i8}> : (tensor<1x32x8x65536xf32>, tensor<1x32x8x65536xf32>) -> tensor<1x32x8x65536xf32>
    %970 = "tosa.reduce_sum"(%969) <{axis = 2 : i32}> : (tensor<1x32x8x65536xf32>) -> tensor<1x32x1x65536xf32>
    %971 = "tosa.reduce_sum"(%970) <{axis = 3 : i32}> : (tensor<1x32x1x65536xf32>) -> tensor<1x32x1x1xf32>
    %972 = "tosa.const"() <{value = dense<5.242880e+05> : tensor<1xf32>}> : () -> tensor<1xf32>
    %973 = "tosa.reciprocal"(%972) : (tensor<1xf32>) -> tensor<1xf32>
    %974 = "tosa.mul"(%973, %971) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %975 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %976 = "tosa.add"(%974, %975) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %977 = "tosa.rsqrt"(%976) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %978 = "tosa.sub"(%962, %967) : (tensor<1x32x8x65536xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x8x65536xf32>
    %979 = "tosa.mul"(%978, %977) <{shift = 0 : i8}> : (tensor<1x32x8x65536xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x8x65536xf32>
    %980 = "tosa.reshape"(%979) <{new_shape = array<i64: 1, 256, 256, 256>}> : (tensor<1x32x8x65536xf32>) -> tensor<1x256x256x256xf32>
    %981 = "tosa.reshape"(%arg101) <{new_shape = array<i64: 1, 256>}> : (tensor<256xf32>) -> tensor<1x256xf32>
    %982 = "tosa.reshape"(%981) <{new_shape = array<i64: 1, 256, 1>}> : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %983 = "tosa.reshape"(%982) <{new_shape = array<i64: 1, 256, 1, 1>}> : (tensor<1x256x1xf32>) -> tensor<1x256x1x1xf32>
    %984 = "tosa.reshape"(%arg102) <{new_shape = array<i64: 1, 256>}> : (tensor<256xf32>) -> tensor<1x256xf32>
    %985 = "tosa.reshape"(%984) <{new_shape = array<i64: 1, 256, 1>}> : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %986 = "tosa.reshape"(%985) <{new_shape = array<i64: 1, 256, 1, 1>}> : (tensor<1x256x1xf32>) -> tensor<1x256x1x1xf32>
    %987 = "tosa.mul"(%980, %986) <{shift = 0 : i8}> : (tensor<1x256x256x256xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x256x256xf32>
    %988 = "tosa.add"(%987, %983) : (tensor<1x256x256x256xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x256x256xf32>
    %989 = "tosa.sigmoid"(%988) : (tensor<1x256x256x256xf32>) -> tensor<1x256x256x256xf32>
    %990 = "tosa.mul"(%988, %989) <{shift = 0 : i8}> : (tensor<1x256x256x256xf32>, tensor<1x256x256x256xf32>) -> tensor<1x256x256x256xf32>
    %991 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %992 = "tosa.transpose"(%990, %991) : (tensor<1x256x256x256xf32>, tensor<4xi32>) -> tensor<1x256x256x256xf32>
    %993 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %994 = "tosa.transpose"(%arg103, %993) : (tensor<256x256x3x3xf32>, tensor<4xi32>) -> tensor<256x3x3x256xf32>
    %995 = "tosa.conv2d"(%992, %994, %arg104) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x256x256x256xf32>, tensor<256x3x3x256xf32>, tensor<256xf32>) -> tensor<1x256x256x256xf32>
    %996 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %997 = "tosa.transpose"(%995, %996) : (tensor<1x256x256x256xf32>, tensor<4xi32>) -> tensor<1x256x256x256xf32>
    %998 = "tosa.reshape"(%997) <{new_shape = array<i64: 1, 32, 8, 65536>}> : (tensor<1x256x256x256xf32>) -> tensor<1x32x8x65536xf32>
    %999 = "tosa.reduce_sum"(%998) <{axis = 2 : i32}> : (tensor<1x32x8x65536xf32>) -> tensor<1x32x1x65536xf32>
    %1000 = "tosa.reduce_sum"(%999) <{axis = 3 : i32}> : (tensor<1x32x1x65536xf32>) -> tensor<1x32x1x1xf32>
    %1001 = "tosa.const"() <{value = dense<5.242880e+05> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1002 = "tosa.reciprocal"(%1001) : (tensor<1xf32>) -> tensor<1xf32>
    %1003 = "tosa.mul"(%1002, %1000) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1004 = "tosa.sub"(%998, %1003) : (tensor<1x32x8x65536xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x8x65536xf32>
    %1005 = "tosa.mul"(%1004, %1004) <{shift = 0 : i8}> : (tensor<1x32x8x65536xf32>, tensor<1x32x8x65536xf32>) -> tensor<1x32x8x65536xf32>
    %1006 = "tosa.reduce_sum"(%1005) <{axis = 2 : i32}> : (tensor<1x32x8x65536xf32>) -> tensor<1x32x1x65536xf32>
    %1007 = "tosa.reduce_sum"(%1006) <{axis = 3 : i32}> : (tensor<1x32x1x65536xf32>) -> tensor<1x32x1x1xf32>
    %1008 = "tosa.const"() <{value = dense<5.242880e+05> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1009 = "tosa.reciprocal"(%1008) : (tensor<1xf32>) -> tensor<1xf32>
    %1010 = "tosa.mul"(%1009, %1007) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1011 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %1012 = "tosa.add"(%1010, %1011) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1013 = "tosa.rsqrt"(%1012) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1014 = "tosa.sub"(%998, %1003) : (tensor<1x32x8x65536xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x8x65536xf32>
    %1015 = "tosa.mul"(%1014, %1013) <{shift = 0 : i8}> : (tensor<1x32x8x65536xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x8x65536xf32>
    %1016 = "tosa.reshape"(%1015) <{new_shape = array<i64: 1, 256, 256, 256>}> : (tensor<1x32x8x65536xf32>) -> tensor<1x256x256x256xf32>
    %1017 = "tosa.reshape"(%arg105) <{new_shape = array<i64: 1, 256>}> : (tensor<256xf32>) -> tensor<1x256xf32>
    %1018 = "tosa.reshape"(%1017) <{new_shape = array<i64: 1, 256, 1>}> : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %1019 = "tosa.reshape"(%1018) <{new_shape = array<i64: 1, 256, 1, 1>}> : (tensor<1x256x1xf32>) -> tensor<1x256x1x1xf32>
    %1020 = "tosa.reshape"(%arg106) <{new_shape = array<i64: 1, 256>}> : (tensor<256xf32>) -> tensor<1x256xf32>
    %1021 = "tosa.reshape"(%1020) <{new_shape = array<i64: 1, 256, 1>}> : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %1022 = "tosa.reshape"(%1021) <{new_shape = array<i64: 1, 256, 1, 1>}> : (tensor<1x256x1xf32>) -> tensor<1x256x1x1xf32>
    %1023 = "tosa.mul"(%1016, %1022) <{shift = 0 : i8}> : (tensor<1x256x256x256xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x256x256xf32>
    %1024 = "tosa.add"(%1023, %1019) : (tensor<1x256x256x256xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x256x256xf32>
    %1025 = "tosa.sigmoid"(%1024) : (tensor<1x256x256x256xf32>) -> tensor<1x256x256x256xf32>
    %1026 = "tosa.mul"(%1024, %1025) <{shift = 0 : i8}> : (tensor<1x256x256x256xf32>, tensor<1x256x256x256xf32>) -> tensor<1x256x256x256xf32>
    %1027 = "tosa.identity"(%1026) : (tensor<1x256x256x256xf32>) -> tensor<1x256x256x256xf32>
    %1028 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1029 = "tosa.transpose"(%1027, %1028) : (tensor<1x256x256x256xf32>, tensor<4xi32>) -> tensor<1x256x256x256xf32>
    %1030 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1031 = "tosa.transpose"(%arg107, %1030) : (tensor<256x256x3x3xf32>, tensor<4xi32>) -> tensor<256x3x3x256xf32>
    %1032 = "tosa.conv2d"(%1029, %1031, %arg108) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x256x256x256xf32>, tensor<256x3x3x256xf32>, tensor<256xf32>) -> tensor<1x256x256x256xf32>
    %1033 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1034 = "tosa.transpose"(%1032, %1033) : (tensor<1x256x256x256xf32>, tensor<4xi32>) -> tensor<1x256x256x256xf32>
    %1035 = "tosa.add"(%960, %1034) : (tensor<1x256x256x256xf32>, tensor<1x256x256x256xf32>) -> tensor<1x256x256x256xf32>
    %1036 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x256x256x256xf32>}> : () -> tensor<1x256x256x256xf32>
    %1037 = "tosa.reciprocal"(%1036) : (tensor<1x256x256x256xf32>) -> tensor<1x256x256x256xf32>
    %1038 = "tosa.mul"(%1035, %1037) <{shift = 0 : i8}> : (tensor<1x256x256x256xf32>, tensor<1x256x256x256xf32>) -> tensor<1x256x256x256xf32>
    %1039 = "tosa.const"() <{value = dense<"0x00000000000000000100000000000000020000000000000003000000000000000400000000000000050000000000000006000000000000000700000000000000080000000000000009000000000000000A000000000000000B000000000000000C000000000000000D000000000000000E000000000000000F0000000000000010000000000000001100000000000000120000000000000013000000000000001400000000000000150000000000000016000000000000001700000000000000180000000000000019000000000000001A000000000000001B000000000000001C000000000000001D000000000000001E000000000000001F0000000000000020000000000000002100000000000000220000000000000023000000000000002400000000000000250000000000000026000000000000002700000000000000280000000000000029000000000000002A000000000000002B000000000000002C000000000000002D000000000000002E000000000000002F0000000000000030000000000000003100000000000000320000000000000033000000000000003400000000000000350000000000000036000000000000003700000000000000380000000000000039000000000000003A000000000000003B000000000000003C000000000000003D000000000000003E000000000000003F0000000000000040000000000000004100000000000000420000000000000043000000000000004400000000000000450000000000000046000000000000004700000000000000480000000000000049000000000000004A000000000000004B000000000000004C000000000000004D000000000000004E000000000000004F0000000000000050000000000000005100000000000000520000000000000053000000000000005400000000000000550000000000000056000000000000005700000000000000580000000000000059000000000000005A000000000000005B000000000000005C000000000000005D000000000000005E000000000000005F0000000000000060000000000000006100000000000000620000000000000063000000000000006400000000000000650000000000000066000000000000006700000000000000680000000000000069000000000000006A000000000000006B000000000000006C000000000000006D000000000000006E000000000000006F0000000000000070000000000000007100000000000000720000000000000073000000000000007400000000000000750000000000000076000000000000007700000000000000780000000000000079000000000000007A000000000000007B000000000000007C000000000000007D000000000000007E000000000000007F0000000000000080000000000000008100000000000000820000000000000083000000000000008400000000000000850000000000000086000000000000008700000000000000880000000000000089000000000000008A000000000000008B000000000000008C000000000000008D000000000000008E000000000000008F0000000000000090000000000000009100000000000000920000000000000093000000000000009400000000000000950000000000000096000000000000009700000000000000980000000000000099000000000000009A000000000000009B000000000000009C000000000000009D000000000000009E000000000000009F00000000000000A000000000000000A100000000000000A200000000000000A300000000000000A400000000000000A500000000000000A600000000000000A700000000000000A800000000000000A900000000000000AA00000000000000AB00000000000000AC00000000000000AD00000000000000AE00000000000000AF00000000000000B000000000000000B100000000000000B200000000000000B300000000000000B400000000000000B500000000000000B600000000000000B700000000000000B800000000000000B900000000000000BA00000000000000BB00000000000000BC00000000000000BD00000000000000BE00000000000000BF00000000000000C000000000000000C100000000000000C200000000000000C300000000000000C400000000000000C500000000000000C600000000000000C700000000000000C800000000000000C900000000000000CA00000000000000CB00000000000000CC00000000000000CD00000000000000CE00000000000000CF00000000000000D000000000000000D100000000000000D200000000000000D300000000000000D400000000000000D500000000000000D600000000000000D700000000000000D800000000000000D900000000000000DA00000000000000DB00000000000000DC00000000000000DD00000000000000DE00000000000000DF00000000000000E000000000000000E100000000000000E200000000000000E300000000000000E400000000000000E500000000000000E600000000000000E700000000000000E800000000000000E900000000000000EA00000000000000EB00000000000000EC00000000000000ED00000000000000EE00000000000000EF00000000000000F000000000000000F100000000000000F200000000000000F300000000000000F400000000000000F500000000000000F600000000000000F700000000000000F800000000000000F900000000000000FA00000000000000FB00000000000000FC00000000000000FD00000000000000FE00000000000000FF0000000000000000010000000000000101000000000000020100000000000003010000000000000401000000000000050100000000000006010000000000000701000000000000080100000000000009010000000000000A010000000000000B010000000000000C010000000000000D010000000000000E010000000000000F0100000000000010010000000000001101000000000000120100000000000013010000000000001401000000000000150100000000000016010000000000001701000000000000180100000000000019010000000000001A010000000000001B010000000000001C010000000000001D010000000000001E010000000000001F0100000000000020010000000000002101000000000000220100000000000023010000000000002401000000000000250100000000000026010000000000002701000000000000280100000000000029010000000000002A010000000000002B010000000000002C010000000000002D010000000000002E010000000000002F0100000000000030010000000000003101000000000000320100000000000033010000000000003401000000000000350100000000000036010000000000003701000000000000380100000000000039010000000000003A010000000000003B010000000000003C010000000000003D010000000000003E010000000000003F0100000000000040010000000000004101000000000000420100000000000043010000000000004401000000000000450100000000000046010000000000004701000000000000480100000000000049010000000000004A010000000000004B010000000000004C010000000000004D010000000000004E010000000000004F0100000000000050010000000000005101000000000000520100000000000053010000000000005401000000000000550100000000000056010000000000005701000000000000580100000000000059010000000000005A010000000000005B010000000000005C010000000000005D010000000000005E010000000000005F0100000000000060010000000000006101000000000000620100000000000063010000000000006401000000000000650100000000000066010000000000006701000000000000680100000000000069010000000000006A010000000000006B010000000000006C010000000000006D010000000000006E010000000000006F0100000000000070010000000000007101000000000000720100000000000073010000000000007401000000000000750100000000000076010000000000007701000000000000780100000000000079010000000000007A010000000000007B010000000000007C010000000000007D010000000000007E010000000000007F0100000000000080010000000000008101000000000000820100000000000083010000000000008401000000000000850100000000000086010000000000008701000000000000880100000000000089010000000000008A010000000000008B010000000000008C010000000000008D010000000000008E010000000000008F0100000000000090010000000000009101000000000000920100000000000093010000000000009401000000000000950100000000000096010000000000009701000000000000980100000000000099010000000000009A010000000000009B010000000000009C010000000000009D010000000000009E010000000000009F01000000000000A001000000000000A101000000000000A201000000000000A301000000000000A401000000000000A501000000000000A601000000000000A701000000000000A801000000000000A901000000000000AA01000000000000AB01000000000000AC01000000000000AD01000000000000AE01000000000000AF01000000000000B001000000000000B101000000000000B201000000000000B301000000000000B401000000000000B501000000000000B601000000000000B701000000000000B801000000000000B901000000000000BA01000000000000BB01000000000000BC01000000000000BD01000000000000BE01000000000000BF01000000000000C001000000000000C101000000000000C201000000000000C301000000000000C401000000000000C501000000000000C601000000000000C701000000000000C801000000000000C901000000000000CA01000000000000CB01000000000000CC01000000000000CD01000000000000CE01000000000000CF01000000000000D001000000000000D101000000000000D201000000000000D301000000000000D401000000000000D501000000000000D601000000000000D701000000000000D801000000000000D901000000000000DA01000000000000DB01000000000000DC01000000000000DD01000000000000DE01000000000000DF01000000000000E001000000000000E101000000000000E201000000000000E301000000000000E401000000000000E501000000000000E601000000000000E701000000000000E801000000000000E901000000000000EA01000000000000EB01000000000000EC01000000000000ED01000000000000EE01000000000000EF01000000000000F001000000000000F101000000000000F201000000000000F301000000000000F401000000000000F501000000000000F601000000000000F701000000000000F801000000000000F901000000000000FA01000000000000FB01000000000000FC01000000000000FD01000000000000FE01000000000000FF01000000000000"> : tensor<512xi64>}> : () -> tensor<512xi64>
    %1040 = "tosa.cast"(%1039) : (tensor<512xi64>) -> tensor<512xf32>
    %1041 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
    %1042 = "tosa.mul"(%1040, %1041) <{shift = 0 : i8}> : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %1043 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
    %1044 = "tosa.add"(%1042, %1043) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %1045 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<512xf32>}> : () -> tensor<512xf32>
    %1046 = "tosa.mul"(%1044, %1045) <{shift = 0 : i8}> : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %1047 = "tosa.cast"(%1046) : (tensor<512xf32>) -> tensor<512xi64>
    %1048 = "tosa.reshape"(%1047) <{new_shape = array<i64: 512, 1>}> : (tensor<512xi64>) -> tensor<512x1xi64>
    %1049 = "tosa.const"() <{value = dense<"0x00000000000000000100000000000000020000000000000003000000000000000400000000000000050000000000000006000000000000000700000000000000080000000000000009000000000000000A000000000000000B000000000000000C000000000000000D000000000000000E000000000000000F0000000000000010000000000000001100000000000000120000000000000013000000000000001400000000000000150000000000000016000000000000001700000000000000180000000000000019000000000000001A000000000000001B000000000000001C000000000000001D000000000000001E000000000000001F0000000000000020000000000000002100000000000000220000000000000023000000000000002400000000000000250000000000000026000000000000002700000000000000280000000000000029000000000000002A000000000000002B000000000000002C000000000000002D000000000000002E000000000000002F0000000000000030000000000000003100000000000000320000000000000033000000000000003400000000000000350000000000000036000000000000003700000000000000380000000000000039000000000000003A000000000000003B000000000000003C000000000000003D000000000000003E000000000000003F0000000000000040000000000000004100000000000000420000000000000043000000000000004400000000000000450000000000000046000000000000004700000000000000480000000000000049000000000000004A000000000000004B000000000000004C000000000000004D000000000000004E000000000000004F0000000000000050000000000000005100000000000000520000000000000053000000000000005400000000000000550000000000000056000000000000005700000000000000580000000000000059000000000000005A000000000000005B000000000000005C000000000000005D000000000000005E000000000000005F0000000000000060000000000000006100000000000000620000000000000063000000000000006400000000000000650000000000000066000000000000006700000000000000680000000000000069000000000000006A000000000000006B000000000000006C000000000000006D000000000000006E000000000000006F0000000000000070000000000000007100000000000000720000000000000073000000000000007400000000000000750000000000000076000000000000007700000000000000780000000000000079000000000000007A000000000000007B000000000000007C000000000000007D000000000000007E000000000000007F0000000000000080000000000000008100000000000000820000000000000083000000000000008400000000000000850000000000000086000000000000008700000000000000880000000000000089000000000000008A000000000000008B000000000000008C000000000000008D000000000000008E000000000000008F0000000000000090000000000000009100000000000000920000000000000093000000000000009400000000000000950000000000000096000000000000009700000000000000980000000000000099000000000000009A000000000000009B000000000000009C000000000000009D000000000000009E000000000000009F00000000000000A000000000000000A100000000000000A200000000000000A300000000000000A400000000000000A500000000000000A600000000000000A700000000000000A800000000000000A900000000000000AA00000000000000AB00000000000000AC00000000000000AD00000000000000AE00000000000000AF00000000000000B000000000000000B100000000000000B200000000000000B300000000000000B400000000000000B500000000000000B600000000000000B700000000000000B800000000000000B900000000000000BA00000000000000BB00000000000000BC00000000000000BD00000000000000BE00000000000000BF00000000000000C000000000000000C100000000000000C200000000000000C300000000000000C400000000000000C500000000000000C600000000000000C700000000000000C800000000000000C900000000000000CA00000000000000CB00000000000000CC00000000000000CD00000000000000CE00000000000000CF00000000000000D000000000000000D100000000000000D200000000000000D300000000000000D400000000000000D500000000000000D600000000000000D700000000000000D800000000000000D900000000000000DA00000000000000DB00000000000000DC00000000000000DD00000000000000DE00000000000000DF00000000000000E000000000000000E100000000000000E200000000000000E300000000000000E400000000000000E500000000000000E600000000000000E700000000000000E800000000000000E900000000000000EA00000000000000EB00000000000000EC00000000000000ED00000000000000EE00000000000000EF00000000000000F000000000000000F100000000000000F200000000000000F300000000000000F400000000000000F500000000000000F600000000000000F700000000000000F800000000000000F900000000000000FA00000000000000FB00000000000000FC00000000000000FD00000000000000FE00000000000000FF0000000000000000010000000000000101000000000000020100000000000003010000000000000401000000000000050100000000000006010000000000000701000000000000080100000000000009010000000000000A010000000000000B010000000000000C010000000000000D010000000000000E010000000000000F0100000000000010010000000000001101000000000000120100000000000013010000000000001401000000000000150100000000000016010000000000001701000000000000180100000000000019010000000000001A010000000000001B010000000000001C010000000000001D010000000000001E010000000000001F0100000000000020010000000000002101000000000000220100000000000023010000000000002401000000000000250100000000000026010000000000002701000000000000280100000000000029010000000000002A010000000000002B010000000000002C010000000000002D010000000000002E010000000000002F0100000000000030010000000000003101000000000000320100000000000033010000000000003401000000000000350100000000000036010000000000003701000000000000380100000000000039010000000000003A010000000000003B010000000000003C010000000000003D010000000000003E010000000000003F0100000000000040010000000000004101000000000000420100000000000043010000000000004401000000000000450100000000000046010000000000004701000000000000480100000000000049010000000000004A010000000000004B010000000000004C010000000000004D010000000000004E010000000000004F0100000000000050010000000000005101000000000000520100000000000053010000000000005401000000000000550100000000000056010000000000005701000000000000580100000000000059010000000000005A010000000000005B010000000000005C010000000000005D010000000000005E010000000000005F0100000000000060010000000000006101000000000000620100000000000063010000000000006401000000000000650100000000000066010000000000006701000000000000680100000000000069010000000000006A010000000000006B010000000000006C010000000000006D010000000000006E010000000000006F0100000000000070010000000000007101000000000000720100000000000073010000000000007401000000000000750100000000000076010000000000007701000000000000780100000000000079010000000000007A010000000000007B010000000000007C010000000000007D010000000000007E010000000000007F0100000000000080010000000000008101000000000000820100000000000083010000000000008401000000000000850100000000000086010000000000008701000000000000880100000000000089010000000000008A010000000000008B010000000000008C010000000000008D010000000000008E010000000000008F0100000000000090010000000000009101000000000000920100000000000093010000000000009401000000000000950100000000000096010000000000009701000000000000980100000000000099010000000000009A010000000000009B010000000000009C010000000000009D010000000000009E010000000000009F01000000000000A001000000000000A101000000000000A201000000000000A301000000000000A401000000000000A501000000000000A601000000000000A701000000000000A801000000000000A901000000000000AA01000000000000AB01000000000000AC01000000000000AD01000000000000AE01000000000000AF01000000000000B001000000000000B101000000000000B201000000000000B301000000000000B401000000000000B501000000000000B601000000000000B701000000000000B801000000000000B901000000000000BA01000000000000BB01000000000000BC01000000000000BD01000000000000BE01000000000000BF01000000000000C001000000000000C101000000000000C201000000000000C301000000000000C401000000000000C501000000000000C601000000000000C701000000000000C801000000000000C901000000000000CA01000000000000CB01000000000000CC01000000000000CD01000000000000CE01000000000000CF01000000000000D001000000000000D101000000000000D201000000000000D301000000000000D401000000000000D501000000000000D601000000000000D701000000000000D801000000000000D901000000000000DA01000000000000DB01000000000000DC01000000000000DD01000000000000DE01000000000000DF01000000000000E001000000000000E101000000000000E201000000000000E301000000000000E401000000000000E501000000000000E601000000000000E701000000000000E801000000000000E901000000000000EA01000000000000EB01000000000000EC01000000000000ED01000000000000EE01000000000000EF01000000000000F001000000000000F101000000000000F201000000000000F301000000000000F401000000000000F501000000000000F601000000000000F701000000000000F801000000000000F901000000000000FA01000000000000FB01000000000000FC01000000000000FD01000000000000FE01000000000000FF01000000000000"> : tensor<512xi64>}> : () -> tensor<512xi64>
    %1050 = "tosa.cast"(%1049) : (tensor<512xi64>) -> tensor<512xf32>
    %1051 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
    %1052 = "tosa.mul"(%1050, %1051) <{shift = 0 : i8}> : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %1053 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
    %1054 = "tosa.add"(%1052, %1053) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %1055 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<512xf32>}> : () -> tensor<512xf32>
    %1056 = "tosa.mul"(%1054, %1055) <{shift = 0 : i8}> : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %1057 = "tosa.cast"(%1056) : (tensor<512xf32>) -> tensor<512xi64>
    %1058 = "tensor.empty"() : () -> tensor<1x256x512x512xf32>
    %1059 = "linalg.generic"(%1048, %1057, %1058) <{indexing_maps = [#map3, #map3, #map4], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg141: i64, %arg142: i64, %arg143: f32):
      %1346 = "arith.index_cast"(%arg141) : (i64) -> index
      %1347 = "arith.index_cast"(%arg142) : (i64) -> index
      %1348 = "linalg.index"() <{dim = 2 : i64}> : () -> index
      %1349 = "tensor.extract"(%1038, %1346, %1347, %1348) : (tensor<1x256x256x256xf32>, index, index, index) -> f32
      "linalg.yield"(%1349) : (f32) -> ()
    }) : (tensor<512x1xi64>, tensor<512xi64>, tensor<1x256x512x512xf32>) -> tensor<1x256x512x512xf32>
    %1060 = "tosa.identity"(%1059) : (tensor<1x256x512x512xf32>) -> tensor<1x256x512x512xf32>
    %1061 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1062 = "tosa.transpose"(%1060, %1061) : (tensor<1x256x512x512xf32>, tensor<4xi32>) -> tensor<1x512x512x256xf32>
    %1063 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1064 = "tosa.transpose"(%arg109, %1063) : (tensor<256x256x3x3xf32>, tensor<4xi32>) -> tensor<256x3x3x256xf32>
    %1065 = "tosa.conv2d"(%1062, %1064, %arg110) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x512x512x256xf32>, tensor<256x3x3x256xf32>, tensor<256xf32>) -> tensor<1x512x512x256xf32>
    %1066 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1067 = "tosa.transpose"(%1065, %1066) : (tensor<1x512x512x256xf32>, tensor<4xi32>) -> tensor<1x256x512x512xf32>
    %1068 = "tosa.identity"(%1067) : (tensor<1x256x512x512xf32>) -> tensor<1x256x512x512xf32>
    %1069 = "tosa.reshape"(%1068) <{new_shape = array<i64: 1, 32, 8, 262144>}> : (tensor<1x256x512x512xf32>) -> tensor<1x32x8x262144xf32>
    %1070 = "tosa.reduce_sum"(%1069) <{axis = 2 : i32}> : (tensor<1x32x8x262144xf32>) -> tensor<1x32x1x262144xf32>
    %1071 = "tosa.reduce_sum"(%1070) <{axis = 3 : i32}> : (tensor<1x32x1x262144xf32>) -> tensor<1x32x1x1xf32>
    %1072 = "tosa.const"() <{value = dense<0x4A000000> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1073 = "tosa.reciprocal"(%1072) : (tensor<1xf32>) -> tensor<1xf32>
    %1074 = "tosa.mul"(%1073, %1071) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1075 = "tosa.sub"(%1069, %1074) : (tensor<1x32x8x262144xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x8x262144xf32>
    %1076 = "tosa.mul"(%1075, %1075) <{shift = 0 : i8}> : (tensor<1x32x8x262144xf32>, tensor<1x32x8x262144xf32>) -> tensor<1x32x8x262144xf32>
    %1077 = "tosa.reduce_sum"(%1076) <{axis = 2 : i32}> : (tensor<1x32x8x262144xf32>) -> tensor<1x32x1x262144xf32>
    %1078 = "tosa.reduce_sum"(%1077) <{axis = 3 : i32}> : (tensor<1x32x1x262144xf32>) -> tensor<1x32x1x1xf32>
    %1079 = "tosa.const"() <{value = dense<0x4A000000> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1080 = "tosa.reciprocal"(%1079) : (tensor<1xf32>) -> tensor<1xf32>
    %1081 = "tosa.mul"(%1080, %1078) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1082 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %1083 = "tosa.add"(%1081, %1082) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1084 = "tosa.rsqrt"(%1083) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1085 = "tosa.sub"(%1069, %1074) : (tensor<1x32x8x262144xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x8x262144xf32>
    %1086 = "tosa.mul"(%1085, %1084) <{shift = 0 : i8}> : (tensor<1x32x8x262144xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x8x262144xf32>
    %1087 = "tosa.reshape"(%1086) <{new_shape = array<i64: 1, 256, 512, 512>}> : (tensor<1x32x8x262144xf32>) -> tensor<1x256x512x512xf32>
    %1088 = "tosa.reshape"(%arg111) <{new_shape = array<i64: 1, 256>}> : (tensor<256xf32>) -> tensor<1x256xf32>
    %1089 = "tosa.reshape"(%1088) <{new_shape = array<i64: 1, 256, 1>}> : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %1090 = "tosa.reshape"(%1089) <{new_shape = array<i64: 1, 256, 1, 1>}> : (tensor<1x256x1xf32>) -> tensor<1x256x1x1xf32>
    %1091 = "tosa.reshape"(%arg112) <{new_shape = array<i64: 1, 256>}> : (tensor<256xf32>) -> tensor<1x256xf32>
    %1092 = "tosa.reshape"(%1091) <{new_shape = array<i64: 1, 256, 1>}> : (tensor<1x256xf32>) -> tensor<1x256x1xf32>
    %1093 = "tosa.reshape"(%1092) <{new_shape = array<i64: 1, 256, 1, 1>}> : (tensor<1x256x1xf32>) -> tensor<1x256x1x1xf32>
    %1094 = "tosa.mul"(%1087, %1093) <{shift = 0 : i8}> : (tensor<1x256x512x512xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x512x512xf32>
    %1095 = "tosa.add"(%1094, %1090) : (tensor<1x256x512x512xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x512x512xf32>
    %1096 = "tosa.sigmoid"(%1095) : (tensor<1x256x512x512xf32>) -> tensor<1x256x512x512xf32>
    %1097 = "tosa.mul"(%1095, %1096) <{shift = 0 : i8}> : (tensor<1x256x512x512xf32>, tensor<1x256x512x512xf32>) -> tensor<1x256x512x512xf32>
    %1098 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1099 = "tosa.transpose"(%1097, %1098) : (tensor<1x256x512x512xf32>, tensor<4xi32>) -> tensor<1x512x512x256xf32>
    %1100 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1101 = "tosa.transpose"(%arg113, %1100) : (tensor<128x256x3x3xf32>, tensor<4xi32>) -> tensor<128x3x3x256xf32>
    %1102 = "tosa.conv2d"(%1099, %1101, %arg114) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x512x512x256xf32>, tensor<128x3x3x256xf32>, tensor<128xf32>) -> tensor<1x512x512x128xf32>
    %1103 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1104 = "tosa.transpose"(%1102, %1103) : (tensor<1x512x512x128xf32>, tensor<4xi32>) -> tensor<1x128x512x512xf32>
    %1105 = "tosa.reshape"(%1104) <{new_shape = array<i64: 1, 32, 4, 262144>}> : (tensor<1x128x512x512xf32>) -> tensor<1x32x4x262144xf32>
    %1106 = "tosa.reduce_sum"(%1105) <{axis = 2 : i32}> : (tensor<1x32x4x262144xf32>) -> tensor<1x32x1x262144xf32>
    %1107 = "tosa.reduce_sum"(%1106) <{axis = 3 : i32}> : (tensor<1x32x1x262144xf32>) -> tensor<1x32x1x1xf32>
    %1108 = "tosa.const"() <{value = dense<0x49800000> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1109 = "tosa.reciprocal"(%1108) : (tensor<1xf32>) -> tensor<1xf32>
    %1110 = "tosa.mul"(%1109, %1107) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1111 = "tosa.sub"(%1105, %1110) : (tensor<1x32x4x262144xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x4x262144xf32>
    %1112 = "tosa.mul"(%1111, %1111) <{shift = 0 : i8}> : (tensor<1x32x4x262144xf32>, tensor<1x32x4x262144xf32>) -> tensor<1x32x4x262144xf32>
    %1113 = "tosa.reduce_sum"(%1112) <{axis = 2 : i32}> : (tensor<1x32x4x262144xf32>) -> tensor<1x32x1x262144xf32>
    %1114 = "tosa.reduce_sum"(%1113) <{axis = 3 : i32}> : (tensor<1x32x1x262144xf32>) -> tensor<1x32x1x1xf32>
    %1115 = "tosa.const"() <{value = dense<0x49800000> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1116 = "tosa.reciprocal"(%1115) : (tensor<1xf32>) -> tensor<1xf32>
    %1117 = "tosa.mul"(%1116, %1114) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1118 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %1119 = "tosa.add"(%1117, %1118) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1120 = "tosa.rsqrt"(%1119) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1121 = "tosa.sub"(%1105, %1110) : (tensor<1x32x4x262144xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x4x262144xf32>
    %1122 = "tosa.mul"(%1121, %1120) <{shift = 0 : i8}> : (tensor<1x32x4x262144xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x4x262144xf32>
    %1123 = "tosa.reshape"(%1122) <{new_shape = array<i64: 1, 128, 512, 512>}> : (tensor<1x32x4x262144xf32>) -> tensor<1x128x512x512xf32>
    %1124 = "tosa.reshape"(%arg115) <{new_shape = array<i64: 1, 128>}> : (tensor<128xf32>) -> tensor<1x128xf32>
    %1125 = "tosa.reshape"(%1124) <{new_shape = array<i64: 1, 128, 1>}> : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %1126 = "tosa.reshape"(%1125) <{new_shape = array<i64: 1, 128, 1, 1>}> : (tensor<1x128x1xf32>) -> tensor<1x128x1x1xf32>
    %1127 = "tosa.reshape"(%arg116) <{new_shape = array<i64: 1, 128>}> : (tensor<128xf32>) -> tensor<1x128xf32>
    %1128 = "tosa.reshape"(%1127) <{new_shape = array<i64: 1, 128, 1>}> : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %1129 = "tosa.reshape"(%1128) <{new_shape = array<i64: 1, 128, 1, 1>}> : (tensor<1x128x1xf32>) -> tensor<1x128x1x1xf32>
    %1130 = "tosa.mul"(%1123, %1129) <{shift = 0 : i8}> : (tensor<1x128x512x512xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x512x512xf32>
    %1131 = "tosa.add"(%1130, %1126) : (tensor<1x128x512x512xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x512x512xf32>
    %1132 = "tosa.sigmoid"(%1131) : (tensor<1x128x512x512xf32>) -> tensor<1x128x512x512xf32>
    %1133 = "tosa.mul"(%1131, %1132) <{shift = 0 : i8}> : (tensor<1x128x512x512xf32>, tensor<1x128x512x512xf32>) -> tensor<1x128x512x512xf32>
    %1134 = "tosa.identity"(%1133) : (tensor<1x128x512x512xf32>) -> tensor<1x128x512x512xf32>
    %1135 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1136 = "tosa.transpose"(%1134, %1135) : (tensor<1x128x512x512xf32>, tensor<4xi32>) -> tensor<1x512x512x128xf32>
    %1137 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1138 = "tosa.transpose"(%arg117, %1137) : (tensor<128x128x3x3xf32>, tensor<4xi32>) -> tensor<128x3x3x128xf32>
    %1139 = "tosa.conv2d"(%1136, %1138, %arg118) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x512x512x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<1x512x512x128xf32>
    %1140 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1141 = "tosa.transpose"(%1139, %1140) : (tensor<1x512x512x128xf32>, tensor<4xi32>) -> tensor<1x128x512x512xf32>
    %1142 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1143 = "tosa.transpose"(%1067, %1142) : (tensor<1x256x512x512xf32>, tensor<4xi32>) -> tensor<1x512x512x256xf32>
    %1144 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1145 = "tosa.transpose"(%arg119, %1144) : (tensor<128x256x1x1xf32>, tensor<4xi32>) -> tensor<128x1x1x256xf32>
    %1146 = "tosa.conv2d"(%1143, %1145, %arg120) <{dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}> : (tensor<1x512x512x256xf32>, tensor<128x1x1x256xf32>, tensor<128xf32>) -> tensor<1x512x512x128xf32>
    %1147 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1148 = "tosa.transpose"(%1146, %1147) : (tensor<1x512x512x128xf32>, tensor<4xi32>) -> tensor<1x128x512x512xf32>
    %1149 = "tosa.add"(%1148, %1141) : (tensor<1x128x512x512xf32>, tensor<1x128x512x512xf32>) -> tensor<1x128x512x512xf32>
    %1150 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x128x512x512xf32>}> : () -> tensor<1x128x512x512xf32>
    %1151 = "tosa.reciprocal"(%1150) : (tensor<1x128x512x512xf32>) -> tensor<1x128x512x512xf32>
    %1152 = "tosa.mul"(%1149, %1151) <{shift = 0 : i8}> : (tensor<1x128x512x512xf32>, tensor<1x128x512x512xf32>) -> tensor<1x128x512x512xf32>
    %1153 = "tosa.identity"(%1152) : (tensor<1x128x512x512xf32>) -> tensor<1x128x512x512xf32>
    %1154 = "tosa.reshape"(%1153) <{new_shape = array<i64: 1, 32, 4, 262144>}> : (tensor<1x128x512x512xf32>) -> tensor<1x32x4x262144xf32>
    %1155 = "tosa.reduce_sum"(%1154) <{axis = 2 : i32}> : (tensor<1x32x4x262144xf32>) -> tensor<1x32x1x262144xf32>
    %1156 = "tosa.reduce_sum"(%1155) <{axis = 3 : i32}> : (tensor<1x32x1x262144xf32>) -> tensor<1x32x1x1xf32>
    %1157 = "tosa.const"() <{value = dense<0x49800000> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1158 = "tosa.reciprocal"(%1157) : (tensor<1xf32>) -> tensor<1xf32>
    %1159 = "tosa.mul"(%1158, %1156) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1160 = "tosa.sub"(%1154, %1159) : (tensor<1x32x4x262144xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x4x262144xf32>
    %1161 = "tosa.mul"(%1160, %1160) <{shift = 0 : i8}> : (tensor<1x32x4x262144xf32>, tensor<1x32x4x262144xf32>) -> tensor<1x32x4x262144xf32>
    %1162 = "tosa.reduce_sum"(%1161) <{axis = 2 : i32}> : (tensor<1x32x4x262144xf32>) -> tensor<1x32x1x262144xf32>
    %1163 = "tosa.reduce_sum"(%1162) <{axis = 3 : i32}> : (tensor<1x32x1x262144xf32>) -> tensor<1x32x1x1xf32>
    %1164 = "tosa.const"() <{value = dense<0x49800000> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1165 = "tosa.reciprocal"(%1164) : (tensor<1xf32>) -> tensor<1xf32>
    %1166 = "tosa.mul"(%1165, %1163) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1167 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %1168 = "tosa.add"(%1166, %1167) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1169 = "tosa.rsqrt"(%1168) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1170 = "tosa.sub"(%1154, %1159) : (tensor<1x32x4x262144xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x4x262144xf32>
    %1171 = "tosa.mul"(%1170, %1169) <{shift = 0 : i8}> : (tensor<1x32x4x262144xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x4x262144xf32>
    %1172 = "tosa.reshape"(%1171) <{new_shape = array<i64: 1, 128, 512, 512>}> : (tensor<1x32x4x262144xf32>) -> tensor<1x128x512x512xf32>
    %1173 = "tosa.reshape"(%arg121) <{new_shape = array<i64: 1, 128>}> : (tensor<128xf32>) -> tensor<1x128xf32>
    %1174 = "tosa.reshape"(%1173) <{new_shape = array<i64: 1, 128, 1>}> : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %1175 = "tosa.reshape"(%1174) <{new_shape = array<i64: 1, 128, 1, 1>}> : (tensor<1x128x1xf32>) -> tensor<1x128x1x1xf32>
    %1176 = "tosa.reshape"(%arg122) <{new_shape = array<i64: 1, 128>}> : (tensor<128xf32>) -> tensor<1x128xf32>
    %1177 = "tosa.reshape"(%1176) <{new_shape = array<i64: 1, 128, 1>}> : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %1178 = "tosa.reshape"(%1177) <{new_shape = array<i64: 1, 128, 1, 1>}> : (tensor<1x128x1xf32>) -> tensor<1x128x1x1xf32>
    %1179 = "tosa.mul"(%1172, %1178) <{shift = 0 : i8}> : (tensor<1x128x512x512xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x512x512xf32>
    %1180 = "tosa.add"(%1179, %1175) : (tensor<1x128x512x512xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x512x512xf32>
    %1181 = "tosa.sigmoid"(%1180) : (tensor<1x128x512x512xf32>) -> tensor<1x128x512x512xf32>
    %1182 = "tosa.mul"(%1180, %1181) <{shift = 0 : i8}> : (tensor<1x128x512x512xf32>, tensor<1x128x512x512xf32>) -> tensor<1x128x512x512xf32>
    %1183 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1184 = "tosa.transpose"(%1182, %1183) : (tensor<1x128x512x512xf32>, tensor<4xi32>) -> tensor<1x512x512x128xf32>
    %1185 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1186 = "tosa.transpose"(%arg123, %1185) : (tensor<128x128x3x3xf32>, tensor<4xi32>) -> tensor<128x3x3x128xf32>
    %1187 = "tosa.conv2d"(%1184, %1186, %arg124) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x512x512x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<1x512x512x128xf32>
    %1188 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1189 = "tosa.transpose"(%1187, %1188) : (tensor<1x512x512x128xf32>, tensor<4xi32>) -> tensor<1x128x512x512xf32>
    %1190 = "tosa.reshape"(%1189) <{new_shape = array<i64: 1, 32, 4, 262144>}> : (tensor<1x128x512x512xf32>) -> tensor<1x32x4x262144xf32>
    %1191 = "tosa.reduce_sum"(%1190) <{axis = 2 : i32}> : (tensor<1x32x4x262144xf32>) -> tensor<1x32x1x262144xf32>
    %1192 = "tosa.reduce_sum"(%1191) <{axis = 3 : i32}> : (tensor<1x32x1x262144xf32>) -> tensor<1x32x1x1xf32>
    %1193 = "tosa.const"() <{value = dense<0x49800000> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1194 = "tosa.reciprocal"(%1193) : (tensor<1xf32>) -> tensor<1xf32>
    %1195 = "tosa.mul"(%1194, %1192) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1196 = "tosa.sub"(%1190, %1195) : (tensor<1x32x4x262144xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x4x262144xf32>
    %1197 = "tosa.mul"(%1196, %1196) <{shift = 0 : i8}> : (tensor<1x32x4x262144xf32>, tensor<1x32x4x262144xf32>) -> tensor<1x32x4x262144xf32>
    %1198 = "tosa.reduce_sum"(%1197) <{axis = 2 : i32}> : (tensor<1x32x4x262144xf32>) -> tensor<1x32x1x262144xf32>
    %1199 = "tosa.reduce_sum"(%1198) <{axis = 3 : i32}> : (tensor<1x32x1x262144xf32>) -> tensor<1x32x1x1xf32>
    %1200 = "tosa.const"() <{value = dense<0x49800000> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1201 = "tosa.reciprocal"(%1200) : (tensor<1xf32>) -> tensor<1xf32>
    %1202 = "tosa.mul"(%1201, %1199) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1203 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %1204 = "tosa.add"(%1202, %1203) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1205 = "tosa.rsqrt"(%1204) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1206 = "tosa.sub"(%1190, %1195) : (tensor<1x32x4x262144xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x4x262144xf32>
    %1207 = "tosa.mul"(%1206, %1205) <{shift = 0 : i8}> : (tensor<1x32x4x262144xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x4x262144xf32>
    %1208 = "tosa.reshape"(%1207) <{new_shape = array<i64: 1, 128, 512, 512>}> : (tensor<1x32x4x262144xf32>) -> tensor<1x128x512x512xf32>
    %1209 = "tosa.reshape"(%arg125) <{new_shape = array<i64: 1, 128>}> : (tensor<128xf32>) -> tensor<1x128xf32>
    %1210 = "tosa.reshape"(%1209) <{new_shape = array<i64: 1, 128, 1>}> : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %1211 = "tosa.reshape"(%1210) <{new_shape = array<i64: 1, 128, 1, 1>}> : (tensor<1x128x1xf32>) -> tensor<1x128x1x1xf32>
    %1212 = "tosa.reshape"(%arg126) <{new_shape = array<i64: 1, 128>}> : (tensor<128xf32>) -> tensor<1x128xf32>
    %1213 = "tosa.reshape"(%1212) <{new_shape = array<i64: 1, 128, 1>}> : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %1214 = "tosa.reshape"(%1213) <{new_shape = array<i64: 1, 128, 1, 1>}> : (tensor<1x128x1xf32>) -> tensor<1x128x1x1xf32>
    %1215 = "tosa.mul"(%1208, %1214) <{shift = 0 : i8}> : (tensor<1x128x512x512xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x512x512xf32>
    %1216 = "tosa.add"(%1215, %1211) : (tensor<1x128x512x512xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x512x512xf32>
    %1217 = "tosa.sigmoid"(%1216) : (tensor<1x128x512x512xf32>) -> tensor<1x128x512x512xf32>
    %1218 = "tosa.mul"(%1216, %1217) <{shift = 0 : i8}> : (tensor<1x128x512x512xf32>, tensor<1x128x512x512xf32>) -> tensor<1x128x512x512xf32>
    %1219 = "tosa.identity"(%1218) : (tensor<1x128x512x512xf32>) -> tensor<1x128x512x512xf32>
    %1220 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1221 = "tosa.transpose"(%1219, %1220) : (tensor<1x128x512x512xf32>, tensor<4xi32>) -> tensor<1x512x512x128xf32>
    %1222 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1223 = "tosa.transpose"(%arg127, %1222) : (tensor<128x128x3x3xf32>, tensor<4xi32>) -> tensor<128x3x3x128xf32>
    %1224 = "tosa.conv2d"(%1221, %1223, %arg128) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x512x512x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<1x512x512x128xf32>
    %1225 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1226 = "tosa.transpose"(%1224, %1225) : (tensor<1x512x512x128xf32>, tensor<4xi32>) -> tensor<1x128x512x512xf32>
    %1227 = "tosa.add"(%1152, %1226) : (tensor<1x128x512x512xf32>, tensor<1x128x512x512xf32>) -> tensor<1x128x512x512xf32>
    %1228 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x128x512x512xf32>}> : () -> tensor<1x128x512x512xf32>
    %1229 = "tosa.reciprocal"(%1228) : (tensor<1x128x512x512xf32>) -> tensor<1x128x512x512xf32>
    %1230 = "tosa.mul"(%1227, %1229) <{shift = 0 : i8}> : (tensor<1x128x512x512xf32>, tensor<1x128x512x512xf32>) -> tensor<1x128x512x512xf32>
    %1231 = "tosa.identity"(%1230) : (tensor<1x128x512x512xf32>) -> tensor<1x128x512x512xf32>
    %1232 = "tosa.reshape"(%1231) <{new_shape = array<i64: 1, 32, 4, 262144>}> : (tensor<1x128x512x512xf32>) -> tensor<1x32x4x262144xf32>
    %1233 = "tosa.reduce_sum"(%1232) <{axis = 2 : i32}> : (tensor<1x32x4x262144xf32>) -> tensor<1x32x1x262144xf32>
    %1234 = "tosa.reduce_sum"(%1233) <{axis = 3 : i32}> : (tensor<1x32x1x262144xf32>) -> tensor<1x32x1x1xf32>
    %1235 = "tosa.const"() <{value = dense<0x49800000> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1236 = "tosa.reciprocal"(%1235) : (tensor<1xf32>) -> tensor<1xf32>
    %1237 = "tosa.mul"(%1236, %1234) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1238 = "tosa.sub"(%1232, %1237) : (tensor<1x32x4x262144xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x4x262144xf32>
    %1239 = "tosa.mul"(%1238, %1238) <{shift = 0 : i8}> : (tensor<1x32x4x262144xf32>, tensor<1x32x4x262144xf32>) -> tensor<1x32x4x262144xf32>
    %1240 = "tosa.reduce_sum"(%1239) <{axis = 2 : i32}> : (tensor<1x32x4x262144xf32>) -> tensor<1x32x1x262144xf32>
    %1241 = "tosa.reduce_sum"(%1240) <{axis = 3 : i32}> : (tensor<1x32x1x262144xf32>) -> tensor<1x32x1x1xf32>
    %1242 = "tosa.const"() <{value = dense<0x49800000> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1243 = "tosa.reciprocal"(%1242) : (tensor<1xf32>) -> tensor<1xf32>
    %1244 = "tosa.mul"(%1243, %1241) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1245 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %1246 = "tosa.add"(%1244, %1245) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1247 = "tosa.rsqrt"(%1246) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1248 = "tosa.sub"(%1232, %1237) : (tensor<1x32x4x262144xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x4x262144xf32>
    %1249 = "tosa.mul"(%1248, %1247) <{shift = 0 : i8}> : (tensor<1x32x4x262144xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x4x262144xf32>
    %1250 = "tosa.reshape"(%1249) <{new_shape = array<i64: 1, 128, 512, 512>}> : (tensor<1x32x4x262144xf32>) -> tensor<1x128x512x512xf32>
    %1251 = "tosa.reshape"(%arg129) <{new_shape = array<i64: 1, 128>}> : (tensor<128xf32>) -> tensor<1x128xf32>
    %1252 = "tosa.reshape"(%1251) <{new_shape = array<i64: 1, 128, 1>}> : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %1253 = "tosa.reshape"(%1252) <{new_shape = array<i64: 1, 128, 1, 1>}> : (tensor<1x128x1xf32>) -> tensor<1x128x1x1xf32>
    %1254 = "tosa.reshape"(%arg130) <{new_shape = array<i64: 1, 128>}> : (tensor<128xf32>) -> tensor<1x128xf32>
    %1255 = "tosa.reshape"(%1254) <{new_shape = array<i64: 1, 128, 1>}> : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %1256 = "tosa.reshape"(%1255) <{new_shape = array<i64: 1, 128, 1, 1>}> : (tensor<1x128x1xf32>) -> tensor<1x128x1x1xf32>
    %1257 = "tosa.mul"(%1250, %1256) <{shift = 0 : i8}> : (tensor<1x128x512x512xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x512x512xf32>
    %1258 = "tosa.add"(%1257, %1253) : (tensor<1x128x512x512xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x512x512xf32>
    %1259 = "tosa.sigmoid"(%1258) : (tensor<1x128x512x512xf32>) -> tensor<1x128x512x512xf32>
    %1260 = "tosa.mul"(%1258, %1259) <{shift = 0 : i8}> : (tensor<1x128x512x512xf32>, tensor<1x128x512x512xf32>) -> tensor<1x128x512x512xf32>
    %1261 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1262 = "tosa.transpose"(%1260, %1261) : (tensor<1x128x512x512xf32>, tensor<4xi32>) -> tensor<1x512x512x128xf32>
    %1263 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1264 = "tosa.transpose"(%arg131, %1263) : (tensor<128x128x3x3xf32>, tensor<4xi32>) -> tensor<128x3x3x128xf32>
    %1265 = "tosa.conv2d"(%1262, %1264, %arg132) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x512x512x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<1x512x512x128xf32>
    %1266 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1267 = "tosa.transpose"(%1265, %1266) : (tensor<1x512x512x128xf32>, tensor<4xi32>) -> tensor<1x128x512x512xf32>
    %1268 = "tosa.reshape"(%1267) <{new_shape = array<i64: 1, 32, 4, 262144>}> : (tensor<1x128x512x512xf32>) -> tensor<1x32x4x262144xf32>
    %1269 = "tosa.reduce_sum"(%1268) <{axis = 2 : i32}> : (tensor<1x32x4x262144xf32>) -> tensor<1x32x1x262144xf32>
    %1270 = "tosa.reduce_sum"(%1269) <{axis = 3 : i32}> : (tensor<1x32x1x262144xf32>) -> tensor<1x32x1x1xf32>
    %1271 = "tosa.const"() <{value = dense<0x49800000> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1272 = "tosa.reciprocal"(%1271) : (tensor<1xf32>) -> tensor<1xf32>
    %1273 = "tosa.mul"(%1272, %1270) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1274 = "tosa.sub"(%1268, %1273) : (tensor<1x32x4x262144xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x4x262144xf32>
    %1275 = "tosa.mul"(%1274, %1274) <{shift = 0 : i8}> : (tensor<1x32x4x262144xf32>, tensor<1x32x4x262144xf32>) -> tensor<1x32x4x262144xf32>
    %1276 = "tosa.reduce_sum"(%1275) <{axis = 2 : i32}> : (tensor<1x32x4x262144xf32>) -> tensor<1x32x1x262144xf32>
    %1277 = "tosa.reduce_sum"(%1276) <{axis = 3 : i32}> : (tensor<1x32x1x262144xf32>) -> tensor<1x32x1x1xf32>
    %1278 = "tosa.const"() <{value = dense<0x49800000> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1279 = "tosa.reciprocal"(%1278) : (tensor<1xf32>) -> tensor<1xf32>
    %1280 = "tosa.mul"(%1279, %1277) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1281 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %1282 = "tosa.add"(%1280, %1281) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1283 = "tosa.rsqrt"(%1282) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1284 = "tosa.sub"(%1268, %1273) : (tensor<1x32x4x262144xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x4x262144xf32>
    %1285 = "tosa.mul"(%1284, %1283) <{shift = 0 : i8}> : (tensor<1x32x4x262144xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x4x262144xf32>
    %1286 = "tosa.reshape"(%1285) <{new_shape = array<i64: 1, 128, 512, 512>}> : (tensor<1x32x4x262144xf32>) -> tensor<1x128x512x512xf32>
    %1287 = "tosa.reshape"(%arg133) <{new_shape = array<i64: 1, 128>}> : (tensor<128xf32>) -> tensor<1x128xf32>
    %1288 = "tosa.reshape"(%1287) <{new_shape = array<i64: 1, 128, 1>}> : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %1289 = "tosa.reshape"(%1288) <{new_shape = array<i64: 1, 128, 1, 1>}> : (tensor<1x128x1xf32>) -> tensor<1x128x1x1xf32>
    %1290 = "tosa.reshape"(%arg134) <{new_shape = array<i64: 1, 128>}> : (tensor<128xf32>) -> tensor<1x128xf32>
    %1291 = "tosa.reshape"(%1290) <{new_shape = array<i64: 1, 128, 1>}> : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %1292 = "tosa.reshape"(%1291) <{new_shape = array<i64: 1, 128, 1, 1>}> : (tensor<1x128x1xf32>) -> tensor<1x128x1x1xf32>
    %1293 = "tosa.mul"(%1286, %1292) <{shift = 0 : i8}> : (tensor<1x128x512x512xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x512x512xf32>
    %1294 = "tosa.add"(%1293, %1289) : (tensor<1x128x512x512xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x512x512xf32>
    %1295 = "tosa.sigmoid"(%1294) : (tensor<1x128x512x512xf32>) -> tensor<1x128x512x512xf32>
    %1296 = "tosa.mul"(%1294, %1295) <{shift = 0 : i8}> : (tensor<1x128x512x512xf32>, tensor<1x128x512x512xf32>) -> tensor<1x128x512x512xf32>
    %1297 = "tosa.identity"(%1296) : (tensor<1x128x512x512xf32>) -> tensor<1x128x512x512xf32>
    %1298 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1299 = "tosa.transpose"(%1297, %1298) : (tensor<1x128x512x512xf32>, tensor<4xi32>) -> tensor<1x512x512x128xf32>
    %1300 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1301 = "tosa.transpose"(%arg135, %1300) : (tensor<128x128x3x3xf32>, tensor<4xi32>) -> tensor<128x3x3x128xf32>
    %1302 = "tosa.conv2d"(%1299, %1301, %arg136) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x512x512x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<1x512x512x128xf32>
    %1303 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1304 = "tosa.transpose"(%1302, %1303) : (tensor<1x512x512x128xf32>, tensor<4xi32>) -> tensor<1x128x512x512xf32>
    %1305 = "tosa.add"(%1230, %1304) : (tensor<1x128x512x512xf32>, tensor<1x128x512x512xf32>) -> tensor<1x128x512x512xf32>
    %1306 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x128x512x512xf32>}> : () -> tensor<1x128x512x512xf32>
    %1307 = "tosa.reciprocal"(%1306) : (tensor<1x128x512x512xf32>) -> tensor<1x128x512x512xf32>
    %1308 = "tosa.mul"(%1305, %1307) <{shift = 0 : i8}> : (tensor<1x128x512x512xf32>, tensor<1x128x512x512xf32>) -> tensor<1x128x512x512xf32>
    %1309 = "tosa.identity"(%1308) : (tensor<1x128x512x512xf32>) -> tensor<1x128x512x512xf32>
    %1310 = "tosa.reshape"(%1309) <{new_shape = array<i64: 1, 32, 4, 262144>}> : (tensor<1x128x512x512xf32>) -> tensor<1x32x4x262144xf32>
    %1311 = "tosa.reduce_sum"(%1310) <{axis = 2 : i32}> : (tensor<1x32x4x262144xf32>) -> tensor<1x32x1x262144xf32>
    %1312 = "tosa.reduce_sum"(%1311) <{axis = 3 : i32}> : (tensor<1x32x1x262144xf32>) -> tensor<1x32x1x1xf32>
    %1313 = "tosa.const"() <{value = dense<0x49800000> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1314 = "tosa.reciprocal"(%1313) : (tensor<1xf32>) -> tensor<1xf32>
    %1315 = "tosa.mul"(%1314, %1312) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1316 = "tosa.sub"(%1310, %1315) : (tensor<1x32x4x262144xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x4x262144xf32>
    %1317 = "tosa.mul"(%1316, %1316) <{shift = 0 : i8}> : (tensor<1x32x4x262144xf32>, tensor<1x32x4x262144xf32>) -> tensor<1x32x4x262144xf32>
    %1318 = "tosa.reduce_sum"(%1317) <{axis = 2 : i32}> : (tensor<1x32x4x262144xf32>) -> tensor<1x32x1x262144xf32>
    %1319 = "tosa.reduce_sum"(%1318) <{axis = 3 : i32}> : (tensor<1x32x1x262144xf32>) -> tensor<1x32x1x1xf32>
    %1320 = "tosa.const"() <{value = dense<0x49800000> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1321 = "tosa.reciprocal"(%1320) : (tensor<1xf32>) -> tensor<1xf32>
    %1322 = "tosa.mul"(%1321, %1319) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1323 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %1324 = "tosa.add"(%1322, %1323) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1325 = "tosa.rsqrt"(%1324) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1326 = "tosa.sub"(%1310, %1315) : (tensor<1x32x4x262144xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x4x262144xf32>
    %1327 = "tosa.mul"(%1326, %1325) <{shift = 0 : i8}> : (tensor<1x32x4x262144xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x4x262144xf32>
    %1328 = "tosa.reshape"(%1327) <{new_shape = array<i64: 1, 128, 512, 512>}> : (tensor<1x32x4x262144xf32>) -> tensor<1x128x512x512xf32>
    %1329 = "tosa.reshape"(%arg137) <{new_shape = array<i64: 1, 128>}> : (tensor<128xf32>) -> tensor<1x128xf32>
    %1330 = "tosa.reshape"(%1329) <{new_shape = array<i64: 1, 128, 1>}> : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %1331 = "tosa.reshape"(%1330) <{new_shape = array<i64: 1, 128, 1, 1>}> : (tensor<1x128x1xf32>) -> tensor<1x128x1x1xf32>
    %1332 = "tosa.reshape"(%arg138) <{new_shape = array<i64: 1, 128>}> : (tensor<128xf32>) -> tensor<1x128xf32>
    %1333 = "tosa.reshape"(%1332) <{new_shape = array<i64: 1, 128, 1>}> : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %1334 = "tosa.reshape"(%1333) <{new_shape = array<i64: 1, 128, 1, 1>}> : (tensor<1x128x1xf32>) -> tensor<1x128x1x1xf32>
    %1335 = "tosa.mul"(%1328, %1334) <{shift = 0 : i8}> : (tensor<1x128x512x512xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x512x512xf32>
    %1336 = "tosa.add"(%1335, %1331) : (tensor<1x128x512x512xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x512x512xf32>
    %1337 = "tosa.sigmoid"(%1336) : (tensor<1x128x512x512xf32>) -> tensor<1x128x512x512xf32>
    %1338 = "tosa.mul"(%1336, %1337) <{shift = 0 : i8}> : (tensor<1x128x512x512xf32>, tensor<1x128x512x512xf32>) -> tensor<1x128x512x512xf32>
    %1339 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1340 = "tosa.transpose"(%1338, %1339) : (tensor<1x128x512x512xf32>, tensor<4xi32>) -> tensor<1x512x512x128xf32>
    %1341 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1342 = "tosa.transpose"(%arg139, %1341) : (tensor<3x128x3x3xf32>, tensor<4xi32>) -> tensor<3x3x3x128xf32>
    %1343 = "tosa.conv2d"(%1340, %1342, %arg140) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x512x512x128xf32>, tensor<3x3x3x128xf32>, tensor<3xf32>) -> tensor<1x512x512x3xf32>
    %1344 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1345 = "tosa.transpose"(%1343, %1344) : (tensor<1x512x512x3xf32>, tensor<4xi32>) -> tensor<1x3x512x512xf32>
    "func.return"(%1345) : (tensor<1x3x512x512xf32>) -> ()
  }) : () -> ()
}) : () -> ()

