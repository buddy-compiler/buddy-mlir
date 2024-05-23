module {
  func.func private @printMemrefF32(%ptr : tensor<*xf32>)

  func.func @forward(%arg0: tensor<64x3x3x3xf32>, %arg1: tensor<64xf32>, %arg2: tensor<128x64x3x3xf32>, %arg3: tensor<128xf32>, %arg4: tensor<256x128x3x3xf32>, %arg5: tensor<256xf32>, %arg6: tensor<256x256x3x3xf32>, %arg7: tensor<256xf32>, %arg8: tensor<512x256x3x3xf32>, %arg9: tensor<512xf32>, %arg10: tensor<512x512x3x3xf32>, %arg11: tensor<512xf32>, %arg12: tensor<512x512x3x3xf32>, %arg13: tensor<512xf32>, %arg14: tensor<512x512x3x3xf32>, %arg15: tensor<512xf32>, %arg16: tensor<4096x25088xf32>, %arg17: tensor<4096xf32>, %arg18: tensor<4096x4096xf32>, %arg19: tensor<4096xf32>, %arg20: tensor<1000x4096xf32>, %arg21: tensor<1000xf32>, %arg22: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32> {
    %0 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1 = tosa.transpose %arg22, %0 : (tensor<1x3x224x224xf32>, tensor<4xi32>) -> tensor<1x224x224x3xf32>
    %2 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3 = tosa.transpose %arg0, %2 : (tensor<64x3x3x3xf32>, tensor<4xi32>) -> tensor<64x3x3x3xf32>
    %4 = tosa.conv2d %1, %3, %arg1 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x224x224x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x224x224x64xf32>
    %5 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6 = tosa.transpose %4, %5 : (tensor<1x224x224x64xf32>, tensor<4xi32>) -> tensor<1x64x224x224xf32>
    %7 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x64x224x224xf32>}> : () -> tensor<1x64x224x224xf32>
    %8 = tosa.maximum %6, %7 : (tensor<1x64x224x224xf32>, tensor<1x64x224x224xf32>) -> tensor<1x64x224x224xf32>
    %9 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %10 = tosa.transpose %8, %9 : (tensor<1x64x224x224xf32>, tensor<4xi32>) -> tensor<1x224x224x64xf32>
    %11 = tosa.max_pool2d %10 {kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x224x224x64xf32>) -> tensor<1x112x112x64xf32>
    %12 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %13 = tosa.transpose %11, %12 : (tensor<1x112x112x64xf32>, tensor<4xi32>) -> tensor<1x64x112x112xf32>
    %14 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %15 = tosa.transpose %13, %14 : (tensor<1x64x112x112xf32>, tensor<4xi32>) -> tensor<1x112x112x64xf32>
    %16 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %17 = tosa.transpose %arg2, %16 : (tensor<128x64x3x3xf32>, tensor<4xi32>) -> tensor<128x3x3x64xf32>
    %18 = tosa.conv2d %15, %17, %arg3 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x112x112x64xf32>, tensor<128x3x3x64xf32>, tensor<128xf32>) -> tensor<1x112x112x128xf32>
    %19 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %20 = tosa.transpose %18, %19 : (tensor<1x112x112x128xf32>, tensor<4xi32>) -> tensor<1x128x112x112xf32>
    %21 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x128x112x112xf32>}> : () -> tensor<1x128x112x112xf32>
    %22 = tosa.maximum %20, %21 : (tensor<1x128x112x112xf32>, tensor<1x128x112x112xf32>) -> tensor<1x128x112x112xf32>
    %23 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %24 = tosa.transpose %22, %23 : (tensor<1x128x112x112xf32>, tensor<4xi32>) -> tensor<1x112x112x128xf32>
    %25 = tosa.max_pool2d %24 {kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x112x112x128xf32>) -> tensor<1x56x56x128xf32>
    %26 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %27 = tosa.transpose %25, %26 : (tensor<1x56x56x128xf32>, tensor<4xi32>) -> tensor<1x128x56x56xf32>
    %28 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %29 = tosa.transpose %27, %28 : (tensor<1x128x56x56xf32>, tensor<4xi32>) -> tensor<1x56x56x128xf32>
    %30 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %31 = tosa.transpose %arg4, %30 : (tensor<256x128x3x3xf32>, tensor<4xi32>) -> tensor<256x3x3x128xf32>
    %32 = tosa.conv2d %29, %31, %arg5 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x56x56x128xf32>, tensor<256x3x3x128xf32>, tensor<256xf32>) -> tensor<1x56x56x256xf32>
    %33 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %34 = tosa.transpose %32, %33 : (tensor<1x56x56x256xf32>, tensor<4xi32>) -> tensor<1x256x56x56xf32>
    %35 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x256x56x56xf32>}> : () -> tensor<1x256x56x56xf32>
    %36 = tosa.maximum %34, %35 : (tensor<1x256x56x56xf32>, tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %37 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %38 = tosa.transpose %36, %37 : (tensor<1x256x56x56xf32>, tensor<4xi32>) -> tensor<1x56x56x256xf32>
    %39 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %40 = tosa.transpose %arg6, %39 : (tensor<256x256x3x3xf32>, tensor<4xi32>) -> tensor<256x3x3x256xf32>
    %41 = tosa.conv2d %38, %40, %arg7 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x56x56x256xf32>, tensor<256x3x3x256xf32>, tensor<256xf32>) -> tensor<1x56x56x256xf32>
    %42 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %43 = tosa.transpose %41, %42 : (tensor<1x56x56x256xf32>, tensor<4xi32>) -> tensor<1x256x56x56xf32>
    %44 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x256x56x56xf32>}> : () -> tensor<1x256x56x56xf32>
    %45 = tosa.maximum %43, %44 : (tensor<1x256x56x56xf32>, tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %46 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %47 = tosa.transpose %45, %46 : (tensor<1x256x56x56xf32>, tensor<4xi32>) -> tensor<1x56x56x256xf32>
    %48 = tosa.max_pool2d %47 {kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x56x56x256xf32>) -> tensor<1x28x28x256xf32>
    %49 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %50 = tosa.transpose %48, %49 : (tensor<1x28x28x256xf32>, tensor<4xi32>) -> tensor<1x256x28x28xf32>
    %51 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %52 = tosa.transpose %50, %51 : (tensor<1x256x28x28xf32>, tensor<4xi32>) -> tensor<1x28x28x256xf32>
    %53 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %54 = tosa.transpose %arg8, %53 : (tensor<512x256x3x3xf32>, tensor<4xi32>) -> tensor<512x3x3x256xf32>
    %55 = tosa.conv2d %52, %54, %arg9 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x28x28x256xf32>, tensor<512x3x3x256xf32>, tensor<512xf32>) -> tensor<1x28x28x512xf32>
    %56 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %57 = tosa.transpose %55, %56 : (tensor<1x28x28x512xf32>, tensor<4xi32>) -> tensor<1x512x28x28xf32>
    %58 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x512x28x28xf32>}> : () -> tensor<1x512x28x28xf32>
    %59 = tosa.maximum %57, %58 : (tensor<1x512x28x28xf32>, tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %60 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %61 = tosa.transpose %59, %60 : (tensor<1x512x28x28xf32>, tensor<4xi32>) -> tensor<1x28x28x512xf32>
    %62 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %63 = tosa.transpose %arg10, %62 : (tensor<512x512x3x3xf32>, tensor<4xi32>) -> tensor<512x3x3x512xf32>
    %64 = tosa.conv2d %61, %63, %arg11 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x28x28x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>) -> tensor<1x28x28x512xf32>
    %65 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %66 = tosa.transpose %64, %65 : (tensor<1x28x28x512xf32>, tensor<4xi32>) -> tensor<1x512x28x28xf32>
    %67 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x512x28x28xf32>}> : () -> tensor<1x512x28x28xf32>
    %68 = tosa.maximum %66, %67 : (tensor<1x512x28x28xf32>, tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %69 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %70 = tosa.transpose %68, %69 : (tensor<1x512x28x28xf32>, tensor<4xi32>) -> tensor<1x28x28x512xf32>
    %71 = tosa.max_pool2d %70 {kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x28x28x512xf32>) -> tensor<1x14x14x512xf32>
    %72 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %73 = tosa.transpose %71, %72 : (tensor<1x14x14x512xf32>, tensor<4xi32>) -> tensor<1x512x14x14xf32>
    %74 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %75 = tosa.transpose %73, %74 : (tensor<1x512x14x14xf32>, tensor<4xi32>) -> tensor<1x14x14x512xf32>
    %76 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %77 = tosa.transpose %arg12, %76 : (tensor<512x512x3x3xf32>, tensor<4xi32>) -> tensor<512x3x3x512xf32>
    %78 = tosa.conv2d %75, %77, %arg13 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x14x14x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>) -> tensor<1x14x14x512xf32>
    %79 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %80 = tosa.transpose %78, %79 : (tensor<1x14x14x512xf32>, tensor<4xi32>) -> tensor<1x512x14x14xf32>
    %81 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x512x14x14xf32>}> : () -> tensor<1x512x14x14xf32>
    %82 = tosa.maximum %80, %81 : (tensor<1x512x14x14xf32>, tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
    %83 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %84 = tosa.transpose %82, %83 : (tensor<1x512x14x14xf32>, tensor<4xi32>) -> tensor<1x14x14x512xf32>
    %85 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %86 = tosa.transpose %arg14, %85 : (tensor<512x512x3x3xf32>, tensor<4xi32>) -> tensor<512x3x3x512xf32>
    %87 = tosa.conv2d %84, %86, %arg15 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x14x14x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>) -> tensor<1x14x14x512xf32>
    %88 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %89 = tosa.transpose %87, %88 : (tensor<1x14x14x512xf32>, tensor<4xi32>) -> tensor<1x512x14x14xf32>
    %90 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x512x14x14xf32>}> : () -> tensor<1x512x14x14xf32>
    %91 = tosa.maximum %89, %90 : (tensor<1x512x14x14xf32>, tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
    %92 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %93 = tosa.transpose %91, %92 : (tensor<1x512x14x14xf32>, tensor<4xi32>) -> tensor<1x14x14x512xf32>
    %94 = tosa.max_pool2d %93 {kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x14x14x512xf32>) -> tensor<1x7x7x512xf32>
    %95 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %96 = tosa.transpose %94, %95 : (tensor<1x7x7x512xf32>, tensor<4xi32>) -> tensor<1x512x7x7xf32>
    %97 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %98 = tosa.transpose %96, %97 : (tensor<1x512x7x7xf32>, tensor<4xi32>) -> tensor<1x7x7x512xf32>
    %99 = tosa.avg_pool2d %98 {acc_type = f32, kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %100 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %101 = tosa.transpose %99, %100 : (tensor<1x7x7x512xf32>, tensor<4xi32>) -> tensor<1x512x7x7xf32>
    %102 = tosa.reshape %101 {new_shape = array<i64: 1, 25088>} : (tensor<1x512x7x7xf32>) -> tensor<1x25088xf32>
    %103 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %104 = tosa.transpose %arg16, %103 : (tensor<4096x25088xf32>, tensor<2xi32>) -> tensor<25088x4096xf32>
    %105 = tosa.reshape %102 {new_shape = array<i64: 1, 1, 25088>} : (tensor<1x25088xf32>) -> tensor<1x1x25088xf32>
    %106 = tosa.reshape %104 {new_shape = array<i64: 1, 25088, 4096>} : (tensor<25088x4096xf32>) -> tensor<1x25088x4096xf32>
    %107 = tosa.matmul %105, %106 : (tensor<1x1x25088xf32>, tensor<1x25088x4096xf32>) -> tensor<1x1x4096xf32>
    %108 = tosa.reshape %107 {new_shape = array<i64: 1, 4096>} : (tensor<1x1x4096xf32>) -> tensor<1x4096xf32>
    %109 = tosa.reshape %arg17 {new_shape = array<i64: 1, 4096>} : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %110 = tosa.add %109, %108 : (tensor<1x4096xf32>, tensor<1x4096xf32>) -> tensor<1x4096xf32>
    %111 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x4096xf32>}> : () -> tensor<1x4096xf32>
    %112 = tosa.maximum %110, %111 : (tensor<1x4096xf32>, tensor<1x4096xf32>) -> tensor<1x4096xf32>
    %113 = tosa.identity %112 : (tensor<1x4096xf32>) -> tensor<1x4096xf32>
    %114 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %115 = tosa.transpose %arg18, %114 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %116 = tosa.reshape %113 {new_shape = array<i64: 1, 1, 4096>} : (tensor<1x4096xf32>) -> tensor<1x1x4096xf32>
    %117 = tosa.reshape %115 {new_shape = array<i64: 1, 4096, 4096>} : (tensor<4096x4096xf32>) -> tensor<1x4096x4096xf32>
    %118 = tosa.matmul %116, %117 : (tensor<1x1x4096xf32>, tensor<1x4096x4096xf32>) -> tensor<1x1x4096xf32>
    %119 = tosa.reshape %118 {new_shape = array<i64: 1, 4096>} : (tensor<1x1x4096xf32>) -> tensor<1x4096xf32>
    %120 = tosa.reshape %arg19 {new_shape = array<i64: 1, 4096>} : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %121 = tosa.add %120, %119 : (tensor<1x4096xf32>, tensor<1x4096xf32>) -> tensor<1x4096xf32>
    %122 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x4096xf32>}> : () -> tensor<1x4096xf32>
    %123 = tosa.maximum %121, %122 : (tensor<1x4096xf32>, tensor<1x4096xf32>) -> tensor<1x4096xf32>
    %124 = tosa.identity %123 : (tensor<1x4096xf32>) -> tensor<1x4096xf32>
    %125 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %126 = tosa.transpose %arg20, %125 : (tensor<1000x4096xf32>, tensor<2xi32>) -> tensor<4096x1000xf32>
    %127 = tosa.reshape %124 {new_shape = array<i64: 1, 1, 4096>} : (tensor<1x4096xf32>) -> tensor<1x1x4096xf32>
    %128 = tosa.reshape %126 {new_shape = array<i64: 1, 4096, 1000>} : (tensor<4096x1000xf32>) -> tensor<1x4096x1000xf32>
    %129 = tosa.matmul %127, %128 : (tensor<1x1x4096xf32>, tensor<1x4096x1000xf32>) -> tensor<1x1x1000xf32>
    %130 = tosa.reshape %129 {new_shape = array<i64: 1, 1000>} : (tensor<1x1x1000xf32>) -> tensor<1x1000xf32>
    %131 = tosa.reshape %arg21 {new_shape = array<i64: 1, 1000>} : (tensor<1000xf32>) -> tensor<1x1000xf32>
    %132 = tosa.add %131, %130 : (tensor<1x1000xf32>, tensor<1x1000xf32>) -> tensor<1x1000xf32>
    return %132 : tensor<1x1000xf32>
  }


  func.func @main() {
    %arg0 = arith.constant dense<1.0> : tensor<64x3x3x3xf32>
    %arg1 = arith.constant dense<1.0> : tensor<64xf32>
    %arg2 = arith.constant dense<1.0> : tensor<128x64x3x3xf32>
    %arg3 = arith.constant dense<1.0> : tensor<128xf32>
    %arg4 = arith.constant dense<1.0> : tensor<256x128x3x3xf32>
    %arg5 = arith.constant dense<1.0> : tensor<256xf32>
    %arg6 = arith.constant dense<1.0> : tensor<256x256x3x3xf32>
    %arg7 = arith.constant dense<1.0> : tensor<256xf32>
    %arg8 = arith.constant dense<1.0> : tensor<512x256x3x3xf32>
    %arg9 = arith.constant dense<1.0> : tensor<512xf32>
    %arg10 = arith.constant dense<1.0> : tensor<512x512x3x3xf32>
    %arg11 = arith.constant dense<1.0> : tensor<512xf32>
    %arg12 = arith.constant dense<1.0> : tensor<512x512x3x3xf32>
    %arg13 = arith.constant dense<1.0> : tensor<512xf32>
    %arg14 = arith.constant dense<1.0> : tensor<512x512x3x3xf32>
    %arg15 = arith.constant dense<1.0> : tensor<512xf32>
    %arg16 = arith.constant dense<1.0> : tensor<4096x25088xf32>
    %arg17 = arith.constant dense<1.0> : tensor<4096xf32>
    %arg18 = arith.constant dense<1.0> : tensor<4096x4096xf32>
    %arg19 = arith.constant dense<1.0> : tensor<4096xf32>
    %arg20 = arith.constant dense<1.0> : tensor<1000x4096xf32>
    %arg21 = arith.constant dense<1.0> : tensor<1000xf32>
    %arg22 = arith.constant dense<2.0> : tensor<1x3x224x224xf32>
    
    %fake_output = call @forward(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22) : (tensor<64x3x3x3xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<4096x25088xf32>, tensor<4096xf32>, tensor<4096x4096xf32>, tensor<4096xf32>, tensor<1000x4096xf32>, tensor<1000xf32>, tensor<1x3x224x224xf32>) -> tensor<1x1000xf32>
  
    %tensor_unranked = tensor.cast %fake_output : tensor<1x1000xf32> to tensor<*xf32>
    call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
    return
  }

}

