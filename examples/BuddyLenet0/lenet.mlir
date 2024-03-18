module {
  func.func @forward(%arg0: tensor<6x1x5x5xf32>, %arg1: tensor<6xf32>, %arg2: tensor<16x6x5x5xf32>, %arg3: tensor<16xf32>, %arg4: tensor<120x16x5x5xf32>, %arg5: tensor<120xf32>, %arg6: tensor<84x120xf32>, %arg7: tensor<84xf32>, %arg8: tensor<10x84xf32>, %arg9: tensor<10xf32>, %arg10: tensor<1x1x28x28xf32>) -> tensor<1x10xf32> {
    %0 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1 = tosa.transpose %arg10, %0 : (tensor<1x1x28x28xf32>, tensor<4xi32>) -> tensor<1x28x28x1xf32>
    %2 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3 = tosa.transpose %arg0, %2 : (tensor<6x1x5x5xf32>, tensor<4xi32>) -> tensor<6x5x5x1xf32>
    %4 = tosa.conv2d %1, %3, %arg1 {dilation = array<i64: 1, 1>, pad = array<i64: 2, 2, 2, 2>, stride = array<i64: 1, 1>} : (tensor<1x28x28x1xf32>, tensor<6x5x5x1xf32>, tensor<6xf32>) -> tensor<1x28x28x6xf32>
    %5 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6 = tosa.transpose %4, %5 : (tensor<1x28x28x6xf32>, tensor<4xi32>) -> tensor<1x6x28x28xf32>
    %7 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x6x28x28xf32>}> : () -> tensor<1x6x28x28xf32>
    %8 = tosa.maximum %6, %7 : (tensor<1x6x28x28xf32>, tensor<1x6x28x28xf32>) -> tensor<1x6x28x28xf32>
    %9 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %10 = tosa.transpose %8, %9 : (tensor<1x6x28x28xf32>, tensor<4xi32>) -> tensor<1x28x28x6xf32>
    %11 = tosa.max_pool2d %10 {kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x28x28x6xf32>) -> tensor<1x14x14x6xf32>
    %12 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %13 = tosa.transpose %11, %12 : (tensor<1x14x14x6xf32>, tensor<4xi32>) -> tensor<1x6x14x14xf32>
    %14 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %15 = tosa.transpose %13, %14 : (tensor<1x6x14x14xf32>, tensor<4xi32>) -> tensor<1x14x14x6xf32>
    %16 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %17 = tosa.transpose %arg2, %16 : (tensor<16x6x5x5xf32>, tensor<4xi32>) -> tensor<16x5x5x6xf32>
    %18 = tosa.conv2d %15, %17, %arg3 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x14x14x6xf32>, tensor<16x5x5x6xf32>, tensor<16xf32>) -> tensor<1x10x10x16xf32>
    %19 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %20 = tosa.transpose %18, %19 : (tensor<1x10x10x16xf32>, tensor<4xi32>) -> tensor<1x16x10x10xf32>
    %21 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x16x10x10xf32>}> : () -> tensor<1x16x10x10xf32>
    %22 = tosa.maximum %20, %21 : (tensor<1x16x10x10xf32>, tensor<1x16x10x10xf32>) -> tensor<1x16x10x10xf32>
    %23 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %24 = tosa.transpose %22, %23 : (tensor<1x16x10x10xf32>, tensor<4xi32>) -> tensor<1x10x10x16xf32>
    %25 = tosa.max_pool2d %24 {kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x10x10x16xf32>) -> tensor<1x5x5x16xf32>
    %26 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %27 = tosa.transpose %25, %26 : (tensor<1x5x5x16xf32>, tensor<4xi32>) -> tensor<1x16x5x5xf32>
    %28 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %29 = tosa.transpose %27, %28 : (tensor<1x16x5x5xf32>, tensor<4xi32>) -> tensor<1x5x5x16xf32>
    %30 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %31 = tosa.transpose %arg4, %30 : (tensor<120x16x5x5xf32>, tensor<4xi32>) -> tensor<120x5x5x16xf32>
    %32 = tosa.conv2d %29, %31, %arg5 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x5x5x16xf32>, tensor<120x5x5x16xf32>, tensor<120xf32>) -> tensor<1x1x1x120xf32>
    %33 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %34 = tosa.transpose %32, %33 : (tensor<1x1x1x120xf32>, tensor<4xi32>) -> tensor<1x120x1x1xf32>
    %35 = tosa.reshape %34 {new_shape = array<i64: 1, 120>} : (tensor<1x120x1x1xf32>) -> tensor<1x120xf32>
    %36 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %37 = tosa.transpose %arg6, %36 : (tensor<84x120xf32>, tensor<2xi32>) -> tensor<120x84xf32>
    %38 = tosa.reshape %35 {new_shape = array<i64: 1, 1, 120>} : (tensor<1x120xf32>) -> tensor<1x1x120xf32>
    %39 = tosa.reshape %37 {new_shape = array<i64: 1, 120, 84>} : (tensor<120x84xf32>) -> tensor<1x120x84xf32>
    %40 = tosa.matmul %38, %39 : (tensor<1x1x120xf32>, tensor<1x120x84xf32>) -> tensor<1x1x84xf32>
    %41 = tosa.reshape %40 {new_shape = array<i64: 1, 84>} : (tensor<1x1x84xf32>) -> tensor<1x84xf32>
    %42 = tosa.reshape %arg7 {new_shape = array<i64: 1, 84>} : (tensor<84xf32>) -> tensor<1x84xf32>
    %43 = tosa.add %42, %41 : (tensor<1x84xf32>, tensor<1x84xf32>) -> tensor<1x84xf32>
    %44 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %45 = tosa.transpose %arg8, %44 : (tensor<10x84xf32>, tensor<2xi32>) -> tensor<84x10xf32>
    %46 = tosa.reshape %43 {new_shape = array<i64: 1, 1, 84>} : (tensor<1x84xf32>) -> tensor<1x1x84xf32>
    %47 = tosa.reshape %45 {new_shape = array<i64: 1, 84, 10>} : (tensor<84x10xf32>) -> tensor<1x84x10xf32>
    %48 = tosa.matmul %46, %47 : (tensor<1x1x84xf32>, tensor<1x84x10xf32>) -> tensor<1x1x10xf32>
    %49 = tosa.reshape %48 {new_shape = array<i64: 1, 10>} : (tensor<1x1x10xf32>) -> tensor<1x10xf32>
    %50 = tosa.reshape %arg9 {new_shape = array<i64: 1, 10>} : (tensor<10xf32>) -> tensor<1x10xf32>
    %51 = tosa.add %50, %49 : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    return %51 : tensor<1x10xf32>
  }
}
