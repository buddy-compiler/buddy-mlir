module {
  func.func @forward(%arg0: tensor<61706xf32>, %arg1: tensor<1x1x28x28xf32>) -> tensor<1x10xf32> {
    %extracted_slice = tensor.extract_slice %arg0[0] [150] [1] : tensor<61706xf32> to tensor<150xf32>
    %expanded = tensor.expand_shape %extracted_slice [[0, 1, 2, 3]] : tensor<150xf32> into tensor<6x1x5x5xf32>
    %extracted_slice_0 = tensor.extract_slice %arg0[150] [6] [1] : tensor<61706xf32> to tensor<6xf32>
    %extracted_slice_1 = tensor.extract_slice %arg0[156] [2400] [1] : tensor<61706xf32> to tensor<2400xf32>
    %expanded_2 = tensor.expand_shape %extracted_slice_1 [[0, 1, 2, 3]] : tensor<2400xf32> into tensor<16x6x5x5xf32>
    %extracted_slice_3 = tensor.extract_slice %arg0[2556] [16] [1] : tensor<61706xf32> to tensor<16xf32>
    %extracted_slice_4 = tensor.extract_slice %arg0[2572] [48000] [1] : tensor<61706xf32> to tensor<48000xf32>
    %expanded_5 = tensor.expand_shape %extracted_slice_4 [[0, 1, 2, 3]] : tensor<48000xf32> into tensor<120x16x5x5xf32>
    %extracted_slice_6 = tensor.extract_slice %arg0[50572] [120] [1] : tensor<61706xf32> to tensor<120xf32>
    %extracted_slice_7 = tensor.extract_slice %arg0[50692] [10080] [1] : tensor<61706xf32> to tensor<10080xf32>
    %expanded_8 = tensor.expand_shape %extracted_slice_7 [[0, 1]] : tensor<10080xf32> into tensor<84x120xf32>
    %extracted_slice_9 = tensor.extract_slice %arg0[60772] [84] [1] : tensor<61706xf32> to tensor<84xf32>
    %extracted_slice_10 = tensor.extract_slice %arg0[60856] [840] [1] : tensor<61706xf32> to tensor<840xf32>
    %expanded_11 = tensor.expand_shape %extracted_slice_10 [[0, 1]] : tensor<840xf32> into tensor<10x84xf32>
    %extracted_slice_12 = tensor.extract_slice %arg0[61696] [10] [1] : tensor<61706xf32> to tensor<10xf32>
    %0 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1 = tosa.transpose %arg1, %0 : (tensor<1x1x28x28xf32>, tensor<4xi32>) -> tensor<1x28x28x1xf32>
    %2 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3 = tosa.transpose %expanded, %2 : (tensor<6x1x5x5xf32>, tensor<4xi32>) -> tensor<6x5x5x1xf32>
    %4 = tosa.conv2d %1, %3, %extracted_slice_0 {dilation = array<i64: 1, 1>, pad = array<i64: 2, 2, 2, 2>, stride = array<i64: 1, 1>} : (tensor<1x28x28x1xf32>, tensor<6x5x5x1xf32>, tensor<6xf32>) -> tensor<1x28x28x6xf32>
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
    %17 = tosa.transpose %expanded_2, %16 : (tensor<16x6x5x5xf32>, tensor<4xi32>) -> tensor<16x5x5x6xf32>
    %18 = tosa.conv2d %15, %17, %extracted_slice_3 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x14x14x6xf32>, tensor<16x5x5x6xf32>, tensor<16xf32>) -> tensor<1x10x10x16xf32>
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
    %31 = tosa.transpose %expanded_5, %30 : (tensor<120x16x5x5xf32>, tensor<4xi32>) -> tensor<120x5x5x16xf32>
    %32 = tosa.conv2d %29, %31, %extracted_slice_6 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x5x5x16xf32>, tensor<120x5x5x16xf32>, tensor<120xf32>) -> tensor<1x1x1x120xf32>
    %33 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %34 = tosa.transpose %32, %33 : (tensor<1x1x1x120xf32>, tensor<4xi32>) -> tensor<1x120x1x1xf32>
    %35 = tosa.reshape %34 {new_shape = array<i64: 1, 120>} : (tensor<1x120x1x1xf32>) -> tensor<1x120xf32>
    %36 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %37 = tosa.transpose %expanded_8, %36 : (tensor<84x120xf32>, tensor<2xi32>) -> tensor<120x84xf32>
    %38 = tosa.reshape %35 {new_shape = array<i64: 1, 1, 120>} : (tensor<1x120xf32>) -> tensor<1x1x120xf32>
    %39 = tosa.reshape %37 {new_shape = array<i64: 1, 120, 84>} : (tensor<120x84xf32>) -> tensor<1x120x84xf32>
    %40 = tosa.matmul %38, %39 : (tensor<1x1x120xf32>, tensor<1x120x84xf32>) -> tensor<1x1x84xf32>
    %41 = tosa.reshape %40 {new_shape = array<i64: 1, 84>} : (tensor<1x1x84xf32>) -> tensor<1x84xf32>
    %42 = tosa.reshape %extracted_slice_9 {new_shape = array<i64: 1, 84>} : (tensor<84xf32>) -> tensor<1x84xf32>
    %43 = tosa.add %42, %41 : (tensor<1x84xf32>, tensor<1x84xf32>) -> tensor<1x84xf32>
    %44 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %45 = tosa.transpose %expanded_11, %44 : (tensor<10x84xf32>, tensor<2xi32>) -> tensor<84x10xf32>
    %46 = tosa.reshape %43 {new_shape = array<i64: 1, 1, 84>} : (tensor<1x84xf32>) -> tensor<1x1x84xf32>
    %47 = tosa.reshape %45 {new_shape = array<i64: 1, 84, 10>} : (tensor<84x10xf32>) -> tensor<1x84x10xf32>
    %48 = tosa.matmul %46, %47 : (tensor<1x1x84xf32>, tensor<1x84x10xf32>) -> tensor<1x1x10xf32>
    %49 = tosa.reshape %48 {new_shape = array<i64: 1, 10>} : (tensor<1x1x10xf32>) -> tensor<1x10xf32>
    %50 = tosa.reshape %extracted_slice_12 {new_shape = array<i64: 1, 10>} : (tensor<10xf32>) -> tensor<1x10xf32>
    %51 = tosa.add %50, %49 : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    return %51 : tensor<1x10xf32>
  }
}
