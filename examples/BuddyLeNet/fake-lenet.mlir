module {
  func.func private @printMemrefF32(%ptr : tensor<*xf32>)
  func.func private @rtclock() -> f64

  func.func @forward(%arg0: tensor<44426xf32>, %arg1: tensor<1x1x28x28xf32>) -> tensor<1x10xf32> {
    %extracted_slice = tensor.extract_slice %arg0[0] [150] [1] : tensor<44426xf32> to tensor<150xf32>
    %expanded = tensor.expand_shape %extracted_slice [[0, 1, 2, 3]] : tensor<150xf32> into tensor<6x1x5x5xf32>
    %extracted_slice_0 = tensor.extract_slice %arg0[150] [6] [1] : tensor<44426xf32> to tensor<6xf32>
    %extracted_slice_1 = tensor.extract_slice %arg0[156] [2400] [1] : tensor<44426xf32> to tensor<2400xf32>
    %expanded_2 = tensor.expand_shape %extracted_slice_1 [[0, 1, 2, 3]] : tensor<2400xf32> into tensor<16x6x5x5xf32>
    %extracted_slice_3 = tensor.extract_slice %arg0[2556] [16] [1] : tensor<44426xf32> to tensor<16xf32>
    %extracted_slice_4 = tensor.extract_slice %arg0[2572] [30720] [1] : tensor<44426xf32> to tensor<30720xf32>
    %expanded_5 = tensor.expand_shape %extracted_slice_4 [[0, 1]] : tensor<30720xf32> into tensor<120x256xf32>
    %extracted_slice_6 = tensor.extract_slice %arg0[33292] [120] [1] : tensor<44426xf32> to tensor<120xf32>
    %extracted_slice_7 = tensor.extract_slice %arg0[33412] [10080] [1] : tensor<44426xf32> to tensor<10080xf32>
    %expanded_8 = tensor.expand_shape %extracted_slice_7 [[0, 1]] : tensor<10080xf32> into tensor<84x120xf32>
    %extracted_slice_9 = tensor.extract_slice %arg0[43492] [84] [1] : tensor<44426xf32> to tensor<84xf32>
    %extracted_slice_10 = tensor.extract_slice %arg0[43576] [840] [1] : tensor<44426xf32> to tensor<840xf32>
    %expanded_11 = tensor.expand_shape %extracted_slice_10 [[0, 1]] : tensor<840xf32> into tensor<10x84xf32>
    %extracted_slice_12 = tensor.extract_slice %arg0[44416] [10] [1] : tensor<44426xf32> to tensor<10xf32>
    %0 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1 = tosa.transpose %arg1, %0 : (tensor<1x1x28x28xf32>, tensor<4xi32>) -> tensor<1x28x28x1xf32>
    %2 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3 = tosa.transpose %expanded, %2 : (tensor<6x1x5x5xf32>, tensor<4xi32>) -> tensor<6x5x5x1xf32>
    %4 = tosa.conv2d %1, %3, %extracted_slice_0 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x28x28x1xf32>, tensor<6x5x5x1xf32>, tensor<6xf32>) -> tensor<1x24x24x6xf32>
    %5 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6 = tosa.transpose %4, %5 : (tensor<1x24x24x6xf32>, tensor<4xi32>) -> tensor<1x6x24x24xf32>
    %7 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x6x24x24xf32>}> : () -> tensor<1x6x24x24xf32>
    %8 = tosa.maximum %6, %7 : (tensor<1x6x24x24xf32>, tensor<1x6x24x24xf32>) -> tensor<1x6x24x24xf32>
    %9 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %10 = tosa.transpose %8, %9 : (tensor<1x6x24x24xf32>, tensor<4xi32>) -> tensor<1x24x24x6xf32>
    %11 = tosa.max_pool2d %10 {kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x24x24x6xf32>) -> tensor<1x12x12x6xf32>
    %12 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %13 = tosa.transpose %11, %12 : (tensor<1x12x12x6xf32>, tensor<4xi32>) -> tensor<1x6x12x12xf32>
    %14 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %15 = tosa.transpose %13, %14 : (tensor<1x6x12x12xf32>, tensor<4xi32>) -> tensor<1x12x12x6xf32>
    %16 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %17 = tosa.transpose %expanded_2, %16 : (tensor<16x6x5x5xf32>, tensor<4xi32>) -> tensor<16x5x5x6xf32>
    %18 = tosa.conv2d %15, %17, %extracted_slice_3 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x12x12x6xf32>, tensor<16x5x5x6xf32>, tensor<16xf32>) -> tensor<1x8x8x16xf32>
    %19 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %20 = tosa.transpose %18, %19 : (tensor<1x8x8x16xf32>, tensor<4xi32>) -> tensor<1x16x8x8xf32>
    %21 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x16x8x8xf32>}> : () -> tensor<1x16x8x8xf32>
    %22 = tosa.maximum %20, %21 : (tensor<1x16x8x8xf32>, tensor<1x16x8x8xf32>) -> tensor<1x16x8x8xf32>
    %23 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %24 = tosa.transpose %22, %23 : (tensor<1x16x8x8xf32>, tensor<4xi32>) -> tensor<1x8x8x16xf32>
    %25 = tosa.max_pool2d %24 {kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x8x8x16xf32>) -> tensor<1x4x4x16xf32>
    %26 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %27 = tosa.transpose %25, %26 : (tensor<1x4x4x16xf32>, tensor<4xi32>) -> tensor<1x16x4x4xf32>
    %28 = tosa.reshape %27 {new_shape = array<i64: 1, 256>} : (tensor<1x16x4x4xf32>) -> tensor<1x256xf32>
    %29 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %30 = tosa.transpose %expanded_5, %29 : (tensor<120x256xf32>, tensor<2xi32>) -> tensor<256x120xf32>
    %31 = tosa.reshape %28 {new_shape = array<i64: 1, 1, 256>} : (tensor<1x256xf32>) -> tensor<1x1x256xf32>
    %32 = tosa.reshape %30 {new_shape = array<i64: 1, 256, 120>} : (tensor<256x120xf32>) -> tensor<1x256x120xf32>
    %33 = tosa.matmul %31, %32 : (tensor<1x1x256xf32>, tensor<1x256x120xf32>) -> tensor<1x1x120xf32>
    %34 = tosa.reshape %33 {new_shape = array<i64: 1, 120>} : (tensor<1x1x120xf32>) -> tensor<1x120xf32>
    %35 = tosa.reshape %extracted_slice_6 {new_shape = array<i64: 1, 120>} : (tensor<120xf32>) -> tensor<1x120xf32>
    %36 = tosa.add %35, %34 : (tensor<1x120xf32>, tensor<1x120xf32>) -> tensor<1x120xf32>
    %37 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x120xf32>}> : () -> tensor<1x120xf32>
    %38 = tosa.maximum %36, %37 : (tensor<1x120xf32>, tensor<1x120xf32>) -> tensor<1x120xf32>
    %39 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %40 = tosa.transpose %expanded_8, %39 : (tensor<84x120xf32>, tensor<2xi32>) -> tensor<120x84xf32>
    %41 = tosa.reshape %38 {new_shape = array<i64: 1, 1, 120>} : (tensor<1x120xf32>) -> tensor<1x1x120xf32>
    %42 = tosa.reshape %40 {new_shape = array<i64: 1, 120, 84>} : (tensor<120x84xf32>) -> tensor<1x120x84xf32>
    %43 = tosa.matmul %41, %42 : (tensor<1x1x120xf32>, tensor<1x120x84xf32>) -> tensor<1x1x84xf32>
    %44 = tosa.reshape %43 {new_shape = array<i64: 1, 84>} : (tensor<1x1x84xf32>) -> tensor<1x84xf32>
    %45 = tosa.reshape %extracted_slice_9 {new_shape = array<i64: 1, 84>} : (tensor<84xf32>) -> tensor<1x84xf32>
    %46 = tosa.add %45, %44 : (tensor<1x84xf32>, tensor<1x84xf32>) -> tensor<1x84xf32>
    %47 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x84xf32>}> : () -> tensor<1x84xf32>
    %48 = tosa.maximum %46, %47 : (tensor<1x84xf32>, tensor<1x84xf32>) -> tensor<1x84xf32>
    %49 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %50 = tosa.transpose %expanded_11, %49 : (tensor<10x84xf32>, tensor<2xi32>) -> tensor<84x10xf32>
    %51 = tosa.reshape %48 {new_shape = array<i64: 1, 1, 84>} : (tensor<1x84xf32>) -> tensor<1x1x84xf32>
    %52 = tosa.reshape %50 {new_shape = array<i64: 1, 84, 10>} : (tensor<84x10xf32>) -> tensor<1x84x10xf32>
    %53 = tosa.matmul %51, %52 : (tensor<1x1x84xf32>, tensor<1x84x10xf32>) -> tensor<1x1x10xf32>
    %54 = tosa.reshape %53 {new_shape = array<i64: 1, 10>} : (tensor<1x1x10xf32>) -> tensor<1x10xf32>
    %55 = tosa.reshape %extracted_slice_12 {new_shape = array<i64: 1, 10>} : (tensor<10xf32>) -> tensor<1x10xf32>
    %56 = tosa.add %55, %54 : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    return %56 : tensor<1x10xf32>
  }

  func.func @main() {
    %fake_params = arith.constant dense<1.0> : tensor<44426xf32>
    %fake_input = arith.constant dense<2.0> : tensor<1x1x28x28xf32>
    
    %t_start = call @rtclock() : () -> f64
    %fake_output = call @forward(%fake_params, %fake_input) : (tensor<44426xf32>, tensor<1x1x28x28xf32>) -> tensor<1x10xf32>
    %t_end = call @rtclock() : () -> f64
  
    %tensor_unranked = tensor.cast %fake_output : tensor<1x10xf32> to tensor<*xf32>
    call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()

    %time = arith.subf %t_end, %t_start : f64
    vector.print %time : f64

    return
  }
}
