module {
    func.func private @printMemrefF32(%ptr : tensor<*xf32>)
    func.func private @rtclock() -> f64

    func.func @forward(%arg0: tensor<1x1x784xf32>, %arg1: tensor<32x784xf32>, %arg2: tensor<32xf32>, %arg3: tensor<32x32xf32>, %arg4: tensor<32xf32>, %arg5: tensor<10x32xf32>, %arg6: tensor<10xf32>) -> tensor<1x1x10xf32> {
        %0 = tosa.reshape %arg0 {new_shape = array<i64: 1, 784>} : (tensor<1x1x784xf32>) -> tensor<1x784xf32>
        %1 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
        %2 = tosa.transpose %arg1, %1 : (tensor<32x784xf32>, tensor<2xi32>) -> tensor<784x32xf32>
        %3 = tosa.reshape %0 {new_shape = array<i64: 1, 1, 784>} : (tensor<1x784xf32>) -> tensor<1x1x784xf32>
        %4 = tosa.reshape %2 {new_shape = array<i64: 1, 784, 32>} : (tensor<784x32xf32>) -> tensor<1x784x32xf32>
        %5 = tosa.matmul %3, %4 : (tensor<1x1x784xf32>, tensor<1x784x32xf32>) -> tensor<1x1x32xf32>
        %6 = tosa.reshape %5 {new_shape = array<i64: 1, 32>} : (tensor<1x1x32xf32>) -> tensor<1x32xf32>
        %7 = tosa.reshape %arg2 {new_shape = array<i64: 1, 32>} : (tensor<32xf32>) -> tensor<1x32xf32>
        %8 = tosa.add %7, %6 : (tensor<1x32xf32>, tensor<1x32xf32>) -> tensor<1x32xf32>
        %9 = tosa.reshape %8 {new_shape = array<i64: 1, 1, 32>} : (tensor<1x32xf32>) -> tensor<1x1x32xf32>
        %10 = tosa.sigmoid %9 : (tensor<1x1x32xf32>) -> tensor<1x1x32xf32>
        %11 = tosa.mul %9, %10 : (tensor<1x1x32xf32>, tensor<1x1x32xf32>) -> tensor<1x1x32xf32>
        %12 = tosa.reshape %11 {new_shape = array<i64: 1, 32>} : (tensor<1x1x32xf32>) -> tensor<1x32xf32>
        %13 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
        %14 = tosa.transpose %arg3, %13 : (tensor<32x32xf32>, tensor<2xi32>) -> tensor<32x32xf32>
        %15 = tosa.reshape %12 {new_shape = array<i64: 1, 1, 32>} : (tensor<1x32xf32>) -> tensor<1x1x32xf32>
        %16 = tosa.reshape %14 {new_shape = array<i64: 1, 32, 32>} : (tensor<32x32xf32>) -> tensor<1x32x32xf32>
        %17 = tosa.matmul %15, %16 : (tensor<1x1x32xf32>, tensor<1x32x32xf32>) -> tensor<1x1x32xf32>
        %18 = tosa.reshape %17 {new_shape = array<i64: 1, 32>} : (tensor<1x1x32xf32>) -> tensor<1x32xf32>
        %19 = tosa.reshape %arg4 {new_shape = array<i64: 1, 32>} : (tensor<32xf32>) -> tensor<1x32xf32>
        %20 = tosa.add %19, %18 : (tensor<1x32xf32>, tensor<1x32xf32>) -> tensor<1x32xf32>
        %21 = tosa.reshape %20 {new_shape = array<i64: 1, 1, 32>} : (tensor<1x32xf32>) -> tensor<1x1x32xf32>
        %22 = tosa.sigmoid %21 : (tensor<1x1x32xf32>) -> tensor<1x1x32xf32>
        %23 = tosa.mul %21, %22 : (tensor<1x1x32xf32>, tensor<1x1x32xf32>) -> tensor<1x1x32xf32>
        %24 = tosa.reshape %23 {new_shape = array<i64: 1, 32>} : (tensor<1x1x32xf32>) -> tensor<1x32xf32>
        %25 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
        %26 = tosa.transpose %arg5, %25 : (tensor<10x32xf32>, tensor<2xi32>) -> tensor<32x10xf32>
        %27 = tosa.reshape %24 {new_shape = array<i64: 1, 1, 32>} : (tensor<1x32xf32>) -> tensor<1x1x32xf32>
        %28 = tosa.reshape %26 {new_shape = array<i64: 1, 32, 10>} : (tensor<32x10xf32>) -> tensor<1x32x10xf32>
        %29 = tosa.matmul %27, %28 : (tensor<1x1x32xf32>, tensor<1x32x10xf32>) -> tensor<1x1x10xf32>
        %30 = tosa.reshape %29 {new_shape = array<i64: 1, 10>} : (tensor<1x1x10xf32>) -> tensor<1x10xf32>
        %31 = tosa.reshape %arg6 {new_shape = array<i64: 1, 10>} : (tensor<10xf32>) -> tensor<1x10xf32>
        %32 = tosa.add %31, %30 : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
        %33 = tosa.reshape %32 {new_shape = array<i64: 1, 1, 10>} : (tensor<1x10xf32>) -> tensor<1x1x10xf32>
        return %33 : tensor<1x1x10xf32>
    }
    func.func @main() {
        %fake_input = arith.constant dense<2.0> : tensor<1x1x784xf32>
        
        %fake_weight1 = arith.constant dense<0.5> : tensor<32x784xf32>
        
        %fake_bias1 = arith.constant dense<0.1> : tensor<32xf32>
        
        %fake_weight2 = arith.constant dense<0.3> : tensor<32x32xf32>
        
        %fake_bias2 = arith.constant dense<0.1> : tensor<32xf32>
        
        %fake_weight3 = arith.constant dense<0.2> : tensor<10x32xf32>
        
        %fake_bias3 = arith.constant dense<0.1> : tensor<10xf32>
        
        %t_start = call @rtclock() : () -> f64
        %fake_output = call @forward(%fake_input, %fake_weight1, 
            %fake_bias1, %fake_weight2, %fake_bias2,
             %fake_weight3, %fake_bias3) : 
             (tensor<1x1x784xf32>, tensor<32x784xf32>, 
                tensor<32xf32>, tensor<32x32xf32>, 
                tensor<32xf32>, tensor<10x32xf32>, 
                tensor<10xf32>) -> tensor<1x1x10xf32>
        %t_end = call @rtclock() : () -> f64
      
        %tensor_unranked = tensor.cast %fake_output : tensor<1x1x10xf32> to tensor<*xf32>
        call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
    
        %time = arith.subf %t_end, %t_start : f64
        vector.print %time : f64
    
        return
      }
}