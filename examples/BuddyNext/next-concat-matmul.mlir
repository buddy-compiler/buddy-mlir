module {
    func.func private @printMemrefF32(tensor<*xf32>)
    func.func private @rtclock() -> f64

    func.func @qkv_compute_original(%arg1: tensor<4096x4096xf32>, %arg2: tensor<4096x4096xf32>, %arg3: tensor<4096x4096xf32>) {
        // %41 = tosa.mul %40, %39 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
        %41 = arith.constant dense<1.0> : tensor<1x40x4096xf32>
        
        %t0_original = call @rtclock() : () -> f64 

        %42 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
        %43 = tosa.transpose %arg1, %42 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
        %44 = tosa.reshape %41 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
        %cst_6 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
        %45 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%44, %43 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_6 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
        %46 = tosa.reshape %45 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
        
        %47 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
        %48 = tosa.transpose %arg2, %47 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
        %49 = tosa.reshape %41 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
        %cst_7 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
        %50 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%49, %48 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_7 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
        %51 = tosa.reshape %50 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
        
        %52 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
        %53 = tosa.transpose %arg3, %52 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
        %54 = tosa.reshape %41 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
        %cst_8 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
        %55 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%54, %53 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_8 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
        %56 = tosa.reshape %55 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>

        %t1_original = call @rtclock() : () -> f64 

        %tensor_unranked_46 = tensor.cast %46 : tensor<1x40x4096xf32> to tensor<*xf32>
        %tensor_unranked_51 = tensor.cast %51 : tensor<1x40x4096xf32> to tensor<*xf32>
        %tensor_unranked_56 = tensor.cast %56 : tensor<1x40x4096xf32> to tensor<*xf32>

        // Print results.
        call @printMemrefF32(%tensor_unranked_46) : (tensor<*xf32>) -> ()
        call @printMemrefF32(%tensor_unranked_51) : (tensor<*xf32>) -> ()
        call @printMemrefF32(%tensor_unranked_56) : (tensor<*xf32>) -> ()

        %t_original = arith.subf %t1_original, %t0_original : f64
        vector.print str "original operation time: "
        vector.print %t_original : f64

        return 
    }

    func.func @qkv_compute_optimized(%arg1: tensor<4096x4096xf32>, %arg2: tensor<4096x4096xf32>, %arg3: tensor<4096x4096xf32>) {
        %41 = arith.constant dense<1.0> : tensor<1x40x4096xf32>

        %t0_optimized = call @rtclock() : () -> f64 

        %42 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
        %43 = tosa.transpose %arg1, %42 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
        %47 = tosa.transpose %arg2, %42 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
        %52 = tosa.transpose %arg3, %42 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>

        %concat_weights = "tosa.concat"(%43, %47, %52) {axis = 1 : i32} : (tensor<4096x4096xf32>, tensor<4096x4096xf32>, tensor<4096x4096xf32>) -> tensor<4096x12288xf32>

        %44 = tosa.reshape %41 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
        %cst_6 = arith.constant dense<0.000000e+00> : tensor<40x12288xf32>
        %45 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%44, %concat_weights : tensor<40x4096xf32>, tensor<4096x12288xf32>) outs(%cst_6 : tensor<40x12288xf32>) -> tensor<40x12288xf32>
        

        %t1_optimized = call @rtclock() : () -> f64 
        // %extracted_slice_15 = tensor.extract_slice %59[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
        %res_w1 = tensor.extract_slice %45[0, 0] [40, 4096] [1, 1] : tensor<40x12288xf32> to tensor<40x4096xf32>
        %res_w2 = tensor.extract_slice %45[0, 4096] [40, 4096] [1, 1] : tensor<40x12288xf32> to tensor<40x4096xf32>
        %res_w3 = tensor.extract_slice %45[0, 8192] [40, 4096] [1, 1] : tensor<40x12288xf32> to tensor<40x4096xf32>

        %46 = tosa.reshape %res_w1 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
        %51 = tosa.reshape %res_w2 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
        %56 = tosa.reshape %res_w3 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>

        // %result_W1 = "tosa.slice"(%45) {offsets = array<i64: 0, 0>, sizes = array<i64: 40, 4096>} : (tensor<40x12288xf32>) -> tensor<40x4096xf32>
        // %result_W2 = "tosa.slice"(%45) {offsets = array<i64: 0, 4096>, sizes = array<i64: 40, 4096>} : (tensor<40x12288xf32>) -> tensor<40x4096xf32>
        // %result_W3 = "tosa.slice"(%45) {offsets = array<i64: 0, 8192>, sizes = array<i64: 40, 4096>} : (tensor<40x12288xf32>) -> tensor<40x4096xf32>

        // %46 = tosa.reshape %result_W1 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
        // %51 = tosa.reshape %result_W2 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
        // %56 = tosa.reshape %result_W3 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>

        %tensor_unranked_46 = tensor.cast %46 : tensor<1x40x4096xf32> to tensor<*xf32>
        %tensor_unranked_51 = tensor.cast %51 : tensor<1x40x4096xf32> to tensor<*xf32>
        %tensor_unranked_56 = tensor.cast %56 : tensor<1x40x4096xf32> to tensor<*xf32>

        // Print results.
        call @printMemrefF32(%tensor_unranked_46) : (tensor<*xf32>) -> ()
        call @printMemrefF32(%tensor_unranked_51) : (tensor<*xf32>) -> ()
        call @printMemrefF32(%tensor_unranked_56) : (tensor<*xf32>) -> ()

        %t_optimized = arith.subf %t1_optimized, %t0_optimized : f64
        vector.print str "optimized operation time: "
        vector.print %t_optimized : f64

        return
    }


    func.func @main() {
        %arg1 = arith.constant dense<0.0> : tensor<4096x4096xf32>
        %arg2 = arith.constant dense<1.0> : tensor<4096x4096xf32>
        %arg3 = arith.constant dense<2.0> : tensor<4096x4096xf32>

        call @qkv_compute_original(%arg1, %arg2, %arg3) : (tensor<4096x4096xf32>, tensor<4096x4096xf32>, tensor<4096x4096xf32>) -> ()

        call @qkv_compute_optimized(%arg1, %arg2, %arg3) : (tensor<4096x4096xf32>, tensor<4096x4096xf32>, tensor<4096x4096xf32>) -> ()

        return
    }
}
