module {
    func.func private @printMemrefF32(tensor<*xf32>)
    func.func private @rtclock() -> f64

    func.func @uvue_original() {
        %t0_original = call @rtclock() : () -> f64 

        %84 = arith.constant dense<2.0> : tensor<1x32x40x128xf32>
        %92 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
        %93 = tosa.add %84, %92 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
        %94 = tosa.reshape %93 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
        
        %t1_original = call @rtclock() : () -> f64
        %tensor_unranked = tensor.cast %94 : tensor<32x40x128xf32> to tensor<*xf32>

        // Print results.
        call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()

        
        %t_original = arith.subf %t1_original, %t0_original : f64
        vector.print str "original operation time: "
        vector.print %t_original : f64
        return 
    }

    func.func @uve_optimized() {
        %t0_optimized = call @rtclock() : () -> f64

        %84 = arith.constant dense<2.0> : tensor<1x32x40x128xf32>
        %94 = tosa.reshape %84 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
        %t1_optimized = call @rtclock() : () -> f64

        %tensor_unranked = tensor.cast %94 : tensor<32x40x128xf32> to tensor<*xf32>

        // Print results.
        call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
        
        %t_optimized = arith.subf %t1_optimized, %t0_optimized : f64
        vector.print str "optimized operation time: "
        vector.print %t_optimized : f64
        return 
    }


    func.func @main() {
        %84 = arith.constant dense<2.0> : tensor<1x32x40x128xf32>

        call @uvue_original() : () -> ()
        
        call @uve_optimized() : () -> ()

        return 
    }
}
