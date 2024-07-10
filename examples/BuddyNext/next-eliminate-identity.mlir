module {
    func.func private @printMemrefF32(tensor<*xf32>)
    func.func private @rtclock() -> f64

    func.func @ie_original() {
        %t0_original = call @rtclock() : () -> f64 

        %119 = arith.constant dense<1.0> : tensor<1x40x32x128xf32>
        %120 = tosa.identity %119 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
        %121 = tosa.reshape %120 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
        %t1_original = call @rtclock() : () -> f64
        
        %tensor_unranked = tensor.cast %121 : tensor<1x40x4096xf32> to tensor<*xf32>

        // Print results.
        call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
        
        %t_original = arith.subf %t1_original, %t0_original : f64
        vector.print str "original operation time: "
        vector.print %t_original : f64
        return 
    }

    func.func @ie_optimized() {
        %t0_optimized = call @rtclock() : () -> f64

        %119 = arith.constant dense<1.0> : tensor<1x40x32x128xf32>
        %121 = tosa.reshape %119 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
        %t1_optimized = call @rtclock() : () -> f64

        %tensor_unranked = tensor.cast %121 : tensor<1x40x4096xf32> to tensor<*xf32>

        // Print results.
        call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
        %t_optimized = arith.subf %t1_optimized, %t0_optimized : f64
        vector.print str "optimized operation time: "
        vector.print %t_optimized : f64
        return 
    }


    func.func @main() {

        call @ie_original() : () -> ()
        call @ie_optimized() : () -> ()

        return 
    }
}
