func.func @matmul(
        %arg0: tensor<5376x2048xf32>, %arg1: tensor<2048x5376xf32>,
        %arg2: tensor<5376x5376xf32>)
            -> tensor<5376x5376xf32> {
        %0 = linalg.matmul 
                            ins(%arg0, %arg1: tensor<5376x2048xf32>, tensor<2048x5376xf32>)
                            outs(%arg2: tensor<5376x5376xf32>)
            -> tensor<5376x5376xf32>
        func.return %0 : tensor<5376x5376xf32>
}
