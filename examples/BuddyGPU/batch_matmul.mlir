!unit = f32
!lhs = tensor<4x768x1024x!unit>
!rhs = tensor<4x1024x768x!unit>
!res = tensor<4x768x768x!unit>

func.func @batch_matmul(
        %arg0: !lhs, %arg1: !rhs)
            -> !res {
       %cst = arith.constant 0.000000e+00 : !unit
        %0 = tensor.empty() : !res
        %1 = linalg.fill ins(%cst : !unit) outs(%0 : !res) -> !res
        %2 = linalg.batch_matmul 
                            ins(%arg0, %arg1: !lhs, !rhs)
                            outs(%1: !res)
            -> !res
        func.return %2 : !res
    }