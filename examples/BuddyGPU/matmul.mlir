!unit = f32
!lhs = tensor<5376x2048x!unit>
!rhs = tensor<2048x5376x!unit>
!res = tensor<5376x5376x!unit>

func.func @matmul(%arg0: !lhs, %arg1: !rhs) -> !res {
  %cst = arith.constant 0.000000e+00 : !unit
  %0 = tensor.empty() : !res
  %1 = linalg.fill ins(%cst : !unit) outs(%0 : !res) -> !res
  %2 = linalg.matmul ins(%arg0, %arg1: !lhs, !rhs) outs(%1: !res) -> !res
  func.return %2 : !res
}
