// LinalgBench: integer/boolean named elementwise operations.
module {
  func.func @named_integer_select(%a: tensor<1024xi32>, %b: tensor<1024xi32>,
                                  %cond: tensor<1024xi1>,
                                  %init: tensor<1024xi32>) -> tensor<1024xi32> {
    %0 = linalg.div_unsigned ins(%a, %b : tensor<1024xi32>, tensor<1024xi32>) outs(%init : tensor<1024xi32>) -> tensor<1024xi32>
    %1 = linalg.select ins(%cond, %0, %a : tensor<1024xi1>, tensor<1024xi32>, tensor<1024xi32>) outs(%init : tensor<1024xi32>) -> tensor<1024xi32>
    return %1 : tensor<1024xi32>
  }
}
