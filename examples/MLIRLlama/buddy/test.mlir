#map = affine_map<(d0, d1, d2) -> (d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module{
  func.func @forward() -> tensor<13x13xi1> {
    %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]> : tensor<13xi64>
    %0 = "tosa.reshape"(%cst) {new_shape = array<i64: 1, 13>} : (tensor<13xi64>) -> tensor<1x13xi64>
    %1 = "tosa.reshape"(%0) {new_shape = array<i64: 1, 13>} : (tensor<1x13xi64>) -> tensor<1x13xi64>
    %cst_0 = arith.constant dense<true> : tensor<1x13xi1>
    %cst_1 = arith.constant dense<-3.40282347E+38> : tensor<13x13xf32>
    %cst_2 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]> : tensor<13xi64>
    %cst_3 = arith.constant dense<1> : tensor<i64>
    %3 = "tosa.reshape"(%cst_2) {new_shape = array<i64: 13, 1>} : (tensor<13xi64>) -> tensor<13x1xi64>
    %4 = tensor.empty() : tensor<13x13xi1>
    %5 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%cst_2, %3 : tensor<13xi64>, tensor<13x1xi64>) outs(%4 : tensor<13x13xi1>) {
    ^bb0(%in: i64, %in_69: i64, %out: i1):
      %6 = arith.cmpi sle, %in, %in_69 : i64
      linalg.yield %6 : i1
    } -> tensor<13x13xi1>
    return %5 : tensor<13x13xi1>
  }
}
