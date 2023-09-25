module {
  func.func @main() -> vector<1x8x1x1xf32> {
    %0:2 = sche.on_device() {targetConfig = "", targetId = "gpu1"} () -> (vector<1x8x1x1xf32>, vector<1x8x1x1xf32>) {
      %cst = arith.constant dense<1.000000e-01> : vector<1x8x1x1xf32>
      %cst_0 = arith.constant dense<0.000000e+00> : vector<1x8x1x1xf32>
      sche.return %cst, %cst_0 : vector<1x8x1x1xf32>, vector<1x8x1x1xf32>
    }
    %1 = sche.on_device(%0#0, %0#1) {targetConfig = "", targetId = "gpu2"} (vector<1x8x1x1xf32>, vector<1x8x1x1xf32>) -> vector<1x8x1x1xf32> {
      %2 = arith.addf %0#0, %0#1 : vector<1x8x1x1xf32>
      sche.return %2 : vector<1x8x1x1xf32>
    }
    return %1 : vector<1x8x1x1xf32>
  }
}

