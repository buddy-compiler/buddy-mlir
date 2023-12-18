func.func @testhostdataop(%a: memref<f32>, %ifCond: i1) -> () {
  %0 = acc.use_device varPtr(%a : memref<f32>) -> memref<f32>
  %true = arith.constant true
  acc.host_data dataOperands(%0 : memref<f32>) if(%true) {
  }
  return
}