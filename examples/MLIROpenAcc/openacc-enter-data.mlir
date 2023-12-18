func.func @testenterdataop(%a: memref<f32>, %ifCond: i1) -> () {
  %0 = acc.create varPtr(%a : memref<f32>) -> memref<f32>
  acc.enter_data if(%ifCond) dataOperands(%0 : memref<f32>)
  return
}