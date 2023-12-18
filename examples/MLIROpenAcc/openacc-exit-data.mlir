func.func @testexitdataop(%a: memref<f32>, %ifCond: i1) -> () {
  %0 = acc.getdeviceptr varPtr(%a : memref<f32>) -> memref<f32>
  acc.exit_data if(%ifCond) dataOperands(%0 : memref<f32>)
  acc.delete accPtr(%0 : memref<f32>)
  return
}