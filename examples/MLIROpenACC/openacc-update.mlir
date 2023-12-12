func.func @update(%a: memref<f32>, %ifCond: i1) -> () {
  %0 = acc.update_device varPtr(%a : memref<f32>) -> memref<f32>
  acc.update if(%ifCond) dataOperands(%0 : memref<f32>)
  return
}