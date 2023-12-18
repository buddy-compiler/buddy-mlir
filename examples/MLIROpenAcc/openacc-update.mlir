func.func @testupdateop(%a: memref<f32>, %b: memref<f32>, %c: memref<f32>) -> () {
  %i64Value = arith.constant 1 : i64
  %i32Value = arith.constant 1 : i32
  %idxValue = arith.constant 1 : index
  %ifCond = arith.constant true
  %0 = acc.update_device varPtr(%a : memref<f32>) -> memref<f32>
  %1 = acc.update_device varPtr(%b : memref<f32>) -> memref<f32>
  %2 = acc.update_device varPtr(%c : memref<f32>) -> memref<f32>
  
  acc.update async(%i64Value: i64) dataOperands(%0: memref<f32>)
  acc.update async(%i32Value: i32) dataOperands(%0: memref<f32>)
  acc.update async(%i32Value: i32) dataOperands(%0: memref<f32>)
  acc.update async(%idxValue: index) dataOperands(%0: memref<f32>)
  acc.update wait_devnum(%i64Value: i64) wait(%i32Value, %idxValue : i32, index) dataOperands(%0: memref<f32>)
  acc.update if(%ifCond) dataOperands(%0: memref<f32>)
  acc.update dataOperands(%0: memref<f32>) attributes {acc.device_types = [#acc.device_type<nvidia>]}
  acc.update dataOperands(%0, %1, %2 : memref<f32>, memref<f32>, memref<f32>)
  acc.update dataOperands(%0, %1, %2 : memref<f32>, memref<f32>, memref<f32>) attributes {async}
  acc.update dataOperands(%0, %1, %2 : memref<f32>, memref<f32>, memref<f32>) attributes {wait}
  acc.update dataOperands(%0, %1, %2 : memref<f32>, memref<f32>, memref<f32>) attributes {ifPresent}
  return
}