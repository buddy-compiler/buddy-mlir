// RUN: mlir-opt %s | \
// RUN: mlir-opt -convert-openacc-to-scf -convert-scf-to-cf -convert-func-to-llvm -convert-cf-to-llvm

// CHECK: func @testupdateop([[ARGA:%.*]]: memref<f32>, [[ARGB:%.*]]: memref<f32>, [[ARGC:%.*]]: memref<f32>) {
// CHECK:   [[I64VALUE:%.*]] = arith.constant 1 : i64
// CHECK:   [[I32VALUE:%.*]] = arith.constant 1 : i32
// CHECK:   [[IDXVALUE:%.*]] = arith.constant 1 : index
// CHECK:   [[IFCOND:%.*]] = arith.constant true
// CHECK:   acc.update async([[I64VALUE]] : i64) dataOperands(%{{.*}} : memref<f32>)
// CHECK:   acc.update async([[I32VALUE]] : i32) dataOperands(%{{.*}} : memref<f32>)
// CHECK:   acc.update async([[I32VALUE]] : i32) dataOperands(%{{.*}} : memref<f32>)
// CHECK:   acc.update async([[IDXVALUE]] : index) dataOperands(%{{.*}} : memref<f32>)
// CHECK:   acc.update wait_devnum([[I64VALUE]] : i64) wait([[I32VALUE]], [[IDXVALUE]] : i32, index) dataOperands(%{{.*}} : memref<f32>)
// CHECK:   acc.update if([[IFCOND]]) dataOperands(%{{.*}} : memref<f32>)
// CHECK:   acc.update dataOperands(%{{.*}} : memref<f32>) attributes {acc.device_types = [#acc.device_type<nvidia>]}
// CHECK:   acc.update dataOperands(%{{.*}}, %{{.*}}, %{{.*}} : memref<f32>, memref<f32>, memref<f32>)
// CHECK:   acc.update dataOperands(%{{.*}}, %{{.*}}, %{{.*}} : memref<f32>, memref<f32>, memref<f32>) attributes {async}
// CHECK:   acc.update dataOperands(%{{.*}}, %{{.*}}, %{{.*}} : memref<f32>, memref<f32>, memref<f32>) attributes {wait}
// CHECK:   acc.update dataOperands(%{{.*}}, %{{.*}}, %{{.*}} : memref<f32>, memref<f32>, memref<f32>) attributes {ifPresent}

// -----

// acc.update: This operation performs data movement between the host and the device. 
// The operation can have one or more data clauses that specify the data movement direction 
// and the data variables. The operation does not enclose any code region, and can be used 
// inside or outside of any other data region.
// 
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