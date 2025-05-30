// RUN:mlir-opt %s | \
// mlir-opt -convert-openacc-to-scf -convert-scf-to-cf -convert-func-to-llvm -convert-cf-to-llvm

// CHECK:      func @testserialop([[ARGA:%.*]]: memref<10xf32>, [[ARGB:%.*]]: memref<10xf32>, [[ARGC:%.*]]: memref<10x10xf32>) {
// CHECK:      [[I64VALUE:%.*]] = arith.constant 1 : i64
// CHECK:      [[I32VALUE:%.*]] = arith.constant 1 : i32
// CHECK:      [[IDXVALUE:%.*]] = arith.constant 1 : index
// CHECK:      acc.serial async([[I64VALUE]] : i64) {
// CHECK-NEXT: }
// CHECK:      acc.serial async([[I32VALUE]] : i32) {
// CHECK-NEXT: }
// CHECK:      acc.serial async([[IDXVALUE]] : index) {
// CHECK-NEXT: }
// CHECK:      acc.serial wait([[I64VALUE]] : i64) {
// CHECK-NEXT: }
// CHECK:      acc.serial wait([[I32VALUE]] : i32) {
// CHECK-NEXT: }
// CHECK:      acc.serial wait([[IDXVALUE]] : index) {
// CHECK-NEXT: }
// CHECK:      acc.serial wait([[I64VALUE]], [[I32VALUE]], [[IDXVALUE]] : i64, i32, index) {
// CHECK-NEXT: }
// CHECK:      %[[FIRSTP:.*]] = acc.firstprivate varPtr([[ARGB]] : memref<10xf32>) -> memref<10xf32>
// CHECK:      acc.serial firstprivate(@firstprivatization_memref_10xf32 -> %[[FIRSTP]] : memref<10xf32>) private(@privatization_memref_10_f32 -> [[ARGA]] : memref<10xf32>, @privatization_memref_10_10_f32 -> [[ARGC]] : memref<10x10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.serial {
// CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>}
// CHECK:      acc.serial {
// CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue present>}
// CHECK:      acc.serial {
// CHECK-NEXT: } attributes {asyncAttr}
// CHECK:      acc.serial {
// CHECK-NEXT: } attributes {waitAttr}
// CHECK:      acc.serial {
// CHECK-NEXT: } attributes {selfAttr}
// CHECK:      acc.serial {
// CHECK:        acc.yield
// CHECK-NEXT: } attributes {selfAttr}

// -----

// acc.serial: This operation creates a serial region that encloses a region of code that operates on the device. 
// The operation can have zero or more serial clauses that specify the serial behavior for the region. 
// The operation also has an implicit attribute that indicates whether the serial region is implicit or explicit. 
// An implicit serial region is created by the compiler when there is no explicit serial region in the code. 
// An explicit serial region is created by the programmer using the acc serial directive.
// 

func.func @testserialop(%a: memref<10xf32>, %b: memref<10xf32>, %c: memref<10x10xf32>) -> () {
  %i64value = arith.constant 1 : i64
  %i32value = arith.constant 1 : i32
  %idxValue = arith.constant 1 : index
  acc.serial async(%i64value: i64) {
  }
  acc.serial async(%i32value: i32) {
  }
  acc.serial async(%idxValue: index) {
  }
  acc.serial wait(%i64value: i64) {
  }
  acc.serial wait(%i32value: i32) {
  }
  acc.serial wait(%idxValue: index) {
  }
  acc.serial wait(%i64value, %i32value, %idxValue : i64, i32, index) {
  }
  %firstprivate = acc.firstprivate varPtr(%b : memref<10xf32>) -> memref<10xf32>
  
  acc.serial {
  } attributes {defaultAttr = #acc<defaultvalue none>}
  acc.serial {
  } attributes {defaultAttr = #acc<defaultvalue present>}
  acc.serial {
  } attributes {asyncAttr}
  acc.serial {
  } attributes {waitAttr}
  acc.serial {
  } attributes {selfAttr}
  acc.serial {
    acc.yield
  } attributes {selfAttr}
  return
}
