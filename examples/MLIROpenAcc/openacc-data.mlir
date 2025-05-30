// RUN:mlir-opt %s | \
// mlir-opt -convert-openacc-to-scf -convert-scf-to-cf -convert-func-to-llvm -convert-cf-to-llvm

// CHECK:      func @testdataop(%[[ARGA:.*]]: memref<f32>, %[[ARGB:.*]]: memref<f32>, %[[ARGC:.*]]: memref<f32>) {

// CHECK:      %[[IFCOND1:.*]] = arith.constant true
// CHECK:      %[[PRESENT_A:.*]] = acc.present varPtr(%[[ARGA]] : memref<f32>) -> memref<f32>
// CHECK:      acc.data if(%[[IFCOND1]]) dataOperands(%[[PRESENT_A]] : memref<f32>) {
// CHECK-NEXT: }

// CHECK:      %[[PRESENT_A:.*]] = acc.present varPtr(%[[ARGA]] : memref<f32>) -> memref<f32>
// CHECK:      acc.data if(%[[IFCOND1]]) dataOperands(%[[PRESENT_A]] : memref<f32>) {
// CHECK-NEXT: }

// CHECK:      %[[PRESENT_A:.*]] = acc.present varPtr(%[[ARGA]] : memref<f32>) -> memref<f32>
// CHECK:      %[[PRESENT_B:.*]] = acc.present varPtr(%[[ARGB]] : memref<f32>) -> memref<f32>
// CHECK:      %[[PRESENT_C:.*]] = acc.present varPtr(%[[ARGC]] : memref<f32>) -> memref<f32>
// CHECK:      acc.data dataOperands(%[[PRESENT_A]], %[[PRESENT_B]], %[[PRESENT_C]] : memref<f32>, memref<f32>, memref<f32>) {
// CHECK-NEXT: }

// CHECK:      %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[ARGA]] : memref<f32>) -> memref<f32>
// CHECK:      %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[ARGB]] : memref<f32>) -> memref<f32>
// CHECK:      %[[COPYIN_C:.*]] = acc.copyin varPtr(%[[ARGC]] : memref<f32>) -> memref<f32>
// CHECK:      acc.data dataOperands(%[[COPYIN_A]], %[[COPYIN_B]], %[[COPYIN_C]] : memref<f32>, memref<f32>, memref<f32>) {
// CHECK-NEXT: }

// CHECK:      %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[ARGA]] : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyin_readonly>}
// CHECK:      %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[ARGB]] : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyin_readonly>}
// CHECK:      %[[COPYIN_C:.*]] = acc.copyin varPtr(%[[ARGC]] : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyin_readonly>}
// CHECK:      acc.data dataOperands(%[[COPYIN_A]], %[[COPYIN_B]], %[[COPYIN_C]] : memref<f32>, memref<f32>, memref<f32>) {
// CHECK-NEXT: }

// CHECK:      %[[CREATE_A:.*]] = acc.create varPtr(%[[ARGA]] : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyout>}
// CHECK:      %[[CREATE_B:.*]] = acc.create varPtr(%[[ARGB]] : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyout>}
// CHECK:      %[[CREATE_C:.*]] = acc.create varPtr(%[[ARGC]] : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyout>}
// CHECK:      acc.data dataOperands(%[[CREATE_A]], %[[CREATE_B]], %[[CREATE_C]] : memref<f32>, memref<f32>, memref<f32>) {
// CHECK-NEXT: }
// CHECK:      acc.copyout accPtr(%[[CREATE_A]] : memref<f32>) to varPtr(%[[ARGA]] : memref<f32>)
// CHECK:      acc.copyout accPtr(%[[CREATE_B]] : memref<f32>) to varPtr(%[[ARGB]] : memref<f32>)
// CHECK:      acc.copyout accPtr(%[[CREATE_C]] : memref<f32>) to varPtr(%[[ARGC]] : memref<f32>)

// CHECK:      %[[CREATE_A:.*]] = acc.create varPtr(%[[ARGA]] : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyout_zero>}
// CHECK:      %[[CREATE_B:.*]] = acc.create varPtr(%[[ARGB]] : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyout_zero>}
// CHECK:      %[[CREATE_C:.*]] = acc.create varPtr(%[[ARGC]] : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyout_zero>}
// CHECK:      acc.data dataOperands(%[[CREATE_A]], %[[CREATE_B]], %[[CREATE_C]] : memref<f32>, memref<f32>, memref<f32>) {
// CHECK-NEXT: }
// CHECK:      acc.copyout accPtr(%[[CREATE_A]] : memref<f32>) to varPtr(%[[ARGA]] : memref<f32>) {dataClause = #acc<data_clause acc_copyout_zero>}
// CHECK:      acc.copyout accPtr(%[[CREATE_B]] : memref<f32>) to varPtr(%[[ARGB]] : memref<f32>) {dataClause = #acc<data_clause acc_copyout_zero>}
// CHECK:      acc.copyout accPtr(%[[CREATE_C]] : memref<f32>) to varPtr(%[[ARGC]] : memref<f32>) {dataClause = #acc<data_clause acc_copyout_zero>}

// CHECK:      %[[CREATE_A:.*]] = acc.create varPtr(%[[ARGA]] : memref<f32>) -> memref<f32>
// CHECK:      %[[CREATE_B:.*]] = acc.create varPtr(%[[ARGB]] : memref<f32>) -> memref<f32>
// CHECK:      %[[CREATE_C:.*]] = acc.create varPtr(%[[ARGC]] : memref<f32>) -> memref<f32>
// CHECK:      acc.data dataOperands(%[[CREATE_A]], %[[CREATE_B]], %[[CREATE_C]] : memref<f32>, memref<f32>, memref<f32>) {
// CHECK-NEXT: }

// CHECK:      %[[CREATE_A:.*]] = acc.create varPtr(%[[ARGA]] : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_create_zero>}
// CHECK:      %[[CREATE_B:.*]] = acc.create varPtr(%[[ARGB]] : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_create_zero>}
// CHECK:      %[[CREATE_C:.*]] = acc.create varPtr(%[[ARGC]] : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_create_zero>}
// CHECK:      acc.data dataOperands(%[[CREATE_A]], %[[CREATE_B]], %[[CREATE_C]] : memref<f32>, memref<f32>, memref<f32>) {
// CHECK-NEXT: }

// CHECK:      %[[NOCREATE_A:.*]] = acc.nocreate varPtr(%[[ARGA]] : memref<f32>) -> memref<f32>
// CHECK:      %[[NOCREATE_B:.*]] = acc.nocreate varPtr(%[[ARGB]] : memref<f32>) -> memref<f32>
// CHECK:      %[[NOCREATE_C:.*]] = acc.nocreate varPtr(%[[ARGC]] : memref<f32>) -> memref<f32>
// CHECK:      acc.data dataOperands(%[[NOCREATE_A]], %[[NOCREATE_B]], %[[NOCREATE_C]] : memref<f32>, memref<f32>, memref<f32>) {
// CHECK-NEXT: }

// CHECK:      %[[DEVICEPTR_A:.*]] = acc.deviceptr varPtr(%[[ARGA]] : memref<f32>) -> memref<f32>
// CHECK:      %[[DEVICEPTR_B:.*]] = acc.deviceptr varPtr(%[[ARGB]] : memref<f32>) -> memref<f32>
// CHECK:      %[[DEVICEPTR_C:.*]] = acc.deviceptr varPtr(%[[ARGC]] : memref<f32>) -> memref<f32>
// CHECK:      acc.data dataOperands(%[[DEVICEPTR_A]], %[[DEVICEPTR_B]], %[[DEVICEPTR_C]] : memref<f32>, memref<f32>, memref<f32>) {
// CHECK-NEXT: }

// CHECK:      %[[ATTACH_A:.*]] = acc.attach varPtr(%[[ARGA]] : memref<f32>) -> memref<f32>
// CHECK:      %[[ATTACH_B:.*]] = acc.attach varPtr(%[[ARGB]] : memref<f32>) -> memref<f32>
// CHECK:      %[[ATTACH_C:.*]] = acc.attach varPtr(%[[ARGC]] : memref<f32>) -> memref<f32>
// CHECK:      acc.data dataOperands(%[[ATTACH_A]], %[[ATTACH_B]], %[[ATTACH_C]] : memref<f32>, memref<f32>, memref<f32>) {
// CHECK-NEXT: }


// CHECK:      %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[ARGA]] : memref<f32>) -> memref<f32>
// CHECK:      %[[CREATE_B:.*]] = acc.create varPtr(%[[ARGB]] : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyout>}
// CHECK:      %[[PRESENT_C:.*]] = acc.present varPtr(%[[ARGC]] : memref<f32>) -> memref<f32>
// CHECK:      acc.data dataOperands(%[[COPYIN_A]], %[[CREATE_B]], %[[PRESENT_C]] : memref<f32>, memref<f32>, memref<f32>) {
// CHECK-NEXT: }
// CHECK:      acc.copyout accPtr(%[[CREATE_B]] : memref<f32>) to varPtr(%[[ARGB]] : memref<f32>)

// CHECK:      %[[PRESENT_A:.*]] = acc.present varPtr(%[[ARGA]] : memref<f32>) -> memref<f32>
// CHECK:      acc.data dataOperands(%[[PRESENT_A]] : memref<f32>) {
// CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>}

// CHECK:      %[[PRESENT_A:.*]] = acc.present varPtr(%[[ARGA]] : memref<f32>) -> memref<f32>
// CHECK:      acc.data dataOperands(%[[PRESENT_A]] : memref<f32>) {
// CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue present>}

// CHECK:      acc.data {
// CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>}

// CHECK:      acc.data {
// CHECK-NEXT: } attributes {async, defaultAttr = #acc<defaultvalue none>}

// CHECK:      acc.data async(%{{.*}} : i64) {
// CHECK-NEXT: } attributes {async, defaultAttr = #acc<defaultvalue none>}

// CHECK:      acc.data {
// CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>, wait}

// CHECK:      acc.data wait(%{{.*}} : i64) {
// CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>, wait}

// CHECK:      acc.data wait_devnum(%{{.*}} : i64) wait(%{{.*}} : i64) {
// CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>, wait}

// -----


// acc.data: This operation creates a data region that encloses a region of code that operates on the device. 
// The operation can have zero or more data clauses that specify the data mapping and movement for the region. 
// The operation also has an implicit attribute that indicates whether the data region is implicit or explicit. 
// An implicit data region is created by the compiler when there is no explicit data region in the code. 
// An explicit data region is created by the programmer using the acc data directive.
// 

func.func @testdataop(%a: memref<f32>, %b: memref<f32>, %c: memref<f32>) -> () {
  %ifCond = arith.constant true

  %0 = acc.present varPtr(%a : memref<f32>) -> memref<f32>
  acc.data if(%ifCond) dataOperands(%0 : memref<f32>) {
  }

  %1 = acc.present varPtr(%a : memref<f32>) -> memref<f32>
  acc.data dataOperands(%1 : memref<f32>) if(%ifCond) {
  }

  %2 = acc.present varPtr(%a : memref<f32>) -> memref<f32>
  %3 = acc.present varPtr(%b : memref<f32>) -> memref<f32>
  %4 = acc.present varPtr(%c : memref<f32>) -> memref<f32>
  acc.data dataOperands(%2, %3, %4 : memref<f32>, memref<f32>, memref<f32>) {
  }

  %5 = acc.copyin varPtr(%a : memref<f32>) -> memref<f32>
  %6 = acc.copyin varPtr(%b : memref<f32>) -> memref<f32>
  %7 = acc.copyin varPtr(%c : memref<f32>) -> memref<f32>
  acc.data dataOperands(%5, %6, %7 : memref<f32>, memref<f32>, memref<f32>) {
  }

  %8 = acc.copyin varPtr(%a : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyin_readonly>}
  %9 = acc.copyin varPtr(%b : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyin_readonly>}
  %10 = acc.copyin varPtr(%c : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyin_readonly>}
  acc.data dataOperands(%8, %9, %10 : memref<f32>, memref<f32>, memref<f32>) {
  }

  %11 = acc.create varPtr(%a : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyout>}
  %12 = acc.create varPtr(%b : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyout>}
  %13 = acc.create varPtr(%c : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyout>}
  acc.data dataOperands(%11, %12, %13 : memref<f32>, memref<f32>, memref<f32>) {
  }
  acc.copyout accPtr(%11 : memref<f32>) to varPtr(%a : memref<f32>)
  acc.copyout accPtr(%12 : memref<f32>) to varPtr(%b : memref<f32>)
  acc.copyout accPtr(%13 : memref<f32>) to varPtr(%c : memref<f32>)

  %14 = acc.create varPtr(%a : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyout_zero>}
  %15 = acc.create varPtr(%b : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyout_zero>}
  %16 = acc.create varPtr(%c : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyout_zero>}
  acc.data dataOperands(%14, %15, %16 : memref<f32>, memref<f32>, memref<f32>) {
  }
  acc.copyout accPtr(%14 : memref<f32>) to varPtr(%a : memref<f32>) {dataClause = #acc<data_clause acc_copyout_zero>}
  acc.copyout accPtr(%15 : memref<f32>) to varPtr(%b : memref<f32>) {dataClause = #acc<data_clause acc_copyout_zero>}
  acc.copyout accPtr(%16 : memref<f32>) to varPtr(%c : memref<f32>) {dataClause = #acc<data_clause acc_copyout_zero>}

  %17 = acc.create varPtr(%a : memref<f32>) -> memref<f32>
  %18 = acc.create varPtr(%b : memref<f32>) -> memref<f32>
  %19 = acc.create varPtr(%c : memref<f32>) -> memref<f32>
  acc.data dataOperands(%17, %18, %19 : memref<f32>, memref<f32>, memref<f32>) {
  }
  acc.delete accPtr(%17 : memref<f32>) {dataClause = #acc<data_clause acc_create>}
  acc.delete accPtr(%18 : memref<f32>) {dataClause = #acc<data_clause acc_create>}
  acc.delete accPtr(%19 : memref<f32>) {dataClause = #acc<data_clause acc_create>}
  
  %20 = acc.create varPtr(%a : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_create_zero>}
  %21 = acc.create varPtr(%b : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_create_zero>}
  %22 = acc.create varPtr(%c : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_create_zero>}
  acc.data dataOperands(%20, %21, %22 : memref<f32>, memref<f32>, memref<f32>) {
  }
  acc.delete accPtr(%20 : memref<f32>) {dataClause = #acc<data_clause acc_create_zero>}
  acc.delete accPtr(%21 : memref<f32>) {dataClause = #acc<data_clause acc_create_zero>}
  acc.delete accPtr(%22 : memref<f32>) {dataClause = #acc<data_clause acc_create_zero>}

  %23 = acc.nocreate varPtr(%a : memref<f32>) -> memref<f32>
  %24 = acc.nocreate varPtr(%b : memref<f32>) -> memref<f32>
  %25 = acc.nocreate varPtr(%c : memref<f32>) -> memref<f32>
  acc.data dataOperands(%23, %24, %25 : memref<f32>, memref<f32>, memref<f32>) {
  }

  %26 = acc.deviceptr varPtr(%a : memref<f32>) -> memref<f32>
  %27 = acc.deviceptr varPtr(%b : memref<f32>) -> memref<f32>
  %28 = acc.deviceptr varPtr(%c : memref<f32>) -> memref<f32>
  acc.data dataOperands(%26, %27, %28 : memref<f32>, memref<f32>, memref<f32>) {
  }

  %29 = acc.attach varPtr(%a : memref<f32>) -> memref<f32>
  %30 = acc.attach varPtr(%b : memref<f32>) -> memref<f32>
  %31 = acc.attach varPtr(%c : memref<f32>) -> memref<f32>
  acc.data dataOperands(%29, %30, %31 : memref<f32>, memref<f32>, memref<f32>) {
  }

  %32 = acc.copyin varPtr(%a : memref<f32>) -> memref<f32>
  %33 = acc.create varPtr(%b : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyout>}
  %34 = acc.present varPtr(%c : memref<f32>) -> memref<f32>
  acc.data dataOperands(%32, %33, %34 : memref<f32>, memref<f32>, memref<f32>) {
  }
  acc.copyout accPtr(%33 : memref<f32>) to varPtr(%b : memref<f32>)

  %35 = acc.present varPtr(%a : memref<f32>) -> memref<f32>
  acc.data dataOperands(%35 : memref<f32>) {
  } attributes { defaultAttr = #acc<defaultvalue none> }
  

  %36 = acc.present varPtr(%a : memref<f32>) -> memref<f32>
  acc.data dataOperands(%36 : memref<f32>) {
  } attributes { defaultAttr = #acc<defaultvalue present> }

  acc.data {
  } attributes { defaultAttr = #acc<defaultvalue none> }

  acc.data {
  } attributes { defaultAttr = #acc<defaultvalue none>, async }

  %a1 = arith.constant 1 : i64
  acc.data async(%a1 : i64) {
  } attributes { defaultAttr = #acc<defaultvalue none>, async }

  acc.data {
  } attributes { defaultAttr = #acc<defaultvalue none>, wait }

  %w1 = arith.constant 1 : i64
  acc.data wait(%w1 : i64) {
  } attributes { defaultAttr = #acc<defaultvalue none>, wait }

  %wd1 = arith.constant 1 : i64
  acc.data wait_devnum(%wd1 : i64) wait(%w1 : i64) {
  } attributes { defaultAttr = #acc<defaultvalue none>, wait }

  return
}
