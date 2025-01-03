// RUN:mlir-opt %s | \
// mlir-opt -convert-openacc-to-scf -convert-scf-to-cf -convert-func-to-llvm -convert-cf-to-llvm

// CHECK: func.func @teststructureddataclauseops([[ARGA:%.*]]: memref<10xf32>, [[ARGB:%.*]]: memref<memref<10xf32>>, [[ARGC:%.*]]: memref<10x20xf32>) {
// CHECK: [[DEVICEPTR:%.*]] = acc.deviceptr varPtr([[ARGA]] : memref<10xf32>) -> memref<10xf32> {name = "arrayA"}
// CHECK-NEXT: acc.parallel dataOperands([[DEVICEPTR]] : memref<10xf32>) {
// CHECK-NEXT: }
// CHECK: [[PRESENT:%.*]] = acc.present varPtr([[ARGA]] : memref<10xf32>) -> memref<10xf32>
// CHECK-NEXT: acc.data dataOperands([[PRESENT]] : memref<10xf32>) {
// CHECK-NEXT: }
// CHECK: [[COPYIN:%.*]] = acc.copyin varPtr([[ARGA]] : memref<10xf32>) -> memref<10xf32>
// CHECK-NEXT: acc.parallel dataOperands([[COPYIN]] : memref<10xf32>) {
// CHECK-NEXT: }
// CHECK: [[COPYINRO:%.*]] = acc.copyin varPtr([[ARGA]] : memref<10xf32>) -> memref<10xf32> {dataClause = #acc<data_clause acc_copyin_readonly>}
// CHECK-NEXT: acc.kernels dataOperands([[COPYINRO]] : memref<10xf32>) {
// CHECK-NEXT: }
// CHECK: [[COPYINCOPY:%.*]] = acc.copyin varPtr([[ARGA]] : memref<10xf32>) -> memref<10xf32> {dataClause = #acc<data_clause acc_copy>}
// CHECK-NEXT: acc.serial dataOperands([[COPYINCOPY]] : memref<10xf32>) {
// CHECK-NEXT: }
// CHECK-NEXT: acc.copyout accPtr([[COPYINCOPY]] : memref<10xf32>) to varPtr([[ARGA]] : memref<10xf32>) {dataClause = #acc<data_clause acc_copy>}
// CHECK: [[CREATE:%.*]] = acc.create varPtr([[ARGA]] : memref<10xf32>) -> memref<10xf32>
// CHECK-NEXT: [[CREATEIMP:%.*]] = acc.create varPtr([[ARGC]] : memref<10x20xf32>) -> memref<10x20xf32> {implicit = true}
// CHECK-NEXT: acc.parallel dataOperands([[CREATE]], [[CREATEIMP]] : memref<10xf32>, memref<10x20xf32>) {
// CHECK-NEXT: }
// CHECK-NEXT: acc.delete accPtr([[CREATE]] : memref<10xf32>) {dataClause = #acc<data_clause acc_create>}
// CHECK-NEXT: acc.delete accPtr([[CREATEIMP]] : memref<10x20xf32>) {dataClause = #acc<data_clause acc_create>, implicit = true}
// CHECK: [[COPYOUTZ:%.*]] = acc.create varPtr([[ARGA]] : memref<10xf32>) -> memref<10xf32> {dataClause = #acc<data_clause acc_copyout_zero>}
// CHECK-NEXT: acc.parallel dataOperands([[COPYOUTZ]] : memref<10xf32>) {
// CHECK-NEXT: }
// CHECK-NEXT: acc.copyout accPtr([[COPYOUTZ]] : memref<10xf32>) to varPtr([[ARGA]] : memref<10xf32>) {dataClause = #acc<data_clause acc_copyout_zero>}
// CHECK: [[ATTACH:%.*]] = acc.attach varPtr([[ARGB]] : memref<memref<10xf32>>) -> memref<memref<10xf32>>
// CHECK-NEXT: acc.parallel dataOperands([[ATTACH]] : memref<memref<10xf32>>) {
// CHECK-NEXT: }
// CHECK-NEXT: acc.detach accPtr([[ATTACH]] : memref<memref<10xf32>>) {dataClause = #acc<data_clause acc_attach>}
// CHECK: [[COPYINP:%.*]] = acc.copyin varPtr([[ARGA]] : memref<10xf32>) varPtrPtr([[ARGB]] : memref<memref<10xf32>>) -> memref<10xf32> {dataClause = #acc<data_clause acc_copy>}
// CHECK-NEXT: acc.parallel dataOperands([[COPYINP]] : memref<10xf32>) {
// CHECK-NEXT: }
// CHECK-NEXT: acc.copyout accPtr([[COPYINP]] : memref<10xf32>) to varPtr([[ARGA]] : memref<10xf32>) {dataClause = #acc<data_clause acc_copy>}
// CHECK-DAG: [[CON0:%.*]] = arith.constant 0 : index
// CHECK-DAG: [[CON1:%.*]] = arith.constant 1 : index
// CHECK-DAG: [[CON4:%.*]] = arith.constant 4 : index
// CHECK-DAG: [[CON9:%.*]] = arith.constant 9 : index
// CHECK-DAG: [[CON20:%.*]] = arith.constant 20 : index
// CHECK: [[BOUNDS1F:%.*]] = acc.bounds lowerbound([[CON0]] : index) upperbound([[CON9]] : index) stride([[CON1]] : index)
// CHECK-NEXT: [[COPYINF1:%.*]] = acc.copyin varPtr([[ARGA]] : memref<10xf32>) bounds([[BOUNDS1F]]) -> memref<10xf32> {name = "arrayA[0:9]"}
// CHECK-NEXT: [[BOUNDS2F:%.*]] = acc.bounds lowerbound([[CON1]] : index) upperbound([[CON20]] : index) extent([[CON20]] : index) stride([[CON4]] : index) startIdx([[CON1]] : index) {strideInBytes = true}
// CHECK-NEXT: [[COPYINF2:%.*]] = acc.copyin varPtr([[ARGC]] : memref<10x20xf32>) bounds([[BOUNDS1F]], [[BOUNDS2F]]) -> memref<10x20xf32>
// CHECK-NEXT: acc.parallel dataOperands([[COPYINF1]], [[COPYINF2]] : memref<10xf32>, memref<10x20xf32>) {
// CHECK-NEXT: }
// CHECK: [[BOUNDS1P:%.*]] = acc.bounds lowerbound([[CON4]] : index) upperbound([[CON9]] : index) stride([[CON1]] : index)
// CHECK-NEXT: [[COPYINPART:%.*]] = acc.copyin varPtr([[ARGA]] : memref<10xf32>) bounds([[BOUNDS1P]]) -> memref<10xf32> {dataClause = #acc<data_clause acc_copy>}
// CHECK-NEXT: acc.parallel dataOperands([[COPYINPART]] : memref<10xf32>) {
// CHECK-NEXT: }
// CHECK-NEXT: acc.copyout accPtr([[COPYINPART]] : memref<10xf32>) bounds([[BOUNDS1P]]) to varPtr([[ARGA]] : memref<10xf32>) {dataClause = #acc<data_clause acc_copy>}

// -----

// acc.bound: This operation specifies the bounds of the device gangs, workers, and vectors for the OpenACC regions. 
// The operation can have zero or more bound clauses that specify the bound behavior for the regions. 
// The operation also has an implicit attribute that indicates whether the bound specification is implicit or explicit. 
// An implicit bound specification is created by the compiler when there is no explicit bound specification in the code. 
// An explicit bound specification is created by the programmer using the acc bound directive.

func.func @teststructureddataclauseops(%a: memref<10xf32>, %b: memref<memref<10xf32>>, %c: memref<10x20xf32>) -> () {
  %deviceptr = acc.deviceptr varPtr(%a : memref<10xf32>) -> memref<10xf32> {name = "arrayA"}
  acc.parallel dataOperands(%deviceptr : memref<10xf32>) {
  }

  %present = acc.present varPtr(%a : memref<10xf32>) -> memref<10xf32>
  acc.data dataOperands(%present : memref<10xf32>) {
  }

  %copyin = acc.copyin varPtr(%a : memref<10xf32>) -> memref<10xf32>
  acc.parallel dataOperands(%copyin : memref<10xf32>) {
  }

  %copyinreadonly = acc.copyin varPtr(%a : memref<10xf32>) -> memref<10xf32> {dataClause = #acc<data_clause acc_copyin_readonly>}
  acc.kernels dataOperands(%copyinreadonly : memref<10xf32>) {
  }

  %copyinfromcopy = acc.copyin varPtr(%a : memref<10xf32>) -> memref<10xf32> {dataClause = #acc<data_clause acc_copy>}
  acc.serial dataOperands(%copyinfromcopy : memref<10xf32>) {
  }
  acc.copyout accPtr(%copyinfromcopy : memref<10xf32>) to varPtr(%a : memref<10xf32>) {dataClause = #acc<data_clause acc_copy>}

  %create = acc.create varPtr(%a : memref<10xf32>) -> memref<10xf32>
  %createimplicit = acc.create varPtr(%c : memref<10x20xf32>) -> memref<10x20xf32> {implicit = true}
  acc.parallel dataOperands(%create, %createimplicit : memref<10xf32>, memref<10x20xf32>) {
  }
  acc.delete accPtr(%create : memref<10xf32>) {dataClause = #acc<data_clause acc_create>}
  acc.delete accPtr(%createimplicit : memref<10x20xf32>) {dataClause = #acc<data_clause acc_create>, implicit = true}

  %copyoutzero = acc.create varPtr(%a : memref<10xf32>) -> memref<10xf32> {dataClause = #acc<data_clause acc_copyout_zero>}
  acc.parallel dataOperands(%copyoutzero: memref<10xf32>) {
  }
  acc.copyout accPtr(%copyoutzero : memref<10xf32>) to varPtr(%a : memref<10xf32>) {dataClause = #acc<data_clause acc_copyout_zero>}

  %attach = acc.attach varPtr(%b : memref<memref<10xf32>>) -> memref<memref<10xf32>>
  acc.parallel dataOperands(%attach : memref<memref<10xf32>>) {
  }
  acc.detach accPtr(%attach : memref<memref<10xf32>>) {dataClause = #acc<data_clause acc_attach>}

  %copyinparent = acc.copyin varPtr(%a : memref<10xf32>) varPtrPtr(%b : memref<memref<10xf32>>) -> memref<10xf32> {dataClause = #acc<data_clause acc_copy>}
  acc.parallel dataOperands(%copyinparent : memref<10xf32>) {
  }
  acc.copyout accPtr(%copyinparent : memref<10xf32>) to varPtr(%a : memref<10xf32>) {dataClause = #acc<data_clause acc_copy>}

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c9 = arith.constant 9 : index
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index

  %bounds1full = acc.bounds lowerbound(%c0 : index) upperbound(%c9 : index) stride(%c1 : index)
  %copyinfullslice1 = acc.copyin varPtr(%a : memref<10xf32>) bounds(%bounds1full) -> memref<10xf32> {name = "arrayA[0:9]"}
  // Specify full-bounds but assume that startIdx of array reference is 1.
  %bounds2full = acc.bounds lowerbound(%c1 : index) upperbound(%c20 : index) extent(%c20 : index) stride(%c4 : index) startIdx(%c1 : index) {strideInBytes = true}
  %copyinfullslice2 = acc.copyin varPtr(%c : memref<10x20xf32>) bounds(%bounds1full, %bounds2full) -> memref<10x20xf32>
  acc.parallel dataOperands(%copyinfullslice1, %copyinfullslice2 : memref<10xf32>, memref<10x20xf32>) {
  }

  %bounds1partial = acc.bounds lowerbound(%c4 : index) upperbound(%c9 : index) stride(%c1 : index)
  %copyinpartial = acc.copyin varPtr(%a : memref<10xf32>) bounds(%bounds1partial) -> memref<10xf32> {dataClause = #acc<data_clause acc_copy>}
  acc.parallel dataOperands(%copyinpartial : memref<10xf32>) {
  }
  acc.copyout accPtr(%copyinpartial : memref<10xf32>) bounds(%bounds1partial) to varPtr(%a : memref<10xf32>) {dataClause = #acc<data_clause acc_copy>}

  return
}
