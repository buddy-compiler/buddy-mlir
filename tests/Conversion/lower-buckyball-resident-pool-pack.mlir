// RUN: buddy-opt %s --target=pebble -lower-buckyball-to-bank-ssa -canonicalize -cse | FileCheck %s

// CHECK-LABEL: func.func @pool_tail(
// CHECK: linalg.fill ins(%c-128_i8 : i8)
// CHECK: arith.cmpi slt, {{.*}}, %c24 : index
// CHECK: memref.load %arg0
// CHECK: arith.divui {{.*}}, %c16 : index
// CHECK: arith.muli {{.*}}, %c36 : index
// CHECK: memref.store
// CHECK: buckyball.bank_maxpool
func.func @pool_tail(%input: memref<1x9x9x24xi8>, %output: memref<1x9x9x24xi8>) {
  buckyball.mega_kernel %input %output : memref<1x9x9x24xi8> memref<1x9x9x24xi8> {
    buckyball.mega_max_pool2d %input %output {finalOutput = false, kernel = 5 : i64, padding = 2 : i64, stride = 1 : i64} : memref<1x9x9x24xi8> memref<1x9x9x24xi8>
  }
  return
}

// CHECK-LABEL: func.func @pool_second_panel_chunk(
// CHECK: arith.addi {{.*}}, %c256 : index
// CHECK: memref.load %arg0
// CHECK: buckyball.bank_maxpool
func.func @pool_second_panel_chunk(%input: memref<1x2x2x272xi8>, %output: memref<1x2x2x272xi8>) {
  buckyball.mega_kernel %input %output : memref<1x2x2x272xi8> memref<1x2x2x272xi8> {
    buckyball.mega_max_pool2d %input %output {finalOutput = false, kernel = 1 : i64, padding = 0 : i64, stride = 1 : i64} : memref<1x2x2x272xi8> memref<1x2x2x272xi8>
  }
  return
}
