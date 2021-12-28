// RUN: buddy-opt -verify-diagnostics %s | buddy-opt | FileCheck %s

// CHECK-LABEL: func @rvv_memory
func @rvv_memory(%v: !rvv.vector<!rvv.m4,i32>,
                 %m: memref<?xi32>,
                 %vl: i64) -> !rvv.vector<!rvv.m4,i32> {
  %c0 = arith.constant 0 : index
  // CHECK: rvv.load {{.*}}: memref<?xi32>, !rvv.vector<!rvv.m4,i32>, i64
  %0 = rvv.load %m[%c0], %vl : memref<?xi32>, !rvv.vector<!rvv.m4,i32>, i64
  // CHECK: rvv.store {{.*}}: !rvv.vector<!rvv.m4,i32>, memref<?xi32>, i64
  rvv.store %v, %m[%c0], %vl : !rvv.vector<!rvv.m4,i32>, memref<?xi32>, i64
  return %0 : !rvv.vector<!rvv.m4,i32>
}

// CHECK-LABEL: func @rvv_arith
func @rvv_arith(%a: !rvv.vector<!rvv.m4,i32>,
                %b: !rvv.vector<!rvv.m4,i32>,
                %c: i32,
                %vl: i64) -> !rvv.vector<!rvv.m4,i32> {
  // CHECK: rvv.add {{.*}} : !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.m4,i32>, i64
  %0 = rvv.add %a, %b, %vl : !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.m4,i32>, i64
  // CHECK: rvv.add {{.*}} : !rvv.vector<!rvv.m4,i32>, i32, i64
  %1 = rvv.add %a, %c, %vl : !rvv.vector<!rvv.m4,i32>, i32, i64
  // CHECK: rvv.sub {{.*}} : !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.m4,i32>, i64
  %2 = rvv.sub %a, %b, %vl : !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.m4,i32>, i64
  // CHECK: rvv.sub {{.*}} : !rvv.vector<!rvv.m4,i32>, i32, i64
  %3 = rvv.sub %a, %c, %vl : !rvv.vector<!rvv.m4,i32>, i32, i64
  // CHECK: rvv.mul {{.*}} : !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.m4,i32>, i64
  %4 = rvv.mul %a, %b, %vl : !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.m4,i32>, i64
  // CHECK: rvv.mul {{.*}} : !rvv.vector<!rvv.m4,i32>, i32, i64
  %5 = rvv.mul %a, %c, %vl : !rvv.vector<!rvv.m4,i32>, i32, i64
  // CHECK: rvv.div {{.*}} : !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.m4,i32>, i64
  %6 = rvv.div %a, %b, %vl : !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.m4,i32>, i64
  // CHECK: rvv.div {{.*}} : !rvv.vector<!rvv.m4,i32>, i32, i64
  %7 = rvv.div %a, %c, %vl : !rvv.vector<!rvv.m4,i32>, i32, i64
  return %7 : !rvv.vector<!rvv.m4,i32>
}

// CHECK-LABEL: func @rvv_masked_arith
func @rvv_masked_arith(%maskedoff: !rvv.vector<!rvv.m4,i32>,
                       %a: !rvv.vector<!rvv.m4,i32>,
                       %b: !rvv.vector<!rvv.m4,i32>,
                       %c: i32,
                       %mask: !rvv.vector<!rvv.mask8,i1>,
                       %vl: i64,
                       %vta: i64) -> !rvv.vector<!rvv.m4,i32> {
  // CHECK: rvv.masked.add {{.*}} : !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.mask8,i1>, i64
  %0 = rvv.masked.add %maskedoff, %a, %b, %mask, %vl, %vta : !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.mask8,i1>, i64
  // CHECK: rvv.masked.add {{.*}} : !rvv.vector<!rvv.m4,i32>, i32, !rvv.vector<!rvv.mask8,i1>, i64
  %1 = rvv.masked.add %maskedoff, %a, %c, %mask, %vl, %vta : !rvv.vector<!rvv.m4,i32>, i32, !rvv.vector<!rvv.mask8,i1>, i64
  // CHECK: rvv.masked.sub {{.*}} : !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.mask8,i1>, i64
  %2 = rvv.masked.sub %maskedoff, %a, %b, %mask, %vl, %vta : !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.mask8,i1>, i64
  // CHECK: rvv.masked.sub {{.*}} : !rvv.vector<!rvv.m4,i32>, i32, !rvv.vector<!rvv.mask8,i1>, i64
  %3 = rvv.masked.sub %maskedoff, %a, %c, %mask, %vl, %vta : !rvv.vector<!rvv.m4,i32>, i32, !rvv.vector<!rvv.mask8,i1>, i64
  // CHECK: rvv.masked.mul {{.*}} : !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.mask8,i1>, i64
  %4 = rvv.masked.mul %maskedoff, %a, %b, %mask, %vl, %vta : !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.mask8,i1>, i64
  // CHECK: rvv.masked.mul {{.*}} : !rvv.vector<!rvv.m4,i32>, i32, !rvv.vector<!rvv.mask8,i1>, i64
  %5 = rvv.masked.mul %maskedoff, %a, %c, %mask, %vl, %vta : !rvv.vector<!rvv.m4,i32>, i32, !rvv.vector<!rvv.mask8,i1>, i64
  // CHECK: rvv.masked.div {{.*}} : !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.mask8,i1>, i64
  %6 = rvv.masked.div %maskedoff, %a, %b, %mask, %vl, %vta : !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.mask8,i1>, i64
  // CHECK: rvv.masked.div {{.*}} : !rvv.vector<!rvv.m4,i32>, i32, !rvv.vector<!rvv.mask8,i1>, i64
  %7 = rvv.masked.div %maskedoff, %a, %c, %mask, %vl, %vta : !rvv.vector<!rvv.m4,i32>, i32, !rvv.vector<!rvv.mask8,i1>, i64
  return %7 : !rvv.vector<!rvv.m4,i32>
}
