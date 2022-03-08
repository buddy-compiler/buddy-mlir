// RUN: buddy-opt %s -lower-rvv -convert-func-to-llvm | buddy-opt | FileCheck %s

func @rvv_memory(%v: !rvv.vector<!rvv.m4,i32>,
                 %m: memref<?xi32>,
                 %vl: i64) -> !rvv.vector<!rvv.m4,i32> {
  %c0 = arith.constant 0 : index
  // CHECK: llvm.extractvalue {{.*}} : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK-NEXT: llvm.getelementptr {{.*}} : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
  // CHECK-NEXT: llvm.bitcast {{.*}} : !llvm.ptr<i32> to !llvm.ptr<vec<? x 8 x i32>>
  // CHECK-NEXT: rvv.intr.vle{{.*}} : (!llvm.ptr<vec<? x 8 x i32>>, i64) -> !llvm.vec<? x 8 x i32>
  %0 = rvv.load %m[%c0], %vl : memref<?xi32>, !rvv.vector<!rvv.m4,i32>, i64
  // CHECK: llvm.extractvalue {{.*}} : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK-NEXT: llvm.getelementptr {{.*}} : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
  // CHECK-NEXT: llvm.bitcast {{.*}} : !llvm.ptr<i32> to !llvm.ptr<vec<? x 8 x i32>>
  // CHECK-NEXT: rvv.intr.vse{{.*}} : (!llvm.vec<? x 8 x i32>, !llvm.ptr<vec<? x 8 x i32>>, i64) -> ()
  rvv.store %v, %m[%c0], %vl : !rvv.vector<!rvv.m4,i32>, memref<?xi32>, i64
  return %0 : !rvv.vector<!rvv.m4,i32>
}

func @rvv_arith(%a: !rvv.vector<!rvv.m4,i32>,
                %b: !rvv.vector<!rvv.m4,i32>,
                %c: i32,
                %vl: i64) -> !rvv.vector<!rvv.m4,i32> {
  // CHECK: rvv.intr.vadd{{.*}} : (!llvm.vec<? x 8 x i32>, !llvm.vec<? x 8 x i32>, i64) -> !llvm.vec<? x 8 x i32>
  %0 = rvv.add %a, %b, %vl : !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.m4,i32>, i64
  // CHECK: rvv.intr.vadd{{.*}} : (!llvm.vec<? x 8 x i32>, i32, i64) -> !llvm.vec<? x 8 x i32>
  %1 = rvv.add %a, %c, %vl : !rvv.vector<!rvv.m4,i32>, i32, i64
  // CHECK: rvv.intr.vsub{{.*}} : (!llvm.vec<? x 8 x i32>, !llvm.vec<? x 8 x i32>, i64) -> !llvm.vec<? x 8 x i32>
  %2 = rvv.sub %a, %b, %vl : !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.m4,i32>, i64
  // CHECK: rvv.intr.vsub{{.*}} : (!llvm.vec<? x 8 x i32>, i32, i64) -> !llvm.vec<? x 8 x i32>
  %3 = rvv.sub %a, %c, %vl : !rvv.vector<!rvv.m4,i32>, i32, i64
  // CHECK: rvv.intr.vmul{{.*}} : (!llvm.vec<? x 8 x i32>, !llvm.vec<? x 8 x i32>, i64) -> !llvm.vec<? x 8 x i32>
  %4 = rvv.mul %a, %b, %vl : !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.m4,i32>, i64
  // CHECK: rvv.intr.vmul{{.*}} : (!llvm.vec<? x 8 x i32>, i32, i64) -> !llvm.vec<? x 8 x i32>
  %5 = rvv.mul %a, %c, %vl : !rvv.vector<!rvv.m4,i32>, i32, i64
  // CHECK: rvv.intr.vdiv{{.*}} : (!llvm.vec<? x 8 x i32>, !llvm.vec<? x 8 x i32>, i64) -> !llvm.vec<? x 8 x i32>
  %6 = rvv.div %a, %b, %vl : !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.m4,i32>, i64
  // CHECK: rvv.intr.vdiv{{.*}} : (!llvm.vec<? x 8 x i32>, i32, i64) -> !llvm.vec<? x 8 x i32>
  %7 = rvv.div %a, %c, %vl : !rvv.vector<!rvv.m4,i32>, i32, i64
  return %7 : !rvv.vector<!rvv.m4,i32>
}

func @rvv_masked_arith(%maskedoff: !rvv.vector<!rvv.m4,i32>,
                       %a: !rvv.vector<!rvv.m4,i32>,
                       %b: !rvv.vector<!rvv.m4,i32>,
                       %c: i32,
                       %mask: !rvv.vector<!rvv.mask8,i1>,
                       %vl: i64,
                       %vta: i64) -> !rvv.vector<!rvv.m4,i32> {
  // CHECK: rvv.intr.vadd_mask{{.*}} : (!llvm.vec<? x 8 x i32>, !llvm.vec<? x 8 x i32>, !llvm.vec<? x 8 x i32>, !llvm.vec<? x 8 x i1>, i64, i64) -> !llvm.vec<? x 8 x i32>
  %0 = rvv.masked.add %maskedoff, %a, %b, %mask, %vl, %vta: !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.mask8,i1>, i64
  // CHECK: rvv.intr.vadd_mask{{.*}} : (!llvm.vec<? x 8 x i32>, !llvm.vec<? x 8 x i32>, i32, !llvm.vec<? x 8 x i1>, i64, i64) -> !llvm.vec<? x 8 x i32>
  %1 = rvv.masked.add %maskedoff, %a, %c, %mask, %vl, %vta: !rvv.vector<!rvv.m4,i32>, i32, !rvv.vector<!rvv.mask8,i1>, i64
  // CHECK: rvv.intr.vsub_mask{{.*}} : (!llvm.vec<? x 8 x i32>, !llvm.vec<? x 8 x i32>, !llvm.vec<? x 8 x i32>, !llvm.vec<? x 8 x i1>, i64, i64) -> !llvm.vec<? x 8 x i32>
  %2 = rvv.masked.sub %maskedoff, %a, %b, %mask, %vl, %vta: !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.mask8,i1>, i64
  // CHECK: rvv.intr.vsub_mask{{.*}} : (!llvm.vec<? x 8 x i32>, !llvm.vec<? x 8 x i32>, i32, !llvm.vec<? x 8 x i1>, i64, i64) -> !llvm.vec<? x 8 x i32>
  %3 = rvv.masked.sub %maskedoff, %a, %c, %mask, %vl, %vta: !rvv.vector<!rvv.m4,i32>, i32, !rvv.vector<!rvv.mask8,i1>, i64
  // CHECK: rvv.intr.vmul_mask{{.*}} : (!llvm.vec<? x 8 x i32>, !llvm.vec<? x 8 x i32>, !llvm.vec<? x 8 x i32>, !llvm.vec<? x 8 x i1>, i64, i64) -> !llvm.vec<? x 8 x i32>
  %4 = rvv.masked.mul %maskedoff, %a, %b, %mask, %vl, %vta: !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.mask8,i1>, i64
  // CHECK: rvv.intr.vmul_mask{{.*}} : (!llvm.vec<? x 8 x i32>, !llvm.vec<? x 8 x i32>, i32, !llvm.vec<? x 8 x i1>, i64, i64) -> !llvm.vec<? x 8 x i32>
  %5 = rvv.masked.mul %maskedoff, %a, %c, %mask, %vl, %vta: !rvv.vector<!rvv.m4,i32>, i32, !rvv.vector<!rvv.mask8,i1>, i64
  // CHECK: rvv.intr.vdiv_mask{{.*}} : (!llvm.vec<? x 8 x i32>, !llvm.vec<? x 8 x i32>, !llvm.vec<? x 8 x i32>, !llvm.vec<? x 8 x i1>, i64, i64) -> !llvm.vec<? x 8 x i32>
  %6 = rvv.masked.div %maskedoff, %a, %b, %mask, %vl, %vta: !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.mask8,i1>, i64
  // CHECK: rvv.intr.vdiv_mask{{.*}} : (!llvm.vec<? x 8 x i32>, !llvm.vec<? x 8 x i32>, i32, !llvm.vec<? x 8 x i1>, i64, i64) -> !llvm.vec<? x 8 x i32>
  %7 = rvv.masked.div %maskedoff, %a, %c, %mask, %vl, %vta: !rvv.vector<!rvv.m4,i32>, i32, !rvv.vector<!rvv.mask8,i1>, i64
  return %7 : !rvv.vector<!rvv.m4,i32>
}
