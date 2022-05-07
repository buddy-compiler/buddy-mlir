func.func @rvv_memory(%v: !rvv.vector<!rvv.m4,i32>,
                 %m: memref<?xi32>,
                 %vl: i64) -> !rvv.vector<!rvv.m4,i32> {
  %c0 = arith.constant 0 : index
  %0 = rvv.load %m[%c0], %vl : memref<?xi32>, !rvv.vector<!rvv.m4,i32>, i64
  rvv.store %v, %m[%c0], %vl : !rvv.vector<!rvv.m4,i32>, memref<?xi32>, i64
  return %0 : !rvv.vector<!rvv.m4,i32>
}
