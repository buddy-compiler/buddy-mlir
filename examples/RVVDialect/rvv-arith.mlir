func.func @rvv_arith(%a: !rvv.vector<!rvv.m4,i32>,
                %b: !rvv.vector<!rvv.m4,i32>,
                %c: i32,
                %vl: i64) -> !rvv.vector<!rvv.m4,i32> {
  %0 = rvv.add %a, %b, %vl : !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.m4,i32>, i64
  %1 = rvv.add %a, %c, %vl : !rvv.vector<!rvv.m4,i32>, i32, i64
  %2 = rvv.sub %a, %b, %vl : !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.m4,i32>, i64
  %3 = rvv.sub %a, %c, %vl : !rvv.vector<!rvv.m4,i32>, i32, i64
  %4 = rvv.mul %a, %b, %vl : !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.m4,i32>, i64
  %5 = rvv.mul %a, %c, %vl : !rvv.vector<!rvv.m4,i32>, i32, i64
  %6 = rvv.div %a, %b, %vl : !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.m4,i32>, i64
  %7 = rvv.div %a, %c, %vl : !rvv.vector<!rvv.m4,i32>, i32, i64
  return %7 : !rvv.vector<!rvv.m4,i32>
}
