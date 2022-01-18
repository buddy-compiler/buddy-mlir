func @rvv_masked_arith(%maskedoff: !rvv.vector<!rvv.m4,i32>,
                       %a: !rvv.vector<!rvv.m4,i32>,
                       %b: !rvv.vector<!rvv.m4,i32>,
                       %c: i32,
                       %mask: !rvv.vector<!rvv.mask8,i1>,
                       %vl: i64,
                       %vta: i64) -> !rvv.vector<!rvv.m4,i32> {
    %0 = rvv.masked.add %maskedoff, %a, %b, %mask, %vl, %vta : !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.mask8,i1>, i64
    %1 = rvv.masked.add %maskedoff, %a, %c, %mask, %vl, %vta : !rvv.vector<!rvv.m4,i32>, i32, !rvv.vector<!rvv.mask8,i1>, i64
    %2 = rvv.masked.sub %maskedoff, %a, %b, %mask, %vl, %vta : !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.mask8,i1>, i64
    %3 = rvv.masked.sub %maskedoff, %a, %c, %mask, %vl, %vta : !rvv.vector<!rvv.m4,i32>, i32, !rvv.vector<!rvv.mask8,i1>, i64
    %4 = rvv.masked.mul %maskedoff, %a, %b, %mask, %vl, %vta : !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.mask8,i1>, i64
    %5 = rvv.masked.mul %maskedoff, %a, %c, %mask, %vl, %vta : !rvv.vector<!rvv.m4,i32>, i32, !rvv.vector<!rvv.mask8,i1>, i64
    %6 = rvv.masked.div %maskedoff, %a, %b, %mask, %vl, %vta : !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.m4,i32>, !rvv.vector<!rvv.mask8,i1>, i64
    %7 = rvv.masked.add %maskedoff, %a, %c, %mask, %vl, %vta : !rvv.vector<!rvv.m4,i32>, i32, !rvv.vector<!rvv.mask8,i1>, i64
  return %7 : !rvv.vector<!rvv.m4,i32>
}
