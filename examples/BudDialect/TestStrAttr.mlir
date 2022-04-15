module {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %res1 = bud.test_enum_attr add %c1, %c2 : i32
  %res2 = bud.test_enum_attr sub %c1, %c2 : i32
  %res3 = "bud.test_enum_attr"(%c1, %c2) {arith = #bud<"test_enum_attr_op sub">} : (i32, i32) -> i32
}
