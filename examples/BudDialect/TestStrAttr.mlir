module {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %res1 = bud.test_str_attr %c1, %c2 {arith = "add"} : i32
  %res2 = bud.test_str_attr %c1, %c2 {arith = "sub"} : i32
  %res3 = bud.test_str_attr %c1, %c2 : i32
}
