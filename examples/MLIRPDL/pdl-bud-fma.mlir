module @patterns {
  pdl.pattern : benefit(1) {
    %vector_type = pdl.type
    %lhs = pdl.operand : %vector_type
    %rhs = pdl.operand : %vector_type
    %acc = pdl.operand : %vector_type

    %attr0 = pdl.attribute = false
    %op0 = pdl.operation "bud.fma" (%lhs, %rhs, %acc : !pdl.value, !pdl.value, !pdl.value) {"splitting" = %attr0} -> (%vector_type : !pdl.type)

    pdl.rewrite %op0 {
      %op1 = pdl.operation "vector.fma" (%lhs, %rhs, %acc: !pdl.value, !pdl.value, !pdl.value) -> (%vector_type : !pdl.type)
      %val1 = pdl.result 0 of %op1
      pdl.replace %op0 with (%val1 : !pdl.value)
    }
  }
}

module @ir {
  memref.global "private" @gv : memref<4x4xf32> = dense<[[0. , 1. , 2. , 3. ],
                                                         [10., 11., 12., 13.],
                                                         [20., 21., 22., 23.],
                                                         [30., 31., 32., 33.]]>
  func.func @main() {
    %mem = memref.get_global @gv : memref<4x4xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %load_vec1 = vector.load %mem[%c0, %c0] : memref<4x4xf32>, vector<4xf32>
    %load_vec2 = vector.load %mem[%c1, %c0] : memref<4x4xf32>, vector<4xf32>
    %load_vec3 = vector.load %mem[%c2, %c0] : memref<4x4xf32>, vector<4xf32>
    %result = "bud.fma"(%load_vec1, %load_vec2, %load_vec3) {splitting = false} : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    vector.print %result : vector<4xf32>
    return
  }
}
