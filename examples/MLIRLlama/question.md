I can create a linalg.generic op in python binding. For example,
```
linalg.GenericOp([tensor_type], [op1], [op2],
                ir.ArrayAttr.get([ir.AffineMapAttr.get(ir.AffineMap.get_permutation([0, 1])),
                   ir.AffineMapAttr.get(ir.AffineMap.get_permutation([0, 1]))]),
                 ir.ArrayAttr.get([ir.StringAttr.get("parallel")]*2))
```
```
%3 = "linalg.generic"(%1, %2) ({
}) {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x13xf32>, tensor<1x13xf32>) -> tensor<1x13xf32>
```
But in this op, I can't create a basic block

I want to get a op like this, it has a '^bb0' in its body
```
linalg.generic #matmul_trait
  ins(%A, %B : memref<?x?xf32, stride_specification>,
               memref<?x?xf32, stride_specification>)
  outs(%C : memref<?x?xf32, stride_specification>)
  {other-optional-attributes} {
  ^bb0(%a: f32, %b: f32, %c: f32) :
    %d = arith.mulf %a, %b: f32
    %e = arith.addf %c, %d: f32
    linalg.yield %e : f32
}
```
How can I get this in python binding?

Thank you for your answer!