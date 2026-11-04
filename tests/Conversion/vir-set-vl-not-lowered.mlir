// RUN: not buddy-opt %s -lower-vir-to-vector='use-scalable=true rvv-vlen=128 rvv-vlen-exact=false' \
// RUN:   -o /dev/null 2>&1 | FileCheck %s

// A vir.set_vl that is not lowered fails the pass (a nonzero exit), with the
// error of its lowering only.
func.func @dot(%a: memref<64xf32>) -> f32 {
  %vl = arith.constant 64 : index
  %z = arith.constant 0.0 : f32
  %r = vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<64xf32> -> !vir.vec<?xf32>
    %s = vir.reduce %x, %z {kind = "add"} : !vir.vec<?xf32>, f32 -> f32
    vector.yield %s : f32
  } -> f32
  return %r : f32
}

// CHECK: error: a vir.set_vl with results on scalable vectors needs the exact VLEN (rvv-vlen-exact)
// CHECK-NOT: : error:
