// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true rvv-vlen=128' \
// RUN: | FileCheck %s --check-prefixes=CHECK,ON
// RUN: buddy-opt %s \
// RUN:     -lower-vir-to-vector='use-scalable=true rvv-vlen=128 rvv-mattr=-zvfbfmin' \
// RUN: | FileCheck %s --check-prefixes=CHECK,OFF

// With use-scalable, a region with dynamic vectors that RVV does not hold
// is lowered with fixed vectors, which LLVM lowers lane by lane: those of an
// f8 type (no RVV extension converts them), and without Zvfbfmin those of
// bf16 in an op that has no lowering on their bits (here an addf).

// CHECK-LABEL: func.func @f32_to_f8
// CHECK-NOT: vector<[
// CHECK: arith.truncf %{{.*}} : vector<{{[0-9]+}}xf32> to vector<{{[0-9]+}}xf8E4M3FN>
// CHECK-NOT: vector<[
// CHECK: return
func.func @f32_to_f8(%a: memref<128xf32>, %o: memref<128xf8E4M3FN>) {
  %vl = arith.constant 128 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<128xf32> -> !vir.vec<?xf32>
    %y = vir.truncf %x : !vir.vec<?xf32> -> !vir.vec<?xf8E4M3FN>
    vir.store %y, %o[] : !vir.vec<?xf8E4M3FN> -> memref<128xf8E4M3FN>
    vector.yield
  }
  return
}

// CHECK-LABEL: func.func @bf16_add
// ON: arith.addf %{{.*}}, %{{.*}} : vector<[{{[0-9]+}}]xbf16>
// OFF-NOT: vector<[
// OFF: arith.addf %{{.*}}, %{{.*}} : vector<{{[0-9]+}}xbf16>
// OFF-NOT: vector<[
// CHECK: return
func.func @bf16_add(%a: memref<128xbf16>, %o: memref<128xbf16>) {
  %vl = arith.constant 128 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<128xbf16> -> !vir.vec<?xbf16>
    %y = arith.addf %x, %x : !vir.vec<?xbf16>
    vir.store %y, %o[] : !vir.vec<?xbf16> -> memref<128xbf16>
    vector.yield
  }
  return
}

// Without Zvfbfmin, bf16 bits are read through a pointer of the address
// space of the memory space: one that is not an integer has none, and the
// region is lowered with fixed vectors.
// OFF-LABEL: func.func @symbolic_space
// OFF-NOT: vector<[
// OFF: vector.load %{{.*}} : memref<128xbf16, "gpu">, vector<{{[0-9]+}}xbf16>
// OFF-NOT: vector<[
// OFF: return
func.func @symbolic_space(%a: memref<128xbf16, "gpu">, %o: memref<128xf32>) {
  %vl = arith.constant 128 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<128xbf16, "gpu"> -> !vir.vec<?xbf16>
    %e = vir.extf %x : !vir.vec<?xbf16> -> !vir.vec<?xf32>
    vir.store %e, %o[] : !vir.vec<?xf32> -> memref<128xf32>
    vector.yield
  }
  return
}
