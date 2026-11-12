// Vector width 1 (as register pressure or a short vl can make it): the
// vector<[1]xbf16> of a broadcast is a part of a vector<[2]xbf16>
// (broadcastTo); without Zvfbfmin its bits are extracted as vector<[1]xi16>.
// RUN: buddy-opt %s \
// RUN:     -lower-vir-to-vector='use-scalable=true vector-width=1 rvv-mattr=-zvfbfmin' \
// RUN: | FileCheck %s
// RUN: buddy-opt %s \
// RUN:     -lower-vir-to-vector='use-scalable=true vector-width=1 rvv-mattr=-zvfbfmin' \
// RUN:     -convert-vector-to-scf -lower-affine -convert-scf-to-cf \
// RUN:     -convert-arith-to-llvm -convert-math-to-llvm -convert-vector-to-llvm \
// RUN:     -expand-strided-metadata -finalize-memref-to-llvm \
// RUN:     -convert-arith-to-llvm -convert-cf-to-llvm -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | buddy-translate --buddy-to-llvmir \
// RUN: | llc -mtriple=riscv64 -mattr=+m,+f,+d,+v,+zfh,+zvfh -O0 \
// RUN:     -filetype=obj -o /dev/null
// The items of rvv-mattr are trimmed, blank ones skipped.
// RUN: buddy-opt %s \
// RUN:     -lower-vir-to-vector='use-scalable=true vector-width=1 rvv-mattr="+v, , -zvfbfmin "' \
// RUN: | FileCheck %s

// CHECK-LABEL: func.func @width_one
// CHECK-NOT: vector<[{{[0-9]+}}]xbf16>
// CHECK: vector.scalable.extract %{{.*}}[0] : vector<[1]xi16> from vector<[2]xi16>
// CHECK-NOT: vector<[{{[0-9]+}}]xbf16>
// CHECK: return
func.func @width_one(%s: bf16, %a: memref<128xbf16>, %b: memref<128xf32>,
                     %o: memref<128xbf16>) {
  %vl = arith.constant 128 : index
  vir.set_vl %vl : index {
    %v = vir.broadcast %s : bf16 -> !vir.vec<?xbf16>
    %c = vir.constant {value = 1.0 : bf16} : !vir.vec<?xbf16>
    %x = vir.load %a[] : memref<128xbf16> -> !vir.vec<?xbf16>
    %f = vir.load %b[] : memref<128xf32> -> !vir.vec<?xf32>
    %t = vir.truncf %f : !vir.vec<?xf32> -> !vir.vec<?xbf16>
    %n = arith.negf %t : !vir.vec<?xbf16>
    %m = math.absf %x : !vir.vec<?xbf16>
    %e = vir.extf %m : !vir.vec<?xbf16> -> !vir.vec<?xf32>
    %k = arith.cmpf olt, %e, %f : !vir.vec<?xf32>
    %y = vir.select %k, %n, %v : !vir.vec<?xi1>, !vir.vec<?xbf16>
    %z = vir.select %k, %y, %c : !vir.vec<?xi1>, !vir.vec<?xbf16>
    vir.store %z, %o[] : !vir.vec<?xbf16> -> memref<128xbf16>
    vector.yield
  }
  return
}
