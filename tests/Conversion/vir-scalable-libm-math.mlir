// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true rvv-vlen=128' \
// RUN: | FileCheck %s
// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true rvv-vlen=128' \
// RUN:     -convert-vector-to-scf -lower-affine -convert-scf-to-cf \
// RUN:     -convert-arith-to-llvm -convert-math-to-llvm -convert-math-to-libm \
// RUN:     -convert-vector-to-llvm -expand-strided-metadata \
// RUN:     -finalize-memref-to-llvm -convert-arith-to-llvm -convert-cf-to-llvm \
// RUN:     -convert-func-to-llvm -reconcile-unrealized-casts \
// RUN: | buddy-translate --buddy-to-llvmir \
// RUN: | llc -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c,+v,+zfh,+zvfh -O0 \
// RUN:     -filetype=asm -o - \
// RUN: | FileCheck %s --check-prefix=ASM

// A math function that LLVM does not lower on scalable vectors (erf, erfc,
// cbrt: libm only; powf, atan2: binary) is applied to the N * vscale lanes
// of a vector<[N]xT> one by one: convert-math-to-libm would call it on the N
// first lanes only and leave the others 0.

// CHECK-LABEL: func.func @erf
// CHECK: %[[X:.*]] = vector.load %{{.*}} : memref<128xf32>, vector<[16]xf32>
// CHECK: %[[VS:.*]] = vector.vscale
// CHECK: %[[N:.*]] = arith.muli %[[VS]], %{{.*}} : index
// CHECK: %[[Y:.*]] = scf.for %[[I:.*]] = %{{.*}} to %[[N]] step %{{.*}} iter_args(%[[ACC:.*]] = %[[X]]) -> (vector<[16]xf32>)
// CHECK:   %[[E:.*]] = vector.extract %[[X]][%[[I]]] : f32 from vector<[16]xf32>
// CHECK:   %[[R:.*]] = math.erf %[[E]] : f32
// CHECK:   %[[INS:.*]] = vector.insert %[[R]], %[[ACC]] [%[[I]]] : f32 into vector<[16]xf32>
// CHECK:   scf.yield %[[INS]]
// CHECK: vector.store %[[Y]], %{{.*}} : memref<128xf32>, vector<[16]xf32>
// CHECK-NOT: math.erf %{{.*}} : vector
// ASM-LABEL: erf:
// ASM: call erff
func.func @erf(%a: memref<128xf32>, %o: memref<128xf32>) {
  %vl = arith.constant 128 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<128xf32> -> !vir.vec<?xf32>
    %y = math.erf %x : !vir.vec<?xf32>
    vir.store %y, %o[] : !vir.vec<?xf32> -> memref<128xf32>
    vector.yield
  }
  return
}

// f16 lanes: the scalar math.erf of f16 is promoted to f32 by
// convert-math-to-libm.
// CHECK-LABEL: func.func @erf_f16
// CHECK: scf.for {{.*}} -> (vector<[32]xf16>)
// CHECK:   math.erf %{{.*}} : f16
// CHECK-NOT: math.erf %{{.*}} : vector
// ASM-LABEL: erf_f16:
// ASM: call erff
func.func @erf_f16(%a: memref<128xf16>, %o: memref<128xf16>) {
  %vl = arith.constant 128 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<128xf16> -> !vir.vec<?xf16>
    %y = math.erf %x : !vir.vec<?xf16>
    vir.store %y, %o[] : !vir.vec<?xf16> -> memref<128xf16>
    vector.yield
  }
  return
}

// A binary one: each lane of each operand.
// CHECK-LABEL: func.func @powf
// CHECK: scf.for {{.*}} -> (vector<[16]xf32>)
// CHECK:   %[[A:.*]] = vector.extract %{{.*}}[%{{.*}}] : f32 from vector<[16]xf32>
// CHECK:   %[[B:.*]] = vector.extract %{{.*}}[%{{.*}}] : f32 from vector<[16]xf32>
// CHECK:   math.powf %[[A]], %[[B]] : f32
// CHECK-NOT: math.powf %{{.*}} : vector
// ASM-LABEL: powf:
// ASM: call powf
func.func @powf(%a: memref<128xf32>, %b: memref<128xf32>,
                %o: memref<128xf32>) {
  %vl = arith.constant 128 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<128xf32> -> !vir.vec<?xf32>
    %e = vir.load %b[] : memref<128xf32> -> !vir.vec<?xf32>
    %y = math.powf %x, %e : !vir.vec<?xf32>
    vir.store %y, %o[] : !vir.vec<?xf32> -> memref<128xf32>
    vector.yield
  }
  return
}

// The others stay on vectors: LLVM lowers them on scalable vectors (exp: a
// loop calling expf on each lane; sqrt: an RVV instruction).
// CHECK-LABEL: func.func @exp_sqrt
// CHECK: math.exp %{{.*}} : vector<[16]xf32>
// CHECK: math.sqrt %{{.*}} : vector<[16]xf32>
// ASM-LABEL: exp_sqrt:
// ASM: call expf
// ASM: vfsqrt.v
func.func @exp_sqrt(%a: memref<128xf32>, %o: memref<128xf32>) {
  %vl = arith.constant 128 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<128xf32> -> !vir.vec<?xf32>
    %e = math.exp %x : !vir.vec<?xf32>
    %y = math.sqrt %e : !vir.vec<?xf32>
    vir.store %y, %o[] : !vir.vec<?xf32> -> memref<128xf32>
    vector.yield
  }
  return
}
