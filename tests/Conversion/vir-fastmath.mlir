// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true rvv-vlen=256' \
// RUN: | FileCheck %s
// RUN: buddy-opt %s -lower-vir-to-vector='rvv-vlen=256' \
// RUN: | FileCheck %s --check-prefix=FIXED
// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true rvv-vlen=256' \
// RUN:     -convert-vector-to-scf -lower-affine -convert-scf-to-cf \
// RUN:     -convert-arith-to-llvm -convert-math-to-llvm -convert-vector-to-llvm \
// RUN:     -expand-strided-metadata -finalize-memref-to-llvm \
// RUN:     -convert-arith-to-llvm -convert-cf-to-llvm -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | buddy-translate --buddy-to-llvmir \
// RUN: | llc -mtriple=riscv64 \
// RUN:     -mattr=+m,+f,+d,+v,+zfh,+zvfh,+zvfbfmin,+zvfbfwma -O2 -o - \
// RUN: | FileCheck %s --check-prefix=ASM

// The fast-math flags of the ops of a vir.set_vl stay on the ops they are
// lowered to, in the vector loop and in the tail: with contract, LLVM fuses
// a multiply and an add (on bf16 with Zvfbfwma, vfwmaccbf16).

// CHECK-LABEL: func.func @muladd
// CHECK: arith.mulf %{{.*}}, %{{.*}} fastmath<contract> : vector<[{{[0-9]+}}]xf32>
// CHECK: arith.addf %{{.*}}, %{{.*}} fastmath<contract> : vector<[{{[0-9]+}}]xf32>
// CHECK: arith.mulf %{{.*}}, %{{.*}} fastmath<contract> : f32
// CHECK: arith.addf %{{.*}}, %{{.*}} fastmath<contract> : f32
// FIXED-LABEL: func.func @muladd
// FIXED: arith.mulf %{{.*}}, %{{.*}} fastmath<contract> : vector<{{[0-9]+}}xf32>
// FIXED: arith.addf %{{.*}}, %{{.*}} fastmath<contract> : vector<{{[0-9]+}}xf32>
// FIXED: arith.mulf %{{.*}}, %{{.*}} fastmath<contract> : f32
// ASM-LABEL: muladd:
// ASM-NOT: vfmul
// ASM: vf{{n?}}m{{acc|add}}.vv
func.func @muladd(%a: memref<100xf32>, %b: memref<100xf32>,
                  %c: memref<100xf32>) {
  %vl = arith.constant 100 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<100xf32> -> !vir.vec<?xf32>
    %y = vir.load %b[] : memref<100xf32> -> !vir.vec<?xf32>
    %z = vir.load %c[] : memref<100xf32> -> !vir.vec<?xf32>
    %m = arith.mulf %x, %y fastmath<contract> : !vir.vec<?xf32>
    %r = arith.addf %z, %m fastmath<contract> : !vir.vec<?xf32>
    vir.store %r, %c[] : !vir.vec<?xf32> -> memref<100xf32>
    vector.yield
  }
  return
}

// CHECK-LABEL: func.func @bf16_muladd
// CHECK: arith.mulf %{{.*}}, %{{.*}} fastmath<contract> : vector<[{{[0-9]+}}]xf32>
// CHECK: arith.addf %{{.*}}, %{{.*}} fastmath<contract> : vector<[{{[0-9]+}}]xf32>
// ASM-LABEL: bf16_muladd:
// ASM: vfwmaccbf16.vv
func.func @bf16_muladd(%a: memref<128xbf16>, %b: memref<128xbf16>,
                       %c: memref<128xf32>) {
  %vl = arith.constant 128 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<128xbf16> -> !vir.vec<?xbf16>
    %y = vir.load %b[] : memref<128xbf16> -> !vir.vec<?xbf16>
    %z = vir.load %c[] : memref<128xf32> -> !vir.vec<?xf32>
    %ex = vir.extf %x : !vir.vec<?xbf16> -> !vir.vec<?xf32>
    %ey = vir.extf %y : !vir.vec<?xbf16> -> !vir.vec<?xf32>
    %m = arith.mulf %ex, %ey fastmath<contract> : !vir.vec<?xf32>
    %r = arith.addf %z, %m fastmath<contract> : !vir.vec<?xf32>
    vir.store %r, %c[] : !vir.vec<?xf32> -> memref<128xf32>
    vector.yield
  }
  return
}

// Without flags, none: no fusion.
// CHECK-LABEL: func.func @plain
// CHECK-NOT: fastmath
// CHECK: return
// ASM-LABEL: plain:
// ASM: vfmul.vv
// ASM: vfadd.vv
func.func @plain(%a: memref<128xf32>, %b: memref<128xf32>,
                 %c: memref<128xf32>) {
  %vl = arith.constant 128 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<128xf32> -> !vir.vec<?xf32>
    %y = vir.load %b[] : memref<128xf32> -> !vir.vec<?xf32>
    %z = vir.load %c[] : memref<128xf32> -> !vir.vec<?xf32>
    %m = arith.mulf %x, %y : !vir.vec<?xf32>
    %r = arith.addf %z, %m : !vir.vec<?xf32>
    vir.store %r, %c[] : !vir.vec<?xf32> -> memref<128xf32>
    vector.yield
  }
  return
}
