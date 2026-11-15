// Zvfbfmin (the default): scalable vectors of bf16, converted by RVV.
// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true rvv-vlen=128' \
// RUN: | FileCheck %s --check-prefix=ON
// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true rvv-vlen=128' \
// RUN:     -convert-vector-to-scf -lower-affine -convert-scf-to-cf \
// RUN:     -convert-arith-to-llvm -convert-math-to-llvm -convert-vector-to-llvm \
// RUN:     -expand-strided-metadata -finalize-memref-to-llvm \
// RUN:     -convert-arith-to-llvm -convert-cf-to-llvm -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | buddy-translate --buddy-to-llvmir \
// RUN: | llc -mtriple=riscv64 -mattr=+m,+f,+d,+v,+zfh,+zvfh,+zvfbfmin -O0 \
// RUN:     -o - \
// RUN: | FileCheck %s --check-prefix=ASM-ON

// -zvfbfmin (the last of the list): scalable vectors of the bf16 bits,
// converted with integer ops, which any RVV lowers.
// RUN: buddy-opt %s \
// RUN:     -lower-vir-to-vector='use-scalable=true rvv-vlen=128 rvv-mattr=+v,+zvfbfmin,-zvfbfmin' \
// RUN: | FileCheck %s --check-prefix=OFF
// RUN: buddy-opt %s \
// RUN:     -lower-vir-to-vector='use-scalable=true rvv-vlen=128 rvv-mattr=-zvfbfmin' \
// RUN:     -convert-vector-to-scf -lower-affine -convert-scf-to-cf \
// RUN:     -convert-arith-to-llvm -convert-math-to-llvm -convert-vector-to-llvm \
// RUN:     -expand-strided-metadata -finalize-memref-to-llvm \
// RUN:     -convert-arith-to-llvm -convert-cf-to-llvm -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | buddy-translate --buddy-to-llvmir \
// RUN: | llc -mtriple=riscv64 -mattr=+m,+f,+d,+v,+zfh,+zvfh -O0 -o - \
// RUN: | FileCheck %s --check-prefix=ASM-OFF

// ON-LABEL: func.func @ext
// ON: vector.load %{{.*}} : memref<128xbf16>, vector<[{{[0-9]+}}]xbf16>
// ON: arith.extf %{{.*}} : vector<[{{[0-9]+}}]xbf16> to vector<[{{[0-9]+}}]xf32>
// ASM-ON-LABEL: ext:
// ASM-ON: vfwcvtbf16.f.f.v
// OFF-LABEL: func.func @ext
// OFF-NOT: vector<[{{[0-9]+}}]xbf16>
// OFF: %[[B:.*]] = llvm.load %{{.*}} <alignment = 2> : !llvm.ptr -> vector<[{{[0-9]+}}]xi16>
// The NaNs are the canonical NaN, as vfwcvtbf16 makes them.
// OFF: %[[M:.*]] = arith.andi %[[B]], %{{.*}} : vector<[{{[0-9]+}}]xi16>
// OFF: %[[N:.*]] = arith.cmpi ugt, %[[M]], %{{.*}} : vector<[{{[0-9]+}}]xi16>
// OFF: %[[C:.*]] = arith.select %[[N]], %{{.*}}, %[[B]] : vector<[{{[0-9]+}}]xi1>, vector<[{{[0-9]+}}]xi16>
// OFF: %[[W:.*]] = arith.extui %[[C]] : vector<[{{[0-9]+}}]xi16> to vector<[{{[0-9]+}}]xi32>
// OFF: %[[H:.*]] = arith.shli %[[W]], %{{.*}} : vector<[{{[0-9]+}}]xi32>
// OFF: arith.bitcast %[[H]] : vector<[{{[0-9]+}}]xi32> to vector<[{{[0-9]+}}]xf32>
// OFF-NOT: vector<[{{[0-9]+}}]xbf16>
// OFF: return
// ASM-OFF-LABEL: ext:
// ASM-OFF-NOT: bf16
// ASM-OFF: vsll.vi
// ASM-OFF-LABEL: trunc_abs:
func.func @ext(%a: memref<128xbf16>, %o: memref<128xf32>) {
  %vl = arith.constant 128 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<128xbf16> -> !vir.vec<?xbf16>
    %y = vir.extf %x : !vir.vec<?xbf16> -> !vir.vec<?xf32>
    vir.store %y, %o[] : !vir.vec<?xf32> -> memref<128xf32>
    vector.yield
  }
  return
}

// ON-LABEL: func.func @trunc_abs
// ON: arith.truncf %{{.*}} : vector<[{{[0-9]+}}]xf32> to vector<[{{[0-9]+}}]xbf16>
// ON: math.absf %{{.*}} : vector<[{{[0-9]+}}]xbf16>
// ASM-ON-LABEL: trunc_abs:
// ASM-ON: vfncvtbf16.f.f.w
// OFF-LABEL: func.func @trunc_abs
// OFF-NOT: vector<[{{[0-9]+}}]xbf16>
// Rounded to nearest even: bits + 0x7fff + bit 16, the high half.
// OFF: %[[X:.*]] = arith.bitcast %{{.*}} : vector<[{{[0-9]+}}]xf32> to vector<[{{[0-9]+}}]xi32>
// OFF: %[[HI:.*]] = arith.shrui %[[X]], %{{.*}} : vector<[{{[0-9]+}}]xi32>
// OFF: %[[B16:.*]] = arith.andi %[[HI]], %{{.*}} : vector<[{{[0-9]+}}]xi32>
// OFF: %[[BIAS:.*]] = arith.addi %[[B16]], %{{.*}} : vector<[{{[0-9]+}}]xi32>
// OFF: %[[SUM:.*]] = arith.addi %[[X]], %[[BIAS]] : vector<[{{[0-9]+}}]xi32>
// OFF: %[[SH:.*]] = arith.shrui %[[SUM]], %{{.*}} : vector<[{{[0-9]+}}]xi32>
// OFF: %[[R:.*]] = arith.trunci %[[SH]] : vector<[{{[0-9]+}}]xi32> to vector<[{{[0-9]+}}]xi16>
// OFF: %[[NAN:.*]] = arith.cmpf uno, %{{.*}}, %{{.*}} : vector<[{{[0-9]+}}]xf32>
// OFF: %[[T:.*]] = arith.select %[[NAN]], %{{.*}}, %[[R]] : vector<[{{[0-9]+}}]xi1>, vector<[{{[0-9]+}}]xi16>
// absf clears the sign bit.
// OFF: %[[A:.*]] = arith.andi %[[T]], %{{.*}} : vector<[{{[0-9]+}}]xi16>
// OFF: llvm.store %[[A]], %{{.*}} <alignment = 2> : vector<[{{[0-9]+}}]xi16>, !llvm.ptr
// OFF-NOT: vector<[{{[0-9]+}}]xbf16>
// OFF: return
// ASM-OFF-NOT: bf16
// ASM-OFF: vnsrl.wi
func.func @trunc_abs(%a: memref<128xf32>, %o: memref<128xbf16>) {
  %vl = arith.constant 128 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<128xf32> -> !vir.vec<?xf32>
    %y = vir.truncf %x : !vir.vec<?xf32> -> !vir.vec<?xbf16>
    %z = math.absf %y : !vir.vec<?xbf16>
    vir.store %z, %o[] : !vir.vec<?xbf16> -> memref<128xbf16>
    vector.yield
  }
  return
}
