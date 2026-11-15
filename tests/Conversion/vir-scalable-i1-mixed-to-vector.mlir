// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true rvv-vlen=256' \
// RUN: | FileCheck %s
// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true rvv-vlen=256' \
// RUN:     -convert-vector-to-scf -lower-affine -convert-scf-to-cf \
// RUN:     -convert-vector-to-llvm -expand-strided-metadata \
// RUN:     -finalize-memref-to-llvm -convert-arith-to-llvm -convert-cf-to-llvm \
// RUN:     -convert-func-to-llvm -reconcile-unrealized-casts \
// RUN: | buddy-translate --buddy-to-llvmir \
// RUN: | llc -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c,+v,+zvl256b \
// RUN:     -riscv-v-vector-bits-max=256 -filetype=asm -o - \
// RUN: | FileCheck %s --check-prefix=ASM

// A region whose first dynamic vector is i1 (a mask) and whose others are
// f32: the vectors are sized for the widest element, f32, so that an f32
// vector is at most a group of 8 registers (vector<[16]xf32>, 64 lanes at
// VLEN 256); the i1 lanes are read as as many bytes (vector<[16]xi8>, LMUL
// 2). Sized for the i1 (as a byte), the f32 vectors were vector<[32]xf32>,
// 16 registers.

// CHECK-LABEL: func.func @select_masked
// CHECK-NOT: vector<[32]xf32>
// CHECK: llvm.load %{{.*}} <alignment = 1> : !llvm.ptr -> vector<[16]xi8>
// CHECK: arith.trunci %{{.*}} : vector<[16]xi8> to vector<[16]xi1>
// CHECK: vector.load %{{.*}} : memref<128xf32>, vector<[16]xf32>
// CHECK: {{(arith|llvm)}}.select %{{.*}}, %{{.*}}, %{{.*}} : vector<[16]xi1>, vector<[16]xf32>
// CHECK: vector.store %{{.*}} : memref<128xf32>, vector<[16]xf32>

// The loop body: the mask bytes as one group of 2 registers, the f32 lanes
// as one group of 8.
// ASM-LABEL: select_masked:
// ASM-NOT: {{m16|m32}}
// ASM: vl2r.v
// ASM: vsetvli {{.*}}, e8, m2
// ASM: vmsne.vi v0
// ASM: vsetvli {{.*}}, e32, m8
// ASM: vle32.v {{.*}}, v0.t
// ASM: vs8r.v
func.func @select_masked(%m: memref<128xi1>, %a: memref<128xf32>,
                         %o: memref<128xf32>) {
  %vl = arith.constant 128 : index
  %z = arith.constant 0.0 : f32
  vir.set_vl %vl : index {
    %k = vir.load %m[] : memref<128xi1> -> !vir.vec<?xi1>
    %x = vir.load %a[] : memref<128xf32> -> !vir.vec<?xf32>
    %zv = vir.broadcast %z : f32 -> !vir.vec<?xf32>
    %y = vir.select %k, %x, %zv : !vir.vec<?xi1>, !vir.vec<?xf32>
    vir.store %y, %o[] : !vir.vec<?xf32> -> memref<128xf32>
    vector.yield
  }
  return
}

// A memref of i1 in memory space 1: the bytes are addressed with a pointer
// of address space 1.
// CHECK-LABEL: func.func @space
// CHECK: llvm.inttoptr %{{.*}} : i64 to !llvm.ptr<1>
func.func @space(%m: memref<128xi1, 1>, %o: memref<128xi1, 1>) {
  %vl = arith.constant 128 : index
  vir.set_vl %vl : index {
    %k = vir.load %m[] : memref<128xi1, 1> -> !vir.vec<?xi1>
    vir.store %k, %o[] : !vir.vec<?xi1> -> memref<128xi1, 1>
    vector.yield
  }
  return
}
