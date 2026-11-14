// RUN: buddy-opt %s -lower-vir-to-vector='rvv-vlen=256' -cse | FileCheck %s
// RUN: buddy-opt %s -lower-vir-to-vector='rvv-vlen=1024' -cse \
// RUN: | FileCheck %s --check-prefix=VLEN1024
// RUN: buddy-opt %s -lower-vir-to-vector='rvv-vlen=256 vector-width=4' -cse \
// RUN: | FileCheck %s --check-prefix=WIDTH
// RUN: not buddy-opt %s -lower-vir-to-vector='rvv-vlen=100' 2>&1 \
// RUN: | FileCheck %s --check-prefix=ERROR

// With rvv-vlen, the vector size is that of the RISC-V Vector target (32
// registers of VLEN bits), whatever the host: LMUL = min(8,
// 2^floor(log2(32 / Req_Group))) registers of VLEN / 32 f32 lanes.

// Three vectors live at the add: 32 / 3 = 10 registers per vector, LMUL 8.
func.func @low_pressure(%a: memref<1000xf32>, %b: memref<1000xf32>) {
  %vl = arith.constant 1000 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<1000xf32> -> !vir.vec<?xf32>
    %y = vir.load %b[] : memref<1000xf32> -> !vir.vec<?xf32>
    %s = arith.addf %x, %y : !vir.vec<?xf32>
    vir.store %s, %a[] : !vir.vec<?xf32> -> memref<1000xf32>
    vector.yield
  }
  return
}

// CHECK-LABEL: func.func @low_pressure
// CHECK: arith.addf {{.*}} : vector<64xf32>
// VLEN1024-LABEL: func.func @low_pressure
// VLEN1024: arith.addf {{.*}} : vector<256xf32>
// WIDTH-LABEL: func.func @low_pressure
// WIDTH: arith.addf {{.*}} : vector<4xf32>

// Ten vectors loaded before they are summed, eleven live at the first add:
// 32 / 11 = 2 registers per vector, LMUL 2.
func.func @high_pressure(%m: memref<10x1000xf32>, %o: memref<1000xf32>) {
  %vl = arith.constant 1000 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %c8 = arith.constant 8 : index
  %c9 = arith.constant 9 : index
  vir.set_vl %vl : index {
    %x0 = vir.load %m[%c0, %c0] : memref<10x1000xf32> -> !vir.vec<?xf32>
    %x1 = vir.load %m[%c1, %c0] : memref<10x1000xf32> -> !vir.vec<?xf32>
    %x2 = vir.load %m[%c2, %c0] : memref<10x1000xf32> -> !vir.vec<?xf32>
    %x3 = vir.load %m[%c3, %c0] : memref<10x1000xf32> -> !vir.vec<?xf32>
    %x4 = vir.load %m[%c4, %c0] : memref<10x1000xf32> -> !vir.vec<?xf32>
    %x5 = vir.load %m[%c5, %c0] : memref<10x1000xf32> -> !vir.vec<?xf32>
    %x6 = vir.load %m[%c6, %c0] : memref<10x1000xf32> -> !vir.vec<?xf32>
    %x7 = vir.load %m[%c7, %c0] : memref<10x1000xf32> -> !vir.vec<?xf32>
    %x8 = vir.load %m[%c8, %c0] : memref<10x1000xf32> -> !vir.vec<?xf32>
    %x9 = vir.load %m[%c9, %c0] : memref<10x1000xf32> -> !vir.vec<?xf32>
    %s1 = arith.addf %x0, %x1 : !vir.vec<?xf32>
    %s2 = arith.addf %s1, %x2 : !vir.vec<?xf32>
    %s3 = arith.addf %s2, %x3 : !vir.vec<?xf32>
    %s4 = arith.addf %s3, %x4 : !vir.vec<?xf32>
    %s5 = arith.addf %s4, %x5 : !vir.vec<?xf32>
    %s6 = arith.addf %s5, %x6 : !vir.vec<?xf32>
    %s7 = arith.addf %s6, %x7 : !vir.vec<?xf32>
    %s8 = arith.addf %s7, %x8 : !vir.vec<?xf32>
    %s9 = arith.addf %s8, %x9 : !vir.vec<?xf32>
    vir.store %s9, %o[] : !vir.vec<?xf32> -> memref<1000xf32>
    vector.yield
  }
  return
}

// CHECK-LABEL: func.func @high_pressure
// CHECK: arith.addf {{.*}} : vector<16xf32>
// VLEN1024-LABEL: func.func @high_pressure
// VLEN1024: arith.addf {{.*}} : vector<64xf32>

// ERROR: rvv-vlen must be a power of two >= 32, got 100
