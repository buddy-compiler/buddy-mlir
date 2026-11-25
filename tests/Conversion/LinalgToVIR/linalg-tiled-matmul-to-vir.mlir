// RUN: buddy-opt %s -lower-linalg-to-vir='target-vector-bits=256 target-vector-regs=16' | FileCheck %s --check-prefix=VIR
// RUN: buddy-opt %s -lower-linalg-to-vir='target-vector-bits=256 target-vector-regs=16' -lower-vir-to-vector='target-vector-bits=256 target-vector-regs=16' | FileCheck %s --check-prefix=FIXED
// RUN: buddy-opt %s -lower-linalg-to-vir='target-vector-bits=256 target-vector-regs=32 use-scalable=true' -lower-vir-to-vector='use-scalable=true rvv-vlen=256 use-avl-loop=true' | FileCheck %s --check-prefix=RVV

func.func @tails(%a: memref<5x7xf32>, %b: memref<7x19xf32>, %c: memref<5x19xf32>) {
  linalg.matmul ins(%a, %b : memref<5x7xf32>, memref<7x19xf32>)
      outs(%c : memref<5x19xf32>)
  return
}

// VIR-LABEL: func.func @tails
// VIR-NOT: linalg.matmul
// VIR: vir.set_vl
// VIR: scf.for {{.*}} iter_args({{.*}}) -> (!vir.vec<?xf32>, !vir.vec<?xf32>, !vir.vec<?xf32>, !vir.vec<?xf32>, !vir.vec<?xf32>, !vir.vec<?xf32>, !vir.vec<?xf32>, !vir.vec<?xf32>)
// VIR-COUNT-8: vir.fma
// VIR: scf.yield
// VIR-COUNT-8: vir.store
// VIR: vir.set_vl
// VIR-COUNT-4: vir.fma
// VIR: vir.set_vl
// VIR-COUNT-2: vir.fma
// FIXED-LABEL: func.func @tails
// FIXED: vector.fma {{.*}} : vector<8xf32>
// FIXED: math.fma {{.*}} : f32
// RVV-LABEL: func.func @tails
// RVV: vector.vscale
// RVV: vector.fma {{.*}} : vector<[4]xf32>
// RVV: llvm.intr.vp.load
// RVV: llvm.intr.vp.store

func.func @dynamic_mixed(%a: memref<?x?xf16>, %b: memref<?x?xf16>, %c: memref<?x?xf32>) {
  linalg.matmul ins(%a, %b : memref<?x?xf16>, memref<?x?xf16>)
      outs(%c : memref<?x?xf32>)
  return
}

// VIR-LABEL: func.func @dynamic_mixed
// VIR: memref.dim
// VIR: arith.remui
// VIR: vir.set_vl
// VIR: vir.extf {{.*}} : !vir.vec<?xf16> -> !vir.vec<?xf32>
// VIR: arith.extf {{.*}} : f16 to f32
// VIR: vir.fma {{.*}} : !vir.vec<?xf32>
// VIR: vir.store
// FIXED-LABEL: func.func @dynamic_mixed
// FIXED: arith.extf {{.*}} : vector<8xf16> to vector<8xf32>
// FIXED: vector.fma {{.*}} : vector<8xf32>
// RVV-LABEL: func.func @dynamic_mixed
// RVV: arith.extf {{.*}} : vector<[4]xf16> to vector<[4]xf32>
// RVV: vector.fma {{.*}} : vector<[4]xf32>

func.func @bf16(%a: memref<8x8xbf16>, %b: memref<8x32xbf16>, %c: memref<8x32xf32>) {
  linalg.matmul ins(%a, %b : memref<8x8xbf16>, memref<8x32xbf16>)
      outs(%c : memref<8x32xf32>)
  return
}

// VIR-LABEL: func.func @bf16
// VIR: vir.extf {{.*}} : !vir.vec<?xbf16> -> !vir.vec<?xf32>
// VIR: vir.fma

func.func @signed(%a: memref<8x7xi8>, %b: memref<7x32xi8>, %c: memref<8x32xi32>) {
  linalg.matmul ins(%a, %b : memref<8x7xi8>, memref<7x32xi8>)
      outs(%c : memref<8x32xi32>)
  return
}

// VIR-LABEL: func.func @signed
// VIR: vir.extsi {{.*}} : !vir.vec<?xi8> -> !vir.vec<?xi32>
// VIR: arith.extsi {{.*}} : i8 to i32
// VIR: arith.muli {{.*}} : !vir.vec<?xi32>
// VIR: arith.addi {{.*}} : !vir.vec<?xi32>
// FIXED-LABEL: func.func @signed
// FIXED: arith.extsi {{.*}} : vector<8xi8> to vector<8xi32>
// FIXED: arith.muli {{.*}} : vector<8xi32>
// FIXED: arith.addi {{.*}} : vector<8xi32>
// RVV-LABEL: func.func @signed
// RVV: arith.extsi {{.*}} : vector<[4]xi8> to vector<[4]xi32>
// RVV: arith.muli {{.*}} : vector<[4]xi32>
// RVV: arith.addi {{.*}} : vector<[4]xi32>

func.func @transpose_b(%a: memref<4x7xf32>, %b: memref<17x7xf32>, %c: memref<4x17xf32>) {
  linalg.matmul indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
    affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>]
    ins(%a, %b : memref<4x7xf32>, memref<17x7xf32>) outs(%c : memref<4x17xf32>)
  return
}

// VIR-LABEL: func.func @transpose_b
// VIR: %[[BT:.*]] = memref.transpose %arg1
// VIR: vir.load %[[BT]]
// VIR: vir.fma

func.func @batch_unsigned(%a: memref<2x4x7xi8>, %b: memref<2x7x32xi8>, %c: memref<2x4x32xi32>) {
  linalg.generic {indexing_maps = [affine_map<(b, m, n, k) -> (b, m, k)>,
    affine_map<(b, m, n, k) -> (b, k, n)>, affine_map<(b, m, n, k) -> (b, m, n)>],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
    ins(%a, %b : memref<2x4x7xi8>, memref<2x7x32xi8>) outs(%c : memref<2x4x32xi32>) {
  ^bb0(%a0: i8, %b0: i8, %c0: i32):
    %x = arith.extui %a0 : i8 to i32
    %y = arith.extui %b0 : i8 to i32
    %xy = arith.muli %x, %y : i32
    %next = arith.addi %xy, %c0 : i32
    linalg.yield %next : i32
  }
  return
}

// VIR-LABEL: func.func @batch_unsigned
// VIR-NOT: linalg.generic
// VIR: scf.for
// VIR: vir.set_vl
// VIR: vir.extui
// VIR: arith.extui {{.*}} : i8 to i32
// VIR: arith.muli
// VIR: arith.addi

func.func @batch_reduce(%a: memref<2x4x7xf32>, %b: memref<2x7x16xf32>, %c: memref<4x16xf32>) {
  linalg.batch_reduce_matmul
      ins(%a, %b : memref<2x4x7xf32>, memref<2x7x16xf32>)
      outs(%c : memref<4x16xf32>)
  return
}

// VIR-LABEL: func.func @batch_reduce
// VIR: vir.set_vl
// VIR: scf.for {{.*}} iter_args
// VIR: scf.for {{.*}} iter_args
// VIR: vir.fma
// VIR: scf.yield
// VIR: scf.yield
// VIR: vir.store

func.func @not_mad(%a: memref<4x7xf32>, %b: memref<7x16xf32>, %c: memref<4x16xf32>) {
  linalg.generic {indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
    affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%a, %b : memref<4x7xf32>, memref<7x16xf32>) outs(%c : memref<4x16xf32>) {
  ^bb0(%a0: f32, %b0: f32, %c0: f32):
    %xy = arith.mulf %a0, %b0 : f32
    %next = arith.subf %c0, %xy : f32
    linalg.yield %next : f32
  }
  return
}

// VIR-LABEL: func.func @not_mad
// VIR-NOT: vir.matmul_width
// VIR-NOT: vir.fma
// VIR: arith.subf
// VIR-NOT: vir.matmul_width
// VIR-NOT: vir.fma
// VIR: return
