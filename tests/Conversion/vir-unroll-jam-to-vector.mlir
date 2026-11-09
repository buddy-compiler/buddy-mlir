// RUN: buddy-opt %s -lower-vir-to-vector="rvv-vlen=1024 use-scalable=true" -cse | FileCheck %s --check-prefix=IR
// RUN: buddy-opt %s -lower-vir-to-vector -cse --convert-linalg-to-loops --convert-vector-to-scf --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | mlir-runner -O0 -e main -entry-point-result=i32 -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext,%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext | FileCheck %s

// In a vir.set_vl without results, a loop whose iterations write elements
// of their own and share a vector load of an inner loop runs iterations
// jammed together, as many as the registers hold: out[o, :] += in[o, c] *
// w[c, :] shares the load of w[c, :]. At VLEN 1024 the 16 lanes are part of
// a register, and 8 iterations (the most) jam: one load of w[c, :] for 8
// fmas, then the 2 iterations left one by one.

// IR-LABEL: func.func @jam
// IR: %[[C8:.*]] = arith.constant 8 : index
// IR: %[[REM:.*]] = arith.remsi %{{.*}}, %[[C8]] : index
// IR: %[[MAIN:.*]] = arith.subi %c10, %[[REM]] : index
// IR: scf.for %{{.*}} = %c0 to %[[MAIN]] step %[[C8]] {
// IR: scf.for {{.*}} iter_args({{.*}}) -> (vector<[1]xf32>, vector<[1]xf32>, vector<[1]xf32>, vector<[1]xf32>, vector<[1]xf32>, vector<[1]xf32>, vector<[1]xf32>, vector<[1]xf32>) {
// IR-NEXT: vector.load %arg1
// IR-NOT: vector.load %arg1
// IR-COUNT-8: vector.fma
// IR: scf.yield
// IR-COUNT-8: vector.store
// IR: scf.for %{{.*}} = %[[MAIN]] to %c10 step %c1 {

// out and in may be the same memory: not jammed (no loop carries several
// vectors).
// IR-LABEL: func.func @may_alias
// IR-NOT: iter_args({{[^)]*}},
// IR-LABEL: func.func @dependence

// iteration o writes out[o + 1], which iteration o + 1 reads: not jammed.
// IR-NOT: iter_args({{[^)]*}},
// IR-LABEL: func.func @main

memref.global "private" constant @in : memref<10x3xf32> = dense<[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0], [7.0, 7.0, 7.0], [8.0, 8.0, 8.0], [9.0, 9.0, 9.0]]>
memref.global "private" constant @w : memref<3x16xf32> = dense<[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]>

func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }

func.func @jam(%in: memref<10x3xf32>, %w: memref<3x16xf32>) -> memref<10x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c10 = arith.constant 10 : index
  %c16 = arith.constant 16 : index
  %zero = arith.constant 0.0 : f32
  %out = memref.alloc() : memref<10x16xf32>
  linalg.fill ins(%zero : f32) outs(%out : memref<10x16xf32>)
  vir.set_vl %c16 : index {
    scf.for %o = %c0 to %c10 step %c1 {
      %acc = vir.load %out[%o, %c0] : memref<10x16xf32> -> !vir.vec<?xf32>
      %r = scf.for %c = %c0 to %c3 step %c1 iter_args(%a = %acc) -> (!vir.vec<?xf32>) {
        %wv = vir.load %w[%c, %c0] : memref<3x16xf32> -> !vir.vec<?xf32>
        %x = memref.load %in[%o, %c] : memref<10x3xf32>
        %xv = vir.broadcast %x : f32 -> !vir.vec<?xf32>
        %n = vir.fma %xv, %wv, %a : !vir.vec<?xf32>
        scf.yield %n : !vir.vec<?xf32>
      }
      vir.store %r, %out[%o, %c0] : !vir.vec<?xf32> -> memref<10x16xf32>
    }
    vector.yield
  }
  return %out : memref<10x16xf32>
}

func.func @may_alias(%in: memref<10x3xf32>, %w: memref<3x16xf32>, %out: memref<10x16xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c10 = arith.constant 10 : index
  %c16 = arith.constant 16 : index
  vir.set_vl %c16 : index {
    scf.for %o = %c0 to %c10 step %c1 {
      %acc = vir.load %out[%o, %c0] : memref<10x16xf32> -> !vir.vec<?xf32>
      %r = scf.for %c = %c0 to %c3 step %c1 iter_args(%a = %acc) -> (!vir.vec<?xf32>) {
        %wv = vir.load %w[%c, %c0] : memref<3x16xf32> -> !vir.vec<?xf32>
        %x = memref.load %in[%o, %c] : memref<10x3xf32>
        %xv = vir.broadcast %x : f32 -> !vir.vec<?xf32>
        %n = vir.fma %xv, %wv, %a : !vir.vec<?xf32>
        scf.yield %n : !vir.vec<?xf32>
      }
      vir.store %r, %out[%o, %c0] : !vir.vec<?xf32> -> memref<10x16xf32>
    }
    vector.yield
  }
  return
}

func.func @dependence(%in: memref<10x3xf32>, %w: memref<3x16xf32>) -> memref<11x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c10 = arith.constant 10 : index
  %c16 = arith.constant 16 : index
  %out = memref.alloc() : memref<11x16xf32>
  vir.set_vl %c16 : index {
    scf.for %o = %c0 to %c10 step %c1 {
      %acc = vir.load %out[%o, %c0] : memref<11x16xf32> -> !vir.vec<?xf32>
      %r = scf.for %c = %c0 to %c3 step %c1 iter_args(%a = %acc) -> (!vir.vec<?xf32>) {
        %wv = vir.load %w[%c, %c0] : memref<3x16xf32> -> !vir.vec<?xf32>
        %x = memref.load %in[%o, %c] : memref<10x3xf32>
        %xv = vir.broadcast %x : f32 -> !vir.vec<?xf32>
        %n = vir.fma %xv, %wv, %a : !vir.vec<?xf32>
        scf.yield %n : !vir.vec<?xf32>
      }
      %o1 = arith.addi %o, %c1 : index
      vir.store %r, %out[%o1, %c0] : !vir.vec<?xf32> -> memref<11x16xf32>
    }
    vector.yield
  }
  return %out : memref<11x16xf32>
}

func.func @main() -> i32 {
  %in = memref.get_global @in : memref<10x3xf32>
  %w = memref.get_global @w : memref<3x16xf32>
  %out = call @jam(%in, %w) : (memref<10x3xf32>, memref<3x16xf32>) -> memref<10x16xf32>
  %p = memref.cast %out : memref<10x16xf32> to memref<*xf32>
  // out[o, f] = o * (f + 2), and o * 18 at f = 15
  // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [10, 16] strides = [16, 1] data =
  // CHECK-NEXT: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  // CHECK-NEXT: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18]
  // CHECK-NEXT: [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 36]
  // CHECK-NEXT: [6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 54]
  // CHECK-NEXT: [8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72]
  // CHECK-NEXT: [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90]
  // CHECK-NEXT: [12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 108]
  // CHECK-NEXT: [14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 105, 112, 126]
  // CHECK-NEXT: [16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 144]
  // CHECK-NEXT: [18, 27, 36, 45, 54, 63, 72, 81, 90, 99, 108, 117, 126, 135, 144, 162]
  call @printMemrefF32(%p) : (memref<*xf32>) -> ()
  memref.dealloc %out : memref<10x16xf32>
  %ret = arith.constant 0 : i32
  return %ret : i32
}
