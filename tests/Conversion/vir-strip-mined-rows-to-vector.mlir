// RUN: buddy-opt %s -lower-vir-to-vector="vector-width=4" | FileCheck %s --check-prefix=IR
// RUN: buddy-opt %s -lower-vir-to-vector="vector-width=4" --convert-vector-to-scf --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | mlir-runner -O0 -e main -entry-point-result=i32 -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext,%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext | FileCheck %s

// In a vir.set_vl without results (a strip-mined loop over vl, in loops over
// the rows), a !vir.vec<2x?xf32> access is as vector.load: the lanes along
// the last memref dim, the rows along the dim before it, both from the
// indices on.

func.func private @printMemrefF32(memref<*xf32>)

// Rows from [1, 0, 1]: a[1][r][1 + c].
func.func @copy_at(%a: memref<3x2x6xf32>, %o: memref<2x5xf32>) {
  %vl = arith.constant 5 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[%c1, %c0, %c1] : memref<3x2x6xf32> -> !vir.vec<2x?xf32>
    vir.store %x, %o[] : !vir.vec<2x?xf32> -> memref<2x5xf32>
    vector.yield
  }
  return
}

// IR-LABEL: func.func @copy_at
// IR: affine.for %[[R:.*]] = {{.*}} {
// IR: affine.for %[[I:.*]] = {{.*}} step 4 {
// IR:   %[[ROW:.*]] = arith.addi %[[R]], %c0{{.*}} : index
// IR:   %[[COL:.*]] = arith.addi %[[I]], %c1{{.*}} : index
// IR:   vector.load %{{.*}}[%c1{{.*}}, %[[ROW]], %[[COL]]] : memref<3x2x6xf32>, vector<4xf32>
// IR:   vector.store %{{.*}}, %{{.*}}[%[[R]], %[[I]]] : memref<2x5xf32>, vector<4xf32>

// No indices on a memref of rank 3: the rows along dim 1, b[0][r][c].
func.func @copy_rank3(%b: memref<2x2x5xf32>, %o: memref<2x5xf32>) {
  %vl = arith.constant 5 : index
  vir.set_vl %vl : index {
    %x = vir.load %b[] : memref<2x2x5xf32> -> !vir.vec<2x?xf32>
    vir.store %x, %o[] : !vir.vec<2x?xf32> -> memref<2x5xf32>
    vector.yield
  }
  return
}

// IR-LABEL: func.func @copy_rank3
// IR: affine.for %[[R:.*]] = {{.*}} {
// IR: affine.for %[[I:.*]] = {{.*}} step 4 {
// IR:   vector.load %{{.*}}[%c0{{.*}}, %[[R]], %[[I]]] : memref<2x2x5xf32>, vector<4xf32>

func.func @main() -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %a = memref.alloc() : memref<3x2x6xf32>
  %b = memref.alloc() : memref<2x2x5xf32>
  // a[i][j][k] = b[i][j][k] = 100 i + 10 j + k
  scf.for %i = %c0 to %c3 step %c1 {
    scf.for %j = %c0 to %c2 step %c1 {
      scf.for %k = %c0 to %c6 step %c1 {
        %h = arith.constant 100 : index
        %t = arith.constant 10 : index
        %x0 = arith.muli %i, %h : index
        %x1 = arith.muli %j, %t : index
        %x2 = arith.addi %x0, %x1 : index
        %x3 = arith.addi %x2, %k : index
        %xi = arith.index_cast %x3 : index to i32
        %xf = arith.sitofp %xi : i32 to f32
        memref.store %xf, %a[%i, %j, %k] : memref<3x2x6xf32>
        %inb = arith.cmpi ult, %i, %c2 : index
        %ink = arith.cmpi ult, %k, %c5 : index
        %in = arith.andi %inb, %ink : i1
        scf.if %in {
          memref.store %xf, %b[%i, %j, %k] : memref<2x2x5xf32>
        }
      }
    }
  }
  %o = memref.alloc() : memref<2x5xf32>
  call @copy_at(%a, %o) : (memref<3x2x6xf32>, memref<2x5xf32>) -> ()
  %po = memref.cast %o : memref<2x5xf32> to memref<*xf32>
  call @printMemrefF32(%po) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [2, 5]
  // CHECK-NEXT: [101, 102, 103, 104, 105]
  // CHECK-NEXT: [111, 112, 113, 114, 115]
  call @copy_rank3(%b, %o) : (memref<2x2x5xf32>, memref<2x5xf32>) -> ()
  call @printMemrefF32(%po) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [2, 5]
  // CHECK-NEXT: [0, 1, 2, 3, 4]
  // CHECK-NEXT: [10, 11, 12, 13, 14]
  memref.dealloc %a : memref<3x2x6xf32>
  memref.dealloc %b : memref<2x2x5xf32>
  memref.dealloc %o : memref<2x5xf32>
  %r = arith.constant 0 : i32
  return %r : i32
}
