// RUN: buddy-opt %s \
// RUN:     -convert-vector-to-scf -lower-affine -convert-scf-to-cf \
// RUN:     -convert-cf-to-llvm \
// RUN:     -convert-vector-to-llvm \
// RUN:     -finalize-memref-to-llvm="use-aligned-alloc=true" \
// RUN:     -convert-func-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=i32 \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

// Demonstrate the usage of alignment attribute in vector.load
func.func @alloc_memref(%in : f32) -> memref<32xf32> {
  %c0 = arith.constant 0 : index
  %mem = memref.alloc() {alignment = 64} : memref<32xf32>
  %vec = vector.broadcast %in : f32 to vector<32xf32>
  vector.store %vec, %mem[%c0] {alignment = 64} : memref<32xf32>, vector<32xf32>
  return %mem : memref<32xf32>
}

// Demonstrate usage of memref.assume_alignment and {alignment = 64} attribute
func.func @fma(%mem1 : memref<32xf32>, %mem2 : memref<32xf32>, %mem3 : memref<32xf32>, %output : memref<32xf32>) -> () {
  memref.assume_alignment %mem1, 64 : memref<32xf32>
  memref.assume_alignment %mem2, 64 : memref<32xf32>
  memref.assume_alignment %mem3, 64 : memref<32xf32>
  memref.assume_alignment %output, 64 : memref<32xf32>
  %c0 = arith.constant 0 : index
  %v1 = vector.load %mem1[%c0] {alignment = 64} : memref<32xf32>, vector<32xf32>
  %v2 = vector.load %mem2[%c0] {alignment = 64} : memref<32xf32>, vector<32xf32>
  %v3 = vector.load %mem3[%c0] {alignment = 64} : memref<32xf32>, vector<32xf32>
  %v4 = vector.fma %v1, %v2, %v3 : vector<32xf32>
  vector.store %v4, %output[%c0] {alignment = 64} : memref<32xf32>, vector<32xf32>
  return
}


func.func @main() -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %input = arith.constant 1.0 : f32
  %input2 = arith.constant 2.0 : f32
  %input3 = arith.constant 3.0 : f32
  %mem1 = call @alloc_memref(%input) : (f32) -> memref<32xf32>
  %mem2 = call @alloc_memref(%input2) : (f32) -> memref<32xf32>
  %mem3 = call @alloc_memref(%input3) : (f32) -> memref<32xf32>
  %output = call @alloc_memref(%input) : (f32) -> memref<32xf32>
  call @fma(%mem1, %mem2, %mem3, %output) : (memref<32xf32>, memref<32xf32>, memref<32xf32>, memref<32xf32>) -> ()
  %v = vector.load %output[%c0] {alignment = 64} : memref<32xf32>, vector<32xf32>
  vector.print %v : vector<32xf32>
  // CHECK: ( 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5 )

  %ret = arith.constant 0 : i32
  return %ret : i32
}
