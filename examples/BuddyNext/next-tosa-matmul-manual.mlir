// RUN: buddy-opt %s \
// RUN:     -convert-linalg-to-affine-loops \
// RUN:     -affine-loop-fusion \
// RUN:     -affine-parallelize \
// RUN:     -convert-vector-to-scf \
// RUN:     -lower-affine \
// RUN:     -func-bufferize-dynamic-offset \
// RUN:     -cse \
// RUN:     -memref-expand \
// RUN:     -arith-expand \
// RUN:     -convert-vector-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-cf-to-llvm \
// RUN:     -convert-openmp-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-math-to-libm \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

#map = affine_map<(d0) -> (d0 mod 8)>
#map1 = affine_map<(d0, d1) -> (d0 - d1)>
#map2 = affine_map<(d0, d1) -> (d0 - d1 + 1)>
module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func private @rtclock() -> f64
  func.func @kernel(%arg0: memref<12x1x1024xf32, strided<[?, ?, ?], offset: ?>>, %arg1: memref<12x1024x128xf32, strided<[?, ?, ?], offset: ?>>) -> memref<12x1x128xf32> {
    %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %arg1 : memref<12x1024x128xf32, strided<[?, ?, ?], offset: ?>> -> memref<f32>, index, index, index, index, index, index, index
    %b = memref.reinterpret_cast %base_buffer to offset: [%offset], sizes: [12, 1024, 128], strides: [%strides#0, %strides#1, 1] : memref<f32> to memref<12x1024x128xf32, strided<[?, ?, 1], offset: ?>>
    %base_buffer_0, %offset_1, %sizes_2:3, %strides_3:3 = memref.extract_strided_metadata %arg0 : memref<12x1x1024xf32, strided<[?, ?, ?], offset: ?>> -> memref<f32>, index, index, index, index, index, index, index
    %a = memref.reinterpret_cast %base_buffer_0 to offset: [%offset_1], sizes: [12, 1, 1024], strides: [%strides_3#0, %strides_3#1, 1] : memref<f32> to memref<12x1x1024xf32, strided<[?, ?, 1], offset: ?>>
    %cst = arith.constant 0.000000e+00 : f32
    %0 = call @rtclock() : () -> f64
    %output = memref.alloc() {alignment = 64 : i64} : memref<12x1x128xf32>
    linalg.fill ins(%cst : f32) outs(%output : memref<12x1x128xf32>)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 64 : index
    %batch_size = memref.dim %a, %c0 : memref<12x1x1024xf32, strided<[?, ?, 1], offset: ?>>
    %m_size = memref.dim %a, %c1 : memref<12x1x1024xf32, strided<[?, ?, 1], offset: ?>>
    %k_size = memref.dim %a, %c2 : memref<12x1x1024xf32, strided<[?, ?, 1], offset: ?>>
    %n_size = memref.dim %output, %c2 : memref<12x1x128xf32>

    scf.parallel (%batch_idx) = (%c0) to (%batch_size) step (%c1) {
      scf.for %n_idx = %c0 to %n_size step %c32 {
        %c_vec = vector.load %output[%batch_idx, %c0, %n_idx] {alignment = 64 : i64} : memref<12x1x128xf32>, vector<64xf32>
        %sum_iter = scf.for %k_idx = %c0 to %k_size step %c1 iter_args(%sum_vec = %c_vec) -> (vector<64xf32>) {
          %a_ele = memref.load %a
          [%batch_idx, %c0, %k_idx] : memref<12x1x1024xf32, strided<[?, ?, 1], offset: ?>>
          %a_vec = vector.broadcast %a_ele : f32 to vector<64xf32>
          %b_vec = vector.load %b[%batch_idx, %k_idx, %n_idx] {alignment = 64 : i64, nontemporal = true} : memref<12x1024x128xf32, strided<[?, ?, 1], offset: ?>>, vector<64xf32>
          %r_vec = vector.fma %a_vec, %b_vec, %sum_vec : vector<64xf32>
          scf.yield %r_vec : vector<64xf32>
        }
        vector.store %sum_iter, %output[%batch_idx, %c0, %n_idx] : memref<12x1x128xf32>, vector<64xf32>
      }
    }
    %4 = call @rtclock() : () -> f64
    %5 = arith.subf %4, %0 : f64
    vector.print %5 : f64
    // CHECK: {{[0-9]+\.[0-9]+}}
    return %output : memref<12x1x128xf32>
  }
  func.func @main() {
    %true = arith.constant true
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 4.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<12x1x1024xf32>
    linalg.fill ins(%cst_0 : f32) outs(%alloc : memref<12x1x1024xf32>)
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<12x1024x128xf32>
    linalg.fill ins(%cst : f32) outs(%alloc_1 : memref<12x1024x128xf32>)
    scf.for %arg0 = %c0 to %c5 step %c1 {
      %cast = memref.cast %alloc : memref<12x1x1024xf32> to memref<12x1x1024xf32, strided<[?, ?, ?], offset: ?>>
      %cast_2 = memref.cast %alloc_1 : memref<12x1024x128xf32> to memref<12x1024x128xf32, strided<[?, ?, ?], offset: ?>>
      %0 = func.call @kernel(%cast, %cast_2) : (memref<12x1x1024xf32, strided<[?, ?, ?], offset: ?>>, memref<12x1024x128xf32, strided<[?, ?, ?], offset: ?>>) -> memref<12x1x128xf32>
      %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %0 : memref<12x1x128xf32> -> memref<f32>, index, index, index, index, index, index, index
      scf.if %true {
        memref.dealloc %base_buffer : memref<f32>
      }
    }
    scf.if %true {
      memref.dealloc %alloc : memref<12x1x1024xf32>
    }
    scf.if %true {
      memref.dealloc %alloc_1 : memref<12x1024x128xf32>
    }
    return
  }
}
