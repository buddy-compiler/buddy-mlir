// RUN: buddy-opt %s \
// RUN:     -convert-linalg-to-affine-loops \
// RUN:     -affine-loop-fusion \
// RUN:     -affine-parallelize \
// RUN:     -convert-vector-to-scf \
// RUN:     -lower-affine \
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

#map = affine_map<(d0) -> (d0 mod 64)>
#map1 = affine_map<(d0) -> (d0 ceildiv 64)>
#map2 = affine_map<(d0) -> (d0)>
module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func private @rtclock() -> f64
  func.func @kernel(%arg0: memref<8960x1536xf32, strided<[?, ?], offset: ?>>) -> memref<1x1536xf32> {
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg0 : memref<8960x1536xf32, strided<[?, ?], offset: ?>> -> memref<f32>, index, index, index, index, index
    %b = memref.reinterpret_cast %base_buffer to offset: [%offset], sizes: [8960, 1536], strides: [%strides#0, 1] : memref<f32> to memref<8960x1536xf32, strided<[?, 1], offset: ?>>
    %true = arith.constant true
    %cst = arith.constant 4.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %a = memref.alloc() {alignment = 64 : i64} : memref<1x8960xf32>
    linalg.fill ins(%cst_0 : f32) outs(%a : memref<1x8960xf32>)
    %c = memref.alloc() {alignment = 64 : i64} : memref<1x1536xf32>
    linalg.fill ins(%cst : f32) outs(%c : memref<1x1536xf32>)
    %0 = call @rtclock() : () -> f64

    memref.assume_alignment %a, 64 : memref<1x8960xf32>
    memref.assume_alignment %b, 64 : memref<8960x1536xf32, strided<[?, 1], offset: ?>>
    memref.assume_alignment %c, 64 : memref<1x1536xf32>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %step = arith.constant 32 : index
    %prefetch_step = arith.constant 1024 : index
    %m = arith.constant 1 : index
    %n = arith.constant 1536 : index
    %k = arith.constant 8960 : index

    scf.parallel (%n_idx) = (%c0) to (%n) step (%step) {
      %c_vec = vector.load %c[%c0, %n_idx] {alignment = 64 : i64} : memref<1x1536xf32>, vector<32xf32>
      %sum_iter = scf.for %k_idx = %c0 to %k step %c1 iter_args(%sum_vec = %c_vec) -> (vector<32xf32>) {
        %k_prefetch = arith.addi %k_idx, %prefetch_step : index
        memref.prefetch %b[%k_prefetch, %n_idx], read, locality<0>, data : memref<8960x1536xf32, strided<[?, 1], offset: ?>>
        %a_ele = memref.load %a[%c0, %k_idx] : memref<1x8960xf32>
        %a_vec = vector.broadcast %a_ele : f32 to vector<32xf32>
        %b_vec = vector.load %b[%k_idx, %n_idx] {alignment = 64 : i64, nontemporal = true} : memref<8960x1536xf32, strided<[?, 1], offset: ?>>, vector<32xf32>
        %r_vec = vector.fma %a_vec, %b_vec, %sum_vec : vector<32xf32>
        scf.yield %r_vec : vector<32xf32>
      }
      vector.store %sum_iter, %c[%c0, %n_idx] {alignment = 64 : i64} : memref<1x1536xf32>, vector<32xf32>
    }

    %5 = call @rtclock() : () -> f64
    %6 = arith.subf %5, %0 : f64
    vector.print %6 : f64
    // CHECK: {{[0-9]+\.[0-9]+}}
    return %c : memref<1x1536xf32>
  }
  func.func @main() {
    %true = arith.constant true
    %cst = arith.constant 3.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<8960x1536xf32>
    linalg.fill ins(%cst : f32) outs(%alloc : memref<8960x1536xf32>)
    %cast = memref.cast %alloc : memref<8960x1536xf32> to memref<8960x1536xf32, strided<[?, ?], offset: ?>>
    %0 = call @kernel(%cast) : (memref<8960x1536xf32, strided<[?, ?], offset: ?>>) -> memref<1x1536xf32>
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %0 : memref<1x1536xf32> -> memref<f32>, index, index, index, index, index
    %alloc_0 = memref.alloc() : memref<2xindex>
    %alloc_1 = memref.alloc() : memref<2xi1>
    %alloc_2 = memref.alloc() : memref<0xindex>
    %intptr = memref.extract_aligned_pointer_as_index %alloc : memref<8960x1536xf32> -> index
    %c0 = arith.constant 0 : index
    memref.store %intptr, %alloc_0[%c0] : memref<2xindex>
    %intptr_3 = memref.extract_aligned_pointer_as_index %base_buffer : memref<f32> -> index
    %c1 = arith.constant 1 : index
    memref.store %intptr_3, %alloc_0[%c1] : memref<2xindex>
    %c0_4 = arith.constant 0 : index
    memref.store %true, %alloc_1[%c0_4] : memref<2xi1>
    %c1_5 = arith.constant 1 : index
    memref.store %true, %alloc_1[%c1_5] : memref<2xi1>
    %cast_6 = memref.cast %alloc_0 : memref<2xindex> to memref<?xindex>
    %cast_7 = memref.cast %alloc_1 : memref<2xi1> to memref<?xi1>
    %cast_8 = memref.cast %alloc_2 : memref<0xindex> to memref<?xindex>
    %alloc_9 = memref.alloc() : memref<2xi1>
    %alloc_10 = memref.alloc() : memref<0xi1>
    %cast_11 = memref.cast %alloc_9 : memref<2xi1> to memref<?xi1>
    %cast_12 = memref.cast %alloc_10 : memref<0xi1> to memref<?xi1>
    call @dealloc_helper(%cast_6, %cast_8, %cast_7, %cast_11, %cast_12) : (memref<?xindex>, memref<?xindex>, memref<?xi1>, memref<?xi1>, memref<?xi1>) -> ()
    %c0_13 = arith.constant 0 : index
    %1 = memref.load %alloc_9[%c0_13] : memref<2xi1>
    scf.if %1 {
      memref.dealloc %alloc : memref<8960x1536xf32>
    }
    %c1_14 = arith.constant 1 : index
    %2 = memref.load %alloc_9[%c1_14] : memref<2xi1>
    scf.if %2 {
      memref.dealloc %base_buffer : memref<f32>
    }
    memref.dealloc %alloc_0 : memref<2xindex>
    memref.dealloc %alloc_2 : memref<0xindex>
    memref.dealloc %alloc_1 : memref<2xi1>
    memref.dealloc %alloc_9 : memref<2xi1>
    memref.dealloc %alloc_10 : memref<0xi1>
    return
  }
  func.func private @dealloc_helper(%arg0: memref<?xindex>, %arg1: memref<?xindex>, %arg2: memref<?xi1>, %arg3: memref<?xi1>, %arg4: memref<?xi1>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %false = arith.constant false
    %dim = memref.dim %arg0, %c0 : memref<?xindex>
    %dim_0 = memref.dim %arg1, %c0 : memref<?xindex>
    scf.for %arg5 = %c0 to %dim_0 step %c1 {
      memref.store %false, %arg4[%arg5] : memref<?xi1>
    }
    scf.for %arg5 = %c0 to %dim step %c1 {
      %0 = memref.load %arg0[%arg5] : memref<?xindex>
      %1 = memref.load %arg2[%arg5] : memref<?xi1>
      %2 = scf.for %arg6 = %c0 to %dim_0 step %c1 iter_args(%arg7 = %true) -> (i1) {
        %5 = memref.load %arg1[%arg6] : memref<?xindex>
        %6 = arith.cmpi eq, %5, %0 : index
        scf.if %6 {
          %9 = memref.load %arg4[%arg6] : memref<?xi1>
          %10 = arith.ori %9, %1 : i1
          memref.store %10, %arg4[%arg6] : memref<?xi1>
        }
        %7 = arith.cmpi ne, %5, %0 : index
        %8 = arith.andi %arg7, %7 : i1
        scf.yield %8 : i1
      }
      %3 = scf.for %arg6 = %c0 to %arg5 step %c1 iter_args(%arg7 = %2) -> (i1) {
        %5 = memref.load %arg0[%arg6] : memref<?xindex>
        %6 = arith.cmpi ne, %5, %0 : index
        %7 = arith.andi %arg7, %6 : i1
        scf.yield %7 : i1
      }
      %4 = arith.andi %3, %1 : i1
      memref.store %4, %arg3[%arg5] : memref<?xi1>
    }
    return
  }
}
