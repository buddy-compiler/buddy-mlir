// RUN: buddy-opt %s \
// RUN:     -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" \
// RUN: | buddy-opt \
// RUN:     -arith-expand \
// RUN:     -eliminate-empty-tensors \
// RUN:     -empty-tensor-to-alloc-tensor \
// RUN:     -one-shot-bufferize \
// RUN:     -convert-linalg-to-affine-loops \
// RUN:     -affine-loop-fusion \
// RUN:     -lower-affine \
// RUN:     -func-bufferize \
// RUN:     -arith-bufferize \
// RUN:     -tensor-bufferize \
// RUN:     -buffer-deallocation \
// RUN:     -finalizing-bufferize \
// RUN:     -convert-vector-to-scf \
// RUN:     -expand-strided-metadata \
// RUN:     -convert-vector-to-llvm \
// RUN:     -memref-expand \
// RUN:     -arith-expand \
// RUN:     -convert-arith-to-llvm \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-openmp-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-math-to-libm  \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

func.func private @rtclock() -> f64
func.func private @printMemrefF32(memref<*xf32>)
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @kernel(%t0: tensor<1x40x1536xf32>, %t1: tensor<1536xf32>) {
  %t_start = call @rtclock() : () -> f64

  %idx_0 = arith.constant 0 : index
  %idx_1 = arith.constant 1 : index
  %idx_40 = arith.constant 40 : index

  %idx_128 = arith.constant 128 : index
  %idx_256 = arith.constant 256 : index
  %idx_384 = arith.constant 384 : index
  %idx_512 = arith.constant 512 : index
  %idx_640 = arith.constant 640 : index
  %idx_768 = arith.constant 768 : index
  %idx_896 = arith.constant 896 : index
  %idx_1024 = arith.constant 1024 : index
  %idx_1152 = arith.constant 1152 : index
  %idx_1280 = arith.constant 1280 : index
  %idx_1408 = arith.constant 1408 : index

  %memref_t0 = bufferization.to_memref %t0 : memref<1x40x1536xf32>

  %memref_t1 = bufferization.to_memref %t1 : memref<1536xf32>
  %weight_0 = vector.load %memref_t1[%idx_0] : memref<1536xf32>, vector<128xf32>
  %weight_1 = vector.load %memref_t1[%idx_128] : memref<1536xf32>, vector<128xf32>
  %weight_2 = vector.load %memref_t1[%idx_256] : memref<1536xf32>, vector<128xf32>
  %weight_3 = vector.load %memref_t1[%idx_384] : memref<1536xf32>, vector<128xf32>
  %weight_4 = vector.load %memref_t1[%idx_512] : memref<1536xf32>, vector<128xf32>
  %weight_5 = vector.load %memref_t1[%idx_640] : memref<1536xf32>, vector<128xf32>
  %weight_6 = vector.load %memref_t1[%idx_768] : memref<1536xf32>, vector<128xf32>
  %weight_7 = vector.load %memref_t1[%idx_896] : memref<1536xf32>, vector<128xf32>
  %weight_8 = vector.load %memref_t1[%idx_1024] : memref<1536xf32>, vector<128xf32>
  %weight_9 = vector.load %memref_t1[%idx_1152] : memref<1536xf32>, vector<128xf32>
  %weight_10 = vector.load %memref_t1[%idx_1280] : memref<1536xf32>, vector<128xf32>
  %weight_11 = vector.load %memref_t1[%idx_1408] : memref<1536xf32>, vector<128xf32>
  
  %zero = arith.constant 0.0 : f32
  %rsqrt_eps = arith.constant 9.99999997E-7 : f32
  %scale = arith.constant 1536.0 : f32
  %result_memref = memref.alloc() : memref<1x40x1536xf32>
  
  scf.parallel (%i) = (%idx_0) to (%idx_40) step (%idx_1) {
    %vec_0 = vector.load %memref_t0[%idx_0, %i, %idx_0] : memref<1x40x1536xf32>, vector<128xf32>
    %vec_1 = vector.load %memref_t0[%idx_0, %i, %idx_128] : memref<1x40x1536xf32>, vector<128xf32>
    %vec_2 = vector.load %memref_t0[%idx_0, %i, %idx_256] : memref<1x40x1536xf32>, vector<128xf32>
    %vec_3 = vector.load %memref_t0[%idx_0, %i, %idx_384] : memref<1x40x1536xf32>, vector<128xf32>
    %square_0 = arith.mulf %vec_0, %vec_0 : vector<128xf32>
    %square_1 = arith.mulf %vec_1, %vec_1 : vector<128xf32>
    %square_2 = arith.mulf %vec_2, %vec_2 : vector<128xf32>
    %square_3 = arith.mulf %vec_3, %vec_3 : vector<128xf32>
    %sum_0 = vector.reduction <add>, %square_0 : vector<128xf32> into f32
    %sum_1 = vector.reduction <add>, %square_1 : vector<128xf32> into f32
    %sum_2 = vector.reduction <add>, %square_2 : vector<128xf32> into f32
    %sum_3 = vector.reduction <add>, %square_3 : vector<128xf32> into f32

    %vec_4 = vector.load %memref_t0[%idx_0, %i, %idx_512] : memref<1x40x1536xf32>, vector<128xf32>
    %vec_5 = vector.load %memref_t0[%idx_0, %i, %idx_640] : memref<1x40x1536xf32>, vector<128xf32>
    %vec_6 = vector.load %memref_t0[%idx_0, %i, %idx_768] : memref<1x40x1536xf32>, vector<128xf32>
    %vec_7 = vector.load %memref_t0[%idx_0, %i, %idx_896] : memref<1x40x1536xf32>, vector<128xf32>
    %square_4 = arith.mulf %vec_4, %vec_4 : vector<128xf32>
    %square_5 = arith.mulf %vec_5, %vec_5 : vector<128xf32>
    %square_6 = arith.mulf %vec_6, %vec_6 : vector<128xf32>
    %square_7 = arith.mulf %vec_7, %vec_7 : vector<128xf32>
    %sum_4 = vector.reduction <add>, %square_4 : vector<128xf32> into f32
    %sum_5 = vector.reduction <add>, %square_5 : vector<128xf32> into f32
    %sum_6 = vector.reduction <add>, %square_6 : vector<128xf32> into f32
    %sum_7 = vector.reduction <add>, %square_7 : vector<128xf32> into f32

    %vec_8 = vector.load %memref_t0[%idx_0, %i, %idx_1024] : memref<1x40x1536xf32>, vector<128xf32>
    %vec_9 = vector.load %memref_t0[%idx_0, %i, %idx_1152] : memref<1x40x1536xf32>, vector<128xf32>
    %vec_10 = vector.load %memref_t0[%idx_0, %i, %idx_1280] : memref<1x40x1536xf32>, vector<128xf32>
    %vec_11 = vector.load %memref_t0[%idx_0, %i, %idx_1408] : memref<1x40x1536xf32>, vector<128xf32>
    %square_8 = arith.mulf %vec_8, %vec_8 : vector<128xf32>
    %square_9 = arith.mulf %vec_9, %vec_9 : vector<128xf32>
    %square_10 = arith.mulf %vec_10, %vec_10 : vector<128xf32>
    %square_11 = arith.mulf %vec_11, %vec_11 : vector<128xf32>
    %sum_8 = vector.reduction <add>, %square_8 : vector<128xf32> into f32
    %sum_9 = vector.reduction <add>, %square_9 : vector<128xf32> into f32
    %sum_10 = vector.reduction <add>, %square_10 : vector<128xf32> into f32
    %sum_11 = vector.reduction <add>, %square_11 : vector<128xf32> into f32

    // level 1
    %l1_0 = arith.addf %sum_0, %sum_1 : f32
    %l1_1 = arith.addf %sum_2, %sum_3 : f32
    %l1_2 = arith.addf %sum_4, %sum_5 : f32
    %l1_3 = arith.addf %sum_6, %sum_7 : f32
    %l1_4 = arith.addf %sum_8, %sum_9 : f32
    %l1_5 = arith.addf %sum_10, %sum_11 : f32
    // level 2
    %l2_0 = arith.addf %l1_0, %l1_1 : f32
    %l2_1 = arith.addf %l1_2, %l1_3 : f32
    %l2_2 = arith.addf %l1_4, %l1_5 : f32
    // level 3
    %l3_0 = arith.addf %l2_0, %l2_1 : f32
    // final sum
    %sum_all = arith.addf %l3_0, %l2_2 : f32
    
    %mean = arith.divf %sum_all, %scale : f32
    %var = arith.addf %mean, %rsqrt_eps : f32
    %inv_std = math.rsqrt %var : f32
    %inv_std_vec = vector.splat %inv_std : vector<128xf32>

    %vec_0_new = arith.mulf %vec_0, %inv_std_vec : vector<128xf32>
    %vec_1_new = arith.mulf %vec_1, %inv_std_vec : vector<128xf32>
    %vec_2_new = arith.mulf %vec_2, %inv_std_vec : vector<128xf32>
    %vec_3_new = arith.mulf %vec_3, %inv_std_vec : vector<128xf32>
    %vec_0_result = arith.mulf %vec_0_new, %weight_0 : vector<128xf32>
    %vec_1_result = arith.mulf %vec_1_new, %weight_1 : vector<128xf32>
    %vec_2_result = arith.mulf %vec_2_new, %weight_2 : vector<128xf32>
    %vec_3_result = arith.mulf %vec_3_new, %weight_3 : vector<128xf32>
    vector.store %vec_0_result, %result_memref[%idx_0, %i, %idx_0] : memref<1x40x1536xf32>, vector<128xf32>
    vector.store %vec_1_result, %result_memref[%idx_0, %i, %idx_128] : memref<1x40x1536xf32>, vector<128xf32>
    vector.store %vec_2_result, %result_memref[%idx_0, %i, %idx_256] : memref<1x40x1536xf32>, vector<128xf32>
    vector.store %vec_3_result, %result_memref[%idx_0, %i, %idx_384] : memref<1x40x1536xf32>, vector<128xf32>

    %vec_4_new = arith.mulf %vec_4, %inv_std_vec : vector<128xf32>
    %vec_5_new = arith.mulf %vec_5, %inv_std_vec : vector<128xf32>
    %vec_6_new = arith.mulf %vec_6, %inv_std_vec : vector<128xf32>
    %vec_7_new = arith.mulf %vec_7, %inv_std_vec : vector<128xf32>
    %vec_4_result = arith.mulf %vec_4_new, %weight_4 : vector<128xf32>
    %vec_5_result = arith.mulf %vec_5_new, %weight_5 : vector<128xf32>
    %vec_6_result = arith.mulf %vec_6_new, %weight_6 : vector<128xf32>
    %vec_7_result = arith.mulf %vec_7_new, %weight_7 : vector<128xf32>
    vector.store %vec_4_result, %result_memref[%idx_0, %i, %idx_512] : memref<1x40x1536xf32>, vector<128xf32>
    vector.store %vec_5_result, %result_memref[%idx_0, %i, %idx_640] : memref<1x40x1536xf32>, vector<128xf32>
    vector.store %vec_6_result, %result_memref[%idx_0, %i, %idx_768] : memref<1x40x1536xf32>, vector<128xf32>
    vector.store %vec_7_result, %result_memref[%idx_0, %i, %idx_896] : memref<1x40x1536xf32>, vector<128xf32>

    %vec_8_new = arith.mulf %vec_8, %inv_std_vec : vector<128xf32>
    %vec_9_new = arith.mulf %vec_9, %inv_std_vec : vector<128xf32>
    %vec_10_new = arith.mulf %vec_10, %inv_std_vec : vector<128xf32>
    %vec_11_new = arith.mulf %vec_11, %inv_std_vec : vector<128xf32>
    %vec_8_result = arith.mulf %vec_8_new, %weight_8 : vector<128xf32>
    %vec_9_result = arith.mulf %vec_9_new, %weight_9 : vector<128xf32>
    %vec_10_result = arith.mulf %vec_10_new, %weight_10 : vector<128xf32>
    %vec_11_result = arith.mulf %vec_11_new, %weight_11 : vector<128xf32>
    vector.store %vec_8_result, %result_memref[%idx_0, %i, %idx_1024] : memref<1x40x1536xf32>, vector<128xf32>
    vector.store %vec_9_result, %result_memref[%idx_0, %i, %idx_1152] : memref<1x40x1536xf32>, vector<128xf32>
    vector.store %vec_10_result, %result_memref[%idx_0, %i, %idx_1280] : memref<1x40x1536xf32>, vector<128xf32>
    vector.store %vec_11_result, %result_memref[%idx_0, %i, %idx_1408] : memref<1x40x1536xf32>, vector<128xf32>
  }

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  %print_result = memref.cast %result_memref : memref<1x40x1536xf32> to memref<*xf32>

  // Print results.
  call @printMemrefF32(%print_result) : (memref<*xf32>) -> ()
  // Print timings.
  vector.print %time : f64

  return
}

func.func @main() {

  %c0 = arith.constant dense<3.0> : tensor<1x40x1536xf32>
  %c1 = arith.constant dense <2.0> : tensor<1536xf32>

  call @kernel(%c0, %c1) : (tensor<1x40x1536xf32>, tensor<1536xf32>) -> ()

  return
}
