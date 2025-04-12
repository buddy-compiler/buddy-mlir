// RUN: buddy-opt -reduce-vectorize="vector-size=16" -verify-diagnostics -lower-affine -expand-strided-metadata -convert-vector-to-scf -convert-vector-to-llvm -finalize-memref-to-llvm -convert-scf-to-cf -convert-arith-to-llvm -convert-func-to-llvm -lower-affine -llvm-request-c-wrappers -convert-arith-to-llvm -reconcile-unrealized-casts %s \
// RUN: | mlir-cpu-runner -O0 -e main -entry-point-result=void \
// RUN: -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext,%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

func.func private @rtclock() -> f64
func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }

// 创建一个12x40x40的输入张量
memref.global "private" @A : memref<12x40x40xf32> = dense<3.0>

func.func @kernel(%a : memref<12x40x40xf32>) {
  %t_start = call @rtclock() : () -> f64
  
  %b = memref.alloc() : memref<12x40xf32>  // 输出张量

  // 初始化常量
  %c0 = arith.constant 0.0 : f32
  %c16 = arith.constant 16 : index
  %c40 = arith.constant 40 : index
  %c0_idx = arith.constant 0 : index
  %c32_idx = arith.constant 32 : index
  %c1 = arith.constant 1 : index

  // 外层循环遍历 (i, j)
  affine.for %i = 0 to 12 {
    affine.for %j = 0 to 40 {
      // 初始化累加器
      %acc = arith.constant 0.0 : f32

      // 第一块：0 到 15
      %vec0 = vector.transfer_read %a[%i, %j, %c0_idx], %c0 {in_bounds = [true]} 
              : memref<12x40x40xf32>, vector<16xf32>
      %sum0 = vector.reduction <add>, %vec0, %c0 
              : vector<16xf32> into f32
      %acc1 = arith.addf %acc, %sum0 : f32

      // 第二块：16 到 31
      %vec1 = vector.transfer_read %a[%i, %j, %c16], %c0 {in_bounds = [true]} 
              : memref<12x40x40xf32>, vector<16xf32>
      %sum1 = vector.reduction <add>, %vec1, %c0 
              : vector<16xf32> into f32
      %acc2 = arith.addf %acc1, %sum1 : f32

      // 剩余块：32 到 39 (8 个元素)
      %vec2 = vector.transfer_read %a[%i, %j, %c32_idx], %c0 {in_bounds = [true]} 
              : memref<12x40x40xf32>, vector<8xf32>
      %sum2 = vector.reduction <add>, %vec2, %c0 
              : vector<8xf32> into f32
      %acc_final = arith.addf %acc2, %sum2 : f32

      // 写入结果
      affine.store %acc_final, %b[%i, %j] : memref<12x40xf32>
    }
  }

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  // 打印结果
  %printed_b = memref.cast %b : memref<12x40xf32> to memref<*xf32>
  call @printMemrefF32(%printed_b) : (memref<*xf32>) -> ()
  
  // 打印时间
  vector.print %time : f64
  
  memref.dealloc %b : memref<12x40xf32>
  return
}

func.func @main() {
  %a = memref.get_global @A : memref<12x40x40xf32>
  call @kernel(%a) : (memref<12x40x40xf32>) -> ()
  return
}